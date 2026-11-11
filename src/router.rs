//! The model router: which strixite node runs a turn.
//!
//! A node's prompt cache holds the state of every conversation it has seen, so a thread's next turn goes to
//! the node that served its last one and reads only the new messages. A new thread goes to the node with the
//! most free slots. While a thread's node is full the turn waits here, and a slot that frees there goes to it
//! before any newcomer. A thread leaves its node only when that node is full and the thread is short enough to
//! read again (`spill_tokens`), or, with `move_threads`, by taking its cached state along over the rail:
//! `POST /cache/export` on the old node streams into `POST /cache/import` on the new one, then the turn runs there.
//!
//! Load is read from each node's /health every second (`slots`, `busy_slots`, `queued` count every client's
//! requests, so a node shared with other work counts as busy); between polls the router adds what it started
//! or finished itself. A node that stops answering is down until it answers again; a turn that never reached
//! it goes to another. While no node is up a turn waits `node_down_wait_s` before it fails.
//!
//! Nodes may serve different models (`node_meta` model; default `model`): a turn runs only on a node serving its
//! request's `model`, so a thread born from a DeepSeek expert lands on the StrixFold pair and every other thread
//! stays on strixite. StrixFold's /health says `max_batch_size` and `live` {connections, waiting} where strixite
//! says `slots`, `busy_slots` and `queued`; both are read.

use crate::config::Config;
use anyhow::{bail, Result};
use serde_json::Value;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, Notify};

const POLL: Duration = Duration::from_secs(1);

#[derive(Debug, Clone)]
struct NodeState {
    url: String,
    up: bool,
    slots: u32,
    busy: Option<u32>,
    mine: u32,
    mine_at_poll: u32,
    wanted: u32,
    held: bool,
    xfer: Option<bool>,
    context_length: u64,
    model: Option<String>,
    /// The model this node serves for placement (`node_meta` model, else the config's `model`).
    serves: String,
}

impl NodeState {
    fn load(&self) -> u32 {
        match self.busy {
            None => self.mine,
            Some(b) => self.mine.max((b + self.mine).saturating_sub(self.mine_at_poll)),
        }
    }
    fn free(&self) -> i64 {
        self.slots as i64 - self.load() as i64
    }
}

#[derive(Default, Debug, Clone, serde::Serialize)]
pub struct Counts {
    pub kept: u64,
    pub new: u64,
    pub spilled: u64,
    pub moved: u64,
    pub move_failed: u64,
    pub parked: u64,
    pub park_missed: u64,
    pub restored: u64,
    pub restore_failed: u64,
    pub restore_skipped: u64,
    pub waited: u64,
    pub broken: u64,
}

pub struct Router {
    cfg: Config,
    client: reqwest::Client,
    xfer_client: reqwest::Client,
    nodes: Mutex<Vec<NodeState>>,
    counts: Mutex<Counts>,
    notify: Notify,
    /// Set when the daemon stops: a turn still waiting for a slot gives up (ChatError::Stopping) instead of holding
    /// the stop for up to slice_timeout_s; a turn already sent to a node finishes.
    stopping: std::sync::atomic::AtomicBool,
    /// Threads whose turns go first (`POST /threads/{id}/priority`; pod-ff marks its expedite cards' threads): while
    /// one of their turns waits for a slot, no other turn takes one. After the 2026-11-09 rail loss the StrixFold
    /// cards' rounds sat behind 33 others for B/C/D's six slots while F+A waited for those rounds (dad 2026-11-10).
    priority: std::sync::Mutex<std::collections::HashSet<String>>,
    /// Priority turns waiting for a slot now, by model: a priority turn holds back only turns that could take
    /// the slots it waits for (one waiting for strixite never holds a DeepSeek turn off the StrixFold pair).
    priority_waiting: std::sync::Mutex<std::collections::HashMap<String, usize>>,
    /// Holds dropped while the nodes were locked: (node, slot rather than wait), applied at the next `lock_nodes`.
    released: std::sync::Mutex<Vec<(usize, bool)>>,
}

/// Counts a priority turn as waiting until it is placed, fails or is dropped (a cancelled slice).
struct PriorityWait<'a>(&'a std::sync::Mutex<std::collections::HashMap<String, usize>>, String);
impl Drop for PriorityWait<'_> {
    fn drop(&mut self) {
        let mut w = self.0.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(n) = w.get_mut(&self.1) { *n = n.saturating_sub(1); if *n == 0 { w.remove(&self.1); } }
    }
}

/// A slot (`mine`) or a wait (`wanted`) a turn holds on a node, given back however the turn ends: a cancelled
/// slice drops the turn's future mid-request, and the counts leaked there until a restart (review 2026-11-11: two
/// cancels made a 2-slot node look full for good). Dropped while the nodes are locked elsewhere, it queues the
/// release for the next lock.
struct Hold<'a> {
    r: &'a Router,
    i: usize,
    slot: bool,
}

impl Hold<'_> {
    /// Give it back under a lock the caller holds already.
    fn release_in(self, nodes: &mut [NodeState]) {
        release(&mut nodes[self.i], self.slot);
        std::mem::forget(self);
    }
}

impl Drop for Hold<'_> {
    fn drop(&mut self) {
        match self.r.nodes.try_lock() {
            Ok(mut nodes) => release(&mut nodes[self.i], self.slot),
            Err(_) => self.r.released.lock().unwrap_or_else(|e| e.into_inner()).push((self.i, self.slot)),
        }
        self.r.notify.notify_waiters();
    }
}

fn release(n: &mut NodeState, slot: bool) {
    if slot { n.mine = n.mine.saturating_sub(1) } else { n.wanted = n.wanted.saturating_sub(1) }
}

pub struct Turn {
    pub message: Value,
    pub node: String,
    pub prompt_tokens: Option<u64>,
    pub cached_tokens: Option<u64>,
    pub finish_reason: Option<String>,
    /// How long the turn waited for a free slot before its request went out.
    pub waited_ms: u64,
    /// From the request going out (a cache move included) to the reply.
    pub ran_ms: u64,
}

#[derive(Debug)]
pub enum ChatError {
    /// No node served the turn; the slice retries later.
    Transient(String),
    /// The node rejected the request itself (400): retrying the same prompt cannot help.
    Rejected(String),
    /// The daemon is stopping and the turn never reached a node: nothing ran, the slice re-queues as it was.
    Stopping,
}
impl std::fmt::Display for ChatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChatError::Transient(s) | ChatError::Rejected(s) => f.write_str(s),
            ChatError::Stopping => f.write_str("the daemon is stopping"),
        }
    }
}

impl Router {
    pub fn new(cfg: &Config, tasks: &mut Vec<tokio::task::JoinHandle<()>>) -> Arc<Router> {
        let nodes = cfg
            .nodes
            .iter()
            .map(|u| {
                let m = cfg.node_meta.get(u);
                NodeState { url: u.clone(), up: false, slots: m.and_then(|m| m.slots).unwrap_or(cfg.node_slots), busy: None, mine: 0, mine_at_poll: 0, wanted: 0, held: false, xfer: None,
                            context_length: m.and_then(|m| m.context_length).unwrap_or(0), model: None, serves: cfg.node_model(u).to_owned() }
            })
            .collect();
        let r = Arc::new(Router {
            cfg: cfg.clone(),
            client: reqwest::Client::builder().timeout(Duration::from_secs(cfg.slice_timeout_s)).connect_timeout(Duration::from_secs(5)).build().expect("client"),
            xfer_client: reqwest::Client::builder().timeout(Duration::from_secs(600)).connect_timeout(Duration::from_secs(5)).build().expect("client"),
            nodes: Mutex::new(nodes),
            counts: Mutex::new(Counts::default()),
            notify: Notify::new(),
            stopping: Default::default(),
            priority: Default::default(),
            priority_waiting: Default::default(),
            released: Default::default(),
        });
        for i in 0..cfg.nodes.len() {
            let r = r.clone();
            tasks.push(tokio::spawn(async move { r.poll(i).await }));
        }
        r
    }

    pub fn counts(&self) -> Counts {
        self.counts.try_lock().map(|c| c.clone()).unwrap_or_default()
    }

    /// Mark (`on`) or unmark a thread's turns as going first; returns whether it is marked now.
    pub fn set_priority(&self, thread: &str, on: bool) -> bool {
        let mut p = self.priority.lock().unwrap();
        if on { p.insert(thread.to_string()); } else { p.remove(thread); }
        self.notify.notify_waiters();
        on
    }

    pub fn priority_threads(&self) -> Vec<String> {
        let mut v: Vec<String> = self.priority.lock().unwrap().iter().cloned().collect();
        v.sort();
        v
    }

    /// The nodes, locked, with the holds dropped since the last lock given back.
    async fn lock_nodes(&self) -> tokio::sync::MutexGuard<'_, Vec<NodeState>> {
        let mut nodes = self.nodes.lock().await;
        for (i, slot) in std::mem::take(&mut *self.released.lock().unwrap_or_else(|e| e.into_inner())) { release(&mut nodes[i], slot); }
        nodes
    }

    pub async fn any_up(&self) -> bool {
        self.lock_nodes().await.iter().any(|n| n.up)
    }

    /// The smallest context length the up nodes serving the default model report, or 0 while none says.
    pub async fn context_length(&self) -> u64 {
        self.context_length_for(&self.cfg.model).await
    }

    /// The smallest context length the up nodes serving `model` report (or `node_meta` gives), 0 while none says.
    pub async fn context_length_for(&self, model: &str) -> u64 {
        self.lock_nodes().await.iter().filter(|n| n.up && n.serves == model && n.context_length > 0).map(|n| n.context_length).min().unwrap_or(0)
    }

    pub async fn view(&self) -> Vec<Value> {
        self.lock_nodes()
            .await
            .iter()
            .map(|n| serde_json::json!({"url": n.url, "up": n.up, "slots": n.slots, "load": n.load(), "mine": n.mine, "waiting": n.wanted, "held": n.held, "xfer": n.xfer, "model": n.model, "serves": n.serves, "context_length": n.context_length}))
            .collect()
    }

    async fn poll(&self, i: usize) {
        let probe = reqwest::Client::builder().timeout(Duration::from_secs(2)).connect_timeout(Duration::from_secs(2)).build().expect("client");
        let url = self.cfg.nodes[i].clone();
        let mut granted_at: Option<Instant> = None;
        loop {
            let granted = self.cfg.node_meta.get(&url).and_then(|m| m.grant) != Some(false);
            if granted && !self.cfg.grant_cmd.is_empty() && granted_at.is_none_or(|t| t.elapsed() >= Duration::from_secs(self.cfg.grant_check_s)) {
                granted_at = Some(Instant::now());
                let held = grant_held(&self.cfg.grant_cmd, &url).await;
                let was = std::mem::replace(&mut self.lock_nodes().await[i].held, held);
                if held != was {
                    tracing::info!("node {url}: {}", if held { "held: no new threads, its threads leave at their next turn" } else { "released" });
                }
            }
            let health = probe.get(format!("{url}/health")).send().await;
            let was_up;
            let (up, parsed) = match health {
                Ok(r) if r.status().is_success() => (true, r.json::<Value>().await.unwrap_or(Value::Null)),
                Ok(r) if r.status().as_u16() == 404 => {
                    // no /health: up while /v1/models answers
                    (probe.get(format!("{url}/v1/models")).send().await.map(|r| r.status().is_success()).unwrap_or(false), Value::Null)
                }
                _ => (false, Value::Null),
            };
            let mut xfer_unknown = false;
            {
                let mut nodes = self.lock_nodes().await;
                let n = &mut nodes[i];
                was_up = n.up;
                read_health(n, up, &parsed);
                if n.up && n.xfer.is_none() { xfer_unknown = true; }
                if !n.up { n.xfer = None; }
            }
            if xfer_unknown {
                // the cache routes exist only with cache-xfer on: an empty export answers 400 there, 404 elsewhere
                let x = match probe.post(format!("{url}/cache/export")).header("content-type", "application/json").body("").send().await {
                    Ok(r) if r.status().as_u16() == 400 => Some(true),
                    Ok(r) if r.status().as_u16() == 404 => Some(false),
                    Ok(_) => Some(false),
                    Err(_) => None,
                };
                self.lock_nodes().await[i].xfer = x;
            }
            if up != was_up {
                tracing::info!("node {url}: {}", if up { "up" } else { "down" });
            }
            self.notify.notify_waiters();
            tokio::time::sleep(POLL).await;
        }
    }

    /// Pick the node for a turn of `thread` (None: not a thread's, an expert being read) last served by `home`,
    /// carrying about `tokens` prompt tokens. Returns (node index, node to move the cached state from). Waits while
    /// the usable nodes are full, and up to `node_down_wait_s` while none is up.
    async fn place(&self, model: &str, thread: Option<&str>, home: Option<&str>, tokens: u64, avoid: &[usize]) -> Result<(Hold<'_>, Option<usize>)> {
        let started = Instant::now();
        let prio = thread.is_some_and(|t| self.priority.lock().unwrap().contains(t));
        let mut _prio_wait: Option<PriorityWait> = None;
        // a newcomer's patience is newcomer_wait_s plus up to half again, so newcomers born together (several
        // goldfish of one turn) do not all turn overdue in the same second
        let patience = Duration::from_millis(self.cfg.newcomer_wait_s * 1000 + rand::random_range(0..=self.cfg.newcomer_wait_s * 500));
        let mut waiting_on: Option<Hold> = None;
        let mut waited = false;
        loop {
            let notified = self.notify.notified();
            {
                let mut nodes = self.lock_nodes().await;
                if let Some(w) = waiting_on.take() { w.release_in(&mut nodes); }
                if self.is_stopping() { bail!("the daemon is stopping"); }
                let up: Vec<usize> = (0..nodes.len()).filter(|&i| nodes[i].up && nodes[i].serves == model && !avoid.contains(&i)).collect();
                if up.is_empty() {
                    if started.elapsed() >= Duration::from_secs(self.cfg.down_wait_s(model)) {
                        bail!("no node serving {model} is up");
                    }
                } else {
                    let h = home.and_then(|h| up.iter().copied().find(|&i| nodes[i].url == h));
                    // a slot freeing on a node goes to the threads waiting for that node (their cache is there)
                    // before a newcomer; past newcomer_wait_s a newcomer stops yielding, or under steady load new
                    // threads (goldfish, expert freezes) would never be placed (pod, 2026-11-08: 31 min). A thread
                    // whose home is down or held has no home to wait for and is a newcomer too (pod, 2026-11-09:
                    // threads of a down node waited 2.6 h for it to come back)
                    let overdue = h.is_none_or(|h| nodes[h].held) && started.elapsed() >= patience;
                    // overdue, a newcomer may take a spare slot home-waiters want, but where two or more of them wait it
                    // leaves them one (a burst of overdue newcomers never takes a node's every slot from its own threads)
                    // a priority turn takes any free slot, home-waiters' included
                    let reserve = |w: u32| if prio { 0 } else if overdue { (w as i64 - 1).max(0) } else { w as i64 };
                    let mut free: Vec<usize> = up.iter().copied().filter(|&i| !nodes[i].held && nodes[i].free() - reserve(nodes[i].wanted) > 0).collect();
                    // a node a thread can later leave with its cache first; then the least loaded
                    free.sort_by_key(|&i| (nodes[i].xfer == Some(false), nodes[i].wanted as i64 - nodes[i].free(), nodes[i].load(), i));
                    let yield_to_priority = !prio && self.priority_waiting.lock().unwrap().get(model).is_some_and(|&n| n > 0);
                    let choice = if yield_to_priority { None } else { match h {
                        Some(h) if !nodes[h].held && nodes[h].free() > 0 => Some((h, None, "kept")),
                        None if !free.is_empty() => Some((free[0], None, "new")),
                        // a held home is left like a full one, and spilled when its state cannot move
                        Some(h) if !free.is_empty() && (tokens <= self.cfg.spill_tokens || (self.cfg.move_threads && nodes[h].xfer == Some(false)) || (nodes[h].held && (!self.cfg.move_threads || nodes[h].xfer == Some(false)))) => Some((free[0], None, "spilled")),
                        Some(h) if !free.is_empty() && self.cfg.move_threads => Some((free[0], Some(h), "moved")),
                        _ => None,
                    } };
                    // an overdue newcomer with no spare slot anywhere waits on one node as that node's own threads do,
                    // and takes a slot there on their terms. Taking only slots no home-waiter wanted, it was never
                    // placed while every node had two or more waiting (pod, 2026-11-11: a djinn thread, 3.5 h)
                    let adopt = if overdue && choice.is_none() && !yield_to_priority {
                        up.iter().copied().filter(|&i| !nodes[i].held).min_by_key(|&i| (nodes[i].wanted as i64 - nodes[i].free(), nodes[i].load(), i))
                    } else { None };
                    let choice = choice.or_else(|| adopt.filter(|&a| nodes[a].free() > 0).map(|a| match h {
                        None => (a, None, "new"),
                        Some(h) if tokens <= self.cfg.spill_tokens || !self.cfg.move_threads || nodes[h].xfer == Some(false) => (a, None, "spilled"),
                        Some(h) => (a, Some(h), "moved"),
                    }));
                    if let Some((n, from, why)) = choice {
                        nodes[n].mine += 1;
                        let mut c = self.counts.lock().await;
                        match why { "kept" => c.kept += 1, "new" => c.new += 1, "spilled" => c.spilled += 1, _ => c.moved += 1 }
                        if waited { c.waited += 1; }
                        if why == "spilled" {
                            tracing::info!("thread leaves {} (full) for {}; it reads its ~{tokens}-token prompt again there", nodes[h.unwrap()].url, nodes[n].url);
                        }
                        return Ok((Hold { r: self, i: n, slot: true }, from));
                    }
                    if let Some(w) = h.filter(|&h| !nodes[h].held).or(adopt) { nodes[w].wanted += 1; waiting_on = Some(Hold { r: self, i: w, slot: false }); }
                    if prio && _prio_wait.is_none() {
                        *self.priority_waiting.lock().unwrap().entry(model.to_owned()).or_default() += 1;
                        _prio_wait = Some(PriorityWait(&self.priority_waiting, model.to_owned()));
                    }
                }
            }
            waited = true;
            let _ = tokio::time::timeout(POLL, notified).await;
        }
    }

    /// The daemon is stopping: turns waiting for a slot give up now (see `stopping`).
    pub fn stop(&self) {
        self.stopping.store(true, std::sync::atomic::Ordering::Relaxed);
        self.notify.notify_waiters();
    }

    fn is_stopping(&self) -> bool {
        self.stopping.load(std::sync::atomic::Ordering::Relaxed)
    }

    async fn down(&self, i: usize) {
        let mut nodes = self.lock_nodes().await;
        if nodes[i].up {
            nodes[i].up = false;
            tracing::warn!("node {}: connection failed, down until its /health answers", nodes[i].url);
        }
    }

    /// Stream a thread's cached state from one node to another; Ok(false) when nothing was cached.
    pub async fn move_state(&self, from: &str, to: &str, body: &Value) -> Result<bool> {
        let exp = self.xfer_client.post(format!("{from}/cache/export")).json(body).send().await?;
        match exp.status().as_u16() {
            200 => {}
            404 => return Ok(false),
            s => bail!("export answered {s}"),
        }
        let len = exp.content_length();
        let mut req = self.xfer_client.post(format!("{to}/cache/import")).header("content-type", "application/octet-stream");
        if let Some(l) = len { req = req.header("content-length", l); }
        let imp = req.body(reqwest::Body::wrap_stream(exp.bytes_stream())).send().await?;
        if !imp.status().is_success() {
            bail!("import answered {}: {}", imp.status(), imp.text().await.unwrap_or_default().chars().take(200).collect::<String>());
        }
        Ok(true)
    }

    /// Copy a thread's cached state from its node into `path` (written beside it and renamed); the bytes
    /// written, or None when the node holds nothing for this conversation.
    pub async fn park_state(&self, from: &str, body: &Value, path: &Path, min_tokens: u64) -> Result<Option<u64>> {
        let mut r = self.save_state_covering(from, body, path, min_tokens).await?;
        if let Some((_, covered)) = r.filter(|&(_, c)| c < min_tokens) {
            // a shorter entry (the system prompt's) is not this thread's state: worth nothing parked
            tracing::info!("{from}: the export covered {covered} of the {min_tokens} tokens wanted; not parked");
            let _ = tokio::fs::remove_file(path).await;
            r = None;
        }
        let r = r.map(|(n, _)| n);
        let mut c = self.counts.lock().await;
        if r.is_some() { c.parked += 1 } else { c.park_missed += 1 }
        Ok(r)
    }

    /// `save_state`, retried until the entry covers at least `min_tokens`: strixite answers a turn first and
    /// caches its entries after, so an export sent the moment a reply arrives gets an older, shorter entry (the
    /// system prompt's). After about 10 s the shorter entry is kept. (bytes, tokens covered).
    pub async fn save_state_covering(&self, from: &str, body: &Value, path: &Path, min_tokens: u64) -> Result<Option<(u64, u64)>> {
        let mut last = None;
        for attempt in 0..20 {
            last = self.save_state(from, body, path).await?;
            match last {
                Some((_, t)) if t >= min_tokens => return Ok(last),
                None if attempt >= 4 => return Ok(None),
                _ => tokio::time::sleep(Duration::from_millis(500)).await,
            }
        }
        if let Some((_, t)) = last { tracing::warn!("{from}: the export covers {t} tokens, under the {min_tokens} wanted; kept"); }
        Ok(last)
    }

    /// The node's best cached entry covering a prefix of `body`'s prompt, written to `path` beside it and
    /// renamed: (bytes, tokens it covers); None when no entry covers any prefix.
    pub async fn save_state(&self, from: &str, body: &Value, path: &Path) -> Result<Option<(u64, u64)>> {
        use futures_util::StreamExt;
        use tokio::io::AsyncWriteExt;
        let exp = self.xfer_client.post(format!("{from}/cache/export")).json(body).send().await?;
        match exp.status().as_u16() {
            200 => {}
            404 => return Ok(None),
            s => bail!("export answered {s}"),
        }
        let tokens = exp.headers().get("x-cache-tokens").and_then(|v| v.to_str().ok()).and_then(|v| v.parse().ok()).unwrap_or(0u64);
        let part = crate::park::part_path(path);
        let written = async {
            let mut f = tokio::fs::File::create(&part).await?;
            let mut n = 0u64;
            let mut stream = exp.bytes_stream();
            while let Some(chunk) = stream.next().await {
                let chunk = chunk?;
                f.write_all(&chunk).await?;
                n += chunk.len() as u64;
            }
            f.flush().await?;
            tokio::fs::rename(&part, path).await?;
            anyhow::Ok(n)
        }.await;
        if written.is_err() { let _ = tokio::fs::remove_file(&part).await; }
        Ok(Some((written?, tokens)))
    }

    /// How many tokens of this request's prompt a node's prompt cache holds (strixite POST /cache/probe); None when
    /// the node cannot say (an older strixite), so the caller sends the state.
    async fn probe(&self, url: &str, body: &Value) -> Option<u64> {
        let r = self.xfer_client.post(format!("{url}/cache/probe")).json(body).timeout(Duration::from_secs(10)).send().await.ok()?;
        if !r.status().is_success() { return None; }
        r.json::<Value>().await.ok()?.get("tokens")?.as_u64()
    }

    /// The steering vectors EVERY up node holds (strixite GET /steer): name -> layer. A thread may be placed on any
    /// node, so a vector one node lacks would fail there: the intersection. A node that answers 404 (no steer-dir)
    /// holds none; one that cannot be reached is tried three times and then named in the error, so a blip is not
    /// mistaken for a missing vector. Err when no node is up or one stays unreachable.
    /// Whether every up node takes `steer_scope` (GET /steer answers "scope": true). Err when no node is up or one
    /// stays unreachable; a node without vectors (404) does not take it.
    pub async fn steer_scope_everywhere(&self) -> Result<bool, String> {
        let mut any = false;
        for url in self.up_nodes().await {
            let mut got = None;
            for attempt in 0..3 {
                match self.client.get(format!("{url}/steer")).timeout(Duration::from_secs(5)).send().await {
                    Ok(r) if r.status().as_u16() == 404 => { got = Some(false); break; }
                    Ok(r) if r.status().is_success() => {
                        let v: Value = r.json().await.unwrap_or(Value::Null);
                        got = Some(v.get("scope").and_then(|s| s.as_bool()) == Some(true));
                        break;
                    }
                    _ => if attempt < 2 { tokio::time::sleep(Duration::from_millis(500)).await },
                }
            }
            match got {
                Some(true) => any = true,
                Some(false) => return Ok(false),
                None => return Err(format!("{url} did not answer GET /steer (three tries); try again")),
            }
        }
        if any { Ok(true) } else { Err("no node is up to check steer_scope".to_owned()) }
    }

    pub async fn steer_vectors(&self) -> Result<std::collections::HashMap<String, i64>, String> {
        let mut all: Option<std::collections::HashMap<String, i64>> = None;
        for url in self.up_nodes().await {
            let mut held = None;
            for attempt in 0..3 {
                match self.client.get(format!("{url}/steer")).timeout(Duration::from_secs(5)).send().await {
                    Ok(r) if r.status().as_u16() == 404 => { held = Some(Default::default()); break; }
                    Ok(r) if r.status().is_success() => {
                        let v: Value = r.json().await.unwrap_or(Value::Null);
                        held = Some(v.get("vectors").and_then(|a| a.as_array()).into_iter().flatten()
                            .filter_map(|x| Some((x.get("name")?.as_str()?.to_owned(), x.get("layer")?.as_i64()?))).collect());
                        break;
                    }
                    _ => if attempt < 2 { tokio::time::sleep(Duration::from_millis(500)).await },
                }
            }
            let Some(held): Option<std::collections::HashMap<String, i64>> = held else {
                return Err(format!("{url} did not answer GET /steer (three tries); try again"));
            };
            all = Some(match all {
                None => held,
                Some(mut a) => { a.retain(|k, l| held.get(k) == Some(l)); a }
            });
        }
        all.ok_or_else(|| "no node is up to list steering vectors".to_owned())
    }

    /// Put saved state (a parked thread's, a frozen expert's) into a node's cache; the bytes sent.
    pub async fn load_state(&self, to: &str, path: &Path) -> Result<u64> {
        use tokio::io::AsyncReadExt;
        let f = tokio::fs::File::open(path).await?;
        let len = f.metadata().await?.len();
        let stream = futures_util::stream::unfold(f, |mut f| async move {
            let mut buf = vec![0u8; 8 << 20];
            match f.read(&mut buf).await {
                Ok(0) => None,
                Ok(n) => { buf.truncate(n); Some((Ok::<_, std::io::Error>(bytes::Bytes::from(buf)), f)) }
                Err(e) => Some((Err(e), f)),
            }
        });
        let imp = self.xfer_client.post(format!("{to}/cache/import")).header("content-type", "application/octet-stream")
            .header("content-length", len).body(reqwest::Body::wrap_stream(stream)).send().await?;
        if !imp.status().is_success() {
            bail!("import answered {}: {}", imp.status(), imp.text().await.unwrap_or_default().chars().take(200).collect::<String>());
        }
        Ok(len)
    }

    /// Run one chat completion for `thread` (None: not a thread's). `home` is the node that served its last turn.
    pub async fn chat(&self, thread: Option<&str>, home: Option<&str>, body: &Value) -> Result<Turn, ChatError> {
        self.chat_restoring(thread, home, body, None).await
    }

    /// `chat`, first putting saved state (if any) into the node the turn is placed on.
    pub async fn chat_restoring(&self, thread: Option<&str>, home: Option<&str>, body: &Value, parked: Option<&Path>) -> Result<Turn, ChatError> {
        let tokens = (serde_json::to_vec(body).map(|b| b.len()).unwrap_or(0) / 4) as u64;
        let model = body.get("model").and_then(|m| m.as_str()).unwrap_or(&self.cfg.model).to_owned();
        let mut avoid = Vec::new();
        let total = self.cfg.nodes.len();
        let placed_at = Instant::now();
        loop {
            let (slot, from) = self.place(&model, thread, home, tokens, &avoid).await
                .map_err(|e| if self.is_stopping() { ChatError::Stopping } else { ChatError::Transient(e.to_string()) })?;
            let i = slot.i;
            let (waited, sent_at) = (placed_at.elapsed(), Instant::now());
            let (url, from_url) = {
                let nodes = self.lock_nodes().await;
                (nodes[i].url.clone(), from.map(|f| nodes[f].url.clone()))
            };
            if let Some(p) = parked.filter(|_| avoid.is_empty()) {
                // saved state (a parked thread's, or the frozen expert a new thread is born from) serves on any node;
                // a node that already holds at least as much of this conversation (probe) gets nothing sent
                let t = Instant::now();
                let have = crate::park::tokens_in(p);
                let held = match have { Some(_) => self.probe(&url, body).await, None => None };
                let skip = matches!((held, have), (Some(h), Some(w)) if h >= w);
                if skip {
                    let (h, want) = (held.unwrap_or(0), have.unwrap_or(0));
                    self.counts.lock().await.restore_skipped += 1;
                    tracing::info!("saved state {} not sent: {url} holds {h} tokens of it already ({want} saved)", p.file_name().and_then(|f| f.to_str()).unwrap_or(""));
                } else { match self.load_state(&url, p).await {
                    Ok(n) => {
                        self.counts.lock().await.restored += 1;
                        tracing::info!("saved state {} ({:.1} GB) restored into {url} in {:.1}s", p.file_name().and_then(|f| f.to_str()).unwrap_or(""), n as f64 / 1e9, t.elapsed().as_secs_f64());
                    }
                    Err(e) => {
                        self.counts.lock().await.restore_failed += 1;
                        tracing::warn!("restoring saved state into {url} failed ({e}); it reads the prompt itself");
                    }
                } }
            } else if let Some(f) = &from_url {
                let t = Instant::now();
                match self.move_state(f, &url, body).await {
                    Ok(true) => tracing::info!("thread moved {f} -> {url} with its cache in {:.1}s", t.elapsed().as_secs_f64()),
                    Ok(false) => tracing::info!("thread moved {f} -> {url}; nothing cached, it reads its prompt there"),
                    Err(e) => {
                        self.counts.lock().await.move_failed += 1;
                        tracing::warn!("cache move {f} -> {url} failed ({e}); {url} reads the prompt itself");
                    }
                }
            }
            let res = self.client.post(format!("{url}/v1/chat/completions")).json(body).send().await;
            let resp = match res {
                Ok(r) => r,
                Err(e) if e.is_connect() => {
                    drop(slot);
                    self.down(i).await;
                    avoid.push(i);
                    if avoid.len() > total { return Err(ChatError::Transient(format!("every node refused: {e}"))); }
                    continue;
                }
                Err(e) => {
                    drop(slot);
                    if e.is_body() || e.is_decode() { self.counts.lock().await.broken += 1; }
                    return Err(ChatError::Transient(format!("{url}: {e}")));
                }
            };
            let status = resp.status();
            let text = match resp.bytes().await {
                Ok(b) => b,
                Err(e) => {
                    drop(slot);
                    self.counts.lock().await.broken += 1;
                    return Err(ChatError::Transient(format!("{url}: reply broke off: {e}")));
                }
            };
            drop(slot);
            if !status.is_success() {
                let msg = format!("{url} answered {}: {}", status, String::from_utf8_lossy(&text).chars().take(300).collect::<String>());
                return Err(if status.as_u16() == 400 { ChatError::Rejected(msg) } else { ChatError::Transient(msg) });
            }
            let v: Value = serde_json::from_slice(&text).map_err(|e| ChatError::Transient(format!("{url}: not JSON: {e}")))?;
            let mut turn = parse_turn(&v, url).map_err(ChatError::Transient)?;
            // until 2026-11-07 waited_ms was placed_at.elapsed() HERE, so the log's "waited N s for a slot" was the
            // whole turn, generation included (pod: "waited 85 s" on turns that never waited)
            turn.waited_ms = waited.as_millis() as u64;
            turn.ran_ms = sent_at.elapsed().as_millis() as u64;
            return Ok(turn);
        }
    }

    /// The up nodes' URLs among those serving the default model (steering lives on strixite only).
    pub async fn up_nodes(&self) -> Vec<String> {
        self.up_nodes_for(&self.cfg.model).await
    }

    /// The up nodes serving `model`.
    pub async fn up_nodes_for(&self, model: &str) -> Vec<String> {
        self.lock_nodes().await.iter().filter(|n| n.up && n.serves == model).map(|n| n.url.clone()).collect()
    }

    /// One completion on one named node, outside placement (warming an expert's prefix into its cache).
    pub async fn chat_on(&self, url: &str, body: &Value) -> Result<Turn> {
        let r = self.client.post(format!("{url}/v1/chat/completions")).json(body).send().await?;
        let status = r.status();
        let text = r.bytes().await?;
        if !status.is_success() {
            bail!("{url} answered {status}: {}", String::from_utf8_lossy(&text).chars().take(300).collect::<String>());
        }
        let v: Value = serde_json::from_slice(&text)?;
        parse_turn(&v, url.to_owned()).map_err(|e| anyhow::anyhow!(e))
    }

    pub fn model(&self) -> &str {
        &self.cfg.model
    }

    /// The fewest tokens an export from `node` must cover for a `prompt_tokens`-token prompt (config min_cover).
    pub fn min_cover(&self, node: &str, prompt_tokens: u64) -> u64 {
        self.cfg.min_cover(node, prompt_tokens)
    }

    /// Whether requests of `model` carry `shared_messages` (config `node_meta` shared_messages).
    pub fn takes_shared_messages(&self, model: &str) -> bool {
        self.cfg.takes_shared_messages(model)
    }
}

/// Apply one /health answer (`up`: it answered) to a node: strixite's `slots`/`busy_slots`/`queued`, or
/// StrixFold's `max_batch_size` and `live` {connections, waiting}. A field the answer lacks keeps its value.
fn read_health(n: &mut NodeState, up: bool, h: &Value) {
    n.up = up && h.get("status").and_then(|s| s.as_str()).is_none_or(|s| s == "ok");
    if let Some(s) = h.get("slots").or_else(|| h.get("max_batch_size")).and_then(|v| v.as_u64()) { n.slots = s.max(1) as u32; }
    let live = h.get("live");
    let busy = h.get("busy_slots").and_then(|v| v.as_u64())
        .or_else(|| h.get("busy").and_then(|b| b.as_bool()).map(|b| if b { n.slots as u64 } else { 0 }))
        .or_else(|| live.and_then(|l| l.get("connections")).and_then(|v| v.as_u64()));
    let queued = h.get("queued").or_else(|| live.and_then(|l| l.get("waiting"))).and_then(|v| v.as_u64()).unwrap_or(0);
    n.busy = busy.map(|b| (b + queued) as u32);
    n.mine_at_poll = n.mine;
    if let Some(c) = h.get("context_length").and_then(|v| v.as_u64()) { n.context_length = c; }
    if let Some(m) = h.get("model").and_then(|v| v.as_str()) { n.model = Some(m.to_owned()); }
}

/// Run the grant command for `node`: exit 1 = held. A command that cannot run counts as free, so a missing
/// grant tool never stops the router.
async fn grant_held(cmd: &[String], node: &str) -> bool {
    let argv: Vec<String> = cmd.iter().map(|a| a.replace("{node}", node)).collect();
    let out = tokio::process::Command::new(&argv[0]).args(&argv[1..]).stdin(std::process::Stdio::null()).stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null()).kill_on_drop(true).status();
    matches!(tokio::time::timeout(Duration::from_secs(10), out).await, Ok(Ok(s)) if s.code() == Some(1))
}

fn parse_turn(v: &Value, url: String) -> Result<Turn, String> {
    let message = v.pointer("/choices/0/message").cloned().ok_or_else(|| format!("{url}: no choices[0].message"))?;
    let usage = v.get("usage");
    Ok(Turn {
        message,
        finish_reason: v.pointer("/choices/0/finish_reason").and_then(|f| f.as_str()).map(str::to_owned),
        node: url,
        prompt_tokens: usage.and_then(|u| u.get("prompt_tokens")).and_then(|v| v.as_u64()),
        cached_tokens: usage.and_then(|u| u.pointer("/prompt_tokens_details/cached_tokens")).and_then(|v| v.as_u64()),
        waited_ms: 0,
        ran_ms: 0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const M: &str = "qwen3.8-flash-next";

    fn serving(mut n: NodeState, model: &str) -> NodeState { n.serves = model.into(); n }

    #[tokio::test]
    async fn a_turn_runs_only_on_nodes_serving_its_model() {
        // a (the default model) is the less loaded node, yet a DeepSeek turn goes to b and a default turn to a
        let mut b = serving(node("b", false, false), "dsv41");
        b.busy = Some(1);
        let r = router(vec![node("a", false, true), b], true);
        assert_eq!(r.placed("dsv41", None, None, 10, &[]).await.unwrap(), (1, None));
        assert_eq!(r.placed(M, None, None, 10, &[]).await.unwrap(), (0, None));
        // a full node of the right model is waited for, never traded for a free node of another model
        let r = router(vec![node("a", false, true), { let mut b = serving(node("b", false, false), "dsv41"); b.busy = Some(2); b }], true);
        assert!(tokio::time::timeout(Duration::from_millis(1500), r.placed("dsv41", None, None, 10, &[])).await.is_err());
    }

    #[tokio::test]
    async fn a_priority_turn_holds_back_only_turns_of_its_own_model() {
        // the only default-model node is full: a priority default turn waits there, and a DeepSeek newcomer
        // still takes the StrixFold node's free slot
        let mut a = node("a", false, true);
        a.busy = Some(2);
        let r = Arc::new(router(vec![a, serving(node("b", false, false), "dsv41")], true));
        r.set_priority("p", true);
        let rp = r.clone();
        let _prio = tokio::spawn(async move { rp.placed(M, Some("p"), None, 10, &[]).await });
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert_eq!(r.priority_waiting.lock().unwrap().get(M).copied(), Some(1));
        let other = tokio::time::timeout(Duration::from_secs(3), r.placed("dsv41", Some("o"), None, 10, &[])).await;
        assert_eq!(other.expect("not held back by another model's priority turn").unwrap(), (1, None));
    }

    #[test]
    fn strixfold_health_reads_as_slots_and_load() {
        let mut n = serving(node("f", false, false), "dsv41");
        n.busy = None;
        let h = serde_json::json!({"status": "ok", "model": "dsv41", "max_batch_size": 1, "warming": false,
                                   "live": {"connections": 1, "waiting": 2, "decode_tokens_per_second": 0.0}});
        read_health(&mut n, true, &h);
        assert!(n.up);
        assert_eq!((n.slots, n.busy, n.model.as_deref()), (1, Some(3), Some("dsv41")));
        // strixite's own fields win where both are present
        let h = serde_json::json!({"status": "ok", "slots": 2, "busy_slots": 1, "queued": 0, "context_length": 262144});
        read_health(&mut n, true, &h);
        assert_eq!((n.slots, n.busy, n.context_length), (2, Some(1), 262144));
        read_health(&mut n, true, &serde_json::json!({"status": "loading"}));
        assert!(!n.up, "a status other than ok is not up");
    }

    fn router(nodes: Vec<NodeState>, move_threads: bool) -> Router {
        let cfg = Config { move_threads, spill_tokens: 1000, ..Config::default() };
        Router { cfg, client: reqwest::Client::new(), xfer_client: reqwest::Client::new(), nodes: Mutex::new(nodes), counts: Mutex::new(Counts::default()), notify: Notify::new(), stopping: Default::default(), priority: Default::default(), priority_waiting: Default::default(), released: Default::default() }
    }

    impl Router {
        /// `place`, the turn keeping its slot (still in flight when the test looks).
        async fn placed(&self, model: &str, thread: Option<&str>, home: Option<&str>, tokens: u64, avoid: &[usize]) -> Result<(usize, Option<usize>)> {
            let (h, from) = self.place(model, thread, home, tokens, avoid).await?;
            let i = h.i;
            std::mem::forget(h);
            Ok((i, from))
        }
    }

    #[tokio::test]
    async fn a_cancelled_turn_gives_back_its_slot_and_its_wait() {
        // review 2026-11-11: Queue::cancel drops the slice's future mid-request, and mine/wanted never came back down
        let r = router(vec![node("a", false, true)], true);
        let (slot, _) = r.place(M, None, None, 10, &[]).await.unwrap();
        let held = r.nodes.lock().await;  // dropped under another's lock: queued, given back at the next lock
        drop(slot);
        drop(held);
        assert_eq!(r.lock_nodes().await[0].mine, 0);
        r.lock_nodes().await[0].mine = 2;  // full: a thread of a waits for it
        let waiting = tokio::time::timeout(Duration::from_millis(1500), r.place(M, Some("t"), Some("a"), 10, &[])).await;
        assert!(waiting.is_err());
        let n = r.lock_nodes().await;
        assert_eq!((n[0].mine, n[0].wanted), (2, 0), "the dropped wait gave its want back");
    }

    fn node(url: &str, held: bool, xfer: bool) -> NodeState {
        NodeState { url: url.into(), up: true, slots: 2, busy: Some(0), mine: 0, mine_at_poll: 0, wanted: 0, held, xfer: Some(xfer), context_length: 0, model: None, serves: M.into() }
    }

    #[tokio::test]
    async fn a_held_node_takes_no_new_thread() {
        let r = router(vec![node("a", true, true), node("b", false, true)], true);
        assert_eq!(r.placed(M, None, None, 10, &[]).await.unwrap(), (1, None));
    }

    #[tokio::test]
    async fn a_thread_leaves_its_held_home_with_its_cache() {
        let r = router(vec![node("a", true, true), node("b", false, true)], true);
        assert_eq!(r.placed(M, None, Some("a"), 50_000, &[]).await.unwrap(), (1, Some(0)));
        assert_eq!(r.counts().moved, 1);
    }

    #[tokio::test]
    async fn a_thread_spills_from_a_held_home_that_cannot_move_it() {
        let r = router(vec![node("a", true, false), node("b", false, true)], true);
        assert_eq!(r.placed(M, None, Some("a"), 50_000, &[]).await.unwrap(), (1, None));
        let r = router(vec![node("a", true, true), node("b", false, true)], false);
        assert_eq!(r.placed(M, None, Some("a"), 50_000, &[]).await.unwrap(), (1, None));
        assert_eq!(r.counts().spilled, 1);
    }

    #[tokio::test]
    async fn a_newcomer_yields_to_waiting_threads_only_until_it_is_overdue() {
        let mut a = node("a", false, true);
        a.mine = 1;
        a.wanted = 1;  // a thread waits for a's last free slot
        let mut r = router(vec![a], true);
        r.cfg.newcomer_wait_s = 1;
        r.cfg.node_down_wait_s = 30;
        let t = Instant::now();
        assert_eq!(r.placed(M, None, None, 10, &[]).await.unwrap(), (0, None));
        assert!(t.elapsed() >= Duration::from_secs(1), "it waited its turn first");
        assert_eq!(r.counts().new, 1);
    }

    #[tokio::test]
    async fn an_overdue_newcomer_waits_on_a_full_node_like_its_own_threads() {
        // pod 2026-11-11: every node full with two to four threads waiting for home, a new djinn thread waited 3.5 h
        let mut a = node("a", false, true);
        a.mine = 2;
        a.wanted = 2;  // full, two threads wait for a slot
        let mut r = router(vec![a], true);
        r.cfg.newcomer_wait_s = 0;
        r.cfg.node_down_wait_s = 30;
        let r = std::sync::Arc::new(r);
        let p = { let r = r.clone(); tokio::spawn(async move { r.placed(M, None, None, 10, &[]).await.unwrap() }) };
        tokio::time::sleep(Duration::from_millis(300)).await;
        assert_eq!(r.lock_nodes().await[0].wanted, 3, "it waits beside the home-waiters");
        r.lock_nodes().await[0].mine = 1;  // a slot frees: it takes it on the same terms as they do
        r.notify.notify_waiters();
        let placed = tokio::time::timeout(Duration::from_secs(3), p).await.expect("placed when a slot frees").unwrap();
        assert_eq!(placed, (0, None));
        assert_eq!(r.lock_nodes().await[0].wanted, 2, "its wait is given back");
    }

    #[tokio::test]
    async fn a_thread_whose_home_is_down_or_held_turns_overdue_like_a_newcomer() {
        for held in [false, true] {
            let mut a = node("a", held, true);
            a.up = held;  // down, or up and held
            let mut b = node("b", false, true);
            b.mine = 1;
            b.wanted = 1;  // a thread waits for b's last free slot
            let mut r = router(vec![a, b], true);
            r.cfg.newcomer_wait_s = 1;
            r.cfg.node_down_wait_s = 30;
            r.cfg.spill_tokens = 100_000;
            let placed = tokio::time::timeout(Duration::from_secs(5), r.placed(M, None, Some("a"), 50_000, &[])).await;
            assert_eq!(placed.expect("it is placed once overdue, not when its home returns").unwrap().0, 1);
        }
    }

    #[tokio::test]
    async fn a_priority_turn_goes_first_and_takes_slots_home_waiters_want() {
        let mut a = node("a", false, true);
        a.mine = 2;  // full
        let r = Arc::new(router(vec![a], true));
        r.set_priority("p", true);
        let rp = r.clone();
        let prio = tokio::spawn(async move { rp.placed(M, Some("p"), None, 10, &[]).await.unwrap() });
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert_eq!(r.priority_waiting.lock().unwrap().get(M).copied().unwrap_or(0), 1);
        // a slot frees: the ordinary turn waiting since before the priority one still yields to it
        let ro = r.clone();
        let other = tokio::spawn(async move { tokio::time::timeout(Duration::from_secs(3), ro.placed(M, Some("o"), Some("a"), 10, &[])).await });
        tokio::time::sleep(Duration::from_millis(100)).await;
        { let mut n = r.nodes.lock().await; n[0].mine = 1; n[0].wanted += 1; }  // freed, and wanted by home-waiters
        r.notify.notify_waiters();
        assert_eq!(prio.await.unwrap(), (0, None));
        assert_eq!(r.priority_waiting.lock().unwrap().get(M).copied().unwrap_or(0), 0, "placed: no longer counted as waiting");
        assert!(other.await.unwrap().is_err(), "the ordinary turn got no slot (the node is full again)");
    }

    #[tokio::test]
    async fn an_unmarked_thread_waits_normally() {
        let r = router(vec![node("a", false, true)], true);
        r.set_priority("p", true);
        r.set_priority("p", false);
        assert_eq!(r.priority_threads(), Vec::<String>::new());
        assert_eq!(r.placed(M, Some("p"), None, 10, &[]).await.unwrap(), (0, None));
    }

    #[tokio::test]
    async fn a_free_home_keeps_its_thread() {
        let r = router(vec![node("a", false, true), node("b", true, true)], true);
        assert_eq!(r.placed(M, None, Some("a"), 50_000, &[]).await.unwrap(), (0, None));
    }

    #[tokio::test]
    async fn the_grant_command_holds_only_on_exit_1() {
        let sh = |code: &str| vec!["sh".to_string(), "-c".to_string(), format!("test {{node}} = http://n:1 || exit 0; exit {code}")];
        assert!(grant_held(&sh("1"), "http://n:1").await);
        assert!(!grant_held(&sh("0"), "http://n:1").await);
        assert!(!grant_held(&sh("2"), "http://n:1").await);
        assert!(!grant_held(&sh("1"), "http://other:1").await, "{{node}} is the node's url");
        assert!(!grant_held(&["/nonexistent/grant".to_string()], "http://n:1").await, "a missing tool is free");
    }
}
