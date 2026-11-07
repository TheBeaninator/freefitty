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

use crate::config::Config;
use anyhow::{bail, Result};
use serde_json::Value;
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
}

pub struct Turn {
    pub message: Value,
    pub node: String,
    pub prompt_tokens: Option<u64>,
    pub cached_tokens: Option<u64>,
    pub finish_reason: Option<String>,
    /// How long the turn waited for a free slot before its request went out.
    pub waited_ms: u64,
}

#[derive(Debug)]
pub enum ChatError {
    /// No node served the turn; the slice retries later.
    Transient(String),
    /// The node rejected the request itself (400): retrying the same prompt cannot help.
    Rejected(String),
}
impl std::fmt::Display for ChatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self { ChatError::Transient(s) | ChatError::Rejected(s) => f.write_str(s) }
    }
}

impl Router {
    pub fn new(cfg: &Config, tasks: &mut Vec<tokio::task::JoinHandle<()>>) -> Arc<Router> {
        let nodes = cfg
            .nodes
            .iter()
            .map(|u| NodeState { url: u.clone(), up: false, slots: cfg.node_slots, busy: None, mine: 0, mine_at_poll: 0, wanted: 0, held: false, xfer: None, context_length: 0, model: None })
            .collect();
        let r = Arc::new(Router {
            cfg: cfg.clone(),
            client: reqwest::Client::builder().timeout(Duration::from_secs(cfg.slice_timeout_s)).connect_timeout(Duration::from_secs(5)).build().expect("client"),
            xfer_client: reqwest::Client::builder().timeout(Duration::from_secs(600)).connect_timeout(Duration::from_secs(5)).build().expect("client"),
            nodes: Mutex::new(nodes),
            counts: Mutex::new(Counts::default()),
            notify: Notify::new(),
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

    pub async fn any_up(&self) -> bool {
        self.nodes.lock().await.iter().any(|n| n.up)
    }

    /// The smallest context length the up nodes report, or 0 while none says.
    pub async fn context_length(&self) -> u64 {
        self.nodes.lock().await.iter().filter(|n| n.up && n.context_length > 0).map(|n| n.context_length).min().unwrap_or(0)
    }

    pub async fn view(&self) -> Vec<Value> {
        self.nodes
            .lock()
            .await
            .iter()
            .map(|n| serde_json::json!({"url": n.url, "up": n.up, "slots": n.slots, "load": n.load(), "mine": n.mine, "waiting": n.wanted, "held": n.held, "xfer": n.xfer, "model": n.model, "context_length": n.context_length}))
            .collect()
    }

    async fn poll(&self, i: usize) {
        let probe = reqwest::Client::builder().timeout(Duration::from_secs(2)).connect_timeout(Duration::from_secs(2)).build().expect("client");
        let url = self.cfg.nodes[i].clone();
        let mut granted_at: Option<Instant> = None;
        loop {
            if !self.cfg.grant_cmd.is_empty() && granted_at.is_none_or(|t| t.elapsed() >= Duration::from_secs(self.cfg.grant_check_s)) {
                granted_at = Some(Instant::now());
                let held = grant_held(&self.cfg.grant_cmd, &url).await;
                let was = std::mem::replace(&mut self.nodes.lock().await[i].held, held);
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
                let mut nodes = self.nodes.lock().await;
                let n = &mut nodes[i];
                was_up = n.up;
                n.up = up && parsed.get("status").and_then(|s| s.as_str()).is_none_or(|s| s == "ok");
                if let Some(s) = parsed.get("slots").and_then(|v| v.as_u64()) { n.slots = s.max(1) as u32; }
                let busy = parsed.get("busy_slots").and_then(|v| v.as_u64()).or_else(|| parsed.get("busy").and_then(|b| b.as_bool()).map(|b| if b { n.slots as u64 } else { 0 }));
                let queued = parsed.get("queued").and_then(|v| v.as_u64()).unwrap_or(0);
                n.busy = busy.map(|b| (b + queued) as u32);
                n.mine_at_poll = n.mine;
                if let Some(c) = parsed.get("context_length").and_then(|v| v.as_u64()) { n.context_length = c; }
                if let Some(m) = parsed.get("model").and_then(|v| v.as_str()) { n.model = Some(m.to_owned()); }
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
                self.nodes.lock().await[i].xfer = x;
            }
            if up != was_up {
                tracing::info!("node {url}: {}", if up { "up" } else { "down" });
            }
            self.notify.notify_waiters();
            tokio::time::sleep(POLL).await;
        }
    }

    /// Pick the node for a turn of a thread last served by `home`, carrying about `tokens` prompt tokens.
    /// Returns (node index, node to move the cached state from). Waits while the usable nodes are full, and
    /// up to `node_down_wait_s` while none is up.
    async fn place(&self, home: Option<&str>, tokens: u64, avoid: &[usize]) -> Result<(usize, Option<usize>)> {
        let started = Instant::now();
        let mut waiting_on: Option<usize> = None;
        let mut waited = false;
        loop {
            let notified = self.notify.notified();
            {
                let mut nodes = self.nodes.lock().await;
                if let Some(w) = waiting_on.take() { nodes[w].wanted -= 1; }
                let up: Vec<usize> = (0..nodes.len()).filter(|&i| nodes[i].up && !avoid.contains(&i)).collect();
                if up.is_empty() {
                    if started.elapsed() >= Duration::from_secs(self.cfg.node_down_wait_s) {
                        bail!("no model node is up");
                    }
                } else {
                    let h = home.and_then(|h| up.iter().copied().find(|&i| nodes[i].url == h));
                    let mut free: Vec<usize> = up.iter().copied().filter(|&i| !nodes[i].held && nodes[i].free() - nodes[i].wanted as i64 > 0).collect();
                    // a node a thread can later leave with its cache first; then the least loaded
                    free.sort_by_key(|&i| (nodes[i].xfer == Some(false), nodes[i].wanted as i64 - nodes[i].free(), nodes[i].load(), i));
                    let choice = match h {
                        Some(h) if !nodes[h].held && nodes[h].free() > 0 => Some((h, None, "kept")),
                        None if !free.is_empty() => Some((free[0], None, "new")),
                        // a held home is left like a full one, and spilled when its state cannot move
                        Some(h) if !free.is_empty() && (tokens <= self.cfg.spill_tokens || (self.cfg.move_threads && nodes[h].xfer == Some(false)) || (nodes[h].held && (!self.cfg.move_threads || nodes[h].xfer == Some(false)))) => Some((free[0], None, "spilled")),
                        Some(h) if !free.is_empty() && self.cfg.move_threads => Some((free[0], Some(h), "moved")),
                        _ => None,
                    };
                    if let Some((n, from, why)) = choice {
                        nodes[n].mine += 1;
                        let mut c = self.counts.lock().await;
                        match why { "kept" => c.kept += 1, "new" => c.new += 1, "spilled" => c.spilled += 1, _ => c.moved += 1 }
                        if waited { c.waited += 1; }
                        if why == "spilled" {
                            tracing::info!("thread leaves {} (full) for {}; it reads its ~{tokens}-token prompt again there", nodes[h.unwrap()].url, nodes[n].url);
                        }
                        return Ok((n, from));
                    }
                    if let Some(h) = h.filter(|&h| !nodes[h].held) { nodes[h].wanted += 1; waiting_on = Some(h); }
                }
            }
            waited = true;
            let _ = tokio::time::timeout(POLL, notified).await;
        }
    }

    async fn done(&self, i: usize) {
        self.nodes.lock().await[i].mine -= 1;
        self.notify.notify_waiters();
    }

    async fn down(&self, i: usize) {
        let mut nodes = self.nodes.lock().await;
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

    /// Run one chat completion for a thread. `home` is the node that served its last turn.
    pub async fn chat(&self, home: Option<&str>, body: &Value) -> Result<Turn, ChatError> {
        let tokens = (serde_json::to_vec(body).map(|b| b.len()).unwrap_or(0) / 4) as u64;
        let mut avoid = Vec::new();
        let total = self.cfg.nodes.len();
        let placed_at = Instant::now();
        loop {
            let (i, from) = self.place(home, tokens, &avoid).await.map_err(|e| ChatError::Transient(e.to_string()))?;
            let (url, from_url) = {
                let nodes = self.nodes.lock().await;
                (nodes[i].url.clone(), from.map(|f| nodes[f].url.clone()))
            };
            if let Some(f) = &from_url {
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
                    self.done(i).await;
                    self.down(i).await;
                    avoid.push(i);
                    if avoid.len() > total { return Err(ChatError::Transient(format!("every node refused: {e}"))); }
                    continue;
                }
                Err(e) => {
                    self.done(i).await;
                    if e.is_body() || e.is_decode() { self.counts.lock().await.broken += 1; }
                    return Err(ChatError::Transient(format!("{url}: {e}")));
                }
            };
            let status = resp.status();
            let text = match resp.bytes().await {
                Ok(b) => b,
                Err(e) => {
                    self.done(i).await;
                    self.counts.lock().await.broken += 1;
                    return Err(ChatError::Transient(format!("{url}: reply broke off: {e}")));
                }
            };
            self.done(i).await;
            if !status.is_success() {
                let msg = format!("{url} answered {}: {}", status, String::from_utf8_lossy(&text).chars().take(300).collect::<String>());
                return Err(if status.as_u16() == 400 { ChatError::Rejected(msg) } else { ChatError::Transient(msg) });
            }
            let v: Value = serde_json::from_slice(&text).map_err(|e| ChatError::Transient(format!("{url}: not JSON: {e}")))?;
            let mut turn = parse_turn(&v, url).map_err(ChatError::Transient)?;
            turn.waited_ms = placed_at.elapsed().as_millis() as u64;
            return Ok(turn);
        }
    }

    /// The up nodes' URLs.
    pub async fn up_nodes(&self) -> Vec<String> {
        self.nodes.lock().await.iter().filter(|n| n.up).map(|n| n.url.clone()).collect()
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
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn router(nodes: Vec<NodeState>, move_threads: bool) -> Router {
        let cfg = Config { move_threads, spill_tokens: 1000, ..Config::default() };
        Router { cfg, client: reqwest::Client::new(), xfer_client: reqwest::Client::new(), nodes: Mutex::new(nodes), counts: Mutex::new(Counts::default()), notify: Notify::new() }
    }

    fn node(url: &str, held: bool, xfer: bool) -> NodeState {
        NodeState { url: url.into(), up: true, slots: 2, busy: Some(0), mine: 0, mine_at_poll: 0, wanted: 0, held, xfer: Some(xfer), context_length: 0, model: None }
    }

    #[tokio::test]
    async fn a_held_node_takes_no_new_thread() {
        let r = router(vec![node("a", true, true), node("b", false, true)], true);
        assert_eq!(r.place(None, 10, &[]).await.unwrap(), (1, None));
    }

    #[tokio::test]
    async fn a_thread_leaves_its_held_home_with_its_cache() {
        let r = router(vec![node("a", true, true), node("b", false, true)], true);
        assert_eq!(r.place(Some("a"), 50_000, &[]).await.unwrap(), (1, Some(0)));
        assert_eq!(r.counts().moved, 1);
    }

    #[tokio::test]
    async fn a_thread_spills_from_a_held_home_that_cannot_move_it() {
        let r = router(vec![node("a", true, false), node("b", false, true)], true);
        assert_eq!(r.place(Some("a"), 50_000, &[]).await.unwrap(), (1, None));
        let r = router(vec![node("a", true, true), node("b", false, true)], false);
        assert_eq!(r.place(Some("a"), 50_000, &[]).await.unwrap(), (1, None));
        assert_eq!(r.counts().spilled, 1);
    }

    #[tokio::test]
    async fn a_free_home_keeps_its_thread() {
        let r = router(vec![node("a", false, true), node("b", true, true)], true);
        assert_eq!(r.place(Some("a"), 50_000, &[]).await.unwrap(), (0, None));
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
