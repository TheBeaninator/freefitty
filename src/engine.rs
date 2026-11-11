//! A slice: one thread, the events that arrived for it, one model turn, one commit.
//!
//! The conversation is loaded, the new events become messages (a person's text, a tool's result, a wake-up),
//! the model answers once. Its text becomes a reply; each tool call becomes a wake-up event (the sleeps) or
//! a RAP invocation. All of it, the consumed events included, is one transaction in the store, and only
//! after that commit are invocations posted and replies handed out. A crash before the commit re-runs the
//! slice from the same events; one after it cannot run the turn twice, because the events are gone and
//! their ids are recorded as processed.
//!
//! What the model sees is the *live view*: the shared prefix (the system prompt, then the thread's expert
//! messages if it has one), the bridge left by the last refocus (a framing line and the model's own notes with
//! the calls still awaited), then the history from the refocus point. The history itself is never changed.

use crate::config::Config;
use crate::rap::{self, ToolBox};
use crate::router::{ChatError, Router};
use crate::store::{now_ms, Commit, Compaction, Event, EventKind, GoldfishOf, Invocation, PendingCall, Store, ThreadMeta};
use crate::window;
use serde_json::{json, Value};
use std::sync::Arc;

pub struct Engine {
    pub cfg: Config,
    pub store: Arc<Store>,
    pub router: Arc<Router>,
    pub tools: Arc<ToolBox>,
    pub system: String,
    /// What `system_prompt_commands` put into `system`, as it was at start.
    pub hoisted: Vec<crate::config::Hoisted>,
    /// Instruction ids whose text is on disk already.
    pub instructions_seen: std::sync::Mutex<std::collections::HashSet<String>>,
    pub compact_prompt: String,
    pub callback_url: String,
    /// The latest node-count-over-estimate ratio any thread saw, times 1000; 0 until a turn has run.
    pub scale_hint: std::sync::atomic::AtomicU64,
    /// Where quiet threads' cached state is parked (`park`); None = off.
    pub parking: Option<Arc<crate::park::Parking>>,
    /// Whether a slice of a thread is running (the queue's own record, waiting for a slot included); set once the
    /// queue exists. A thread is never parked mid-turn.
    pub inflight: Arc<std::sync::OnceLock<Box<dyn Fn(&str) -> bool + Send + Sync>>>,
    /// This engine, for tasks that outlive a call (a park firing later); set once the engine is in its Arc.
    pub me: std::sync::OnceLock<std::sync::Weak<Engine>>,
    pub experts: Arc<crate::experts::Experts>,
    pub metrics: Arc<crate::metrics::Metrics>,
    /// Queues an event from outside a slice (a commissioned expert's result); set once the queue exists.
    pub deliver: Arc<std::sync::OnceLock<Box<dyn Fn(Event) + Send + Sync>>>,
}

/// What a goldfish's turn said, for its elephant: the text of its assistant messages.
fn text_for_elephant(msgs: &[Value]) -> String {
    msgs.iter().filter(|m| m["role"] == "assistant").filter_map(|m| m["content"].as_str()).map(str::trim).filter(|t| !t.is_empty()).collect::<Vec<_>>().join("\n")
}


pub enum Outcome {
    Done,
    /// Nothing ran (the daemon is stopping and the turn never reached a node): the events stay as they were, no
    /// attempt counted, and run after the restart.
    Requeue,
    /// Nothing was committed; the events run again after `retry_after_s`, and dead-letter past `max_attempts`.
    Transient(String),
    /// Nothing was committed and no retry can help; the events dead-letter now and the thread is closed.
    Fatal(String),
}

const UNKNOWN_TOOL_ROUNDS: usize = 3;
/// Turns cut off at the output limit with no whole tool call that a slice answers with a note and goes on.
const CUT_OFF_ROUNDS: usize = 2;

pub fn tokens(v: &[Value]) -> u64 {
    v.iter().map(window::weigh).sum()
}

/// The bytes / 4 estimate scaled up by `scale` (never down: a node that reports fewer tokens than the estimate,
/// a fake, a different accounting, never makes the guard looser than bytes / 4).
pub fn calibrated(est: u64, scale: f64) -> u64 {
    if scale <= 1.0 { est } else { (est as f64 * scale) as u64 }
}

/// The bytes / 4 of a request's tool definitions: counted by the node, left out of the messages' estimate. On a
/// short thread they are most of the prompt (4,258 tokens over a 450-token estimate on 2026-11-08), and that ratio,
/// as the shared hint, closed a new goldfish as "context full" at ten times its size.
pub fn overhead_tokens(body: &Value) -> u64 {
    body.get("tools").map(|t| serde_json::to_vec(t).map(|b| b.len()).unwrap_or(0) as u64 / 4).unwrap_or(0)
}

/// The most the latest ratio of some other thread can scale a thread that has no count of its own yet. Qwen runs
/// about 1.3 over bytes / 4; a thread's own count, or the node's overflow count, still scales past this.
pub const HINT_SCALE_MAX: f64 = 2.0;

/// A thread's own node-count-over-estimate ratio from its last turn, if it has one and it is over 1.
pub fn own_scale(meta: &ThreadMeta) -> Option<f64> {
    (meta.prompt_est > 0 && meta.prompt_tokens > meta.prompt_est).then(|| meta.prompt_tokens as f64 / meta.prompt_est as f64)
}

/// The node's count in a context-overflow message ("the prompt is N tokens"), if it says.
pub fn overflow_count(msg: &str) -> Option<u64> {
    let i = msg.find("prompt is ")? + "prompt is ".len();
    let digits: String = msg[i..].chars().take_while(|c| c.is_ascii_digit()).collect();
    digits.parse().ok().filter(|&n| n > 0)
}

/// A node's 400 that says the prompt does not fit its context: no retry fixes it.
pub fn is_context_overflow(msg: &str) -> bool {
    // the servers' own phrasings only: an overflow closes the thread, so any other 400 that happens to mention
    // context and tokens must not read as one (review 2026-11-11)
    let m = msg.to_ascii_lowercase();
    ["the context holds", "maximum context length", "exceeds the available context size", "exceed the context window", "exceeds the context window", "context_length_exceeded"]
        .iter().any(|p| m.contains(p))
}

/// The shared prefix: the system prompt, then the expert's own messages. The system message is always the
/// first and the same for every thread, so every node's cache serves it to all of them.
pub fn prefix(system: &str, store: &Store, meta: &ThreadMeta) -> anyhow::Result<Vec<Value>> {
    let mut p = vec![json!({"role": "system", "content": system})];
    if let Some(key) = &meta.expert {
        let e = store.expert(key)?.ok_or_else(|| anyhow::anyhow!("expert {key} is not stored"))?;
        p.extend(e.messages);
    }
    Ok(p)
}

/// The paging rule's `page_results_after_turns` for this thread: 0 (never advances) for a goldfish. A goldfish
/// lives a few turns on its expert's frozen state, so a paging batch only breaks the cache it was born to reuse;
/// an elephant lives long enough to need the room (dad 2026-11-09). Handles already shown stay.
pub fn page_after(after_turns: u64, meta: &ThreadMeta) -> u64 {
    if meta.goldfish_of.is_some() { 0 } else { after_turns }
}

/// What the model sees: the prefix, then (after a refocus) the bridge and notes, then (after a window cut) the
/// `[Paged.]` notice, then history from the window's start. A result whose call is not in the shown tail gets a
/// stub call on the notes or notice side so the template sees a call before every result. `pending` is the
/// thread's pending list as read at slice start.
pub fn live_view(prefix: &[Value], meta: &ThreadMeta, history: &[Value], pending: &[PendingCall], page_min_bytes: u64) -> Vec<Value> {
    let mut v: Vec<Value> = prefix.to_vec();
    let low = meta.compactions.last().map(|c| c.from as usize).unwrap_or(0).min(history.len());
    let from = (meta.window_from as usize).max(low).min(history.len());
    // pending calls need a stub only once the view no longer starts at the beginning (their own message may have left)
    let stubs = window::orphan_calls(history, from, if from > 0 { pending } else { &[] });
    let mut stubs_placed = false;
    if let Some(c) = meta.compactions.last() {
        let focus = c.focus.as_deref().map(|f| format!(" Focus: {f}")).unwrap_or_default();
        v.push(json!({"role": "user", "content": format!("[Refocus. The conversation so far is replaced by your notes, which follow.{focus}]")}));
        let mut notes = json!({"role": "assistant", "content": c.summary});
        if !stubs.is_empty() && from == low { notes["tool_calls"] = Value::Array(stubs.clone()); stubs_placed = true; }
        v.push(notes);
    }
    if from > low {
        let resident: Vec<u64> = meta.resident.iter().copied().filter(|&i| (i as usize) < from).collect();
        v.push(json!({"role": "user", "content": format!("{}{}", window::notice(history, low, from), window::resident_text(history, &resident))}));
        if !stubs.is_empty() { v.push(json!({"role": "assistant", "content": "", "tool_calls": stubs.clone()})); stubs_placed = true; }
    }
    if !stubs.is_empty() && !stubs_placed {
        // no refocus and no cut, yet a result's call is missing (a migrated history): a stub turn before the tail
        v.push(json!({"role": "user", "content": "[The calls these results answer were made before this view begins.]"}));
        v.push(json!({"role": "assistant", "content": "", "tool_calls": stubs}));
    }
    // old tool results below the paging mark are handles; the text stays stored and `recall` returns it
    let paged_before = (meta.paged_before as usize).min(history.len());
    let calls = if page_min_bytes > 0 && paged_before > from { window::calls_by_id(history) } else { Default::default() };
    for i in from..history.len() {
        if page_min_bytes > 0 && i < paged_before && window::pageable(history, i, page_min_bytes, &meta.pinned) {
            v.push(window::handle(i, &history[i], &calls));
        } else {
            v.push(history[i].clone());
        }
    }
    v
}

/// A `commission_expert` call checked and built during a slice, registered after its commit.
struct Commissioned {
    expert: crate::store::Expert,
    definition: Option<(std::path::PathBuf, String)>,
    thread: String,
    call: String,
    provider_call_id: Option<String>,
}

impl Engine {
    /// The instructions this turn runs under (the system prompt and the thread's tool definitions, by the hash an
    /// expert's frozen context carries too): recorded on the thread where they change, their text kept once under
    /// `<data_dir>/instructions/<id>.json`, so every turn in every log traces to exactly what the model was told
    /// (dad 2026-11-11: an instruction change invalidates what earlier logs say about a tool's use).
    fn note_instructions(&self, meta: &mut ThreadMeta, from: u64, defs: &[Value]) {
        let id = crate::experts::context(&self.system, defs);
        if meta.instructions.last().map(|(_, i)| i) != Some(&id) { meta.instructions.push((from, id.clone())); }
        if !self.instructions_seen.lock().unwrap().insert(id.clone()) { return; }
        let dir = self.cfg.data_dir.join("instructions");
        let body = json!({"id": id, "system": self.system, "tools": defs, "hoisted": self.hoisted, "build": crate::BUILD, "first_used_ms": now_ms()});
        let wrote = std::fs::create_dir_all(&dir)
            .and_then(|_| std::fs::OpenOptions::new().write(true).create_new(true).open(dir.join(format!("{id}.json"))))
            .and_then(|mut f| std::io::Write::write_all(&mut f, &serde_json::to_vec_pretty(&body).unwrap_or_default()));
        match wrote {
            Ok(()) => tracing::info!("instructions {id}: first used; kept in {}", dir.display()),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(e) => {
                tracing::warn!("instructions {id}: not kept ({e}); tried again on the next turn");
                self.instructions_seen.lock().unwrap().remove(&id);
            }
        }
    }

    pub async fn slice(&self, thread: &str, batch: &[(u64, Event)]) -> Outcome {
        match self.run(thread, batch).await {
            Ok(o) => o,
            Err(e) => Outcome::Transient(format!("store: {e:#}")),
        }
    }

    /// The live view of a stored thread, for the API.
    pub fn view(&self, thread: &str) -> anyhow::Result<Vec<Value>> {
        let meta = self.store.thread(thread)?.unwrap_or_default();
        let history = self.store.messages(thread)?;
        let p = prefix(&self.system, &self.store, &meta)?;
        Ok(live_view(&p, &meta, &history, &meta.pending, self.page_min()))
    }

    async fn run(&self, thread: &str, batch: &[(u64, Event)]) -> anyhow::Result<Outcome> {
        let mut meta = self.store.thread(thread)?.unwrap_or_default();
        // the view's named events: a cut, a paging batch, a refocus (images fade only at one of them)
        let marks = (meta.window_from, meta.paged_before, meta.compactions.len());
        if let Some(why) = &meta.closed {
            return Ok(Outcome::Fatal(format!("thread is closed: {why}")));
        }
        // events already run (a redrive of something that had committed) are consumed without effect
        let mut live: Vec<&(u64, Event)> = Vec::new();
        for e in batch {
            if self.store.is_processed(&e.1.id)? {
                tracing::info!("{thread}: event {} already ran, dropped", e.1.id);
            } else {
                live.push(e);
            }
        }
        let done: Vec<u64> = batch.iter().map(|(s, _)| *s).collect();
        let done_ids: Vec<String> = batch.iter().map(|(_, e)| e.id.clone()).collect();
        if live.is_empty() {
            self.store.commit(Commit { thread, meta, messages: vec![], done, done_ids, replies: vec![], events: vec![], invocations: vec![], new_threads: vec![] })?;
            return Ok(Outcome::Done);
        }

        // ---- the new messages ------------------------------------------------------------------
        let pending_at_start = meta.pending.clone();
        let mut new_messages: Vec<Value> = Vec::new();
        let mut reply_to: Option<String> = None;
        let mut refocus: Option<Option<String>> = None;
        // images go after every tool message of the batch: a user message between two results would split them
        let mut image_messages: Vec<Value> = Vec::new();
        for (_, ev) in &live {
            match &ev.kind {
                EventKind::User { text, .. } if !self.cfg.test_panic_on.is_empty() && *text == self.cfg.test_panic_on => {
                    panic!("test_panic_on: {text}");
                }
                EventKind::User { text, .. } => {
                    // a person may speak while tool calls are outstanding: the calls stay pending and their
                    // results land later as tool messages (the templates render that)
                    new_messages.push(json!({"role": "user", "content": text}));
                    reply_to = Some(ev.id.clone());
                }
                EventKind::ToolResult { call, text, images, .. } => {
                    // a result for a call nobody is waiting on (a late one, a thread moved here from another
                    // host) is said in the user role: a tool message with no call before it breaks the template
                    // the stored tool message names the call by the id the assistant message carries (the model's
                    // own when it gave one), so a template or validator that pairs them finds the pair
                    let awaited = meta.pending.iter().any(|p| &p.id == call);
                    let shown_id = meta.pending.iter().find(|p| &p.id == call).and_then(|p| p.provider_call_id.clone()).unwrap_or_else(|| call.clone());
                    meta.pending.retain(|p| &p.id != call);
                    let text = if images.is_empty() { text.clone() } else { format!("{text}\n[{} image(s) follow]", images.len()) };
                    new_messages.push(if awaited {
                        json!({"role": "tool", "tool_call_id": shown_id, "content": text})
                    } else {
                        json!({"role": "user", "content": format!("[late result for tool call {call}] {text}")})
                    });
                    if !images.is_empty() { image_messages.push(window::image_message(&format!("[image(s) from tool call {shown_id}]"), images)); }
                }
                EventKind::Subscription { call, text, r#final } => {
                    if *r#final { meta.pending.retain(|p| &p.id != call); }
                    new_messages.push(json!({"role": "user", "content": format!("[event for tool call {call}] {text}")}));
                }
                EventKind::Compact { focus } => {
                    if self.cfg.compaction { refocus = Some(focus.clone()); } else { tracing::warn!("{thread}: a refocus was asked for but compaction is off"); }
                }
            }
        }
        new_messages.extend(image_messages);
        if reply_to.is_none() { reply_to = live.last().map(|(_, e)| e.id.clone()); }

        let prefix = prefix(&self.system, &self.store, &meta)?;
        let history = self.store.messages(thread)?;
        let tools = self.tools.current().await;
        let mut defs = rap::builtin_defs(self.cfg.compaction, self.cfg.window);
        defs.extend(tools.defs.iter().cloned());
        let defs = self.experts.with_tools(defs).await;
        // a thread born with a tool list (a goldfish, an expert that declares one) sees and may call only those
        let defs = rap::scope_defs(defs, meta.tools.as_deref());
        let allowed: std::collections::HashSet<String> = defs.iter().filter_map(|d| d.pointer("/function/name").and_then(|n| n.as_str()).map(str::to_owned)).collect();
        self.note_instructions(&mut meta, history.len() as u64, &defs);
        let mut asked = 0usize;

        // ---- refocus -----------------------------------------------------------------------------
        let mut view = live_view(&prefix, &meta, &history, &pending_at_start, self.page_min());
        let mut est = calibrated(tokens(&view) + tokens(&new_messages), self.scale(&meta));
        if self.cfg.compaction && (refocus.is_some() || (self.cfg.compact_at_tokens > 0 && est > self.cfg.compact_at_tokens)) {
            let focus = refocus.clone().flatten();
            match self.refocus(thread, &meta, &view, &history, &defs, focus, est).await? {
                Refocused::Done(c) => {
                    meta.window_from = c.from;
                    meta.compactions.push(c);
                    meta.node = None;
                    view = live_view(&prefix, &meta, &history, &pending_at_start, self.page_min());
                    est = calibrated(tokens(&view) + tokens(&new_messages), self.scale(&meta));
                    if self.cfg.compact_at_tokens > 0 && est > self.cfg.compact_at_tokens {
                        tracing::warn!("{thread}: still ~{est} tokens after refocusing (the kept tail or the new messages are large)");
                    }
                }
                Refocused::Nothing => {
                    if refocus.is_some() { tracing::info!("{thread}: refocus asked for, nothing to fold yet"); }
                }
                Refocused::Failed(why) => return Ok(Outcome::Transient(why)),
                Refocused::Stopping => return Ok(Outcome::Requeue),
            }
        }
        if new_messages.is_empty() {
            // only a refocus was asked for: no turn
            self.store.commit(Commit { thread, meta, messages: vec![], done, done_ids, replies: vec![], events: vec![], invocations: vec![], new_threads: vec![] })?;
            return Ok(Outcome::Done);
        }

        // ---- paging of old tool results ------------------------------------------------------------
        let after = page_after(self.cfg.page_results_after_turns, &meta);
        if after > 0 {
            let low = meta.compactions.last().map(|c| c.from as usize).unwrap_or(0);
            let from = (meta.window_from as usize).max(low).min(history.len());
            let boundary = window::page_boundary(&history, from, after);
            if let Some(to) = window::page_advance(&history, from, meta.paged_before as usize, boundary, self.cfg.page_results_min_bytes, self.cfg.page_results_batch, &meta.pinned) {
                let n = (from.max(meta.paged_before as usize)..to).filter(|&i| window::pageable(&history, i, self.cfg.page_results_min_bytes, &meta.pinned)).count();
                meta.paged_before = to as u64;
                view = live_view(&prefix, &meta, &history, &pending_at_start, self.page_min());
                let before = est;
                est = calibrated(tokens(&view) + tokens(&new_messages), self.scale(&meta));
                tracing::info!("{thread}: paged {n} tool results below #{to} (view ~{before} -> ~{est} tokens)");
            }
        }

        // ---- the window --------------------------------------------------------------------------
        if let Some((v, e)) = self.window_cut(thread, &mut meta, &prefix, &history, &pending_at_start, &new_messages, est).await {
            view = v;
            est = e;
        }

        // ---- room in the context -----------------------------------------------------------------
        let model = self.model_of(&meta);
        let limit = if self.cfg.max_context_tokens > 0 { self.cfg.max_context_tokens } else { self.router.context_length_for(&model).await.saturating_sub(self.cfg.max_output_tokens) };
        if limit > 0 && est > limit {
            return Ok(Outcome::Fatal(format!("context full: about {est} tokens, the limit is {limit}")));
        }

        // ---- the model's turn --------------------------------------------------------------------
        let mut messages = view.clone();
        messages.extend(new_messages.iter().cloned());
        if marks != (meta.window_from, meta.paged_before, meta.compactions.len()) {
            self.keep_newest_images(&mut meta, &history, &new_messages);
        }
        self.fade(&mut messages, &meta, &history);
        let sent_est = tokens(&messages);
        let mut body = json!({"model": model, "messages": messages, "tools": defs, "max_tokens": self.cfg.max_output_tokens, "stream": false});
        self.steer_into(&mut body, &meta);

        let mut reply_text = String::new();
        let mut events: Vec<Event> = Vec::new();
        let mut invocations: Vec<Invocation> = Vec::new();
        // goldfish threads and commissioned experts take effect with the commit, not before it: a slice that fails or
        // is cancelled and runs again would otherwise leave an orphan goldfish, or find its own expert "exists
        // already" while the first freeze's result arrives late (review 2026-11-11)
        let mut new_threads: Vec<(String, ThreadMeta)> = Vec::new();
        let mut commissions: Vec<Commissioned> = Vec::new();
        let mut rounds = 0;
        let mut recut = false;
        let mut cut_off = 0;
        let mut waits_for_event = false;
        let mut restore = self.parking.as_ref().and_then(|p| p.has(thread));
        // a thread born from an expert, with no node holding it yet: the expert's frozen state goes into whichever
        // node takes the turn; not frozen for this context yet, it is frozen now for the next thread
        if let (None, None, Some(key)) = (&restore, &meta.node, &meta.expert) {
            let ctx = crate::experts::context(&self.system, &defs);
            restore = self.experts.state(key, &ctx);
            if restore.is_none() {
                let (experts, store, router, system, defs, key, max) = (self.experts.clone(), self.store.clone(), self.router.clone(), self.system.clone(), defs.clone(), key.clone(), self.cfg.max_output_tokens);
                tokio::spawn(async move {
                    if let Err(e) = experts.freeze(&key, &store, &router, &system, &defs, max).await { tracing::info!("expert {key}: not frozen: {e:#}"); }
                });
            }
        }
        loop {
            rounds += 1;
            let r = restore.take();
            let turn = match self.router.chat_restoring(Some(thread), meta.node.as_deref(), &body, r.as_deref()).await {
                Ok(t) => t,
                // the node says the prompt does not fit. With the window on and the node's own count in the message,
                // the estimate was wrong: learn the ratio, cut, and send once more; otherwise the thread is done
                Err(ChatError::Rejected(e)) if is_context_overflow(&e) => {
                    if self.cfg.window && rounds == 1 && !recut {
                        if let Some(n) = overflow_count(&e) {
                            meta.prompt_tokens = n;
                            meta.prompt_est = sent_est.max(1);
                            self.scale_hint.store((own_scale(&meta).unwrap_or(1.0) * 1000.0) as u64, std::sync::atomic::Ordering::Relaxed);
                            let est = calibrated(tokens(&view) + tokens(&new_messages), self.scale(&meta));
                            tracing::warn!("{thread}: the node counted {n} tokens where the estimate said {sent_est}; recounting at that ratio and cutting");
                            if let Some((v, _)) = self.window_cut(thread, &mut meta, &prefix, &history, &pending_at_start, &new_messages, est).await {
                                let mut msgs = v;
                                msgs.extend(new_messages.iter().cloned());
                                self.keep_newest_images(&mut meta, &history, &new_messages);
                                self.fade(&mut msgs, &meta, &history);
                                body["messages"] = Value::Array(msgs);
                                recut = true;
                                rounds -= 1;
                                continue;
                            }
                        }
                    }
                    return Ok(Outcome::Fatal(format!("context full: {e}")));
                }
                // a node refusing the thread's steering (no such vector, no steer-dir) refuses every retry the same
                Err(ChatError::Rejected(e)) if body.get("steer").is_some() && e.contains("steer") => {
                    return Ok(Outcome::Fatal(format!("the node refuses this thread's steering: {e}")));
                }
                Err(ChatError::Stopping) => return Ok(Outcome::Requeue),
                Err(ChatError::Transient(e)) | Err(ChatError::Rejected(e)) => return Ok(Outcome::Transient(e)),
            };
            meta.node = Some(turn.node.clone());
            self.metrics.turn(&turn.node, turn.prompt_tokens, turn.cached_tokens, turn.ran_ms, turn.waited_ms);
            // the parenthesis closes right after the cache count: harvesters match "(N prompt tokens, M from its cache)"
            let timing = format!(" in {:.0} s{}", turn.ran_ms as f64 / 1000.0,
                                 if turn.waited_ms >= 1000 { format!(", after {} s waiting for a slot", turn.waited_ms / 1000) } else { String::new() });
            if let (Some(p), Some(c)) = (turn.prompt_tokens, turn.cached_tokens) {
                tracing::info!("{thread}: turn on {} ({p} prompt tokens, {c} from its cache){timing}", turn.node);
            } else {
                tracing::info!("{thread}: turn on {}{timing}", turn.node);
            }
            if let Some(p) = turn.prompt_tokens.filter(|&p| p > 0) {
                meta.prompt_tokens = p;
                // the node's count includes the tool definitions and the template, which the estimate leaves out:
                // learn a ratio only when the messages outweigh them, else a short thread's ratio runs to 10
                meta.prompt_est = if overhead_tokens(&body) * 4 <= sent_est { sent_est.max(1) } else { 0 };
                if let Some(r) = own_scale(&meta) { self.scale_hint.store((r * 1000.0) as u64, std::sync::atomic::Ordering::Relaxed); }
            }
            let (assistant, calls) = normalize(&turn.message);
            new_messages.push(assistant.clone());
            if let Some(t) = assistant.get("content").and_then(|c| c.as_str()) {
                if !t.trim().is_empty() {
                    if !reply_text.is_empty() { reply_text.push('\n'); }
                    reply_text.push_str(t.trim());
                }
            }
            // cut off at the output limit with no whole tool call: the model was acting (a file too long for one
            // reply, usually), so the text is not its answer and the slice must not end on it (a goldfish would
            // answer with it, and its elephant wait on a call that never comes). Say so and let it go on.
            if calls.is_empty() && turn.finish_reason.as_deref() == Some("length") && cut_off < CUT_OFF_ROUNDS {
                cut_off += 1;
                tracing::warn!("{thread}: the reply hit the output limit ({} tokens) with no whole tool call; asking it to go on in smaller calls", self.cfg.max_output_tokens);
                let note = json!({"role": "user", "content": format!("[FreeFitty] Your reply reached the output limit ({} tokens) and was cut off; a tool call in it was lost. Go on from where you were, keeping each tool call well under the limit: write a long file in parts (write_file the first part, then add the rest with further calls).", self.cfg.max_output_tokens)});
                new_messages.push(note.clone());
                let mut msgs = body["messages"].as_array().cloned().unwrap_or_default();
                msgs.push(assistant);
                msgs.push(note);
                body["messages"] = Value::Array(msgs);
                continue;
            }
            let mut unknown = Vec::new();
            let mut recalled: Vec<(Call, String)> = Vec::new();
            for call in &calls {
                match call.name.as_str() {
                    rap::EXPERTS => recalled.push((call.clone(), self.experts.catalog(&self.store).await)),
                    rap::GOLDFISH if allowed.contains(rap::GOLDFISH) => match self.goldfish(thread, call).await {
                        Ok((child_meta, first)) => {
                            tracing::info!("{thread}: asks goldfish {}", first.thread);
                            new_threads.push((first.thread.clone(), child_meta));
                            events.push(first);
                            meta.pending.push(call.pending());
                            asked += 1;
                            reply_text.push_str(&format!("\n[Tool Call: goldfish with arguments {}]", call.args));
                        }
                        Err(e) => recalled.push((call.clone(), format!("Error: {e:#}"))),
                    },
                    rap::COMMISSION if allowed.contains(rap::COMMISSION) => match self.commission(thread, call, &commissions).await {
                        Ok(c) => {
                            commissions.push(c);
                            meta.pending.push(call.pending());
                            asked += 1;
                            reply_text.push_str(&format!("\n[Tool Call: commission_expert with arguments {}]", call.args));
                        }
                        Err(e) => recalled.push((call.clone(), format!("Error: {e:#}"))),
                    },
                    rap::RECALL if self.cfg.window => {
                        // answered from the store within the slice: a fault costs one more model pass, no tool round trip
                        let text = match window::recall_indices(&call.args, history.len()) {
                            Ok(idx) => {
                                // what the model asked for again stays in full from now on
                                for &i in &idx { if !meta.pinned.contains(&(i as u64)) { meta.pinned.push(i as u64); } }
                                window::recall_text(&history, &idx)
                            }
                            Err(e) => format!("Error: {e}"),
                        };
                        recalled.push((call.clone(), text));
                    }
                    rap::SLEEP => {
                        let secs = call.args.get("seconds").and_then(|s| s.as_f64()).unwrap_or(0.0).clamp(0.0, 366.0 * 86400.0);
                        events.push(wake(thread, call, now_ms() + (secs * 1000.0) as u64, format!("Slept for {secs} seconds")));
                        meta.pending.push(call.pending());
                        reply_text.push_str(&format!("\n[Tool Call: sleep with arguments {}]", call.args));
                    }
                    rap::SLEEP_UNTIL => {
                        match until(&call.args) {
                            Ok(due) => {
                                events.push(wake(thread, call, due, format!("Woke up: it is now {}", call.args.get("date").and_then(|d| d.as_str()).unwrap_or(""))));
                                meta.pending.push(call.pending());
                            }
                            Err(e) => events.push(wake(thread, call, 0, format!("Error: {e}"))),
                        }
                        reply_text.push_str(&format!("\n[Tool Call: sleep_until with arguments {}]", call.args));
                    }
                    rap::SLEEP_UNTIL_EVENT => {
                        // nothing is pending for this call: the next event or message wakes the thread anyway. Its
                        // result is written now, so the transcript is whole and a watcher does not read a stall
                        new_messages.push(json!({"role": "tool", "tool_call_id": call.shown_id(), "content": "Waiting for the next event or message."}));
                        waits_for_event = true;
                        reply_text.push_str(&format!("\n[Tool Call: sleep_until_event_or_input with arguments {}]", call.args));
                    }
                    rap::REFOCUS if self.cfg.compaction => {
                        // the refocus runs at the start of the next slice, before the result is answered
                        let focus = call.args.get("focus").and_then(|f| f.as_str()).map(|f| f.chars().take(2000).collect::<String>());
                        events.push(Event { id: format!("refocus-{}", call.id), thread: thread.to_owned(), due_ms: 0, created_ms: 0, attempts: 0, kind: EventKind::Compact { focus } });
                        events.push(wake(thread, call, 0, "Refocused: your notes now stand in for the conversation before them.".into()));
                        meta.pending.push(call.pending());
                        reply_text.push_str(&format!("\n[Tool Call: refocus with arguments {}]", call.args));
                    }
                    name => match tools.get(name).filter(|_| allowed.contains(name)) {
                        Some(tool) => {
                            invocations.push(Invocation {
                                call: call.id.clone(),
                                thread: thread.to_owned(),
                                endpoint: tool.endpoint.clone(),
                                body: rap::invocation(thread, &call.id, call.provider_call_id.as_deref(), name, call.args.clone(), &self.callback_url),
                                attempts: 0,
                                created_ms: 0,
                                next_ms: 0,
                            });
                            meta.pending.push(call.pending());
                            reply_text.push_str(&format!("\n[Tool Call: {name} with arguments {}]", call.args));
                        }
                        None => match self.experts.tool(name).await.filter(|_| allowed.contains(name)) {
                            // a tool an expert answers: a goldfish of it, briefed with the call's arguments
                            Some((expert, spec)) if meta.goldfish_of.is_none() => {
                                let ask = Call { args: json!({"expert": expert, "question": spec.brief(&call.args)}), ..call.clone() };
                                match self.goldfish(thread, &ask).await {
                                    Ok((child_meta, first)) => {
                                        tracing::info!("{thread}: {name} goes to goldfish {}", first.thread);
                                        new_threads.push((first.thread.clone(), child_meta));
                                        events.push(first);
                                        meta.pending.push(call.pending());
                                        asked += 1;
                                        reply_text.push_str(&format!("\n[Tool Call: {name} with arguments {}]", call.args));
                                    }
                                    Err(e) => recalled.push((call.clone(), format!("Error: {e:#}"))),
                                }
                            }
                            Some((expert, _)) => recalled.push((call.clone(), format!("Error: {name} is answered by a goldfish of the expert {expert}, and a goldfish cannot ask one; say in your answer that {name} should be called"))),
                            None => unknown.push(call.clone()),
                        },
                    },
                }
            }
            if unknown.is_empty() && recalled.is_empty() { break; }
            // recalls are answered here; tools that do not exist are explained; both within the same slice
            let names: Vec<String> = defs.iter().filter_map(|d| d.pointer("/function/name").and_then(|n| n.as_str()).map(str::to_owned)).collect();
            let answered = unknown.len() + recalled.len();
            for (c, text) in &recalled {
                new_messages.push(json!({"role": "tool", "tool_call_id": c.shown_id(), "content": text}));
                reply_text.push_str(&format!("\n[Tool Call: recall with arguments {}]", c.args));
            }
            for u in &unknown {
                new_messages.push(json!({"role": "tool", "tool_call_id": u.shown_id(), "content": format!("Error: there is no tool named {:?}. The tools are: {}", u.name, names.join(", "))}));
            }
            if !unknown.is_empty() && (rounds >= UNKNOWN_TOOL_ROUNDS || !events.is_empty() || !invocations.is_empty() || asked > 0) {
                reply_text.push_str(&format!("\n[Error: the model called tools that do not exist: {}]", unknown.iter().map(|u| u.name.as_str()).collect::<Vec<_>>().join(", ")));
                break;
            }
            // a turn that also slept or called a real tool ends here; the recall's text waits in the history for its next turn
            if rounds >= UNKNOWN_TOOL_ROUNDS || !events.is_empty() || !invocations.is_empty() || asked > 0 { break; }
            let mut msgs = body["messages"].as_array().cloned().unwrap_or_default();
            msgs.extend(new_messages[new_messages.len() - 1 - answered..].iter().cloned());
            body["messages"] = Value::Array(msgs);
        }

        // ---- commit ------------------------------------------------------------------------------
        let mut replies = Vec::new();
        let text = reply_text.trim().to_owned();
        if !text.is_empty() {
            replies.push((text, reply_to, "text".to_owned(), meta.metadata.clone()));
        }
        // parking: at once for a long sleep or a wait on events alone; after park_after_s for a tool call out or
        // a finished turn; never for a sleep that ends sooner (the node still has the state then)
        // (a goldfish asked or an expert commissioned counts as a tool call out; a goldfish's first event is
        // its own thread's, not this one's)
        let after_ms = self.cfg.park_after_s * 1000;
        let out = !invocations.is_empty() || asked > 0;
        let park_in = match events.iter().filter(|e| e.thread == thread).map(|e| e.due_ms).min() {
            Some(due) if !out && due >= now_ms() + after_ms => Some(0),
            Some(_) => None,
            None if waits_for_event && !out => Some(0),
            None => Some(after_ms),
        };
        // a goldfish that ended a turn with nothing out has answered: its text returns to the elephant as the
        // call's result, and the goldfish is done
        if let Some(g) = meta.goldfish_of.clone() {
            if !out && meta.pending.is_empty() && !events.iter().any(|e| e.thread == thread) {
                let answer = if text_for_elephant(&new_messages).is_empty() { "(the goldfish ended without an answer)".to_owned() } else { text_for_elephant(&new_messages) };
                events.push(Event { id: format!("goldfish-{}", g.call), thread: g.thread.clone(), due_ms: 0, created_ms: 0, attempts: 0,
                                    kind: EventKind::ToolResult { call: g.call.clone(), provider_call_id: g.provider_call_id.clone(), text: answer, images: vec![] } });
                meta.closed = Some("answered its elephant".into());
                tracing::info!("{thread}: answered {}", g.thread);
            }
        }
        let node = meta.node.clone();
        let closed = meta.closed.is_some();
        let covered = meta.node.as_deref().map_or(meta.prompt_tokens.saturating_sub(64), |n| self.cfg.min_cover(n, meta.prompt_tokens));
        self.store.commit(Commit { thread, meta, messages: new_messages, done, done_ids, replies, events, invocations, new_threads })?;
        for c in commissions { self.register(c).await; }
        if let Some(p) = &self.parking {
            // the node's cache is newer than any parked copy from here on, including one whose write began
            // before this slice and ended during it
            p.forget(thread);
            // a closed thread (a goldfish that answered) never runs again: nothing to park
            if let (Some(node), Some(delay), false) = (node, park_in, closed) {
                self.park_later(thread, node, delay, covered);
            }
        }
        Ok(Outcome::Done)
    }

    /// At a named event: images before the newest `keep_images` (counting this turn's) fade from now on.
    fn keep_newest_images(&self, meta: &mut ThreadMeta, history: &[Value], new_messages: &[Value]) {
        let total = window::count_images(history) + window::count_images(new_messages);
        meta.images_kept_from = meta.images_kept_from.max(total.saturating_sub(self.cfg.keep_images as u64));
    }

    /// Stub the images the thread's fade mark says are out of view, in messages whose history part starts at the
    /// view's start.
    fn fade(&self, messages: &mut [Value], meta: &ThreadMeta, history: &[Value]) {
        let low = meta.compactions.last().map(|c| c.from as usize).unwrap_or(0);
        let from = (meta.window_from as usize).max(low).min(history.len());
        window::fade_images_before(messages, window::count_images(&history[..from]), meta.images_kept_from);
    }

    /// The model this thread runs on: its expert's, else the config's (the router places the turn on a node
    /// serving it).
    fn model_of(&self, meta: &ThreadMeta) -> String {
        meta.expert.as_ref().and_then(|k| self.store.expert(k).ok().flatten()).and_then(|e| e.model).unwrap_or_else(|| self.cfg.model.clone())
    }

    /// The steering every request of this thread carries: its expert's (a thread born from a steered expert is
    /// that persona on every turn, and the nodes cache its state under the steering's own key).
    fn steer_into(&self, body: &mut Value, meta: &ThreadMeta) {
        if let Some(e) = meta.expert.as_ref().and_then(|k| self.store.expert(k).ok().flatten()) {
            crate::experts::steer_into(body, &e);
            crate::experts::shared_into(body, &e, self.cfg.takes_shared_messages(&self.model_of(meta)));
        }
    }

    /// A goldfish for a `goldfish` call: its thread's meta and first event. Born from the expert, with the
    /// expert's tool list (or the call's), never the tools only an elephant may call.
    async fn goldfish(&self, thread: &str, call: &Call) -> anyhow::Result<(ThreadMeta, Event)> {
        let name = call.args.get("expert").and_then(|e| e.as_str()).unwrap_or("");
        let question = call.args.get("question").and_then(|q| q.as_str()).unwrap_or("").trim();
        if question.is_empty() { anyhow::bail!("goldfish needs a question"); }
        let Some(key) = self.experts.key(name).await else {
            anyhow::bail!("there is no expert named {name:?}.\n{}", self.experts.catalog(&self.store).await)
        };
        let expert = self.store.expert(&key)?.ok_or_else(|| anyhow::anyhow!("expert {key} is not stored"))?;
        let granted: Vec<String> = match call.args.get("tools").and_then(|t| t.as_array()) {
            Some(a) => a.iter().filter_map(|t| t.as_str().map(str::to_owned)).collect(),
            None => expert.tools.clone().unwrap_or_default(),
        };
        let mut text = String::from("You are a goldfish: you were born for this one brief and answer it once; your answer returns to the agent that asked. You cannot ask goldfish or commission experts. If another expert should look at something, say so in your answer and name it (`experts` lists them).\n\n");
        let mut total = 0usize;
        for p in call.args.get("documents").and_then(|d| d.as_array()).into_iter().flatten().filter_map(|p| p.as_str()) {
            let doc = std::fs::read_to_string(self.cfg.model_read(p)?).map_err(|e| anyhow::anyhow!("reading document {p}: {e}"))?;
            if doc.len() > 256 << 10 { anyhow::bail!("document {p} is {} bytes, over 256 KiB", doc.len()); }
            total += doc.len();
            if total > 1 << 20 { anyhow::bail!("the documents are over 1 MiB together"); }
            text.push_str(&format!("<document path=\"{p}\">\n{doc}\n</document>\n\n"));
        }
        text.push_str(question);
        let tag: String = call.id.chars().filter(|c| c.is_ascii_alphanumeric()).take(8).collect();
        let mut child = format!("{thread}.gf-{name}-{tag}");
        if child.len() > 128 { child = format!("gf-{name}-{tag}-{}", crate::config::id()).chars().take(128).collect(); }
        let meta = ThreadMeta {
            created_ms: now_ms(), updated_ms: now_ms(),
            metadata: json!({"goldfish_of": thread, "expert": key}),
            expert: Some(key),
            tools: Some(rap::goldfish_tools(&granted)),
            goldfish_of: Some(GoldfishOf { thread: thread.to_owned(), call: call.id.clone(), provider_call_id: call.provider_call_id.clone() }),
            ..Default::default()
        };
        let first = Event { id: format!("gf-{}", call.id), thread: child, due_ms: 0, created_ms: 0, attempts: 0, kind: EventKind::User { text, metadata: Value::Null } };
        Ok((meta, first))
    }

    /// Check and build a `commission_expert` call's expert; `register` makes it real once the slice commits.
    async fn commission(&self, thread: &str, call: &Call, earlier: &[Commissioned]) -> anyhow::Result<Commissioned> {
        let a = &call.args;
        let name = a.get("name").and_then(|n| n.as_str()).unwrap_or("");
        if self.experts.key(name).await.is_some() || earlier.iter().any(|c| c.expert.name == name) { anyhow::bail!("an expert named {name:?} exists already"); }
        let reads: Vec<crate::experts::Read> = serde_json::from_value(a.get("reads").cloned().unwrap_or(json!([])))
            .map_err(|e| anyhow::anyhow!("reads must be a list of {{path}} or {{text}}: {e}"))?;
        let tools: Vec<String> = a.get("tools").and_then(|t| t.as_array()).map(|t| t.iter().filter_map(|x| x.as_str().map(str::to_owned)).collect()).unwrap_or_default();
        let steer: Vec<crate::experts::Steer> = serde_json::from_value(a.get("steer").cloned().unwrap_or(json!([])))
            .map_err(|e| anyhow::anyhow!("steer must be a list of {{vector, scale, layer?}}: {e}"))?;
        let steer_scope = a.get("steer_scope").and_then(|s| s.as_str());
        crate::experts::steer_scope_of(steer_scope, &steer)?;
        if !steer.is_empty() {
            if !self.cfg.commission_steer {
                anyhow::bail!("commissioned experts cannot be steered yet (commission_steer is off until a steered goldfish is measured against a plain one); commission it without steer");
            }
            if let Some(s) = steer.iter().find(|s| !s.scale.is_finite() || s.scale.abs() > self.cfg.max_steer_scale) {
                anyhow::bail!("steer scale {} for {} is over the limit of {}", s.scale, s.vector, self.cfg.max_steer_scale);
            }
            // checked now, against what the nodes hold: a wrong name would fail every goldfish of the expert later
            match self.router.steer_vectors().await {
                Ok(held) => if let Some(s) = steer.iter().find(|s| !held.contains_key(&s.vector)) {
                    let mut names: Vec<&String> = held.keys().collect();
                    names.sort();
                    anyhow::bail!("not every node holds a steering vector named {:?} (a thread may run on any of them); all of them hold: {}", s.vector, if names.is_empty() { "none".to_owned() } else { names.iter().map(|n| n.as_str()).collect::<Vec<_>>().join(", ") });
                },
                Err(e) => anyhow::bail!("the steering could not be checked: {e}"),
            }
            if steer_scope == Some("prefix") {
                match self.router.steer_scope_everywhere().await {
                    Ok(true) => {}
                    Ok(false) => anyhow::bail!("steer_scope \"prefix\" needs every node to take steer_scope; one does not (commission with scope \"all\", or not at all)"),
                    Err(e) => anyhow::bail!("steer_scope could not be checked: {e}"),
                }
            }
        }
        let (expert, definition) = crate::experts::commission(&self.cfg, name, a.get("description").and_then(|d| d.as_str()).unwrap_or(""), a.get("ack").and_then(|d| d.as_str()), &reads, rap::goldfish_tools(&tools), steer, steer_scope, thread)?;
        Ok(Commissioned { expert, definition, thread: thread.to_owned(), call: call.id.clone(), provider_call_id: call.provider_call_id.clone() })
    }

    /// A committed commission: the expert is registered now and frozen in the background; the result returns to
    /// the caller when the freeze ends, or at once when registering fails.
    async fn register(&self, c: Commissioned) {
        let Commissioned { expert: e, definition, thread, call: call_id, provider_call_id: provider } = c;
        let deliver = self.deliver.clone();
        let result = move |text: String| if let Some(d) = deliver.get() {
            d(Event { id: format!("commission-{call_id}"), thread, due_ms: 0, created_ms: 0, attempts: 0,
                      kind: EventKind::ToolResult { call: call_id, provider_call_id: provider, text, images: vec![] } });
        };
        let added = match definition.as_ref().map(crate::experts::write_definition).transpose() {
            Ok(_) => self.experts.add(&self.store, &e).await,
            Err(err) => Err(err),
        };
        let key = match added {
            Ok(k) => k,
            Err(err) => return result(format!("Error: expert {} could not be registered: {err:#}", e.name)),
        };
        tracing::info!("commissioned expert {key}");
        // frozen in the context its goldfish will see: the plain built-ins and its own tools (from the whole set,
        // not the commissioning thread's, which may be scoped)
        let mut full = rap::builtin_defs(self.cfg.compaction, self.cfg.window);
        full.extend(self.tools.current().await.defs.iter().cloned());
        let full = self.experts.with_tools(full).await;
        let scoped = rap::scope_defs(full, e.tools.as_deref());
        let (experts, store, router, system, max) = (self.experts.clone(), self.store.clone(), self.router.clone(), self.system.clone(), self.cfg.max_output_tokens);
        tokio::spawn(async move {
            let text = match experts.freeze(&key, &store, &router, &system, &scoped, max).await {
                Ok(r) => format!("Expert {key} is ready: frozen from {} prompt tokens ({:.2} GB of cached state). Ask it with goldfish {{\"expert\": \"{}\", ...}}.",
                                 r["prompt_tokens"].as_u64().unwrap_or(0), r["bytes"].as_u64().unwrap_or(0) as f64 / 1e9, key.split('@').next().unwrap_or("")),
                Err(e) => format!("Expert {key} is registered, but freezing it failed ({e:#}); a goldfish born from it reads it instead."),
            };
            result(text);
        });
    }

    /// The error result a failed goldfish owes its elephant, if the thread is one.
    pub fn goldfish_failed(&self, thread: &str, why: &str) -> Option<Event> {
        let g = self.store.thread(thread).ok().flatten()?.goldfish_of?;
        Some(Event { id: format!("goldfish-{}", g.call), thread: g.thread, due_ms: 0, created_ms: 0, attempts: 0,
                     kind: EventKind::ToolResult { call: g.call, provider_call_id: g.provider_call_id, text: format!("Error: the goldfish failed: {why}"), images: vec![] } })
    }

    /// Park the thread's state `delay_ms` from now unless it has run another turn by then. The request the export
    /// is sent with is rebuilt then from the store (the view the node cached, as of the last turn), so a sleeper
    /// holds no messages or images in memory while it waits; the engine is held weakly until then.
    fn park_later(&self, thread: &str, node: String, delay_ms: u64, min_tokens: u64) {
        let (Some(park), Ok(Some(m)), Some(me)) = (self.parking.clone(), self.store.thread(thread), self.me.get().cloned()) else { return };
        let (thread, stamp) = (thread.to_owned(), m.updated_ms);
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
            let Some(engine) = me.upgrade() else { return };
            // no turn committed since, and none running now (a turn waiting for a slot would never read the copy)
            let unchanged = |e: &Engine| !e.inflight.get().is_some_and(|f| f(&thread)) && matches!(e.store.thread(&thread), Ok(Some(m)) if m.updated_ms == stamp);
            if !unchanged(&engine) { return; }
            let Some(body) = engine.park_body(&thread).await else { return };
            let path = park.path(&thread);
            let t = std::time::Instant::now();
            match engine.router.park_state(&node, &body, &path, min_tokens).await {
                Ok(Some(n)) => {
                    if unchanged(&engine) {
                        tracing::info!("{thread}: parked its cached state from {node} ({:.1} GB in {:.1}s)", n as f64 / 1e9, t.elapsed().as_secs_f64());
                        park.prune();
                    } else {
                        park.forget(&thread);  // a turn ran while it was written
                    }
                }
                Ok(None) => tracing::info!("{thread}: nothing to park, {node} no longer holds its state"),
                Err(e) => tracing::warn!("{thread}: parking its state from {node} failed: {e:#}"),
            }
        });
    }

    /// The request a parked thread's state is exported with: its view now (what the node cached for its last
    /// turn, extended by the reply), with the thread's tools and steering, as a turn would send it.
    async fn park_body(&self, thread: &str) -> Option<Value> {
        let meta = self.store.thread(thread).ok()??;
        let history = self.store.messages(thread).ok()?;
        let p = prefix(&self.system, &self.store, &meta).ok()?;
        let mut defs = rap::builtin_defs(self.cfg.compaction, self.cfg.window);
        defs.extend(self.tools.current().await.defs.iter().cloned());
        let defs = rap::scope_defs(self.experts.with_tools(defs).await, meta.tools.as_deref());
        let mut messages = live_view(&p, &meta, &history, &meta.pending, self.page_min());
        self.fade(&mut messages, &meta, &history);
        let mut body = json!({"model": self.model_of(&meta), "messages": messages, "tools": defs, "max_tokens": self.cfg.max_output_tokens, "stream": false});
        self.steer_into(&mut body, &meta);
        Some(body)
    }

    /// The paging size threshold, or 0 while paging is off.
    fn page_min(&self) -> u64 {
        if self.cfg.page_results_after_turns > 0 { self.cfg.page_results_min_bytes.max(1) } else { 0 }
    }

    /// How far over bytes / 4 to count this thread: its own last ratio, else the latest any thread saw, else the
    /// config's default.
    fn scale(&self, meta: &ThreadMeta) -> f64 {
        own_scale(meta).unwrap_or_else(|| {
            let hint = self.scale_hint.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1000.0;
            if hint > 1.0 { hint.min(HINT_SCALE_MAX) } else { self.cfg.est_scale_default }
        })
    }

    /// Move the window if the view (`est`, calibrated) is over the budget. Returns the new view and estimate when
    /// it moved.
    #[allow(clippy::too_many_arguments)]
    async fn window_cut(&self, thread: &str, meta: &mut ThreadMeta, prefix: &[Value], history: &[Value], pending: &[PendingCall], new_messages: &[Value], est: u64) -> Option<(Vec<Value>, u64)> {
        if !self.cfg.window { return None; }
        let (budget, target) = self.window_limits(&self.model_of(meta)).await?;
        if est <= budget { return None; }
        let low = meta.compactions.last().map(|c| c.from as usize).unwrap_or(0);
        let from = (meta.window_from as usize).max(low).min(history.len());
        let scale = self.scale(meta);
        // the tail may use what the budget leaves after the prefix, the bridges and the new messages
        let tail_est = calibrated(tokens(&history[from..]), scale);
        let fixed = est.saturating_sub(tail_est);
        let new_from = window::cut(history, from, target.saturating_sub(fixed), scale);
        if new_from <= from { return None; }
        meta.window_from = new_from as u64;
        meta.window_cuts += 1;
        // pins below the new start become resident: the notice shows them until the next cut (newest 32)
        let mut pins: Vec<u64> = meta.pinned.iter().copied().filter(|&i| (i as usize) < new_from).collect();
        pins.sort_unstable();
        pins.dedup();
        meta.resident = pins.split_off(pins.len().saturating_sub(32));
        // the cut costs the cache anyway: page every result the rule allows at the same time
        let after = page_after(self.cfg.page_results_after_turns, meta);
        if after > 0 {
            let b = window::page_boundary(history, new_from, after) as u64;
            meta.paged_before = meta.paged_before.max(b);
        }
        let view = live_view(prefix, meta, history, pending, self.page_min());
        let after = calibrated(tokens(&view) + tokens(new_messages), scale);
        tracing::info!("{thread}: paged out messages #{from} to #{} (view ~{est} -> ~{after} tokens, budget {budget}, cut {})", new_from - 1, meta.window_cuts);
        Some((view, after))
    }

    /// The window's budget and trim target: the config's numbers, or from the nodes' context length (none while
    /// no node has said).
    async fn window_limits(&self, model: &str) -> Option<(u64, u64)> {
        const MARGIN: u64 = 8192;
        let budget = if self.cfg.window_tokens > 0 {
            self.cfg.window_tokens
        } else {
            let ctx = if self.cfg.max_context_tokens > 0 { self.cfg.max_context_tokens } else { self.router.context_length_for(model).await };
            if ctx == 0 { return None; }
            ctx.saturating_sub(self.cfg.max_output_tokens + MARGIN)
        };
        if budget == 0 { return None; }
        let target = if self.cfg.window_trim_tokens > 0 { self.cfg.window_trim_tokens.min(budget - 1) } else { budget * 5 / 8 };
        Some((budget, target))
    }

    /// Ask the model for its notes and record the refocus. The cut keeps `compact_keep_messages` newest
    /// history messages (fewer while they alone are over half of `compact_at_tokens`), moved earlier past
    /// leading tool results so the tail never opens with an answer whose call is gone.
    #[allow(clippy::too_many_arguments)]
    async fn refocus(&self, thread: &str, meta: &ThreadMeta, view: &[Value], history: &[Value], defs: &[Value], focus: Option<String>, est: u64) -> anyhow::Result<Refocused> {
        let n = history.len() as u64;
        let from = meta.compactions.last().map(|c| c.from).unwrap_or(0);
        let mut keep = self.cfg.compact_keep_messages.min(n.saturating_sub(from));
        if self.cfg.compact_at_tokens > 0 {
            while keep > 1 && tokens(&history[(n - keep) as usize..]) > self.cfg.compact_at_tokens / 2 { keep -= 1; }
        }
        let mut keep_from = n - keep;
        while keep_from > from && history[keep_from as usize].get("role").and_then(|r| r.as_str()) == Some("tool") { keep_from -= 1; }
        if keep_from <= from {
            return Ok(Refocused::Nothing);
        }
        let mut instruction = self.compact_prompt.clone();
        if let Some(f) = &focus { instruction.push_str(&format!("\n\nThe focal point to write around: {f}")); }
        let awaited: Vec<String> = meta.pending.iter().map(|p| format!("{} ({})", p.id, p.name)).collect();
        if !awaited.is_empty() { instruction.push_str(&format!("\n\nTool calls still awaiting results: {}", awaited.join(", "))); }
        // the messages the notes replace: the view up to the cut (the prefix and bridge included), then the ask
        let cut = view.len() - (n - keep_from) as usize;
        let mut messages: Vec<Value> = view[..cut].to_vec();
        messages.push(json!({"role": "user", "content": instruction}));
        self.fade(&mut messages, meta, history);
        // the same tools as every turn, so the node's cache serves the view; the prompt asks for text only
        // tool_choice none keeps the model from answering the ask with a tool call; a node that does not know
        // the field (400) gets the request again without it
        // unsteered: the notes are stored text the thread will believe, so memory stays separate from voice (a
        // persona's steering never colors what a later freeze of these notes bakes in). The cost: the node reads
        // this view without the steered cache.
        let mut body = json!({"model": self.model_of(meta), "messages": messages, "tools": defs, "tool_choice": "none", "max_tokens": self.cfg.compact_summary_tokens, "stream": false});
        // under scope "prefix" the generation is plain by construction, so the refocus keeps the steering fields and
        // the node serves the view from the expert's cached (steered) reading instead of reading it again
        if let Some(e) = meta.expert.as_ref().and_then(|k| self.store.expert(k).ok().flatten()) {
            if e.steer_scope.as_deref() == Some("prefix") { crate::experts::steer_into(&mut body, &e); }
        }
        let mut truncated = false;
        let mut summary = String::new();
        for attempt in 0..2 {
            let turn = match self.router.chat(Some(thread), meta.node.as_deref(), &body).await {
                Ok(t) => t,
                Err(ChatError::Stopping) => return Ok(Refocused::Stopping),
                Err(ChatError::Transient(e)) => return Ok(Refocused::Failed(format!("refocus: {e}"))),
                Err(ChatError::Rejected(_)) if body.get("tool_choice").is_some() => {
                    body.as_object_mut().unwrap().remove("tool_choice");
                    match self.router.chat(Some(thread), meta.node.as_deref(), &body).await {
                        Ok(t) => t,
                        Err(e) => return Ok(Refocused::Failed(format!("refocus: {e}"))),
                    }
                }
                Err(ChatError::Rejected(e)) => return Ok(Refocused::Failed(format!("refocus: the node refused the request: {e}"))),
            };
            let content = turn.message.get("content").and_then(|c| c.as_str()).unwrap_or("").trim().to_owned();
            if content.is_empty() {
                let calls = turn.message.get("tool_calls").and_then(|t| t.as_array()).map(|a| a.len()).unwrap_or(0);
                let reasoning = turn.message.get("reasoning_content").or(turn.message.get("reasoning")).and_then(|r| r.as_str()).map(|r| r.len()).unwrap_or(0);
                return Ok(Refocused::Failed(format!("refocus: the model wrote no notes (finish_reason {:?}, {calls} tool call(s), {reasoning} chars of reasoning)", turn.finish_reason)));
            }
            let cut_short = turn.finish_reason.as_deref() == Some("length");
            summary = content;
            if !cut_short { truncated = false; break; }
            truncated = true;
            if attempt == 0 {
                tracing::info!("{thread}: notes hit the cap, asking for half the length");
                body["messages"].as_array_mut().unwrap().last_mut().unwrap()["content"] = json!(format!("{instruction}\n\nWrite at most half of what you would otherwise; the previous attempt was cut off."));
            }
        }
        let c = Compaction { from: keep_from, summary, focus, at_ms: now_ms(), summarized: keep_from - from, tokens_before: est, truncated };
        self.store.compact(thread, &c)?;
        tracing::info!("{thread}: refocused at {keep_from} ({} messages folded, ~{est} tokens before{})", c.summarized, if truncated { ", notes truncated" } else { "" });
        Ok(Refocused::Done(c))
    }
}

enum Refocused {
    Done(Compaction),
    Nothing,
    Failed(String),
    /// The daemon is stopping before the notes' request reached a node.
    Stopping,
}

#[derive(Debug, Clone)]
struct Call {
    id: String,
    provider_call_id: Option<String>,
    name: String,
    args: Value,
}

impl Call {
    /// The id the stored assistant message carries for this call: the model's own when it gave one.
    fn shown_id(&self) -> String {
        self.provider_call_id.clone().unwrap_or_else(|| self.id.clone())
    }

    fn pending(&self) -> PendingCall {
        PendingCall { id: self.id.clone(), name: self.name.clone(), provider_call_id: self.provider_call_id.clone(), arguments: self.args.to_string() }
    }
}

fn wake(thread: &str, call: &Call, due_ms: u64, text: String) -> Event {
    Event {
        id: format!("wake-{}", call.id),
        thread: thread.to_owned(),
        due_ms,
        created_ms: 0,
        attempts: 0,
        kind: EventKind::ToolResult { call: call.id.clone(), provider_call_id: call.provider_call_id.clone(), text, images: vec![] },
    }
}

/// The assistant message as stored and resent: role, content, tool_calls with arguments that always render
/// (a model sometimes JSON-encodes them twice, or writes a bare value; a node refuses the whole thread then).
/// Reasoning is dropped: it is not resent.
fn normalize(m: &Value) -> (Value, Vec<Call>) {
    let content = m.get("content").and_then(|c| c.as_str()).map(|s| s.to_owned());
    let mut calls = Vec::new();
    let mut tool_calls = Vec::new();
    for (i, tc) in m.get("tool_calls").and_then(|t| t.as_array()).into_iter().flatten().enumerate() {
        let f = tc.get("function").cloned().unwrap_or(Value::Null);
        let name = f.get("name").and_then(|n| n.as_str()).unwrap_or("").to_owned();
        let args = repair_args(f.get("arguments").unwrap_or(&Value::Null));
        let provider_call_id = tc.get("id").and_then(|i| i.as_str()).map(str::to_owned);
        // the id the thread knows the call by: ours, unique across the store; the model's id rides along
        let id = format!("{}-{}", crate::config::id(), i);
        tool_calls.push(json!({"id": provider_call_id.clone().unwrap_or_else(|| id.clone()), "type": "function",
            "function": {"name": name, "arguments": args.to_string()}}));
        calls.push(Call { id, provider_call_id, name, args });
    }
    let mut out = json!({"role": "assistant", "content": content});
    if !tool_calls.is_empty() { out["tool_calls"] = Value::Array(tool_calls); }
    (out, calls)
}

fn repair_args(a: &Value) -> Value {
    let v = match a {
        Value::String(s) => serde_json::from_str::<Value>(s).unwrap_or(Value::String(s.clone())),
        other => other.clone(),
    };
    let v = match v {
        Value::String(s) => serde_json::from_str::<Value>(&s).unwrap_or(Value::String(s)),
        other => other,
    };
    match v {
        Value::Object(_) => v,
        Value::Null => json!({}),
        other => json!({"value": other}),
    }
}

fn until(args: &Value) -> Result<u64, String> {
    use chrono::{NaiveDate, NaiveTime, TimeZone};
    let date = args.get("date").and_then(|d| d.as_str()).ok_or("date is required")?;
    let time = args.get("time").and_then(|t| t.as_str()).ok_or("time is required")?;
    let tz: chrono_tz::Tz = args.get("timezone").and_then(|t| t.as_str()).unwrap_or("UTC").parse().map_err(|_| "unknown timezone".to_owned())?;
    let d = NaiveDate::parse_from_str(date, "%Y-%m-%d").map_err(|_| "date must be YYYY-MM-DD".to_owned())?;
    let t = NaiveTime::parse_from_str(time, "%H:%M:%S").or_else(|_| NaiveTime::parse_from_str(time, "%H:%M")).map_err(|_| "time must be HH:MM or HH:MM:SS".to_owned())?;
    let local = tz.from_local_datetime(&d.and_time(t)).single().ok_or("that local time does not exist")?;
    Ok((local.timestamp_millis().max(0)) as u64)
}

#[cfg(test)]
mod tests {
    #[test]
    fn resident_pins_show_in_the_notice_region_until_the_next_cut() {
        let h: Vec<Value> = (0..6).map(|i| json!({"role": if i % 2 == 0 { "user" } else { "assistant" }, "content": format!("message {i}")})).collect();
        let p = vec![json!({"role": "system", "content": "s"})];
        let mut meta = ThreadMeta { window_from: 4, pinned: vec![1], ..Default::default() };
        let v = live_view(&p, &meta, &h, &[], 0);
        assert!(!v[1]["content"].as_str().unwrap().contains("Kept in view"), "a pin alone moves nothing");
        meta.resident = vec![1];
        let v = live_view(&p, &meta, &h, &[], 0);
        let notice = v[1]["content"].as_str().unwrap();
        assert!(notice.starts_with("[Paged.") && notice.contains("Kept in view") && notice.contains("#1 assistant: message 1"), "{notice}");
        assert_eq!(v[2]["content"], "message 4");
    }

    use super::*;

    #[test]
    fn goldfish_never_page() {
        let elephant = ThreadMeta::default();
        let goldfish = ThreadMeta { goldfish_of: Some(GoldfishOf { thread: "e".into(), call: "c".into(), provider_call_id: None }), ..Default::default() };
        assert_eq!(page_after(4, &elephant), 4);
        assert_eq!(page_after(4, &goldfish), 0);
        assert_eq!(page_after(0, &elephant), 0);
    }

    #[test]
    fn estimates_scale_up_only() {
        let mut m = ThreadMeta::default();
        assert_eq!(own_scale(&m), None);
        m.prompt_est = 1000; m.prompt_tokens = 1300;
        assert_eq!(calibrated(2000, own_scale(&m).unwrap()), 2600);
        m.prompt_tokens = 10;
        assert_eq!(own_scale(&m), None, "a node that counts fewer never loosens the guard");
        assert_eq!(calibrated(2000, 0.5), 2000);
        assert_eq!(overflow_count("400 Bad Request: {\"error\":{\"message\":\"the prompt is 265526 tokens; the context holds 262144\"}}"), Some(265526));
        assert_eq!(overflow_count("maximum context length is 262144 tokens"), None);
        let defs = json!([{"type": "function", "function": {"name": "read_file", "description": "x".repeat(4000)}}]);
        assert!(overhead_tokens(&json!({"tools": defs})) > 1000);
    }

    #[test]
    fn overflow_messages_are_fatal() {
        assert!(is_context_overflow("http://n:5300 answered 400 Bad Request: {\"error\":{\"message\":\"the prompt is 267832 tokens; the context holds 262144\"}}"));
        assert!(is_context_overflow("This model's maximum context length is 262144 tokens"));
        assert!(!is_context_overflow("400 Bad Request: unknown field tool_choice"));
        assert!(is_context_overflow("the request exceeds the available context size, try increasing it"));
        assert!(!is_context_overflow("400 Bad Request: max_tokens must be under the context length in tokens"));
    }

    #[test]
    fn args_repaired() {
        assert_eq!(repair_args(&json!("{\"a\":1}")), json!({"a": 1}));
        assert_eq!(repair_args(&json!("\"{\\\"a\\\":1}\"")), json!({"a": 1}));
        assert_eq!(repair_args(&json!("hello")), json!({"value": "hello"}));
        assert_eq!(repair_args(&json!(null)), json!({}));
        assert_eq!(repair_args(&json!({"x": 2})), json!({"x": 2}));
    }

    #[test]
    fn until_parses() {
        let due = until(&json!({"date": "2030-01-01", "time": "00:00", "timezone": "UTC"})).unwrap();
        assert_eq!(due, 1_893_456_000_000);
        assert!(until(&json!({"date": "2030-01-01"})).is_err());
    }

    #[test]
    fn live_view_bridges_pending_calls_once() {
        let prefix = vec![json!({"role": "system", "content": "s"})];
        let history = vec![
            json!({"role": "user", "content": "a"}),
            json!({"role": "assistant", "content": null, "tool_calls": [{"id": "p1", "type": "function", "function": {"name": "run", "arguments": "{}"}}]}),
            json!({"role": "user", "content": "b"}),
        ];
        let pending = vec![
            PendingCall { id: "x-0".into(), name: "run".into(), provider_call_id: Some("p1".into()), arguments: "{}".into() },
            PendingCall { id: "y-0".into(), name: "echo".into(), provider_call_id: None, arguments: "{\"t\":1}".into() },
        ];
        let mut meta = ThreadMeta::default();
        assert_eq!(live_view(&prefix, &meta, &history, &pending, 0).len(), 4);
        meta.compactions.push(Compaction { from: 1, summary: "notes".into(), focus: Some("f".into()), at_ms: 0, summarized: 1, tokens_before: 0, truncated: false });
        let v = live_view(&prefix, &meta, &history, &pending, 0);
        assert_eq!(v.len(), 1 + 2 + 2);
        assert_eq!(v[2]["content"], "notes");
        let calls = v[2]["tool_calls"].as_array().unwrap();
        assert_eq!(calls.len(), 1, "p1 is in the kept tail; only y-0 is bridged: {calls:?}");
        assert_eq!(calls[0]["id"], "y-0");
        assert!(v[1]["content"].as_str().unwrap().contains("Focus: f"));
    }
}
