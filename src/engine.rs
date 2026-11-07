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
use crate::store::{now_ms, Commit, Compaction, Event, EventKind, Invocation, PendingCall, Store, ThreadMeta};
use crate::window;
use serde_json::{json, Value};
use std::sync::Arc;

pub struct Engine {
    pub cfg: Config,
    pub store: Arc<Store>,
    pub router: Arc<Router>,
    pub tools: Arc<ToolBox>,
    pub system: String,
    pub compact_prompt: String,
    pub callback_url: String,
    /// The latest node-count-over-estimate ratio any thread saw, times 1000; 0 until a turn has run.
    pub scale_hint: std::sync::atomic::AtomicU64,
}

pub enum Outcome {
    Done,
    /// Nothing was committed; the events run again after `retry_after_s`, and dead-letter past `max_attempts`.
    Transient(String),
    /// Nothing was committed and no retry can help; the events dead-letter now and the thread is closed.
    Fatal(String),
}

const UNKNOWN_TOOL_ROUNDS: usize = 3;

pub fn tokens(v: &[Value]) -> u64 {
    (serde_json::to_vec(v).map(|b| b.len()).unwrap_or(0) / 4) as u64
}

/// The bytes / 4 estimate scaled up by `scale` (never down: a node that reports fewer tokens than the estimate,
/// a fake, a different accounting, never makes the guard looser than bytes / 4).
pub fn calibrated(est: u64, scale: f64) -> u64 {
    if scale <= 1.0 { est } else { (est as f64 * scale) as u64 }
}

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
    let m = msg.to_ascii_lowercase();
    m.contains("context") && (m.contains("token") || m.contains("length"))
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

/// What the model sees: the prefix, then (after a refocus) the bridge and notes, then (after a window cut) the
/// `[Paged.]` notice, then history from the window's start. A result whose call is not in the shown tail gets a
/// stub call on the notes or notice side so the template sees a call before every result. `pending` is the
/// thread's pending list as read at slice start.
pub fn live_view(prefix: &[Value], meta: &ThreadMeta, history: &[Value], pending: &[PendingCall]) -> Vec<Value> {
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
        v.push(json!({"role": "user", "content": window::notice(history, low, from)}));
        if !stubs.is_empty() { v.push(json!({"role": "assistant", "content": "", "tool_calls": stubs.clone()})); stubs_placed = true; }
    }
    if !stubs.is_empty() && !stubs_placed {
        // no refocus and no cut, yet a result's call is missing (a migrated history): a stub turn before the tail
        v.push(json!({"role": "user", "content": "[The calls these results answer were made before this view begins.]"}));
        v.push(json!({"role": "assistant", "content": "", "tool_calls": stubs}));
    }
    v.extend(history[from..].iter().cloned());
    v
}

impl Engine {
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
        Ok(live_view(&p, &meta, &history, &meta.pending))
    }

    async fn run(&self, thread: &str, batch: &[(u64, Event)]) -> anyhow::Result<Outcome> {
        let mut meta = self.store.thread(thread)?.unwrap_or_default();
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
            self.store.commit(Commit { thread, meta, messages: vec![], done, done_ids, replies: vec![], events: vec![], invocations: vec![] })?;
            return Ok(Outcome::Done);
        }

        // ---- the new messages ------------------------------------------------------------------
        let pending_at_start = meta.pending.clone();
        let mut new_messages: Vec<Value> = Vec::new();
        let mut reply_to: Option<String> = None;
        let mut refocus: Option<Option<String>> = None;
        for (_, ev) in &live {
            match &ev.kind {
                EventKind::User { text, .. } => {
                    // a person may speak while tool calls are outstanding: the calls stay pending and their
                    // results land later as tool messages (the templates render that)
                    new_messages.push(json!({"role": "user", "content": text}));
                    reply_to = Some(ev.id.clone());
                }
                EventKind::ToolResult { call, text, .. } => {
                    // a result for a call nobody is waiting on (a late one, a thread moved here from another
                    // host) is said in the user role: a tool message with no call before it breaks the template
                    // the stored tool message names the call by the id the assistant message carries (the model's
                    // own when it gave one), so a template or validator that pairs them finds the pair
                    let awaited = meta.pending.iter().any(|p| &p.id == call);
                    let shown_id = meta.pending.iter().find(|p| &p.id == call).and_then(|p| p.provider_call_id.clone()).unwrap_or_else(|| call.clone());
                    meta.pending.retain(|p| &p.id != call);
                    new_messages.push(if awaited {
                        json!({"role": "tool", "tool_call_id": shown_id, "content": text})
                    } else {
                        json!({"role": "user", "content": format!("[late result for tool call {call}] {text}")})
                    });
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
        if reply_to.is_none() { reply_to = live.last().map(|(_, e)| e.id.clone()); }

        let prefix = prefix(&self.system, &self.store, &meta)?;
        let history = self.store.messages(thread)?;
        let tools = self.tools.current().await;
        let mut defs = rap::builtin_defs(self.cfg.compaction, self.cfg.window);
        defs.extend(tools.defs.iter().cloned());

        // ---- refocus -----------------------------------------------------------------------------
        let mut view = live_view(&prefix, &meta, &history, &pending_at_start);
        let mut est = calibrated(tokens(&view) + tokens(&new_messages), self.scale(&meta));
        if self.cfg.compaction && (refocus.is_some() || (self.cfg.compact_at_tokens > 0 && est > self.cfg.compact_at_tokens)) {
            let focus = refocus.clone().flatten();
            match self.refocus(thread, &meta, &view, &history, &defs, focus, est).await? {
                Refocused::Done(c) => {
                    meta.window_from = c.from;
                    meta.compactions.push(c);
                    meta.node = None;
                    view = live_view(&prefix, &meta, &history, &pending_at_start);
                    est = calibrated(tokens(&view) + tokens(&new_messages), self.scale(&meta));
                    if self.cfg.compact_at_tokens > 0 && est > self.cfg.compact_at_tokens {
                        tracing::warn!("{thread}: still ~{est} tokens after refocusing (the kept tail or the new messages are large)");
                    }
                }
                Refocused::Nothing => {
                    if refocus.is_some() { tracing::info!("{thread}: refocus asked for, nothing to fold yet"); }
                }
                Refocused::Failed(why) => return Ok(Outcome::Transient(why)),
            }
        }
        if new_messages.is_empty() {
            // only a refocus was asked for: no turn
            self.store.commit(Commit { thread, meta, messages: vec![], done, done_ids, replies: vec![], events: vec![], invocations: vec![] })?;
            return Ok(Outcome::Done);
        }

        // ---- the window --------------------------------------------------------------------------
        if let Some((v, e)) = self.window_cut(thread, &mut meta, &prefix, &history, &pending_at_start, &new_messages, est).await {
            view = v;
            est = e;
        }

        // ---- room in the context -----------------------------------------------------------------
        let limit = if self.cfg.max_context_tokens > 0 { self.cfg.max_context_tokens } else { self.router.context_length().await.saturating_sub(self.cfg.max_output_tokens) };
        if limit > 0 && est > limit {
            return Ok(Outcome::Fatal(format!("context full: about {est} tokens, the limit is {limit}")));
        }

        // ---- the model's turn --------------------------------------------------------------------
        let mut messages = view.clone();
        messages.extend(new_messages.iter().cloned());
        let sent_est = tokens(&messages);
        let mut body = json!({"model": self.router.model(), "messages": messages, "tools": defs, "max_tokens": self.cfg.max_output_tokens, "stream": false});

        let mut reply_text = String::new();
        let mut events: Vec<Event> = Vec::new();
        let mut invocations: Vec<Invocation> = Vec::new();
        let mut rounds = 0;
        let mut recut = false;
        loop {
            rounds += 1;
            let turn = match self.router.chat(meta.node.as_deref(), &body).await {
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
                                body["messages"] = Value::Array(msgs);
                                recut = true;
                                rounds -= 1;
                                continue;
                            }
                        }
                    }
                    return Ok(Outcome::Fatal(format!("context full: {e}")));
                }
                Err(ChatError::Transient(e)) | Err(ChatError::Rejected(e)) => return Ok(Outcome::Transient(e)),
            };
            meta.node = Some(turn.node.clone());
            let waited = if turn.waited_ms >= 1000 { format!(", waited {} s for a slot", turn.waited_ms / 1000) } else { String::new() };
            if let (Some(p), Some(c)) = (turn.prompt_tokens, turn.cached_tokens) {
                tracing::info!("{thread}: turn on {} ({p} prompt tokens, {c} from its cache{waited})", turn.node);
            } else if !waited.is_empty() {
                tracing::info!("{thread}: turn on {}{waited}", turn.node);
            }
            if let Some(p) = turn.prompt_tokens.filter(|&p| p > 0) {
                meta.prompt_tokens = p;
                meta.prompt_est = tokens(body["messages"].as_array().map(|a| a.as_slice()).unwrap_or(&[])).max(1);
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
            let mut unknown = Vec::new();
            let mut recalled: Vec<(Call, String)> = Vec::new();
            for call in &calls {
                match call.name.as_str() {
                    rap::RECALL if self.cfg.window => {
                        // answered from the store within the slice: a fault costs one more model pass, no tool round trip
                        let text = match window::recall_indices(&call.args, history.len()) {
                            Ok(idx) => window::recall_text(&history, &idx),
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
                    name => match tools.get(name) {
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
                        None => unknown.push(call.clone()),
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
            if !unknown.is_empty() && (rounds >= UNKNOWN_TOOL_ROUNDS || !events.is_empty() || !invocations.is_empty()) {
                reply_text.push_str(&format!("\n[Error: the model called tools that do not exist: {}]", unknown.iter().map(|u| u.name.as_str()).collect::<Vec<_>>().join(", ")));
                break;
            }
            // a turn that also slept or called a real tool ends here; the recall's text waits in the history for its next turn
            if rounds >= UNKNOWN_TOOL_ROUNDS || !events.is_empty() || !invocations.is_empty() { break; }
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
        self.store.commit(Commit { thread, meta, messages: new_messages, done, done_ids, replies, events, invocations })?;
        Ok(Outcome::Done)
    }

    /// How far over bytes / 4 to count this thread: its own last ratio, else the latest any thread saw, else the
    /// config's default.
    fn scale(&self, meta: &ThreadMeta) -> f64 {
        own_scale(meta).unwrap_or_else(|| {
            let hint = self.scale_hint.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1000.0;
            if hint > 1.0 { hint } else { self.cfg.est_scale_default }
        })
    }

    /// Move the window if the view (`est`, calibrated) is over the budget. Returns the new view and estimate when
    /// it moved.
    #[allow(clippy::too_many_arguments)]
    async fn window_cut(&self, thread: &str, meta: &mut ThreadMeta, prefix: &[Value], history: &[Value], pending: &[PendingCall], new_messages: &[Value], est: u64) -> Option<(Vec<Value>, u64)> {
        if !self.cfg.window { return None; }
        let (budget, target) = self.window_limits().await?;
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
        let view = live_view(prefix, meta, history, pending);
        let after = calibrated(tokens(&view) + tokens(new_messages), scale);
        tracing::info!("{thread}: paged out messages #{from} to #{} (view ~{est} -> ~{after} tokens, budget {budget}, cut {})", new_from - 1, meta.window_cuts);
        Some((view, after))
    }

    /// The window's budget and trim target: the config's numbers, or from the nodes' context length (none while
    /// no node has said).
    async fn window_limits(&self) -> Option<(u64, u64)> {
        const MARGIN: u64 = 8192;
        let budget = if self.cfg.window_tokens > 0 {
            self.cfg.window_tokens
        } else {
            let ctx = if self.cfg.max_context_tokens > 0 { self.cfg.max_context_tokens } else { self.router.context_length().await };
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
        // the same tools as every turn, so the node's cache serves the view; the prompt asks for text only
        // tool_choice none keeps the model from answering the ask with a tool call; a node that does not know
        // the field (400) gets the request again without it
        let mut body = json!({"model": self.router.model(), "messages": messages, "tools": defs, "tool_choice": "none", "max_tokens": self.cfg.compact_summary_tokens, "stream": false});
        let mut truncated = false;
        let mut summary = String::new();
        for attempt in 0..2 {
            let turn = match self.router.chat(meta.node.as_deref(), &body).await {
                Ok(t) => t,
                Err(ChatError::Transient(e)) => return Ok(Refocused::Failed(format!("refocus: {e}"))),
                Err(ChatError::Rejected(_)) if body.get("tool_choice").is_some() => {
                    body.as_object_mut().unwrap().remove("tool_choice");
                    match self.router.chat(meta.node.as_deref(), &body).await {
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
        kind: EventKind::ToolResult { call: call.id.clone(), provider_call_id: call.provider_call_id.clone(), text },
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
    use super::*;

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
    }

    #[test]
    fn overflow_messages_are_fatal() {
        assert!(is_context_overflow("http://n:5300 answered 400 Bad Request: {\"error\":{\"message\":\"the prompt is 267832 tokens; the context holds 262144\"}}"));
        assert!(is_context_overflow("This model's maximum context length is 262144 tokens"));
        assert!(!is_context_overflow("400 Bad Request: unknown field tool_choice"));
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
        assert_eq!(live_view(&prefix, &meta, &history, &pending).len(), 4);
        meta.compactions.push(Compaction { from: 1, summary: "notes".into(), focus: Some("f".into()), at_ms: 0, summarized: 1, tokens_before: 0, truncated: false });
        let v = live_view(&prefix, &meta, &history, &pending);
        assert_eq!(v.len(), 1 + 2 + 2);
        assert_eq!(v[2]["content"], "notes");
        let calls = v[2]["tool_calls"].as_array().unwrap();
        assert_eq!(calls.len(), 1, "p1 is in the kept tail; only y-0 is bridged: {calls:?}");
        assert_eq!(calls[0]["id"], "y-0");
        assert!(v[1]["content"].as_str().unwrap().contains("Focus: f"));
    }
}
