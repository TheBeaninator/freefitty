//! A slice: one thread, the events that arrived for it, one model turn, one commit.
//!
//! The conversation is loaded, the new events become messages (a person's text, a tool's result, a wake-up),
//! the model answers once. Its text becomes a reply; each tool call becomes a wake-up event (the sleeps) or
//! a RAP invocation. All of it, the consumed events included, is one transaction in the store, and only
//! after that commit are invocations posted and replies handed out. A crash before the commit re-runs the
//! slice from the same events; one after it cannot run the turn twice, because the events are gone and
//! their ids are recorded as processed.

use crate::config::Config;
use crate::rap::{self, ToolBox};
use crate::router::{ChatError, Router};
use crate::store::{now_ms, Commit, Event, EventKind, Invocation, PendingCall, Store, ThreadMeta};
use serde_json::{json, Value};
use std::sync::Arc;

pub struct Engine {
    pub cfg: Config,
    pub store: Arc<Store>,
    pub router: Arc<Router>,
    pub tools: Arc<ToolBox>,
    pub system: String,
    pub callback_url: String,
}

pub enum Outcome {
    Done,
    /// Nothing was committed; the events run again after `retry_after_s`, and dead-letter past `max_attempts`.
    Transient(String),
    /// Nothing was committed and no retry can help; the events dead-letter now and the thread is closed.
    Fatal(String),
}

const UNKNOWN_TOOL_ROUNDS: usize = 3;

impl Engine {
    pub async fn slice(&self, thread: &str, batch: &[(u64, Event)]) -> Outcome {
        match self.run(thread, batch).await {
            Ok(o) => o,
            Err(e) => Outcome::Transient(format!("store: {e:#}")),
        }
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
        let mut new_messages: Vec<Value> = Vec::new();
        let mut reply_to: Option<String> = None;
        for (_, ev) in &live {
            match &ev.kind {
                EventKind::User { text, .. } => {
                    // a person spoke while tool calls are outstanding: the template needs a result per call
                    for p in meta.pending.drain(..) {
                        new_messages.push(json!({"role": "tool", "tool_call_id": p.id, "content": "(no result yet; it arrives as a later message)"}));
                    }
                    new_messages.push(json!({"role": "user", "content": text}));
                    reply_to = Some(ev.id.clone());
                }
                EventKind::ToolResult { call, text, .. } => {
                    meta.pending.retain(|p| &p.id != call);
                    new_messages.push(json!({"role": "tool", "tool_call_id": call, "content": text}));
                }
                EventKind::Subscription { call, text, r#final } => {
                    if *r#final { meta.pending.retain(|p| &p.id != call); }
                    new_messages.push(json!({"role": "user", "content": format!("[event for tool call {call}] {text}")}));
                }
            }
        }
        if reply_to.is_none() { reply_to = live.last().map(|(_, e)| e.id.clone()); }

        let mut messages = vec![json!({"role": "system", "content": self.system})];
        messages.extend(self.store.messages(thread)?);
        messages.extend(new_messages.iter().cloned());

        // ---- room in the context -----------------------------------------------------------------
        let est = (serde_json::to_vec(&messages)?.len() / 4) as u64;
        let limit = if self.cfg.max_context_tokens > 0 { self.cfg.max_context_tokens } else { self.router.context_length().await.saturating_sub(self.cfg.max_output_tokens) };
        if limit > 0 && est > limit {
            return Ok(Outcome::Fatal(format!("context full: about {est} tokens, the limit is {limit}")));
        }

        // ---- the model's turn --------------------------------------------------------------------
        let tools = self.tools.current().await;
        let mut defs = rap::builtin_defs();
        defs.extend(tools.defs.iter().cloned());
        let mut body = json!({"model": self.router.model(), "messages": messages, "tools": defs, "max_tokens": self.cfg.max_output_tokens, "stream": false});

        let mut reply_text = String::new();
        let mut events: Vec<Event> = Vec::new();
        let mut invocations: Vec<Invocation> = Vec::new();
        let mut rounds = 0;
        loop {
            rounds += 1;
            let turn = match self.router.chat(meta.node.as_deref(), &body).await {
                Ok(t) => t,
                Err(ChatError::Transient(e)) | Err(ChatError::Rejected(e)) => return Ok(Outcome::Transient(e)),
            };
            meta.node = Some(turn.node.clone());
            if let (Some(p), Some(c)) = (turn.prompt_tokens, turn.cached_tokens) {
                tracing::info!("{thread}: turn on {} ({p} prompt tokens, {c} from its cache)", turn.node);
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
            for call in &calls {
                match call.name.as_str() {
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
                    rap::SLEEP_UNTIL_EVENT => {}
                    name => match tools.get(name) {
                        Some(tool) => {
                            invocations.push(Invocation {
                                call: call.id.clone(),
                                thread: thread.to_owned(),
                                endpoint: tool.endpoint.clone(),
                                body: rap::invocation(thread, &call.id, call.provider_call_id.as_deref(), name, call.args.clone(), &self.callback_url),
                                attempts: 0,
                                created_ms: 0,
                            });
                            meta.pending.push(call.pending());
                            reply_text.push_str(&format!("\n[Tool Call: {name} with arguments {}]", call.args));
                        }
                        None => unknown.push(call.clone()),
                    },
                }
            }
            if unknown.is_empty() { break; }
            // the model asked for tools that do not exist: tell it in the same slice and let it try again
            let names: Vec<String> = tools.defs.iter().filter_map(|d| d.pointer("/function/name").and_then(|n| n.as_str()).map(str::to_owned)).collect();
            for u in &unknown {
                new_messages.push(json!({"role": "tool", "tool_call_id": u.id, "content": format!("Error: there is no tool named {:?}. The tools are: sleep, sleep_until, sleep_until_event_or_input, {}", u.name, names.join(", "))}));
            }
            if rounds >= UNKNOWN_TOOL_ROUNDS || !events.is_empty() || !invocations.is_empty() {
                reply_text.push_str(&format!("\n[Error: the model called tools that do not exist: {}]", unknown.iter().map(|u| u.name.as_str()).collect::<Vec<_>>().join(", ")));
                break;
            }
            let mut msgs = body["messages"].as_array().cloned().unwrap_or_default();
            msgs.extend(new_messages[new_messages.len() - 1 - unknown.len()..].iter().cloned());
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
}

#[derive(Debug, Clone)]
struct Call {
    id: String,
    provider_call_id: Option<String>,
    name: String,
    args: Value,
}

impl Call {
    fn pending(&self) -> PendingCall {
        PendingCall { id: self.id.clone(), name: self.name.clone(), provider_call_id: self.provider_call_id.clone() }
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

#[allow(dead_code)]
pub fn thread_meta_default() -> ThreadMeta {
    ThreadMeta::default()
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
