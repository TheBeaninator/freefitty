//! RAP, the Reactive Agent Protocol: tools are HTTP servers. Each publishes a manifest at
//! /.well-known/rap-toolset naming its tools and an invoke endpoint; the agent posts an invocation there and
//! the server acknowledges at once, then posts the result to the invocation's callback URL whenever it is done.
//!
//! `Tools` holds every tool the configured servers offer, as the model sees them. `Dispatcher` owns the
//! invocations a slice committed and posts them until a server acknowledges; one that never does gets its
//! result written as an error so the thread is not stuck waiting for a server that is gone.

use crate::config::Config;
use crate::store::{Event, EventKind, Invocation, Store};
use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{Notify, RwLock};

#[derive(Debug, Clone)]
pub struct Tool {
    pub name: String,
    pub endpoint: String,
    pub def: Value,
}

#[derive(Default)]
pub struct Tools {
    by_name: HashMap<String, Tool>,
    /// The `tools` array of a chat completion request, built once.
    pub defs: Vec<Value>,
}

pub const SLEEP: &str = "sleep";
pub const SLEEP_UNTIL: &str = "sleep_until";
pub const SLEEP_UNTIL_EVENT: &str = "sleep_until_event_or_input";
pub const REFOCUS: &str = "refocus";
pub const RECALL: &str = "recall";

/// The built-in tools; `refocus` only with compaction on, `recall` only with the window on.
pub fn builtin_defs(compaction: bool, window: bool) -> Vec<Value> {
    let mut v = vec![
        json!({"type": "function", "function": {"name": SLEEP,
            "description": "Sleep for a number of seconds, then continue. The agent holds nothing while asleep; any message that arrives first wakes it early.",
            "parameters": {"type": "object", "properties": {
                "seconds": {"type": "number", "description": "Seconds to sleep"},
                "reason": {"type": "string", "description": "Why, in a few words"}}, "required": ["seconds"]}}}),
        json!({"type": "function", "function": {"name": SLEEP_UNTIL,
            "description": "Sleep until a date and time, then continue.",
            "parameters": {"type": "object", "properties": {
                "date": {"type": "string", "description": "YYYY-MM-DD"},
                "time": {"type": "string", "description": "HH:MM or HH:MM:SS, 24-hour"},
                "timezone": {"type": "string", "description": "IANA zone name; UTC by default"}}, "required": ["date", "time"]}}}),
        json!({"type": "function", "function": {"name": SLEEP_UNTIL_EVENT,
            "description": "Wait with no timer: the agent runs again only when a message, tool result or event arrives.",
            "parameters": {"type": "object", "properties": {}}}}),
    ];
    if compaction {
        v.push(json!({"type": "function", "function": {"name": REFOCUS,
            "description": "Refocus: before your next turn you will write notes around a focal point, and those notes replace the conversation before them as your memory. Use it when the context has grown muddled with superseded ideas, stale output or finished branches. The full history stays stored.",
            "parameters": {"type": "object", "properties": {
                "focus": {"type": "string", "description": "The focal point the notes should be written around, in a sentence"}}}}}));
    }
    if window {
        v.push(json!({"type": "function", "function": {"name": RECALL,
            "description": "Return stored messages of this conversation by index, exactly as written: the ones a [Paged.] notice says left the view, or anything earlier. Up to 16 KiB per call; ask for a narrower range if it is cut.",
            "parameters": {"type": "object", "properties": {
                "indices": {"type": "array", "items": {"type": "integer"}, "description": "Message indices (#n in the notice)"},
                "from": {"type": "integer", "description": "First index of a range"},
                "to": {"type": "integer", "description": "Last index of the range, inclusive"}}}}}));
    }
    v
}

impl Tools {
    pub fn get(&self, name: &str) -> Option<&Tool> {
        self.by_name.get(name)
    }

    fn from_manifests(manifests: &[(String, Value)]) -> Result<Tools> {
        let mut t = Tools::default();
        for (url, m) in manifests {
            let endpoint = m.get("endpoint").and_then(|e| e.as_str()).with_context(|| format!("{url}: manifest has no endpoint"))?.to_owned();
            for tool in m.get("tools").and_then(|t| t.as_array()).into_iter().flatten() {
                let name = tool.get("name").and_then(|n| n.as_str()).with_context(|| format!("{url}: a tool has no name"))?.to_owned();
                if [SLEEP, SLEEP_UNTIL, SLEEP_UNTIL_EVENT, REFOCUS, RECALL].contains(&name.as_str()) || t.by_name.contains_key(&name) {
                    bail!("{url}: tool name {name:?} is taken");
                }
                let def = json!({"type": "function", "function": {
                    "name": name,
                    "description": tool.get("description").and_then(|d| d.as_str()).unwrap_or(""),
                    "parameters": tool.get("inputSchema").cloned().unwrap_or(json!({"type": "object", "properties": {}})),
                }});
                t.defs.push(def.clone());
                t.by_name.insert(name.clone(), Tool { name, endpoint: endpoint.clone(), def });
            }
        }
        Ok(t)
    }
}

/// Read every server's manifest; a server that does not answer keeps the manifest last stored for it.
pub async fn load_tools(cfg: &Config, store: &Store, client: &reqwest::Client) -> Result<Tools> {
    let mut manifests = Vec::new();
    for url in &cfg.tool_servers {
        let base = url.trim_end_matches('/');
        match client.get(format!("{base}/.well-known/rap-toolset")).send().await {
            Ok(r) if r.status().is_success() => match r.json::<Value>().await {
                Ok(m) => {
                    store.save_toolset(base, &m)?;
                    manifests.push((base.to_owned(), m));
                }
                Err(e) => tracing::warn!("tool server {base}: manifest is not JSON ({e})"),
            },
            Ok(r) => tracing::warn!("tool server {base}: manifest answered {}", r.status()),
            Err(e) => tracing::warn!("tool server {base}: {e}"),
        }
        if manifests.last().is_none_or(|(u, _)| u != base) {
            if let Some(m) = store.toolset(base)? {
                tracing::info!("tool server {base}: using the manifest stored earlier");
                manifests.push((base.to_owned(), m));
            }
        }
    }
    let t = Tools::from_manifests(&manifests)?;
    tracing::info!("tools: {} from {} server(s)", t.defs.len(), manifests.len());
    Ok(t)
}

/// Keeps `Tools` fresh: reloaded every `toolset_refresh_s`.
pub struct ToolBox {
    pub tools: RwLock<Arc<Tools>>,
}

impl ToolBox {
    pub async fn start(cfg: Config, store: Arc<Store>, tasks: &mut Vec<tokio::task::JoinHandle<()>>) -> Result<Arc<ToolBox>> {
        let client = reqwest::Client::builder().timeout(Duration::from_secs(10)).build()?;
        let tools = load_tools(&cfg, &store, &client).await?;
        let tb = Arc::new(ToolBox { tools: RwLock::new(Arc::new(tools)) });
        let t = tb.clone();
        tasks.push(tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(cfg.toolset_refresh_s.max(10))).await;
                match load_tools(&cfg, &store, &client).await {
                    Ok(tools) => *t.tools.write().await = Arc::new(tools),
                    Err(e) => tracing::warn!("tools: reload failed: {e}"),
                }
            }
        }));
        Ok(tb)
    }

    pub async fn current(&self) -> Arc<Tools> {
        self.tools.read().await.clone()
    }
}

/// Posts committed invocations to their tool servers until acknowledged. On start it resumes those a
/// crashed daemon left behind: a tool server sees the same call id again and treats it as it must (pod-hands
/// names its unit after the call id, so a repeat is refused there and the first run's result still arrives).
pub struct Dispatcher {
    store: Arc<Store>,
    client: reqwest::Client,
    notify: Notify,
    on_result: Box<dyn Fn(Event) + Send + Sync>,
}

const GIVE_UP_AFTER: u32 = 12; // ~ 6 minutes of retries at 30 s

impl Dispatcher {
    pub fn start(store: Arc<Store>, on_result: Box<dyn Fn(Event) + Send + Sync>, tasks: &mut Vec<tokio::task::JoinHandle<()>>) -> Arc<Dispatcher> {
        let d = Arc::new(Dispatcher {
            store,
            client: reqwest::Client::builder().timeout(Duration::from_secs(30)).connect_timeout(Duration::from_secs(5)).build().expect("client"),
            notify: Notify::new(),
            on_result,
        });
        let me = d.clone();
        tasks.push(tokio::spawn(async move { me.run().await }));
        d
    }

    pub fn kick(&self) {
        self.notify.notify_one();
    }

    async fn run(&self) {
        loop {
            let pending = match self.store.invocations() {
                Ok(p) => p,
                Err(e) => {
                    tracing::error!("dispatcher: {e}");
                    Vec::new()
                }
            };
            for inv in pending {
                self.post(inv).await;
            }
            let _ = tokio::time::timeout(Duration::from_secs(30), self.notify.notified()).await;
        }
    }

    async fn post(&self, inv: Invocation) {
        let r = self.client.post(&inv.endpoint).json(&inv.body).send().await;
        match r {
            Ok(r) if r.status().is_success() => {
                if let Err(e) = self.store.invocation_done(&inv.call) { tracing::error!("dispatcher: {e}"); }
            }
            other => {
                let why = match other { Ok(r) => format!("HTTP {}", r.status()), Err(e) => e.to_string() };
                let attempts = inv.attempts + 1;
                if attempts >= GIVE_UP_AFTER || matches!(&other_status(&why), Some(s) if (400..500).contains(s) && *s != 429) {
                    tracing::warn!("invocation {} to {}: giving up after {attempts} attempts ({why})", inv.call, inv.endpoint);
                    let _ = self.store.invocation_done(&inv.call);
                    (self.on_result)(Event {
                        id: format!("fail-{}", inv.call),
                        thread: inv.thread.clone(),
                        due_ms: 0,
                        created_ms: 0,
                        attempts: 0,
                        kind: EventKind::ToolResult {
                            call: inv.call.clone(),
                            provider_call_id: inv.body.get("call_id").and_then(|v| v.as_str()).map(str::to_owned),
                            text: format!("Error: the tool server at {} did not accept this call ({why}).", inv.endpoint),
                        },
                    });
                } else {
                    tracing::warn!("invocation {} to {}: {why}; attempt {attempts}", inv.call, inv.endpoint);
                    let _ = self.store.invocation_attempt(&inv.call, attempts);
                }
            }
        }
    }
}

fn other_status(why: &str) -> Option<u16> {
    why.strip_prefix("HTTP ").and_then(|s| s.split_whitespace().next()).and_then(|s| s.parse().ok())
}

/// The invocation body a tool server receives (rap_protocol::RapInvocation).
pub fn invocation(thread: &str, call: &str, provider_call_id: Option<&str>, operation: &str, arguments: Value, callback_url: &str) -> Value {
    json!({
        "operation": operation,
        "arguments": arguments,
        "id": call,
        "call_id": provider_call_id,
        "callback_url": callback_url,
        "group_id": thread,
    })
}

/// A callback body (rap_protocol::RapCallback) as the event it queues; None for kinds that carry nothing
/// for the conversation (view updates).
pub fn callback_event(body: &Value) -> Result<Option<Event>> {
    let kind = body.get("type").and_then(|t| t.as_str()).unwrap_or("tool_result");
    let thread = body.get("group_id").and_then(|g| g.as_str()).context("callback has no group_id")?.to_owned();
    if thread.is_empty() || thread.len() > 128 || !thread.bytes().all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b)) {
        bail!("callback group_id is not a thread id");
    }
    let provider_call_id = body.get("call_id").and_then(|c| c.as_str()).map(str::to_owned);
    let ev = match kind {
        "tool_result" => {
            let call = body.get("id").and_then(|i| i.as_str()).context("tool_result has no id")?.to_owned();
            let mut text = body.get("text").and_then(|t| t.as_str()).unwrap_or("").to_owned();
            for c in body.get("content").and_then(|c| c.as_array()).into_iter().flatten() {
                match c.get("type").and_then(|t| t.as_str()) {
                    Some("text") => {
                        if !text.is_empty() { text.push('\n'); }
                        text.push_str(c.get("text").and_then(|t| t.as_str()).unwrap_or(""));
                    }
                    Some(other) => text.push_str(&format!("\n[{other} content omitted]")),
                    None => {}
                }
            }
            Event { id: format!("result-{call}-{}", short_hash(&text)), thread, due_ms: 0, created_ms: 0, attempts: 0, kind: EventKind::ToolResult { call, provider_call_id, text } }
        }
        "subscription_event" => {
            let call = body.get("tool_call_id").and_then(|i| i.as_str()).context("subscription_event has no tool_call_id")?.to_owned();
            let text = body.get("text").and_then(|t| t.as_str()).unwrap_or("").to_owned();
            let fin = body.get("final").and_then(|f| f.as_bool()).unwrap_or(false);
            Event { id: format!("sub-{call}-{}", short_hash(&text)), thread, due_ms: 0, created_ms: 0, attempts: 0, kind: EventKind::Subscription { call, text, r#final: fin } }
        }
        "oauth" | "user_choice" => {
            // the person must act; it reaches them as a message in the conversation and as a reply
            let text = match kind {
                "oauth" => format!("[A tool needs authorization: {}]", body.get("auth_url").and_then(|u| u.as_str()).unwrap_or("?")),
                _ => format!("[A tool asks: {} Choices: {}]", body.get("prompt").and_then(|p| p.as_str()).unwrap_or("?"), body.get("choices").map(|c| c.to_string()).unwrap_or_default()),
            };
            let call = body.get("id").and_then(|i| i.as_str()).unwrap_or("choice").to_owned();
            Event { id: format!("{kind}-{call}-{}", short_hash(&text)), thread, due_ms: 0, created_ms: 0, attempts: 0, kind: EventKind::Subscription { call, text, r#final: false } }
        }
        _ => return Ok(None),
    };
    Ok(Some(ev))
}

/// A short stable hash, so a tool server that posts the same result twice queues it once.
pub fn short_hash(s: &str) -> String {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    format!("{h:016x}")
}
