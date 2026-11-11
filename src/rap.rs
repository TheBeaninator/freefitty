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
pub const EXPERTS: &str = "experts";
pub const GOLDFISH: &str = "goldfish";
pub const COMMISSION: &str = "commission_expert";

/// Built-ins every thread keeps whatever its tool list says.
const PLAIN: [&str; 6] = [SLEEP, SLEEP_UNTIL, SLEEP_UNTIL_EVENT, REFOCUS, RECALL, EXPERTS];
/// Built-ins a goldfish never has, whatever it was granted: it suggests a goldfish to its elephant instead.
pub const ELEPHANT_ONLY: [&str; 2] = [GOLDFISH, COMMISSION];

/// A name FreeFitty answers itself, whatever the ToolBox and the experts define.
pub fn is_builtin(name: &str) -> bool {
    PLAIN.contains(&name) || ELEPHANT_ONLY.contains(&name)
}

/// The definitions a thread with this tool list sees: the plain built-ins and the named tools; None = all.
pub fn scope_defs(defs: Vec<Value>, allow: Option<&[String]>) -> Vec<Value> {
    let Some(allow) = allow else { return defs };
    defs.into_iter().filter(|d| {
        let n = d.pointer("/function/name").and_then(|n| n.as_str()).unwrap_or("");
        PLAIN.contains(&n) || allow.iter().any(|a| a == n)
    }).collect()
}

/// A goldfish's tool list: what it was granted, less what only an elephant may call.
pub fn goldfish_tools(granted: &[String]) -> Vec<String> {
    granted.iter().filter(|t| !ELEPHANT_ONLY.contains(&t.as_str())).cloned().collect()
}

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
        json!({"type": "function", "function": {"name": EXPERTS,
            "description": "List the experts: each one's name, what it has read in a line, and the tools a goldfish born from it may use. An expert is the model's cached state after reading; a goldfish born from one starts already knowing what it read.",
            "parameters": {"type": "object", "properties": {}}}}),
        json!({"type": "function", "function": {"name": GOLDFISH,
            "description": "Ask a goldfish: a fresh thread born from an expert (see `experts`) that gets your question and any documents, answers once and is gone. Your turn ends and you sleep until its answer returns as this call's result; several calls in one turn run at once. A goldfish cannot ask goldfish or commission experts; it may suggest that you do.",
            "parameters": {"type": "object", "properties": {
                "expert": {"type": "string", "description": "The expert's name"},
                "question": {"type": "string", "description": "What it should do or answer: the whole brief, since it knows nothing of your conversation"},
                "documents": {"type": "array", "items": {"type": "string"}, "description": "Paths of files to put in front of the question (on the host running the agents; up to 256 KiB each)"},
                "tools": {"type": "array", "items": {"type": "string"}, "description": "Tools it may use, instead of the expert's own list"}}, "required": ["expert", "question"]}}}),
        json!({"type": "function", "function": {"name": COMMISSION,
            "description": "Commission a new expert: it reads the documents once and is frozen as cached state under its name. You sleep until it is ready; the result gives its name and version, and from then on any thread can ask goldfish born from it.",
            "parameters": {"type": "object", "properties": {
                "name": {"type": "string", "description": "1-64 of [A-Za-z0-9._-], not yet taken"},
                "description": {"type": "string", "description": "One line: what it knows, for the catalog"},
                "reads": {"type": "array", "items": {"type": "object", "properties": {
                    "path": {"type": "string", "description": "A file to read"},
                    "text": {"type": "string", "description": "Or text to read"}}}, "description": "What it reads, in order"},
                "ack": {"type": "string", "description": "Its one-sentence acknowledgement, in its own voice (a persona goes here)"},
                "tools": {"type": "array", "items": {"type": "string"}, "description": "Tools its goldfish may use; none by default"},
                "steer": {"type": "array", "items": {"type": "object", "properties": {
                    "vector": {"type": "string", "description": "A steering vector the nodes hold"},
                    "scale": {"type": "number", "description": "How hard to push along it"},
                    "layer": {"type": "integer", "description": "The layer it is added after; the vector's own by default"}}},
                    "description": "Activation steering: a persona the expert is on every turn of its goldfish"},
                "steer_scope": {"type": "string", "enum": ["all", "prefix"], "description": "Where the steering applies: all (every turn, the default) or prefix (its reading only; the goldfish answers unsteered)"}}, "required": ["name", "description", "reads"]}}}),
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
                if PLAIN.contains(&name.as_str()) || ELEPHANT_ONLY.contains(&name.as_str()) {
                    // one server shadowing a built-in must not take every other tool down with it
                    tracing::warn!("{url}: tool {name:?} is a built-in name; that tool is left out");
                    continue;
                }
                if t.by_name.contains_key(&name) {
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
            let now = crate::store::now_ms();
            for inv in pending {
                if inv.next_ms > now { continue; }   // backing off after a failure; the 30 s tick returns to it
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
                            images: vec![],
                        },
                    });
                } else {
                    // 30 s, 60 s, 120 s ... up to 10 min between attempts, so a kick from every commit does not burn them
                    let wait_ms = (30_000u64 << (attempts - 1).min(5)).min(600_000);
                    tracing::warn!("invocation {} to {}: {why}; attempt {attempts}, next in {} s", inv.call, inv.endpoint, wait_ms / 1000);
                    let _ = self.store.invocation_attempt(&inv.call, attempts, crate::store::now_ms() + wait_ms);
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

/// Images one tool result may carry; more are named in its text and left out.
pub const MAX_IMAGES: usize = 4;

/// An image content part as a data URL: MCP's `{"type":"image","data":<base64>,"mimeType":...}` or an OpenAI
/// `image_url` part. Only inline data is taken: a node asked to fetch a URL could be pointed anywhere.
fn image_url(c: &Value) -> Option<String> {
    let u = match c.get("data").and_then(|d| d.as_str()) {
        Some(d) => format!("data:{};base64,{d}", c.get("mimeType").or_else(|| c.get("mime_type")).and_then(|m| m.as_str()).unwrap_or("image/png")),
        None => c.pointer("/image_url/url").and_then(|u| u.as_str())?.to_owned(),
    };
    u.starts_with("data:image/").then_some(u)
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
            let mut images = Vec::new();
            for c in body.get("content").and_then(|c| c.as_array()).into_iter().flatten() {
                match c.get("type").and_then(|t| t.as_str()) {
                    Some("text") => {
                        if !text.is_empty() { text.push('\n'); }
                        text.push_str(c.get("text").and_then(|t| t.as_str()).unwrap_or(""));
                    }
                    Some("image") | Some("image_url") => match image_url(c) {
                        Some(u) if images.len() < MAX_IMAGES => images.push(u),
                        Some(_) => text.push_str(&format!("\n[an image beyond the first {MAX_IMAGES} was left out]")),
                        None => text.push_str("\n[an image that is not inline data was left out]"),
                    },
                    Some(other) => text.push_str(&format!("\n[{other} content omitted]")),
                    None => {}
                }
            }
            let id = format!("result-{call}-{}", short_hash(&format!("{text}{}", images.concat())));
            Event { id, thread, due_ms: 0, created_ms: 0, attempts: 0, kind: EventKind::ToolResult { call, provider_call_id, text, images } }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_result_keeps_inline_images_only() {
        let body = json!({"type": "tool_result", "group_id": "thread-1", "id": "c1", "text": "look",
            "content": [{"type": "text", "text": "a frame"},
                        {"type": "image", "data": "AAA=", "mimeType": "image/jpeg"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,BBB="}},
                        {"type": "image_url", "image_url": {"url": "https://elsewhere/x.png"}},
                        {"type": "image", "data": "CCC=", "mimeType": "text/html"}]});
        let ev = callback_event(&body).unwrap().unwrap();
        let EventKind::ToolResult { text, images, .. } = &ev.kind else { panic!("{ev:?}") };
        assert_eq!(images, &vec!["data:image/jpeg;base64,AAA=".to_owned(), "data:image/png;base64,BBB=".to_owned()]);
        assert!(text.starts_with("look\na frame"), "{text}");
        assert_eq!(text.matches("not inline data was left out").count(), 2, "{text}");
        // the same text with other images is another event, not a duplicate
        let mut other = body.clone();
        other["content"][1]["data"] = json!("ZZZ=");
        assert_ne!(callback_event(&other).unwrap().unwrap().id, ev.id);
    }

    #[test]
    fn past_four_images_are_named_not_sent() {
        let parts: Vec<Value> = (0..6).map(|i| json!({"type": "image", "data": format!("D{i}"), "mimeType": "image/png"})).collect();
        let ev = callback_event(&json!({"type": "tool_result", "group_id": "t", "id": "c", "content": parts})).unwrap().unwrap();
        let EventKind::ToolResult { text, images, .. } = &ev.kind else { panic!() };
        assert_eq!(images.len(), MAX_IMAGES);
        assert_eq!(text.matches("beyond the first 4").count(), 2);
    }
}
