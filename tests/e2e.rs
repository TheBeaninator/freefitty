//! The whole daemon against a scripted model node and a RAP tool server, both in-process.

use axum::{extract::State, routing::{get, post}, Json, Router};
use freefitty::config::Config;
use serde_json::{json, Value};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

struct FakeModel {
    fail: AtomicBool,
    /// slots the node reports busy with other work
    busy: AtomicUsize,
    calls: AtomicUsize,
    /// the next notes answer is cut off (finish_reason length)
    cut_notes: AtomicBool,
    bodies: std::sync::Mutex<Vec<Value>>,
    /// what /cache/export answers (None: 404, nothing cached), and every body /cache/import got
    kv: std::sync::Mutex<Option<Vec<u8>>>,
    imports: std::sync::Mutex<Vec<Vec<u8>>>,
    /// the next export answers an older, shorter entry, as strixite does while a turn's entries are being saved
    short_first: AtomicBool,
    /// what /cache/probe says the node holds (u64::MAX: the route does not exist), GET /steer's answer, and
    /// whether a steered request is refused (no such vector)
    probe_tokens: std::sync::atomic::AtomicU64,
    steer_list: std::sync::Mutex<Option<Value>>,
    refuse_steer: AtomicBool,
    /// GET /steer answers 500 (a node in trouble, not one without vectors)
    steer_fail: AtomicBool,
}

async fn health(State(m): State<Arc<FakeModel>>) -> Json<Value> {
    Json(json!({"status": "ok", "model": "fake", "slots": 2, "busy_slots": m.busy.load(Ordering::Relaxed), "queued": 0, "context_length": 262144}))
}

/// The script: a user message starting with "sleep" calls the sleep tool for 1 s, "echo X" calls the echo
/// tool, "nothing" ends the turn with no text and no call; a tool result is answered with "done: <result>";
/// anything else with "hi: <text>".
async fn completions(State(m): State<Arc<FakeModel>>, Json(body): Json<Value>) -> (axum::http::StatusCode, Json<Value>) {
    m.calls.fetch_add(1, Ordering::Relaxed);
    m.bodies.lock().unwrap().push(body.clone());
    if m.fail.load(Ordering::Relaxed) {
        return (axum::http::StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": "fake outage"})));
    }
    let msgs = body["messages"].as_array().cloned().unwrap_or_default();
    assert_eq!(msgs[0]["role"], "system");
    if body.get("steer").is_some() && m.refuse_steer.load(Ordering::Relaxed) {
        return (axum::http::StatusCode::BAD_REQUEST, Json(json!({"error": {"message": "steer[0]: no vector named 'villain24' (0 loaded)", "param": "steer"}})));
    }
    let last = msgs.last().cloned().unwrap_or(Value::Null);
    let content = last["content"].as_str().unwrap_or("").to_owned();
    if content.starts_with("wall") && msgs.len() > 6 {
        // the node's count disagrees with the estimate: too big until the view is cut
        return (axum::http::StatusCode::BAD_REQUEST, Json(json!({"error": {"message": "the prompt is 300000 tokens; the context holds 262144 (prompt + at least 1 generated token)"}})));
    }
    if content.starts_with("overflow") {
        // what strixite answers when the prompt does not fit
        return (axum::http::StatusCode::BAD_REQUEST, Json(json!({"error": {"message": "the prompt is 267832 tokens; the context holds 262144 (prompt + at least max_tokens)"}})));
    }
    let tool_names: Vec<&str> = body["tools"].as_array().expect("every request carries tools").iter().map(|t| t["function"]["name"].as_str().unwrap()).collect();
    // a goldfish sees only the plain built-ins and what it was granted; everyone else sees every tool
    let goldfish = msgs.iter().any(|m| m["content"].as_str().is_some_and(|c| c.starts_with("You are a goldfish")));
    assert!(tool_names.contains(&"sleep") && (goldfish || tool_names.contains(&"echo")), "tools offered: {tool_names:?}");
    if content.starts_with("Stop working and refocus") {
        // the refocus ask: notes name how many messages they replace and the focus
        let focus = content.split("The focal point to write around: ").nth(1).map(|f| f.lines().next().unwrap_or("").to_owned());
        let cut = m.cut_notes.swap(false, Ordering::Relaxed);
        let text = format!("NOTES over {} messages{}{}", msgs.len() - 1, focus.map(|f| format!(" focus={f}")).unwrap_or_default(), if cut { " (cut" } else { "" });
        return (axum::http::StatusCode::OK, Json(json!({"choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": if cut { "length" } else { "stop" }}], "usage": {"prompt_tokens": 1}})));
    }
    if msgs.len() > 1 && msgs[1]["content"].as_str().is_some_and(|c| c.starts_with("[Refocus.")) {
        // after a refocus the model sees the bridge: answer with what it sees so the test can check it
        assert_eq!(msgs[2]["role"], "assistant");
        let bridged_calls = msgs[2]["tool_calls"].as_array().map(|a| a.len()).unwrap_or(0);
        if last["role"] == "user" && !content.starts_with("refocus") {
            let message = json!({"role": "assistant", "content": format!("after refocus: {} messages, notes={}, bridged={bridged_calls}: {content}", msgs.len(), msgs[2]["content"].as_str().unwrap_or(""))});
            return (axum::http::StatusCode::OK, Json(json!({"choices": [{"message": message}], "usage": {"prompt_tokens": 1}})));
        }
    }
    if last["role"] == "user" && (content.starts_with("cutoff") || content.starts_with("[FreeFitty] Your reply reached the output limit")) {
        // a write_file call too long for the output limit: cut off twice, then written in parts
        let notes = msgs.iter().filter(|m| m["content"].as_str().is_some_and(|c| c.starts_with("[FreeFitty] Your reply reached"))).count();
        let forever = msgs.iter().any(|m| m["content"].as_str() == Some("cutoff forever"));
        if notes < 2 || forever {
            return (axum::http::StatusCode::OK, Json(json!({"choices": [{"message": {"role": "assistant", "content": format!("writing the file ({notes})")}, "finish_reason": "length"}], "usage": {"prompt_tokens": 100}})));
        }
        return (axum::http::StatusCode::OK, Json(json!({"choices": [{"message": {"role": "assistant", "content": format!("wrote it in parts after {notes} notes")}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 100}})));
    }
    let shown: Vec<String> = last["content"].as_array().into_iter().flatten()
        .filter_map(|p| p.pointer("/image_url/url").and_then(|u| u.as_str()).map(str::to_owned)).collect();
    let gf_call = |expert: &str, q: &str| json!({"role": "assistant", "content": null, "tool_calls": [{"id": format!("call_g{}", msgs.len()), "type": "function", "function": {"name": "goldfish", "arguments": json!({"expert": expert, "question": q}).to_string()}}]});
    let message = if goldfish && last["role"] == "user" {
        // the goldfish: tries to ask a goldfish when told to, else answers with the tools it was offered
        if content.ends_with("try goldfish") {
            json!({"role": "assistant", "content": null, "tool_calls": [{"id": "call_gg", "type": "function", "function": {"name": "goldfish", "arguments": "{\"expert\": \"railhand\", \"question\": \"x\"}"}}]})
        } else {
            let mut t: Vec<&str> = tool_names.clone();
            t.sort();
            json!({"role": "assistant", "content": format!("fish(tools={}): {}", t.join(","), content.lines().last().unwrap_or(""))})
        }
    } else if last["role"] == "user" && content == "catalog" {
        json!({"role": "assistant", "content": null, "tool_calls": [{"id": "call_c", "type": "function", "function": {"name": "experts", "arguments": "{}"}}]})
    } else if let Some(q) = content.strip_prefix("ask ").filter(|_| last["role"] == "user") {
        gf_call("railhand", q)
    } else if let Some(q) = content.strip_prefix("askr2 ").filter(|_| last["role"] == "user") {
        gf_call("rail2", q)
    } else if content == "askbad" {
        gf_call("nobody", "x")
    } else if let Some(n) = content.strip_prefix("commission ").filter(|_| last["role"] == "user") {
        json!({"role": "assistant", "content": null, "tool_calls": [{"id": "call_m", "type": "function", "function": {"name": "commission_expert", "arguments": json!({"name": n, "description": "knows rail two", "reads": [{"text": "C is .52"}], "tools": ["echo", "goldfish"], "steer": if n.starts_with("persona") { json!([{"vector": "villain24", "scale": 6.0}]) } else { json!([]) }}).to_string()}}]})
    } else if !shown.is_empty() {
        // a tool's images arrive as a user message after its result: say what was seen and what came before
        let before = &msgs[msgs.len() - 2];
        json!({"role": "assistant", "content": format!("saw {} image(s) {} after {}: {}", shown.len(), shown.join(","), before["role"].as_str().unwrap_or(""), before["content"].as_str().unwrap_or(""))})
    } else if last["role"] == "tool" {
        json!({"role": "assistant", "content": format!("done: {content}")})
    } else if let Some(n) = content.strip_prefix("recall ") {
        // the model asks for a stored message back
        json!({"role": "assistant", "content": null, "tool_calls": [{"id": "call_r", "type": "function", "function": {"name": "recall", "arguments": format!("{{\"indices\": [{n}]}}")}}]})
    } else if content.starts_with("long ") {
        json!({"role": "assistant", "content": format!("long answer: {}", "y".repeat(600))})
    } else if content.starts_with("sleep") {
        json!({"role": "assistant", "content": null, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "sleep", "arguments": "{\"seconds\": 1}"}}]})
    } else if let Some(x) = content.strip_prefix("echo ") {
        // arguments JSON-encoded twice, as a model sometimes writes them
        json!({"role": "assistant", "content": "calling echo", "tool_calls": [{"id": "call_2", "type": "function", "function": {"name": "echo", "arguments": serde_json::to_string(&json!({"text": x}).to_string()).unwrap()}}]})
    } else if content.starts_with("refocus") {
        let focus = content.strip_prefix("refocus ").unwrap_or("").to_owned();
        json!({"role": "assistant", "content": "ok, refocusing", "tool_calls": [{"id": "call_r", "type": "function", "function": {"name": "refocus", "arguments": json!({"focus": focus}).to_string()}}]})
    } else if content.starts_with("bogus") {
        json!({"role": "assistant", "content": null, "tool_calls": [{"id": "call_3", "type": "function", "function": {"name": "no_such_tool", "arguments": "{}"}}]})
    } else if content.starts_with("nothing") {
        json!({"role": "assistant", "content": ""})
    } else {
        json!({"role": "assistant", "content": format!("hi: {content}"), "reasoning_content": "dropped"})
    };
    (axum::http::StatusCode::OK, Json(json!({"choices": [{"message": message}], "usage": {"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": 90}}})))
}

async fn export(State(m): State<Arc<FakeModel>>) -> axum::response::Response {
    use axum::response::IntoResponse;
    match m.kv.lock().unwrap().clone() {
        // the entry covers the whole prompt the scripted turns report (100 tokens)
        Some(b) => ([("x-cache-tokens", if m.short_first.swap(false, Ordering::Relaxed) { "10" } else { "100" })], b).into_response(),
        None => axum::http::StatusCode::NOT_FOUND.into_response(),
    }
}

async fn probe(State(m): State<Arc<FakeModel>>) -> axum::response::Response {
    use axum::response::IntoResponse;
    match m.probe_tokens.load(Ordering::Relaxed) {
        u64::MAX => axum::http::StatusCode::NOT_FOUND.into_response(),
        n => Json(json!({"tokens": n, "ram": true})).into_response(),
    }
}

async fn steer_list(State(m): State<Arc<FakeModel>>) -> axum::response::Response {
    use axum::response::IntoResponse;
    if m.steer_fail.load(Ordering::Relaxed) { return axum::http::StatusCode::INTERNAL_SERVER_ERROR.into_response(); }
    match m.steer_list.lock().unwrap().clone() {
        Some(v) => Json(v).into_response(),
        None => axum::http::StatusCode::NOT_FOUND.into_response(),
    }
}

/// A saved state file in strixite's prompt cache layout: one entry of `n` tokens.
fn kv_file(n: u64) -> Vec<u8> {
    let state = 64usize;
    let state_at = (4096 + n as usize * 4).div_ceil(4096) * 4096;
    let mut f = vec![0u8; state_at + state];
    f[..8].copy_from_slice(b"STRXPCCH");
    f[8..12].copy_from_slice(&2u32.to_le_bytes());
    f[12..16].copy_from_slice(&2u32.to_le_bytes());
    f[16..24].copy_from_slice(&n.to_le_bytes());
    f[24..32].copy_from_slice(&(state as u64).to_le_bytes());
    f[32..40].copy_from_slice(&(state_at as u64).to_le_bytes());
    f
}

async fn import(State(m): State<Arc<FakeModel>>, body: bytes::Bytes) -> Json<Value> {
    m.imports.lock().unwrap().push(body.to_vec());
    Json(json!({"status": "ok"}))
}

struct FakeTools {
    base: String,
    invocations: AtomicUsize,
}

async fn manifest(State(t): State<Arc<FakeTools>>) -> Json<Value> {
    Json(json!({"name": "fake", "endpoint": format!("{}/invoke", t.base), "tools": [{"name": "echo", "description": "echoes", "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}}}]}))
}

async fn invoke(State(t): State<Arc<FakeTools>>, Json(inv): Json<Value>) -> &'static str {
    t.invocations.fetch_add(1, Ordering::Relaxed);
    assert_eq!(inv["operation"], "echo");
    assert!(inv["arguments"]["text"].is_string(), "arguments were repaired to an object: {}", inv["arguments"]);
    let cb = inv["callback_url"].as_str().unwrap().to_owned();
    let mut result = json!({"type": "tool_result", "group_id": inv["group_id"], "id": inv["id"], "call_id": inv["call_id"], "text": format!("echo: {}", inv["arguments"]["text"].as_str().unwrap())});
    if inv["arguments"]["text"] == "picture" {
        // an MCP image part beside the text, and one that is a remote URL (left out: only inline data goes on)
        result["content"] = json!([{"type": "image", "data": "iVBORw0KGgo=", "mimeType": "image/png"},
                                   {"type": "image_url", "image_url": {"url": "http://example.invalid/x.png"}}]);
    }
    let delay = if inv["arguments"]["text"] == "slow" { 6000 } else { 200 };
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(delay)).await;
        let c = reqwest::Client::new();
        // posted twice: the second must be a duplicate, not a second turn
        for _ in 0..2 {
            let r = c.post(&cb).json(&result).send().await.unwrap();
            assert!(r.status().is_success(), "callback answered {}", r.status());
        }
    });
    "OK"
}

async fn serve(router: Router) -> String {
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = l.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(l, router).await.unwrap() });
    format!("http://{addr}")
}

struct Harness {
    model: Arc<FakeModel>,
    tools: Arc<FakeTools>,
    cfg: Config,
    _dir: tempfile::TempDir,
}

async fn harness() -> Harness {
    let model = Arc::new(FakeModel { fail: AtomicBool::new(false), busy: AtomicUsize::new(0), calls: AtomicUsize::new(0), cut_notes: AtomicBool::new(false), bodies: std::sync::Mutex::new(Vec::new()), kv: std::sync::Mutex::new(None), imports: std::sync::Mutex::new(Vec::new()), short_first: AtomicBool::new(false), probe_tokens: std::sync::atomic::AtomicU64::new(u64::MAX), steer_list: std::sync::Mutex::new(None), refuse_steer: AtomicBool::new(false), steer_fail: AtomicBool::new(false) });
    let model_url = serve(Router::new().route("/health", get(health)).route("/v1/chat/completions", post(completions)).route("/cache/export", post(export)).route("/cache/import", post(import)).route("/cache/probe", post(probe)).route("/steer", get(steer_list)).with_state(model.clone())).await;
    let tools = Arc::new(FakeTools { base: String::new(), invocations: AtomicUsize::new(0) });
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let tools_url = format!("http://{}", l.local_addr().unwrap());
    let tools = Arc::new(FakeTools { base: tools_url.clone(), invocations: AtomicUsize::new(tools.invocations.load(Ordering::Relaxed)) });
    let r = Router::new().route("/.well-known/rap-toolset", get(manifest)).route("/invoke", post(invoke)).with_state(tools.clone());
    tokio::spawn(async move { axum::serve(l, r).await.unwrap() });
    let dir = tempfile::tempdir().unwrap();
    let mut cfg = Config::fresh();
    cfg.listen = "127.0.0.1:0".into();
    cfg.data_dir = dir.path().join("data");
    cfg.nodes = vec![model_url];
    cfg.tool_servers = vec![tools_url];
    cfg.retry_after_s = 1;
    cfg.node_down_wait_s = 2;
    Harness { model, tools, cfg, _dir: dir }
}

struct Api {
    base: String,
    token: String,
    c: reqwest::Client,
}

impl Api {
    fn new(r: &freefitty::Running) -> Api {
        Api { base: format!("http://{}", r.addr), token: r.app.cfg.operator_token.clone(), c: reqwest::Client::new() }
    }
    async fn say(&self, text: &str, thread: Option<&str>) -> Value {
        let mut b = json!({"text": text});
        if let Some(t) = thread { b["thread"] = json!(t); }
        self.c.post(format!("{}/say", self.base)).bearer_auth(&self.token).json(&b).send().await.unwrap().json().await.unwrap()
    }
    async fn output(&self, thread: &str, wait: u64) -> Option<Value> {
        let r = self.c.get(format!("{}/output?wait={wait}&thread={thread}", self.base)).bearer_auth(&self.token).send().await.unwrap();
        if r.status().as_u16() == 204 { None } else { Some(r.json().await.unwrap()) }
    }
    async fn get(&self, path: &str) -> Value {
        self.c.get(format!("{}{path}", self.base)).bearer_auth(&self.token).send().await.unwrap().json().await.unwrap()
    }
    async fn post(&self, path: &str, body: Value) -> Value {
        self.c.post(format!("{}{path}", self.base)).bearer_auth(&self.token).json(&body).send().await.unwrap().json().await.unwrap()
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn turns_sleeps_tools_and_dead_letters() {
    let h = harness().await;
    let r = freefitty::start(h.cfg.clone()).await.unwrap();
    let api = Api::new(&r);

    // no token: refused; wrong path: 404
    let s = reqwest::Client::new().post(format!("{}/say", api.base)).json(&json!({"text": "x"})).send().await.unwrap().status();
    assert_eq!(s.as_u16(), 401);

    // a plain turn
    let sent = api.say("hello", None).await;
    let t = sent["thread"].as_str().unwrap().to_owned();
    let out = api.output(&t, 10).await.expect("a reply");
    assert_eq!(out["text"], "hi: hello");
    assert_eq!(out["metadata"]["reply_to"], sent["id"]);
    assert_eq!(out["metadata"]["thread"], t);

    // a sleep: the slice ends, the wake-up comes 1 s later, the next slice answers the tool result
    api.say("sleep please", Some(&t)).await;
    let out = api.output(&t, 10).await.expect("the tool call line");
    assert!(out["text"].as_str().unwrap().contains("[Tool Call: sleep"), "{out}");
    let out = api.output(&t, 10).await.expect("the reply after waking");
    assert_eq!(out["text"], "done: Slept for 1 seconds");

    // a RAP tool: invoked once, its result posted twice, answered once
    api.say("echo foo", Some(&t)).await;
    let out = api.output(&t, 10).await.expect("text + call line");
    assert!(out["text"].as_str().unwrap().starts_with("calling echo\n[Tool Call: echo"), "{out}");
    let out = api.output(&t, 10).await.expect("the reply after the tool");
    assert_eq!(out["text"], "done: echo: foo");
    assert_eq!(h.tools.invocations.load(Ordering::Relaxed), 1);
    assert!(api.output(&t, 1).await.is_none(), "no second turn for the duplicate result");

    // the conversation is stored in OpenAI shape, reasoning dropped, arguments repaired
    let msgs = api.get(&format!("/threads/{t}/messages")).await;
    let msgs = msgs.as_array().unwrap();
    assert_eq!(msgs[0]["role"], "user");
    assert!(msgs.iter().all(|m| m.get("reasoning_content").is_none()));
    let echo_call = msgs.iter().find(|m| m["tool_calls"][0]["function"]["name"] == "echo").unwrap();
    assert_eq!(echo_call["tool_calls"][0]["function"]["arguments"], "{\"text\":\"foo\"}");
    let th = api.get(&format!("/threads/{t}")).await;
    assert_eq!(th["pending"].as_array().unwrap().len(), 0);
    assert!(th["node"].as_str().unwrap().starts_with("http://127.0.0.1:"));

    // an unknown tool is explained to the model in the same slice; it sees the error as a tool result
    api.say("bogus", Some(&t)).await;
    let out = api.output(&t, 10).await.expect("a reply");
    assert!(out["text"].as_str().unwrap().starts_with("done: Error: there is no tool named \"no_such_tool\""), "{out}");

    // a turn with nothing in it produces no reply
    api.say("nothing", Some(&t)).await;
    assert!(api.output(&t, 2).await.is_none());

    // the model fails: three attempts, then a dead letter and an error reply; redrive after it recovers
    h.model.fail.store(true, Ordering::Relaxed);
    let sent = api.say("after outage", Some(&t)).await;
    let out = api.output(&t, 20).await.expect("the error reply");
    assert_eq!(out["metadata"]["kind"], "error");
    assert_eq!(out["metadata"]["reply_to"], sent["id"]);
    let dlq = api.get("/dlq").await;
    assert_eq!(dlq.as_array().unwrap().len(), 1);
    assert_eq!(dlq[0]["attempts"], 3);
    h.model.fail.store(false, Ordering::Relaxed);
    let rd = api.post("/dlq/redrive", json!({"thread": t})).await;
    assert_eq!(rd["redriven"], 1);
    let out = api.output(&t, 10).await.expect("the redriven turn's reply");
    assert_eq!(out["text"], "hi: after outage");
    assert_eq!(out["metadata"]["reply_to"], sent["id"], "a redriven event keeps its id");
    assert!(api.get("/dlq").await.as_array().unwrap().is_empty());

    // two threads at once, each strictly ordered
    let a = api.say("one", Some("thread-a")).await;
    let b = api.say("one", Some("thread-b")).await;
    api.say("two", Some("thread-a")).await;
    assert_eq!(api.output("thread-a", 10).await.unwrap()["metadata"]["reply_to"], a["id"]);
    assert_eq!(api.output("thread-b", 10).await.unwrap()["metadata"]["reply_to"], b["id"]);
    assert_eq!(api.output("thread-a", 10).await.unwrap()["text"], "hi: two");

    // health reports the node and the counts
    let hv = api.get("/health").await;
    assert_eq!(hv["nodes"][0]["up"], true);
    assert_eq!(hv["tools"], 1);
    assert!(hv["router"]["kept"].as_u64().unwrap() >= 5, "{}", hv["router"]);
    // idle: nothing waits to start, and the starvation gauge says so
    assert_eq!(hv["queue"]["runnable"], 0, "{}", hv["queue"]);
    assert_eq!(hv["queue"]["oldest_runnable_s"], 0, "{}", hv["queue"]);

    r.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn a_restart_keeps_the_queue_and_the_sleep() {
    let h = harness().await;
    let r = freefitty::start(h.cfg.clone()).await.unwrap();
    let api = Api::new(&r);
    let sent = api.say("sleep then", None).await;
    let t = sent["thread"].as_str().unwrap().to_owned();
    let out = api.output(&t, 10).await.unwrap();
    assert!(out["text"].as_str().unwrap().contains("[Tool Call: sleep"));
    // the wake-up is queued for 1 s from now; stop before it runs, and queue a message that cannot run
    r.shutdown().await;
    let calls_before = h.model.calls.load(Ordering::Relaxed);
    tokio::time::sleep(Duration::from_millis(1500)).await;
    assert_eq!(h.model.calls.load(Ordering::Relaxed), calls_before, "nothing runs while stopped");

    let r = freefitty::start(h.cfg.clone()).await.unwrap();
    let api = Api::new(&r);
    let out = api.output(&t, 10).await.expect("the wake-up ran after the restart");
    assert_eq!(out["text"], "done: Slept for 1 seconds");
    // the conversation is intact
    let msgs = api.get(&format!("/threads/{t}/messages")).await;
    assert_eq!(msgs.as_array().unwrap().len(), 4, "{msgs}");
    r.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn callbacks_need_their_token_and_a_thread() {
    let h = harness().await;
    let r = freefitty::start(h.cfg.clone()).await.unwrap();
    let c = reqwest::Client::new();
    let base = format!("http://{}", r.addr);
    let body = json!({"type": "tool_result", "group_id": "nothread", "id": "x", "text": "late"});
    let s = c.post(format!("{base}/rap/callback/{}", r.app.cfg.operator_token)).json(&body).send().await.unwrap().status();
    assert_eq!(s.as_u16(), 401, "the operator token is not the callback token");
    let s = c.post(format!("{base}/rap/callback")).bearer_auth(&r.app.cfg.callback_token).json(&json!({"type": "tool_result"})).send().await.unwrap().status();
    assert_eq!(s.as_u16(), 400);
    let s = c.post(format!("{base}/rap/callback/{}", r.app.cfg.callback_token)).json(&body).send().await.unwrap().status();
    assert_eq!(s.as_u16(), 200, "a result for a thread that has no pending call still queues; the model sees it");
    let api = Api::new(&r);
    let out = api.output("nothread", 10).await.unwrap();
    assert_eq!(out["text"], "hi: [late result for tool call x] late", "a result nobody awaits is said in the user role");
    r.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn refocus_folds_history_and_keeps_pending_calls() {
    let mut h = harness().await;
    h.cfg.compaction = true;
    h.cfg.compact_at_tokens = 0;
    h.cfg.compact_keep_messages = 2;
    let r = freefitty::start(h.cfg.clone()).await.unwrap();
    let api = Api::new(&r);
    let t = "refocus-1";
    for i in 0..3 {
        api.say(&format!("turn {i}"), Some(t)).await;
        assert_eq!(api.output(t, 10).await.unwrap()["text"], format!("hi: turn {i}"));
    }
    // a pending RAP call whose result comes late, then the model asks to refocus
    api.say("echo slow", Some(t)).await;
    api.output(t, 10).await.unwrap();
    api.say("refocus the slow echo", Some(t)).await;
    let out = api.output(t, 10).await.unwrap();
    assert!(out["text"].as_str().unwrap().contains("[Tool Call: refocus"), "{out}");
    // the next slice refocuses first, then answers the refocus tool result with the bridge in view
    let out = api.output(t, 10).await.unwrap();
    assert!(out["text"].as_str().unwrap().starts_with("done: Refocused"), "{out}");
    let th = api.get(&format!("/threads/{t}")).await;
    let cp = &th["compaction"];
    assert_eq!(cp["count"], 1, "{th}");
    assert_eq!(cp["focus"], "the slow echo");
    assert_eq!(cp["truncated"], false);
    let n = th["messages"].as_u64().unwrap();
    // the cut keeps the last 2 history messages, moved back past leading tool results
    let hist = api.get(&format!("/threads/{t}/messages")).await;
    let hist = hist.as_array().unwrap();
    assert_eq!(hist.len() as u64, n, "the full history is intact");
    let from = cp["from"].as_u64().unwrap() as usize;
    assert_ne!(hist[from]["role"], "tool");
    // the notes request carried the view up to the cut and the same tools
    let bodies = h.model.bodies.lock().unwrap().clone();
    let notes_req = bodies.iter().find(|b| b["messages"].as_array().unwrap().last().unwrap()["content"].as_str().unwrap().starts_with("Stop working")).unwrap();
    assert!(notes_req["tools"].as_array().unwrap().iter().any(|d| d["function"]["name"] == "refocus"));
    assert!(notes_req["messages"].as_array().unwrap().last().unwrap()["content"].as_str().unwrap().contains("Tool calls still awaiting results:"), "pending calls are named");
    // the live view: system, framing, notes with the pending echo call bridged, kept tail
    let live = api.get(&format!("/threads/{t}/messages?view=live")).await;
    let live = live.as_array().unwrap();
    assert_eq!(live[0]["role"], "system");
    assert!(live[1]["content"].as_str().unwrap().starts_with("[Refocus."));
    assert!(live[2]["content"].as_str().unwrap().starts_with("NOTES over"), "{}", live[2]);
    let bridged: Vec<&str> = live[2]["tool_calls"].as_array().map(|a| a.iter().map(|c| c["function"]["name"].as_str().unwrap()).collect()).unwrap_or_default();
    assert!(bridged.contains(&"echo"), "the awaited echo call is bridged: {bridged:?}");
    assert_eq!(live.len(), 3 + (hist.len() - from));
    // the late result still resolves: the model answers it
    let out = api.output(t, 10).await.expect("the slow echo's result");
    assert_eq!(out["text"], "done: echo: slow");
    // a refocus with only the kept tail to fold does nothing: no second record, no notes call
    let t2 = "refocus-2";
    for i in 0..2 { api.say(&format!("t2 {i}"), Some(t2)).await; api.output(t2, 10).await.unwrap(); }
    api.post(&format!("/threads/{t2}/compact"), json!({})).await;
    tokio::time::sleep(Duration::from_millis(800)).await;
    assert_eq!(api.get(&format!("/threads/{t2}")).await["compaction"]["from"], 2);
    let calls_before = h.model.calls.load(Ordering::Relaxed);
    api.post(&format!("/threads/{t2}/compact"), json!({})).await;
    tokio::time::sleep(Duration::from_millis(800)).await;
    assert_eq!(api.get(&format!("/threads/{t2}")).await["compaction"]["count"], 1);
    assert_eq!(h.model.calls.load(Ordering::Relaxed), calls_before, "no notes call when there is nothing to fold");
    // after more turns an operator refocus with a focus hint; the model sees the bridge on its next turn
    for i in 0..3 { api.say(&format!("more {i}"), Some(t)).await; api.output(t, 10).await.unwrap(); }
    h.model.cut_notes.store(true, Ordering::Relaxed);
    let v = api.post(&format!("/threads/{t}/compact"), json!({"focus": "colours"})).await;
    assert_eq!(v["queued"], true);
    api.say("what now", Some(t)).await;
    let out = api.output(t, 10).await.unwrap();
    let text = out["text"].as_str().unwrap();
    assert!(text.starts_with("after refocus:") && text.contains("focus=colours") && text.contains("bridged=0") && text.ends_with("what now"), "{text}");
    let th = api.get(&format!("/threads/{t}")).await;
    assert_eq!(th["compaction"]["count"], 2);
    assert_eq!(th["compaction"]["truncated"], false, "a cut-off notes answer was asked again at half length");
    let cut_requests = h.model.bodies.lock().unwrap().iter().filter(|b| b["messages"].as_array().unwrap().last().unwrap()["content"].as_str().unwrap().contains("half of what you would")).count();
    assert_eq!(cut_requests, 1);
    // the notes failing is transient: three times, then dead letters, no record
    h.model.fail.store(true, Ordering::Relaxed);
    for i in 0..3 { api.say(&format!("again {i}"), Some(t)).await; api.output(t, 10).await; }
    h.model.fail.store(false, Ordering::Relaxed);
    r.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn token_trigger_refocuses_once() {
    let mut h = harness().await;
    h.cfg.compaction = true;
    h.cfg.compact_at_tokens = 1200;
    h.cfg.compact_summary_tokens = 100;
    h.cfg.compact_keep_messages = 2;
    let r = freefitty::start(h.cfg.clone()).await.unwrap();
    let api = Api::new(&r);
    let t = "tokens-1";
    let big = "x".repeat(900);
    for i in 0..6 {
        api.say(&format!("{big} {i}"), Some(t)).await;
        api.output(t, 10).await.unwrap();
    }
    let th = api.get(&format!("/threads/{t}")).await;
    let count = th["compaction"]["count"].as_u64().unwrap();
    assert!(count >= 1, "{th}");
    // every refocus folded something new: no two records share a `from`
    let msgs_live = api.get(&format!("/threads/{t}/messages?view=live")).await;
    assert!(msgs_live.as_array().unwrap().len() < 8, "the live view is short");
    assert_eq!(api.get(&format!("/threads/{t}/messages")).await.as_array().unwrap().len(), 12, "history intact");
    r.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn experts_share_their_prefix_bytes() {
    let mut h = harness().await;
    h.cfg.compaction = true;
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("doc.md"), "The rail is 10.44.0.0/24.\n").unwrap();
    std::fs::write(dir.path().join("railhand.toml"), format!("description = \"knows the rail\"\nack = \"Read.\"\n[[reads]]\npath = \"{}\"\n[[reads]]\ntext = \"B is .51\"\n", dir.path().join("doc.md").display())).unwrap();
    h.cfg.experts_dir = Some(dir.path().to_path_buf());
    h.cfg.warm_on_start = vec!["railhand".into()];
    let r = freefitty::start(h.cfg.clone()).await.unwrap();
    let api = Api::new(&r);
    let list = api.get("/experts").await;
    assert_eq!(list[0]["name"], "railhand");
    let key = list[0]["key"].as_str().unwrap().to_owned();
    assert!(key.starts_with("railhand@"));
    // two threads born from the expert send identical first three messages
    let a = api.post("/say", json!({"text": "hello a", "thread": "ex-a", "expert": "railhand"})).await;
    assert_eq!(a["thread"], "ex-a");
    api.post("/say", json!({"text": "hello b", "thread": "ex-b", "expert": "railhand"})).await;
    assert_eq!(api.output("ex-a", 10).await.unwrap()["text"], "hi: hello a");
    assert_eq!(api.output("ex-b", 10).await.unwrap()["text"], "hi: hello b");
    let bodies = h.model.bodies.lock().unwrap().clone();
    let of = |thread_text: &str| bodies.iter().find(|b| b["messages"].as_array().unwrap().last().unwrap()["content"] == thread_text).cloned().unwrap();
    let (ba, bb) = (of("hello a"), of("hello b"));
    assert_eq!(ba["messages"].as_array().unwrap()[0..3], bb["messages"].as_array().unwrap()[0..3]);
    assert!(ba["messages"][1]["content"].as_str().unwrap().contains("<document path=") && ba["messages"][1]["content"].as_str().unwrap().contains("B is .51"));
    assert_eq!(ba["messages"][2]["content"], "Read.");
    assert_eq!(api.get("/threads/ex-a").await["expert"], key);
    // the plain thread shares the system message with them
    api.say("plain", Some("ex-plain")).await;
    api.output("ex-plain", 10).await.unwrap();
    let bp = h.model.bodies.lock().unwrap().iter().find(|b| b["messages"].as_array().unwrap().last().unwrap()["content"] == "plain").cloned().unwrap();
    assert_eq!(bp["messages"][0], ba["messages"][0]);
    // warm on start ran: the expert read on the node (it waits for the node, then a moment); this node keeps no
    // state to export, so the reading is the warm
    let mut warmed = false;
    for _ in 0..40 {
        warmed = h.model.bodies.lock().unwrap().iter().any(|b| b["messages"].as_array().unwrap().last().unwrap()["content"] == "Say ready." && b["messages"][2]["content"] == "Read.");
        if warmed { break; }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    assert!(warmed, "warm_on_start read the prefix on the node");
    let w = api.post("/experts/railhand/warm", json!({})).await;
    assert_eq!(w["nodes"].as_array().unwrap().len(), 1);
    // an existing thread cannot be re-pointed at an expert; a wrong name is 404
    let s = reqwest::Client::new().post(format!("{}/say", api.base)).bearer_auth(&api.token).json(&json!({"text": "x", "thread": "ex-plain", "expert": "railhand"})).send().await.unwrap().status();
    assert_eq!(s.as_u16(), 409);
    let s = reqwest::Client::new().post(format!("{}/say", api.base)).bearer_auth(&api.token).json(&json!({"text": "x", "expert": "nobody"})).send().await.unwrap().status();
    assert_eq!(s.as_u16(), 404);
    // a changed file is a new version; ex-a keeps the bytes it was born with
    std::fs::write(dir.path().join("doc.md"), "The rail is 10.44.0.0/24, 100 GbE.\n").unwrap();
    api.post("/experts/reload", json!({})).await;
    let list = api.get("/experts").await;
    assert_ne!(list[0]["key"], key);
    assert_eq!(list[0]["threads_on_older_versions"], 2);
    api.say("still a", Some("ex-a")).await;
    api.output("ex-a", 10).await.unwrap();
    let again = h.model.bodies.lock().unwrap().iter().find(|b| b["messages"].as_array().unwrap().last().unwrap()["content"] == "still a").cloned().unwrap();
    assert_eq!(again["messages"][1], ba["messages"][1]);
    // freeze a thread's focal point as an expert
    api.post("/say", json!({"text": "refocus rail facts", "thread": "ex-a"})).await;
    api.output("ex-a", 10).await.unwrap();
    api.output("ex-a", 10).await.unwrap();
    let f = api.post("/experts/from-thread/ex-a", json!({"name": "rail-focus", "description": "frozen"})).await;
    assert!(f["key"].as_str().unwrap().starts_with("rail-focus@"), "{f}");
    api.post("/say", json!({"text": "hello c", "thread": "ex-c", "expert": "rail-focus"})).await;
    api.output("ex-c", 10).await.unwrap();
    let bc = h.model.bodies.lock().unwrap().iter().find(|b| b["messages"].as_array().unwrap().last().unwrap()["content"] == "hello c").cloned().unwrap();
    assert!(bc["messages"][1]["content"].as_str().unwrap().contains("NOTES over"), "{}", bc["messages"][1]);
    r.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn a_migrated_history_continues_under_its_thread_id() {
    let h = harness().await;
    let r = freefitty::start(h.cfg.clone()).await.unwrap();
    let api = Api::new(&r);
    let history = json!([
        {"role": "user", "content": "first ask"},
        {"role": "assistant", "content": "calling", "tool_calls": [{"id": "call_old", "type": "function", "function": {"name": "echo", "arguments": "{\"text\":\"x\"}"}}]},
        {"role": "tool", "tool_call_id": "call_old", "content": "echo: x"},
    ]);
    let c = reqwest::Client::new();
    let put = |body: Value| c.put(format!("{}/threads/moved-1/messages", api.base)).bearer_auth(&api.token).json(&body);
    let s = put(json!({"messages": history})).send().await.unwrap();
    assert_eq!(s.status().as_u16(), 200);
    let meta = api.get("/threads/moved-1").await;
    assert_eq!(meta["messages"], 3);
    assert_eq!(meta["last"], "tool_result");
    assert_eq!(put(json!({"messages": [{"role": "user", "content": "again"}]})).send().await.unwrap().status().as_u16(), 409, "a thread with messages keeps them");
    assert_eq!(put(json!({"messages": [{"content": "no role"}]})).send().await.unwrap().status().as_u16(), 400);
    api.say("hello after the move", Some("moved-1")).await;
    let out = api.output("moved-1", 10).await.unwrap();
    assert_eq!(out["text"], "hi: hello after the move");
    let bodies = h.model.bodies.lock().unwrap().clone();
    let seen = bodies.iter().find(|b| b["messages"].as_array().unwrap().last().unwrap()["content"] == "hello after the move").unwrap();
    let msgs = seen["messages"].as_array().unwrap();
    assert_eq!(msgs.len(), 5, "system + the three imported + the new one");
    assert_eq!(msgs[1]["content"], "first ask");
    assert_eq!(msgs[3]["tool_call_id"], "call_old");
    assert_eq!(api.get("/threads/moved-1/messages?tail=1").await.as_array().unwrap().len(), 1);
    r.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn a_full_context_closes_the_thread_at_once() {
    let h = harness().await;
    let r = freefitty::start(h.cfg.clone()).await.unwrap();
    let api = Api::new(&r);
    api.say("hello", Some("thread-full")).await;
    assert_eq!(api.output("thread-full", 10).await.unwrap()["text"], "hi: hello");
    let before = h.model.calls.load(Ordering::Relaxed);
    let sent = api.say("overflow now", Some("thread-full")).await;
    let out = api.output("thread-full", 10).await.expect("the error reply");
    assert_eq!(out["metadata"]["kind"], "error");
    assert_eq!(out["metadata"]["reply_to"], sent["id"]);
    assert!(out["text"].as_str().unwrap().contains("context full"), "{out}");
    assert_eq!(h.model.calls.load(Ordering::Relaxed), before + 1, "one request: a two-message thread has nothing the window could cut, so no retry");
    let th = api.get("/threads/thread-full").await;
    assert!(th["closed"].as_str().unwrap().contains("context full"), "{th}");
    assert_eq!(api.get("/dlq").await.as_array().unwrap().len(), 1);
    assert!(th["prompt_tokens"].as_u64().unwrap() > 0 && th["prompt_est"].as_u64().unwrap() > 0, "{th}");
    assert_eq!(th["window"]["cuts"], json!(0), "{th}");
    r.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn waiting_for_a_slot_is_not_a_timeout() {
    let mut h = harness().await;
    h.cfg.slice_timeout_s = 1;
    h.model.busy.store(2, Ordering::Relaxed);
    let r = freefitty::start(h.cfg.clone()).await.unwrap();
    let api = Api::new(&r);
    let sent = api.say("hello", Some("thread-wait")).await;
    assert!(api.output("thread-wait", 3).await.is_none(), "the node is full: nothing runs");
    assert_eq!(h.model.calls.load(Ordering::Relaxed), 0);
    h.model.busy.store(0, Ordering::Relaxed);
    let out = api.output("thread-wait", 10).await.expect("the turn after the slot frees");
    assert_eq!(out["text"], "hi: hello");
    assert_eq!(out["metadata"]["reply_to"], sent["id"]);
    assert!(api.get("/dlq").await.as_array().unwrap().is_empty(), "a 3 s wait under a 1 s slice timeout is not a failure");
    assert!(api.get("/health").await["router"]["waited"].as_u64().unwrap() >= 1);
    r.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn a_first_message_waits_out_a_dead_node_and_a_full_one() {
    // FF-LOSTQ 9aff78d02a: one node down, the other full; the round's first message queues, then runs once a slot frees
    let mut h = harness().await;
    h.cfg.slice_timeout_s = 1;
    h.cfg.nodes.insert(0, "http://127.0.0.1:9".into());
    h.model.busy.store(2, Ordering::Relaxed);
    let r = freefitty::start(h.cfg.clone()).await.unwrap();
    let api = Api::new(&r);
    let sent = api.say("round one", Some("thread-lowcap")).await;
    assert!(api.output("thread-lowcap", 4).await.is_none(), "no free slot anywhere: nothing runs");
    assert_eq!(h.model.calls.load(Ordering::Relaxed), 0);
    h.model.busy.store(0, Ordering::Relaxed);
    let out = api.output("thread-lowcap", 15).await.expect("the first message runs once capacity returns");
    assert_eq!(out["text"], "hi: round one");
    assert_eq!(out["metadata"]["reply_to"], sent["id"]);
    assert!(api.get("/dlq").await.as_array().unwrap().is_empty(), "the wait is not a lost message");
    r.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn the_window_pages_old_turns_out_and_recall_brings_them_back() {
    let mut h = harness().await;
    h.cfg.window_tokens = 1500;
    h.cfg.window_trim_tokens = 900;
    let r = freefitty::start(h.cfg.clone()).await.unwrap();
    let api = Api::new(&r);
    let t = "thread-window";
    api.say("first thing: the password is swordfish", Some(t)).await;
    assert_eq!(api.output(t, 10).await.unwrap()["text"], "hi: first thing: the password is swordfish");
    // long turns until the view passes the budget
    let mut cuts = 0;
    // the cut happens at the start of the slice whose view would pass the budget
    for i in 0..12 {
        api.say(&format!("long {i}"), Some(t)).await;
        let out = api.output(t, 10).await.expect("a reply");
        assert!(out["text"].as_str().unwrap().starts_with("long answer"), "{out}");
        let th = api.get(&format!("/threads/{t}")).await;
        eprintln!("turn {i}: live_tokens {} window {}", th["live_tokens"], th["window"]);
        cuts = th["window"]["cuts"].as_u64().unwrap();
        if cuts > 0 { break; }
    }
    assert!(cuts > 0, "the window never cut");
    let th = api.get(&format!("/threads/{t}")).await;
    let from = th["window"]["from"].as_u64().unwrap();
    assert!(from > 0 && th["live_tokens"].as_u64().unwrap() < 1500, "{th}");
    // what the model saw: the notice after the prefix, then the tail; the first message is only in the notice
    let bodies = h.model.bodies.lock().unwrap().clone();
    let last = bodies.last().unwrap();
    let msgs = last["messages"].as_array().unwrap();
    assert!(msgs[1]["content"].as_str().unwrap().starts_with("[Paged. Messages #0 to #"), "{}", msgs[1]);
    assert!(msgs[1]["content"].as_str().unwrap().contains("#0: \"first thing: the password is swordfish\""), "{}", msgs[1]);
    assert_eq!(msgs[2]["role"], "user", "the tail starts at an input");
    assert!(msgs.iter().skip(2).all(|m| m["content"].as_str() != Some("first thing: the password is swordfish")));
    let tools: Vec<&str> = last["tools"].as_array().unwrap().iter().map(|t| t["function"]["name"].as_str().unwrap()).collect();
    assert!(tools.contains(&"recall"), "{tools:?}");
    // the frontier holds: the next turn's notice is the same text (the cache sees the same prefix)
    let n_before = bodies.len();
    api.say("short one", Some(t)).await;
    assert_eq!(api.output(t, 10).await.unwrap()["text"], "hi: short one");
    let bodies = h.model.bodies.lock().unwrap().clone();
    assert_eq!(bodies.len(), n_before + 1);
    assert_eq!(bodies[n_before]["messages"][1], msgs[1], "the cut point did not move");
    assert_eq!(api.get(&format!("/threads/{t}")).await["window"]["cuts"], json!(cuts));
    // recall returns the exact stored text within the same slice
    api.say("recall 0", Some(t)).await;
    let out = api.output(t, 10).await.expect("the recall reply");
    let text = out["text"].as_str().unwrap();
    assert!(text.starts_with("[Tool Call: recall with arguments {\"indices\":[0]}]"), "{out}");
    assert!(text.ends_with("done: #0 user: first thing: the password is swordfish"), "{out}");
    let stored = api.get(&format!("/threads/{t}/messages?tail=3")).await;
    assert_eq!(stored[1]["role"], "tool", "the recall result is stored as a tool message: {stored}");
    // replay over the same history agrees that cuts happened and nothing faulted
    let hist = api.get(&format!("/threads/{t}/messages")).await;
    let rp = freefitty::window::replay(hist.as_array().unwrap(), 1500, 900, 1.0, 200, 0, 0, 1);
    assert!(rp.cuts >= 1 && rp.peak_windowed < rp.peak_unwindowed, "{rp:?}");
    r.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn a_wrong_estimate_is_corrected_by_the_node_and_the_turn_runs() {
    let h = harness().await;
    let r = freefitty::start(h.cfg.clone()).await.unwrap();
    let api = Api::new(&r);
    let t = "thread-wall";
    for i in 0..4 {
        api.say(&format!("hello {i}"), Some(t)).await;
        api.output(t, 10).await.expect("a reply");
    }
    let before = h.model.calls.load(Ordering::Relaxed);
    let sent = api.say("wall", Some(t)).await;
    let out = api.output(t, 10).await.expect("the reply after the cut");
    assert_eq!(out["text"], "hi: wall", "{out}");
    assert_eq!(out["metadata"]["reply_to"], sent["id"]);
    assert_eq!(h.model.calls.load(Ordering::Relaxed), before + 2, "the refused request and the one after the cut");
    let th = api.get(&format!("/threads/{t}")).await;
    assert!(th["closed"].is_null(), "{th}");
    assert_eq!(th["window"]["cuts"], json!(1));
    assert_eq!(th["prompt_tokens"], json!(100), "the node's real count from the successful turn replaces the overflow's");
    assert!(api.get("/dlq").await.as_array().unwrap().is_empty());
    r.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn a_tools_image_reaches_the_model() {
    let h = harness().await;
    let r = freefitty::start(h.cfg.clone()).await.unwrap();
    let api = Api::new(&r);
    let sent = api.say("echo picture", None).await;
    let t = sent["thread"].as_str().unwrap().to_owned();
    // the first output is the turn that calls the tool; the answer to the result comes after it
    let mut text = String::new();
    for _ in 0..3 {
        let out = api.output(&t, 15).await.expect("a reply");
        text = out["text"].as_str().unwrap().to_owned();
        if text.starts_with("saw") { break; }
    }
    assert!(text.starts_with("saw 1 image(s) data:image/png;base64,iVBORw0KGgo= after tool: echo: picture"), "{text}");
    assert!(text.contains("[1 image(s) follow]"), "the tool message says images follow: {text}");
    assert!(text.contains("not inline data was left out"), "the remote URL is named, not fetched: {text}");
    r.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn old_results_page_out_in_batches_and_recall_pins_them() {
    let mut h = harness().await;
    h.cfg.page_results_after_turns = 1;
    h.cfg.page_results_min_bytes = 100;
    h.cfg.page_results_batch = 2;
    let r = freefitty::start(h.cfg.clone()).await.unwrap();
    let api = Api::new(&r);
    let t = "thread-page";
    let big = "x".repeat(300);
    for i in 0..3 {
        api.say(&format!("echo {i}-{big}"), Some(t)).await;
        api.output(t, 10).await.expect("the call line");
        let out = api.output(t, 10).await.expect("the reply after the tool");
        assert!(out["text"].as_str().unwrap().starts_with("done: echo: "), "{out}");
    }
    // an error result, which must never page
    api.say("bogus", Some(t)).await;
    api.output(t, 10).await.expect("the unknown-tool reply");
    // the paging mark moves at the start of a slice, in batches of 2: this plain turn shows the handles
    api.say("hello", Some(t)).await;
    assert_eq!(api.output(t, 10).await.unwrap()["text"], "hi: hello");
    let th = api.get(&format!("/threads/{t}")).await;
    assert!(th["paging"]["before"].as_u64().unwrap() > 0, "{th}");
    let live = api.get(&format!("/threads/{t}/messages?view=live")).await;
    let live = live.as_array().unwrap();
    let handles: Vec<&Value> = live.iter().filter(|m| m["content"].as_str().is_some_and(|c| c.starts_with("[Paged out: echo("))).collect();
    assert!(handles.len() >= 2, "{} handles in {}", handles.len(), serde_json::to_string_pretty(live).unwrap());
    assert!(handles[0]["content"].as_str().unwrap().contains("answered with 3"), "{}", handles[0]);
    let errors: Vec<&Value> = live.iter().filter(|m| m["role"] == "tool" && m["content"].as_str().is_some_and(|c| c.starts_with("Error: there is no tool"))).collect();
    assert_eq!(errors.len(), 1, "the error result stays in full");
    // the full texts are still stored
    let stored = api.get(&format!("/threads/{t}/messages")).await;
    assert!(stored.as_array().unwrap().iter().filter(|m| m["role"] == "tool" && m["content"].as_str().is_some_and(|c| c.starts_with("echo: 0-"))).count() == 1);
    // recall the first result: it comes back exactly and is pinned, so the live view shows it in full again
    let idx = stored.as_array().unwrap().iter().position(|m| m["role"] == "tool" && m["content"].as_str().is_some_and(|c| c.starts_with("echo: 0-"))).unwrap();
    api.say(&format!("recall {idx}"), Some(t)).await;
    let out = api.output(t, 10).await.expect("the recall reply");
    assert!(out["text"].as_str().unwrap().contains(&format!("#{idx} tool (result of ")), "{out}");
    assert!(out["text"].as_str().unwrap().contains("echo: 0-xxxx"), "{out}");
    let th = api.get(&format!("/threads/{t}")).await;
    assert_eq!(th["paging"]["pinned"], json!(1), "{th}");
    let live = api.get(&format!("/threads/{t}/messages?view=live")).await;
    assert!(live.as_array().unwrap().iter().any(|m| m["content"].as_str().is_some_and(|c| c.starts_with("echo: 0-"))), "the pinned result is back in full");
    // the model saw handles on the last request
    let bodies = h.model.bodies.lock().unwrap().clone();
    let last = bodies.last().unwrap()["messages"].as_array().unwrap().clone();
    assert!(last.iter().any(|m| m["content"].as_str().is_some_and(|c| c.starts_with("[Paged out:"))));
    r.shutdown().await;
}

#[tokio::test]
async fn a_sleeper_is_parked_and_its_state_restored_on_waking() {
    let mut h = harness().await;
    let _ = tracing_subscriber::fmt().with_test_writer().try_init();
    // any sleep counts as long here; the scripted one is 1 s
    h.cfg.park_after_s = 0;
    *h.model.kv.lock().unwrap() = Some(vec![7u8; 3 << 19]);
    let parked = h.cfg.data_dir.join("parked");
    let r = freefitty::start(h.cfg.clone()).await.unwrap();
    let api = Api::new(&r);
    let t = api.say("sleep please", None).await["thread"].as_str().unwrap().to_owned();
    let out = api.output(&t, 10).await.expect("the tool call line");
    assert!(out["text"].as_str().unwrap().contains("[Tool Call: sleep"), "{out}");
    let out = api.output(&t, 10).await.expect("the reply after waking");
    assert_eq!(out["text"], "done: Slept for 1 seconds");
    let imports = h.model.imports.lock().unwrap().clone();
    assert_eq!(imports.len(), 1, "the parked state went back into the node once");
    assert_eq!(imports[0], vec![7u8; 3 << 19], "byte for byte what the node exported");
    // the turn that ran made the copy stale and removed it; the finished turn parks again after park_after_s
    // (0 here), and what lands is a whole copy, never a half-written one
    let c = api.get("/health").await;
    assert_eq!(c["router"]["parked"].as_u64().unwrap() >= 1, true, "{c}");
    assert_eq!(c["router"]["restored"], 1, "{c}");
    tokio::time::sleep(Duration::from_millis(500)).await;
    for e in std::fs::read_dir(&parked).unwrap().flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        assert!(name.ends_with(".kv") && e.metadata().unwrap().len() == 3 << 19, "{name}: only whole copies stay");
    }

    // a node with nothing cached: nothing is parked, and nothing is restored
    *h.model.kv.lock().unwrap() = None;
    for e in std::fs::read_dir(&parked).unwrap().flatten() { std::fs::remove_file(e.path()).unwrap(); }
    let t2 = api.say("sleep please", None).await["thread"].as_str().unwrap().to_owned();
    api.output(&t2, 10).await.expect("the tool call line");
    api.output(&t2, 10).await.expect("the reply after waking");
    assert_eq!(h.model.imports.lock().unwrap().len(), 1);
    assert!(!parked.join(format!("{t2}.kv")).exists());
    // a copy that lands while a turn runs is stale once the turn commits: the commit removes it
    std::fs::write(parked.join(format!("{t2}.kv")), b"stale").unwrap();
    api.say("hello", Some(&t2)).await;
    assert_eq!(api.output(&t2, 10).await.expect("a reply")["text"], "hi: hello");
    assert!(!parked.join(format!("{t2}.kv")).exists());

    // deleting a thread removes its parked state
    std::fs::write(parked.join(format!("{t}.kv")), b"x").unwrap();
    let s = reqwest::Client::new().delete(format!("{}/threads/{t}", api.base)).bearer_auth(&api.token).send().await.unwrap().status();
    assert!(s.is_success(), "{s}");
    assert!(!parked.join(format!("{t}.kv")).exists());
    r.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn an_expert_is_its_frozen_state_and_threads_are_born_into_it() {
    let mut h = harness().await;
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("railhand.toml"), "description = \"knows the rail\"\nack = \"Read.\"\n[[reads]]\ntext = \"B is .51\"\n").unwrap();
    h.cfg.experts_dir = Some(dir.path().to_path_buf());
    *h.model.kv.lock().unwrap() = Some(vec![9u8; 3 << 19]);
    // the export sent right after the reading gets the system prompt's entry; the freeze waits for the real one
    h.model.short_first.store(true, Ordering::Relaxed);
    let r = freefitty::start(h.cfg.clone()).await.unwrap();
    let api = Api::new(&r);
    // frozen once a node is up: read once, its state exported to disk
    let mut frozen = Value::Null;
    for _ in 0..40 {
        frozen = api.get("/experts").await[0]["frozen_bytes"].clone();
        if !frozen.is_null() { break; }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    assert_eq!(frozen, 3 << 19, "the expert's state was frozen");
    let reads = |m: &FakeModel| m.bodies.lock().unwrap().iter().filter(|b| b["messages"].as_array().unwrap().last().unwrap()["content"] == "Say ready.").count();
    assert_eq!(reads(&h.model), 1, "read once");
    // two threads born from it: each first turn gets the frozen state put into its node first, byte for byte
    for t in ["fz-a", "fz-b"] {
        api.post("/say", json!({"text": format!("hello {t}"), "thread": t, "expert": "railhand"})).await;
        assert_eq!(api.output(t, 10).await.unwrap()["text"], format!("hi: hello {t}"));
    }
    let imports = h.model.imports.lock().unwrap().clone();
    assert_eq!(imports.len(), 2);
    assert!(imports.iter().all(|i| *i == vec![9u8; 3 << 19]));
    // a later turn of a born thread runs on its node's own cache: no import
    api.say("again", Some("fz-a")).await;
    api.output("fz-a", 10).await.unwrap();
    assert_eq!(h.model.imports.lock().unwrap().len(), 2);
    // a plain thread is born into nothing
    api.say("plain", Some("fz-plain")).await;
    api.output("fz-plain", 10).await.unwrap();
    assert_eq!(h.model.imports.lock().unwrap().len(), 2);
    r.shutdown().await;
    // across a restart the expert is still its state: not read again, and a new thread is born into it
    let r = freefitty::start(h.cfg.clone()).await.unwrap();
    let api = Api::new(&r);
    api.post("/say", json!({"text": "hello c", "thread": "fz-c", "expert": "railhand"})).await;
    assert_eq!(api.output("fz-c", 10).await.unwrap()["text"], "hi: hello c");
    tokio::time::sleep(Duration::from_millis(500)).await;
    assert_eq!(reads(&h.model), 1, "not read again after the restart");
    assert_eq!(h.model.imports.lock().unwrap().len(), 3);
    r.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn elephants_find_ask_and_commission_goldfish() {
    let mut h = harness().await;
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("railhand.toml"), "description = \"knows the rail\"\nack = \"Read.\"\n[[reads]]\ntext = \"B is .51\"\n").unwrap();
    h.cfg.experts_dir = Some(dir.path().to_path_buf());
    // parking at once and a node with state to export: a finished turn would park, so a closed goldfish shows it is not
    h.cfg.park_after_s = 0;
    *h.model.kv.lock().unwrap() = Some(vec![5u8; 1 << 10]);
    let r = freefitty::start(h.cfg.clone()).await.unwrap();
    let api = Api::new(&r);
    // finding: the catalog names each expert, its line and its goldfish's tools
    api.say("catalog", Some("el")).await;
    let out = api.output("el", 10).await.unwrap()["text"].as_str().unwrap().to_owned();
    assert!(out.contains("- railhand: knows the rail (no tools as a goldfish)"), "{out}");
    // asking: the elephant sleeps, the goldfish is born from the expert with only the plain built-ins, and its
    // answer returns as the call's result
    api.say("ask what is B", Some("el")).await;
    assert!(api.output("el", 10).await.unwrap()["text"].as_str().unwrap().contains("[Tool Call: goldfish"));
    let out = api.output("el", 10).await.unwrap()["text"].as_str().unwrap().to_owned();
    assert!(out.starts_with("done: fish(tools=") && out.ends_with("): what is B"), "{out}");
    assert!(out.contains("experts") && out.contains("sleep") && !out.contains("echo") && !out.contains("goldfish,") && !out.contains("commission"), "{out}");
    let threads = api.get("/threads").await;
    let fish: Vec<&Value> = threads.as_array().unwrap().iter().filter(|t| t["thread"].as_str().unwrap().starts_with("el.gf-railhand-")).collect();
    assert_eq!(fish.len(), 1);
    assert_eq!(fish[0]["closed"], "answered its elephant");
    assert!(!h.cfg.data_dir.join("parked").join(format!("{}.kv", fish[0]["thread"].as_str().unwrap())).exists(), "a closed goldfish is not parked");
    let fish_body = h.model.bodies.lock().unwrap().iter().find(|b| b["messages"].as_array().unwrap().last().unwrap()["content"].as_str().is_some_and(|c| c.ends_with("what is B") && c.starts_with("You are a goldfish"))).cloned().unwrap();
    assert!(fish_body["messages"][1]["content"].as_str().unwrap().contains("B is .51"), "born from the expert");
    // a goldfish cannot ask goldfish, even when it tries: it is told there is no such tool, and says so
    api.say("ask try goldfish", Some("el")).await;
    api.output("el", 10).await.unwrap();
    let out = api.output("el", 10).await.unwrap()["text"].as_str().unwrap().to_owned();
    assert!(out.contains("there is no tool named \"goldfish\""), "{out}");
    // an unknown expert is answered in the same turn, with the catalog
    api.say("askbad", Some("el")).await;
    let out = api.output("el", 10).await.unwrap()["text"].as_str().unwrap().to_owned();
    assert!(out.contains("no expert named \"nobody\"") && out.contains("- railhand:"), "{out}");
    // commissioning: registered at once, frozen, the definition written with its provenance; a goldfish asked
    // for it was granted `goldfish` by the elephant, which no goldfish keeps
    api.say("commission rail2", Some("el")).await;
    api.output("el", 10).await.unwrap();
    let out = api.output("el", 10).await.unwrap()["text"].as_str().unwrap().to_owned();
    assert!(out.starts_with("done: Expert rail2@") && out.contains("is ready"), "{out}");
    let def = std::fs::read_to_string(dir.path().join("rail2.toml")).unwrap();
    assert!(def.starts_with("# commissioned by thread el") && def.contains("C is .52") && !def.contains("goldfish"), "{def}");
    api.say("askr2 what is C", Some("el")).await;
    api.output("el", 10).await.unwrap();
    let out = api.output("el", 10).await.unwrap()["text"].as_str().unwrap().to_owned();
    assert!(out.contains("echo") && !out.contains("goldfish,") && out.ends_with("): what is C"), "{out}");
    assert!(h.model.imports.lock().unwrap().len() >= 1, "the goldfish was born into the frozen expert");
    r.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn a_steered_expert_steers_every_request_of_its_threads() {
    let mut h = harness().await;
    let dir = tempfile::tempdir().unwrap();
    let def = "description = \"a villain\"\nack = \"Read.\"\n[[reads]]\ntext = \"You scheme.\"\n";
    std::fs::write(dir.path().join("villain.toml"), format!("{def}[[steer]]\nvector = \"villain24\"\nscale = 6.0\nlayer = 24\n")).unwrap();
    std::fs::write(dir.path().join("plainvillain.toml"), def).unwrap();
    h.cfg.experts_dir = Some(dir.path().to_path_buf());
    *h.model.kv.lock().unwrap() = Some(vec![3u8; 1 << 10]);
    let r = freefitty::start(h.cfg.clone()).await.unwrap();
    let api = Api::new(&r);
    let list = api.get("/experts").await;
    let of = |n: &str| list.as_array().unwrap().iter().find(|e| e["name"] == n).cloned().unwrap();
    assert_eq!(of("villain")["steer"], json!([{"vector": "villain24", "scale": 6.0, "layer": 24}]));
    // the same reading unsteered is another version: the nodes cache steered state under its own key
    assert_ne!(of("villain")["key"].as_str().unwrap().split('@').nth(1), of("plainvillain")["key"].as_str().unwrap().split('@').nth(1));
    let steer = json!([{"vector": "villain24", "scale": 6.0, "layer": 24}]);
    // the freeze read it steered
    let mut frozen = false;
    for _ in 0..40 {
        frozen = h.model.bodies.lock().unwrap().iter().any(|b| b["steer"] == steer && b["messages"].as_array().unwrap().last().unwrap()["content"] == "Say ready.");
        if frozen { break; }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    assert!(frozen, "the expert was read under its steering");
    // every turn of a thread born from it carries it; a plain thread's carry none
    api.post("/say", json!({"text": "hello v", "thread": "sv", "expert": "villain"})).await;
    api.output("sv", 10).await.unwrap();
    api.say("again v", Some("sv")).await;
    api.output("sv", 10).await.unwrap();
    api.say("plain p", Some("sp")).await;
    api.output("sp", 10).await.unwrap();
    let bodies = h.model.bodies.lock().unwrap().clone();
    let by = |t: &str| bodies.iter().find(|b| b["messages"].as_array().unwrap().last().unwrap()["content"] == t).cloned().unwrap();
    assert_eq!(by("hello v")["steer"], steer);
    assert_eq!(by("again v")["steer"], steer);
    assert!(by("plain p").get("steer").is_none());
    // the catalog says so
    api.say("catalog", Some("sc")).await;
    let out = api.output("sc", 10).await.unwrap()["text"].as_str().unwrap().to_owned();
    assert!(out.contains("- villain: a villain (no tools as a goldfish; steered: villain24 x6.0)"), "{out}");
    r.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn a_node_that_holds_the_state_is_not_sent_it_again() {
    let mut h = harness().await;
    h.cfg.park_after_s = 0;
    *h.model.kv.lock().unwrap() = Some(kv_file(100));
    let r = freefitty::start(h.cfg.clone()).await.unwrap();
    let api = Api::new(&r);
    // the node says it holds all 100 tokens the parked copy covers: nothing is sent
    h.model.probe_tokens.store(100, Ordering::Relaxed);
    let t = api.say("sleep please", None).await["thread"].as_str().unwrap().to_owned();
    api.output(&t, 10).await.expect("the tool call line");
    api.output(&t, 10).await.expect("the reply after waking");
    assert_eq!(h.model.imports.lock().unwrap().len(), 0);
    assert_eq!(api.get("/health").await["router"]["restore_skipped"], 1);
    // it holds less: the copies go (the one parked after the last turn, then the one parked for this sleep)
    h.model.probe_tokens.store(50, Ordering::Relaxed);
    api.say("sleep again", Some(&t)).await;
    api.output(&t, 10).await.expect("the tool call line");
    api.output(&t, 10).await.expect("the reply after waking");
    assert_eq!(h.model.imports.lock().unwrap().len(), 2);
    r.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn a_commissioned_persona_is_checked_before_it_exists() {
    let mut h = harness().await;
    let dir = tempfile::tempdir().unwrap();
    h.cfg.experts_dir = Some(dir.path().to_path_buf());
    let r = freefitty::start(h.cfg.clone()).await.unwrap();
    let api = Api::new(&r);
    // steered commissions are off until measured; the fake elephant asks with steer
    api.say("commission persona1", Some("ce")).await;
    let out = api.output("ce", 10).await.unwrap()["text"].as_str().unwrap().to_owned();
    assert!(out.contains("cannot be steered yet"), "{out}");
    assert!(!dir.path().join("persona1.toml").exists());
    r.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn a_node_refusing_a_threads_steering_closes_it_at_once() {
    let mut h = harness().await;
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("villain.toml"), "description = \"v\"\nack = \"Read.\"\n[[reads]]\ntext = \"x\"\n[[steer]]\nvector = \"villain24\"\nscale = 6.0\n").unwrap();
    h.cfg.experts_dir = Some(dir.path().to_path_buf());
    h.model.refuse_steer.store(true, Ordering::Relaxed);
    let r = freefitty::start(h.cfg.clone()).await.unwrap();
    let api = Api::new(&r);
    api.post("/say", json!({"text": "hello", "thread": "rv", "expert": "villain"})).await;
    let out = api.output("rv", 10).await.unwrap();
    assert!(out["text"].as_str().unwrap().contains("refuses this thread's steering"), "{out}");
    assert!(api.get("/threads/rv").await["closed"].as_str().unwrap().contains("steering"));
    assert!(api.get("/dlq").await.as_array().unwrap().iter().all(|e| e["thread"] != "rv" || e["attempts"].as_u64().unwrap() <= 1), "not retried");
    r.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn a_persona_needs_its_vector_on_every_node() {
    let mut h = harness().await;
    let dir = tempfile::tempdir().unwrap();
    h.cfg.experts_dir = Some(dir.path().to_path_buf());
    h.cfg.commission_steer = true;
    *h.model.steer_list.lock().unwrap() = Some(json!({"vectors": [{"name": "villain24", "layer": 24}]}));
    // a second node that holds no vectors: villain24 is not on every node
    let other = Arc::new(FakeModel { fail: AtomicBool::new(false), busy: AtomicUsize::new(2), calls: AtomicUsize::new(0), cut_notes: AtomicBool::new(false), bodies: std::sync::Mutex::new(Vec::new()), kv: std::sync::Mutex::new(None), imports: std::sync::Mutex::new(Vec::new()), short_first: AtomicBool::new(false), probe_tokens: std::sync::atomic::AtomicU64::new(u64::MAX), steer_list: std::sync::Mutex::new(Some(json!({"vectors": []}))), refuse_steer: AtomicBool::new(false), steer_fail: AtomicBool::new(false) });
    let other_url = serve(Router::new().route("/health", get(health)).route("/steer", get(steer_list)).with_state(other.clone())).await;
    h.cfg.nodes.push(other_url);
    let r = freefitty::start(h.cfg.clone()).await.unwrap();
    let api = Api::new(&r);
    api.say("commission persona2", Some("cv")).await;
    let out = api.output("cv", 10).await.unwrap()["text"].as_str().unwrap().to_owned();
    assert!(out.contains("not every node holds a steering vector named \"villain24\""), "{out}");
    assert!(!dir.path().join("persona2.toml").exists());
    // a node in trouble is named, not taken for one without the vector
    other.steer_fail.store(true, Ordering::Relaxed);
    api.say("commission persona4", Some("cv")).await;
    let out = api.output("cv", 15).await.unwrap()["text"].as_str().unwrap().to_owned();
    assert!(out.contains("did not answer GET /steer"), "{out}");
    other.steer_fail.store(false, Ordering::Relaxed);
    // the other node gains it: the commission goes through
    *other.steer_list.lock().unwrap() = Some(json!({"vectors": [{"name": "villain24", "layer": 24}]}));
    api.say("commission persona3", Some("cv")).await;
    assert!(api.output("cv", 10).await.unwrap()["text"].as_str().unwrap().contains("[Tool Call: commission_expert"));
    let out = api.output("cv", 10).await.unwrap()["text"].as_str().unwrap().to_owned();
    assert!(out.contains("persona3@") && dir.path().join("persona3.toml").exists(), "{out}");
    r.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn a_slice_that_panics_is_a_counted_failure_and_dead_letters() {
    let mut h = harness().await;
    h.cfg.test_panic_on = "poison".into();
    let r = freefitty::start(h.cfg.clone()).await.unwrap();
    let api = Api::new(&r);
    api.say("poison", Some("pz")).await;
    // three attempts (retry_after_s 1), then the dead-letter queue; the daemon and other threads carry on
    let mut dead = Value::Null;
    for _ in 0..80 {
        dead = api.get("/dlq").await;
        if dead.as_array().is_some_and(|d| d.iter().any(|e| e["thread"] == "pz")) { break; }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    let e = dead.as_array().unwrap().iter().find(|e| e["thread"] == "pz").cloned().expect("dead-lettered");
    assert_eq!(e["attempts"], 3, "{e}");
    api.say("hello", Some("ok1")).await;
    assert_eq!(api.output("ok1", 10).await.unwrap()["text"], "hi: hello");
    r.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn metrics_health_and_a_backup_on_demand() {
    let h = harness().await;
    let r = freefitty::start(h.cfg.clone()).await.unwrap();
    let api = Api::new(&r);
    api.say("hello", Some("m1")).await;
    api.output("m1", 10).await.unwrap();
    assert!(api.get("/health").await["build"].as_str().unwrap().starts_with(env!("CARGO_PKG_VERSION")));
    let m = reqwest::get(format!("{}/metrics", api.base)).await.unwrap().text().await.unwrap();
    assert!(m.contains("freefitty_build_info{version="), "{m}");
    assert!(m.contains("freefitty_slices_total{outcome=\"done\"} 1"), "{m}");
    assert!(m.lines().any(|l| l.starts_with("freefitty_node_cached_tokens_total{node=") && l.ends_with(" 90")), "the fake node's 90 cached tokens: {m}");
    assert!(m.contains("freefitty_turn_seconds_count 1") && m.contains("freefitty_queue_queued"), "{m}");
    assert!(m.contains("freefitty_threads_over_three_quarter_budget 0") && m.contains("freefitty_threads_closed_for_context 0")
            && m.contains("freefitty_thread_prompt_tokens_max 100"), "context pressure gauges: {m}");
    let b = api.post("/admin/backup", json!({})).await;
    let p = std::path::PathBuf::from(b["backup"].as_str().unwrap_or_else(|| panic!("{b}")));
    assert!(p.exists() && b["bytes"].as_u64().unwrap() > 0, "{b}");
    assert_eq!(freefitty::store::Store::schema_of(&p).unwrap(), freefitty::store::SCHEMA);
    r.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn a_stop_does_not_wait_for_turns_still_queued_for_a_slot() {
    let mut h = harness().await;
    h.cfg.slice_timeout_s = 60;
    h.model.busy.store(2, Ordering::Relaxed);
    let r = freefitty::start(h.cfg.clone()).await.unwrap();
    let api = Api::new(&r);
    api.say("hello", Some("st1")).await;
    assert!(api.output("st1", 2).await.is_none(), "the node is full: the turn waits for a slot");
    let t = std::time::Instant::now();
    r.shutdown().await;
    assert!(t.elapsed() < Duration::from_secs(10), "the stop took {:?}: a waiting turn held it", t.elapsed());
    assert_eq!(h.model.calls.load(Ordering::Relaxed), 0, "nothing reached the node");
    // after a restart with a free slot, the turn runs, and nothing counted as a failed attempt
    h.model.busy.store(0, Ordering::Relaxed);
    let r = freefitty::start(h.cfg.clone()).await.unwrap();
    let api = Api::new(&r);
    assert_eq!(api.output("st1", 10).await.expect("the re-queued turn runs")["text"], "hi: hello");
    assert!(api.get("/dlq").await.as_array().unwrap().is_empty());
    let m = reqwest::get(format!("{}/metrics", api.base)).await.unwrap().text().await.unwrap();
    assert!(!m.contains("outcome=\"transient\""), "a stop is not a failure: {m}");
    r.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn a_reply_cut_off_mid_call_goes_on_in_the_same_slice() {
    let h = harness().await;
    let r = freefitty::start(h.cfg.clone()).await.unwrap();
    let api = Api::new(&r);
    // cut off twice, told so each time, then written in parts: one slice, three model calls
    api.say("cutoff", Some("thread-cut")).await;
    let out = api.output("thread-cut", 10).await.expect("a reply");
    let text = out["text"].as_str().unwrap();
    assert!(text.contains("writing the file (0)") && text.ends_with("wrote it in parts after 2 notes"), "{out}");
    assert_eq!(h.model.calls.load(Ordering::Relaxed), 3);
    // a model that never fits gets two notes, then the slice ends on its text as before
    api.say("cutoff forever", Some("thread-cut2")).await;
    let out = api.output("thread-cut2", 10).await.expect("a reply");
    assert!(out["text"].as_str().unwrap().ends_with("writing the file (2)"), "{out}");
    assert_eq!(h.model.calls.load(Ordering::Relaxed), 6);
    r.shutdown().await;
}
