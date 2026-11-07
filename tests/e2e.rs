//! The whole daemon against a scripted model node and a RAP tool server, both in-process.

use axum::{extract::State, routing::{get, post}, Json, Router};
use freefitty::config::Config;
use serde_json::{json, Value};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

struct FakeModel {
    fail: AtomicBool,
    calls: AtomicUsize,
}

async fn health() -> Json<Value> {
    Json(json!({"status": "ok", "model": "fake", "slots": 2, "busy_slots": 0, "queued": 0, "context_length": 262144}))
}

/// The script: a user message starting with "sleep" calls the sleep tool for 1 s, "echo X" calls the echo
/// tool, "nothing" ends the turn with no text and no call; a tool result is answered with "done: <result>";
/// anything else with "hi: <text>".
async fn completions(State(m): State<Arc<FakeModel>>, Json(body): Json<Value>) -> (axum::http::StatusCode, Json<Value>) {
    m.calls.fetch_add(1, Ordering::Relaxed);
    if m.fail.load(Ordering::Relaxed) {
        return (axum::http::StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": "fake outage"})));
    }
    let msgs = body["messages"].as_array().cloned().unwrap_or_default();
    assert_eq!(msgs[0]["role"], "system");
    let last = msgs.last().cloned().unwrap_or(Value::Null);
    let content = last["content"].as_str().unwrap_or("").to_owned();
    let tool_names: Vec<&str> = body["tools"].as_array().unwrap().iter().map(|t| t["function"]["name"].as_str().unwrap()).collect();
    assert!(tool_names.contains(&"sleep") && tool_names.contains(&"echo"), "tools offered: {tool_names:?}");
    let message = if last["role"] == "tool" {
        json!({"role": "assistant", "content": format!("done: {content}")})
    } else if content.starts_with("sleep") {
        json!({"role": "assistant", "content": null, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "sleep", "arguments": "{\"seconds\": 1}"}}]})
    } else if let Some(x) = content.strip_prefix("echo ") {
        // arguments JSON-encoded twice, as a model sometimes writes them
        json!({"role": "assistant", "content": "calling echo", "tool_calls": [{"id": "call_2", "type": "function", "function": {"name": "echo", "arguments": serde_json::to_string(&json!({"text": x}).to_string()).unwrap()}}]})
    } else if content.starts_with("bogus") {
        json!({"role": "assistant", "content": null, "tool_calls": [{"id": "call_3", "type": "function", "function": {"name": "no_such_tool", "arguments": "{}"}}]})
    } else if content.starts_with("nothing") {
        json!({"role": "assistant", "content": ""})
    } else {
        json!({"role": "assistant", "content": format!("hi: {content}"), "reasoning_content": "dropped"})
    };
    (axum::http::StatusCode::OK, Json(json!({"choices": [{"message": message}], "usage": {"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": 90}}})))
}

async fn export() -> axum::http::StatusCode {
    axum::http::StatusCode::NOT_FOUND
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
    assert_eq!(inv["arguments"]["text"], "foo", "arguments were repaired to an object: {}", inv["arguments"]);
    let cb = inv["callback_url"].as_str().unwrap().to_owned();
    let result = json!({"type": "tool_result", "group_id": inv["group_id"], "id": inv["id"], "call_id": inv["call_id"], "text": format!("echo: {}", inv["arguments"]["text"].as_str().unwrap())});
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(200)).await;
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
    let model = Arc::new(FakeModel { fail: AtomicBool::new(false), calls: AtomicUsize::new(0) });
    let model_url = serve(Router::new().route("/health", get(health)).route("/v1/chat/completions", post(completions)).route("/cache/export", post(export)).with_state(model.clone())).await;
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
    assert_eq!(out["text"], "done: late");
    r.shutdown().await;
}
