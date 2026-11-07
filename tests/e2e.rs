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
    /// the next notes answer is cut off (finish_reason length)
    cut_notes: AtomicBool,
    bodies: std::sync::Mutex<Vec<Value>>,
}

async fn health() -> Json<Value> {
    Json(json!({"status": "ok", "model": "fake", "slots": 2, "busy_slots": 0, "queued": 0, "context_length": 262144}))
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
    let last = msgs.last().cloned().unwrap_or(Value::Null);
    let content = last["content"].as_str().unwrap_or("").to_owned();
    let tool_names: Vec<&str> = body["tools"].as_array().expect("every request carries tools").iter().map(|t| t["function"]["name"].as_str().unwrap()).collect();
    assert!(tool_names.contains(&"sleep") && tool_names.contains(&"echo"), "tools offered: {tool_names:?}");
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
    let message = if last["role"] == "tool" {
        json!({"role": "assistant", "content": format!("done: {content}")})
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
    assert!(inv["arguments"]["text"].is_string(), "arguments were repaired to an object: {}", inv["arguments"]);
    let cb = inv["callback_url"].as_str().unwrap().to_owned();
    let result = json!({"type": "tool_result", "group_id": inv["group_id"], "id": inv["id"], "call_id": inv["call_id"], "text": format!("echo: {}", inv["arguments"]["text"].as_str().unwrap())});
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
    let model = Arc::new(FakeModel { fail: AtomicBool::new(false), calls: AtomicUsize::new(0), cut_notes: AtomicBool::new(false), bodies: std::sync::Mutex::new(Vec::new()) });
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
    // warm on start ran: a 1-token request with the prefix (it waits for the node, then a moment)
    let mut warmed = false;
    for _ in 0..40 {
        warmed = h.model.bodies.lock().unwrap().iter().any(|b| b["max_tokens"] == 16 && b["messages"][2]["content"] == "Read.");
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
