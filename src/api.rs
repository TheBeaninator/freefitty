//! The HTTP surface, the same one pod-ff and pod-hands speak to freefinity:
//!
//!   POST /say                 {text, thread?, metadata?} -> {thread, id}       operator
//!   GET  /output?wait&thread  the oldest uncollected reply, removed once sent   operator
//!   GET  /threads, /threads/{id}, /threads/{id}/messages; DELETE /threads/{id} operator
//!   GET  /dlq; POST /dlq/redrive, /dlq/drop  {thread?}                         operator
//!   POST /rap/callback, /rap/callback/{token}  a tool's result                 callback token
//!   GET  /health                                                                open
//!
//! Two tokens, least privilege each: the operator drives the model and reads every reply; a tool server's
//! token can only deliver results. Bodies are capped; nothing user-supplied is logged.

use crate::config::{self, Config};
use crate::engine::{self, Engine};
use crate::experts::{self, Experts};
use crate::queue::Queue;
use crate::rap::{self, ToolBox};
use crate::router::Router;
use crate::store::{now_ms, Event, EventKind, Store};
use axum::{
    body::Bytes,
    extract::{Path, Query, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router as AxumRouter,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{Mutex, Notify};

pub struct App {
    pub cfg: Config,
    pub store: Arc<Store>,
    pub queue: Arc<Queue>,
    pub router: Arc<Router>,
    pub tools: Arc<ToolBox>,
    pub replies: Arc<Notify>,
    pub engine: Arc<Engine>,
    pub experts: Arc<Experts>,
    pub started_ms: u64,
    output_lock: Mutex<()>,
}

impl App {
    #[allow(clippy::too_many_arguments)]
    pub fn new(cfg: Config, store: Arc<Store>, queue: Arc<Queue>, router: Arc<Router>, tools: Arc<ToolBox>, replies: Arc<Notify>, engine: Arc<Engine>, experts: Arc<Experts>) -> Arc<App> {
        Arc::new(App { cfg, store, queue, router, tools, replies, engine, experts, started_ms: now_ms(), output_lock: Mutex::new(()) })
    }
}

type Shared = State<Arc<App>>;

/// One message's text. Threads read files through tools, but a round brief should fit in one message.
const MAX_TEXT: usize = 256 << 10;

pub fn routes(app: Arc<App>) -> AxumRouter {
    AxumRouter::new()
        .route("/health", get(health))
        .route("/metrics", get(metrics))
        .route("/admin/backup", post(admin_backup))
        .route("/say", post(say))
        .route("/output", get(output))
        .route("/threads", get(threads))
        .route("/threads/{id}", get(thread).delete(delete_thread))
        .route("/threads/{id}/messages", get(messages).put(import_messages))
        .route("/threads/{id}/compact", post(compact))
        .route("/experts", get(experts_list))
        .route("/experts/reload", post(experts_reload))
        .route("/experts/{name}/warm", post(experts_warm))
        .route("/experts/from-thread/{id}", post(experts_from_thread))
        .route("/dlq", get(dlq))
        .route("/dlq/redrive", post(redrive))
        .route("/dlq/drop", post(drop_dlq))
        .route("/rap/callback", post(callback))
        .route("/rap/callback/{token}", post(callback_path))
        .layer(axum::extract::DefaultBodyLimit::max(app.cfg.max_body_bytes))
        .with_state(app)
}

fn err(status: StatusCode, msg: &str) -> Response {
    (status, Json(json!({"error": msg}))).into_response()
}

fn bearer(headers: &HeaderMap) -> Option<&str> {
    headers.get("authorization")?.to_str().ok()?.strip_prefix("Bearer ").map(str::trim)
}

fn same(a: &str, b: &str) -> bool {
    // constant time over equal lengths: a token is not guessed byte by byte
    a.len() == b.len() && a.bytes().zip(b.bytes()).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

fn operator(app: &App, headers: &HeaderMap) -> Result<(), Box<Response>> {
    match bearer(headers) {
        Some(t) if same(t, &app.cfg.operator_token) => Ok(()),
        _ => Err(Box::new(err(StatusCode::UNAUTHORIZED, "operator token required"))),
    }
}

fn thread_id_ok(t: &str) -> bool {
    !t.is_empty() && t.len() <= 128 && t.bytes().all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
}

async fn health(State(app): Shared) -> Json<Value> {
    Json(json!({
        "status": "ok",
        "build": crate::BUILD,
        "uptime_s": now_ms().saturating_sub(app.started_ms) / 1000,
        "model": app.cfg.model,
        "nodes": app.router.view().await,
        "router": app.router.counts(),
        "queue": app.queue.stats(),
        "store": app.store.counts().unwrap_or(Value::Null),
        "tools": app.tools.current().await.defs.len(),
    }))
}

/// A store snapshot now, into <data_dir>/backups (newest backup_keep kept).
async fn admin_backup(State(app): Shared, headers: HeaderMap) -> Response {
    if let Err(r) = operator(&app, &headers) { return *r; }
    let (store, dir, keep) = (app.store.clone(), app.cfg.data_dir.join("backups"), app.cfg.backup_keep.max(1));
    match tokio::task::spawn_blocking(move || crate::backup_now(&store, &dir, keep)).await {
        Ok(Ok((p, n))) => Json(json!({"backup": p, "bytes": n})).into_response(),
        Ok(Err(e)) => err(StatusCode::INTERNAL_SERVER_ERROR, &format!("{e:#}")),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
}

/// Prometheus text: the engine's counters and histograms, then the router, queue and store numbers read now.
async fn metrics(State(app): Shared) -> Response {
    let mut counters = Vec::new();
    if let Ok(Value::Object(m)) = serde_json::to_value(app.router.counts()) {
        for (k, v) in m { if let Some(n) = v.as_f64() { counters.push((format!("freefitty_router_{k}_total"), n)); } }
    }
    let mut gauges = vec![("freefitty_uptime_seconds".to_owned(), (now_ms().saturating_sub(app.started_ms) / 1000) as f64),
                          ("freefitty_tools".to_owned(), app.tools.current().await.defs.len() as f64),
                          ("freefitty_nodes_up".to_owned(), app.router.up_nodes().await.len() as f64)];
    for (prefix, v) in [("queue", app.queue.stats()), ("store", app.store.counts().unwrap_or(Value::Null))] {
        if let Value::Object(m) = v {
            for (k, v) in m { if let Some(n) = v.as_f64() { gauges.push((format!("freefitty_{prefix}_{k}"), n)); } }
        }
    }
    if let Some(p) = &app.engine.parking { gauges.push(("freefitty_parked_bytes".to_owned(), p.bytes() as f64)); }
    // context pressure (dad 2026-11-08: watch whether threads run out of room before a bigger context per slot is
    // worth it): the view budget, the largest open thread by the node's own count, how many open threads are past
    // half and three quarters of the budget, the window cuts so far, and threads closed because the context was full
    let budget = if app.cfg.window_tokens > 0 { app.cfg.window_tokens } else {
        app.router.context_length().await.saturating_sub(app.cfg.max_output_tokens + 8192)
    };
    let threads = app.store.threads().unwrap_or_default();
    // Pressure counts threads with a turn in the last 6 h: finished work often leaves its thread open, and a
    // day-old giant says nothing about whether today's work fits.
    let since = now_ms().saturating_sub(6 * 3_600_000);
    let open: Vec<_> = threads.iter().filter(|(_, m)| m.closed.is_none() && m.updated_ms >= since).collect();
    let over = |f: f64| open.iter().filter(|(_, m)| budget > 0 && m.prompt_tokens as f64 > f * budget as f64).count() as f64;
    gauges.push(("freefitty_context_budget_tokens".to_owned(), budget as f64));
    gauges.push(("freefitty_thread_prompt_tokens_max".to_owned(), open.iter().map(|(_, m)| m.prompt_tokens).max().unwrap_or(0) as f64));
    gauges.push(("freefitty_threads_over_half_budget".to_owned(), over(0.5)));
    gauges.push(("freefitty_threads_over_three_quarter_budget".to_owned(), over(0.75)));
    gauges.push(("freefitty_window_cuts".to_owned(), threads.iter().map(|(_, m)| m.window_cuts).sum::<u64>() as f64));
    gauges.push(("freefitty_threads_closed_for_context".to_owned(),
                 threads.iter().filter(|(_, m)| m.closed.as_deref().is_some_and(|c| c.contains("context"))).count() as f64));
    ([(axum::http::header::CONTENT_TYPE, "text/plain; version=0.0.4")], app.engine.metrics.render(crate::BUILD, &counters, &gauges)).into_response()
}

#[derive(Deserialize)]
struct Say {
    text: String,
    #[serde(default)]
    thread: Option<String>,
    #[serde(default)]
    metadata: Option<Value>,
    /// Start the thread from this expert's prefix (new threads only).
    #[serde(default)]
    expert: Option<String>,
}

async fn say(State(app): Shared, headers: HeaderMap, body: Bytes) -> Response {
    if let Err(r) = operator(&app, &headers) { return *r; }
    let req: Say = match serde_json::from_slice(&body) { Ok(r) => r, Err(_) => return err(StatusCode::BAD_REQUEST, "body must be JSON with text") };
    if req.text.trim().is_empty() { return err(StatusCode::BAD_REQUEST, "text is empty"); }
    if req.text.len() > MAX_TEXT { return err(StatusCode::BAD_REQUEST, "text is over 256 KiB"); }
    let thread = req.thread.unwrap_or_else(config::id);
    if !thread_id_ok(&thread) { return err(StatusCode::BAD_REQUEST, "thread must be 1-128 of [A-Za-z0-9._-]"); }
    let metadata = req.metadata.unwrap_or(Value::Null);
    if !metadata.is_null() && (!metadata.is_object() || metadata.to_string().len() > 4096) {
        return err(StatusCode::BAD_REQUEST, "metadata must be a JSON object under 4 KiB");
    }
    let existing = match app.store.thread(&thread) { Ok(t) => t, Err(e) => return err(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()) };
    let mut expert_key = None;
    if let Some(name) = &req.expert {
        let Some(key) = app.experts.key(name).await else { return err(StatusCode::NOT_FOUND, "no such expert") };
        match &existing {
            Some(t) if t.expert.as_deref().is_some_and(|k| k.starts_with(&format!("{name}@"))) => {}
            Some(_) => return err(StatusCode::CONFLICT, "the thread exists and was not started from that expert"),
            None => expert_key = Some(key),
        }
    }
    let id = config::id();
    let ev = Event { id: id.clone(), thread: thread.clone(), due_ms: 0, created_ms: 0, attempts: 0, kind: EventKind::User { text: req.text, metadata } };
    if let Some(key) = expert_key {
        if let Err(e) = app.store.new_thread_with_expert(&thread, &key, &ev) { return err(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()); }
    }
    match app.queue.push(ev) {
        Ok(_) => Json(json!({"thread": thread, "id": id})).into_response(),
        Err(e) => {
            tracing::error!("say: {e:#}");
            err(StatusCode::INTERNAL_SERVER_ERROR, "could not store the message")
        }
    }
}

#[derive(Deserialize)]
struct OutputQuery {
    #[serde(default)]
    wait: Option<u64>,
    #[serde(default)]
    thread: Option<String>,
}

async fn output(State(app): Shared, headers: HeaderMap, Query(q): Query<OutputQuery>) -> Response {
    if let Err(r) = operator(&app, &headers) { return *r; }
    let deadline = tokio::time::Instant::now() + Duration::from_secs(q.wait.unwrap_or(0).min(60));
    loop {
        let notified = app.replies.notified();
        {
            let _g = app.output_lock.lock().await;
            match app.store.replies(q.thread.as_deref()) {
                Ok(mut rs) if !rs.is_empty() => {
                    let r = rs.remove(0);
                    if let Err(e) = app.store.remove_reply(r.seq) { tracing::error!("output: {e:#}"); }
                    let mut meta = match r.metadata { Value::Object(m) => m, _ => Default::default() };
                    meta.insert("thread".into(), json!(r.thread));
                    meta.insert("reply_to".into(), json!(r.reply_to));
                    meta.insert("kind".into(), json!(r.kind));
                    return Json(json!({"text": r.text, "metadata": meta, "created_ms": r.created_ms})).into_response();
                }
                Ok(_) => {}
                Err(e) => {
                    tracing::error!("output: {e:#}");
                    return err(StatusCode::INTERNAL_SERVER_ERROR, "could not read replies");
                }
            }
        }
        let now = tokio::time::Instant::now();
        if now >= deadline { return StatusCode::NO_CONTENT.into_response(); }
        let _ = tokio::time::timeout(deadline - now, notified).await;
    }
}

async fn threads(State(app): Shared, headers: HeaderMap) -> Response {
    if let Err(r) = operator(&app, &headers) { return *r; }
    match app.store.threads() {
        Ok(ts) => Json(ts.into_iter().map(|(id, m)| json!({"thread": id, "created_ms": m.created_ms, "updated_ms": m.updated_ms, "node": m.node, "messages": m.messages, "bytes": m.bytes, "pending": m.pending.len(), "last": m.last, "closed": m.closed})).collect::<Vec<_>>()).into_response(),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
}

async fn thread(State(app): Shared, headers: HeaderMap, Path(id): Path<String>) -> Response {
    if let Err(r) = operator(&app, &headers) { return *r; }
    match app.store.thread(&id) {
        Ok(Some(m)) => {
            let live_tokens = app.engine.view(&id).map(|v| engine::tokens(&v)).unwrap_or(0);
            let compaction = m.compactions.last().map(|c| json!({"from": c.from, "at_ms": c.at_ms, "summarized": c.summarized, "focus": c.focus, "truncated": c.truncated, "tokens_before": c.tokens_before, "count": m.compactions.len()}));
            Json(json!({"thread": id, "created_ms": m.created_ms, "updated_ms": m.updated_ms, "node": m.node, "messages": m.messages, "bytes": m.bytes, "live_tokens": live_tokens, "prompt_tokens": m.prompt_tokens, "prompt_est": m.prompt_est, "window": json!({"from": m.window_from, "cuts": m.window_cuts}), "paging": json!({"before": m.paged_before, "pinned": m.pinned.len()}), "pending": m.pending, "last": m.last, "closed": m.closed, "expert": m.expert, "compaction": compaction, "metadata": m.metadata})).into_response()
        }
        Ok(None) => err(StatusCode::NOT_FOUND, "no such thread"),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
}

#[derive(Deserialize, Default)]
struct ViewQuery {
    #[serde(default)]
    view: Option<String>,
    /// Only the newest N messages of the chosen view.
    #[serde(default)]
    tail: Option<usize>,
}

/// The stored history, or with `?view=live` what the model sees (prefix, refocus bridge, kept history);
/// `?tail=N` keeps the newest N.
async fn messages(State(app): Shared, headers: HeaderMap, Path(id): Path<String>, Query(q): Query<ViewQuery>) -> Response {
    if let Err(r) = operator(&app, &headers) { return *r; }
    let r = if q.view.as_deref() == Some("live") { app.engine.view(&id) } else { app.store.messages(&id) };
    match r {
        Ok(mut ms) => {
            if let Some(n) = q.tail {
                ms.drain(..ms.len().saturating_sub(n));
            }
            Json(ms).into_response()
        }
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
}

#[derive(Deserialize)]
struct Import {
    messages: Vec<Value>,
}

/// Give a thread with no messages a history: OpenAI chat messages, oldest first (a migration from another
/// host). The thread id is the path; it may be new. 409 when the thread already has messages.
async fn import_messages(State(app): Shared, headers: HeaderMap, Path(id): Path<String>, body: Bytes) -> Response {
    if let Err(r) = operator(&app, &headers) { return *r; }
    if !thread_id_ok(&id) { return err(StatusCode::BAD_REQUEST, "thread must be 1-128 of [A-Za-z0-9._-]"); }
    let imp: Import = match serde_json::from_slice(&body) { Ok(i) => i, Err(_) => return err(StatusCode::BAD_REQUEST, "body must be JSON with messages") };
    if imp.messages.is_empty() { return err(StatusCode::BAD_REQUEST, "messages is empty"); }
    if imp.messages.iter().any(|m| !m.get("role").is_some_and(|r| r.is_string())) {
        return err(StatusCode::BAD_REQUEST, "every message needs a role");
    }
    match app.store.import_messages(&id, &imp.messages) {
        Ok(()) => Json(json!({"thread": id, "messages": imp.messages.len()})).into_response(),
        Err(e) if e.to_string().contains("already has") => err(StatusCode::CONFLICT, &e.to_string()),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
}

#[derive(Deserialize, Default)]
struct CompactArg {
    #[serde(default)]
    focus: Option<String>,
}

/// Queue a refocus for the thread: it runs before the thread's next turn, now if nothing else is queued.
async fn compact(State(app): Shared, headers: HeaderMap, Path(id): Path<String>, body: Bytes) -> Response {
    if let Err(r) = operator(&app, &headers) { return *r; }
    if !app.cfg.compaction { return err(StatusCode::CONFLICT, "compaction is off in the config"); }
    let a: CompactArg = if body.is_empty() { CompactArg::default() } else { match serde_json::from_slice(&body) { Ok(a) => a, Err(_) => return err(StatusCode::BAD_REQUEST, "body must be JSON") } };
    if !thread_id_ok(&id) { return err(StatusCode::BAD_REQUEST, "bad thread id"); }
    match app.store.thread(&id) {
        Ok(Some(_)) => {}
        Ok(None) => return err(StatusCode::NOT_FOUND, "no such thread"),
        Err(e) => return err(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
    let focus = a.focus.filter(|f| !f.trim().is_empty()).map(|f| f.chars().take(2000).collect::<String>());
    let ev = Event { id: format!("compact-{}", config::id()), thread: id.clone(), due_ms: 0, created_ms: 0, attempts: 0, kind: EventKind::Compact { focus } };
    match app.queue.push(ev) {
        Ok(_) => Json(json!({"thread": id, "queued": true})).into_response(),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
}

/// Freeze the current experts that have no state for this context, in the background.
async fn freeze_in_background(app: &Arc<App>) {
    let mut defs = rap::builtin_defs(app.cfg.compaction, app.cfg.window);
    defs.extend(app.tools.current().await.defs.iter().cloned());
    let app = app.clone();
    tokio::spawn(async move {
        app.experts.freeze_all(&app.store, &app.router, &app.engine.system, &defs, app.cfg.max_output_tokens).await;
    });
}

async fn experts_list(State(app): Shared, headers: HeaderMap) -> Response {
    if let Err(r) = operator(&app, &headers) { return *r; }
    let mut defs = rap::builtin_defs(app.cfg.compaction, app.cfg.window);
    defs.extend(app.tools.current().await.defs.iter().cloned());
    let ctx = experts::context(&app.engine.system, &defs);
    let current = app.experts.current.read().await.clone();
    let threads = app.store.threads().unwrap_or_default();
    let all = app.store.experts().unwrap_or_default();
    let out: Vec<Value> = current.iter().map(|(name, key)| {
        let e = all.iter().find(|e| format!("{}@{}", e.name, e.hash) == *key);
        let on_current = threads.iter().filter(|(_, m)| m.expert.as_deref() == Some(key)).count();
        let on_older = threads.iter().filter(|(_, m)| m.expert.as_deref().is_some_and(|k| k.starts_with(&format!("{name}@")) && k != key)).count();
        json!({"name": name, "key": key, "description": e.map(|e| e.description.clone()), "steer": e.and_then(|e| e.steer.clone()), "tools": e.and_then(|e| e.tools.clone()), "bytes": e.map(|e| e.bytes), "est_tokens": e.map(|e| e.bytes / 4), "threads": on_current, "threads_on_older_versions": on_older,
               "frozen_bytes": app.experts.state(key, &ctx).and_then(|p| std::fs::metadata(p).ok()).map(|m| m.len())})
    }).collect();
    Json(out).into_response()
}

async fn experts_reload(State(app): Shared, headers: HeaderMap) -> Response {
    if let Err(r) = operator(&app, &headers) { return *r; }
    match app.experts.load(&app.cfg, &app.store).await {
        Ok(names) => {
            freeze_in_background(&app).await;
            Json(json!({"loaded": names})).into_response()
        }
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
}

async fn experts_warm(State(app): Shared, headers: HeaderMap, Path(name): Path<String>) -> Response {
    if let Err(r) = operator(&app, &headers) { return *r; }
    let mut defs = rap::builtin_defs(app.cfg.compaction, app.cfg.window);
    defs.extend(app.tools.current().await.defs.iter().cloned());
    match app.experts.warm(&name, &app.store, &app.router, &app.engine.system, &defs, app.cfg.max_output_tokens).await {
        Ok(nodes) => Json(json!({"expert": name, "nodes": nodes})).into_response(),
        Err(e) => err(StatusCode::NOT_FOUND, &e.to_string()),
    }
}

#[derive(Deserialize)]
struct Freeze {
    name: String,
    #[serde(default)]
    description: String,
}

/// Freeze a thread's focal point as an expert: its latest refocus notes become what the expert has read.
async fn experts_from_thread(State(app): Shared, headers: HeaderMap, Path(id): Path<String>, body: Bytes) -> Response {
    if let Err(r) = operator(&app, &headers) { return *r; }
    let f: Freeze = match serde_json::from_slice(&body) { Ok(f) => f, Err(_) => return err(StatusCode::BAD_REQUEST, "body must be JSON with name") };
    if !experts::name_ok(&f.name) { return err(StatusCode::BAD_REQUEST, "expert names are 1-64 of [A-Za-z0-9._-]"); }
    let notes = match app.store.thread(&id) {
        Ok(Some(m)) => match m.compactions.last() { Some(c) => c.summary.clone(), None => return err(StatusCode::CONFLICT, "the thread has no refocus notes yet; POST /threads/{id}/compact first") },
        Ok(None) => return err(StatusCode::NOT_FOUND, "no such thread"),
        Err(e) => return err(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    };
    let e = experts::from_notes(&f.name, &f.description, &id, &notes);
    match app.experts.add(&app.store, &e).await {
        Ok(key) => {
            freeze_in_background(&app).await;
            Json(json!({"expert": f.name, "key": key, "bytes": e.bytes})).into_response()
        }
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
}

async fn delete_thread(State(app): Shared, headers: HeaderMap, Path(id): Path<String>) -> Response {
    if let Err(r) = operator(&app, &headers) { return *r; }
    if app.queue.is_inflight(&id) { return err(StatusCode::CONFLICT, "a slice of this thread is running; try again when it has committed"); }
    app.queue.forget_thread(&id);
    if let Some(p) = &app.engine.parking { p.forget(&id); }
    match app.store.delete_thread(&id) {
        Ok(()) => Json(json!({"deleted": id})).into_response(),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
}

async fn dlq(State(app): Shared, headers: HeaderMap) -> Response {
    if let Err(r) = operator(&app, &headers) { return *r; }
    match app.store.dlq() {
        Ok(d) => Json(d.into_iter().map(|(seq, e)| json!({"seq": seq, "id": e.id, "thread": e.thread, "attempts": e.attempts, "created_ms": e.created_ms, "kind": e.kind})).collect::<Vec<_>>()).into_response(),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
}

#[derive(Deserialize, Default)]
struct ThreadArg {
    #[serde(default)]
    thread: Option<String>,
}

async fn redrive(State(app): Shared, headers: HeaderMap, body: Bytes) -> Response {
    if let Err(r) = operator(&app, &headers) { return *r; }
    let a: ThreadArg = if body.is_empty() { ThreadArg::default() } else { match serde_json::from_slice(&body) { Ok(a) => a, Err(_) => return err(StatusCode::BAD_REQUEST, "body must be JSON") } };
    match app.store.redrive(a.thread.as_deref()) {
        Ok(items) => {
            let n = items.len();
            app.queue.requeue(items);
            Json(json!({"redriven": n})).into_response()
        }
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
}

async fn drop_dlq(State(app): Shared, headers: HeaderMap, body: Bytes) -> Response {
    if let Err(r) = operator(&app, &headers) { return *r; }
    let a: ThreadArg = if body.is_empty() { ThreadArg::default() } else { match serde_json::from_slice(&body) { Ok(a) => a, Err(_) => return err(StatusCode::BAD_REQUEST, "body must be JSON") } };
    match app.store.drop_dlq(a.thread.as_deref()) {
        Ok(n) => Json(json!({"dropped": n})).into_response(),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
}

async fn callback_path(State(app): Shared, headers: HeaderMap, Path(token): Path<String>, body: Bytes) -> Response {
    deliver(&app, bearer(&headers).unwrap_or(&token), &body)
}

async fn callback(State(app): Shared, headers: HeaderMap, body: Bytes) -> Response {
    deliver(&app, bearer(&headers).unwrap_or(""), &body)
}

fn deliver(app: &App, token: &str, body: &[u8]) -> Response {
    if !same(token, &app.cfg.callback_token) { return err(StatusCode::UNAUTHORIZED, "callback token required"); }
    let v: Value = match serde_json::from_slice(body) { Ok(v) => v, Err(_) => return err(StatusCode::BAD_REQUEST, "body must be JSON") };
    match rap::callback_event(&v) {
        Ok(Some(ev)) => match app.queue.push(ev) {
            Ok(Some(_)) => Json(json!({"ok": true})).into_response(),
            Ok(None) => Json(json!({"ok": true, "duplicate": true})).into_response(),
            Err(e) => {
                tracing::error!("callback: {e:#}");
                err(StatusCode::INTERNAL_SERVER_ERROR, "could not store the result")
            }
        },
        Ok(None) => Json(json!({"ok": true, "ignored": true})).into_response(),
        Err(e) => err(StatusCode::BAD_REQUEST, &e.to_string()),
    }
}
