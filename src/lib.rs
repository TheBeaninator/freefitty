//! FreeFitty: durable agents on a cluster of Strix Halos. One binary, one data file.
//!
//! A thread is its stored conversation. Events for it (a message, a tool's result, a wake-up) queue in the
//! store; a slice runs the model once over them and commits the turn. Between slices a thread holds no
//! process. The router keeps each thread on the node whose prompt cache has it, and moves the cache over the
//! rail when a node is full. Tools are RAP servers; their results come back as events.

pub mod api;
pub mod config;
/// The version and the commit this binary was built from ("-dirty" when the tree had changes).
pub const BUILD: &str = concat!(env!("CARGO_PKG_VERSION"), " (", env!("FREEFITTY_COMMIT"), ")");

pub mod engine;
pub mod park;
pub mod experts;
pub mod metrics;
pub mod queue;
pub mod rap;
pub mod router;
pub mod store;
pub mod window;

use anyhow::{Context, Result};
use std::sync::Arc;
use tokio::sync::Notify;

pub struct Running {
    pub app: Arc<api::App>,
    pub addr: std::net::SocketAddr,
    tasks: Vec<tokio::task::JoinHandle<()>>,
    server: tokio::task::JoinHandle<()>,
    stop: Arc<Notify>,
    /// Fires when the scheduler or the server ends on its own: the process must exit so the service manager
    /// brings back a whole daemon, never one that answers /health and takes no work.
    pub fault: Arc<Notify>,
}

impl Running {
    /// Stop taking work, let running slices finish (they commit atomically), then stop serving.
    pub async fn shutdown(self) {
        self.app.queue.stop();
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(self.app.cfg.slice_timeout_s + 5);
        while self.app.queue.running() > 0 && tokio::time::Instant::now() < deadline {
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
        for t in &self.tasks { t.abort(); }
        for t in self.tasks { let _ = t.await; }
        // the server closes idle connections and finishes the requests in flight
        self.stop.notify_one();
        let _ = tokio::time::timeout(std::time::Duration::from_secs(5), self.server).await;
    }
}

/// Open the store, read the tools, start the router's polls, the dispatcher, the scheduler and the API.
pub async fn start(cfg: config::Config) -> Result<Running> {
    cfg.check()?;
    let mut tasks = Vec::new();
    let store = Arc::new(store::Store::open(&cfg.data_dir, cfg.compact_on_start)?);
    // bound first: with no callback_base the tool servers call back to the port actually taken
    let listener = tokio::net::TcpListener::bind(&cfg.listen).await.with_context(|| format!("binding {}", cfg.listen))?;
    let addr = listener.local_addr()?;
    let mut cfg = cfg;
    if cfg.listen.ends_with(":0") { cfg.listen = addr.to_string(); }
    let router = router::Router::new(&cfg, &mut tasks);
    let tools = rap::ToolBox::start(cfg.clone(), store.clone(), &mut tasks).await?;
    let system = cfg.system_prompt()?;
    let compact_prompt = cfg.compact_prompt()?;
    let experts = experts::Experts::new(Some(cfg.data_dir.join("experts")));
    let loaded = experts.load(&cfg, &store).await?;
    if !loaded.is_empty() { tracing::info!("experts: {}", loaded.join(", ")); }
    let parking = if cfg.park {
        let dir = cfg.park_dir.clone().unwrap_or_else(|| cfg.data_dir.join("parked"));
        let p = park::Parking::new(dir.clone(), cfg.park_max_gib).with_context(|| format!("park directory {}", dir.display()))?;
        tracing::info!("parking sleepers' cached state in {} ({} GiB parked now, cap {} GiB)", dir.display(), p.bytes() >> 30, cfg.park_max_gib);
        Some(Arc::new(p))
    } else { None };
    let engine = Arc::new(engine::Engine { cfg: cfg.clone(), store: store.clone(), router: router.clone(), tools: tools.clone(), system, compact_prompt, callback_url: cfg.callback_url(), scale_hint: std::sync::atomic::AtomicU64::new(0), parking, inflight: Default::default(), me: Default::default(), experts: experts.clone(), deliver: Default::default(), metrics: Default::default() });
    let _ = engine.me.set(Arc::downgrade(&engine));
    let replies = Arc::new(Notify::new());

    // the dispatcher's failures become events; the queue is built after it, so it reaches the queue through a slot
    let queue_slot: Arc<std::sync::OnceLock<std::sync::Weak<queue::Queue>>> = Arc::new(std::sync::OnceLock::new());
    let slot = queue_slot.clone();
    let dispatcher = rap::Dispatcher::start(store.clone(), Box::new(move |ev| {
        if let Some(q) = slot.get().and_then(|w| w.upgrade()) {
            if let Err(e) = q.push(ev) { tracing::error!("dispatcher: could not queue a failure: {e:#}"); }
        }
    }), &mut tasks);
    let d = dispatcher.clone();
    let queue = queue::Queue::new(store.clone(), engine.clone(), replies.clone(), Box::new(move || d.kick()))?;
    let _ = queue_slot.set(Arc::downgrade(&queue));
    let wq = Arc::downgrade(&queue);
    let _ = engine.inflight.set(Box::new(move |t| wq.upgrade().is_some_and(|q| q.is_inflight(t))));
    let weak = Arc::downgrade(&queue);
    let _ = engine.deliver.set(Box::new(move |ev| {
        if let Some(q) = weak.upgrade() {
            if let Err(e) = q.push(ev) { tracing::error!("could not queue an event: {e:#}"); }
        }
    }));

    let app = api::App::new(cfg.clone(), store.clone(), queue.clone(), router.clone(), tools.clone(), replies, engine.clone(), experts.clone());

    // once a node is up: every expert without frozen state for this context is frozen, then the experts named in
    // warm_on_start are put into every node's cache
    if !experts.current.read().await.is_empty() {
        let (cfg2, store2, router2, tools2, experts2, engine2) = (cfg.clone(), store.clone(), router.clone(), tools.clone(), experts.clone(), engine.clone());
        tasks.push(tokio::spawn(async move {
            while !router2.any_up().await { tokio::time::sleep(std::time::Duration::from_secs(1)).await; }
            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
            let mut defs = rap::builtin_defs(cfg2.compaction, cfg2.window);
            defs.extend(tools2.current().await.defs.iter().cloned());
            experts2.freeze_all(&store2, &router2, &engine2.system, &defs, cfg2.max_output_tokens).await;
            for name in &cfg2.warm_on_start {
                match experts2.warm(name, &store2, &router2, &engine2.system, &defs, cfg2.max_output_tokens).await {
                    Ok(r) => tracing::info!("warm {name}: {}", serde_json::to_string(&r).unwrap_or_default()),
                    Err(e) => tracing::warn!("warm {name}: {e:#}"),
                }
            }
        }));
    }
    let routes = api::routes(app.clone());
    let fault = Arc::new(Notify::new());
    let stop = Arc::new(Notify::new());
    let stop_rx = stop.clone();
    let f = fault.clone();
    let server = tokio::spawn(async move {
        if let Err(e) = axum::serve(listener, routes).with_graceful_shutdown(async move { stop_rx.notified().await }).await { tracing::error!("server: {e}"); }
        f.notify_one();
    });
    let f = fault.clone();
    let q = queue.clone();
    tasks.push(tokio::spawn(async move {
        q.clone().run().await;
        if q.is_stopping() { tracing::info!("the scheduler stopped"); } else { tracing::error!("the scheduler ended"); }
        f.notify_one();
    }));

    // snapshots of the store, newest backup_keep kept: one at start when the newest is older than the interval
    if cfg.backup_every_h > 0 {
        let (s, dir, every, keep) = (store.clone(), cfg.data_dir.join("backups"), cfg.backup_every_h * 3600, cfg.backup_keep.max(1));
        tasks.push(tokio::spawn(async move {
            loop {
                let newest = newest_backup_age(&dir);
                if newest.is_none_or(|age| age >= every) {
                    match backup_now(&s, &dir, keep) {
                        Ok((p, n)) => tracing::info!("store snapshot {} ({:.1} MB)", p.display(), n as f64 / 1e6),
                        Err(e) => tracing::warn!("store snapshot failed: {e:#}"),
                    }
                }
                tokio::time::sleep(std::time::Duration::from_secs(3600)).await;
            }
        }));
    }

    // replies nobody collected expire
    let s = store.clone();
    let days = cfg.reply_retention_days;
    tasks.push(tokio::spawn(async move {
        loop {
            tokio::time::sleep(std::time::Duration::from_secs(3600)).await;
            match s.expire_replies(store::now_ms().saturating_sub(days * 86_400_000)) {
                Ok(n) if n > 0 => tracing::info!("{n} uncollected repl{} expired", if n == 1 { "y" } else { "ies" }),
                Err(e) => tracing::warn!("expiring replies: {e:#}"),
                _ => {}
            }
            // processed event ids guard against a duplicate delivery; none comes weeks later
            match s.expire_processed(store::now_ms().saturating_sub(4 * days * 86_400_000)) {
                Ok(n) if n > 0 => tracing::info!("{n} processed event id(s) forgotten"),
                Err(e) => tracing::warn!("expiring processed ids: {e:#}"),
                _ => {}
            }
        }
    }));

    tracing::info!("freefitty serving on {addr}; {} node(s), data in {}", cfg.nodes.len(), cfg.data_dir.display());
    Ok(Running { app, addr, tasks, server, stop, fault })
}

/// Seconds since the newest snapshot in `dir` was written, if there is one.
fn newest_backup_age(dir: &std::path::Path) -> Option<u64> {
    std::fs::read_dir(dir).ok()?.flatten()
        .filter(|e| e.file_name().to_string_lossy().ends_with(".redb"))
        .filter_map(|e| e.metadata().ok()?.modified().ok()?.elapsed().ok())
        .min().map(|d| d.as_secs())
}

/// Take a snapshot now into `dir` (freefitty-<utc>.redb) and keep the newest `keep`: its path and size.
pub fn backup_now(store: &store::Store, dir: &std::path::Path, keep: usize) -> Result<(std::path::PathBuf, u64)> {
    // one at a time (the start-of-day snapshot and an operator's can coincide), each under its own name
    static ONE: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    let path = dir.join(format!("freefitty-{}.redb", chrono::Utc::now().format("%Y%m%dT%H%M%S%.3fZ")));
    let n = store.backup(&path)?;
    let mut all: Vec<_> = std::fs::read_dir(dir)?.flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "redb")).collect();
    all.sort();
    for old in all.iter().take(all.len().saturating_sub(keep)) { let _ = std::fs::remove_file(old); }
    Ok((path, n))
}
