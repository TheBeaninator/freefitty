//! Experts: the model's cached state after reading, kept as memory with an address, a version and a
//! residence. An expert has read documents and acknowledged them. Frozen, it is the node's state at the end of
//! that reading: read once on one node, exported (`POST /cache/export`) into `<data_dir>/experts/<name>@<hash>-
//! <context>.kv`, and imported (`POST /cache/import`) into whichever node runs the first turn of a thread born
//! from it, so that turn reads only its own message, on any node, across restarts, however much the expert has
//! read. The context part of the name is a hash of the system prompt and the tool definitions, which render
//! before the expert: when either changes, the expert is frozen again for the new context, and a thread born
//! meanwhile reads the expert itself. The stored messages are the expert's provenance and what it is rebuilt
//! from; every thread born from it sends those exact bytes after the shared system prompt.
//!
//! Defined as `<experts_dir>/<name>.toml`:
//!
//! ```toml
//! description = "Has read the pod runbook"
//! ack = "I have read the runbook and will work within it."
//! [[reads]]
//! path = "/data/pod/objective/RUNBOOK.md"
//! [[reads]]
//! text = "The board is /data/pod/kanban.db."
//! ```
//!
//! Or frozen from a thread's focal point: its latest refocus notes become what the expert has read.
//! Versions are stored by content hash and never pruned: a thread keeps the bytes it was born with when a
//! file changes. The system prompt is not part of the stored expert, so a prompt edit changes every thread
//! alike and the system block stays shared.

use crate::config::Config;
use crate::rap;
use crate::router::Router;
use crate::store::{now_ms, Expert, Store};
use anyhow::{bail, Context, Result};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Deserialize)]
struct Definition {
    #[serde(default)]
    description: String,
    #[serde(default = "default_ack")]
    ack: String,
    #[serde(default)]
    reads: Vec<Read>,
}

#[derive(Deserialize)]
struct Read {
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    text: Option<String>,
}

fn default_ack() -> String {
    "I have read these documents and will work within them.".into()
}

pub fn name_ok(n: &str) -> bool {
    !n.is_empty() && n.len() <= 64 && n.bytes().all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
}

/// The expert's two messages from its documents: byte-identical for the same documents in the same order.
fn build(name: &str, description: &str, ack: &str, docs: &[(Option<String>, String)]) -> Expert {
    let mut body = format!("<expert name=\"{name}\">\n");
    for (path, text) in docs {
        match path {
            Some(p) => body.push_str(&format!("<document path=\"{p}\">\n{text}\n</document>\n")),
            None => body.push_str(&format!("<document>\n{text}\n</document>\n")),
        }
    }
    body.push_str("</expert>\nYou have read these documents. Acknowledge in one sentence.");
    let messages = vec![json!({"role": "user", "content": body}), json!({"role": "assistant", "content": ack})];
    let bytes = serde_json::to_vec(&messages).map(|b| b.len()).unwrap_or(0) as u64;
    let hash = rap::short_hash(&serde_json::to_string(&messages).unwrap_or_default());
    Expert { name: name.to_owned(), hash, description: description.to_owned(), messages, bytes, loaded_ms: now_ms() }
}

fn load_file(path: &std::path::Path) -> Result<Expert> {
    let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("").to_owned();
    if !name_ok(&name) { bail!("{}: expert names are 1-64 of [A-Za-z0-9._-]", path.display()); }
    let def: Definition = toml::from_str(&std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?).with_context(|| format!("parsing {}", path.display()))?;
    let mut docs = Vec::new();
    for r in &def.reads {
        match (&r.path, &r.text) {
            (Some(p), _) => docs.push((Some(p.clone()), std::fs::read_to_string(p).with_context(|| format!("{}: reading {p}", path.display()))?)),
            (None, Some(t)) => docs.push((None, t.clone())),
            (None, None) => bail!("{}: a [[reads]] entry needs path or text", path.display()),
        }
    }
    Ok(build(&name, &def.description, &def.ack, &docs))
}

/// An expert whose reading is a thread's focal point: its latest refocus notes.
pub fn from_notes(name: &str, description: &str, thread: &str, notes: &str) -> Expert {
    build(name, description, "I hold this understanding and will continue from it.", &[(Some(format!("notes-of-thread-{thread}")), notes.to_owned())])
}

/// What renders before an expert's messages: the system prompt and the tool definitions.
pub fn context(system: &str, defs: &[Value]) -> String {
    rap::short_hash(&format!("{system}\u{0}{}", serde_json::to_string(defs).unwrap_or_default()))
}

/// The request an expert is read with, and the one its state is exported with. Strixite saves a checkpoint at
/// the start of the last user message, which here is the end of the expert's acknowledgement; exporting with a
/// different last message picks that checkpoint, the prefix every thread born from the expert shares.
fn bodies(e: &Expert, system: &str, defs: &[Value], model: &str, max_tokens: u64) -> (Value, Value) {
    let with = |last: &str| {
        let mut messages = vec![json!({"role": "system", "content": system})];
        messages.extend(e.messages.iter().cloned());
        messages.push(json!({"role": "user", "content": last}));
        json!({"model": model, "messages": messages, "tools": defs, "max_tokens": max_tokens, "stream": false})
    };
    (with("Say ready."), with("."))
}

/// The experts by name, each at its current version key ("name@hash"), and where their frozen state lives.
pub struct Experts {
    pub current: RwLock<HashMap<String, String>>,
    state_dir: Option<PathBuf>,
    freezing: std::sync::Mutex<HashSet<PathBuf>>,
}

impl Experts {
    pub fn new(state_dir: Option<PathBuf>) -> Arc<Experts> {
        if let Some(d) = &state_dir {
            if let Err(e) = std::fs::create_dir_all(d) { tracing::warn!("experts: state directory {}: {e}", d.display()); }
        }
        Arc::new(Experts { current: RwLock::new(HashMap::new()), state_dir, freezing: Default::default() })
    }

    fn state_path(&self, key: &str, ctx: &str) -> Option<PathBuf> {
        self.state_dir.as_ref().map(|d| d.join(format!("{key}-{ctx}.kv")))
    }

    /// The frozen state of an expert version in a context, if it has been frozen.
    pub fn state(&self, key: &str, ctx: &str) -> Option<PathBuf> {
        self.state_path(key, ctx).filter(|p| p.is_file())
    }

    /// Freeze an expert version for the current context: read it once on a node and export that node's state.
    /// A version already frozen is left alone; two freezes of the same one never run at once.
    pub async fn freeze(&self, key: &str, store: &Store, router: &Router, system: &str, defs: &[Value], max_tokens: u64) -> Result<Value> {
        let ctx = context(system, defs);
        let path = self.state_path(key, &ctx).context("no expert state directory")?;
        if path.is_file() {
            return Ok(json!({"key": key, "state": path, "bytes": std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0), "frozen_before": true}));
        }
        if !self.freezing.lock().unwrap().insert(path.clone()) { bail!("{key} is being frozen already"); }
        let r = self.freeze_into(key, &path, store, router, system, defs, max_tokens).await;
        self.freezing.lock().unwrap().remove(&path);
        if r.is_ok() {
            // the state of older contexts of this version renders before nothing that is sent any more
            if let (Some(dir), Some(name)) = (&self.state_dir, path.file_name().and_then(|f| f.to_str())) {
                for e in std::fs::read_dir(dir).into_iter().flatten().flatten() {
                    let f = e.file_name().to_string_lossy().into_owned();
                    if f.starts_with(&format!("{key}-")) && f.ends_with(".kv") && f != name { let _ = std::fs::remove_file(e.path()); }
                }
            }
        }
        r
    }

    async fn freeze_into(&self, key: &str, path: &Path, store: &Store, router: &Router, system: &str, defs: &[Value], max_tokens: u64) -> Result<Value> {
        let e = store.expert(key)?.with_context(|| format!("expert {key} is not stored"))?;
        let (read, export) = bodies(&e, system, defs, router.model(), max_tokens);
        let t = std::time::Instant::now();
        let turn = router.chat(None, &read).await.map_err(|e| anyhow::anyhow!("reading the expert: {e:?}"))?;
        let read_s = t.elapsed().as_secs_f64();
        match router.save_state(&turn.node, &export, path).await? {
            Some(n) => {
                tracing::info!("expert {key}: frozen on {} ({:?} prompt tokens, {:?} from its cache, read in {read_s:.0} s): {:.2} GB", turn.node, turn.prompt_tokens, turn.cached_tokens, n as f64 / 1e9);
                Ok(json!({"key": key, "node": turn.node, "prompt_tokens": turn.prompt_tokens, "cached_tokens": turn.cached_tokens, "read_s": read_s, "state": path, "bytes": n}))
            }
            None => bail!("{} kept no state for the expert (is cache-xfer on?)", turn.node),
        }
    }

    /// Freeze every current expert that has no state for this context; failures are logged.
    pub async fn freeze_all(&self, store: &Store, router: &Router, system: &str, defs: &[Value], max_tokens: u64) {
        let keys: Vec<String> = self.current.read().await.values().cloned().collect();
        for key in keys {
            if let Err(e) = self.freeze(&key, store, router, system, defs, max_tokens).await { tracing::warn!("expert {key}: not frozen: {e:#}"); }
        }
    }

    /// Read `experts_dir`; the names loaded. A file that fails is reported and skipped, the others load.
    pub async fn load(&self, cfg: &Config, store: &Store) -> Result<Vec<String>> {
        let Some(dir) = &cfg.experts_dir else { return Ok(vec![]) };
        let mut names = Vec::new();
        let mut entries: Vec<_> = std::fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))?.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "toml")).collect();
        entries.sort();
        for path in entries {
            match load_file(&path) {
                Ok(e) => {
                    let key = store.save_expert(&e)?;
                    self.current.write().await.insert(e.name.clone(), key);
                    names.push(e.name);
                }
                Err(err) => tracing::warn!("experts: {err:#}"),
            }
        }
        Ok(names)
    }

    pub async fn add(&self, store: &Store, e: &Expert) -> Result<String> {
        let key = store.save_expert(e)?;
        self.current.write().await.insert(e.name.clone(), key.clone());
        Ok(key)
    }

    pub async fn key(&self, name: &str) -> Option<String> {
        self.current.read().await.get(name).cloned()
    }

    /// Put an expert into every up node's cache before any thread starts from it: its frozen state, frozen
    /// first if need be, or, where the nodes cannot take state, a reading on each. Not needed for a thread to be
    /// born from it on any node; it saves the import on that thread's first turn.
    pub async fn warm(&self, name: &str, store: &Store, router: &Router, system: &str, defs: &[Value], max_tokens: u64) -> Result<Vec<Value>> {
        let key = self.key(name).await.with_context(|| format!("no expert named {name}"))?;
        let e = store.expert(&key)?.context("expert vanished")?;
        if let Err(err) = self.freeze(&key, store, router, system, defs, max_tokens).await { tracing::warn!("expert {key}: not frozen: {err:#}"); }
        if let Some(state) = self.state(&key, &context(system, defs)) {
            let mut out = Vec::new();
            for url in router.up_nodes().await {
                let t = std::time::Instant::now();
                match router.load_state(&url, &state).await {
                    Ok(n) => out.push(json!({"url": url, "imported_bytes": n, "seconds": t.elapsed().as_secs_f64()})),
                    Err(err) => out.push(json!({"url": url, "error": err.to_string()})),
                }
            }
            tracing::info!("expert {name}: frozen state put into {} node(s)", out.len());
            return Ok(out);
        }
        let (body, _) = bodies(&e, system, defs, router.model(), max_tokens);
        let mut out = Vec::new();
        for url in router.up_nodes().await {
            let t = std::time::Instant::now();
            match router.chat_on(&url, &body).await {
                Ok(turn) => out.push(json!({"url": url, "prompt_tokens": turn.prompt_tokens, "cached_tokens": turn.cached_tokens, "seconds": t.elapsed().as_secs_f64()})),
                Err(err) => out.push(json!({"url": url, "error": err.to_string()})),
            }
        }
        tracing::info!("expert {name}: warmed on {} node(s)", out.len());
        Ok(out)
    }
}
