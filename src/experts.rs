//! Preloaded experts: a named prefix a thread can be born with. An expert has read documents and
//! acknowledged them; every thread started from it sends those exact bytes after the shared system prompt,
//! so a node's prompt cache serves the first turn of each such thread, however much the expert has read.
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
use std::collections::HashMap;
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

/// The experts by name, each at its current version key ("name@hash").
pub struct Experts {
    pub current: RwLock<HashMap<String, String>>,
}

impl Experts {
    pub fn new() -> Arc<Experts> {
        Arc::new(Experts { current: RwLock::new(HashMap::new()) })
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

    /// Read an expert's prefix once on every up node, so each node's cache holds it before any thread
    /// starts from it. Best effort: a node's cache may evict it later, and the next thread reads it again.
    pub async fn warm(&self, name: &str, store: &Store, router: &Router, system: &str, defs: &[Value]) -> Result<Vec<Value>> {
        let key = self.key(name).await.with_context(|| format!("no expert named {name}"))?;
        let e = store.expert(&key)?.context("expert vanished")?;
        let mut messages = vec![json!({"role": "system", "content": system})];
        messages.extend(e.messages.iter().cloned());
        // a short real answer: a node keeps the entry of a completed turn, not of one cut at a token
        messages.push(json!({"role": "user", "content": "Say ready."}));
        let body = json!({"model": router.model(), "messages": messages, "tools": defs, "max_tokens": 16, "stream": false});
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
