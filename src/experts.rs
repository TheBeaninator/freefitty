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
//! `model = "dsv41"` puts the expert and every thread born from it on the nodes serving that model (the StrixFold
//! pair); unset = the config's `model`. A node that cannot export state (StrixFold today) is not frozen on: the
//! thread reads the expert there, and the node's own prompt cache serves the next thread born from it.
//!
//! `[tool]` makes the expert answer a tool: `name`, `description` and JSON-schema `parameters` define a tool every
//! thread sees beside the others, and a call to it is a goldfish of this expert with the call's arguments as its brief
//! (`ask`, optional: what the answer must look like). To the caller it is one more tool with a result; behind it is
//! a model that has read what the expert read. A ToolBox tool of the same name wins, and a goldfish cannot call one.
//! `fallback` names an expert that answers instead while no node of this expert's model is up and free.
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
    /// The tools a thread born from it may call; unset = every tool (an elephant's expert), [] = none.
    #[serde(default)]
    tools: Option<Vec<String>>,
    /// Activation steering: `[[steer]] vector = "villain24", scale = 6.0, layer = 24 (optional)`.
    #[serde(default)]
    steer: Vec<Steer>,
    /// `steer_scope = "prefix"`: steer the expert's reading only, not the thread's turns or the generation
    /// (strixite `steer_scope`); unset or "all" = every row.
    #[serde(default)]
    steer_scope: Option<String>,
    /// The model the expert and its threads run on; unset = the config's `model`.
    #[serde(default)]
    model: Option<String>,
    /// A tool this expert answers (see the module docs).
    #[serde(default)]
    tool: Option<ToolSpec>,
}

/// A tool answered by a goldfish of an expert: looks like any tool to the caller.
#[derive(Deserialize, Clone, Debug)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    #[serde(default = "no_parameters")]
    pub parameters: Value,
    /// What the answer must look like; unset = the result only, in the shape the description promises.
    #[serde(default)]
    pub ask: Option<String>,
    /// Another expert that answers in its place while no node of this one's model can take a turn (the StrixFold
    /// pair down or held for a card window): a call never waits hours on one model when another has read the same.
    #[serde(default)]
    pub fallback: Option<String>,
}

fn no_parameters() -> Value {
    json!({"type": "object", "properties": {}})
}

impl ToolSpec {
    pub fn def(&self) -> Value {
        json!({"type": "function", "function": {"name": self.name, "description": self.description, "parameters": self.parameters}})
    }

    /// The goldfish's brief for one call.
    pub fn brief(&self, args: &Value) -> String {
        let args = serde_json::to_string_pretty(args).unwrap_or_default();
        let ask = self.ask.as_deref().unwrap_or("Answer as the tool's result: what was asked for, in the shape its description promises, with no preamble and nothing after it.");
        format!("You are answering a call to the tool `{}` ({}). Its arguments:\n{args}\n\n{ask}", self.name, self.description)
    }
}

/// A definition's `steer_scope`, checked: None (= "all") or Some("prefix").
pub fn steer_scope_of(scope: Option<&str>, steer: &[Steer]) -> Result<Option<String>> {
    match scope {
        None | Some("all") => Ok(None),
        Some("prefix") if steer.is_empty() => bail!("steer_scope = \"prefix\" needs [[steer]] vectors to scope"),
        Some("prefix") => Ok(Some("prefix".to_owned())),
        Some(other) => bail!("steer_scope {other:?}: \"all\" or \"prefix\""),
    }
}

/// The steering fields of a request for a thread of this expert: strixite's `steer`, and under scope "prefix"
/// its `steer_scope` {"messages": k} with k the shared prefix's message count (the system prompt and the expert's
/// messages, which every request of the thread starts with). The nodes steer the prompt those k messages render to
/// and nothing after it.
pub fn steer_into(body: &mut Value, e: &Expert) {
    if let Some(s) = &e.steer {
        body["steer"] = s.clone();
        if e.steer_scope.as_deref() == Some("prefix") { body["steer_scope"] = json!({"messages": 1 + e.messages.len()}); }
    }
}

/// The request's `shared_messages` for a thread of this expert: the system prompt and the expert's messages, the prefix
/// every thread born from it starts with. A node that takes it (StrixFold) keeps that prefix's state as a shared one,
/// and on disk with --learn: the expert stays read across the node's restarts. Only for nodes that take the field.
pub fn shared_into(body: &mut Value, e: &Expert, on: bool) {
    if on { body["shared_messages"] = json!(1 + e.messages.len()); }
}

/// One steering vector of an expert, by its name in the nodes' steer-dir.
#[derive(Deserialize, serde::Serialize, Clone, Debug)]
pub struct Steer {
    pub vector: String,
    pub scale: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layer: Option<i64>,
}

/// The request's `steer` field for these vectors; None when there are none.
pub fn steer_field(steer: &[Steer]) -> Option<Value> {
    (!steer.is_empty()).then(|| serde_json::to_value(steer).unwrap_or(Value::Null))
}

#[derive(Deserialize, serde::Serialize, Clone)]
pub struct Read {
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

/// The expert's two messages from its documents: byte-identical for the same documents in the same order. Its
/// tool list is part of its identity (the definitions render before it), so it is in the hash when set.
#[allow(clippy::too_many_arguments)]
fn build(name: &str, description: &str, ack: &str, docs: &[(Option<String>, String)], tools: Option<Vec<String>>, steer: Option<Value>, steer_scope: Option<String>, model: Option<String>) -> Expert {
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
    let mut hashed = serde_json::to_string(&messages).unwrap_or_default();
    if let Some(t) = &tools { hashed.push_str(&format!("\u{0}tools={}", t.join(","))); }
    if let Some(s) = &steer { hashed.push_str(&format!("\u{0}steer={s}")); }
    if let Some(sc) = &steer_scope { hashed.push_str(&format!("\u{0}steer_scope={sc}")); }
    if let Some(m) = &model { hashed.push_str(&format!("\u{0}model={m}")); }
    let hash = rap::short_hash(&hashed);
    Expert { name: name.to_owned(), hash, description: description.to_owned(), messages, bytes, loaded_ms: now_ms(), tools, steer, steer_scope, model }
}

/// Read a definition's documents: (path, text) pairs, each file at most `max_file` bytes.
pub fn read_docs(reads: &[Read], max_file: usize) -> Result<Vec<(Option<String>, String)>> {
    let mut docs = Vec::new();
    for r in reads {
        match (&r.path, &r.text) {
            (Some(p), _) => {
                let text = std::fs::read_to_string(p).with_context(|| format!("reading {p}"))?;
                if text.len() > max_file { bail!("{p} is {} bytes, over {max_file}", text.len()); }
                docs.push((Some(p.clone()), text));
            }
            (None, Some(t)) => docs.push((None, t.clone())),
            (None, None) => bail!("a read needs path or text"),
        }
    }
    Ok(docs)
}

/// An expert an elephant commissioned, and its definition for `experts_dir` (when set; `write_definition`) so it
/// survives a restart, with who commissioned it. Nothing is written here: the slice has not committed yet.
pub fn commission(cfg: &Config, name: &str, description: &str, ack: Option<&str>, reads: &[Read], tools: Vec<String>, steer: Vec<Steer>, steer_scope: Option<&str>, by: &str) -> Result<(Expert, Option<(std::path::PathBuf, String)>)> {
    if !name_ok(name) { bail!("expert names are 1-64 of [A-Za-z0-9._-]"); }
    if reads.is_empty() { bail!("an expert needs something to read"); }
    let reads = reads.iter().map(|r| Ok(Read { path: r.path.as_deref().map(|p| cfg.model_read(p)).transpose()?.map(|p| p.display().to_string()), text: r.text.clone() }))
        .collect::<Result<Vec<_>>>()?;
    let reads = reads.as_slice();
    let docs = read_docs(reads, 1 << 20)?;
    let ack = ack.map(str::to_owned).unwrap_or_else(default_ack);
    let scope = steer_scope_of(steer_scope, &steer)?;
    let e = build(name, description, &ack, &docs, Some(tools.clone()), steer_field(&steer), scope.clone(), None);
    let mut definition = None;
    if let Some(dir) = &cfg.experts_dir {
        let path = dir.join(format!("{name}.toml"));
        if path.exists() { bail!("an expert definition named {name} exists already"); }
        #[derive(serde::Serialize)]
        struct Def<'a> { description: &'a str, ack: &'a str, tools: &'a [String], reads: &'a [Read], #[serde(skip_serializing_if = "<[Steer]>::is_empty")] steer: &'a [Steer], #[serde(skip_serializing_if = "Option::is_none")] steer_scope: Option<&'a str> }
        let body = toml::to_string(&Def { description, ack: &ack, tools: &tools, reads, steer: &steer, steer_scope: scope.as_deref() })?;
        definition = Some((path, format!("# commissioned by thread {by} on {}\n{body}", chrono::Utc::now().to_rfc3339())));
    }
    Ok((e, definition))
}

/// Write a commissioned expert's definition, refusing one that appeared since it was checked.
pub fn write_definition((path, text): &(std::path::PathBuf, String)) -> Result<()> {
    use std::io::Write;
    let mut f = std::fs::OpenOptions::new().write(true).create_new(true).open(path).with_context(|| format!("writing {}", path.display()))?;
    f.write_all(text.as_bytes()).with_context(|| format!("writing {}", path.display()))
}

fn load_file(path: &std::path::Path) -> Result<(Expert, Option<ToolSpec>)> {
    let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("").to_owned();
    if !name_ok(&name) { bail!("{}: expert names are 1-64 of [A-Za-z0-9._-]", path.display()); }
    let def: Definition = toml::from_str(&std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?).with_context(|| format!("parsing {}", path.display()))?;
    let docs = read_docs(&def.reads, usize::MAX).with_context(|| path.display().to_string())?;
    let scope = steer_scope_of(def.steer_scope.as_deref(), &def.steer).with_context(|| path.display().to_string())?;
    if let Some(t) = &def.tool {
        if !name_ok(&t.name) || t.name.contains('.') { bail!("{}: a tool name is 1-64 of [A-Za-z0-9_-]", path.display()); }
        if rap::is_builtin(&t.name) { bail!("{}: {} is a built-in tool", path.display(), t.name); }
        if t.parameters.get("type").and_then(|x| x.as_str()) != Some("object") { bail!("{}: tool parameters must be a JSON schema of type object", path.display()); }
    }
    Ok((build(&name, &def.description, &def.ack, &docs, def.tools, steer_field(&def.steer), scope, def.model), def.tool))
}

/// An expert whose reading is a thread's focal point: its latest refocus notes.
pub fn from_notes(name: &str, description: &str, thread: &str, notes: &str) -> Expert {
    build(name, description, "I hold this understanding and will continue from it.", &[(Some(format!("notes-of-thread-{thread}")), notes.to_owned())], None, None, None, None)
}

/// What renders before an expert's messages: the system prompt and the tool definitions.
pub fn context(system: &str, defs: &[Value]) -> String {
    rap::short_hash(&format!("{system}\u{0}{}", serde_json::to_string(defs).unwrap_or_default()))
}

/// The request an expert is read with, and the one its state is exported with. Strixite saves a checkpoint at
/// the start of the last user message, which here is the end of the expert's acknowledgement; exporting with a
/// different last message picks that checkpoint, the prefix every thread born from the expert shares.
fn bodies(e: &Expert, system: &str, defs: &[Value], model: &str, max_tokens: u64, shared: bool) -> (Value, Value) {
    let with = |last: &str| {
        let mut messages = vec![json!({"role": "system", "content": system})];
        messages.extend(e.messages.iter().cloned());
        messages.push(json!({"role": "user", "content": last}));
        let mut b = json!({"model": model, "messages": messages, "tools": defs, "max_tokens": max_tokens, "stream": false});
        steer_into(&mut b, e);
        shared_into(&mut b, e, shared);
        b
    };
    (with("Say ready."), with("."))
}

/// The experts by name, each at its current version key ("name@hash"), and where their frozen state lives.
pub struct Experts {
    pub current: RwLock<HashMap<String, String>>,
    /// The tools experts answer, by tool name: (expert name, spec).
    tools: RwLock<HashMap<String, (String, ToolSpec)>>,
    state_dir: Option<PathBuf>,
    freezing: std::sync::Mutex<HashSet<PathBuf>>,
}

impl Experts {
    pub fn new(state_dir: Option<PathBuf>) -> Arc<Experts> {
        if let Some(d) = &state_dir {
            if let Err(e) = std::fs::create_dir_all(d) { tracing::warn!("experts: state directory {}: {e}", d.display()); }
        }
        Arc::new(Experts { current: RwLock::new(HashMap::new()), tools: Default::default(), state_dir, freezing: Default::default() })
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
        // released however the freeze ends: an error, a panic, or the task being dropped at an await
        struct Unmark<'a>(&'a std::sync::Mutex<HashSet<PathBuf>>, PathBuf);
        impl Drop for Unmark<'_> {
            fn drop(&mut self) { self.0.lock().unwrap_or_else(|e| e.into_inner()).remove(&self.1); }
        }
        let _unmark = Unmark(&self.freezing, path.clone());
        let r = self.freeze_into(key, &path, store, router, system, defs, max_tokens).await;
        // other contexts of this version stay: a goldfish (scoped tools) and a thread born with every tool are
        // two live contexts of the same expert
        r
    }

    async fn freeze_into(&self, key: &str, path: &Path, store: &Store, router: &Router, system: &str, defs: &[Value], max_tokens: u64) -> Result<Value> {
        let e = store.expert(key)?.with_context(|| format!("expert {key} is not stored"))?;
        if e.steer_scope.as_deref() == Some("prefix") {
            // a node that does not know steer_scope would ignore it and steer the whole thread: never freeze there
            match router.steer_scope_everywhere().await {
                Ok(true) => {}
                Ok(false) => bail!("steer_scope = \"prefix\" needs every up node to take steer_scope (GET /steer \"scope\": true); one does not"),
                Err(why) => bail!("steer_scope could not be checked: {why}"),
            }
        }
        let model = e.model.as_deref().unwrap_or(router.model());
        let (read, export) = bodies(&e, system, defs, model, max_tokens, router.takes_shared_messages(model));
        let t = std::time::Instant::now();
        let turn = router.chat(None, None, &read).await.map_err(|e| anyhow::anyhow!("reading the expert: {e:?}"))?;
        let read_s = t.elapsed().as_secs_f64();
        // the entry wanted ends where the read's last message (a few tokens) begins
        let want = router.min_cover(&turn.node, turn.prompt_tokens.unwrap_or(0));
        match router.save_state_covering(&turn.node, &export, path, want).await? {
            Some((n, covered)) if covered < want => {
                let _ = std::fs::remove_file(path);
                bail!("{} exported only {covered} of the expert's ~{want} tokens ({:.2} GB)", turn.node, n as f64 / 1e9)
            }
            Some((n, _)) => {
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
            // frozen in the context a thread born from it sees: the tools its list allows
            let tools = store.expert(&key).ok().flatten().and_then(|e| e.tools);
            let scoped = rap::scope_defs(defs.to_vec(), tools.as_deref());
            if let Err(e) = self.freeze(&key, store, router, system, &scoped, max_tokens).await { tracing::warn!("expert {key}: not frozen: {e:#}"); }
        }
    }

    /// Read `experts_dir`; the names loaded. A file that fails is reported and skipped, the others load.
    pub async fn load(&self, cfg: &Config, store: &Store) -> Result<Vec<String>> {
        let Some(dir) = &cfg.experts_dir else { return Ok(vec![]) };
        let mut names = Vec::new();
        let mut tools: HashMap<String, (String, ToolSpec)> = HashMap::new();
        let mut entries: Vec<_> = std::fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))?.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "toml")).collect();
        entries.sort();
        for path in entries {
            match load_file(&path) {
                Ok((e, tool)) => {
                    let key = store.save_expert(&e)?;
                    self.current.write().await.insert(e.name.clone(), key);
                    if let Some(t) = tool {
                        match tools.get(&t.name) {
                            Some((other, _)) => tracing::warn!("experts: {} and {other} both answer the tool {}; {other} keeps it", e.name, t.name),
                            None => { tools.insert(t.name.clone(), (e.name.clone(), t)); }
                        }
                    }
                    names.push(e.name);
                }
                Err(err) => tracing::warn!("experts: {err:#}"),
            }
        }
        *self.tools.write().await = tools;
        Ok(names)
    }

    pub async fn add(&self, store: &Store, e: &Expert) -> Result<String> {
        let key = store.save_expert(e)?;
        self.current.write().await.insert(e.name.clone(), key.clone());
        Ok(key)
    }

    /// The catalog an elephant chooses goldfish from: each expert's name, its line, its goldfish's tools.
    pub async fn catalog(&self, store: &Store) -> String {
        let mut cur: Vec<(String, String)> = self.current.read().await.iter().map(|(n, k)| (n.clone(), k.clone())).collect();
        cur.sort();
        if cur.is_empty() { return "No experts yet. commission_expert makes one.".into(); }
        let mut out = String::from("Experts (ask one with goldfish {expert, question}):\n");
        let answers = self.tools.read().await.clone();
        for (name, key) in cur {
            let e = store.expert(&key).ok().flatten();
            let desc = e.as_ref().map(|e| e.description.clone()).filter(|d| !d.is_empty()).unwrap_or_else(|| "(no description)".into());
            let prefix_only = e.as_ref().is_some_and(|e| e.steer_scope.as_deref() == Some("prefix"));
            let steer = e.as_ref().and_then(|e| e.steer.as_ref()).and_then(|s| s.as_array()).map(|a| {
                let v: Vec<String> = a.iter().map(|x| format!("{} x{}", x["vector"].as_str().unwrap_or("?"), x["scale"])).collect();
                format!("; steered: {}{}", v.join(", "), if prefix_only { " on its reading only" } else { "" })
            }).unwrap_or_default();
            let tools = match e.as_ref().and_then(|e| e.tools.clone()) {
                Some(t) if t.is_empty() => "no tools".to_owned(),
                Some(t) => format!("tools: {}", t.join(", ")),
                None => "no tools as a goldfish".to_owned(),
            };
            let on = e.as_ref().and_then(|e| e.model.as_deref()).map(|m| format!("; runs on {m}")).unwrap_or_default();
            let answers = answers.iter().filter(|(_, (n, _))| n == &name).map(|(t, _)| format!("; answers the tool {t}")).collect::<String>();
            out.push_str(&format!("- {name}: {desc} ({tools}{steer}{on}{answers})\n"));
        }
        out
    }

    /// `defs` (the built-ins and the ToolBox's) and after them the tools experts answer, in name order; one whose
    /// name is taken already stays out, as the dispatch gives that name to the tool already there.
    pub async fn with_tools(&self, mut defs: Vec<Value>) -> Vec<Value> {
        let taken: HashSet<String> = defs.iter().filter_map(|d| d.pointer("/function/name").and_then(|n| n.as_str()).map(str::to_owned)).collect();
        let tools = self.tools.read().await;
        let mut mine: Vec<&ToolSpec> = tools.values().map(|(_, t)| t).filter(|t| !taken.contains(&t.name)).collect();
        mine.sort_by(|a, b| a.name.cmp(&b.name));
        defs.extend(mine.into_iter().map(ToolSpec::def));
        defs
    }

    /// The expert that answers the tool `name`, and its spec.
    pub async fn tool(&self, name: &str) -> Option<(String, ToolSpec)> {
        self.tools.read().await.get(name).cloned()
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
            for url in router.up_nodes_for(e.model.as_deref().unwrap_or(router.model())).await {
                let t = std::time::Instant::now();
                match router.load_state(&url, &state).await {
                    Ok(n) => out.push(json!({"url": url, "imported_bytes": n, "seconds": t.elapsed().as_secs_f64()})),
                    Err(err) => out.push(json!({"url": url, "error": err.to_string()})),
                }
            }
            tracing::info!("expert {name}: frozen state put into {} node(s)", out.len());
            return Ok(out);
        }
        let model = e.model.clone().unwrap_or_else(|| router.model().to_owned());
        let (body, _) = bodies(&e, system, defs, &model, max_tokens, router.takes_shared_messages(&model));
        let mut out = Vec::new();
        for url in router.up_nodes_for(&model).await {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_expert_tool_needs_a_free_name_and_an_object_schema() {
        let dir = tempfile::tempdir().unwrap();
        let file = |name: &str, tool: &str| { let p = dir.path().join(format!("{name}.toml")); std::fs::write(&p, format!("description = \"x\"\n[tool]\n{tool}")).unwrap(); p };
        let (_, t) = load_file(&file("ok", "name = \"look_up\"\ndescription = \"d\"\n")).unwrap();
        assert_eq!(t.unwrap().def()["function"]["parameters"]["type"], "object", "no parameters = an empty object");
        assert!(format!("{:#}", load_file(&file("b", "name = \"goldfish\"\ndescription = \"d\"\n")).unwrap_err()).contains("built-in"));
        assert!(load_file(&file("c", "name = \"x\"\ndescription = \"d\"\nparameters = { type = \"string\" }\n")).is_err());
        assert!(load_file(&file("d", "name = \"a.b\"\ndescription = \"d\"\n")).is_err());
    }
}
