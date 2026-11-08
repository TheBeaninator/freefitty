//! One redb file holds everything that must survive a restart. Every write is a transaction that is
//! fsynced before it returns, so what `Store` has acknowledged is on disk.
//!
//! Tables (keys -> JSON values):
//!   threads      thread id            -> ThreadMeta
//!   messages     (thread id, n)       -> one chat message, in conversation order
//!   events       seq                  -> Event waiting to run (the queue; due time inside)
//!   processed    event id             -> seq that ran it (dedup: a redelivered event is not run twice)
//!   dlq          seq                  -> Event that failed max_attempts times
//!   replies      seq                  -> Reply nobody has collected yet
//!   invocations  call id              -> a RAP invocation not yet acknowledged by its tool server
//!   toolsets     server url           -> the manifest last read from it
//!   meta         name                 -> u64 counters (seq)

use anyhow::{bail, Context, Result};
use redb::{Database, ReadableDatabase, ReadableTable, ReadableTableMetadata, TableDefinition};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::Value;
use std::path::Path;

const THREADS: TableDefinition<&str, &[u8]> = TableDefinition::new("threads");
const MESSAGES: TableDefinition<(&str, u64), &[u8]> = TableDefinition::new("messages");
const EVENTS: TableDefinition<u64, &[u8]> = TableDefinition::new("events");
const PROCESSED: TableDefinition<&str, u64> = TableDefinition::new("processed");
const DLQ: TableDefinition<u64, &[u8]> = TableDefinition::new("dlq");
const REPLIES: TableDefinition<u64, &[u8]> = TableDefinition::new("replies");
const INVOCATIONS: TableDefinition<&str, &[u8]> = TableDefinition::new("invocations");
const TOOLSETS: TableDefinition<&str, &[u8]> = TableDefinition::new("toolsets");
const EXPERTS: TableDefinition<&str, &[u8]> = TableDefinition::new("experts");
const META: TableDefinition<&str, u64> = TableDefinition::new("meta");

pub fn now_ms() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EventKind {
    /// A person's (or pod-ff's) message.
    User { text: String, #[serde(default)] metadata: Value },
    /// A tool's result, including a sleep's wake-up: `call` is the tool call id the model made.
    /// `images` are data:image/... URLs the tool returned beside its text; the model sees them in a user message
    /// right after the batch's tool messages (a tool message carries text only in the templates).
    ToolResult {
        call: String,
        #[serde(default)]
        provider_call_id: Option<String>,
        text: String,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        images: Vec<String>,
    },
    /// A subscription event a tool posted on its own.
    Subscription { call: String, text: String, #[serde(default)] r#final: bool },
    /// Refocus the thread before its next turn: fold what came before into notes around `focus`.
    Compact { #[serde(default)] focus: Option<String> },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Event {
    /// Unique per event; what `processed` dedups on. A redriven event keeps it.
    pub id: String,
    pub thread: String,
    /// Not run before this wall-clock time: a sleep is an event with a due time, nothing more.
    pub due_ms: u64,
    pub created_ms: u64,
    pub attempts: u32,
    #[serde(flatten)]
    pub kind: EventKind,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PendingCall {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub provider_call_id: Option<String>,
    /// The arguments JSON the model wrote, so the call can be shown again after a compaction.
    #[serde(default)]
    pub arguments: String,
}

/// One refocus: from here on the model sees `summary` in place of the history before `from`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Compaction {
    /// History index the live view starts at.
    pub from: u64,
    pub summary: String,
    #[serde(default)]
    pub focus: Option<String>,
    pub at_ms: u64,
    /// Messages this summary replaced (since the previous `from`).
    pub summarized: u64,
    pub tokens_before: u64,
    /// The summary hit `compact_summary_tokens` and may be missing its end.
    #[serde(default)]
    pub truncated: bool,
}

/// A preloaded expert: two messages (documents read, acknowledgement) a thread may start from, after the
/// system prompt. Stored by "name@hash"; a thread pins the version it was born with.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Expert {
    pub name: String,
    pub hash: String,
    pub description: String,
    pub messages: Vec<Value>,
    pub bytes: u64,
    pub loaded_ms: u64,
    /// The tools a thread born from it may call, besides the plain built-ins; None = every tool.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<String>>,
    /// Activation steering every request of a thread born from it carries (strixite `steer`: [{vector, scale,
    /// layer?}]); None = unsteered. Part of its version: the nodes cache steered state under its own key.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub steer: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ThreadMeta {
    pub created_ms: u64,
    pub updated_ms: u64,
    /// The node whose prompt cache holds this thread, as given in the config.
    #[serde(default)]
    pub node: Option<String>,
    /// What the first /say carried; every reply carries it back.
    #[serde(default)]
    pub metadata: Value,
    pub messages: u64,
    /// Bytes of stored messages: the only token measure kept (tokens ~ bytes / 4).
    pub bytes: u64,
    /// Tool calls the model made whose results have not arrived.
    #[serde(default)]
    pub pending: Vec<PendingCall>,
    /// Set when a turn ended in an error the thread cannot recover from (context full).
    #[serde(default)]
    pub closed: Option<String>,
    /// The node's own prompt-token count for the last turn, and the bytes / 4 estimate of what was sent then:
    /// their ratio corrects the estimate for the next turn (the tokenizer is denser than 4 bytes a token).
    #[serde(default)]
    pub prompt_tokens: u64,
    #[serde(default)]
    pub prompt_est: u64,
    /// Where the window starts the view (at or past the last refocus's `from`), and how often it has moved.
    #[serde(default)]
    pub window_from: u64,
    #[serde(default)]
    pub window_cuts: u64,
    /// Tool results below this history index that qualify are shown as handles (monotone; moves in batches).
    #[serde(default)]
    pub paged_before: u64,
    /// History indices the model recalled: shown in full however old.
    #[serde(default)]
    pub pinned: Vec<u64>,
    /// Pinned messages below the window's start, set at each cut and shown in the notice region from then on:
    /// a pin means resident (until the next cut), not just "never a handle".
    #[serde(default)]
    pub resident: Vec<u64>,
    /// The thread's images in history order: those before this ordinal are shown as stubs. Moved only at a cut,
    /// a paging batch or a refocus, so fading never breaks the prefix between those named events.
    #[serde(default)]
    pub images_kept_from: u64,
    /// What the newest stored message is: user, tool_result, assistant_text, tool_call or assistant_empty.
    /// A watcher reads "stopped talking" here: assistant_empty, or tool_call with nothing pending.
    #[serde(default)]
    pub last: String,
    /// Refocuses so far, oldest first; the last one shapes the live view.
    #[serde(default)]
    pub compactions: Vec<Compaction>,
    /// "name@hash" of the expert prefix this thread was born with.
    #[serde(default)]
    pub expert: Option<String>,
    /// The tools this thread may call besides the plain built-ins (sleeps, recall, refocus, experts); None = all.
    #[serde(default)]
    pub tools: Option<Vec<String>>,
    /// Set on a goldfish: the elephant's thread and the call its answer returns as.
    #[serde(default)]
    pub goldfish_of: Option<GoldfishOf>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct GoldfishOf {
    pub thread: String,
    pub call: String,
    #[serde(default)]
    pub provider_call_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Reply {
    pub seq: u64,
    pub thread: String,
    pub text: String,
    #[serde(default)]
    pub reply_to: Option<String>,
    #[serde(default = "default_kind")]
    pub kind: String,
    pub metadata: Value,
    pub created_ms: u64,
}
fn default_kind() -> String { "text".into() }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Invocation {
    pub call: String,
    pub thread: String,
    pub endpoint: String,
    pub body: Value,
    pub attempts: u32,
    pub created_ms: u64,
    /// Not posted again before this time (backoff after a failed attempt).
    #[serde(default)]
    pub next_ms: u64,
}

/// Everything one slice changes, committed in one transaction after the model answered.
pub struct Commit<'a> {
    pub thread: &'a str,
    pub meta: ThreadMeta,
    pub messages: Vec<Value>,
    pub done: Vec<u64>,
    pub done_ids: Vec<String>,
    pub replies: Vec<(String, Option<String>, String, Value)>,
    pub events: Vec<Event>,
    pub invocations: Vec<Invocation>,
}

pub struct Store {
    db: Database,
}

pub fn message_kind(m: &Value) -> &'static str {
    match m.get("role").and_then(|r| r.as_str()) {
        Some("user") => "user",
        Some("tool") => "tool_result",
        Some("assistant") if m.get("tool_calls").and_then(|t| t.as_array()).is_some_and(|t| !t.is_empty()) => "tool_call",
        Some("assistant") if m.get("content").and_then(|c| c.as_str()).is_some_and(|c| !c.trim().is_empty()) => "assistant_text",
        Some("assistant") => "assistant_empty",
        _ => "other",
    }
}

fn enc<T: Serialize>(v: &T) -> Vec<u8> {
    serde_json::to_vec(v).expect("serializable")
}
fn dec<T: DeserializeOwned>(b: &[u8]) -> Result<T> {
    serde_json::from_slice(b).context("decoding a stored value")
}

impl Store {
    pub fn open(dir: &Path) -> Result<Store> {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        let db = Database::create(dir.join("freefitty.redb")).context("opening the data file")?;
        let w = db.begin_write()?;
        {
            w.open_table(THREADS)?;
            w.open_table(MESSAGES)?;
            w.open_table(EVENTS)?;
            w.open_table(PROCESSED)?;
            w.open_table(DLQ)?;
            w.open_table(REPLIES)?;
            w.open_table(INVOCATIONS)?;
            w.open_table(TOOLSETS)?;
            w.open_table(EXPERTS)?;
            w.open_table(META)?;
        }
        w.commit()?;
        Ok(Store { db })
    }

    fn next_seq(meta: &mut redb::Table<&str, u64>) -> Result<u64> {
        let n = meta.get("seq")?.map(|v| v.value()).unwrap_or(0) + 1;
        meta.insert("seq", n)?;
        Ok(n)
    }

    // ---- threads ---------------------------------------------------------------------------------

    pub fn thread(&self, id: &str) -> Result<Option<ThreadMeta>> {
        let r = self.db.begin_read()?;
        let t = r.open_table(THREADS)?;
        Ok(match t.get(id)? { Some(v) => Some(dec(v.value())?), None => None })
    }

    pub fn threads(&self) -> Result<Vec<(String, ThreadMeta)>> {
        let r = self.db.begin_read()?;
        let t = r.open_table(THREADS)?;
        let mut out = Vec::new();
        for e in t.iter()? {
            let (k, v) = e?;
            out.push((k.value().to_owned(), dec(v.value())?));
        }
        Ok(out)
    }

    pub fn messages(&self, thread: &str) -> Result<Vec<Value>> {
        let r = self.db.begin_read()?;
        let t = r.open_table(MESSAGES)?;
        let mut out = Vec::new();
        for e in t.range((thread, 0)..(thread, u64::MAX))? {
            let (_, v) = e?;
            out.push(dec(v.value())?);
        }
        Ok(out)
    }

    /// Forget a thread: its messages, meta, replies and queued events. Dead letters stay until dropped.
    pub fn delete_thread(&self, thread: &str) -> Result<()> {
        let w = self.db.begin_write()?;
        {
            let mut t = w.open_table(THREADS)?;
            t.remove(thread)?;
            let mut m = w.open_table(MESSAGES)?;
            let keys: Vec<u64> = m.range((thread, 0)..(thread, u64::MAX))?.map(|e| e.map(|(k, _)| k.value().1)).collect::<Result<_, _>>()?;
            for k in keys { m.remove((thread, k))?; }
            let mut ev = w.open_table(EVENTS)?;
            let seqs: Vec<u64> = ev.iter()?.filter_map(|e| e.ok()).filter(|(_, v)| dec::<Event>(v.value()).map(|e| e.thread == thread).unwrap_or(false)).map(|(k, _)| k.value()).collect();
            for s in seqs { ev.remove(s)?; }
            let mut rp = w.open_table(REPLIES)?;
            let seqs: Vec<u64> = rp.iter()?.filter_map(|e| e.ok()).filter(|(_, v)| dec::<Reply>(v.value()).map(|r| r.thread == thread).unwrap_or(false)).map(|(k, _)| k.value()).collect();
            for s in seqs { rp.remove(s)?; }
            // its dead letters and the invocations still to post, whose callbacks would otherwise recreate it
            let mut dlq = w.open_table(DLQ)?;
            let seqs: Vec<u64> = dlq.iter()?.filter_map(|e| e.ok()).filter(|(_, v)| dec::<Event>(v.value()).map(|e| e.thread == thread).unwrap_or(false)).map(|(k, _)| k.value()).collect();
            for s in seqs { dlq.remove(s)?; }
            let mut inv = w.open_table(INVOCATIONS)?;
            let calls: Vec<String> = inv.iter()?.filter_map(|e| e.ok()).filter(|(_, v)| dec::<Invocation>(v.value()).map(|i| i.thread == thread).unwrap_or(false)).map(|(k, _)| k.value().to_owned()).collect();
            for c in calls { inv.remove(c.as_str())?; }
        }
        w.commit()?;
        Ok(())
    }

    // ---- the queue -------------------------------------------------------------------------------

    /// Queue an event; its seq. A `metadata` on a first user message becomes the thread's.
    pub fn push(&self, mut ev: Event) -> Result<u64> {
        let w = self.db.begin_write()?;
        let seq;
        {
            let mut meta = w.open_table(META)?;
            seq = Self::next_seq(&mut meta)?;
            if ev.created_ms == 0 { ev.created_ms = now_ms(); }
            let mut threads = w.open_table(THREADS)?;
            if threads.get(ev.thread.as_str())?.is_none() {
                let md = match &ev.kind { EventKind::User { metadata, .. } => metadata.clone(), _ => Value::Null };
                let t = ThreadMeta { created_ms: now_ms(), updated_ms: now_ms(), metadata: md, ..Default::default() };
                threads.insert(ev.thread.as_str(), enc(&t).as_slice())?;
            }
            let mut events = w.open_table(EVENTS)?;
            events.insert(seq, enc(&ev).as_slice())?;
        }
        w.commit()?;
        Ok(seq)
    }

    /// Give a thread that has no messages yet a history (a migration from another host). Its meta counts
    /// the messages like a commit would; a trailing assistant tool call is not pending (no result is coming).
    pub fn import_messages(&self, thread: &str, msgs: &[Value]) -> Result<()> {
        let w = self.db.begin_write()?;
        {
            let mut threads = w.open_table(THREADS)?;
            let mut m: ThreadMeta = threads.get(thread)?.map(|v| dec(v.value())).transpose()?.unwrap_or_default();
            if m.messages > 0 { bail!("thread {thread} already has {} messages", m.messages); }
            if m.created_ms == 0 { m.created_ms = now_ms(); }
            let mut messages = w.open_table(MESSAGES)?;
            for msg in msgs {
                let b = enc(msg);
                messages.insert((thread, m.messages), b.as_slice())?;
                m.messages += 1;
                m.bytes += b.len() as u64;
                m.last = message_kind(msg).to_owned();
            }
            m.updated_ms = now_ms();
            threads.insert(thread, enc(&m).as_slice())?;
        }
        w.commit()?;
        Ok(())
    }

    /// Create a thread with this meta unless it exists; whether it was created.
    pub fn new_thread(&self, thread: &str, meta: &ThreadMeta) -> Result<bool> {
        let w = self.db.begin_write()?;
        let created;
        {
            let mut threads = w.open_table(THREADS)?;
            created = threads.get(thread)?.is_none();
            if created { threads.insert(thread, enc(meta).as_slice())?; }
        }
        w.commit()?;
        Ok(created)
    }

    /// Create a thread pinned to an expert version before its first event is queued.
    pub fn new_thread_with_expert(&self, thread: &str, key: &str, first: &Event) -> Result<()> {
        let w = self.db.begin_write()?;
        {
            let mut threads = w.open_table(THREADS)?;
            if threads.get(thread)?.is_none() {
                let md = match &first.kind { EventKind::User { metadata, .. } => metadata.clone(), _ => Value::Null };
                let t = ThreadMeta { created_ms: now_ms(), updated_ms: now_ms(), metadata: md, expert: Some(key.to_owned()), ..Default::default() };
                threads.insert(thread, enc(&t).as_slice())?;
            }
        }
        w.commit()?;
        Ok(())
    }

    pub fn events(&self) -> Result<Vec<(u64, Event)>> {
        let r = self.db.begin_read()?;
        let t = r.open_table(EVENTS)?;
        let mut out = Vec::new();
        for e in t.iter()? {
            let (k, v) = e?;
            out.push((k.value(), dec(v.value())?));
        }
        Ok(out)
    }

    /// Forget processed ids older than `before_ms` (a duplicate of an event that old is not coming). Rows written
    /// before 2026-11-07 hold a seq, which reads as 1970 and goes with the first sweep.
    pub fn expire_processed(&self, before_ms: u64) -> Result<usize> {
        let w = self.db.begin_write()?;
        let n;
        {
            let mut t = w.open_table(PROCESSED)?;
            let old: Vec<String> = t.iter()?.filter_map(|e| e.ok()).filter(|(_, v)| v.value() < before_ms).map(|(k, _)| k.value().to_owned()).collect();
            n = old.len();
            for k in old { t.remove(k.as_str())?; }
        }
        w.commit()?;
        Ok(n)
    }

    pub fn is_processed(&self, id: &str) -> Result<bool> {
        let r = self.db.begin_read()?;
        Ok(r.open_table(PROCESSED)?.get(id)?.is_some())
    }

    /// A slice failed: count the attempt and set when it runs again. Past `max`, the events dead-letter.
    /// Returns the events that dead-lettered.
    pub fn failed(&self, seqs: &[u64], retry_ms: u64, max: u32) -> Result<Vec<Event>> {
        let w = self.db.begin_write()?;
        let mut dead = Vec::new();
        {
            let mut events = w.open_table(EVENTS)?;
            let mut dlq = w.open_table(DLQ)?;
            for &s in seqs {
                let Some(v) = events.get(s)? else { continue };
                let mut ev: Event = dec(v.value())?;
                drop(v);
                ev.attempts += 1;
                if ev.attempts >= max {
                    events.remove(s)?;
                    dlq.insert(s, enc(&ev).as_slice())?;
                    dead.push(ev);
                } else {
                    ev.due_ms = now_ms() + retry_ms;
                    events.insert(s, enc(&ev).as_slice())?;
                }
            }
        }
        w.commit()?;
        Ok(dead)
    }

    /// Dead-letter at once (an error no retry fixes); the thread is closed with the reason.
    pub fn bury(&self, seqs: &[u64], thread: &str, reason: &str) -> Result<Vec<Event>> {
        let w = self.db.begin_write()?;
        let mut dead = Vec::new();
        {
            let mut events = w.open_table(EVENTS)?;
            let mut dlq = w.open_table(DLQ)?;
            for &s in seqs {
                let Some(v) = events.get(s)? else { continue };
                let ev: Event = dec(v.value())?;
                drop(v);
                events.remove(s)?;
                dlq.insert(s, enc(&ev).as_slice())?;
                dead.push(ev);
            }
            let mut threads = w.open_table(THREADS)?;
            let found = threads.get(thread)?.map(|v| dec::<ThreadMeta>(v.value())).transpose()?;
            if let Some(mut m) = found {
                m.closed = Some(reason.to_owned());
                threads.insert(thread, enc(&m).as_slice())?;
            }
        }
        w.commit()?;
        Ok(dead)
    }

    pub fn dlq(&self) -> Result<Vec<(u64, Event)>> {
        let r = self.db.begin_read()?;
        let t = r.open_table(DLQ)?;
        let mut out = Vec::new();
        for e in t.iter()? {
            let (k, v) = e?;
            out.push((k.value(), dec(v.value())?));
        }
        Ok(out)
    }

    /// Put dead letters back on the queue, with their attempts reset and their ids kept (a redriven event
    /// whose turn already ran is skipped by `processed`). Returns the events requeued.
    pub fn redrive(&self, thread: Option<&str>) -> Result<Vec<(u64, Event)>> {
        let w = self.db.begin_write()?;
        let mut out = Vec::new();
        {
            let mut dlq = w.open_table(DLQ)?;
            let mut events = w.open_table(EVENTS)?;
            let mut meta = w.open_table(META)?;
            let mut threads = w.open_table(THREADS)?;
            let all: Vec<(u64, Event)> = dlq.iter()?.map(|e| e.map(|(k, v)| (k.value(), dec::<Event>(v.value())))).collect::<Result<Vec<_>, _>>()?.into_iter().map(|(k, v)| v.map(|v| (k, v))).collect::<Result<_>>()?;
            for (s, mut ev) in all {
                if thread.is_some_and(|t| t != ev.thread) { continue; }
                dlq.remove(s)?;
                ev.attempts = 0;
                ev.due_ms = 0;
                let seq = Self::next_seq(&mut meta)?;
                events.insert(seq, enc(&ev).as_slice())?;
                let found = threads.get(ev.thread.as_str())?.map(|v| dec::<ThreadMeta>(v.value())).transpose()?;
                if let Some(mut m) = found {
                    if m.closed.is_some() {
                        m.closed = None;
                        threads.insert(ev.thread.as_str(), enc(&m).as_slice())?;
                    }
                }
                out.push((seq, ev));
            }
        }
        w.commit()?;
        Ok(out)
    }

    pub fn drop_dlq(&self, thread: Option<&str>) -> Result<usize> {
        let w = self.db.begin_write()?;
        let mut n = 0;
        {
            let mut dlq = w.open_table(DLQ)?;
            let seqs: Vec<u64> = dlq.iter()?.filter_map(|e| e.ok()).filter(|(_, v)| thread.is_none_or(|t| dec::<Event>(v.value()).map(|e| e.thread == t).unwrap_or(false))).map(|(k, _)| k.value()).collect();
            for s in seqs { dlq.remove(s)?; n += 1; }
        }
        w.commit()?;
        Ok(n)
    }

    // ---- the slice's commit ----------------------------------------------------------------------

    /// Apply a finished slice in one transaction: the new messages, the events it consumed (and their ids as
    /// processed), the replies it produced, the events it schedules (wake-ups) and the tool invocations to
    /// dispatch. Returns the seqs of the new events and replies.
    pub fn commit(&self, c: Commit<'_>) -> Result<(Vec<u64>, Vec<u64>)> {
        let w = self.db.begin_write()?;
        let mut new_events = Vec::new();
        let mut new_replies = Vec::new();
        {
            let mut meta = w.open_table(META)?;
            let mut threads = w.open_table(THREADS)?;
            let mut messages = w.open_table(MESSAGES)?;
            let mut events = w.open_table(EVENTS)?;
            let mut processed = w.open_table(PROCESSED)?;
            let mut replies = w.open_table(REPLIES)?;
            let mut invocations = w.open_table(INVOCATIONS)?;
            let mut m = c.meta;
            for msg in &c.messages {
                let b = enc(msg);
                messages.insert((c.thread, m.messages), b.as_slice())?;
                m.messages += 1;
                m.bytes += b.len() as u64;
                m.last = message_kind(msg).to_owned();
            }
            m.updated_ms = now_ms();
            threads.insert(c.thread, enc(&m).as_slice())?;
            for s in &c.done { events.remove(*s)?; }
            let ran_at = now_ms();
            for id in &c.done_ids { processed.insert(id.as_str(), ran_at)?; }
            for (text, reply_to, kind, extra) in c.replies {
                let seq = Self::next_seq(&mut meta)?;
                let r = Reply { seq, thread: c.thread.to_owned(), text, reply_to, kind, metadata: extra, created_ms: now_ms() };
                replies.insert(seq, enc(&r).as_slice())?;
                new_replies.push(seq);
            }
            for mut ev in c.events {
                let seq = Self::next_seq(&mut meta)?;
                if ev.created_ms == 0 { ev.created_ms = now_ms(); }
                events.insert(seq, enc(&ev).as_slice())?;
                new_events.push(seq);
            }
            for inv in &c.invocations {
                invocations.insert(inv.call.as_str(), enc(inv).as_slice())?;
            }
        }
        w.commit()?;
        Ok((new_events, new_replies))
    }

    // ---- replies ---------------------------------------------------------------------------------

    pub fn replies(&self, thread: Option<&str>) -> Result<Vec<Reply>> {
        let r = self.db.begin_read()?;
        let t = r.open_table(REPLIES)?;
        let mut out = Vec::new();
        for e in t.iter()? {
            let (_, v) = e?;
            let rp: Reply = dec(v.value())?;
            if thread.is_none_or(|th| th == rp.thread) { out.push(rp); }
        }
        Ok(out)
    }

    /// A reply outside a slice: an error the operator must see.
    pub fn push_reply(&self, thread: &str, text: &str, reply_to: Option<&str>, kind: &str) -> Result<u64> {
        let w = self.db.begin_write()?;
        let seq;
        {
            let mut meta = w.open_table(META)?;
            seq = Self::next_seq(&mut meta)?;
            let md = w.open_table(THREADS)?.get(thread)?.map(|v| dec::<ThreadMeta>(v.value()).map(|m| m.metadata).unwrap_or(Value::Null)).unwrap_or(Value::Null);
            let r = Reply { seq, thread: thread.to_owned(), text: text.to_owned(), reply_to: reply_to.map(str::to_owned), kind: kind.to_owned(), metadata: md, created_ms: now_ms() };
            w.open_table(REPLIES)?.insert(seq, enc(&r).as_slice())?;
        }
        w.commit()?;
        Ok(seq)
    }

    pub fn remove_reply(&self, seq: u64) -> Result<()> {
        let w = self.db.begin_write()?;
        w.open_table(REPLIES)?.remove(seq)?;
        w.commit()?;
        Ok(())
    }

    pub fn expire_replies(&self, older_than_ms: u64) -> Result<usize> {
        let w = self.db.begin_write()?;
        let mut n = 0;
        {
            let mut t = w.open_table(REPLIES)?;
            let seqs: Vec<u64> = t.iter()?.filter_map(|e| e.ok()).filter(|(_, v)| dec::<Reply>(v.value()).map(|r| r.created_ms < older_than_ms).unwrap_or(true)).map(|(k, _)| k.value()).collect();
            for s in seqs { t.remove(s)?; n += 1; }
        }
        w.commit()?;
        Ok(n)
    }

    // ---- invocations and toolsets ----------------------------------------------------------------

    pub fn invocations(&self) -> Result<Vec<Invocation>> {
        let r = self.db.begin_read()?;
        let t = r.open_table(INVOCATIONS)?;
        let mut out = Vec::new();
        for e in t.iter()? {
            let (_, v) = e?;
            out.push(dec(v.value())?);
        }
        Ok(out)
    }

    pub fn invocation_done(&self, call: &str) -> Result<()> {
        let w = self.db.begin_write()?;
        w.open_table(INVOCATIONS)?.remove(call)?;
        w.commit()?;
        Ok(())
    }

    pub fn invocation_attempt(&self, call: &str, attempts: u32, next_ms: u64) -> Result<()> {
        let w = self.db.begin_write()?;
        {
            let mut t = w.open_table(INVOCATIONS)?;
            let found = t.get(call)?.map(|v| dec::<Invocation>(v.value())).transpose()?;
            if let Some(mut inv) = found {
                inv.attempts = attempts;
                inv.next_ms = next_ms;
                t.insert(call, enc(&inv).as_slice())?;
            }
        }
        w.commit()?;
        Ok(())
    }

    pub fn toolset(&self, url: &str) -> Result<Option<Value>> {
        let r = self.db.begin_read()?;
        let t = r.open_table(TOOLSETS)?;
        Ok(match t.get(url)? { Some(v) => Some(dec(v.value())?), None => None })
    }

    pub fn save_toolset(&self, url: &str, manifest: &Value) -> Result<()> {
        let w = self.db.begin_write()?;
        w.open_table(TOOLSETS)?.insert(url, enc(manifest).as_slice())?;
        w.commit()?;
        Ok(())
    }

    // ---- compaction and experts -------------------------------------------------------------------

    /// Record a refocus: its own transaction, touching only the thread's compactions and node, so a crash
    /// between it and the slice's commit leaves a consistent thread (the events are still queued; the
    /// re-run finds the compacted view and does not summarize again).
    pub fn compact(&self, thread: &str, c: &Compaction) -> Result<()> {
        let w = self.db.begin_write()?;
        {
            let mut threads = w.open_table(THREADS)?;
            let found = threads.get(thread)?.map(|v| dec::<ThreadMeta>(v.value())).transpose()?;
            if let Some(mut m) = found {
                m.compactions.push(c.clone());
                m.node = None;
                threads.insert(thread, enc(&m).as_slice())?;
            }
        }
        w.commit()?;
        Ok(())
    }

    pub fn expert(&self, key: &str) -> Result<Option<Expert>> {
        let r = self.db.begin_read()?;
        let t = r.open_table(EXPERTS)?;
        Ok(match t.get(key)? { Some(v) => Some(dec(v.value())?), None => None })
    }

    pub fn experts(&self) -> Result<Vec<Expert>> {
        let r = self.db.begin_read()?;
        let t = r.open_table(EXPERTS)?;
        let mut out = Vec::new();
        for e in t.iter()? {
            let (_, v) = e?;
            out.push(dec(v.value())?);
        }
        Ok(out)
    }

    /// Store a version; the key. Same bytes, same key: a no-op.
    pub fn save_expert(&self, e: &Expert) -> Result<String> {
        let key = format!("{}@{}", e.name, e.hash);
        let w = self.db.begin_write()?;
        {
            let mut t = w.open_table(EXPERTS)?;
            if t.get(key.as_str())?.is_none() {
                t.insert(key.as_str(), enc(e).as_slice())?;
            }
        }
        w.commit()?;
        Ok(key)
    }

    pub fn counts(&self) -> Result<Value> {
        let r = self.db.begin_read()?;
        Ok(serde_json::json!({
            "threads": r.open_table(THREADS)?.len()?,
            "queued": r.open_table(EVENTS)?.len()?,
            "dlq": r.open_table(DLQ)?.len()?,
            "replies": r.open_table(REPLIES)?.len()?,
            "invocations": r.open_table(INVOCATIONS)?.len()?,
        }))
    }
}
