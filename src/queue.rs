//! The scheduler: which thread runs next.
//!
//! Every queued event is in the store and mirrored here in seq order. A slice takes all of one thread's
//! due events; a thread with a slice in flight is skipped, so a thread never runs twice at once, and a
//! timer is just an event whose due time has not come. Threads run concurrently up to `MAX_SLICES`; the
//! router holds a turn while the nodes are full. While no node is up nothing starts, so an outage costs
//! no delivery attempts.

use crate::engine::{Engine, Outcome};
use crate::store::{now_ms, Event, Store};
use anyhow::Result;
use std::collections::{BTreeMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::Notify;

const MAX_SLICES: usize = 64;

/// One thread's due events, in order.
type Batch = (String, Vec<(u64, Event)>);

pub struct Queue {
    store: Arc<Store>,
    engine: Arc<Engine>,
    events: Mutex<BTreeMap<u64, Event>>,
    inflight: Mutex<HashSet<String>>,
    running: AtomicUsize,
    pub notify: Notify,
    /// Replies landed: /output readers wake.
    pub replies: Arc<Notify>,
    /// Invocations committed: the dispatcher posts them.
    pub kick: Box<dyn Fn() + Send + Sync>,
    stopping: std::sync::atomic::AtomicBool,
}

impl Queue {
    pub fn new(store: Arc<Store>, engine: Arc<Engine>, replies: Arc<Notify>, kick: Box<dyn Fn() + Send + Sync>) -> Result<Arc<Queue>> {
        let events: BTreeMap<u64, Event> = store.events()?.into_iter().collect();
        tracing::info!("queue: {} event(s) waiting", events.len());
        Ok(Arc::new(Queue {
            store,
            engine,
            events: Mutex::new(events),
            inflight: Mutex::new(HashSet::new()),
            running: AtomicUsize::new(0),
            notify: Notify::new(),
            replies,
            kick,
            stopping: std::sync::atomic::AtomicBool::new(false),
        }))
    }

    /// Queue an event; its seq, or None when an event with that id already ran (a tool server posting a
    /// result twice).
    pub fn push(&self, mut ev: Event) -> Result<Option<u64>> {
        if self.store.is_processed(&ev.id)? {
            return Ok(None);
        }
        // the same result posted twice before a slice picked it: the second copy must not queue either
        if self.events.lock().unwrap().values().any(|e| e.id == ev.id) {
            return Ok(None);
        }
        if ev.created_ms == 0 { ev.created_ms = now_ms(); }   // the mirror's copy ages from now too
        let seq = self.store.push(ev.clone())?;
        self.events.lock().unwrap().insert(seq, ev);
        self.notify.notify_one();
        Ok(Some(seq))
    }

    pub fn requeue(&self, items: Vec<(u64, Event)>) {
        let mut ev = self.events.lock().unwrap();
        for (s, e) in items { ev.insert(s, e); }
        drop(ev);
        self.notify.notify_one();
    }

    /// A slice of this thread is running right now.
    pub fn is_inflight(&self, thread: &str) -> bool {
        self.inflight.lock().unwrap().contains(thread)
    }

    pub fn forget_thread(&self, thread: &str) {
        self.events.lock().unwrap().retain(|_, e| e.thread != thread);
    }

    pub fn stats(&self) -> serde_json::Value {
        let ev = self.events.lock().unwrap();
        let inflight = self.inflight.lock().unwrap();
        let now = now_ms();
        let age = |e: &Event| now.saturating_sub(e.due_ms.max(e.created_ms)) / 1000;
        let due = || ev.values().filter(|e| e.due_ms <= now);
        // a due event of a thread whose slice is running waits for that slice, as it must; one of an idle
        // thread waits for nothing but a start: starvation shows in oldest_runnable_s, not in a green status
        let runnable = || due().filter(|e| !inflight.contains(&e.thread));
        serde_json::json!({
            "queued": ev.len(),
            "due": due().count(),
            "oldest_due_s": due().map(age).max().unwrap_or(0),
            "runnable": runnable().count(),
            "oldest_runnable_s": runnable().map(age).max().unwrap_or(0),
            "running": self.running.load(Ordering::Relaxed),
            "inflight": inflight.len(),
        })
    }

    pub fn stop(&self) {
        self.stopping.store(true, Ordering::Relaxed);
        self.notify.notify_one();
    }

    pub fn is_stopping(&self) -> bool {
        self.stopping.load(Ordering::Relaxed)
    }

    pub fn running(&self) -> usize {
        self.running.load(Ordering::Relaxed)
    }

    pub async fn run(self: Arc<Self>) {
        loop {
            if self.stopping.load(Ordering::Relaxed) { return; }
            if !self.engine.router.any_up().await {
                let _ = tokio::time::timeout(Duration::from_secs(1), self.notify.notified()).await;
                continue;
            }
            let now = now_ms();
            let (batches, next_due) = self.pick(now);
            for (thread, batch) in batches {
                self.running.fetch_add(1, Ordering::Relaxed);
                let me = self.clone();
                tokio::spawn(async move { me.slice(thread, batch).await });
            }
            let wait = next_due.map(|d| Duration::from_millis(d.saturating_sub(now_ms()).clamp(10, 1000))).unwrap_or(Duration::from_secs(1));
            let _ = tokio::time::timeout(wait, self.notify.notified()).await;
        }
    }

    /// One batch per thread with due events and no slice in flight, in seq order; and the earliest due time
    /// among what stays.
    fn pick(&self, now: u64) -> (Vec<Batch>, Option<u64>) {
        let events = self.events.lock().unwrap();
        let mut inflight = self.inflight.lock().unwrap();
        let room = MAX_SLICES.saturating_sub(self.running.load(Ordering::Relaxed));
        let mut batches: Vec<Batch> = Vec::new();
        let mut skipped: HashSet<String> = HashSet::new();
        let mut next_due: Option<u64> = None;
        for (seq, ev) in events.iter() {
            if inflight.contains(&ev.thread) || skipped.contains(&ev.thread) { continue; }
            if ev.due_ms > now {
                next_due = Some(next_due.map_or(ev.due_ms, |d| d.min(ev.due_ms)));
                // a timer lets later events of its thread run first; a failed event waiting for its retry
                // holds the thread, so what came after it never runs ahead of it
                if ev.attempts > 0 { skipped.insert(ev.thread.clone()); }
                continue;
            }
            if let Some(b) = batches.iter_mut().find(|(t, _)| t == &ev.thread) {
                b.1.push((*seq, ev.clone()));
            } else if batches.len() < room {
                batches.push((ev.thread.clone(), vec![(*seq, ev.clone())]));
            } else {
                skipped.insert(ev.thread.clone());
            }
        }
        for (t, _) in &batches { inflight.insert(t.clone()); }
        (batches, next_due)
    }

    async fn slice(&self, thread: String, batch: Vec<(u64, Event)>) {
        // the in-flight mark and the running count are released however this ends, a panic included
        struct Release<'a>(&'a Queue, &'a str);
        impl Drop for Release<'_> {
            fn drop(&mut self) {
                self.0.inflight.lock().unwrap().remove(self.1);
                self.0.running.fetch_sub(1, Ordering::Relaxed);
                self.0.notify.notify_one();
            }
        }
        let _release = Release(self, &thread);
        let seqs: Vec<u64> = batch.iter().map(|(s, _)| *s).collect();
        let reply_to = batch.iter().rev().find_map(|(_, e)| match e.kind { crate::store::EventKind::User { .. } => Some(e.id.clone()), _ => None });
        // no clock here: `slice_timeout_s` bounds each request to a node (the router's client), and the wait for a
        // free slot is not the thread's fault. Before 2026-11-07 a slice that queued behind long prompts for 15 min
        // failed "slice timed out" three times and dead-lettered; `/health` oldest_due_s is where the wait shows.
        let outcome = self.engine.slice(&thread, &batch).await;
        let result = match outcome {
            Outcome::Done => {
                let mut ev = self.events.lock().unwrap();
                for s in &seqs { ev.remove(s); }
                // wake-ups and replies the slice committed
                drop(ev);
                self.reload_new();
                (self.kick)();
                self.replies.notify_waiters();
                Ok(())
            }
            Outcome::Transient(why) => {
                tracing::warn!("{thread}: slice failed ({why})");
                match self.store.failed(&seqs, self.engine.cfg.retry_after_s * 1000, self.engine.cfg.max_attempts) {
                    Ok(dead) => {
                        if !dead.is_empty() {
                            tracing::error!("{thread}: {} event(s) dead-lettered after {} attempts", dead.len(), self.engine.cfg.max_attempts);
                            let _ = self.store.push_reply(&thread, &format!("Error: this turn failed {} times and was set aside ({why}). `freefitty dlq redrive` runs it again.", self.engine.cfg.max_attempts), reply_to.as_deref(), "error");
                            self.replies.notify_waiters();
                            let mut ev = self.events.lock().unwrap();
                            for s in &seqs { ev.remove(s); }
                        }
                        self.reload_due(&seqs);
                        Ok(())
                    }
                    Err(e) => Err(e),
                }
            }
            Outcome::Fatal(why) => {
                tracing::error!("{thread}: {why}; its events are set aside and the thread is closed");
                let r = self.store.bury(&seqs, &thread, &why).map(|_| ());
                let _ = self.store.push_reply(&thread, &format!("Error: {why}. The thread is closed; start a new one."), reply_to.as_deref(), "error");
                self.replies.notify_waiters();
                let mut ev = self.events.lock().unwrap();
                for s in &seqs { ev.remove(s); }
                r
            }
        };
        if let Err(e) = result { tracing::error!("{thread}: store failed after the slice: {e:#}"); }
    }

    /// Mirror events the store has that the map does not (the slice's new wake-ups).
    fn reload_new(&self) {
        if let Ok(all) = self.store.events() {
            let mut ev = self.events.lock().unwrap();
            for (s, e) in all {
                ev.entry(s).or_insert(e);
            }
        }
    }

    /// Mirror the retried events' new due times and attempt counts.
    fn reload_due(&self, seqs: &[u64]) {
        if let Ok(all) = self.store.events() {
            let mut ev = self.events.lock().unwrap();
            for (s, e) in all {
                if seqs.contains(&s) { ev.insert(s, e); }
            }
        }
    }
}
