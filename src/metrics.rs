//! Counters and histograms for `GET /metrics` (Prometheus text exposition, version 0.0.4): what a scraper needs
//! without parsing logs. Slices by outcome and how long they took; per node the turns, prompt tokens and the tokens
//! its cache served (their ratio is the cache hit rate); how long a turn ran and how long it waited for a slot. The
//! router, queue and store numbers are read live when the page is rendered.

use std::collections::BTreeMap;
use std::fmt::Write;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::Mutex;

/// A cumulative histogram with fixed upper bounds in seconds.
pub struct Hist {
    bounds: &'static [f64],
    counts: Vec<AtomicU64>,
    sum_ms: AtomicU64,
    n: AtomicU64,
}

impl Hist {
    fn new(bounds: &'static [f64]) -> Hist {
        Hist { bounds, counts: bounds.iter().map(|_| AtomicU64::new(0)).collect(), sum_ms: AtomicU64::new(0), n: AtomicU64::new(0) }
    }

    pub fn observe(&self, secs: f64) {
        for (b, c) in self.bounds.iter().zip(&self.counts) {
            if secs <= *b { c.fetch_add(1, Relaxed); }
        }
        self.sum_ms.fetch_add((secs * 1000.0) as u64, Relaxed);
        self.n.fetch_add(1, Relaxed);
    }

    fn render(&self, out: &mut String, name: &str, help: &str) {
        let _ = writeln!(out, "# HELP {name} {help}\n# TYPE {name} histogram");
        for (b, c) in self.bounds.iter().zip(&self.counts) {
            let _ = writeln!(out, "{name}_bucket{{le=\"{b}\"}} {}", c.load(Relaxed));
        }
        let n = self.n.load(Relaxed);
        let _ = writeln!(out, "{name}_bucket{{le=\"+Inf\"}} {n}\n{name}_sum {}\n{name}_count {n}", self.sum_ms.load(Relaxed) as f64 / 1000.0);
    }
}

const SLICE_S: &[f64] = &[1.0, 5.0, 15.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1800.0];
const TURN_S: &[f64] = &[0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0];
const WAIT_S: &[f64] = &[0.1, 1.0, 5.0, 15.0, 30.0, 60.0, 120.0, 300.0, 900.0];

#[derive(Default, Clone, Copy)]
struct NodeTotals {
    turns: u64,
    prompt_tokens: u64,
    cached_tokens: u64,
}

pub struct Metrics {
    slices: Mutex<BTreeMap<&'static str, u64>>,
    pub slice_seconds: Hist,
    pub turn_seconds: Hist,
    pub slot_wait_seconds: Hist,
    nodes: Mutex<BTreeMap<String, NodeTotals>>,
}

impl Default for Metrics {
    fn default() -> Self {
        Metrics { slices: Mutex::default(), slice_seconds: Hist::new(SLICE_S), turn_seconds: Hist::new(TURN_S), slot_wait_seconds: Hist::new(WAIT_S), nodes: Mutex::default() }
    }
}

impl Metrics {
    /// A slice ended: "done", "transient", "fatal" or "panic", after `secs`.
    pub fn slice(&self, outcome: &'static str, secs: f64) {
        *self.slices.lock().unwrap().entry(outcome).or_default() += 1;
        self.slice_seconds.observe(secs);
    }

    /// A node answered a turn.
    pub fn turn(&self, node: &str, prompt: Option<u64>, cached: Option<u64>, ran_ms: u64, waited_ms: u64) {
        let mut n = self.nodes.lock().unwrap();
        let t = n.entry(node.to_owned()).or_default();
        t.turns += 1;
        t.prompt_tokens += prompt.unwrap_or(0);
        t.cached_tokens += cached.unwrap_or(0);
        drop(n);
        self.turn_seconds.observe(ran_ms as f64 / 1000.0);
        self.slot_wait_seconds.observe(waited_ms as f64 / 1000.0);
    }

    /// The page: these counters, then `gauges` (name, help, value) and `counters` read live by the caller.
    pub fn render(&self, build: &str, counters: &[(String, f64)], gauges: &[(String, f64)]) -> String {
        let mut o = String::new();
        let _ = writeln!(o, "# HELP freefitty_build_info The running binary.\n# TYPE freefitty_build_info gauge\nfreefitty_build_info{{version=\"{build}\"}} 1");
        let _ = writeln!(o, "# HELP freefitty_slices_total Slices that ended, by outcome.\n# TYPE freefitty_slices_total counter");
        for (k, v) in self.slices.lock().unwrap().iter() {
            let _ = writeln!(o, "freefitty_slices_total{{outcome=\"{k}\"}} {v}");
        }
        self.slice_seconds.render(&mut o, "freefitty_slice_seconds", "Wall time of a slice, slot waits included.");
        self.turn_seconds.render(&mut o, "freefitty_turn_seconds", "Time a node took to answer a turn.");
        self.slot_wait_seconds.render(&mut o, "freefitty_slot_wait_seconds", "Time a turn waited for a node slot.");
        let nodes = self.nodes.lock().unwrap().clone();
        for (name, help, f) in [("freefitty_node_turns_total", "Turns a node answered.", (|t: &NodeTotals| t.turns) as fn(&NodeTotals) -> u64),
                                ("freefitty_node_prompt_tokens_total", "Prompt tokens sent to a node.", |t| t.prompt_tokens),
                                ("freefitty_node_cached_tokens_total", "Prompt tokens a node's cache served (cached/prompt = hit rate).", |t| t.cached_tokens)] {
            let _ = writeln!(o, "# HELP {name} {help}\n# TYPE {name} counter");
            for (node, t) in &nodes {
                let _ = writeln!(o, "{name}{{node=\"{node}\"}} {}", f(t));
            }
        }
        for (name, v) in counters {
            let _ = writeln!(o, "# TYPE {name} counter\n{name} {v}");
        }
        for (name, v) in gauges {
            let _ = writeln!(o, "# TYPE {name} gauge\n{name} {v}");
        }
        o
    }
}
