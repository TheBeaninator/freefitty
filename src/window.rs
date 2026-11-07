//! The budget window: what keeps a thread's view under the node's context however long the thread lives.
//!
//! The model sees `live_view`: the shared prefix, then the stored history from an index on. Refocusing moves that
//! index by asking the model for notes (engine.rs). The window moves it without asking: when the view would pass
//! `budget` tokens, the oldest whole units leave until it is under `target`, and the index then stays put until
//! the budget is crossed again, so the node's cache sees the same prefix for many turns (one cut = one cache miss,
//! not one a turn). Nothing is removed from the store; a `[Paged.]` notice at the cut names what left and the
//! `recall` tool returns any stored message exactly. A unit starts at an input (a user message or a tool result),
//! so the view never opens with the model's own words, and a tool result whose call left the view gets a stub
//! call in the notice so the template still sees a call before every result.
//!
//! `replay` runs the same cut over a stored history as if it had been live, to measure before deciding.
//! After "The Missing Memory Hierarchy" (arXiv 2603.09023) and pi-context-paging (dad 2026-11-07).

use crate::store::PendingCall;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};

/// Bytes / 4 of one message, as everywhere else.
pub fn weigh(m: &Value) -> u64 {
    (serde_json::to_vec(m).map(|b| b.len()).unwrap_or(0) / 4) as u64
}

fn role(m: &Value) -> &str {
    m.get("role").and_then(|r| r.as_str()).unwrap_or("")
}

/// A view may start here: an input the model answers (a person's words or a tool's result).
pub fn is_boundary(m: &Value) -> bool {
    matches!(role(m), "user" | "tool")
}

/// Advance `from` past whole units until the messages from the new index weigh at most `target` (bytes / 4 times
/// `scale`), never past the newest input and never backwards. Returns the new index.
pub fn cut(history: &[Value], from: usize, target: u64, scale: f64) -> usize {
    let n = history.len();
    if from >= n { return from; }
    let last = (from..n).rev().find(|&i| is_boundary(&history[i])).unwrap_or(from);
    let w = |m: &Value| (weigh(m) as f64 * scale) as u64;
    let mut weight: u64 = history[from..].iter().map(w).sum();
    let mut i = from;
    while weight > target && i < last {
        let next = (i + 1..=last).find(|&j| is_boundary(&history[j])).unwrap_or(last);
        weight = weight.saturating_sub(history[i..next].iter().map(w).sum());
        i = next;
    }
    i
}

/// Stub tool calls for the results in `history[from..]` whose calls are not there: calls that left the view, and
/// pending calls (answers still to come) whose assistant message left. The template needs a call before a result.
pub fn orphan_calls(history: &[Value], from: usize, pending: &[PendingCall]) -> Vec<Value> {
    let tail = &history[from.min(history.len())..];
    let shown: HashSet<&str> = tail
        .iter()
        .flat_map(|m| m.get("tool_calls").and_then(|t| t.as_array()).into_iter().flatten())
        .filter_map(|tc| tc.get("id").and_then(|i| i.as_str()))
        .collect();
    let mut earlier: HashMap<&str, &Value> = HashMap::new();
    for m in &history[..from.min(history.len())] {
        for tc in m.get("tool_calls").and_then(|t| t.as_array()).into_iter().flatten() {
            if let Some(id) = tc.get("id").and_then(|i| i.as_str()) { earlier.insert(id, tc); }
        }
    }
    let mut out = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for m in tail {
        if role(m) != "tool" { continue; }
        let Some(id) = m.get("tool_call_id").and_then(|i| i.as_str()) else { continue };
        if shown.contains(id) || !seen.insert(id.to_owned()) { continue; }
        out.push(match earlier.get(id) {
            Some(tc) => (*tc).clone(),
            None => json!({"id": id, "type": "function", "function": {"name": "unknown", "arguments": "{}"}}),
        });
    }
    for p in pending {
        let id = p.provider_call_id.clone().unwrap_or_else(|| p.id.clone());
        if shown.contains(id.as_str()) || shown.contains(p.id.as_str()) || !seen.insert(id.clone()) { continue; }
        out.push(json!({"id": id, "type": "function", "function": {"name": p.name, "arguments": if p.arguments.is_empty() { "{}".to_owned() } else { p.arguments.clone() }}}));
    }
    out
}

const NOTICE_MSG_CHARS: usize = 300;
const NOTICE_CHARS: usize = 6000;

/// The `[Paged.]` notice: what left the view between `low` (where the stored view begins: 0 or the last
/// refocus) and `from`, with the user messages among it copied in short, and how to get any of it back.
pub fn notice(history: &[Value], low: usize, from: usize) -> String {
    let gone = &history[low.min(from)..from.min(history.len())];
    let tokens: u64 = gone.iter().map(weigh).sum();
    let mut s = format!(
        "[Paged. Messages #{low} to #{} ({} messages, ~{tokens} tokens) left this view to keep it short; every message is stored and \
         recall(indices) returns any of them exactly as written. Continue the current task. The user messages among them:",
        from.saturating_sub(1), gone.len());
    let mut used = 0usize;
    let mut skipped = 0usize;
    for (k, m) in gone.iter().enumerate() {
        if role(m) != "user" { continue; }
        let text: String = m.get("content").and_then(|c| c.as_str()).unwrap_or("").chars().take(NOTICE_MSG_CHARS).collect();
        let line = format!(" #{}: {:?}", low + k, text.trim());
        if used + line.len() > NOTICE_CHARS { skipped += 1; continue; }
        used += line.len();
        s.push_str(&line);
    }
    if skipped > 0 { s.push_str(&format!(" and {skipped} more user messages (recall them by index)")); }
    s.push(']');
    s
}

/// A stored message as `recall` returns it: index, role, the text, calls and results named.
pub fn render(i: usize, m: &Value) -> String {
    let mut s = format!("#{i} {}", role(m));
    if let Some(id) = m.get("tool_call_id").and_then(|v| v.as_str()) { s.push_str(&format!(" (result of {id})")); }
    s.push_str(": ");
    if let Some(c) = m.get("content").and_then(|c| c.as_str()) { s.push_str(c); }
    for tc in m.get("tool_calls").and_then(|t| t.as_array()).into_iter().flatten() {
        s.push_str(&format!("\n  [call {} {}({})]", tc.get("id").and_then(|v| v.as_str()).unwrap_or(""),
            tc.pointer("/function/name").and_then(|v| v.as_str()).unwrap_or(""),
            tc.pointer("/function/arguments").and_then(|v| v.as_str()).unwrap_or("")));
    }
    s
}

pub const RECALL_CHARS: usize = 16 * 1024;

/// The indices a `recall` call asks for: `indices: [..]`, or `from`/`to` (inclusive), each under `len`.
pub fn recall_indices(args: &Value, len: usize) -> Result<Vec<usize>, String> {
    let mut v: Vec<usize> = Vec::new();
    if let Some(a) = args.get("indices").and_then(|a| a.as_array()) {
        for x in a { if let Some(i) = x.as_u64() { v.push(i as usize); } }
    }
    if let Some(f) = args.get("from").and_then(|f| f.as_u64()) {
        let t = args.get("to").and_then(|t| t.as_u64()).unwrap_or(f);
        if t < f { return Err("`to` is before `from`".into()); }
        v.extend((f..=t.min(f + 200)).map(|i| i as usize));
    }
    v.retain(|&i| i < len);
    v.sort_unstable();
    v.dedup();
    if v.is_empty() { return Err(format!("no stored message among those indices; the stored messages are #0 to #{}", len.saturating_sub(1))); }
    Ok(v)
}

/// The text `recall` answers with, cut at `RECALL_CHARS` with a note.
pub fn recall_text(history: &[Value], indices: &[usize]) -> String {
    let mut out = String::new();
    for (k, &i) in indices.iter().enumerate() {
        let r = render(i, &history[i]);
        if out.len() + r.len() + 1 > RECALL_CHARS {
            out.push_str(&format!("\n[{} of the {} messages asked for did not fit in 16 KiB; ask for #{i} onward in a smaller call]", indices.len() - k, indices.len()));
            break;
        }
        if !out.is_empty() { out.push('\n'); }
        out.push_str(&r);
    }
    out
}

/// What a replay of one stored history under a window (and, with `tau > 0`, under tool-result paging) would
/// have done. Faults count later tool calls identical (name and arguments) to one whose text had left the view.
#[derive(Debug, Default, Clone, serde::Serialize)]
pub struct Replay {
    pub messages: usize,
    pub slices: usize,
    /// Bytes / 4 times the scale, of the whole history.
    pub tokens: u64,
    /// The largest view a slice saw without the window, and with it.
    pub peak_unwindowed: u64,
    pub peak_windowed: u64,
    pub cuts: usize,
    /// Messages that left the view over the history, and the token-slices they would have cost resident.
    pub removed: usize,
    pub token_slices_total: u64,
    pub token_slices_saved: u64,
    pub window_faults: usize,
    /// Tool-result paging at `tau` user turns and `min_bytes`.
    pub gc_evicted: usize,
    pub gc_faults: usize,
    pub gc_token_slices_saved: u64,
}

fn call_keys(m: &Value) -> Vec<String> {
    m.get("tool_calls").and_then(|t| t.as_array()).into_iter().flatten()
        .map(|tc| format!("{}|{}", tc.pointer("/function/name").and_then(|v| v.as_str()).unwrap_or(""), tc.pointer("/function/arguments").and_then(|v| v.as_str()).unwrap_or("")))
        .collect()
}

pub fn replay(history: &[Value], budget: u64, target: u64, scale: f64, fixed: u64, tau: usize, min_bytes: u64) -> Replay {
    let mut r = Replay { messages: history.len(), ..Default::default() };
    let w: Vec<u64> = history.iter().map(|m| (weigh(m) as f64 * scale) as u64).collect();
    r.tokens = w.iter().sum();
    // the call behind each result, so an evicted result can be matched to a later identical call
    let mut call_of: HashMap<&str, String> = HashMap::new();
    for m in history {
        for tc in m.get("tool_calls").and_then(|t| t.as_array()).into_iter().flatten() {
            if let Some(id) = tc.get("id").and_then(|i| i.as_str()) {
                call_of.insert(id, format!("{}|{}", tc.pointer("/function/name").and_then(|v| v.as_str()).unwrap_or(""), tc.pointer("/function/arguments").and_then(|v| v.as_str()).unwrap_or("")));
            }
        }
    }
    let mut from = 0usize;
    let mut gone_calls: HashSet<String> = HashSet::new();
    let mut gc_gone: HashSet<String> = HashSet::new();
    let mut user_turns_at: Vec<usize> = Vec::new(); // index of each user message, for tau
    for s in 0..history.len() {
        if !is_boundary(&history[s]) { continue; }
        r.slices += 1;
        if role(&history[s]) == "user" { user_turns_at.push(s); }
        // the view this slice would see: history[from..=s] plus the fixed prefix
        let unwindowed: u64 = fixed + w[..=s].iter().sum::<u64>();
        r.peak_unwindowed = r.peak_unwindowed.max(unwindowed);
        let mut est: u64 = fixed + w[from..=s].iter().sum::<u64>();
        if budget > 0 && est > budget {
            let new_from = cut(&history[..=s], from, target.saturating_sub(fixed), scale);
            if new_from > from {
                r.cuts += 1;
                r.removed += new_from - from;
                for m in &history[from..new_from] { for k in call_keys(m) { gone_calls.insert(k); } }
                from = new_from;
                est = fixed + w[from..=s].iter().sum::<u64>();
            }
        }
        r.peak_windowed = r.peak_windowed.max(est);
        r.token_slices_total += w[..=s].iter().sum::<u64>();
        r.token_slices_saved += w[..from].iter().sum::<u64>();
        // the model's answer to this slice: the assistant messages up to the next boundary
        let next = (s + 1..history.len()).find(|&j| is_boundary(&history[j])).unwrap_or(history.len());
        for m in &history[s + 1..next] {
            for k in call_keys(m) {
                if gone_calls.contains(&k) { r.window_faults += 1; }
                if gc_gone.contains(&k) { r.gc_faults += 1; }
            }
        }
        // tool-result paging: results older than tau user turns and over min_bytes are out of the view
        let cutoff = if tau > 0 && user_turns_at.len() > tau { Some(user_turns_at[user_turns_at.len() - 1 - tau]) } else { None };
        if let Some(cutoff) = cutoff.filter(|&c| c >= from) {
            for (i, m) in history[from..=s.min(cutoff)].iter().enumerate() {
                let i = i + from;
                if role(m) != "tool" || (weigh(m) * 4) < min_bytes { continue; }
                let id = m.get("tool_call_id").and_then(|v| v.as_str()).unwrap_or("");
                let key = call_of.get(id).cloned().unwrap_or_else(|| format!("#{i}"));
                if gc_gone.insert(key) { r.gc_evicted += 1; }
                r.gc_token_slices_saved += w[i];
            }
        }
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    fn user(t: &str) -> Value { json!({"role": "user", "content": t}) }
    fn asst(t: &str) -> Value { json!({"role": "assistant", "content": t}) }
    fn call(id: &str, name: &str) -> Value { json!({"role": "assistant", "content": null, "tool_calls": [{"id": id, "type": "function", "function": {"name": name, "arguments": "{\"a\":1}"}}]}) }
    fn result(id: &str, t: &str) -> Value { json!({"role": "tool", "tool_call_id": id, "content": t}) }

    #[test]
    fn cuts_whole_units_and_keeps_the_newest_input() {
        let h = vec![user("one"), asst("1"), user("two"), call("c1", "echo"), result("c1", &"x".repeat(400)), asst("2"), user("three"), asst("3")];
        // a tiny target: everything but the last unit goes, and the cut lands on an input
        let f = cut(&h, 0, 10, 1.0);
        assert_eq!(f, 6);
        assert!(is_boundary(&h[f]));
        // a target the whole tail meets: no move
        assert_eq!(cut(&h, 2, 10_000, 1.0), 2);
        // never backwards, never past the newest input
        assert_eq!(cut(&h, 6, 1, 1.0), 6);
        assert_eq!(cut(&h, 7, 1, 1.0), 7);
    }

    #[test]
    fn a_result_whose_call_left_gets_a_stub() {
        let h = vec![user("go"), call("c1", "echo"), result("c1", "r1"), asst("ok"), user("more")];
        let stubs = orphan_calls(&h, 2, &[]);
        assert_eq!(stubs.len(), 1);
        assert_eq!(stubs[0]["function"]["name"], "echo");
        assert!(orphan_calls(&h, 1, &[]).is_empty(), "the call is in the tail");
        let p = PendingCall { id: "c9".into(), name: "sleep".into(), provider_call_id: None, arguments: String::new() };
        let stubs = orphan_calls(&h, 4, &[p]);
        assert_eq!(stubs.len(), 1);
        assert_eq!(stubs[0]["function"]["name"], "sleep");
    }

    #[test]
    fn the_notice_copies_user_messages_and_recall_returns_exact_text() {
        let h = vec![user("first ask"), asst("a"), user("second ask"), asst("b"), user("third")];
        let n = notice(&h, 0, 4);
        assert!(n.starts_with("[Paged. Messages #0 to #3 (4 messages"), "{n}");
        assert!(n.contains("#0: \"first ask\"") && n.contains("#2: \"second ask\""), "{n}");
        let idx = recall_indices(&json!({"indices": [2, 99]}), h.len()).unwrap();
        assert_eq!(idx, vec![2]);
        assert_eq!(recall_text(&h, &idx), "#2 user: second ask");
        let idx = recall_indices(&json!({"from": 0, "to": 1}), h.len()).unwrap();
        assert_eq!(recall_text(&h, &idx), "#0 user: first ask\n#1 assistant: a");
        assert!(recall_indices(&json!({"indices": [99]}), h.len()).is_err());
    }

    #[test]
    fn replay_counts_cuts_and_faults() {
        let mut h = Vec::new();
        for i in 0..20 {
            h.push(user(&format!("ask {i}")));
            h.push(call(&format!("c{i}"), if i % 5 == 0 { "repeat" } else { "echo" }));
            h.push(result(&format!("c{i}"), &"r".repeat(200)));
            h.push(asst("done"));
        }
        let r = replay(&h, 300, 150, 1.0, 0, 2, 100);
        assert!(r.cuts >= 2, "{r:?}");
        assert!(r.peak_windowed <= 300 + 60 && r.peak_unwindowed > 1000, "{r:?}");
        assert!(r.window_faults >= 2, "the repeated identical call after its first copy left: {r:?}");
        assert_eq!(r.gc_evicted, 0, "the window took every old result before tau could: {r:?}");
        let gc = replay(&h, 0, 0, 1.0, 0, 2, 100);
        assert!(gc.gc_evicted > 0 && gc.gc_faults >= 1 && gc.cuts == 0, "{gc:?}");
        let none = replay(&h, 0, 0, 1.0, 0, 0, 0);
        assert_eq!(none.cuts, 0);
        assert_eq!(none.peak_windowed, none.peak_unwindowed);
    }
}
