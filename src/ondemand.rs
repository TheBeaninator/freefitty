//! On-demand window engines: a model served only while a window is open (Ornith-1.5 397B on F D A C, card
//! 83a59b0e07; dad 2026-11-14, DECIDE 005bc072f2: "Yes, FreeFitty may trigger it").
//!
//! A node whose `node_meta` names `window_url` and `window_up` is not polled into service by anyone else: when a
//! turn of its model finds no node serving that model up, the router asks `window_url` (exit 0 = a window is open,
//! 2 = one is queued or loading, 1 = none) and on 1 runs `window_up`, which must return at once (a systemd-run of
//! the blocking `up`, so the window outlives a FreeFitty restart). FreeFitty never closes a window: its end is the
//! window job's own time limit, so the keepers always get the nodes back.
//!
//! While the window loads, at most `window_park` turns wait for it (each waiting turn holds a queue worker); the
//! rest are deferred by `window_retry_s` without counting an attempt. A window that never opened (`window_url`
//! never said 0 after a start) is started again at most `WINDOW_STARTS` times in `WINDOW_RESET`; past that the
//! turns fail as transient (and dead-letter) with the reason, so a broken engine never takes the nodes over and over.

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Starts in a row that never reached an open window before starting stops (a retry cap escalates, not loops).
pub const WINDOW_STARTS: u32 = 2;
/// How long the cap holds before one more start is allowed.
pub const WINDOW_RESET: Duration = Duration::from_secs(6 * 3600);
/// How often `window_url` is asked while turns wait.
const CHECK_EVERY: Duration = Duration::from_secs(15);
/// After `window_up`, how long `window_url` saying "none" is taken as the start not having registered yet.
const START_GRACE: Duration = Duration::from_secs(120);

/// A turn of a window model cannot run now: deferred (not an attempt) by `retry_s`.
#[derive(Debug)]
pub struct WindowPending {
    pub msg: String,
    pub retry_s: u64,
}
impl std::fmt::Display for WindowPending {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str(&self.msg) }
}
impl std::error::Error for WindowPending {}

/// One window engine's commands (from `node_meta`).
#[derive(Clone, Debug)]
pub struct WindowCmds {
    pub url: String,
    pub up: String,
    pub park: u32,
    pub retry_s: u64,
}

#[derive(Default)]
struct State {
    parked: u32,
    checked: Option<Instant>,
    /// The last `window_url` answer: 0 open, 2 pending, 1 none.
    last: Option<i32>,
    started: Option<Instant>,
    /// Starts since the last time `window_url` said open.
    failed_starts: u32,
    capped_at: Option<Instant>,
}

#[derive(Default)]
pub struct Windows {
    st: std::sync::Mutex<HashMap<String, State>>,
}

/// Counts a turn as parked on a model's window until it is placed, fails or is dropped.
pub struct Parked<'a>(&'a Windows, String);
impl Drop for Parked<'_> {
    fn drop(&mut self) {
        let mut st = self.0.st.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(s) = st.get_mut(&self.1) { s.parked = s.parked.saturating_sub(1); }
    }
}

impl Windows {
    /// Park a turn on `model`'s window, or say it is deferred because `park` turns already wait.
    pub fn park<'a>(&'a self, model: &str, w: &WindowCmds) -> Result<Parked<'a>, WindowPending> {
        let mut st = self.st.lock().unwrap();
        let s = st.entry(model.to_owned()).or_default();
        if s.parked >= w.park {
            return Err(WindowPending { msg: format!("{model} window pending: {} turns already wait for it; retry in {} s", s.parked, w.retry_s), retry_s: w.retry_s });
        }
        s.parked += 1;
        Ok(Parked(self, model.to_owned()))
    }

    /// Ask the window's state (at most every CHECK_EVERY) and start one when none is open or pending.
    /// Err = the start cap is hit: the turn fails with the reason.
    pub async fn poke(&self, model: &str, w: &WindowCmds) -> anyhow::Result<()> {
        {
            let st = self.st.lock().unwrap();
            if st.get(model).and_then(|s| s.checked).is_some_and(|c| c.elapsed() < CHECK_EVERY) {
                return match st.get(model).and_then(|s| s.capped_at) { Some(_) => Err(capped(model)), None => Ok(()) };
            }
        }
        let rc = run(&w.url, Duration::from_secs(20)).await;
        let start = {
            let mut st = self.st.lock().unwrap();
            let s = st.entry(model.to_owned()).or_default();
            s.checked = Some(Instant::now());
            if s.last != Some(rc) { tracing::info!("{model} window: {}", match rc { 0 => "open", 2 => "pending", 1 => "none", _ => "state unknown (window_url failed)" }); }
            s.last = Some(rc);
            if rc == 0 { s.failed_starts = 0; s.capped_at = None; }
            if s.capped_at.is_some_and(|c| c.elapsed() >= WINDOW_RESET) { s.capped_at = None; s.failed_starts = WINDOW_STARTS - 1; }
            if s.capped_at.is_some() { return Err(capped(model)); }
            let fresh = s.started.is_some_and(|t| t.elapsed() < START_GRACE);
            if rc == 1 && !fresh {
                if s.failed_starts >= WINDOW_STARTS {
                    s.capped_at = Some(Instant::now());
                    tracing::error!("{model} window: {WINDOW_STARTS} starts in a row never opened; no more starts for {} h", WINDOW_RESET.as_secs() / 3600);
                    return Err(capped(model));
                }
                s.failed_starts += 1;
                s.started = Some(Instant::now());
                true
            } else { false }
        };
        if start {
            tracing::warn!("{model} window: none open; starting one ({})", w.up);
            let rc = run(&w.up, Duration::from_secs(60)).await;
            if rc != 0 { tracing::error!("{model} window: window_up exited {rc}"); }
        }
        Ok(())
    }
}

fn capped(model: &str) -> anyhow::Error {
    anyhow::anyhow!("{model} window: {WINDOW_STARTS} starts in a row never opened; starts are stopped for {} h (check the window engine)", WINDOW_RESET.as_secs() / 3600)
}

/// `sh -c cmd`, its exit code (-1 when it could not run or ran past `limit`); output is dropped.
async fn run(cmd: &str, limit: Duration) -> i32 {
    let child = tokio::process::Command::new("sh").arg("-c").arg(cmd)
        .stdin(std::process::Stdio::null()).stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null())
        .kill_on_drop(true).spawn();
    let Ok(mut child) = child else { return -1 };
    match tokio::time::timeout(limit, child.wait()).await {
        Ok(Ok(s)) => s.code().unwrap_or(-1),
        _ => -1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cmds(dir: &std::path::Path) -> WindowCmds {
        let st = dir.join("state");
        WindowCmds { url: format!("exit $(cat {} 2>/dev/null || echo 1)", st.display()), up: format!("echo up >> {}; echo 2 > {}", dir.join("ups").display(), st.display()), park: 2, retry_s: 60 }
    }

    #[tokio::test]
    async fn none_starts_once_then_pending_and_open_reset_the_cap() {
        let d = tempfile::tempdir().unwrap();
        let w = cmds(d.path());
        let ws = Windows::default();
        ws.poke("orn", &w).await.unwrap();
        assert_eq!(std::fs::read_to_string(d.path().join("ups")).unwrap().lines().count(), 1);
        // pending: no second start, even past the check interval
        ws.st.lock().unwrap().get_mut("orn").unwrap().checked = None;
        ws.poke("orn", &w).await.unwrap();
        assert_eq!(std::fs::read_to_string(d.path().join("ups")).unwrap().lines().count(), 1);
        std::fs::write(d.path().join("state"), "0").unwrap();
        ws.st.lock().unwrap().get_mut("orn").unwrap().checked = None;
        ws.poke("orn", &w).await.unwrap();
        assert_eq!(ws.st.lock().unwrap()["orn"].failed_starts, 0, "an open window resets the start count");
    }

    #[tokio::test]
    async fn a_window_that_never_opens_stops_after_the_cap() {
        let d = tempfile::tempdir().unwrap();
        // window_up does nothing: the state stays "none"
        let w = WindowCmds { up: "true".into(), ..cmds(d.path()) };
        let ws = Windows::default();
        for _ in 0..WINDOW_STARTS {
            ws.poke("orn", &w).await.unwrap();
            let mut st = ws.st.lock().unwrap();
            let s = st.get_mut("orn").unwrap();
            s.checked = None;
            s.started = Some(Instant::now() - START_GRACE);
        }
        let e = ws.poke("orn", &w).await.unwrap_err();
        assert!(e.to_string().contains("starts are stopped"), "{e}");
        assert_eq!(ws.st.lock().unwrap()["orn"].failed_starts, WINDOW_STARTS);
    }

    #[test]
    fn park_caps_the_waiting_turns() {
        let d = tempfile::tempdir().unwrap();
        let w = cmds(d.path());
        let ws = Windows::default();
        let a = ws.park("orn", &w).unwrap();
        let _b = ws.park("orn", &w).unwrap();
        let e = ws.park("orn", &w).err().unwrap();
        assert_eq!(e.retry_s, 60);
        drop(a);
        assert!(ws.park("orn", &w).is_ok(), "a placed turn frees its park place");
    }
}
