//! Parking: a thread that will be quiet a while has its node's cached state copied to this host's disk, and
//! the state goes back into whichever node runs its next turn.
//!
//! A node's prompt cache spills to its own disk, but that tier is small: a worker's drive is shared with
//! weights, and the cache caps its writes an hour, so under memory pressure a sleeper's state is dropped and its
//! next turn reads the whole prompt again (the pod, 2026-11-07: 12 of 621 resumes on node C came from disk).
//! FreeFitty knows which threads sleep and for how long, so it saves them itself: `POST /cache/export` on the
//! thread's node into `<park_dir>/<thread>.kv`, and `POST /cache/import` of that file on the node the next turn
//! runs on, which may be any node. A thread is parked at once when it sleeps at least `park_after_s` or waits
//! for an event with nothing else out, and after `park_after_s` when a tool call is out or its turn ended
//! with nothing pending and nothing new has come in. A short sleep is never parked: the node still has it.
//! The file is written beside its final name and renamed, so a restore never reads half of one; a turn that
//! runs removes it, since the node's cache is newer from then on.

use std::path::{Path, PathBuf};

pub struct Parking {
    dir: PathBuf,
    max_bytes: u64,
}

impl Parking {
    pub fn new(dir: PathBuf, max_gib: u64) -> std::io::Result<Parking> {
        std::fs::create_dir_all(&dir)?;
        let p = Parking { dir, max_bytes: max_gib.saturating_mul(1 << 30) };
        // a write cut short by a stop leaves its temporary file
        if let Ok(rd) = std::fs::read_dir(&p.dir) {
            for e in rd.flatten() {
                if e.file_name().to_string_lossy().ends_with(".part") { let _ = std::fs::remove_file(e.path()); }
            }
        }
        p.prune(|_, _| false);
        Ok(p)
    }

    /// Where a thread's state is parked (thread ids are `[A-Za-z0-9._-]`, so the name is safe).
    pub fn path(&self, thread: &str) -> PathBuf {
        self.dir.join(format!("{thread}.kv"))
    }

    pub fn has(&self, thread: &str) -> Option<PathBuf> {
        let p = self.path(thread);
        p.is_file().then_some(p)
    }

    pub fn forget(&self, thread: &str) {
        let _ = std::fs::remove_file(self.path(thread));
    }

    /// Bytes parked now.
    pub fn bytes(&self) -> u64 {
        self.files().iter().map(|f| f.1).sum()
    }

    /// Parked states that can never be restored go (see `prune_dir`), then the least recently used until the
    /// rest fit under `park_max_gib`.
    pub fn prune(&self, superseded: impl Fn(&Path, std::time::SystemTime) -> bool) -> Pruned {
        prune_dir(&self.dir, self.max_bytes, &[], "parked state", superseded)
    }

    fn files(&self) -> Vec<(PathBuf, u64, std::time::SystemTime)> {
        kv_files(&self.dir).into_iter().map(|f| (f.path, f.len, f.used)).collect()
    }
}

pub struct KvFile {
    pub path: PathBuf,
    pub len: u64,
    /// When it was written or last marked used by `touch`: the time that says what it was made under.
    pub written: std::time::SystemTime,
    /// The later of that and its last read (a relatime mount records a read within a day).
    pub used: std::time::SystemTime,
}

/// The `.kv` files in a directory.
pub fn kv_files(dir: &Path) -> Vec<KvFile> {
    let Ok(rd) = std::fs::read_dir(dir) else { return vec![] };
    rd.flatten()
        .filter(|e| e.file_name().to_string_lossy().ends_with(".kv"))
        .filter_map(|e| {
            let m = e.metadata().ok()?;
            let written = m.modified().ok()?;
            Some(KvFile { path: e.path(), len: m.len(), written, used: written.max(m.accessed().unwrap_or(written)) })
        })
        .collect()
}

#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct Pruned {
    pub kept: usize,
    pub kept_bytes: u64,
    pub superseded_bytes: u64,
    pub over_cap_bytes: u64,
}

/// A directory of saved states kept within bounds. First every `.kv` file `superseded(path, written)` names goes:
/// state that can never be imported again (its expert has a newer version, or it was made under a system prompt or
/// tool set no longer in use). Age alone never removes one. Then, only if the rest are over `max_bytes`, the least
/// recently used go until they fit. `keep` are never removed (a file being written).
pub fn prune_dir(dir: &Path, max_bytes: u64, keep: &[PathBuf], what: &str, superseded: impl Fn(&Path, std::time::SystemTime) -> bool) -> Pruned {
    let mut out = Pruned::default();
    let mut rest = Vec::new();
    for f in kv_files(dir).into_iter().filter(|f| !keep.contains(&f.path)) {
        if superseded(&f.path, f.written) && std::fs::remove_file(&f.path).is_ok() {
            tracing::info!("{what} {} dropped: superseded", f.path.display());
            out.superseded_bytes += f.len;
        } else {
            rest.push(f);
        }
    }
    rest.sort_by_key(|f| f.used);
    let mut total: u64 = rest.iter().map(|f| f.len).sum();
    for f in rest {
        if total > max_bytes && std::fs::remove_file(&f.path).is_ok() {
            tracing::warn!("{what} {} dropped: the directory is over {} GiB and it was used least recently", f.path.display(), max_bytes >> 30);
            total -= f.len;
            out.over_cap_bytes += f.len;
        } else {
            out.kept += 1;
            out.kept_bytes += f.len;
        }
    }
    out
}

/// Mark a saved state as used now: written under what is current, and recently needed.
pub fn touch(path: &Path) {
    if let Ok(f) = std::fs::File::options().write(true).open(path) {
        let _ = f.set_modified(std::time::SystemTime::now());
    }
}

/// How many tokens a saved state file covers: the last entry's count (a file is one entry, or a base then its
/// delta, in strixite's prompt cache layout: n at byte 16, state size at 24, state offset at 32 of each header).
pub fn tokens_in(path: &Path) -> Option<u64> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = std::fs::File::open(path).ok()?;
    let len = f.metadata().ok()?.len();
    let (mut off, mut n) = (0u64, None);
    while off + 40 <= len {
        let mut h = [0u8; 40];
        f.seek(SeekFrom::Start(off)).ok()?;
        f.read_exact(&mut h).ok()?;
        if &h[..8] != b"STRXPCCH" { return None; }
        let u = |a: usize| u64::from_le_bytes(h[a..a + 8].try_into().unwrap());
        n = Some(u(16));
        off += u(32) + u(24);
    }
    n
}

/// The temporary name a parked file is written under.
pub fn part_path(path: &Path) -> PathBuf {
    let mut s = path.as_os_str().to_owned();
    s.push(".part");
    PathBuf::from(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_oldest_files_go_first_and_parts_are_cleared() {
        let d = tempfile::tempdir().unwrap();
        std::fs::write(d.path().join("old.kv"), vec![0u8; 1 << 20]).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(20));
        std::fs::write(d.path().join("new.kv"), vec![0u8; 1 << 20]).unwrap();
        std::fs::write(d.path().join("cut.kv.part"), b"x").unwrap();
        let mut p = Parking::new(d.path().to_owned(), 1).unwrap();
        assert!(!d.path().join("cut.kv.part").exists());
        assert_eq!(p.bytes(), 2 << 20);
        p.max_bytes = 1 << 20;
        p.prune(|_, _| false);
        assert!(p.has("new").is_some() && p.has("old").is_none());
        p.forget("new");
        assert_eq!(p.bytes(), 0);
    }

    #[test]
    fn superseded_states_go_first_and_age_alone_removes_nothing() {
        let d = tempfile::tempdir().unwrap();
        let year_ago = std::time::SystemTime::now() - std::time::Duration::from_secs(365 * 86_400);
        for n in ["old-current", "superseded", "writing", "recent"] {
            let p = d.path().join(format!("{n}.kv"));
            std::fs::write(&p, vec![0u8; 1 << 10]).unwrap();
            if n != "recent" {
                let t = std::fs::FileTimes::new().set_accessed(year_ago).set_modified(year_ago);
                std::fs::File::options().write(true).open(&p).unwrap().set_times(t).unwrap();
            }
        }
        let keep = [d.path().join("writing.kv")];
        let gone = |p: &Path, _| p.ends_with("superseded.kv") || p.ends_with("writing.kv");
        let r = prune_dir(d.path(), u64::MAX, &keep, "state", gone);
        assert_eq!((r.kept, r.superseded_bytes, r.over_cap_bytes), (2, 1 << 10, 0));
        assert!(d.path().join("old-current.kv").exists() && d.path().join("writing.kv").exists());
        // over the cap, the least recently used goes; touching one makes it the most recent
        touch(&d.path().join("old-current.kv"));
        let r = prune_dir(d.path(), 1 << 10, &keep, "state", |_, _| false);
        assert_eq!((r.kept, r.over_cap_bytes), (1, 1 << 10));
        assert!(!d.path().join("recent.kv").exists() && d.path().join("old-current.kv").exists());
    }
}
