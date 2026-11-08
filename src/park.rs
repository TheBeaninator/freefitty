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
        let p = Parking { dir, max_bytes: max_gib << 30 };
        // a write cut short by a stop leaves its temporary file
        if let Ok(rd) = std::fs::read_dir(&p.dir) {
            for e in rd.flatten() {
                if e.file_name().to_string_lossy().ends_with(".part") { let _ = std::fs::remove_file(e.path()); }
            }
        }
        p.prune();
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

    /// The oldest files go until the rest fit under `park_max_gib`.
    pub fn prune(&self) {
        let mut files = self.files();
        let mut total: u64 = files.iter().map(|f| f.1).sum();
        files.sort_by_key(|f| f.2);
        for (path, len, _) in files {
            if total <= self.max_bytes { break; }
            if std::fs::remove_file(&path).is_ok() {
                total -= len;
                tracing::info!("parked state {} dropped: the park directory is over {} GiB", path.display(), self.max_bytes >> 30);
            }
        }
    }

    fn files(&self) -> Vec<(PathBuf, u64, std::time::SystemTime)> {
        let Ok(rd) = std::fs::read_dir(&self.dir) else { return vec![] };
        rd.flatten()
            .filter(|e| e.file_name().to_string_lossy().ends_with(".kv"))
            .filter_map(|e| { let m = e.metadata().ok()?; Some((e.path(), m.len(), m.modified().ok()?)) })
            .collect()
    }
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
        p.prune();
        assert!(p.has("new").is_some() && p.has("old").is_none());
        p.forget("new");
        assert_eq!(p.bytes(), 0);
    }
}
