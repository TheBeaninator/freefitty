//! freefitty.toml: everything the daemon and the CLI need, in one file. `freefitty init` writes it with
//! fresh tokens; nothing is read from the environment, so a running daemon is exactly its config file.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    /// Where the operator API listens. Loopback by default: the tokens are the only auth, and a front door
    /// with TLS is someone else's job (Caddy, a wireguard peer).
    #[serde(default = "d_listen")]
    pub listen: String,
    /// One redb file lives here: threads, conversations, the event queue, replies, dead letters.
    #[serde(default = "d_data_dir")]
    pub data_dir: PathBuf,
    /// The model id every node serves.
    #[serde(default = "d_model")]
    pub model: String,
    /// The strixite nodes, http://host:port (the rail addresses on the pod). A thread stays on the node that
    /// holds its prompt cache; a full node's long thread moves with its cache when `move_threads` is on.
    #[serde(default)]
    pub nodes: Vec<String>,
    /// Slots a node is assumed to have while its /health does not say.
    #[serde(default = "d_slots")]
    pub node_slots: u32,
    /// A thread whose prompt is under this many tokens may spill to another node, which reads it again.
    #[serde(default = "d_spill")]
    pub spill_tokens: u64,
    /// Move longer threads between nodes with their cached state (strixite cache-xfer = on).
    #[serde(default = "d_true")]
    pub move_threads: bool,
    /// A command that says whether a node may take work now, run every `grant_check_s` per node with `{node}`
    /// replaced by its URL: exit 1 = held (no new, spilled or moved threads; a thread living there leaves at its
    /// next turn), anything else = free. Empty = every node is free.
    #[serde(default)]
    pub grant_cmd: Vec<String>,
    #[serde(default = "d_grant_check")]
    pub grant_check_s: u64,
    /// RAP tool servers; each one's /.well-known/rap-toolset is read at start and every `toolset_refresh_s`.
    #[serde(default)]
    pub tool_servers: Vec<String>,
    #[serde(default = "d_toolset_refresh")]
    pub toolset_refresh_s: u64,
    /// The address tool servers post results to. They append /rap/callback/<callback_token>.
    #[serde(default)]
    pub callback_base: String,
    /// Bearer for /say, /output, /threads, /dlq: the operator.
    #[serde(default)]
    pub operator_token: String,
    /// Bearer (or last path segment) for /rap/callback: tool servers. It can only deliver tool results.
    #[serde(default)]
    pub callback_token: String,
    /// Tokens the model may write per turn.
    #[serde(default = "d_max_output")]
    pub max_output_tokens: u64,
    /// Prompt tokens a thread may reach (bytes / 4) before its turns end in an error reply; 0 = the nodes'
    /// context length minus the output cap.
    #[serde(default)]
    pub max_context_tokens: u64,
    /// Seconds one slice may take: the model call plus the commit.
    #[serde(default = "d_slice_timeout")]
    pub slice_timeout_s: u64,
    /// Delivery attempts before a thread's events dead-letter.
    #[serde(default = "d_attempts")]
    pub max_attempts: u32,
    /// Seconds a failed slice waits before its retry.
    #[serde(default = "d_retry")]
    pub retry_after_s: u64,
    /// Seconds a turn waits for any node to come up before it fails (a node restart costs a pause, not a retry).
    #[serde(default = "d_down_wait")]
    pub node_down_wait_s: u64,
    /// Largest request body on the operator API, bytes.
    #[serde(default = "d_max_body")]
    pub max_body_bytes: usize,
    /// A file whose contents replace the built-in system prompt.
    #[serde(default)]
    pub system_prompt_file: Option<PathBuf>,
    /// Extra text appended to the system prompt (every thread shares it, so the prompt cache does too).
    #[serde(default)]
    pub system_prompt_extra: String,
    /// Replies nobody collected are kept this many days.
    #[serde(default = "d_reply_days")]
    pub reply_retention_days: u64,

    /// Refocusing (compaction): the model gets a `refocus` tool, the operator `POST /threads/{id}/compact`,
    /// and the thread's history before the cut is replaced, for the model, by notes it wrote around a focal
    /// point. The history itself stays on disk. Off: none of that exists.
    #[serde(default)]
    pub compaction: bool,
    /// Live-view tokens (bytes / 4) past which a thread refocuses on its own; 0 = only when asked.
    #[serde(default)]
    pub compact_at_tokens: u64,
    /// Newest history messages kept verbatim after the notes (the tail is also held under half of
    /// `compact_at_tokens` when that is set).
    #[serde(default = "d_keep")]
    pub compact_keep_messages: u64,
    /// Cap on the notes the model writes.
    #[serde(default = "d_summary_tokens")]
    pub compact_summary_tokens: u64,
    /// A file whose contents replace the built-in refocus instruction (src/compact.md).
    #[serde(default)]
    pub compact_prompt_file: Option<PathBuf>,
    /// Directory of `<name>.toml` expert definitions, read at start and on `POST /experts/reload`; empty = none.
    #[serde(default)]
    pub experts_dir: Option<PathBuf>,
    /// Experts whose prefix is read into every node's cache once the nodes are up.
    #[serde(default)]
    pub warm_on_start: Vec<String>,
}

fn d_listen() -> String { "127.0.0.1:8081".into() }
fn d_data_dir() -> PathBuf { PathBuf::from("data") }
fn d_model() -> String { "qwen3.8-flash-next".into() }
fn d_slots() -> u32 { 2 }
fn d_spill() -> u64 { 16_384 }
fn d_true() -> bool { true }
fn d_grant_check() -> u64 { 5 }
fn d_toolset_refresh() -> u64 { 300 }
fn d_max_output() -> u64 { 4096 }
fn d_slice_timeout() -> u64 { 900 }
fn d_attempts() -> u32 { 3 }
fn d_retry() -> u64 { 15 }
fn d_down_wait() -> u64 { 120 }
fn d_max_body() -> usize { 1 << 20 }
fn d_reply_days() -> u64 { 4 }
fn d_keep() -> u64 { 2 }
fn d_summary_tokens() -> u64 { 2048 }

impl Default for Config {
    fn default() -> Self {
        toml::from_str("").expect("every field has a default")
    }
}

impl Config {
    pub fn load(path: &Path) -> Result<Config> {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let c: Config = toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        c.check()?;
        Ok(c)
    }

    pub fn check(&self) -> Result<()> {
        for (name, v) in [("operator_token", &self.operator_token), ("callback_token", &self.callback_token)] {
            if v.len() < 16 || !v.bytes().all(|b| b.is_ascii_alphanumeric() || b"._~-".contains(&b)) {
                bail!("{name} must be 16+ URL-safe characters; `freefitty init` writes one");
            }
        }
        if self.operator_token == self.callback_token {
            bail!("operator_token and callback_token must differ");
        }
        for n in &self.nodes {
            if !(n.starts_with("http://") || n.starts_with("https://")) || n.ends_with('/') {
                bail!("node {n:?} must be http(s)://host:port without a trailing slash");
            }
        }
        if !self.callback_base.is_empty() && !self.callback_base.starts_with("http") {
            bail!("callback_base must be a URL");
        }
        if self.compact_at_tokens > 0 {
            if !self.compaction { bail!("compact_at_tokens needs compaction = true"); }
            if self.max_context_tokens > 0 && self.compact_at_tokens >= self.max_context_tokens {
                bail!("compact_at_tokens must be under max_context_tokens");
            }
            if self.compact_at_tokens <= 4 * self.compact_summary_tokens {
                bail!("compact_at_tokens must be over four times compact_summary_tokens");
            }
        }
        if !self.warm_on_start.is_empty() && self.experts_dir.is_none() {
            bail!("warm_on_start needs experts_dir");
        }
        Ok(())
    }

    /// A config with fresh tokens, the model's address list to be filled in by hand.
    pub fn fresh() -> Config {
        Config { operator_token: token(), callback_token: token(), ..Config::default() }
    }

    pub fn callback_url(&self) -> String {
        let base = if self.callback_base.is_empty() { format!("http://{}", self.listen) } else { self.callback_base.clone() };
        format!("{}/rap/callback/{}", base.trim_end_matches('/'), self.callback_token)
    }

    pub fn compact_prompt(&self) -> Result<String> {
        Ok(match &self.compact_prompt_file {
            Some(f) => std::fs::read_to_string(f).with_context(|| format!("reading {}", f.display()))?,
            None => include_str!("compact.md").to_owned(),
        }
        .trim()
        .to_owned())
    }

    pub fn system_prompt(&self) -> Result<String> {
        let mut p = match &self.system_prompt_file {
            Some(f) => std::fs::read_to_string(f).with_context(|| format!("reading {}", f.display()))?,
            None => include_str!("prompt.md").to_owned(),
        };
        if !self.system_prompt_extra.is_empty() {
            p.push_str("\n\n");
            p.push_str(self.system_prompt_extra.trim());
        }
        Ok(p.trim().to_owned())
    }
}

/// 32 URL-safe characters from the OS's randomness.
pub fn token() -> String {
    use rand::RngCore;
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-";
    let mut bytes = [0u8; 32];
    rand::rng().fill_bytes(&mut bytes);
    bytes.iter().map(|b| ALPHABET[(*b & 63) as usize] as char).collect()
}

/// 16 hex characters: thread ids, event ids, call ids.
pub fn id() -> String {
    use rand::RngCore;
    let mut b = [0u8; 8];
    rand::rng().fill_bytes(&mut b);
    b.iter().map(|x| format!("{x:02x}")).collect()
}
