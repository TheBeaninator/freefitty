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
    /// A new thread yields freed slots to threads waiting for their home node for at most this long, then takes
    /// any free slot (else steady load starves new threads: goldfish, expert freezes).
    #[serde(default = "d_newcomer_wait")]
    pub newcomer_wait_s: u64,
    /// Park a quiet thread's cached state on this host's disk and put it back into the node that runs its next
    /// turn (strixite cache-xfer = on). A node's own disk tier is small and capped, so a sleeper's state is
    /// otherwise often dropped and its next turn reads the whole prompt again.
    #[serde(default = "d_true")]
    pub park: bool,
    /// Where parked state goes; unset = `<data_dir>/parked`.
    #[serde(default)]
    pub park_dir: Option<PathBuf>,
    /// A thread sleeping at least this long is parked at once; one with a tool call out, or done with nothing
    /// pending, is parked after this long if nothing came in.
    #[serde(default = "d_park_after")]
    pub park_after_s: u64,
    /// Whether `commission_expert` may give the expert steering. Off until a steered goldfish is measured against a
    /// plain one (verdict agreement and malformed tool calls: steering touches every token, tool-call JSON
    /// included); expert definitions written by hand may steer either way.
    #[serde(default)]
    pub commission_steer: bool,
    /// The largest |scale| a commissioned expert's steering may use (a model should not commission a scale-50
    /// version of itself; a 4-prompt vector broke the model past 16 on the pod).
    #[serde(default = "d_steer_scale")]
    pub max_steer_scale: f64,
    /// Tests only: a slice whose user message is exactly this text panics (the panic path's test). Empty = never.
    #[doc(hidden)]
    #[serde(default)]
    pub test_panic_on: String,
    /// Give back redb's free space when the daemon starts (redb never shrinks its file on its own).
    #[serde(default = "d_true")]
    pub compact_on_start: bool,
    /// A consistent snapshot of the store into `<data_dir>/backups/` this often (hours; 0 = never) ...
    #[serde(default = "d_backup_every")]
    pub backup_every_h: u64,
    /// ... keeping the newest this many.
    #[serde(default = "d_backup_keep")]
    pub backup_keep: usize,
    /// The park directory's cap; the oldest files go first.
    #[serde(default = "d_park_gib")]
    pub park_max_gib: u64,
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
    /// Prompt tokens a thread may reach before its turns end in an error reply; 0 = the nodes' context length
    /// minus the output cap. The estimate is bytes / 4, scaled up by how the node counted the last turn.
    #[serde(default)]
    pub max_context_tokens: u64,
    /// Seconds one request to a node may take. The wait for a free slot is not counted: a turn queued behind
    /// other threads waits as long as it must (the queue's `oldest_runnable_s` shows it).
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
    /// The budget window: when a thread's view would pass `window_tokens`, its oldest whole turns leave the view
    /// (they stay stored; the model gets a `recall` tool and a notice naming them) until it is under
    /// `window_trim_tokens`, and the cut then holds until the budget is crossed again. On by default: it is
    /// the guard against the node's context limit and does nothing until a thread reaches it.
    #[serde(default = "d_true")]
    pub window: bool,
    /// Images a request shows as pictures, newest first; older ones become a line saying they left the view.
    /// Every image in view is paid for on every turn (about 1000 tokens each on strixite).
    #[serde(default = "d_keep_images")]
    pub keep_images: usize,
    /// Tokens the view may reach before a cut; 0 = the nodes' context length minus the output cap and a margin.
    #[serde(default)]
    pub window_tokens: u64,
    /// Tokens a cut trims the view to; 0 = five eighths of the budget.
    #[serde(default)]
    pub window_trim_tokens: u64,
    /// Paging of old tool results (FF-PAGE-RESULTS, on): a result the model has answered this many turns ago and
    /// that is over `page_results_min_bytes` is shown as a handle naming the call and `recall(#i)`; the text stays
    /// stored. 0 = off. Paging advances in batches of `page_results_batch` results (or at a window cut), because
    /// each advance changes the view at an old position and costs the node one prefix recompute.
    #[serde(default = "d_page_turns")]
    pub page_results_after_turns: u64,
    #[serde(default = "d_page_min")]
    pub page_results_min_bytes: u64,
    #[serde(default = "d_page_batch")]
    pub page_results_batch: u64,
    /// Tokens per 4 bytes of a message before any node has counted: Qwen tokenizers run about a third over
    /// bytes / 4 on pod traffic. A thread's own ratio (the node's count over the estimate) replaces it after its
    /// first turn, and the latest ratio any thread saw stands in before that.
    #[serde(default = "d_scale")]
    pub est_scale_default: f64,
    /// Directory of `<name>.toml` expert definitions, read at start and on `POST /experts/reload`; empty = none.
    #[serde(default)]
    pub experts_dir: Option<PathBuf>,
    /// Experts whose frozen state is put into every node's cache once the nodes are up (every expert is frozen then).
    #[serde(default)]
    pub warm_on_start: Vec<String>,
}

fn d_listen() -> String { "127.0.0.1:8081".into() }
fn d_data_dir() -> PathBuf { PathBuf::from("data") }
fn d_model() -> String { "qwen3.8-flash-next".into() }
fn d_slots() -> u32 { 2 }
fn d_spill() -> u64 { 16_384 }
fn d_true() -> bool { true }
fn d_park_after() -> u64 { 300 }
fn d_newcomer_wait() -> u64 { 30 }
fn d_backup_every() -> u64 { 24 }
fn d_backup_keep() -> usize { 7 }
fn d_steer_scale() -> f64 { 8.0 }
fn d_park_gib() -> u64 { 64 }
fn d_keep_images() -> usize { 4 }
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
fn d_scale() -> f64 { 1.3 }
fn d_page_turns() -> u64 { 4 }
fn d_page_min() -> u64 { 500 }
fn d_page_batch() -> u64 { 32 }

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
        if self.window_trim_tokens > 0 && self.window_tokens > 0 && self.window_trim_tokens >= self.window_tokens {
            bail!("window_trim_tokens must be under window_tokens");
        }
        if self.max_context_tokens > 0 && self.window_tokens > self.max_context_tokens {
            bail!("window_tokens must not exceed max_context_tokens");
        }
        if !(1.0..=4.0).contains(&self.est_scale_default) { bail!("est_scale_default must be between 1 and 4"); }
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
