//! The freefitty command: `serve` runs the daemon; everything else talks to a running one over its API with
//! the operator token from the same config file.

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use freefitty::config::Config;
use serde_json::{json, Value};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "freefitty", version, about = "Durable agents on a cluster of Strix Halos")]
struct Cli {
    /// The config file.
    #[arg(short, long, default_value = "freefitty.toml", global = true)]
    config: PathBuf,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Write a config file with fresh tokens.
    Init {
        /// Model nodes, http://host:port, comma-separated.
        #[arg(long, value_delimiter = ',')]
        nodes: Vec<String>,
        /// RAP tool servers, comma-separated.
        #[arg(long, value_delimiter = ',')]
        tool_servers: Vec<String>,
        #[arg(long)]
        listen: Option<String>,
        #[arg(long)]
        model: Option<String>,
        #[arg(long)]
        data_dir: Option<PathBuf>,
        /// The address tool servers call back (default: the listen address).
        #[arg(long)]
        callback_base: Option<String>,
    },
    /// Run the daemon.
    Serve,
    /// Send a message; prints the thread and message id.
    Say {
        text: String,
        #[arg(short, long)]
        thread: Option<String>,
    },
    /// Send a message and print the thread's replies as they arrive (Ctrl-C to stop).
    Chat {
        text: String,
        #[arg(short, long)]
        thread: Option<String>,
    },
    /// Print replies as they arrive, every thread's or one thread's.
    Watch {
        #[arg(short, long)]
        thread: Option<String>,
    },
    /// List threads.
    Threads,
    /// Print a thread's stored conversation.
    Show { thread: String },
    /// Forget a thread: its conversation, queued events and replies.
    Rm { thread: String },
    /// Dead letters: list, redrive or drop them (one thread's with --thread).
    Dlq {
        #[arg(default_value = "list")]
        action: String,
        #[arg(short, long)]
        thread: Option<String>,
    },
    /// The daemon's health: nodes, queue, store.
    Status,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Init { nodes, tool_servers, listen, model, data_dir, callback_base } => {
            if cli.config.exists() { bail!("{} exists; delete it to start over", cli.config.display()); }
            let mut c = Config::fresh();
            c.nodes = nodes;
            c.tool_servers = tool_servers;
            if let Some(l) = listen { c.listen = l; }
            if let Some(m) = model { c.model = m; }
            if let Some(d) = data_dir { c.data_dir = d; }
            if let Some(b) = callback_base { c.callback_base = b; }
            let text = format!("# freefitty.toml: the whole configuration. Tokens are secrets: keep this file 0600.\n{}", toml::to_string_pretty(&c)?);
            write_private(&cli.config, &text)?;
            println!("wrote {}", cli.config.display());
            if c.nodes.is_empty() { println!("add at least one node under `nodes` before `freefitty serve`"); }
            Ok(())
        }
        Cmd::Serve => {
            tracing_subscriber::fmt().with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into())).with_target(false).init();
            let cfg = Config::load(&cli.config)?;
            if cfg.nodes.is_empty() { bail!("no nodes in {}", cli.config.display()); }
            let running = freefitty::start(cfg).await?;
            let fault = running.fault.clone();
            let faulted = tokio::select! {
                _ = shutdown_signal() => false,
                _ = fault.notified() => true,
            };
            if faulted {
                tracing::error!("a core task ended; exiting so the service manager restarts the daemon");
                std::process::exit(3);
            }
            tracing::info!("stopping: running slices finish, nothing new starts");
            running.shutdown().await;
            Ok(())
        }
        Cmd::Say { text, thread } => {
            let c = Client::new(&cli.config)?;
            println!("{}", c.say(&text, thread.as_deref()).await?);
            Ok(())
        }
        Cmd::Chat { text, thread } => {
            let c = Client::new(&cli.config)?;
            let sent = c.say(&text, thread.as_deref()).await?;
            let t = sent["thread"].as_str().unwrap_or_default().to_owned();
            eprintln!("thread {t}");
            c.watch(Some(&t)).await
        }
        Cmd::Watch { thread } => Client::new(&cli.config)?.watch(thread.as_deref()).await,
        Cmd::Threads => {
            let v = Client::new(&cli.config)?.get("/threads").await?;
            println!("{:<18} {:>5} {:>9} {:>4}  {:<15} {:<24} closed", "thread", "msgs", "bytes", "pend", "last", "node");
            for t in v.as_array().into_iter().flatten() {
                let n = |k: &str| t[k].as_u64().unwrap_or(0);
                println!("{:<18} {:>5} {:>9} {:>4}  {:<15} {:<24} {}", s(&t["thread"]), n("messages"), n("bytes"), n("pending"), s(&t["last"]), s(&t["node"]), s(&t["closed"]));
            }
            Ok(())
        }
        Cmd::Show { thread } => {
            let v = Client::new(&cli.config)?.get(&format!("/threads/{thread}/messages")).await?;
            for m in v.as_array().into_iter().flatten() {
                let role = s(&m["role"]);
                let content = m["content"].as_str().unwrap_or("");
                println!("--- {role}");
                if !content.is_empty() { println!("{content}"); }
                for tc in m["tool_calls"].as_array().into_iter().flatten() {
                    println!("[call {} {}]", s(&tc["function"]["name"]), s(&tc["function"]["arguments"]));
                }
            }
            Ok(())
        }
        Cmd::Rm { thread } => {
            let v = Client::new(&cli.config)?.delete(&format!("/threads/{thread}")).await?;
            println!("{v}");
            Ok(())
        }
        Cmd::Dlq { action, thread } => {
            let c = Client::new(&cli.config)?;
            let body = thread.as_ref().map(|t| json!({"thread": t})).unwrap_or(json!({}));
            match action.as_str() {
                "list" => {
                    let v = c.get("/dlq").await?;
                    let items: Vec<&Value> = v.as_array().into_iter().flatten().filter(|d| thread.as_deref().is_none_or(|t| d["thread"] == t)).collect();
                    if items.is_empty() { println!("no dead letters"); }
                    for d in items { println!("seq={} thread={} attempts={} id={} kind={}", d["seq"], s(&d["thread"]), d["attempts"], s(&d["id"]), s(&d["kind"]["kind"])); }
                }
                "redrive" => println!("{}", c.post("/dlq/redrive", &body).await?),
                "drop" => println!("{}", c.post("/dlq/drop", &body).await?),
                other => bail!("dlq action must be list, redrive or drop, not {other:?}"),
            }
            Ok(())
        }
        Cmd::Status => {
            let v = Client::new(&cli.config)?.get("/health").await?;
            println!("{}", serde_json::to_string_pretty(&v)?);
            Ok(())
        }
    }
}

fn s(v: &Value) -> String {
    match v { Value::String(s) => s.clone(), Value::Null => "-".into(), other => other.to_string() }
}

fn write_private(path: &std::path::Path, text: &str) -> Result<()> {
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;
    let mut f = std::fs::OpenOptions::new().write(true).create_new(true).mode(0o600).open(path).with_context(|| format!("creating {}", path.display()))?;
    f.write_all(text.as_bytes())?;
    Ok(())
}

async fn shutdown_signal() {
    use tokio::signal::unix::{signal, SignalKind};
    let mut term = signal(SignalKind::terminate()).expect("signal");
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {}
        _ = term.recv() => {}
    }
}

struct Client {
    base: String,
    token: String,
    http: reqwest::Client,
}

impl Client {
    fn new(config: &std::path::Path) -> Result<Client> {
        let cfg = Config::load(config)?;
        let host = if cfg.listen.starts_with("0.0.0.0") { cfg.listen.replacen("0.0.0.0", "127.0.0.1", 1) } else { cfg.listen.clone() };
        Ok(Client { base: format!("http://{host}"), token: cfg.operator_token, http: reqwest::Client::builder().timeout(std::time::Duration::from_secs(90)).build()? })
    }

    async fn check(r: reqwest::Response) -> Result<Value> {
        let status = r.status();
        if status.as_u16() == 204 { return Ok(Value::Null); }
        let text = r.text().await?;
        if !status.is_success() { bail!("{status}: {text}"); }
        Ok(serde_json::from_str(&text).unwrap_or(Value::String(text)))
    }

    async fn get(&self, path: &str) -> Result<Value> {
        Self::check(self.http.get(format!("{}{path}", self.base)).bearer_auth(&self.token).send().await.context("is the daemon running?")?).await
    }

    async fn post(&self, path: &str, body: &Value) -> Result<Value> {
        Self::check(self.http.post(format!("{}{path}", self.base)).bearer_auth(&self.token).json(body).send().await.context("is the daemon running?")?).await
    }

    async fn delete(&self, path: &str) -> Result<Value> {
        Self::check(self.http.delete(format!("{}{path}", self.base)).bearer_auth(&self.token).send().await.context("is the daemon running?")?).await
    }

    async fn say(&self, text: &str, thread: Option<&str>) -> Result<Value> {
        let mut body = json!({"text": text});
        if let Some(t) = thread { body["thread"] = json!(t); }
        self.post("/say", &body).await
    }

    async fn watch(&self, thread: Option<&str>) -> Result<()> {
        loop {
            let mut path = "/output?wait=20".to_owned();
            if let Some(t) = thread { path.push_str(&format!("&thread={t}")); }
            let v = self.get(&path).await?;
            if v.is_null() { continue; }
            let meta = &v["metadata"];
            let when = chrono::Local::now().format("%H:%M:%S");
            println!("[{when} {}{}] {}", s(&meta["thread"]), if meta["kind"] == "error" { " error" } else { "" }, v["text"].as_str().unwrap_or(""));
        }
    }
}
