// The commit a binary was built from, for `freefitty --version`, /health and /metrics: "<short sha>[-dirty]", or
// "unknown" outside a git checkout. So the binary on a host can be told apart from any other build.
use std::process::Command;

fn main() {
    let git = |args: &[&str]| Command::new("git").args(args).output().ok().filter(|o| o.status.success()).map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned());
    let commit = git(&["rev-parse", "--short=12", "HEAD"]).unwrap_or_else(|| "unknown".into());
    let dirty = git(&["status", "--porcelain", "--untracked-files=no"]).is_some_and(|s| !s.is_empty());
    println!("cargo:rustc-env=FREEFITTY_COMMIT={commit}{}", if dirty { "-dirty" } else { "" });
    println!("cargo:rerun-if-changed=.git/HEAD");
    println!("cargo:rerun-if-changed=.git/index");
}
