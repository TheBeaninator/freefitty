//! Against real strixite nodes: `FREEFITTY_NODES=http://a:5300,http://b:5300 cargo test --test real_nodes -- --ignored`.
//! Both nodes must run the same model with cache-xfer on. Proves the cache move the router makes when a node
//! is full: a conversation's state exported from one node and imported into the other over the rail.

use freefitty::config::Config;
use freefitty::router::Router;
use serde_json::json;
use std::time::Duration;

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn cache_moves_between_nodes() {
    let nodes: Vec<String> = std::env::var("FREEFITTY_NODES").expect("FREEFITTY_NODES").split(',').map(|s| s.trim().to_owned()).collect();
    assert!(nodes.len() >= 2);
    let mut cfg = Config::fresh();
    cfg.nodes = nodes.clone();
    let mut tasks = Vec::new();
    let router = Router::new(&cfg, &mut tasks);
    tokio::time::sleep(Duration::from_secs(2)).await;
    let view = router.view().await;
    assert!(view.iter().all(|n| n["up"] == true && n["xfer"] == true), "{view:?}");

    // a turn on the first node puts the conversation in its cache
    // a nonce no node can have cached from an earlier run: what the second node reuses came from the import
    let system = cfg.system_prompt().unwrap();
    let nonce = freefitty::config::id();
    let body = json!({"model": cfg.model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": format!("Run {nonce}. Reply with the single word: moved.")}], "max_tokens": 16, "stream": false});
    let c = reqwest::Client::new();
    let r = c.post(format!("{}/v1/chat/completions", nodes[0])).json(&body).send().await.unwrap();
    let status = r.status();
    let text = r.text().await.unwrap();
    assert!(status.is_success(), "{status}: {text}");
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    let reply = v["choices"][0]["message"].clone();
    let mut msgs = body["messages"].as_array().unwrap().clone();
    msgs.push(json!({"role": "assistant", "content": reply["content"]}));
    msgs.push(json!({"role": "user", "content": "And again, the single word: moved."}));
    let next = json!({"model": cfg.model, "messages": msgs, "max_tokens": 16, "stream": false});

    let t = std::time::Instant::now();
    let moved = router.move_state(&nodes[0], &nodes[1], &next).await.unwrap();
    assert!(moved, "the first node had the conversation cached");
    eprintln!("moved in {:.2}s", t.elapsed().as_secs_f64());

    // the second node resumes from the imported state: the prompt comes from its cache
    let r = c.post(format!("{}/v1/chat/completions", nodes[1])).json(&next).send().await.unwrap();
    let v: serde_json::Value = r.json().await.unwrap();
    let prompt = v["usage"]["prompt_tokens"].as_u64().unwrap();
    let cached = v["usage"]["prompt_tokens_details"]["cached_tokens"].as_u64().unwrap();
    eprintln!("second node: {prompt} prompt tokens, {cached} cached");
    // only the new user message and its template tokens are read: everything before came with the state
    assert!(cached + 64 >= prompt && cached > 150, "the move did not carry the state: {cached}/{prompt} cached");
    for t in tasks { t.abort(); }
}
