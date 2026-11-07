# FreeFitty

Durable agents on a cluster of Strix Halos. One Rust binary, one data file, no containers, no cloud stand-ins.

A thread is its stored conversation. Something arrives for it (a message, a tool's result, a wake-up it asked
for), the model runs once over the conversation, the turn is committed, and the thread holds nothing again. Tools
are HTTP servers that answer whenever they are done. Sleeping is free, however long.

FreeFitty is what [freefinity](https://github.com/TheBeaninator/freefinity) was for, built for one place: a
cluster of [strixite](https://github.com/shawnshekari/strixite) nodes on a 100 GbE rail. freefinity ran
hydro-project/infinity's AWS durable mode on stand-ins: ElasticMQ, moto, dynalite, Postgres, two Lambda
emulators, a Python bridge, a Python router, fourteen containers. Everything those did that mattered is here,
in about 2,500 lines:

| was | is |
|---|---|
| SQS FIFO + delay queue + EventBridge Scheduler + dead-letter queues | one event table: an event has a thread and a due time; a sleep of any length is an event due later |
| DynamoDB thread state + Postgres history | one [redb](https://github.com/cberner/redb) file, every write a fsynced transaction |
| Lambda RIE agent replicas, `AGENT_WORKERS` sized by hand | threads run concurrently up to what the nodes' slots allow; the router holds a turn while they are full |
| the bridge's dispatcher, relay, clock, outbox, receiver, callback service | the scheduler, the dispatcher and the API, in one process |
| `model-router` | the router, same policy, state moves over the rail |
| rootless podman, CNI, DNS pauses, ghosts, scope stops | a systemd service |

## Run it

```
cargo build --release
./target/release/freefitty init --nodes http://10.44.0.51:5300,http://10.44.0.52:5300 \
    --tool-servers http://10.42.0.31:8096 --listen 127.0.0.1:8191
./target/release/freefitty serve
./target/release/freefitty chat "Sleep for 20 seconds, then tell me what you did."
```

`init` writes `freefitty.toml` with two fresh tokens (0600; every setting is in that file, nothing comes from the
environment). `serve` is the daemon. The rest talk to it over its API with the operator token from the same
file: `say`, `chat`, `watch`, `threads`, `show`, `rm`, `dlq`, `status`.

What you see for that prompt: one turn runs the model, the model calls `sleep`, the turn is committed and the
thread is idle. Twenty seconds later its wake-up is due, a turn runs with the whole conversation, and the reply
arrives. Nothing waited in between. On the cluster the second turn read 738 of its 815 prompt tokens from the
node's cache, and a fresh thread's first turn read 719 of 740, the system prompt every thread shares.

## How it works

**Store.** `data/freefitty.redb` holds threads, their messages in OpenAI chat shape, the event queue, processed
event ids, dead letters, uncollected replies, unacknowledged tool invocations and the tool manifests. A slice's
whole result is one transaction: the new messages, the events it consumed, the replies it produced, the wake-ups
it scheduled, the invocations to post. A crash before the commit re-runs the slice from the same events; a crash
after it cannot run the turn twice, because the events are gone and their ids are recorded. Dead letters and
their deletion are as durable as everything else. Two daemons cannot open one data file.

**Queue.** Events are mirrored in memory in order. A slice takes all of one thread's due events; a thread with a
slice in flight is skipped, so a conversation never runs twice at once and stays strictly ordered however many
threads run. A failed slice retries after `retry_after_s` and dead-letters after `max_attempts`, with an error
reply so the operator sees it; `dlq redrive` puts the events back with their ids kept, so a turn that had
committed is not run again. An error no retry fixes (the context is full) dead-letters at once and closes the
thread. While no node is up nothing starts, so an outage costs no attempts. On SIGTERM nothing new starts and
running slices finish; give the unit `TimeoutStopSec` at least `slice_timeout_s`. If the scheduler or the server
ever ends on its own the process exits with status 3, so systemd restarts a whole daemon rather than leaving one
that answers `/health` and takes no work.

**Slice.** The events become messages: a person's text, a tool's result, a wake-up (a tool result for the sleep
call). The model answers once, with the built-in tools (`sleep`, `sleep_until`, `sleep_until_event_or_input`)
and every tool the RAP servers offer. Text becomes a reply; each tool call becomes a wake-up event or a RAP
invocation; the thread records its pending calls. Every tool is asynchronous, so a turn with tool calls ends the
slice. A tool the model named that does not exist is explained to it within the same slice, up to three times.
Reasoning content is dropped before storing. Tool-call arguments a model JSON-encoded twice, or wrote as a bare
value, are repaired before storing, since a node refuses to render a thread with such a call in it.

**Router.** Each node's `/health` is read every second: `slots`, `busy_slots` and `queued` count every client's
requests, so a node shared with other work counts as busy; between polls the router adds what it started itself.
A thread's next turn goes to the node that served its last one, where its state is in the prompt cache. A new
thread goes to the node with the most free slots, nodes with the cache transfer routes first. While a thread's
node is full the turn waits in the router, and a slot that frees there goes to it before any newcomer. A thread
leaves its node only when the node is full and the thread is under `spill_tokens` (it reads its prompt again) or,
with `move_threads`, by taking its state along: `POST /cache/export` on the old node streams into `POST
/cache/import` on the new one (strixite with `cache-xfer = on`), then the turn runs there. A node that refuses a
connection is down until its `/health` answers; a turn that never reached it goes to another. While no node is up
a turn waits `node_down_wait_s`. A reply a node breaks off is an error the slice retries, never a shorter turn.
A node whose `/health` reports only `busy` counts as full while busy.

**Tools.** Each server in `tool_servers` publishes `/.well-known/rap-toolset`; the manifests are read at start
and every `toolset_refresh_s`, and the last one read is kept for a server that is down. An invocation is posted
to the manifest's endpoint after its slice committed; one the server does not acknowledge is posted again every
30 s and, after twelve tries or a 4xx, becomes an error result so the thread is not stuck forever. On start the
daemon resumes invocations a crash left unacknowledged. Results come back to
`<callback_base>/rap/callback/<callback_token>` (or with the token as a bearer); a result posted twice queues
once. A result for a call the thread no longer waits on is still appended as a tool message.

**API** (`listen`, loopback by default):

```
POST /say                 {text, thread?, metadata?}       -> {thread, id}         operator token
GET  /output?wait&thread  the oldest uncollected reply; removed once sent          operator token
GET  /threads             id, size, pending calls, last message kind, node, closed  operator token
GET  /threads/{id}        and /threads/{id}/messages; DELETE forgets the thread     operator token
GET  /dlq                 POST /dlq/redrive, /dlq/drop with {thread?}              operator token
POST /rap/callback        or /rap/callback/{token}: a tool's result                callback token
GET  /health              nodes, router counts, queue (with the oldest due wait), store counts
```

A reply carries `metadata.thread`, `metadata.reply_to` (the id `/say` returned for the message it answers),
`metadata.kind` (`text` or `error`) and the thread's own metadata. A message is at most 256 KiB, a body 1 MiB.
Nothing user-supplied is logged.

## Lessons carried over from running freefinity

These came from the pod's weeks on freefinity; each one is a design point here rather than a patch.

- A thread must never run twice at once, and several messages of one thread run together in order. The queue's
  in-flight set and per-thread batches do that; there is no visibility timeout to leave a message invisible for
  16 minutes after a kill.
- A redelivered event must not run its turn again. Processed ids are written in the same transaction as the turn.
- Dead letters, and deleting them, must survive a restart. They are rows in the same transactional file.
- A reply cut short must be an error, not an empty turn: replies are not streamed, and a broken body fails the
  slice. A thread that stops talking is visible: `/threads` shows each thread's last message kind
  (`assistant_empty`, `tool_call` with nothing pending) for a watcher to act on.
- A green status must not hide starvation: `/health` reports how long the oldest runnable event has waited.
- A model that cannot be reached must cost a pause, not retries: the scheduler starts nothing while no node is
  up, and the router waits for a node that blinks.
- The model must not be asked while its reply would go nowhere: a slice commits before anything leaves the process.
- A tool server that is down must not lose a call: invocations are stored and re-posted; a result posted while
  the daemon was down lands when the server retries, as pod-hands does.
- A node's `/health` can lack slot counts, and other clients load it: both are counted.
- Nothing in the system prompt may differ between threads, or the nodes' caches cannot share it.
- A core task that dies must take the process with it.
- Containers, their pause process, their DNS and the memory guard that stopped their scopes are not involved.

## Deploy on the pod

`deploy/freefitty.service` is a user unit for pod-control: binary in `~/.local/bin`, config and data under
`/data/pod/freefitty`. Name the unit as it is or add it to pod-mem-guard's protected patterns; the guard stops
untagged idle transient units under memory pressure. pod-ff and pod-hands speak this API as they spoke
freefinity's: point them at the new port and token.

## Tests

`cargo test` runs the unit tests and `tests/e2e.rs`: the daemon against a scripted model node and a RAP tool
server, in process. It covers a plain turn, a sleep through the queue, a RAP tool whose result is posted twice,
double-encoded arguments, an unknown tool, a turn with nothing in it, three failures to the dead-letter queue and
a redrive that keeps the input's id, two threads at once, a restart with a wake-up pending, and the token checks.
`FREEFITTY_NODES=http://a:5300,http://b:5300 cargo test --test real_nodes -- --ignored` proves the cache move on
two real strixite nodes.

## License

Apache-2.0.
