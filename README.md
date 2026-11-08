# FreeFitty

Durable agents on a cluster of Strix Halos. One Rust binary, one data file, a systemd unit.

A thread is its stored conversation. Something arrives for it (a person's message, a tool's result, a wake-up it
asked for), the model runs once over the conversation, the turn is committed, and the thread holds nothing again
until the next thing arrives. Tools are HTTP servers that answer whenever they are done. Sleeping is free, however
long. A thread lives as long as its work does, days if need be, and never grows past the node's context.

Put differently: FreeFitty is a durable actor system whose actor state is a transcript and whose behaviour function
is a language model. The model has no loop; the scheduler is the loop and the model is a pure step. Every tool call
ends the step, so sleeping costs nothing, a retry re-runs a step that committed nothing, and running a turn exactly
once falls out of the commit boundary. Everything else in this file follows from that one choice.

## Where it runs

The pod: pod-control (the scheduler) and three to five strixite nodes on a 100 GbE rail, each serving the same
model with two slots and a 262,144-token context. The pod's work cards run as FreeFitty threads: pod-ff opens a
thread per card round and watches it, pod-hands is the tool server the threads call, and the board decides what
runs. FreeFitty replaced the freefinity stack on 2026-11-07: hydro-project/infinity's AWS durable mode on stand-ins
(ElasticMQ, moto, dynalite, Postgres, two Lambda emulators, a Python bridge, a Python router, fourteen containers).
What those did that mattered is here, in about 4,100 lines: a queue with due times instead of SQS, delay queues and
a scheduler; one [redb](https://github.com/cberner/redb) file instead of DynamoDB and Postgres; the nodes' own slot
counts instead of hand-sized worker replicas; one process instead of a bridge, a relay, a clock, an outbox, a
receiver, a callback service and a model router; a unit instead of rootless podman.

## Run it

```
cargo build --release
./target/release/freefitty init --nodes http://10.44.0.51:5300,http://10.44.0.52:5300,http://10.44.0.53:5300 \
    --tool-servers http://10.42.0.31:8096 --listen 127.0.0.1:8191
./target/release/freefitty serve
./target/release/freefitty chat "Sleep for 20 seconds, then tell me what you did."
```

`init` writes `freefitty.toml` with two fresh tokens (0600; every setting is in that file, nothing comes from the
environment). `serve` is the daemon. Everything else talks to it over its API with the operator token from the same
file: `say`, `chat`, `watch`, `threads`, `show`, `rm`, `dlq`, `compact`, `experts`, `replay`, `status`.

What you see for that prompt: one turn runs the model, the model calls `sleep`, the turn is committed and the thread
is idle. Twenty seconds later its wake-up is due, a turn runs with the whole conversation, and the reply arrives.
Nothing waited in between. On the cluster the second turn read 738 of its 815 prompt tokens from the node's cache,
and a fresh thread's first turn read 719 of 740: the system prompt every thread shares.

## How a turn happens

**Store.** `data/freefitty.redb` holds threads, their messages in OpenAI chat shape, the event queue, processed
event ids, dead letters, uncollected replies, unacknowledged tool invocations, tool manifests and experts. A slice's
whole result is one transaction: the new messages, the events it consumed, the replies it produced, the wake-ups it
scheduled, the invocations to post. A crash before the commit re-runs the slice from the same events; a crash after
it cannot run the turn twice, because the events are gone and their ids are recorded. Dead letters and their
deletion are as durable as everything else. Two daemons cannot open one data file.

**Queue.** Events are mirrored in memory in order. A slice takes all of one thread's due events, so several
messages that arrived together are one turn that sees them all; a thread with a slice in flight is skipped, so a
conversation never runs twice at once and stays strictly ordered however many threads run. A failed slice retries
after `retry_after_s` and dead-letters after `max_attempts`, with an error reply so the operator sees it; `dlq
redrive` puts the events back with their ids kept, so a turn that had committed is not run again, and reopens a
closed thread. While no node is up nothing starts, so an outage costs no attempts. `slice_timeout_s` bounds one
request to a node; a turn queued behind other threads waits for its slot as long as it must, and that wait shows in
`/health` as `oldest_runnable_s` (events of idle threads; `oldest_due_s` also counts those waiting behind
their own thread's running slice) and on the turn's log line. On SIGTERM nothing new starts and running slices finish;
give the unit `TimeoutStopSec` at least `slice_timeout_s`. If the scheduler or the server ever ends on its own the
process exits with status 3, so systemd restarts a whole daemon rather than leaving one that answers `/health` and
takes no work. A slice that panics releases its thread.

**Slice.** The events become messages: a person's text, a tool's result, a wake-up (a tool result for the sleep
call). The model answers once, with the built-in tools (`sleep`, `sleep_until`, `sleep_until_event_or_input`,
`recall`, and `refocus` when compaction is on) and every tool the RAP servers offer. Text becomes a reply; each tool
call becomes a wake-up event or a RAP invocation; the thread records its pending calls. Every real tool is
asynchronous, so a turn with tool calls ends the slice; `recall` and a tool that does not exist are answered within
the slice, up to three rounds. A stored tool message names its call by the id the assistant message carries.
Reasoning content is dropped before storing. Tool-call arguments a model JSON-encoded twice, or wrote as a bare
value, are repaired before storing, since a node refuses to render a thread with such a call in it.

**Router.** Each node's `/health` is read every second: `slots`, `busy_slots` and `queued` count every client's
requests, so a node shared with other work counts as busy; between polls the router adds what it started itself;
`context_length` sets the budget below. A thread's next turn goes to the node that served its last one, where its
state is in the prompt cache. A new thread goes to the node with the most free slots, nodes with the cache transfer
routes first. While a thread's node is full the turn waits in the router, and a slot that frees there goes to it
before any newcomer. A thread leaves its node only when the node is full and the thread is under `spill_tokens` (it
reads its prompt again) or, with `move_threads`, by taking its state along: `POST /cache/export` on the old node
streams into `POST /cache/import` on the new one (strixite with `cache-xfer = on`), then the turn runs there. A node
that refuses a connection is down until its `/health` answers; a turn that never reached it goes to another. While
no node is up a turn waits `node_down_wait_s`. A reply a node breaks off is an error the slice retries, never a
shorter turn. With `grant_cmd` set (for example `["pod-worker", "may", "{node}", "--owner", "freefitty"]`), each
node's command runs every `grant_check_s` seconds; while it exits 1 the node is held: it gets no new, spilled or
moved threads, and a thread living there leaves at its next turn, with its state when the node can export it.

**Parking.** A node's prompt cache spills to its own disk, but that tier is small: the drive is shared with
weights and the cache caps its writes an hour, so under memory pressure a sleeper's state is dropped and its next
turn reads the whole prompt again (on the pod on 2026-11-07, 12 of 621 resumes on one node came from disk). With
`park` (on) the daemon keeps quiet threads' state itself: `POST /cache/export` on the thread's node, with its last
request as the body, writes `<park_dir>/<thread>.kv` (default `<data_dir>/parked`), and the next turn first
`POST /cache/import`s that file into whichever node it is placed on. A thread is parked at once when it sleeps at
least `park_after_s` (300) or waits on events with nothing else out, and after `park_after_s` when a tool call is
out or its turn ended with nothing pending, if no turn ran in between and none is running or waiting for a slot. A shorter sleep is never parked: the node
still has it. Files are written under a temporary name and renamed; a turn that commits removes its thread's file,
deleting a thread removes it, and the oldest files go past `park_max_gib` (64). On because the node's own disk
tier was measured failing on the pod; a node without cache transfer answers the export with an error and nothing
is parked. `/health` counts `parked`, `park_missed` (the node held nothing), `restored` and `restore_failed`.

**Tools.** Each server in `tool_servers` publishes `/.well-known/rap-toolset`; the manifests are read at start and
every `toolset_refresh_s`, and the last one read is kept for a server that is down. An invocation is posted to the
manifest's endpoint after its slice committed; one the server does not acknowledge is posted again with a growing
wait (30 s to 10 min) and, after twelve tries or a 4xx, becomes an error result so the thread is not stuck forever.
On start the daemon resumes invocations a crash left unacknowledged. Results come back to
`<callback_base>/rap/callback/<callback_token>` (or with the token as a bearer); a result posted twice queues once,
however close together. A result for a call the thread no longer waits on is said to the model in the user role.

**Images.** A result may carry images beside its text: MCP `{"type": "image", "data", "mimeType"}` parts or
`image_url` parts with a `data:image/...` URL, up to four per result. The tool message says how many follow, and
after the batch's tool messages one user message per result shows them as `image_url` parts, so a strixite node
with vision on sees them. A remote image URL is named in the text and not sent (a node asked to fetch it could be
pointed anywhere). Each image counts as 1400 tokens in the estimate, not its base64 bytes, and a request shows only
the newest `keep_images` (4) as pictures; older ones become a line saying they left the view. The callback body
limit (`max_body_bytes`) bounds what a tool can send, so a tool should downscale before it posts.

**API** (`listen`, loopback by default; a front door with TLS is Caddy's or a wireguard peer's job):

```
POST /say                 {text, thread?, expert?, metadata?}  -> {thread, id}         operator token
GET  /output?wait&thread  the oldest uncollected reply; removed once sent              operator token
GET  /threads             id, size, pending calls, last message kind, node, closed      operator token
GET  /threads/{id}        meta: live_tokens, prompt_tokens, window, compaction, expert  operator token
GET  /threads/{id}/messages[?view=live][&tail=N]   the store, or what the model sees    operator token
PUT  /threads/{id}/messages {messages}   a history for a thread that has none           operator token
POST /threads/{id}/compact {focus?}      DELETE /threads/{id} (409 while a slice runs)  operator token
GET  /dlq                 POST /dlq/redrive, /dlq/drop with {thread?}                  operator token
GET  /experts             POST /experts/reload, /experts/{name}/warm, /experts/from-thread/{id}
POST /rap/callback        or /rap/callback/{token}: a tool's result                    callback token
GET  /health              nodes (context_length, load, waiting), router counts, queue (oldest_runnable_s, oldest_due_s), store
```

A reply carries `metadata.thread`, `metadata.reply_to` (the id `/say` returned for the message it answers),
`metadata.kind` (`text` or `error`) and the thread's own metadata. A message is at most 256 KiB, a body 1 MiB.
Nothing user-supplied is logged.

## What the model sees

The store is the truth and is never edited. What the model sees is a *view* of it: the shared prefix (the system
prompt, then an expert's reading if the thread has one), then bridges, then the stored history from some index on.
The node's KV cache is derived from the view and worth exactly as much as the view's stable prefix, so every rule
here protects that prefix: the system prompt is the same bytes for every thread, tool definitions are sent in one
order, a thread stays on its node, state moves with it over the rail, and the view's head changes only at a few
named events, each one cache miss, never one a turn. The transcript is durable; the cache is a cost, never
corruption.

Three mechanisms move the view's head. Each has a default and the reason for it, decided from the pod's own
numbers, not from a reflex either way.

**The budget window, on.** A node's context is finite and a pod thread is not: a card's thread runs for days and
its tool results pile up. When a thread's next view would pass `window_tokens` (default: the nodes' context length
minus `max_output_tokens` and an 8k margin, 249,856 on today's nodes), its oldest whole units leave the view until
it is under `window_trim_tokens` (default five eighths of the budget), and the cut then holds until the budget is
crossed again. A unit starts at an input, a person's message or a tool result, so the view never opens with the
model's own words; a result whose call left the view gets a stub call so the template still sees a call before
every result. The view opens with a `[Paged.]` notice that names the range that left, copies the user messages
among it in short, and says how to get any of it back: the `recall` tool returns stored messages by index, exactly
as written, answered from the store within the same slice (a fault costs one more model pass, no tool round trip,
at most 16 KiB a call). On because it is the guard against the wall and does nothing until a thread reaches it; the
day it shipped, six live threads were past 150k tokens and three had hit the wall. `window = false`
turns it off.

The estimate behind it is bytes / 4 scaled up by how the node counted: the thread's own ratio after its first turn,
else the latest ratio any thread saw, else `est_scale_default` (1.3; measured 1.26 to 1.61 on pod traffic). A node
that still answers 400 with its count recalibrates the thread, cuts, and sends once more within the slice; only a
thread with nothing left to cut is closed. `GET /threads/{id}` reports `window.from`, `window.cuts`, `prompt_tokens`
(the node's count) and `live_tokens` (the estimate); the journal logs each cut with the view before and after.

**Refocusing, off.** On strixite the coherence of the context matters before its size: a long thread fills with
superseded ideas and finished branches. With `compaction = true` a thread can refocus: the model writes notes
around a focal point, and from then on those notes stand in for the conversation before them, as the model's own
assistant turn carrying any tool calls still awaited. Three ways in: the model's `refocus` tool, the operator's
`freefitty compact THREAD --focus "..."`, or `compact_at_tokens`. The instruction (`src/compact.md`, or
`compact_prompt_file`) puts first what is needed next; notes that hit `compact_summary_tokens` are asked for again
at half the length. The refocus is its own transaction before the turn's commit, the window restarts at its cut,
and the thread's node is forgotten since its cached state now covers only the prefix. Off because it changes what
the model believes and no measurement yet says when a model-written summary beats the exact history behind a
window plus `recall`; on real nodes it also needed `tool_choice = "none"` and a real token budget to produce notes
at all. `freefitty show THREAD --live` prints what the model sees after one.

**Experts, off.** An expert is the model's cached state after reading: memory with an address (its name), a
version (the content hash of what it read) and a residence (a state file on this host). What it reads is
documents and an acknowledgement, after the shared system prompt, defined as `<experts_dir>/<name>.toml`:

```toml
description = "Has read the pod runbook"
ack = "I have read the runbook and will work within it."
[[reads]]
path = "/data/pod/objective/RUNBOOK.md"
[[reads]]
text = "The board is /data/pod/kanban.db."
```

Freezing reads it once on one node and exports that node's state at the end of the acknowledgement (strixite's
checkpoint at the start of the next user message) into `<data_dir>/experts/<name>@<hash>-<context>.kv`. A thread
born from it (`freefitty say "..." --expert NAME`) gets that state imported into whichever node takes its first
turn, which then reads only its own message: on any node, after a restart, however much the expert read. The
context is a hash of the system prompt and the tool definitions, which render before the expert; when either
changes the expert is frozen again, and a thread born meanwhile reads the expert itself (and triggers the freeze).
Every current expert without state is frozen once a node is up, after `experts reload`, and after `experts freeze
THREAD NAME`, which makes a thread's refocus notes what an expert has read. The stored messages stay as its
provenance: every born thread sends those exact bytes, and the state is rebuilt from them. Versions are never
pruned: a thread keeps the bytes it was born with when a file changes. `experts warm NAME` puts the frozen state
into every up node (a reading on each, where nodes cannot take state); `warm_on_start` does that at boot.
`experts list` shows each expert's frozen bytes. Off until the experts exist: the mechanism is cheap, the practice
is the work, and it is a board card (FF-EXPERTS) to grow a small set from the kinds of work the pod repeats,
measure their cache reuse, and retire what nothing uses.

**Paging old tool results, on.** Most of a pod thread is tool output, and most of it is read once. A result the
model has answered `page_results_after_turns` (4) model turns ago and that is over `page_results_min_bytes` (500)
is shown as a handle in its place, `[Paged out: name(args...) answered with N bytes; recall(#i) returns it]`, under
the same call id so the template still pairs them; the text stays stored and `recall` brings it back, after which
that index is pinned and shown in full however old. An error result is never paged: the model needs it while it
debugs. Paging advances in batches of `page_results_batch` (32) results, or all at once when the window cuts,
because every advance changes the view at an old position and costs the node a recompute of everything after it,
about as much work as a dozen turns' worth of the attention it saves; between advances the view is stable and the
cache serves it, and a cut, which pays that recompute anyway, pages everything it can. On because the replay below measured it on the pod's
own threads first; `page_results_after_turns = 0` turns it off. `GET /threads/{id}` reports `paging.before` (the
index below which results are handles) and `paging.pinned`.

**Measuring before deciding.** `freefitty replay` runs the window's cut and the paging rule over the stored
histories as if they had been live and prints, per thread and in total: the largest view a slice saw with and
without the window, cuts, messages gone, the share of token-slices the node would not have re-read, faults (a
later tool call identical to one whose text had left the view), and for paging (`--tau N --min-bytes B --batch K`)
the results paged, their faults and their share of token-slices. On 2026-11-07 that replay over 28 live threads
and 8,448 messages chose the defaults above: results the model had answered four turns ago were half of everything
the nodes re-read (51 % of token-slices at tau 4 paged one by one, 49 % at tau 8, 52 % at tau 2), with one
identical re-call among 1,851 paged results. The batch trades that saving against recomputes: 45 % at 32, 31 % at
64, 6 % at 128, because a thread has to live long enough to fill a batch; 32 keeps most of the saving and pays the
recompute once in some thirty turns.

## Operating it on the pod

`deploy/freefitty.service` is a user unit for pod-control: binary in `~/.local/bin`, config and data under
`/data/pod/freefitty`. It is an enabled unit, not a transient one, so pod-mem-guard never takes it for an idle
stray. pod-ff and pod-hands speak this API as they spoke freefinity's: `/data/pod/freefitty/.env` carries
`FREEFINITY_PORT`, `BRIDGE_TOKEN` (the operator token) and `CALLBACK_TOKEN` for them, and their units set `FF_STACK`
to that directory. pod-ff reads thread sizes and last turns from `/threads` and prunes with `DELETE`. The Isekai'd
director rounds run on the same daemon under thread ids starting `isk-`.

Upgrading: build on a machine with the pod's glibc or older, copy the binary over, `install` it, and
`systemctl --user restart freefitty`; running slices finish first, and a turn that was waiting re-queues. Keep the
old binary as `freefitty.bak-<stamp>`. Reading the daemon: the journal has one line per turn (`(N prompt tokens, M
from its cache) in R s, after S s waiting for a slot`), one per cut, one per failed slice; `freefitty status` shows the nodes and
counts; `freefitty threads` the threads with their last message kind, so a watcher sees one that stopped talking
(`assistant_empty`, or `tool_call` with nothing pending). Dead letters are read with `freefitty dlq` and put back
with `dlq redrive`, which also reopens a thread the context had closed, now that the window can shorten it.

## Lessons carried over from running freefinity

These came from the pod's weeks on freefinity; each one is a design point here rather than a patch.

- A thread must never run twice at once, and several messages of one thread run together in order. The queue's
  in-flight set and per-thread batches do that; there is no visibility timeout to leave a message invisible for
  16 minutes after a kill.
- A redelivered event must not run its turn again. Processed ids are written in the same transaction as the turn
  and kept four times the reply retention.
- Dead letters, and deleting them, must survive a restart. They are rows in the same transactional file.
- A reply cut short must be an error, not an empty turn: replies are not streamed, and a broken body fails the
  slice. A thread that stops talking is visible: `/threads` shows each thread's last message kind.
- A green status must not hide starvation: `/health` reports how long the oldest runnable event has waited, and a
  wait for a node slot is never counted against the thread.
- A model that cannot be reached must cost a pause, not retries: the scheduler starts nothing while no node is
  up, and the router waits for a node that blinks.
- The model must not be asked while its reply would go nowhere: a slice commits before anything leaves the process.
- A tool server that is down must not lose a call: invocations are stored and re-posted with backoff; a result
  posted while the daemon was down lands when the server retries, as pod-hands does.
- A node's `/health` can lack slot counts, and other clients load it: both are counted.
- Nothing in the system prompt may differ between threads, or the nodes' caches cannot share it.
- A thread must outlive the node's context: the window, not a closed thread, is the answer to the wall, and the
  estimate learns the tokenizer from the node rather than trusting bytes / 4.
- A core task that dies must take the process with it.
- Containers, their pause process, their DNS and the memory guard that stopped their scopes are not involved.

## Tests

`cargo test` runs the unit tests (the window's cut, stubs, notice, paging boundary, batch and handle, and replay; the estimate's scaling; argument
repair; the router's placement with held nodes) and `tests/e2e.rs`: the daemon against a scripted model node and a
RAP tool server, in process. It covers a plain turn, a sleep through the queue, a RAP tool whose result is posted
twice, double-encoded arguments, an unknown tool, a turn with nothing in it, three failures to the dead-letter queue
and a redrive that keeps the input's id, two threads at once, a restart with a wake-up pending; a refocus from the
tool and from the operator with the pending call bridged and its late result answered, notes cut off and asked
again, nothing-to-fold, the token trigger, failed notes; experts sharing their prefix bytes, version pinning across
a file change, warm on start, and an expert frozen from a thread's notes; an expert frozen once into a state file
that each born thread's first turn imports byte for byte, a later turn importing nothing, and a restart keeping the
state without a second reading; the window cutting whole units at the
budget, the notice carrying the first message, the cut point holding across turns, `recall` returning the exact
text within the slice, and the replay agreeing with the live run; old results paging out in a batch as handles, an
error result staying, and a recalled result pinned back in full; a node's count correcting a wrong estimate so
the turn runs; a full context closing a thread that has nothing to cut; a wait for a node slot longer than the
slice timeout that is not a failure; a tool's images reaching the model; and a sleeper parked and its exact bytes
restored into the node when it wakes, nothing parked when the node holds nothing, and its file removed with the
thread. `FREEFITTY_NODES=http://a:5300,http://b:5300
cargo test --test real_nodes -- --ignored` proves the cache move on two real strixite nodes.

## License

Apache-2.0.
