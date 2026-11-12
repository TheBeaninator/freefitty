# FreeFitty

Durable agents on a cluster of Strix Halos. One Rust binary, one data file, a systemd unit.

A thread is its stored conversation. Something arrives for it (a person's message, a tool's result, a wake-up it
asked for), the model runs once over the conversation, the turn is committed, and the thread holds nothing again
until the next thing arrives. Tools are HTTP servers that answer whenever they are done, over the Reactive Agent
Protocol ([RAP](https://infinity.hydro.run/docs/rap/what-is-rap), from
[hydro-project/infinity](https://github.com/hydro-project/infinity);
[spec](https://infinity.hydro.run/docs/rap/spec/overview)). Sleeping is free, however long. A thread lives as long as its work does, days if need be, and never grows past the node's context.

Put differently: FreeFitty is a durable actor system whose actor state is a transcript and whose behaviour function
is a language model. The model has no loop; the scheduler is the loop and the model is a pure step. Every tool call
ends the step, so sleeping costs nothing, a retry re-runs a step that committed nothing, and running a turn exactly
once falls out of the commit boundary. Everything else in this file follows from that one choice.

## Why

[hydro-project/infinity](https://github.com/hydro-project/infinity) showed the model this file builds on: an agent
that holds nothing between turns, tools that answer whenever they are done, and a sleep that costs nothing. Its
durable mode runs on AWS (SQS, DynamoDB, EventBridge, Lambda). Off AWS, on the pod, that meant fourteen containers of
emulated AWS plus a bridge and a router in Python, and the failures lived in the seams between them: a message
invisible for sixteen minutes after a kill, a redelivered event running its turn twice, a status page green while
work starved. FreeFitty keeps the model and replaces the stack with one process and one transactional file, and
adds what a cluster of local nodes makes possible: a thread stays on the node that holds its cache, and that cache
can be saved, moved and shared (parking, cache moves, frozen experts).

## Where it runs

The pod: pod-control (the scheduler) and the [strixite](https://github.com/shawnshekari/strixite) nodes on a 100 GbE
rail, all Strix Halo machines. Each node serves one model, by default `model`; `node_meta` names another per node,
with its context length and slots, so the strixite workers (two slots, 262,144 tokens) and the StrixFold pair
(DeepSeek V4.1 across two nodes; [StrixFold](https://github.com/TheBeaninator/StrixFold), from
[TensorFold](https://github.com/ashhart/TensorFold)) are nodes of one router, and a turn runs only on nodes serving
its model. The pod's work cards run as FreeFitty threads: pod-ff opens a thread per card round and watches it, pod-hands is the tool server the threads call, and the board decides what runs.
In about 6,900 lines of Rust: a queue with due times and a scheduler; one [redb](https://github.com/cberner/redb)
file for every table; the nodes' own slot counts for capacity; one process for the API, the queue, the router, the
tool calls and their callbacks; a systemd unit.

## What it needs from a node

FreeFitty is built for the pod's own inference servers, and several features depend on endpoints only they have.
Those endpoints are the pod's patches to strixite and StrixFold, and the patches are not in the public strixite or
StrixFold repositories, so outside the pod those features are off, and a plain OpenAI-compatible endpoint is not
enough to run FreeFitty as described here.

| Needs | Used for | Where it comes from |
|---|---|---|
| `POST /v1/chat/completions` with OpenAI tool calls | every turn | any OpenAI-compatible server |
| `GET /health` with `slots`, `busy_slots`, `queued`, `context_length` | placement and the context budget | upstream strixite; StrixFold's `max_batch_size` and `live` are read too |
| `POST /cache/export`, `/cache/import`, `/cache/probe` | parking, moving a thread with its cache, frozen experts | pod patches (strixite `cache-xfer`, StrixFold); the contract is [docs/cache-routes.md](docs/cache-routes.md) |
| `GET /steer`, `steer` and `steer_scope` in requests | persona experts | pod patches to strixite |
| `shared_messages`, `keep_step` | an expert's shared prefix on StrixFold | pod patches to StrixFold |

Against a server that has only the first row (vLLM, llama.cpp's server, and the like): a 404 on `/health` falls back
to `GET /v1/models` to tell whether the node is up; nothing is parked, moved with its cache or frozen (each answers
with an error and the thread reads its prompt again); set `max_context_tokens`, since without a context length the
budget window and the pre-send check are off; leave steering unset. A node behind an API key gets it as
`node_meta."<url>".bearer_token`, sent as `Authorization: Bearer` on every request to that node. Placement assumes each node keeps a prefix cache per
conversation; without one every turn re-reads its whole prompt.

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
file: `say`, `chat`, `watch`, `threads`, `show`, `rm`, `priority`, `dlq`, `compact`, `experts`, `prompt`,
`replay`, `status`, `backup`; `restore` works on the stopped daemon's store.

What you see for that prompt: one turn runs the model, the model calls `sleep`, the turn is committed and the thread
is idle. Twenty seconds later its wake-up is due, a turn runs with the whole conversation, and the reply arrives.
Nothing waited in between. On the cluster the second turn read 738 of its 815 prompt tokens from the node's cache,
and a fresh thread's first turn read 719 of 740: the system prompt every thread shares.

## Configuration

Every setting is in `freefitty.toml`; nothing comes from the environment except `FREEFITTY_LOG_FORMAT`. The sections
below explain each one; this is the list, with defaults.

| Key | Default | What it is |
|---|---|---|
| `listen` | `127.0.0.1:8081` | API address (`init --listen` sets it) |
| `data_dir` | `data` | store, parked and frozen states, backups, instructions |
| `model` | `qwen3.8-flash-next` | the model a thread runs on unless its expert names another, and every node's unless `node_meta` says |
| `nodes` | none | node URLs |
| `node_meta."<url>"` | none | per node: `model`, `context_length`, `slots`, `bearer_token`, `grant` (false: not under `grant_cmd`), `down_wait_s`, `shared_messages`, `keep_step` |
| `node_slots` | 2 | slots assumed while a node's `/health` does not say |
| `spill_tokens` | 16384 | a thread under this may leave a full node and read its prompt again |
| `move_threads` | on | a thread over it leaves a full node with its cache |
| `newcomer_wait_s` | 30 | how long home-waiters keep a freed slot from a new thread |
| `node_down_wait_s` | 120 | how long a turn waits while no node serving its model is up |
| `grant_cmd`, `grant_check_s` | none, 5 | the command that says a node is held, and how often it runs |
| `tool_servers`, `toolset_refresh_s` | none, 300 | RAP servers and how often their manifests are read |
| `callback_base` | `http://<listen>` | the URL tool servers call back |
| `operator_token`, `callback_token` | written by `init` | 16+ URL-safe characters each, different |
| `max_output_tokens` | 16384 | a turn's output cap |
| `max_context_tokens` | 0 | 0: the node's context minus `max_output_tokens` |
| `slice_timeout_s` | 900 | one request to a node |
| `max_attempts`, `retry_after_s` | 3, 15 | a failed slice's retries, and the wait between them |
| `max_body_bytes` | 1 MiB | the largest request body |
| `reply_retention_days` | 4 | uncollected replies; processed ids are kept four times as long |
| `system_prompt_file`, `system_prompt_extra`, `system_prompt_commands` | built-in, none, none | the system prompt |
| `window`, `window_tokens`, `window_trim_tokens` | on, derived, 5/8 of the budget | the budget window |
| `keep_images` | 4 | images shown as pictures |
| `page_results_after_turns`, `page_results_min_bytes`, `page_results_batch` | 4, 500, 32 | paging old tool results |
| `est_scale_default` | 1.3 | tokens per bytes/4 before a node has counted |
| `compaction`, `compact_at_tokens`, `compact_keep_messages`, `compact_summary_tokens`, `compact_prompt_file` | off, 0 (off), 2, 2048, built-in | refocusing |
| `experts_dir`, `warm_on_start` | none, none | expert definitions, and the ones put into every node at boot |
| `commission_steer`, `max_steer_scale` | off, 8 | whether a commissioned expert may steer, and how hard |
| `read_roots` | none | where paths a model names may be read |
| `park`, `park_dir`, `park_after_s`, `park_max_gib` | on, `<data_dir>/parked`, 300, 64 | parking |
| `expert_states_max_gib` | 32 | the backstop cap on frozen expert states |
| `compact_on_start`, `backup_every_h`, `backup_keep` | on, 24, 7 | the store's upkeep |

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
closed thread. While no node at all is up nothing starts, so an outage costs no attempts; while only one model's
nodes are down, its turns start, wait for them (see Router) and count an attempt if they never come. At most 64
slices run at once. `slice_timeout_s` bounds one
request to a node; a turn queued behind other threads waits for its slot as long as it must, and that wait shows in
`/health` as `oldest_runnable_s` (events of idle threads; `oldest_due_s` also counts those waiting behind
their own thread's running slice) and on the turn's log line. On SIGTERM nothing new starts and running slices finish;
give the unit `TimeoutStopSec` at least `slice_timeout_s`. If the scheduler or the server ever ends on its own the
process exits with status 3, so systemd restarts a whole daemon rather than leaving one that answers `/health` and
takes no work. A slice that panics releases its thread. A context overflow the window cannot fix, or a node
refusing a thread's steering, dead-letters at once and closes the thread rather than retrying.

**Slice.** The events become messages: a person's text, a tool's result, a wake-up (a tool result for the sleep
call). The model answers once, with the built-in tools (`sleep`, `sleep_until`, `sleep_until_event_or_input`,
`experts`, `goldfish`, `commission_expert`, `recall` when the window is on, `refocus` when compaction is on) and every
tool the RAP servers offer (a sleep is capped at 366 days). Text becomes a reply; each tool
call becomes a wake-up event or a RAP invocation; the thread records its pending calls. Every real tool is
asynchronous, so a turn with tool calls ends the slice; `recall` and a tool that does not exist are answered within
the slice, up to three rounds. A stored tool message names its call by the id the assistant message carries.
Reasoning content is dropped before storing. Tool-call arguments a model JSON-encoded twice, or wrote as a bare
value, are repaired before storing, since a node refuses to render a thread with such a call in it. A reply cut
off at the output limit with no whole tool call in it gets a note and the slice goes on, at most twice.

The model takes a turn only once it can see every result of the calls it made last: tool results alone never start a
turn while a call of the latest turn is still out; they are stored and wait (a person's message, a wake-up or a
refocus still wakes the thread, and sleeps are not waited for). Otherwise the model acts without its newest results,
concludes its calls did nothing and repeats them. Repeats are watched as a harness bug, not steered away: the
journal warns from the fourth identical call (same name and arguments) in a row, and a goldfish at its twenty-first
is ended, its elephant getting the error.

**Router.** Each node's `/health` is read every second: `slots`, `busy_slots` and `queued` count every client's
requests, so a node shared with other work counts as busy; between polls the router adds what it started itself;
`context_length` sets the budget below. A thread's next turn goes to the node that served its last one, where its
state is in the prompt cache. A new thread goes to the node with the most free slots, nodes with the cache transfer
routes first. While a thread's node is full the turn waits in the router, and a slot that frees there goes to it
before any newcomer, for at most `newcomer_wait_s` (30, plus up to half again at random): past that a newcomer
takes any free slot, or steady load would never place a new thread (a goldfish, an expert's freeze); where two or
more threads wait for their home, one slot stays theirs, and with no free slot at all an overdue newcomer queues on
one node. A thread whose node is down or held counts as a newcomer. A thread marked priority (`POST /threads/{id}/priority`, in memory until a restart) goes first: while one of its turns waits for a slot no other turn takes one, and it takes any free slot, one that home-waiters want included. A thread leaves its node only when the node is full and the thread is under `spill_tokens` (it
reads its prompt again) or, with `move_threads`, by taking its state along: `POST /cache/export` on the old node
streams into `POST /cache/import` on the new one (strixite with `cache-xfer = on`), then the turn runs there. A node
that refuses a connection is down until its `/health` answers; a turn that never reached it goes to another. While
no node serving its model is up a turn waits `node_down_wait_s` (120), or the longest `down_wait_s` that
`node_meta` gives a node of that model. A reply a node breaks off is an error the slice retries, never a
shorter turn. With `grant_cmd` set (for example `["pod-worker", "may", "{node}", "--owner", "freefitty"]`), each
node's command runs every `grant_check_s` (5) seconds; while it exits 1 the node is held (a command that cannot run,
or runs past 10 s, counts as free): it gets no new, spilled or
moved threads, and a thread living there leaves at its next turn, with its state when the node can export it.

**Parking.** A node's prompt cache spills to its own disk, but that tier is small: the drive is shared with
weights and the cache caps its writes an hour, so under memory pressure a sleeper's state is dropped and its next
turn reads the whole prompt again (on the pod on 2026-11-07, 12 of 621 resumes on one node came from disk). With
`park` (on) the daemon keeps quiet threads' state itself: `POST /cache/export` on the thread's node, with its last
request as the body, writes `<park_dir>/<thread>.kv` (default `<data_dir>/parked`), and the next turn first
`POST /cache/import`s that file into whichever node it is placed on. A thread is parked at once when it sleeps at
least `park_after_s` (300) or waits on events with nothing else out, and after `park_after_s` when a tool call (a
goldfish, a commission) is out or its turn ended with nothing pending, if no turn ran in between and none is running
or waiting for a slot. A thread with both a wake-up and a call out is not parked.
A shorter sleep is never parked: the node still has it, and neither is a closed thread. The request the export is
sent with is rebuilt from the store when the park fires, so a sleeper holds nothing in memory meanwhile, and an
export that comes back shorter than the thread's last prompt (an older entry) is not kept. Before a restore the
daemon asks the node how much of the conversation it already holds (strixite `POST /cache/probe`); a node holding
at least what the file covers is sent nothing (`restore_skipped`), which saves up to gigabytes over the rail when a
thread wakes on the node it slept on. Files are written under a temporary name and renamed; a turn that commits removes its thread's file,
deleting a thread removes it, the hourly pass (see Experts) drops a file made under a system prompt or tool set
no longer in use or whose thread is gone, and past `park_max_gib` (64) the least recently used go. On because the node's own disk
tier was measured failing on the pod; a node without cache transfer answers the export with an error and nothing
is parked. `/health` counts `parked`, `park_missed` (the node held nothing), `restored`, `restore_skipped` and
`restore_failed`.

**Tools.** Each server in `tool_servers` publishes `/.well-known/rap-toolset`; the last manifest read is kept for a
server that is down. An invocation is posted to the
manifest's endpoint after its slice committed; one the server does not acknowledge is posted again with a growing
wait (30 s to 10 min) and, after twelve tries or a 4xx other than 429, becomes an error result so the thread is
not stuck forever. Manifests are read at start and every `toolset_refresh_s` (300, at least 10).
On start the daemon resumes invocations a crash left unacknowledged. Results come back to
`<callback_base>/rap/callback/<callback_token>` (or with the token as a bearer); a result posted twice queues once,
however close together; the callback also takes RAP's `subscription_event`, `oauth` and `user_choice` messages. A
result for a call the thread no longer waits on is said to the model in the user role.
Delivery to a tool server is **at least once**: a crash between posting an invocation and recording its
acknowledgement, or a lost acknowledgement, posts it again with the same `call_id`. A tool whose action must not
happen twice (a payment, a deploy, a message) dedupes on `call_id`; reads and idempotent writes need nothing.

**Images.** A result may carry images beside its text: MCP `{"type": "image", "data", "mimeType"}` parts or
`image_url` parts with a `data:image/...` URL, up to four per result. The tool message says how many follow, and
after the batch's tool messages one user message per result shows them as `image_url` parts, so a strixite node
with vision on sees them. A remote image URL is named in the text and not sent (a node asked to fetch it could be
pointed anywhere). Each image counts as 1400 tokens in the estimate, not its base64 bytes, and a request shows only
the newest `keep_images` (4) as pictures; older ones become a line saying they left the view. The line moves only at
the view's named events (a cut, a paging batch, a refocus): between them a new image never turns an older one into a
stub at an old position, so images are shown beyond four until the next event rather than breaking the prefix on
every image-bearing turn. The callback body
limit (`max_body_bytes`) bounds what a tool can send, so a tool should downscale before it posts.

**API** (`listen`, loopback by default; a front door with TLS is Caddy's or a wireguard peer's job):

```
POST /say                 {text, thread?, expert?, metadata?}  -> {thread, id}         operator token
GET  /output?wait&thread  the oldest uncollected reply; removed once sent; 204 if none  operator token
GET  /threads             id, size, pending calls, last message kind, node, closed      operator token
GET  /threads/{id}        meta: live_tokens, prompt_tokens, window, compaction, expert  operator token
GET  /threads/{id}/messages[?view=live][&tail=N]   the store, or what the model sees    operator token
PUT  /threads/{id}/messages {messages}   a history for a thread that has none           operator token
POST /threads/{id}/compact {focus?}      DELETE /threads/{id} (409 while a slice runs)  operator token
POST /threads/{id}/cancel                stop its running slice now; nothing committed  operator token
DELETE /threads/{id}?cancel=true         cancel the running slice, then delete          operator token
POST /threads/{id}/priority              its turns go first for slots; DELETE unmarks    operator token
GET  /dlq                 POST /dlq/redrive, /dlq/drop with {thread?}                  operator token
GET  /experts             POST /experts/reload, /experts/{name}/warm, /experts/from-thread/{id}   operator token
GET  /prompt              each hoisted command's version, and whether its output changed  operator token
POST /rap/callback        or /rap/callback/{token}: a tool's result                    callback token
GET  /health              status, build, uptime_s, model, priority, tools, prompt.hoisted, nodes, router counts, queue, store
GET  /metrics             Prometheus text: slices by outcome, slice/turn/slot-wait histograms, per-node turns and cached tokens
POST /admin/backup        a consistent snapshot of the store into <data_dir>/backups                    operator token
```

A reply carries `metadata.thread`, `metadata.reply_to` (the id `/say` returned for the message it answers),
`metadata.kind` (`text` or `error`) and the thread's own metadata. A message is at most 256 KiB, a body 1 MiB
(`max_body_bytes`), `metadata` a JSON object under 4 KiB, a thread id 1-128 of `[A-Za-z0-9._-]`; `/output?wait` waits
at most 60 s. `/health` per node: `up`, `slots`, `mine` (turns it runs for this daemon), `held`, `xfer` (takes cache
transfer), `serves` (its model), `context_length`, load and waiting; the router counts `kept`, `new`, `spilled`,
`moved`, `move_failed`, `waited` and `broken`; the queue `oldest_runnable_s` and `oldest_due_s`.
Nothing user-supplied is logged.

## What the model sees

The store is the truth and is never edited. What the model sees is a *view* of it: the shared prefix (the system
prompt, then an expert's reading if the thread has one), then bridges, then the stored history from some index on.
The node's KV cache is derived from the view and worth exactly as much as the view's stable prefix, so every rule
here protects that prefix: the system prompt is the same bytes for every thread, tool definitions are sent in one
order, a thread stays on its node, state moves with it over the rail, and the view's head changes only at a few
named events, each one cache miss, never one a turn. The transcript is durable; the cache is a cost, never
corruption.

The system prompt is the built-in one (or `system_prompt_file`), then `system_prompt_extra`, then the output of
each of `system_prompt_commands`, run at start. A shared tool describes itself once and FreeFitty hoists only what
every thread needs: on the pod `akb describe --brief --verbs read,comment,move,query,list,update` (1.9 KB, usage
lines generated from akb's own parameters, and the rules it enforces), with `akb help <verb>` for the rest. The
first line of each output names the tool's version; `/health` shows it, and `freefitty prompt` runs the commands
again and marks the ones whose output changed (a tool upgraded: restart to hoist it); `GET /prompt` shows the same.
A command that fails, prints nothing or runs past 30 s stops the daemon's start. Any change to the prompt is one
cache miss for every thread and a refreeze of every expert.

Every request carries a sampling `seed` derived from the thread and the index its reply is stored at (`engine::turn_seed`), so a replay of a stored transcript samples each turn with the same seed; a node picks a random one without it.

Every turn records the instructions it ran under: the id (a hash of the system prompt and the thread's tool
definitions, the same one an expert's frozen context carries) goes on the thread where it changes (`GET
/threads/{id}` `instructions`: [message index, id] pairs), and the text is kept once in
`<data_dir>/instructions/<id>.json`. A study of how threads use a tool reads only the turns told the same thing.

Three mechanisms move the view's head. Each has a default and the reason for it, decided from the pod's own
numbers, not from a reflex either way.

**The budget window, on.** A node's context is finite and a pod thread is not: a card's thread runs for days and
its tool results pile up. When a thread's next view would pass `window_tokens` (default: the nodes' context length
minus `max_output_tokens` and an 8k margin, 237,568 on today's nodes with the 16k default), its oldest whole units leave the view until
it is under `window_trim_tokens` (default five eighths of the budget), and the cut then holds until the budget is
crossed again. A unit starts at an input, a person's message or a tool result, so the view never opens with the
model's own words; a result whose call left the view gets a stub call so the template still sees a call before
every result. The view opens with a `[Paged.]` notice that names the range that left, copies the user messages
among it in short, and says how to get any of it back: the `recall` tool returns stored messages by index, exactly
as written, answered from the store within the same slice (a fault costs one more model pass, no tool round trip,
at most 16 KiB a call). The recalled text is appended to the history as the call's result, never re-inserted at
its old position, so a fault-in costs no recompute of the prefix: the view stays the bytes the node already holds,
plus the answer. That, more than anything, is the case for the window and recall over a refocus, which writes new
head bytes and pays a prefill. A recalled message below the window's start is pinned; at the next cut it becomes
resident, shown in the notice region (exact text, newest 32, 16 KiB) until the cut after, so a fact recalled
again and again is held once instead of appended each time. On because it is the guard against the wall and does
nothing until a thread reaches it, and pod card threads do reach it. `window = false` turns it off;
`max_context_tokens` (0: the node's context minus `max_output_tokens`) is the limit the budget starts from.

The estimate behind it is bytes / 4 scaled up by how the node counted (the daemon holds no tokenizer, so this is the
weak point for code-heavy threads, whose bytes run denser than prose until the node has counted one turn): the
thread's own ratio after its first turn,
else the latest ratio any thread saw (capped at 2.0), else `est_scale_default` (1.3; measured 1.26 to 1.61 on pod
traffic). A ratio is learned only from a turn whose messages outweigh its tool definitions, which the node counts
too. A node that still answers 400 with its count recalibrates the thread, cuts, and sends once more within the
slice. The thread is closed when nothing is left to cut, on a second overflow, when the 400 carries no count, when
the window is off, or when the estimate is over the limit before anything is sent. `GET /threads/{id}` reports `window.from`, `window.cuts`, `prompt_tokens`
(the node's count) and `live_tokens` (the estimate); the journal logs each cut with the view before and after.

**Refocusing, off.** On strixite the coherence of the context matters before its size: a long thread fills with
superseded ideas and finished branches. With `compaction = true` a thread can refocus: the model writes notes
around a focal point, and from then on those notes stand in for the conversation before them, as the model's own
assistant turn carrying any tool calls still awaited. Three ways in: the model's `refocus` tool, the operator's
`freefitty compact THREAD --focus "..."` (a focus is at most 2,000 characters), or `compact_at_tokens`; the last
`compact_keep_messages` (2) messages stay after the notes. The instruction (`src/compact.md`, or
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

An expert may also be a persona: `[[steer]] vector = "villain24", scale = 6.0, layer = 24` (layer optional) names
activation-steering vectors the nodes hold (strixite `steer-dir`). Every request of a thread born from it carries
them as strixite's `steer`, its freeze reads it steered, and the steering is part of its version, since the nodes
cache steered state under its own key. The expert is then the bundle of everything two threads must share to share
cache: prompt bytes, tool list, steering. A node refusing the steering (no such vector) closes the thread at once
rather than retrying. Refocus runs unsteered: its notes are stored text the thread will believe, so memory stays
separate from voice and a later freeze of those notes bakes in no persona (the price: that one request reads the
view without the steered cache). Steering touches every token, tool-call JSON included, so `commission_expert` does
not steer (`commission_steer`, off): no measurement yet compares a steered goldfish's verdicts and malformed calls
with a plain one's. With it on, a commission's scale is bounded by `max_steer_scale` (8) and its vector
names are checked against what every up node lists (strixite `GET /steer`; the intersection, since a thread may
run on any node, and a node that cannot list counts as holding none). Steering is a composition barrier: experts
under different steering have different cache keys and cannot be spliced. The vectors are the one durable source
that is not in the transcript: a reader of the store cannot rebuild a steered view without the steer-dir, so the
vectors must be kept like the store.

`steer_scope = "prefix"` steers the reading only: every request still carries the vectors, with strixite's
`steer_scope` naming the shared prefix (system prompt + the expert's messages) as the steered span, so the nodes steer
what the expert read and nothing after it - the thread's turns and every generated token are the model's own. The
cached expert is the steered reading; the goldfish that attends to it answers plain. It is its own version (another
cache key), its refocus keeps the fields (the generation is plain anyway, so the view serves from cache), and a freeze
or commission of one refuses unless every up node answers `GET /steer` with `"scope": true` - an older node would
ignore the field and steer the whole thread. Measured on the pod reviewer: whole-thread steering past scale 2 made it
answer without reading the card; the prefix scope exists to test whether the reading alone carries the direction.

`model = "dsv41"` in a definition puts the expert, and every thread born from it, on the nodes serving that model.

Freezing reads it once on one node and exports that node's state at the end of the acknowledgement (strixite's
checkpoint at the start of the next user message) into `<data_dir>/experts/<name>@<hash>-<context>.kv`. strixite
answers a turn before it caches the turn's entries, so an export is retried until the entry it returns covers the
reading (about 10 s at most); parking a thread waits for its last prompt's entry the same way. A thread
born from it (`freefitty say "..." --expert NAME`) gets that state imported into whichever node takes its first
turn, which then reads only its own message: on any node, after a restart, however much the expert read. The
context is a hash of the system prompt and the tool definitions, which render before the expert; when either
changes the expert is frozen again, and a thread born meanwhile reads the expert itself (and triggers the freeze).
Every current expert without state is frozen once a node is up, after `experts reload`, and after `experts freeze
THREAD NAME`, which makes a thread's refocus notes what an expert has read. The stored messages stay as its
provenance: every born thread sends those exact bytes, and the state is rebuilt from them. Versions are never
pruned: a thread keeps the bytes it was born with when a file changes. Frozen states are pruned, at start and
hourly: a state that can never be imported again goes - one of a replaced version, or one not written or used since
the system prompt or tool set last changed (`<data_dir>/state-epoch.json` records since when the current pair has
held; using a state marks it, and at start every current expert's state in the current context is marked). Age
alone removes nothing. `expert_states_max_gib` (32) is a backstop on what is left, least recently used first;
each pass logs what it kept and freed. `experts warm NAME` puts the frozen state
into every up node (a reading on each, where nodes cannot take state); `warm_on_start` does that at boot.
`experts list` shows each expert's frozen bytes. Off until `experts_dir` is set: the mechanism is cheap, and the
practice of choosing what an expert reads is the work.

**Goldfish.** Elephants are long threads that hold a piece of work; goldfish are threads born for one brief,
which answer once and are gone (the names are from Dave Rensin's
[Elephant-Goldfish Model](https://research.google/pubs/elephants-goldfish-and-the-new-golden-age-of-software-engineering/);
see also [google/elephant-goldfish-model](https://github.com/google/elephant-goldfish-model)). Three built-ins make that general:
- `experts` is the catalog an elephant chooses from: each expert's name, its line, and the tools its goldfish may
  use. It is a tool, not part of the tool definitions, so a new expert never changes the bytes every thread
  shares.
- `goldfish {expert, question, documents?, tools?}` makes a thread `<elephant>.gf-<expert>-<call>` born from the
  expert (its frozen state is imported into the node that takes its first turn), puts the documents (paths on
  this host, 256 KiB each, 1 MiB together) and the question in its first message, and ends the elephant's turn;
  several in one turn run at once. When the goldfish ends a turn with nothing out, its text returns as the call's
  result and the goldfish is closed; if it fails or dead-letters, the elephant gets the error as the result.
- `commission_expert {name, description, reads, ack?, tools?, steer?}` (each read file at most 1 MiB) registers a new expert at once, writes its
  definition to `experts_dir` with the commissioning thread named, and freezes it; the elephant sleeps until the
  freeze ends and wakes to the expert's name and version.
- An expert whose definition has a `[tool]` table (`name`, `description`, JSON-schema `parameters`, `ask?`) answers
  that tool: every thread sees it beside the other tools, and a call to it is a goldfish of the expert briefed with
  the call's arguments, whose answer returns as the result. A RAP server's tool of the same name wins; a goldfish cannot
  call one. `fallback` names an expert that answers instead while no node of the expert's model is up and free
  (on the pod `ark_lookup` is the librarian on the StrixFold pair, and the same librarian on the cluster while
  the pair is down or held for a window).

The paths a model names (a goldfish's `documents`, a commission's `reads`) are resolved, symlinks and `..` included,
and read only under `read_roots` (empty, the default: no files at all). FreeFitty's own data, park and config
directories stay closed whatever `read_roots` says, since the config holds both tokens; expert definitions in
`experts_dir` stay readable. Files the operator names (definitions, prompt files) are not checked.

A goldfish sees and may call only the plain built-ins (the sleeps, `recall`, `refocus`, `experts`) and the tools
it was granted: the call's `tools`, else its expert's `tools = [...]`, else none. An expert's tool list is part of
its version, since the definitions render before it, and it is frozen in that context. `goldfish` and
`commission_expert` are never a goldfish's, whatever it was granted: it says in its answer which expert should look
next, and the elephant decides. An expert without a tool list gives a thread born from it with `/say` every
tool. A tool server publishing a built-in's name has that tool left out, with a warning.

**Paging old tool results, on.** Most of a pod thread is tool output, and most of it is read once. A result the
model has answered `page_results_after_turns` (4) model turns ago and that is over `page_results_min_bytes` (500)
is shown as a handle in its place, `[Paged out: name(args...) answered with N bytes; recall(#i) returns it]`, under
the same call id so the template still pairs them; the text stays stored and `recall` brings it back, after which
that index is pinned and shown in full however old. An error result is never paged: the model needs it while it
debugs. Paging advances in batches of `page_results_batch` (32) results, or all at once when the window cuts,
because every advance changes the view at an old position and costs the node a recompute of everything after it,
about as much work as a dozen turns' worth of the attention it saves; between advances the view is stable and the
cache serves it, and a cut, which pays that recompute anyway, pages everything it can. On because the replay below measured it on the pod's
own threads first; `page_results_after_turns = 0` turns it off. A goldfish never pages: it lives a few turns on its
expert's frozen state, and a paging batch would only break the cache it was born to reuse. `GET /threads/{id}` reports `paging.before` (the
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
stray. pod-ff (the pod's card runner) and pod-hands (the pod's RAP tool server) speak this API. They were written
for freefinity, FreeFitty's predecessor, and keep its variable names: `/data/pod/freefitty/.env` carries
`FREEFINITY_PORT` (this daemon's port), `BRIDGE_TOKEN` (the operator token) and `CALLBACK_TOKEN` for them, and their
units set `FF_STACK` to that directory. pod-ff reads thread sizes and last turns from `/threads` and prunes with `DELETE`. The Isekai'd
director rounds run on the same daemon under thread ids starting `isk-`.

Upgrading: build on a machine with the pod's glibc or older, copy the binary over, `install` it, and
`systemctl --user restart freefitty`; turns already at a node finish first, and a turn still waiting for a slot
gives up at once and re-queues as it was (no attempt counted), so a stop takes as long as the longest running turn,
not the slice timeout. A kill is safe too: a slice commits all or nothing, so an interrupted one just runs again. Keep the
old binary as `freefitty.bak-<stamp>`. Reading the daemon: the journal has one line per turn (`(N prompt tokens, M
from its cache) in R s, after S s waiting for a slot`), one per cut, one per failed slice; `freefitty status` shows the nodes and
counts; `freefitty threads` the threads with their last message kind, so a watcher sees one that stopped talking
(`assistant_empty`, or `tool_call` with nothing pending). Dead letters are read with `freefitty dlq` and put back
with `dlq redrive`, which also reopens a closed thread. A slice
that panics (a poison event, a bug) is caught and counted as a failed attempt like any other, so it dead-letters at
`max_attempts` instead of looping.

**Metrics and logs.** `GET /metrics` (no token, like `/health`; the daemon listens on loopback) is Prometheus text:
`freefitty_slices_total{outcome}` (done, requeued, transient, fatal, panic), the histograms `freefitty_slice_seconds`,
`freefitty_turn_seconds` and `freefitty_slot_wait_seconds`, per node `freefitty_node_turns_total`,
`freefitty_node_prompt_tokens_total` and `freefitty_node_cached_tokens_total` (their ratio is the cache hit rate),
the router's counters as `freefitty_router_*_total`, the queue and store numbers as gauges, `freefitty_parked_bytes`,
`freefitty_uptime_seconds`, `freefitty_tools`, `freefitty_nodes_up` and `freefitty_build_info{version}`. Context pressure: `freefitty_context_budget_tokens` (the view budget),
`freefitty_thread_prompt_tokens_max` (the largest open thread with a turn in the last 6 h, by the node's own count),
`freefitty_threads_over_half_budget` / `_over_three_quarter_budget` (same threads), `freefitty_window_cuts` (all threads) and
`freefitty_threads_closed_for_context`: what says whether a bigger context per slot would earn its memory. Counters restart at zero with the daemon. `FREEFITTY_LOG_FORMAT=json` in the
unit's environment writes one JSON object a line instead of text, for a log shipper.

**The store's lifecycle.** One redb file, `<data_dir>/freefitty.redb`. Its layout version is the meta key `schema`
(1 today): a binary refuses a store written by a newer layout, and a change of layout ships its migration in
`Store::open`. redb never shrinks its file; `compact_on_start` (on) gives the free space back each start, which takes
seconds. A consistent snapshot of every table is written to `<data_dir>/backups/freefitty-<utc>.redb` every
`backup_every_h` (24) hours and on start when the newest is older, keeping `backup_keep` (7); `freefitty backup` takes
one now while the daemon runs. Restoring: `systemctl --user stop freefitty`, `freefitty restore <snapshot>` (it
refuses while the daemon answers and keeps the current store as `freefitty.redb.before-restore-<utc>`), start. Parked
state and frozen experts are files beside it and are derived: losing them costs prefills, not data.

**Provenance.** Every binary carries the commit it was built from: `freefitty --version`, `/health` `build` and
`freefitty_build_info` say `0.1.0 (<sha>)`, with `-dirty` when the tree had uncommitted changes, so the binary on
pod-control can be told from any other build. CI (`.github/workflows/ci.yml`) runs the tests and `cargo deny`
(advisories, licenses from `deny.toml`, banned and unknown sources) on every push; a `v*` tag builds the release on
Ubuntu 22.04 and publishes the binary with its CycloneDX SBOM, `SHA256SUMS` and `VERSION`. A deployed binary is
checked with `sha256sum ~/.local/bin/freefitty` against the release's `SHA256SUMS`; a binary built by hand carries its
commit but no published checksum.

**Secrets.** The operator and callback tokens (`operator_token`, `callback_token`: 16 or more URL-safe characters,
and different from each other) live in `freefitty.toml` (0600) and the pod's `.env` for pod-ff and
pod-hands. Rotating one is: write the new token in both places, restart the daemon, restart the tool servers that
call back; nothing reloads tokens at runtime, by design, so a rotation is one visible restart.

## Design rules

Each of these is a property of the design, not a patch on it.

- A thread never runs twice at once, and several messages of one thread run together in order: the queue's in-flight
  set and per-thread batches. A killed slice leaves nothing invisible; its events are simply due again.
- A redelivered event never runs its turn again: processed ids are written in the same transaction as the turn and
  kept four times `reply_retention_days` (4); uncollected replies go after `reply_retention_days`.
- Dead letters, and deleting them, survive a restart: they are rows in the same transactional file.
- A reply cut short is an error, not an empty turn: replies are not streamed, and a broken body fails the slice. A
  thread that stops talking is visible: `/threads` shows each thread's last message kind.
- A green status does not hide starvation: `/health` reports how long the oldest runnable event has waited, and a
  wait for a node slot is never counted against the thread.
- A model that cannot be reached costs a pause, not retries: the scheduler starts nothing while no node is up, and
  the router waits for a node that blinks.
- Nothing leaves the process before the slice commits, so the model is never asked for a reply that would go nowhere.
- A tool server that is down does not lose a call: invocations are stored and re-posted with backoff; a result
  posted while the daemon is down lands when the server retries, as pod-hands does.
- A node's `/health` can lack slot counts (`node_slots`, 2, stands in), and other clients load it: both are counted.
- Nothing in the system prompt differs between threads, or the nodes' caches could not share it.
- A thread outlives the node's context: the window, not a closed thread, is the answer to the wall, and the estimate
  learns the tokenizer from the node rather than trusting bytes / 4.
- A core task that dies takes the process with it.

## Tests

`cargo test` runs the unit tests (the window's cut, stubs, notice, paging boundary, batch and handle, and replay; the
estimate's scaling; argument repair; the router's placement with held nodes; saved-state pruning and its epoch) and
`tests/e2e.rs`: the daemon against a scripted model node and a RAP tool server, in process. What the e2e tests cover:

- **Turns and the queue:** a plain turn; a sleep through the queue; a RAP tool whose result is posted twice;
  double-encoded arguments; an unknown tool; a turn with nothing in it; three failures to the dead-letter queue and
  a redrive that keeps the input's id; two threads at once; a restart with a wake-up pending; a slice that panics,
  counted as a failed attempt and dead-lettered; a stop that does not wait for a turn still queued for a slot, which
  runs after the restart uncounted; a running slice cancelled at once, its thread deleted, no goldfish or expert
  left behind.
- **What the model sees:** a turn's calls answered together, and a late result not starting a turn while the newest
  call is out; a reply cut off mid-call going on in the same slice; a goldfish stuck on one call ended and its
  elephant told; every turn tracing to the instructions it ran under; every reply carrying a seed a replay
  recomputes.
- **Refocus:** from the tool and from the operator, with the pending call bridged and its late result answered;
  notes cut off and asked again; nothing to fold; the token trigger; failed notes.
- **Window and paging:** whole units cut at the budget; the notice carrying the first message; the cut point holding
  across turns; `recall` returning the exact text within the slice; the replay agreeing with the live run; old
  results paging out in a batch as handles, an error result staying, a recalled result pinned back in full; a
  node's count correcting a wrong estimate so the turn runs; a full context closing a thread with nothing to cut.
- **Experts and goldfish:** experts sharing their prefix bytes; version pinning across a file change; warm on start;
  an expert frozen from a thread's notes; a state frozen once that each born thread's first turn imports byte for
  byte, a later turn importing nothing, and a restart keeping it without a second reading; an elephant reading the
  catalog and asking a goldfish that holds only the plain built-ins; a goldfish that tries to ask a goldfish; an
  unknown expert answered with the catalog; a commissioned expert frozen, written and granted its tools; an expert
  tool falling back while its model has no free node; a model-pinned goldfish running only on nodes serving its
  model.
- **Steering:** a steered expert's every request carrying it, a plain thread's none, the steering a version of its
  own; prefix-scoped steering, and its refusal on nodes without the scope; a commissioned persona checked before it
  exists; a node refusing a thread's steering closing it at once; a persona's vector checked on every node.
- **Nodes and state:** a wait for a node slot longer than the slice timeout that is not a failure; overdue newcomers
  leaving one slot where two threads wait for their home; a sleeper parked and its exact bytes restored when it
  wakes, nothing parked when the node holds nothing, its file removed with the thread; a node already holding a
  state not sent it again; a node behind an API key getting its bearer token; a node naming another cache-routes
  version not used for transfers; a tool's images reaching the model;
  the operator marking and unmarking a priority thread.
- **Operations:** `/metrics` after a turn (outcome, cached tokens per node, histograms); a snapshot on demand that
  reopens with its schema; a store from a newer schema refused.

`FREEFITTY_NODES=http://a:5300,http://b:5300 cargo test --test real_nodes -- --ignored` proves the cache move on two
real strixite nodes.

## License

Apache-2.0.
