# C.R.E.A.M.: Cache-Resident Experts As Memory. A cache-native architecture for autonomous agents on an air-gapped Strix Halo cluster

2026-11-07, revised 2026-11-11. The revision adds a second engine on the same state contract (section 3.1), the experiments run since the first version (sections 5, 6, 8 and 9) and what running it taught (section 10).

## Abstract

We describe a small, air-gapped cluster of AMD Strix Halo nodes that runs autonomous software agents around one organizing idea: the inference engine's cached state is the system's working memory, and everything else is a view onto it. Four layers carry the idea.

- **strixite** is the engine, serving one 262k-context model per node with a token-prefix prompt cache that survives in RAM and on disk and can be exported between nodes over a 100 GbE rail. A second engine, StrixFold, serves a much larger model across two nodes behind the same three state routes, so the contract is no longer one engine's.
- **FreeFitty** is a durable actor system whose actor state is a transcript and whose step function is the model; it never edits the transcript and spends all its design on keeping each thread's cache prefix stable.
- **The Elephant-Goldfish Model (EGM)** is the division of labour: long-lived elephant threads hold context, short cold goldfish threads do work and die.
- **The Optical Memory Fabric (OMF)** is the research track asking how much memory a cached token can carry, through rendered pages, vision tokens and direct reuse of KV segments.

The paper states the thesis, reports the measurements that support it so far (bit-exact KV restore 2.5 to 17.7x faster than prefill, 6x state compression with facts intact, order-independent splicing of two segments, a three-level chain of state diffs that reproduces prefill exactly at 30x the speed, ~95% cache reuse on live work, a 65k-token reading restored on a two-node DeepSeek engine in 2.2 s against 62 s cold), and names the four experiments that join the layers: an expert that reads rendered pages, a thread composed from several frozen experts, goldfish born from experts that elephants can commission (now built and in daily use), and revert-with-note as an alternative to compaction (piloted). Section 10 collects what the first four days of running this taught, including two negative results: steering an expert to be brief made it stop checking its sources, and a week of cache experiments measured a mechanism the code had already decided.

## 1. Introduction

The pod is a five-worker cluster with no route to the internet that runs its own software agents against a kanban board, around the clock, on open weights. That setting forces three constraints that larger deployments can buy their way out of. Compute is fixed: each node is one Strix Halo APU, so a token of prefill costs the same today as it will next year. Context is finite but work is not: a card's thread runs for days and its tool output grows without bound. And nothing can be fetched: every model, paper and wheel arrives through a courier mirror called the ark, so the system must get more out of what it already holds.

Under those constraints the expensive operation is reading. A 262k-token context prefills at roughly 1,400 tokens a second, so a thread that re-reads its history each turn spends minutes before it says a word. The engine's prompt cache turns that into a one-time cost if, and only if, the bytes it is handed are the same bytes as last time. Every layer described here is shaped by that single fact.

The architecture therefore inverts the usual relationship between memory and model. In a retrieval-augmented design the store is primary and the model's context is a scratch view of it. Here the engine's cached state is primary: it is what a thread wakes into, what an expert is, what moves between nodes, and what the memory research is trying to pack more densely. The transcript, the board and the documents are the durable sources from which that state is derived and against which it can be rebuilt. We call this cache-native, and the rest of the paper describes the four layers that implement it and what has been measured on each.

Measurements cited are from the pod's own reports with kept raw data and a verify script, except the few marked as operational observations. Dates are 2026 unless stated.

## 2. Substrate

Six identical boxes, one of them the controller, joined by a lossless 100 GbE rail. Each is an AMD Ryzen AI MAX+ 395 with a Radeon 8060S (ROCm gfx1151) and 128 GB of unified memory, of which about 124 GiB is addressable by either CPU or GPU. Unified memory is what makes the design possible on this class of hardware: a model's weights, its KV state and the prompt-cache files share one pool, so a cache entry is a memcpy away from the GPU rather than a PCIe transfer.

| Part | What it is | Role |
| --- | --- | --- |
| Controller | Strix Halo, 2 TB NVMe | scheduler, FreeFitty daemon, board, weight staging |
| Five workers | Strix Halo, NVMe scratch | three serve strixite (2 slots each); two (F and A) serve StrixFold as one tensor-parallel pair |
| Rail | Mellanox SN2700, ConnectX cards, RoCE v2, MTU 9000, PFC + ECN on the rail ports | cache export/import, tensor-parallel runs, NFSv3 at ~1 GB/s |
| Management | 2.5 GbE switch, separate subnet | ssh, board, tool servers |
| The ark | offline mirror of models, papers, wheels and repos, pulled by a courier host | the only way anything enters |

The pod is air-gapped by construction, not by firewall. The controller has no default route, workers route only through the controller, external DNS fails everywhere, and unprivileged user namespaces are disabled. The posture is a security decision and a design input: the system cannot look anything up, so whatever it knows has to be resident, cached or on the ark.

One data-path rule governs everything that touches weights or state: slow storage feeds NVMe, NVMe feeds the rail, the rail feeds VRAM. A load that reads a USB drive or an NFS export of one directly is wrong by definition. The same rule explains why cache transfer between nodes is a first-class operation: a thread's 100 MB to 7 GB of state crosses the rail faster than the node could recompute it.

## 3. strixite: the engine and its cache

strixite is the engine on every worker: a native ROCm server for Qwen3.8-Flash-Next with a 262,144-token context, two slots per node, and a prompt cache that is the architectural hinge of everything above it. It is third-party code (shawnshekari/strixite) that the pod wires, measures and patches on merit.

The model matters for the thesis. Qwen3.8-Flash-Next is a hybrid: 36 gated-delta-net (GDN) recurrent layers carry most of the sequence in a fixed-size state, and 12 full-attention layers plus a multi-token-prediction draft head keep per-token KV rows. Measured on the pod, a cached segment's size is exactly 115.67 MB plus 27,456 bytes per token: the constant is the recurrent state, the slope is the KV rows. For an 800-token segment the recurrent state alone is 82% of the file. Synthetic needle probes read back with recall 1.00 from 2k to 128k tokens, so the recurrent summary is not lossy at the lengths the pod uses.

The prompt cache is keyed on the pure token-id prefix plus a weights fingerprint. There is no conversation id, so any thread whose rendered prefix matches reuses the state, across threads and across processes. Entries live in a RAM LRU and are written back to disk after an hour idle; there is no time-to-live, so entries leave only by eviction or process exit. The disk tier refuses writes that would breach a 32 GiB free margin. Inside a live process, wake hits of 98.3 to 98.6% were measured at idle gaps up to 30 minutes.

Three features turned the cache from an optimisation into a substrate:

- **Export and import.** `POST /cache/export` streams a thread's state to another node's `/cache/import` over the rail. Full-state restore is bit-exact against re-prefill, and moving a live thread between two nodes took 3.7 s.
- **Constrained decoding.** v0.5.0 accepts `json_schema` response formats, which lets code, not prose, gate what an agent writes to the board.
- **Vision on every node.** Every worker serves image input, so a rendered page is a legal prefix component anywhere, not on one pinned worker.
- **Tool-call grammar.** Since 2026-11-09 the pod's strixite build constrains tool-call syntax during decoding, so a malformed call cannot be produced. It was built after a steered expert broke its first tool call (section 10).
- **Steering vectors.** A request may add named activation vectors at a layer and scale, over the whole thread or over its first k messages only (`steer_scope`). A steered expert's vectors are part of its version. What steering is good for is still open; section 10 says what it is not good for.

The fleet runs strixite v0.7.0 with the pod's patches. The same build boots at 524,288 tokens per slot (the model's trained context with YaRN 2) on all five workers without memory trims; the fleet stays at 262,144 because no thread has needed more.

Published serving figures put one node at 44 tokens a second decode and 1,330 to 1,400 tokens a second prefill at 64k, with tensor-parallel over two nodes at 58; a later measurement at a full context pool gave 39 cached decode. Those numbers are why prefill avoided is the currency of the system: at 1,400 tokens a second a 250k view costs three minutes, and a cached one costs a fraction of a second.

### 3.1 StrixFold: the same contract on a second engine

StrixFold is a third-party engine (Shoestring-dev/Strixfold, built on TensorFold), forked by the pod, that serves DeepSeek-V4.1-Flash, a mixture-of-experts model far too large for one node, tensor-parallel over two workers: rank 0 on F serves the API, rank 1 on A follows it over a link between the two boxes. It arrived after the first version of this paper and tested whether the architecture depends on strixite.

It does not. Between 2026-11-08 and 2026-11-11 StrixFold gained the same three routes (`POST /cache/probe`, `/cache/export`, `/cache/import`) and the same `/health` fields FreeFitty reads (slots, live streams, context length), and FreeFitty places, parks and restores DeepSeek threads through them with no engine-specific code beyond a per-node model name. Two differences from strixite are instructive:

- **The state is small.** DeepSeek-V4.1's compressed attention keeps a 63,488-token state in 63.4 MB per rank, about 1 KB a token, against strixite's 115 MB constant plus 27.5 KB a token. Exporting that state takes 0.4 s and importing it 0.3 s. The two ranks' state payloads are byte-identical.
- **Each rank keeps its own copy.** Kept states (`--learn`) are written by each rank to its own disk under the same key and read back from there, so a state never crosses the link between the ranks, and a learned reading survives a restart of the pair. A reply after a restart and a restore is byte-identical to the reply before it.

States are kept at 2,048-token chunk ends, so a cached prefix is reused up to the last chunk boundary, not to the last token; FreeFitty knows this per node (`keep_step`) and does not count the tail as a miss.

Measured on the pair at quiet power (2026-11-11): a fresh 65k-token prompt prefills in 62 s (1,055 tokens a second); the same 65k reading, learned once and asked a new question, returns its first token in 2.2 s with 63,488 tokens from cache. Decode runs 47 tokens a second on prose and 71 on code for one stream, and four concurrent streams share 67 tokens a second (85 to 89 while all four are producing). A goldfish born from a 70k-token DeepSeek expert takes 75 s cold and 7 s after its reading is learned, across restarts.

## 4. FreeFitty: the durable work host

FreeFitty is a durable actor system in one Rust binary whose actor state is a transcript and whose behaviour function is the model. A thread is its stored conversation. When something arrives for it (a person's message, a tool's result, a wake-up it scheduled) the model runs once over the conversation, the turn is committed in one transaction, and the thread holds nothing until the next event. The model has no loop; the scheduler is the loop and the model is a pure step. Sleeping is free however long it lasts, a retry re-runs a step that committed nothing, and exactly-once turns fall out of the commit boundary. It hosts every pod work card as a thread: since 2026-11-08 the pod's older session lanes are retired, and nothing else runs cards.

The store is the truth and is never edited. What the model sees is a view: the shared system prompt, an expert's reading if the thread has one, bridges, then the stored history from some index on. The node's cache is derived from that view and is worth exactly as much as the view's stable prefix, so every rule in the system protects that prefix. The system prompt is byte-identical across threads. Tool definitions are sent in one order. A thread stays on the node that served its last turn, and leaves either with its state over the rail or, while small, by re-reading its prompt elsewhere. The view's head moves at a few named events, each one cache miss, never one a turn.

Four mechanisms move the head, each with a default decided from the pod's own replay rather than reflex:

| Mechanism | Default | What it does | Why that default |
| --- | --- | --- | --- |
| Budget window | on | oldest whole units leave the view when it would pass the node's context minus output and margin; `recall` returns any of them by index within the slice | live card threads routinely pass 150k tokens and would otherwise hit the context wall |
| Paging old results | on | a tool result answered 4 turns ago and over 500 B becomes a handle; advances in batches of 32 | replay over 28 threads and 8,448 messages: such results were half of what nodes re-read; batch 32 keeps 45% of the saving and pays one recompute in ~30 turns |
| Refocusing | off | the model writes notes at a focal point that stand in for the earlier conversation | changes what the model believes; no measurement yet says when notes beat exact history plus recall |
| Preloaded experts | on (since 2026-11-08) | a named prefix (documents read plus an acknowledgement) a thread is born with; versions pinned by content hash; the reading's state is exported once and imported wherever a thread born from it starts | off until the practice existed; the review expert alone has served 257 threads, and three experts are warmed into every node at start |

The expert is the piece this paper turns on. Every thread born from an expert sends the same bytes after the system prompt, so a node's cache serves the first turn of each such thread however much the expert has read: a 100k-token reading is one prefill and then seconds per thread. Live card threads show roughly 95% cache reuse, and a fresh thread reads 719 of its 740 first-turn prompt tokens from cache. An expert is thus a memory with an address, a version and a residence, which is exactly the object the memory research in section 6 was trying to construct from the other direction.

Since the first version FreeFitty also parks: a thread going to sleep has its node state exported to a file, and its next turn imports it on whichever node takes it, so a long sleep no longer costs a re-read when the node has evicted the thread meanwhile (on the order of 200 parks a day, restores in under a second; an operational observation). It routes by model, so strixite and StrixFold threads share one scheduler, and a thread can be marked priority so its turns take the next free slot of its model. A thread born from an expert names the length of its shared prefix to engines that can keep a state there (`shared_messages`), so the reading is learned once even when no thread has yet asked the same question.

FreeFitty also carries images: a tool result may include up to four pictures, which the view shows as image parts to a vision-capable node, keeps as the newest four, and estimates at 1,400 tokens each. Experts cannot yet read images. Closing that gap is the first of the two experiments in section 9.

## 5. EGM: elephants and goldfish

The Elephant-Goldfish Model is a division of labour between agents that remember and agents that do not. It comes from Google's elephant-goldfish-model repository and David Rensin's essay, and the pod adopted it because it maps directly onto what the cache makes cheap and what it makes expensive. An elephant is a long-lived thread that holds the context of a piece of work: a card, a design, a world. A goldfish is a short thread born cold with one document and one question, which answers and dies. Elephants are expensive to build and cheap to keep because their prefix is cached. Goldfish are cheap to spawn because they share a persona prefix and carry nothing else.

On FreeFitty the pattern is a tool server. The elephant calls `goldfish {document, mode, question}`, ends its turn and sleeps. The server opens a new thread holding the document and the question, and when that thread answers it posts the verdict back as the elephant's next tool result. Three goldfish personas ship as modes, taken verbatim from the upstream repository: a hostile design critic, a fresh-reader comprehension audit, and an implementation-readiness gate.

An end-to-end test on FreeFitty's predecessor, against the same engine and cache, ran a design loop to convergence: the elephant drafted a small CLI tool, a design-critic goldfish failed it four times for real defects, and the fifth draft passed. Over the 1,314-second run the elephant slept 74% of the wall; no model call ran in it between issuing a goldfish and receiving its verdict. Cache reuse across the eleven model calls was 70%, climbing from 56% to 86% as the elephant's prefix grew, with fresh goldfish threads steady at 64 to 66% from their shared persona. A long-lived interactive session holding the same loop would have held a slice and a context for 22 minutes; the actor system held nothing. The goldfish server now runs as a FreeFitty tool server.

EGM also serves as a quality gate. The goldfish-intake experiment put a cold reader in front of a card, asking it to restate the card from its text and named files alone, and ran it retrospectively over twelve review-failed and eighteen clean cards. It caught two of the twelve misreads at a mean of 94.5 GPU-seconds with no false positives, and showed that nine of the twelve review failures were execution failures, not comprehension failures; it is adopted on the re-queue path, not on every pick. The same shape drives a generative game's creative director: one elephant holds a world's canon, goldfish generators produce slices of it.

What EGM contributes to the architecture is the claim that memory should be unevenly distributed. Not every agent needs the history; most need a document and a question. The cache makes the elephant affordable, the actor system makes its sleep free, and the goldfish keeps the expensive context from leaking into work that does not need it. The expert is where the two meet: a goldfish born from an expert borrows a reading as a cache hit instead of a per-spawn paste, and an elephant can commission a new expert the way it commissions any goldfish: it names the persona and the reading, sleeps, and wakes to the expert's name. The reading and the freeze happen in a builder thread. What the elephant holds is the north star and a catalog: each expert's name and a one-line description, enough to choose the right goldfish for a job and nothing of what that goldfish will read. The expert registry already stores exactly that pair, so the catalog is a view of it.

This is built. `goldfish {expert, question, documents?, tools?}` starts a thread born from a frozen expert, and `commission_expert {name, description, reads, tools?}` registers and freezes a new one while the elephant sleeps. Both take effect only when the elephant's turn commits, so a turn that fails or is cancelled and runs again leaves no orphan goldfish and no half-made expert. Files a model names in either are read only under configured roots and never from FreeFitty's own directories. The companion paper (*Frozen Experts*, docs/frozen-experts.md) reports the mechanism in detail and the first expert built around a catalog, not documents: a librarian whose 69k-token reading is the card catalog of the pod's offline library, whose 1.27 GB state restores on a second and third node in 1.6 and 3.0 s, and whose goldfish read 69,164 of 70,019 first-turn tokens from cache.

What an expert should read turned out to be the real question, and the answer is less than first assumed. An expert's reading is mostly a lens on what the model already knows (a primer the model writes on the topic, focused by the task), plus the corrections the model cannot know (version changes newer than its training) and the situation (paths, hosts, binaries, factmap article ids): tens of thousands of tokens, not the hundreds of thousands of raw documentation first proposed. Raw documentation stays on the ark for lookup and fact-checking.

## 6. OMF: how much memory a token can carry

The Optical Memory Fabric starts from a question about rendered text: if the reasoner can read a page as an image, and a page costs a fixed number of vision tokens regardless of how much text is printed on it, how much text can be packed per token before the model stops reading it correctly? An A4 page at 96 dpi costs exactly 875 vision tokens on strixite, from the patch arithmetic. The ratio of text tokens carried to vision tokens paid is rho. Measurement gives a single fidelity curve with a knee, and one safety finding that constrains every design: for text rendered too small, the reader never says a page is unreadable. It invents plausible text instead. The protocol must never give it the chance.

From that curve follows a fractal structure: store every memory as a ladder of resolutions, keep only the coarsest rung resident, and let the reasoner zoom into detail by handle the way a deep-zoom renderer resolves a Mandelbrot set. The mechanisms are those of a real-time fractal renderer: level-of-detail subdivision, reprojection of the last frame, frame pacing under a token budget per turn, and an autopilot that dives while the next level is still rendering. The design reduces to one object: memory is a database of nodes viewed through a render function. Visual tokens and page images are a per-request byproduct of a view, never the stored thing, and the writer that fills the database is a curator with a strict archivist persona, not a bulk ingest.

More than forty experiments ran on the real pod corpus (OMF-REAL: the pod's own reports and cards, with blind dev and test splits) against an engine-matched BM25 retrieval baseline. The results sort into three findings.

- **Structure beats flat pages, and does not yet beat retrieval on cost.** The telescopic reader (resident one-line gists, zoom to verbatim leaves) reached 67.3% on dev against 65.3% for retrieval and 22.4% for the gist tree alone, with zero confabulation. The combined reader scored 62.7% against 60.8% on the held-out test, a margin that is not statistically robust, while spending 1.7x retrieval's tokens. Concept layers carry comprehension, verbatim leaves carry precision, and BM25's token efficiency is brutal at this corpus size.
- **A resident haystack fails; state on demand holds.** Holding eleven cached segments resident as a 70k-token prefix answered 0 to 5% of questions where retrieval answered 20 to 45%. Loading the one relevant segment on a miss, chosen by a code router with retrieval as fallback, answered 52.4% against retrieval's 44.9% on a small subset (parity within noise) at 0.28 to 0.46x of its wall-clock. The model spends its budget on the haystack, not the needle.
- **Cached KV state is a usable memory store.** The KV-state study (OMF-R3) exported segments from strixite's cache and measured them: restore is bit-exact and 2.5 to 17.7x faster than re-prefill, the state survives 6.1x quantization with every planted fact read back exactly (the recurrent summary tolerates int4 where per-token rows do not), and two independently cached segments re-based with the engine's own rotary embedding and imported consecutively return both segments' facts in either order. The honest ranking is by job: text in context for bytes, KV segments for latency.

A fourth result came after the first version. **A tree of state diffs is exact and fast (KVTREE).** A 1.7k-token table of contents, three ~33k-token volumes saved as diffs on it, and ~7.5k-token documentation layers saved as diffs on each volume: importing the chain root to volume to docs reproduces the prefill of the same 42k tokens exactly (30 of 30 probes identical to six decimals, recurrent state included) in 1.2 to 1.3 s against 38 to 39 s, 30 to 34x faster. Loading the right volume answers 28 of 30 questions, as well as prefilling everything; loading the wrong one answers 3 of 30, and the table of contents alone 7. Composition works; choosing what to load is the whole game. The bytes follow strixite's formula (the whole 125k-token tree is 4.24 GB uncompressed).

The third finding is where OMF meets FreeFitty. An expert is a cached prefix; a frozen expert is a thread's distillation of itself; the third finding says such prefixes can be stored, compressed and composed. The optical question has not gone away, it has found a consumer: the rendered page is a candidate for what an expert reads, and it pays off only if it carries more exact facts per cached token than the typed text would.

## 7. The unifying principle

Cached state is the memory. Everything else is a view onto it or a durable source it can be rebuilt from. Stated as rules, the four layers each enforce one consequence:

1. **Bytes are identity.** The cache key is the token prefix, so two things are the same memory if and only if they render to the same bytes. FreeFitty's byte-identical system prompt, ordered tool manifests and hash-pinned expert versions all exist to make this true.
2. **The view moves at named events only.** Window cuts, paging batches, refocus and expert birth are the only moments the prefix changes, each paid once and logged. A design that changes the head every turn has thrown the cache away.
3. **State has a residence and can travel.** A thread lives on one node and leaves only with its state over the rail. Export, import, splice and compression of KV segments are the operations of a memory system, and section 6 measured each.
4. **Memory is unevenly distributed.** Elephants hold it, goldfish borrow a document, experts are born with it. The expensive context goes only where it is needed.
5. **The durable thing is never the cache.** Transcript, board, documents and the curated memory database are the sources; the cache is a cost, never corruption, and any view can be regenerated from them.

The diagram below places the parts. Sources sit at the bottom, the engine's cache in the middle, and the agents that read through views at the top.

```
  Agents (EGM)        [Elephant thread]   [Goldfish thread]   [Expert-born thread]
                                   |  a turn: one model step per event
                                   v
  FreeFitty            [ the view the model sees: byte-identical prompt · expert prefix ·
                         budget window · paged results · recall ]
                                   |  the view's bytes, identical each turn
                                   v
  strixite workers     [cache: worker 1] <-> [cache: worker 2] <-> [cache: worker 3]
                         100 GbE rail: export and import move a thread's state
                                   ^  any view is rebuilt from the sources
  Durable sources      [Transcript]  [Board]  [Documents]  [Memory database (OMF, curated)]
```

Agents read through FreeFitty's view, whose bytes land in a worker's prompt cache; the rail moves cached state between workers; the durable sources at the bottom can rebuild any view and are never the cache.

## 8. Measurements to date

Each number below is from a pod report with a kept raw file and a verify script unless its source says otherwise, measured on strixite serving Qwen3.8-Flash-Next on a worker. Engine versions differ between rows (v0.3.0 candidate for most OMF work, v0.5.0 for FreeFitty), and latencies on a shared node include fleet contention.

| Layer | Measurement | Value | Source |
| --- | --- | --- | --- |
| strixite | cached segment size | 115.67 MB + 27,456 B per token | OMF-R3 |
| strixite | needle recall, 2k to 128k context | 1.00 | needle-recall report |
| strixite | wake cache hit inside a live process, up to 30 min idle | 98.3 to 98.6% | FF-CACHE-STRIXITE |
| strixite | decode, one node / two nodes TP | 44 / 58 tok/s | published serving notes |
| strixite | prefill at 64k | 1,330 to 1,400 tok/s | published serving notes |
| strixite | context per slot that boots on all five workers | 524,288 (2 slots, no trims) | CTX-512K |
| StrixFold | state per cached token, DeepSeek-V4.1 TP2 | ~1 KB per rank (63.4 MB for 63,488 tokens) | SFF-PARK |
| StrixFold | export / import of that state | 0.4 s / 0.3 s, reply byte-identical after restart | SFF-PARK |
| StrixFold | 65k reading: cold prefill vs learned | 62 s (1,055 tok/s) vs 2.2 s to first token | SFF-PARK bench, 2026-11-11 |
| StrixFold | decode, one stream / four streams aggregate | 47 prose, 71 code / 67 tok/s | SFF-PARK bench, SPEED-SLOTS |
| FreeFitty | first-turn cache reuse, fresh thread | 719 of 740 prompt tokens | README |
| FreeFitty | live cache move between nodes | 3.7 s | operational observation |
| FreeFitty | paged-out results as share of tokens nodes re-read | ~50% at tau 4 | replay, 28 threads |
| FreeFitty | frozen librarian (69k tokens): state, restore on 2nd / 3rd node | 1.27 GB, 1.6 s / 3.0 s | experts report 2026-11-08 |
| FreeFitty | librarian goldfish first turn from cache | 69,164 of 70,019 tokens | experts report 2026-11-08 |
| FreeFitty | first turn after a rewind with note, cached share (revert / refocus / window) | 19.9% / 1.7% / 19.4% pilot; 1.9% / 1.4% / 17.0% quiet control | FF-REVERT |
| EGM | elephant slept share of a 5-round design loop | 74.4% of 1,314 s | FREEFINITY-EGM |
| EGM | prompt reuse across 11 calls | 70%, rising to 86% | FREEFINITY-EGM |
| EGM | cold-reader catches on review-failed cards | 2 of 12, 0 false positives, 94.5 GPU-s | goldfish-intake |
| OMF | A4 page at 96 dpi | 875 vision tokens | OMF-6 |
| OMF | telescopic reader vs retrieval, dev | 67.3% vs 65.3% | OMF-TELESCOPE |
| OMF | combined reader vs retrieval, test | 62.7% vs 60.8%, 1.7x tokens | OMF-COMBO |
| OMF | resident 11-segment prefix vs retrieval | 0 to 5% vs 20 to 45% | OMF-RESIDENT-LIVE |
| OMF | load-on-miss vs retrieval, cap 80, n=21 (parity) | 52.4% vs 44.9% at 0.28 to 0.46x wall | OMF-RESIDENT-LIVE |
| OMF | KV restore vs re-prefill | bit-exact, 2.5 to 17.7x faster | OMF-R3 |
| OMF | KV compression with facts exact | 6.14x | OMF-R3 |
| OMF | two spliced segments, fact loss | within 0.75 nats of each alone | OMF-R3 |
| OMF | 3-level state-diff chain vs prefill of the same 42k tokens | exact (30/30), 1.2 s vs 38 s | KVTREE |
| OMF | right volume vs wrong volume loaded | 28/30 vs 3/30 exact | KVTREE |
| steering | reviewer expert's first action matching plain, brevity vector scale 2 / 4 / 6 (whole thread) | 16 / 5 / 0 of 23 | STEER-GOLDFISH |
| steering | same, vector on the reading only | 16 / 12 / 10 of 23; still acts on 25 to 29 of 30 | STEER-PREFIX |

The table reads as one argument. The engine's state is exact, compact beyond a few hundred tokens, and fast to restore, on two engines and two very different models. The actor system keeps that state warm with little waste. The division of labour keeps most agents from needing it. And the memory research has shown that packing state in is easy while packing the right state in, on demand, is what pays.

## 9. Open questions and the four experiments

Four experiments join the layers. As of 2026-11-11 the third is built and in daily use, the fourth has a pilot, and the first two are not yet measured. Each is restated as first planned, then its status.

1. **An expert that reads rendered pages**. Experts gain an image read kind built as image parts. Threads born from an image expert and from a text expert of the same facts are compared at equal prompt tokens on a dedicated worker: exact-fact recall, first-turn cached tokens and time to first token over ten or more threads each. The open question is whether an image prefix is cached byte-identically across threads at all, and whether a page carries more exact facts per cached token than its text. *Status: in progress (FF-OPTICAL-EXPERT), no result yet.*
2. **A thread composed from several frozen experts**. A thread starts from two experts and reuses the cache of both. The report measures recall against one concatenated expert and time to first token against re-prefill, and states whether the splice runs through strixite's export and import of segments, as the splice measurement did offline, or through prefix ordering alone. *Status: not started (FF-EXPERT-SPLICE). KVTREE has since shown the engine side: a chain of state diffs composes exactly. What remains is two independently frozen experts, not a parent and its diff.*
3. **Goldfish born from experts, elephants that make them.** The goldfish tool gains an expert parameter, so a specialized goldfish is one cached prefix shared by every spawn, and an elephant gains a tool that commissions a named expert from a persona and a reading list; a builder thread does the reading and the freeze off the elephant's context, and the elephant wakes to the expert's name and hash, spawnable in the same round. The report compares expert goldfish against inline-document goldfish on cached tokens, time to first token and verdict agreement, and commissioned experts against hand-built ones on exact-fact recall, with the elephant's context growth per commission and every elephant-made expert listed with its provenance. *Status: built (section 5) and in daily use. The comparison against inline-document goldfish has not been run as a study; the first-turn cache numbers above come from live use.*
4. **Revert with a note, as an alternative to compaction.** A thread marks checkpoints and can later rewind its view to one, carrying forward a single bridge message written by its reverted self; the store is untouched and recall still reaches everything after the cut. A revert returns to a prefix the node already holds, so the next turn is a cache hit; a refocus writes new head bytes and pays a prefill. The report replays real threads past 150k tokens with revert, refocus and window-only arms, scoring cached tokens after the event, recall of pre-cut facts, task completion, and how much of a note of 256 to 4k tokens the future self carries forward correctly. *Status: piloted (FF-REVERT), not built. On eleven real threads past 150k tokens the first turn after a rewind reused 19.9% of its prompt against 1.7% for a refocus, but window-only reused 19.4%, and on a quiet node the rewind fell to 1.9% while window-only kept 17%. The rewind's advantage belongs to whether the node still holds the old prefix, not to the rewind itself. Note size does not matter to the cache (the note sits at the tail of the view), and every arm carried forward only one or two of six facts from the dropped region. The next version should park the state at each checkpoint, which FreeFitty can now do, so the rewind no longer depends on residency.*

Behind them sit the questions the measurements have not answered:

- **When do a model's own notes beat exact history?** Refocusing is off because nothing has measured it against the window plus recall on real threads, and the fourth experiment adds revert-with-note as a third arm. The frozen expert depends on the answer.
- **Who curates?** The curated memory database needs an archivist that rejects entries without provenance and re-verifies old nodes. Its cost per accepted fact and its effect on recall against an uncurated ingest are unmeasured.
- **Does state on demand scale past one segment?** Load-on-miss matched retrieval with one segment loaded, at a fraction of the wall-clock. The splice result suggests several can be loaded; the resident result warns that loading too many destroys the answer.
- **Is the fertile framework found?** The standard is not one winning arm but a framework where many variants clear the bar and gains compose, followed by a craft phase. The telescopic reader and the KV store each cleared a bar once. Neither has yet shown a slope.
- **What does the cache cost the operator?** Disk-tier capacity, recomputes at paging advances, and state files of 100 MB to 7 GB per thread are real operating costs with no budget model yet. StrixFold's ~1 KB a token shows the cost is set by the model's attention design as much as by the engine.
- **Can a state contract be engine-neutral?** Two engines now serve the same probe/export/import routes, but both are pod forks. Mapping the routes onto a mainstream engine's KV connector (vLLM, LMCache) would show whether the contract is an architecture or a convention.

## 10. What four days of running it taught

The first version was written from measurements; the following come from operating the system and from experiments that failed in instructive ways.

1. **Name the decision before measuring.** A week of cache experiments (FF-CACHE) measured whether sleeping threads keep their cached state, through two hours of wake gaps on the DeepSeek pair. Every row was zero, and the code said why before the first run: that engine keeps no state for prompts under 4,096 tokens and the replay slices were about 2,000, and its cache clock was a counter, not wall time, so the gaps tested a variable that did not exist. Worse, the decision the experiment informed (persist state across sleeps) had already shipped as parking. About ten hours of the only DeepSeek pair went into a guaranteed result. The rule since: name the decision, check the system as built, read the mechanism, predict every row, include a positive control, and spend the cheapest instrument first.
2. **Steering an expert to be brief made it stop checking.** A brevity vector on the librarian cut its cost eightfold and it began inventing line numbers, fields and a standard that does not exist. On the review expert the same vector kept 16 of 23 first actions at scale 2 and 5 at scale 4; from scale 4 on, most briefs got a text answer instead of a read of the card. Applying the vector to the expert's reading only kept the goldfish acting (25 to 29 of 30 briefs opened with a tool call) but did not keep it the same reviewer. A prompt-only turn budget in the librarian's lens halved its turns, cut its tokens to a third, and every spot-checked claim was right. Change the lens before the activations, and judge a cheaper expert by spot-checked claims, not by turn counts.
3. **Make the bad output unrepresentable.** The steered librarian's first failure was a malformed tool call. Constraining tool-call syntax in the engine removed that class of failure on the whole fleet (0 malformed calls at any steering scale), and left the real question, whether the agent still does its job, easier to see.
4. **Residency decides.** The rewind experiment's advantage disappeared on a node whose cache had turned over, and the cache-reuse figures throughout this paper assume the node still holds the prefix. Parking (an explicit export at sleep, an import at wake) turns residency from luck into a decision, and every cache-dependent design should say which of the two it relies on.
5. **The commit boundary has to cover side effects too.** An outside review found the one place the runtime's invariant did not hold: a goldfish thread and a commissioned expert were created inside the turn, before its commit, so a cancelled turn that ran again left an orphan goldfish or found its own expert "exists already". A cancelled turn also leaked its node slot, and a model could name any file the daemon could read, its own tokens included. All three were fixed on 2026-11-11 (goldfish and experts take effect with the commit, slots are released by drop guards, model-named files resolve under configured roots), each with a test that fails on the old code.
6. **Defaults that were right for a test become wrong in service.** A 4-stream, 512k-context DeepSeek configuration "crashed on long prompts"; the cause was a 2-second collective deadline set for short first-light runs, which rank 1 overran by one to three seconds per chunk once kept states were on. At 15 seconds the configuration serves. A context length copied into the scheduler's config went stale the same way, and the scheduler now reads it from the engine's own `/health`.

## 11. Related work and sources

The nearest published ideas:

- **Durable execution.** FreeFitty's predecessor was hydro-project/infinity's AWS durable mode. FreeFitty keeps its one idea, the transcript as actor state with the model as a pure step, and drops the emulated cloud.
- **Prompt-cache reuse and session snapshots.** hipfire's resident session-cache snapshot (256k cached prefix appended at 197 tok/s, from its own published figures; the pod has read hipfire, not run it) is the same move strixite's export and import make, on the same hardware class.
- **KV-cache persistence and compression.** LMCache and kvpress address storing and shrinking KV state outside the engine; OMF-R3 reproduces the core claims on a hybrid GDN model and adds the splice result.
- **Cache-aware serving.** llm-d routes on an index of the exact token prefixes each replica holds and tiers cache entries to CPU memory and SSD; LMCache can locate, move, pin and compress KV state. Both are ahead of FreeFitty on routing precision (FreeFitty uses home-node affinity plus a probe) and act on eviction pressure. What neither has is the scheduler's knowledge that a conversation is about to sleep for six hours, which is what lets FreeFitty park on intent.
- **Durable execution for agents.** Temporal, Restate, DBOS and Inngest get crash recovery by checkpointing and re-running the interrupted step. FreeFitty gets the same guarantee without a replay engine because the model is the only step, and unlike them it knows where each thread's KV state lives. It is also far narrower: one process, one database file, no high availability.
- **Vision-centric token compression** (arXiv 2502.00791) and DeepSeek-OCR-2 are the published forms of the optical-page question; OMF-SERVE ran DeepSeek-OCR-2 as a bake-off arm on ROCm.
- **LEANN** (arXiv 2506.08276) stores a pruned proximity graph and recomputes embeddings on demand, the recompute-not-store principle the curated memory database adopts for its placement layer.
- **The Elephant-Goldfish Model** (google/elephant-goldfish-model, Rensin's essay) supplied the personas verbatim and the memory-distribution argument.
- **Gated DeltaNet and linear attention** (arXiv 2102.11174 lineage) explain why the cached state has a large constant and a small slope, and why its recurrent part tolerates aggressive quantization.

Pod sources, all under /data/pod/reports: OMF-R3, OMF-TELESCOPE, OMF-COMBO, OMF-RESIDENT-LIVE, OMF-SERVE and the fractal-memory design note (2026-10); FREEFINITY-EGM, goldfish-intake, FF-CACHE-STRIXITE, the FreeFitty context-paging review, KVTREE, FF-REVERT, CTX-512K, SFF-CACHE-ON, SPEED-SLOTS (strixfold-slots), STEER-GOLDFISH, STEER-PREFIX, librarian-brevity, ff-cache-planning-failure and experts/2026-11-08 (2026-11); the SFF-PARK result at /data/pod/plans/d74bd09137/RESULT.md; the FreeFitty README at github.com/TheBeaninator/freefitty; strixite's serve/prompt_cache sources.
