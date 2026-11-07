# C.R.E.A.M.: Cache-Resident Experts As Memory. A cache-native architecture for autonomous agents on an air-gapped Strix Halo cluster

2026-11-07

## Abstract

We describe a small, air-gapped cluster of AMD Strix Halo nodes that runs autonomous software agents around one organizing idea: the inference engine's cached state is the system's working memory, and everything else is a view onto it. Four layers carry the idea.

- **strixite** is the engine, serving one 262k-context model per node with a token-prefix prompt cache that survives in RAM and on disk and can be exported between nodes over a 100 GbE rail.
- **FreeFitty** is a durable actor system whose actor state is a transcript and whose step function is the model; it never edits the transcript and spends all its design on keeping each thread's cache prefix stable.
- **The Elephant-Goldfish Model (EGM)** is the division of labour: long-lived elephant threads hold context, short cold goldfish threads do work and die.
- **The Optical Memory Fabric (OMF)** is the research track asking how much memory a cached token can carry, through rendered pages, vision tokens and direct reuse of KV segments.

The paper states the thesis, reports the measurements that support it so far (bit-exact KV restore 2.5 to 17.7x faster than prefill, 6x state compression with facts intact, order-independent splicing of two segments, ~95% cache reuse on live work), and names the two experiments that join the layers: an expert that reads rendered pages, and a thread composed from several frozen experts.

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
| Five workers | Strix Halo, NVMe scratch (one worker holds its weights on a USB drive) | one strixite node each, 2 slots |
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

Published serving figures put one node at 44 tokens a second decode and 1,330 to 1,400 tokens a second prefill at 64k, with tensor-parallel over two nodes at 58; a later measurement at a full context pool gave 39 cached decode. Those numbers are why prefill avoided is the currency of the system: at 1,400 tokens a second a 250k view costs three minutes, and a cached one costs a fraction of a second.

## 4. FreeFitty: the durable work host

FreeFitty is a durable actor system in one Rust binary whose actor state is a transcript and whose behaviour function is the model. A thread is its stored conversation. When something arrives for it (a person's message, a tool's result, a wake-up it scheduled) the model runs once over the conversation, the turn is committed in one transaction, and the thread holds nothing until the next event. The model has no loop; the scheduler is the loop and the model is a pure step. Sleeping is free however long it lasts, a retry re-runs a step that committed nothing, and exactly-once turns fall out of the commit boundary. It hosts every pod work card as a thread.

The store is the truth and is never edited. What the model sees is a view: the shared system prompt, an expert's reading if the thread has one, bridges, then the stored history from some index on. The node's cache is derived from that view and is worth exactly as much as the view's stable prefix, so every rule in the system protects that prefix. The system prompt is byte-identical across threads. Tool definitions are sent in one order. A thread stays on the node that served its last turn, and leaves either with its state over the rail or, while small, by re-reading its prompt elsewhere. The view's head moves at a few named events, each one cache miss, never one a turn.

Four mechanisms move the head, each with a default decided from the pod's own replay rather than reflex:

| Mechanism | Default | What it does | Why that default |
| --- | --- | --- | --- |
| Budget window | on | oldest whole units leave the view when it would pass the node's context minus output and margin; `recall` returns any of them by index within the slice | live card threads routinely pass 150k tokens and would otherwise hit the context wall |
| Paging old results | on | a tool result answered 4 turns ago and over 500 B becomes a handle; advances in batches of 32 | replay over 28 threads and 8,448 messages: such results were half of what nodes re-read; batch 32 keeps 45% of the saving and pays one recompute in ~30 turns |
| Refocusing | off | the model writes notes at a focal point that stand in for the earlier conversation | changes what the model believes; no measurement yet says when notes beat exact history plus recall |
| Preloaded experts | off | a named prefix (documents read plus an acknowledgement) a thread is born with; versions pinned by content hash; `freeze` turns a thread's notes into one | the mechanism is cheap; the practice of choosing what an expert reads is the work |

The expert is the piece this paper turns on. Every thread born from an expert sends the same bytes after the system prompt, so a node's cache serves the first turn of each such thread however much the expert has read: a 100k-token reading is one prefill and then seconds per thread. Live card threads show roughly 95% cache reuse, and a fresh thread reads 719 of its 740 first-turn prompt tokens from cache. An expert is thus a memory with an address, a version and a residence, which is exactly the object the memory research in section 6 was trying to construct from the other direction.

FreeFitty also carries images: a tool result may include up to four pictures, which the view shows as image parts to a vision-capable node, keeps as the newest four, and estimates at 1,400 tokens each. Experts cannot yet read images. Closing that gap is the first of the two experiments in section 9.

## 5. EGM: elephants and goldfish

The Elephant-Goldfish Model is a division of labour between agents that remember and agents that do not. It comes from Google's elephant-goldfish-model repository and David Rensin's essay, and the pod adopted it because it maps directly onto what the cache makes cheap and what it makes expensive. An elephant is a long-lived thread that holds the context of a piece of work: a card, a design, a world. A goldfish is a short thread born cold with one document and one question, which answers and dies. Elephants are expensive to build and cheap to keep because their prefix is cached. Goldfish are cheap to spawn because they share a persona prefix and carry nothing else.

On FreeFitty the pattern is a tool server. The elephant calls `goldfish {document, mode, question}`, ends its turn and sleeps. The server opens a new thread holding the document and the question, and when that thread answers it posts the verdict back as the elephant's next tool result. Three goldfish personas ship as modes, taken verbatim from the upstream repository: a hostile design critic, a fresh-reader comprehension audit, and an implementation-readiness gate.

An end-to-end test on FreeFitty's predecessor, against the same engine and cache, ran a design loop to convergence: the elephant drafted a small CLI tool, a design-critic goldfish failed it four times for real defects, and the fifth draft passed. Over the 1,314-second run the elephant slept 74% of the wall; no model call ran in it between issuing a goldfish and receiving its verdict. Cache reuse across the eleven model calls was 70%, climbing from 56% to 86% as the elephant's prefix grew, with fresh goldfish threads steady at 64 to 66% from their shared persona. A long-lived interactive session holding the same loop would have held a slice and a context for 22 minutes; the actor system held nothing. The goldfish server now runs as a FreeFitty tool server.

EGM also serves as a quality gate. The goldfish-intake experiment put a cold reader in front of a card, asking it to restate the card from its text and named files alone, and ran it retrospectively over twelve review-failed and eighteen clean cards. It caught two of the twelve misreads at a mean of 94.5 GPU-seconds with no false positives, and showed that nine of the twelve review failures were execution failures, not comprehension failures; it is adopted on the re-queue path, not on every pick. The same shape drives a generative game's creative director: one elephant holds a world's canon, goldfish generators produce slices of it.

What EGM contributes to the architecture is the claim that memory should be unevenly distributed. Not every agent needs the history; most need a document and a question. The cache makes the elephant affordable, the actor system makes its sleep free, and the goldfish keeps the expensive context from leaking into work that does not need it.

## 6. OMF: how much memory a token can carry

The Optical Memory Fabric starts from a question about rendered text: if the reasoner can read a page as an image, and a page costs a fixed number of vision tokens regardless of how much text is printed on it, how much text can be packed per token before the model stops reading it correctly? An A4 page at 96 dpi costs exactly 875 vision tokens on strixite, from the patch arithmetic. The ratio of text tokens carried to vision tokens paid is rho. Measurement gives a single fidelity curve with a knee, and one safety finding that constrains every design: for text rendered too small, the reader never says a page is unreadable. It invents plausible text instead. The protocol must never give it the chance.

From that curve follows a fractal structure: store every memory as a ladder of resolutions, keep only the coarsest rung resident, and let the reasoner zoom into detail by handle the way a deep-zoom renderer resolves a Mandelbrot set. The mechanisms are those of a real-time fractal renderer: level-of-detail subdivision, reprojection of the last frame, frame pacing under a token budget per turn, and an autopilot that dives while the next level is still rendering. The design reduces to one object: memory is a database of nodes viewed through a render function. Visual tokens and page images are a per-request byproduct of a view, never the stored thing, and the writer that fills the database is a curator with a strict archivist persona, not a bulk ingest.

More than forty experiments ran on the real pod corpus (OMF-REAL: the pod's own reports and cards, with blind dev and test splits) against an engine-matched BM25 retrieval baseline. The results sort into three findings.

- **Structure beats flat pages, and does not yet beat retrieval on cost.** The telescopic reader (resident one-line gists, zoom to verbatim leaves) reached 67.3% on dev against 65.3% for retrieval and 22.4% for the gist tree alone, with zero confabulation. The combined reader scored 62.7% against 60.8% on the held-out test, a margin that is not statistically robust, while spending 1.7x retrieval's tokens. Concept layers carry comprehension, verbatim leaves carry precision, and BM25's token efficiency is brutal at this corpus size.
- **A resident haystack fails; state on demand holds.** Holding eleven cached segments resident as a 70k-token prefix answered 0 to 5% of questions where retrieval answered 20 to 45%. Loading the one relevant segment on a miss, chosen by a code router with retrieval as fallback, answered 52.4% against retrieval's 44.9% on a small subset (parity within noise) at 0.28 to 0.46x of its wall-clock. The model spends its budget on the haystack, not the needle.
- **Cached KV state is a usable memory store.** The KV-state study (OMF-R3) exported segments from strixite's cache and measured them: restore is bit-exact and 2.5 to 17.7x faster than re-prefill, the state survives 6.1x quantization with every planted fact read back exactly (the recurrent summary tolerates int4 where per-token rows do not), and two independently cached segments re-based with the engine's own rotary embedding and imported consecutively return both segments' facts in either order. The honest ranking is by job: text in context for bytes, KV segments for latency.

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
| FreeFitty | first-turn cache reuse, fresh thread | 719 of 740 prompt tokens | README |
| FreeFitty | live cache move between nodes | 3.7 s | operational observation |
| FreeFitty | paged-out results as share of tokens nodes re-read | ~50% at tau 4 | replay, 28 threads |
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

The table reads as one argument. The engine's state is exact, compact beyond a few hundred tokens, and fast to restore. The actor system keeps that state warm with little waste. The division of labour keeps most agents from needing it. And the memory research has shown that packing state in is easy while packing the right state in, on demand, is what pays.

## 9. Open questions and the next two experiments

Two experiments are queued to join the layers, the second depending on the first.

1. **An expert that reads rendered pages**. Experts gain an image read kind built as image parts. Threads born from an image expert and from a text expert of the same facts are compared at equal prompt tokens on a dedicated worker: exact-fact recall, first-turn cached tokens and time to first token over ten or more threads each. The open question is whether an image prefix is cached byte-identically across threads at all, and whether a page carries more exact facts per cached token than its text.
2. **A thread composed from several frozen experts**. A thread starts from two experts and reuses the cache of both. The report measures recall against one concatenated expert and time to first token against re-prefill, and states whether the splice runs through strixite's export and import of segments, as the splice measurement did offline, or through prefix ordering alone.

Behind them sit the questions the measurements have not answered:

- **When do a model's own notes beat exact history?** Refocusing is off because nothing has measured it against the window plus recall on real threads. The frozen expert depends on the answer.
- **Who curates?** The curated memory database needs an archivist that rejects entries without provenance and re-verifies old nodes. Its cost per accepted fact and its effect on recall against an uncurated ingest are unmeasured.
- **Does state on demand scale past one segment?** Load-on-miss matched retrieval with one segment loaded, at a fraction of the wall-clock. The splice result suggests several can be loaded; the resident result warns that loading too many destroys the answer.
- **Is the fertile framework found?** The standard is not one winning arm but a framework where many variants clear the bar and gains compose, followed by a craft phase. The telescopic reader and the KV store each cleared a bar once. Neither has yet shown a slope.
- **What does the cache cost the operator?** Disk-tier capacity, recomputes at paging advances, and state files of 100 MB to 7 GB per thread are real operating costs with no budget model yet.

## 10. Related work and sources

The nearest published ideas:

- **Durable execution.** FreeFitty's predecessor was hydro-project/infinity's AWS durable mode. FreeFitty keeps its one idea, the transcript as actor state with the model as a pure step, and drops the emulated cloud.
- **Prompt-cache reuse and session snapshots.** hipfire's resident session-cache snapshot (256k cached prefix appended at 197 tok/s, from its own published figures; the pod has read hipfire, not run it) is the same move strixite's export and import make, on the same hardware class.
- **KV-cache persistence and compression.** LMCache and kvpress address storing and shrinking KV state outside the engine; OMF-R3 reproduces the core claims on a hybrid GDN model and adds the splice result.
- **Vision-centric token compression** (arXiv 2502.00791) and DeepSeek-OCR-2 are the published forms of the optical-page question; OMF-SERVE ran DeepSeek-OCR-2 as a bake-off arm on ROCm.
- **LEANN** (arXiv 2506.08276) stores a pruned proximity graph and recomputes embeddings on demand, the recompute-not-store principle the curated memory database adopts for its placement layer.
- **The Elephant-Goldfish Model** (google/elephant-goldfish-model, Rensin's essay) supplied the personas verbatim and the memory-distribution argument.
- **Gated DeltaNet and linear attention** (arXiv 2102.11174 lineage) explain why the cached state has a large constant and a small slope, and why its recurrent part tolerates aggressive quantization.

Pod sources, all under /data/pod/reports: OMF-R3, OMF-TELESCOPE, OMF-COMBO, OMF-RESIDENT-LIVE, OMF-SERVE and the fractal-memory design note (2026-10); FREEFINITY-EGM, goldfish-intake, FF-CACHE-STRIXITE and the FreeFitty context-paging review (2026-11); the FreeFitty README at github.com/TheBeaninator/freefitty; strixite's serve/prompt_cache sources.
