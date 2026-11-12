# Frozen Experts: Cached Model State as Addressable, Transferable Memory for Autonomous Agents, and a Librarian that Knows the Library Without Having Read the Books

Draft. Companion to *C.R.E.A.M.: Cache-Resident Experts As Memory* (docs/cream-cache-resident-experts-as-memory.md), which states the architecture; this paper is about the one object the architecture produced, measured on the live system.

## Abstract

An autonomous agent that works from documents pays to read them on every thread it starts. On a fixed, air-gapped cluster that cost is the dominant one: a 70k-token reading prefills in about a minute on an idle node and in far longer under contention, and the cluster described here starts hundreds of agent threads a week. We describe the *frozen expert*: a named reading whose identity is the exact bytes of its documents and acknowledgement, whose value is the inference engine's cached state at the end of that reading, and whose state is read once on one node, exported to a file, and imported into whichever node takes the first turn of any thread born from it. The expert then has a name, a content-hash version, a state file on disk, and can be imported onto any node after any restart. A thread born from it only has to read its own message. We give the mechanism as built in FreeFitty, the durable agent runtime the cluster runs its work on, and the practice that grew around it: the *lens* method for deciding what an expert reads, goldfish (short cold threads) born from experts, and elephants (long threads) that commission new experts. We then report on the *librarian*, a reference librarian for the cluster's offline library whose entire 1,551-title card catalog is one 69k-token frozen reading. Its 1.27 GB state restores onto a second and a third node in 1.6 and 3.0 seconds; its goldfish's first turns read 69,164 of 70,019 and 69,164 of 69,405 prompt tokens from cache; and it conducts a tool-using reference interview that turns a task into a reading list with local paths, checking the shelf before it answers. We give the measurements that exist so far, say what has not been shown yet, and list the experiments queued to show it.

## 1. Introduction

The system we describe (we call it the pod) is five AMD Strix Halo workers and a controller, air-gapped by construction, running software agents around the clock on open-weight models. The agents take their work from a shared task board; we call each work item a card. Nothing can be fetched from the internet; everything the agents know is resident, cached or on the ark, an offline mirror of models, papers and repositories that is loaded by hand. Compute is fixed. Under those constraints the expensive operation is not generating a token but reading one: the engine prefills at roughly 1,400 tokens a second when idle, and a thread that begins by reading a runbook, a checklist, a design and a situation sheet spends its first minute before it says anything, then does so again on the next thread, and the next.

The engine already has the fix for a single thread: a prompt cache keyed on the exact token prefix, so the second turn of a conversation reuses the state of the first. It has no fix across threads, nodes and restarts, and that is where the pod's cost actually is. Ninety review threads a week each re-read the same identity, objective and checklist; every thread that touches the ark re-discovers where things are.

The idea in this paper is to make the reading itself an object. A *frozen expert* is a set of documents and a one-sentence acknowledgement, rendered to bytes that never change, read once by the model on one node, with the node's cached state at the end of that reading exported to a file. A thread "born from" the expert sends those same bytes after the shared system prompt, and the runtime imports the state file into whichever node takes its first turn, so the node has already read everything the expert read and the thread pays only for its own message. The expert has a name, a version pinned by content hash, a file on disk, and moves between nodes over the 100 GbE rail in far less time than the reading would take.

This is a small mechanism, and we built it as a cache optimisation. We think it matters more than that, for three reasons:

1. **Memory gets an address.** An expert is the first object on the pod that is at once what the model has read, where that state lives, and which version a given thread holds. Section 3.
2. **Memory can be unevenly distributed on purpose.** The Elephant-Goldfish Model's division between agents that remember and agents that do not gains a concrete substrate: a goldfish born from an expert borrows a reading as a cache hit, and an elephant commissions new experts without ever holding their material. Section 4.
3. **A reading can be index-shaped.** The librarian expert holds the catalog of the library, not the books, and converts a task into a list of what to read and where it is. A catalog is a different kind of knowledge from a document, and it happens to be the kind an air-gapped agent most lacks and a cache serves well. Sections 5 and 6.

Section 7 collects the measurements. Section 8 states limitations plainly. Section 9 places the work beside prompt-cache and KV-reuse literature. Section 10 lists the experiments already queued on the board.

## 2. Setting

The substrate is described in the companion paper; we restate only what this paper depends on.

**Hardware.** Each node is a Ryzen AI MAX+ 395 with a Radeon 8060S (ROCm gfx1151) and 128 GB of unified memory. The five workers (named A to F, with no E) and the controller sit on a lossless 100 GbE network (Mellanox SN2700, RoCE v2), which we call the rail, and a separate 2.5 GbE management network.

**Engine.** Each worker runs strixite, a native ROCm server for Qwen3.8-Flash-Next at a 262,144-token context, serving two requests at a time (two slots). The model is a hybrid: gated-delta-net recurrent layers carry most of the sequence in a fixed-size state, and a minority of full-attention layers keep per-token KV rows. The prompt cache is keyed on the pure token-id prefix plus a weights fingerprint; there is no conversation id, so any request whose rendered prefix matches reuses the state across threads and processes. Two engine features make this paper possible: `POST /cache/export` and `POST /cache/import` move a cached state between nodes as a file, and the engine saves a checkpoint entry at the start of the last user message of a request, which gives the runtime a precise point to export.

**Runtime.** FreeFitty is a durable actor system in one Rust binary. A thread's state is its stored transcript; the model is a pure step run once per event; a turn commits in one transaction. The model never sees the store directly but a *view*: the byte-identical system prompt, then an expert's reading if the thread has one, then the stored history from some index on. Every rule in the runtime protects the stable prefix of that view, because the engine's cache is worth exactly as much as the view's stable prefix. Tool definitions render in one fixed order. A thread stays on the node that served its last turn unless it leaves with its state.

All of the pod's work (cards, code reviews, design rounds, content generation for a game project) runs as FreeFitty threads.

## 3. The frozen expert

### 3.1 Definition

An expert is defined in a TOML file in the runtime's `experts_dir`:

```toml
description = "Has read the pod runbook"
ack = "I have read the runbook and will work within it."
tools = ["read_file", "run"]          # optional: what a goldfish born from it may call
[[reads]]
path = "/data/pod/objective/RUNBOOK.md"
[[reads]]
text = "Deployments go through the staging node first."
```

From the definition the runtime builds exactly two messages: a user message wrapping each document in a `<document path=...>` element inside an `<expert name=...>` element, ending with "You have read these documents. Acknowledge in one sentence."; and the assistant's acknowledgement. Those two messages, serialised, are the expert's *bytes*. Their hash, together with the tool list and any steering (3.5), is the expert's *version*. The system prompt is deliberately not part of the stored expert: an edit to it changes every thread alike and the system block stays shared, which is the same reason the system prompt is byte-identical across threads.

Versions are stored by content hash and never pruned. A thread keeps the bytes it was born with when a file changes; a new thread gets the new version. The stored messages are the expert's provenance and what it is rebuilt from, so the state file is derived data: losing it costs a prefill, never information.

### 3.2 Freezing

Freezing an expert version for a *context* (the hash of the system prompt and the rendered tool definitions, which precede the expert in every request) is one read and one export:

1. Send the node a request consisting of the system prompt, the expert's two messages, and a last user message ("Say ready."). The node prefills the reading and generates a few tokens.
2. Export the node's cached state for the same prefix with a different last message ("."). Because the engine checkpoints at the start of the last user message, this selects the entry ending exactly at the end of the acknowledgement: the prefix every thread born from the expert will share, with no trace of the probe message.
3. Verify that the exported state covers the expert's token count less a small margin; refuse and delete the file otherwise.
4. Write `<data_dir>/experts/<name>@<hash>-<context>.kv`.

The path encodes name, version and context, so one expert may have several live frozen states: a goldfish with a scoped tool list and an elephant with every tool are two contexts of the same version, and both are kept. When the system prompt or tool definitions change, the context hash changes and the expert is frozen again on first use; a thread born meanwhile reads the expert itself and triggers the freeze. Two freezes of one version never run at once.

### 3.3 Birth and residence

A thread is born from an expert by `POST /say {text, expert}` or, as a goldfish, by the `goldfish` tool (Section 4). Its view is the system prompt, the expert's two messages, then its own history. On its first turn the router picks a node and, if the expert's state file for this context exists, imports it into that node before sending the turn. The node then reports the whole expert as cached and prefills only the thread's own message. Restores are logged; the librarian's 1.3 GB state restored into worker B in 1.6 s and into worker C in 3.0 s (Section 7).

Nothing about the node is special. The thread can be born on any node that is up, after any restart of the runtime or the node, and the import happens once per node per expert context: the engine then holds the entry in its own cache for later threads. `experts warm NAME` pushes a frozen state into every up node ahead of demand, and `warm_on_start` lists the experts to warm when a node comes up; the log line "frozen state put into 5 node(s)" is that operation.

### 3.4 Experts from threads, experts from elephants

Two further sources of experts exist beside the definition file.

An expert may be frozen *from a thread*. The runtime has a compaction step we call refocusing, in which the model writes notes that stand in for its earlier history; a thread's latest notes can become what an expert has read, with the acknowledgement "I hold this understanding and will continue from it." This is the frozen form of what long-running threads otherwise lose at compaction.

An expert may be *commissioned by an elephant* through the `commission_expert {name, description, reads, ack?, tools?}` tool. The runtime registers it at once, writes its definition to `experts_dir` with a header naming the commissioning thread and the time, and freezes it; the elephant sleeps until the freeze ends and wakes to one tool result naming the expert and its version. The elephant never holds the material; its context carries only the catalog (Section 4.1) and the goal of its own work.

### 3.5 Steering as identity, and the composition barrier

An expert may also be a persona: `[[steer]] vector = "villain24", scale = 6.0` names activation-steering vectors the nodes hold. Every request of a thread born from it carries them, its freeze reads it steered, and the steering is part of its version, since the engine caches steered state under its own key. The expert is then the bundle of everything two threads must share to share cache: prompt bytes, tool list, steering.

One consequence is a real constraint rather than an implementation detail: steering is a composition barrier. Two experts under different steering have different cache keys and their frozen states cannot be spliced into one thread. Refocusing runs unsteered for a related reason: its notes are stored text the thread will believe, so a later freeze of those notes bakes in no persona. Commissioned experts may not steer until a steered goldfish has been measured against a plain one for verdict agreement and malformed tool calls (Section 10, experiment 4).

## 4. Goldfish born from experts, elephants that commission them

The Elephant-Goldfish Model (Google's `elephant-goldfish-model`, after Rensin's essay) divides agents into long-lived elephants that hold context and short cold goldfish that receive one document and one question, answer, and die. The pod adopted it for design critique and task intake. Before experts, a goldfish's persona and document were pasted into each spawn; the only shared cache was the persona prefix. With experts, three built-in tools, available to every thread, make the division concrete.

### 4.1 The catalog

`experts` returns the catalog: each expert's name, its one-line description, and the tools its goldfish may use. It is a tool, not part of the tool definitions, so a new expert never changes the bytes every thread sends. This is what an elephant chooses from. The live catalog at the time of these runs read, in part:

```
- ff-reviewer: Independent reviewer of a card: the shared identity, the review checklist and the
  verdict wording every review round re-reads. (no tools as a goldfish)
- isk-director: Director of a generated game's content rounds: the canon, world manifest and
  director notes every round opens by reading. (no tools as a goldfish)
- librarian: The ark's reference librarian: knows every title in the offline library and
  roughly what each is, turns a task description into a short reading list (title, local path,
  which part, why), checks the shelf before answering, and names what the ark lacks as
  candidate rows (tools: read_file, run)
- strixfold-engineer: Engineer for a second inference engine ... (tools: read_file, write_file, edit_file, run)
- webxr-designer: WebXR art director and A-Frame 1.7 engineer for a virtual office project ...
  (tools: read_file, write_file, edit_file, run, view_image)
```

### 4.2 Asking

`goldfish {expert, question, documents?, tools?}` makes a thread named `<elephant>.gf-<expert>-<call>` born from the expert, puts any documents (local paths) and the question in its first message, and runs it. Several goldfish asked in one turn run at once. When the goldfish ends a turn with nothing pending, its text returns as the elephant's tool result and the goldfish is closed; a failure returns as the result too. A goldfish sees and may call only the plain built-ins (sleeping, recalling its own history, refocusing, the catalog) and the tools it was granted: the call's `tools`, else its expert's list, else none. `goldfish` and `commission_expert` are never a goldfish's, whatever it was granted: it says in its answer which expert should look next, and the elephant decides. That keeps the tree of spawns one level deep and the elephant the only thing that remembers.

### 4.3 Commissioning

`commission_expert` is described in 3.4. The shape it enables is the one the system's operator asked for when the mechanism was tied to EGM: "the elephant should be able to use specialized goldfish, maybe even make new ones", refined to "the elephant *commissions* a specialized goldfish, it does not build one in its own context" and "the elephant holds the north star, not the goldfish's material."

## 5. What an expert reads: the lens method

The mechanism is cheap. The practice of choosing what an expert reads is the work, and the first instinct (preload the documentation) was wrong. The correction, in the operator's words, was that an expert's context "would itself just sort of be lensing information from the model itself about the topic, as well as providing some specific context like dev environment, project locations and descriptions, resource locations, [knowledge-base] article IDs."

The method that followed has three readings, in this order:

1. **The lens.** A primer the pod's own model writes about the topic from what it already knows ("everything a senior A-Frame engineer keeps in mind for a Quest 3 interior"), then corrected by hand against the primary source. The raw and corrected primers are both kept. The corrected primer begins by saying so, and by saying that the deltas reading wins any disagreement.
2. **The deltas.** What is newer than the model's training: version changes, renamed APIs, the fork's own departures, read from the code.
3. **The situation.** Where things are on this system: binaries, project layout, paths into the ark, entries in the pod's internal knowledge base, what the pod has already measured, and the tools the expert has.

Two experts were built this way. `webxr-designer` (4,646 tokens: lens on beautiful fast WebXR interiors, A-Frame 1.7.0 deltas, the situation of a virtual-office project) was asked by a long-lived thread to make the ground floor of that office beautiful; the goldfish worked for hours with file, run and image tools, built a texture pipeline, rendered and committed, and was blocked only by a node lease it did not own. `strixfold-engineer` (6,681 tokens: lens on a two-box mixture-of-experts engine as a node for durable threads, the engine fork's deltas, the situation) was asked to make that second engine a FreeFitty node and worked through dozens of turns of code reading and measurement. Neither is a controlled measurement; both are the experts that made the next one obvious.

Readings are small by design: 2k to 17k tokens for every expert but the librarian. Experiment 3 in Section 10 measures the layers against each other with headless-scored task sets.

## 6. The librarian

### 6.1 The brief

The librarian began as a side request from the operator, stated in one paragraph:

> What do librarians do? They answer questions about what's in the library. As in, they know the titles of the books and a little bit about them, but chances are they haven't read the entire book themselves. They just know about it, or where to look for it. They convert task descriptions into a list of reference material sources, for example.

The ark already looks like a library with a card index. Its manifest names every repository, model, dataset, container, paper and reference shelf the pod has asked for; an index file says what has actually landed and where. No agent on the pod had ever held the whole index. Every thread that needed something from the ark searched for it from nothing, and the agents' most common failure on research tasks was to miss material the operator had staged for them.

### 6.2 Construction

The librarian is a lens-method expert whose "deltas" reading is the catalog itself.

- **The lens** (2.8 KB) describes what a reference librarian does, in five numbered duties: the *reference interview* (turn the task into the questions it contains before naming any source); sources per question, best first, primary over commentary, the pod's own reports and knowledge-base entries over outside material when the pod has measured the thing; say what is not here, as a candidate manifest row; check the shelf before answering (a lookup command says whether a title landed and where); stay brief. It fixes the shape of an answer (Task / Questions with sources and `[landed | not found]` / Pod's own / Not in the ark) and three prohibitions: do not invent titles, paths, versions or paper numbers; do not read whole repositories or papers; do not change the ark.
- **The catalog** (198 KB, 1,663 lines, 1,551 items) is rendered from the manifest by a 60-line script, one line per item: title, local path, a note trimmed to 150 characters, grouped under 48 headings (272 models, 79 inference repos, 59 agent repos, 205 reference URLs, 61 MPI papers, and so on). It is re-rendered after large ark drops and the expert reloaded, which produces a new version; the two versions frozen so far differ by about 1,100 tokens.
- **The situation** (2 KB) says where the library's stable view and its inbox of new arrivals are, how the catalog's paths map to real ones, which tools exist (the shelf lookup, knowledge-base search, `ls`, `grep`, `head`), that hand-dropped items appear on the shelves without a catalog line, and how to write a candidate row.

The expert's tools are `read_file` and `run`, nothing that writes. Its acknowledgement is: "I am the ark's librarian: I know the catalog, not every page. Give me a task and I name the questions inside it, the best sources in the library for each with their paths and which part to open, what the pod has already measured, and what the library is missing."

### 6.3 Why the librarian is the interesting case

Three things make it more than just a large expert.

**It is index-shaped knowledge.** A document expert knows a thing. The librarian knows *about* 1,551 things and where they are. You cannot retrieve that into context on demand, since the point is to see all of it at once and choose; and you cannot give it to every thread, since 69k tokens is a quarter of the context. A frozen reading is the one form in which it is both whole and cheap.

**It separates knowing-about from having-read.** The lens forbids reading the books. The goldfish opens a README, a table of contents, a directory listing or an abstract: enough to point at the right part. Its answers are lists of pointers a cold reader can act on, which is the handoff the Elephant-Goldfish Model needs and previously had no source for.

**It fixes a long-standing failure.** The pod's research tasks are deliberately not primed with sources, to test whether the agents find them on their own. Often they did not. With a librarian in the catalog, any elephant can ask for the reading list in one tool call and spend its own context on the work.

### 6.4 First runs

The librarian's first freeze ran on worker F: 69,177 prompt tokens, read in 406 s under contention, 1.27 GB of state. Two parent threads then asked it the same question, a reading list for making a second inference engine (a two-box DeepSeek-V4.1-Flash server) a FreeFitty node. Both goldfish waited for a free slot (the cluster was running code reviews and two other experts' goldfish), and both were then served on nodes other than the one that read the catalog:

```
saved state librarian@e9af…-f3c4….kv (1.3 GB) restored into worker B in 1.6s
lib-1.gf-librarian: turn on B (70019 prompt tokens, 69164 from its cache) in 17 s
saved state librarian@e9af…-f3c4….kv (1.3 GB) restored into worker C in 3.0s
lib-2.gf-librarian: turn on C (69405 prompt tokens, 69164 from its cache) in 29 s
```

Each goldfish then ran a reference interview with its tools: listing the engine's source tree, grepping its cache routes, checking which papers had landed, listing the documentation shelves, reading the pod's own earlier study of the same question, and reading the runtime's config. Over the first 25 minutes each took 20 to 30 turns, its prompt growing from 70k to 100k and 127k tokens, with 97 to 99 per cent of each turn's prompt served from cache and model time per turn of 7 to 105 s. This draft was prepared before either goldfish returned its reading list; Section 7 records the numbers that exist and Section 8 says what that means for the claims here.

One behaviour is already visible and should be recorded. The lens tells the librarian to stay brief and to open only a README, a listing or an abstract. Both goldfish instead ran long interviews: dozens of shell calls, the pod's earlier study read in full, the engine source grepped, the runtime's config and service file read. Diligent, but also 30 turns and 60k tokens of growth for an expert whose premise is that it does not read the books. Whether the brief needs a turn budget, whether the question (five sub-systems in one task) was too large for one interview, or whether the thoroughness pays for itself in the list, is the first thing to judge once answers are in hand.

## 7. Measurements

All numbers are from the runtime's service log or its weekly experts report, on strixite v0.5.0 serving Qwen3.8-Flash-Next on the workers, over one week of production use. "Read in" is wall time of the freeze's read request including any wait for a slot, so it is an upper bound on prefill. Latencies are on a shared cluster under review load.

### 7.1 Frozen experts

| Expert | Reading (prompt tokens) | State file | Read in | Note |
| --- | --- | --- | --- | --- |
| ff-reviewer | 4,665 to 5,293 | 0.16 to 0.22 GB | 12 s (3,083 cached) to 419 s | four contexts frozen as tools changed |
| webxr-designer | 6,519 | 0.23 GB | 46 s | lens method |
| strixfold-engineer | 6,681 to 6,963 | 0.23 GB | 4,153 s and 6,504 s (slot waits) | lens method |
| isk-director | 12,986 | 0.33 GB | 2,213 s (12,981 cached) | game project canon |
| strixfold-lab | 16,822 | 0.41 GB | 1,651 s (2,455 cached) | engine reports |
| librarian v1 | 69,177 | 1.27 GB | 406 s | the catalog |
| librarian v2 | 68,106 | 1.11 GB | 2,424 s (slot wait) | re-rendered catalog |

State size grows with the reading but below the per-token slope an earlier study on this cluster measured for raw cached segments on an earlier engine build (115.67 MB plus 27,456 bytes per token would predict about 2.0 GB for the librarian). We report the files as written and draw no conclusion from the difference beyond that the export format is not that study's.

### 7.2 Birth from a frozen state

| Event | Value |
| --- | --- |
| 1.3 GB librarian state imported into worker B | 1.6 s |
| the same into worker C | 3.0 s |
| first goldfish turn, B: cached / prompt tokens | 69,164 / 70,019 (98.8%) |
| first goldfish turn, C: cached / prompt tokens | 69,164 / 69,405 (99.7%) |
| later turns, both goldfish, cached share | 97 to 99% |
| code-review threads born from ff-reviewer in one week | 90 |
| ff-reviewer first-turn cached / prompt tokens (typical) | 6,250 / 8,100 to 11,300 |
| frozen states pushed to all up nodes at start (ff-reviewer, isk-director, strixfold-lab) | 5, 5, 4 nodes |

The ff-reviewer rows are the mechanism at scale rather than in a demonstration: every independent code review the pod ran in that week was born from that expert, matched automatically by the card's labels and title, and each first turn read the identity, objective and checklist from cache. The 6,250 figure is constant across threads because it is the shared prefix (system prompt plus expert); the remainder is the round's own brief.

### 7.3 The cost avoided

At the published idle prefill rate of about 1,400 tokens a second, the librarian's 69k-token reading costs about 50 s per thread when the node is free, and the 406 s and 2,424 s reads above are what it costs when it is not. The two imports cost 1.6 s and 3.0 s. For the reviewer the saving is a few seconds per thread, ninety times a week. For the librarian it decides whether the expert can be asked casually at all.

## 8. What has and has not been shown

Shown, on the live system:

- An expert's reading is frozen once and born into threads on other nodes, with the first turn reading the whole expert from cache (7.2).
- The mechanism carries the pod's most repeated work (90 reviewer threads in a week) without anyone choosing to use it per thread (7.2).
- A 69k-token index-shaped reading behaves as a cached object end to end: freeze, export, import, birth, multi-turn tool use at 97 to 99 per cent cache reuse (6.4, 7.2).

Not yet shown:

- **That the librarian's answers are good, or brief.** No reading list had been returned when this draft was prepared, and both goldfish had run far longer than the lens intends (6.4). The quality claim rests on the design and on the transcripts' visible method (reference interview, shelf checks, reading the pod's own prior study first), not on a scored answer. A revision should carry the first answers verbatim, their turn and token cost, and a judgement of both.
- **Verdict agreement and recall against alternatives.** No arm has compared an expert-born goldfish with an inline-document goldfish on cached tokens, time to first token and agreement (experiment 1 in Section 10), nor a commissioned expert with a hand-built one on exact-fact recall.
- **Which layers of the lens matter** (experiment 3), or at what scale steering breaks tool calls (experiment 4).
- **Composition.** A thread born from two experts does not yet reuse both caches; whether that runs through prefix ordering or the engine's segment splice is open (experiment 2), and steering is known to forbid it.

Threats to the numbers: n is small (two librarian goldfish, one freeze per version); read times include slot waits and are upper bounds; the import times are two observations; the ff-reviewer cache figures come from an automated report of which only the per-thread table has been independently verified. The engine is one hybrid model whose recurrent state makes a cached segment small and its restore cheap; the mechanism is engine-agnostic (it needs export, import and a checkpoint at the last user message) but the economics may not transfer to a pure-attention model at the same context, where state per token is larger. Finally, the catalog goes stale as the ark grows; re-rendering it is a new version and a new freeze, by design, and nothing yet does that on a schedule.

## 9. Related work

**Prompt caching as a serving optimisation.** Prompt Cache (Gim et al., MLSys 2024) precomputes attention states for declared *prompt modules* and reuses them across requests with a schema that keeps positions consistent, reporting time-to-first-token reductions from 1.5x to 10x on GPU. Hosted APIs offer prefix caching of system prompts and documents with the same motivation. The frozen expert is a prompt module that has acquired a name, a version, a file and a residence, and that the runtime routes births to; the schema is the fixed order of system prompt, expert, history.

**KV state as a transferable artifact.** CacheGen (SIGCOMM 2024) compresses KV caches into bitstreams and streams them to the serving node; CacheBlend (EuroSys 2025) fuses several precomputed chunk caches by recomputing a small share of tokens; LMCache and kvpress store and shrink KV state outside the engine; KVzip (2025) makes compressed caches reusable across queries over the same context. An earlier study on this cluster reproduced the core claims on a hybrid recurrent model (bit-exact restore 2.5 to 17.7x faster than prefill, 6x compression with planted facts intact, order-independent splice of two segments). This paper does none of that engineering; it uses the engine's export and import as given and asks what agents can be once a reading is a file.

**KV reuse as agent memory.** MemArt treats reusable KV blocks as the external memory substrate of long-term agents; KVCOMM (2025) reuses cache across homogeneous multi-agent workflows where shared text sits under different prefixes; AgentKVShift (2026) corrects reused caches for agentic memory retrieval; "Can I Buy Your KV Cache?" (2026) proposes publishers hosting precomputed caches that independent agents load, verifying equivalence with prefill under greedy decoding. These are the nearest ideas. The differences are of position and of practice: the frozen expert is the identity of an agent persona rather than a retrieved unit; births, not retrievals, are the operation; experts are commissioned by other agents and frozen from their notes; and the librarian is a worked example of a reading whose value is that it is an index.

**Division of labour.** The Elephant-Goldfish Model (Google, after Rensin) supplied the elephant and goldfish roles and the argument that the agent that knows least reviews best. Its goldfish are given a document; ours are born from an expert and may be given a document too.

**Reference librarianship.** The reference interview, the distinction between knowing a collection and having read it, and the duty to say what the collection lacks are standard library practice. The librarian's lens is a paraphrase of that practice for an agent with a card index and a shell.

## 10. Queued experiments

Each is queued on the pod's task board with a report path and a verification script; none has run.

1. **Goldfish from experts.** Expert goldfish against inline-document goldfish, ten or more spawns per arm on a dedicated worker: first-turn cached tokens, time to first token, verdict agreement; and elephant-commissioned experts against hand-built ones of the same material on exact-fact recall and bytes, with every elephant-made expert listed with provenance.
2. **Splicing experts.** A thread born from two experts reusing both caches; recall against one concatenated expert; time to first token against re-prefill; and whether the splice runs through the engine's segment import or prefix ordering alone.
3. **Lens layers.** Blender and Godot experts in four arms (none, lens, lens plus situation, lens plus deltas plus situation), scored by headless checkers over eight or more tasks each.
4. **Steered goldfish.** Steered against plain reviewer goldfish at scales 0 to 8 on thirty or more real briefs: verdict agreement and malformed tool-call rate.
5. **An image-reading expert** (the companion paper's first experiment): whether an image prefix caches byte-identically across threads and whether a rendered page carries more exact facts per cached token than its text.

Two more the librarian itself suggests and that are not yet queued:

6. **A librarian evaluation protocol.** For a set of past research tasks whose eventual sources are known, ask the librarian cold and score its list against what the successful thread actually read: precision of titles, correctness of paths, and the share of operator-staged material it surfaces that the original thread missed.
7. **Shelf librarians commissioned by the librarian's elephant.** When a reading list is dominated by one shelf (the 79 inference repos, the 61 MPI papers), the elephant commissions a shelf expert whose catalog is that shelf's READMEs and abstracts, one level deeper than titles; the test is whether the two-level library answers more of the protocol's questions at the same cached tokens.

## 11. Conclusion

A reading that is read once, frozen and born into threads on any node is a small mechanism, about 350 lines of runtime code around two engine endpoints. It turns the engine's cached state into memory with a name, a version and a place on disk; it lets long threads commission short ones without carrying their material; and it already carries the pod's most repeated work without anyone opting in per thread. Its first large instance is a librarian that holds the index of an air-gapped library as a single cached reading, restores onto a new node in seconds, and conducts a reference interview with a shell. Whether its reading lists are good is the next thing to measure. That they are cheap enough to ask for, we have measured.

## Addendum. The librarian's first answers, and a brevity comparison

Of three goldfish born from the plain librarian and asked the same question, one answered after 20 turns and 42 tool calls (230k live tokens); the other two passed 130 and 240 turns without answering and were cancelled. The answer that came was good: it corrected a stale knowledge-base entry, found the pod's own unmerged fix for the engine's disabled cache, and named the source-file line where one rank mirrors the other's state. Its cost was the problem, as Section 6.4 anticipated.

Two levers were tried on the same question. The first was activation steering: a "brief pointer answer minus exhaustive researcher" direction extracted at layer 20 from eight matched text pairs, attached to a copy of the expert so that the steering is part of its version (3.5). At a scale strong enough to change behaviour the first tool call came out malformed and the thread died. At a weaker scale the goldfish answered in 9 turns and 8 tool calls, an eightfold saving, but spot checks found invented file line numbers, an invented cache-key composition, a wrong RFC number and two sources marked as present that the library does not hold. The direction bought brevity by skipping the shelf checks the lens requires, which is the failure the lens exists to prevent.

The second lever was a paragraph added to the lens: at most eight tool calls, an answer by the sixth turn, anything unchecked marked as such. The goldfish overshot its own cap (12 turns, 18 tool calls, 85k tokens) but answered, at under half the plain run's turns and about a third of its tokens, and every one of its spot-checked claims was correct, including two source line ranges and a hash function named from the engine's code. That lens is now the librarian's.

Two conclusions for the paper's argument. First, the cheap part of the system is the frozen reading; the expensive part is the interview, and the interview is governed by the lens, not the cache. Second, steering an expert's disposition is a blunt instrument where the behaviour wanted is "check before you say": the cases where it saved tokens were the cases where it stopped checking. Constrained decoding of the tool-call block (a declared tool name and its parameters, once the model opens a call) was then built into the engine and removed the syntax failure: with it, a goldfish told to call a tool produced well-formed calls in 10 of 10 trials even at a steering scale that previously broke every call. It does not restore the checks; the lens does.

## Appendix A. Sources

Code: github.com/TheBeaninator/freefitty, `src/experts.rs`, `src/rap.rs`, README sections "Experts" and "Goldfish"; commits 256fbe4 (experts are their frozen cached state), 8d827da (exports wait for the turn's entry), d03dd06 (goldfish, catalog, commission), da7b51b (keep every context's frozen state), 55e0562 (steering as part of an expert). Pod: `/data/pod/freefitty/experts/*.toml`, `/data/pod/plans/librarian/{lens.md,catalog.md,situation.md,mkcatalog.py}`, `/data/pod/plans/{webxr-designer,strixfold-engineer}/`, `/data/pod/reports/experts/2026-11-08.md`, the FreeFitty service journal on the controller for 2026-11-08, board cards 59d9fd4454, 057ef28373, 6867d0ffd1, eab25b14ed, 5de61f94bf. Literature: Gim et al., Prompt Cache, arXiv 2311.04934; Liu et al., CacheGen, arXiv 2310.07240; Yao et al., CacheBlend, arXiv 2405.16444; KVzip, arXiv 2505.23416; AgentKVShift, arXiv 2607.21604; "Can I Buy Your KV Cache?", arXiv 2606.13361; google/elephant-goldfish-model and Rensin, "Elephants, Goldfish and the New Golden Age of Software Engineering".

## Appendix B. The librarian's lens, in full

See `/data/pod/plans/librarian/lens.md`; the duties and prohibitions are paraphrased in 6.2. The answer shape it fixes:

```
Task: <the task, one line>
Questions:
  Q1 <sub-question>
    - <title> — <local path> — <why / which part>        [landed | not found]
  Q2 ...
Pod's own: <reports, knowledge-base entries, plans that already answer part of it>
Not in the ark: <what is missing> — candidate row: <kind, url, shelf, one-line note>
```
