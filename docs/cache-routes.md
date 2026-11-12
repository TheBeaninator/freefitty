# Cache routes: moving a conversation's cached state between inference servers

Three HTTP routes let a client save an inference server's cached state for a conversation, put it into another
server of the same model, and ask how much of a conversation a server already holds. FreeFitty uses them for
parking (a sleeping thread's state saved on the scheduler's disk), moving a thread to another node with its cache,
and frozen experts (a reading's state saved once and imported into any node). Without them FreeFitty still runs, and
every one of those becomes a re-read of the prompt.

The reference implementation is the pod's strixite build (`cache-xfer = on`); the pod's StrixFold build serves
export and import the same way. Neither patch is in the public repositories yet, so this document is the contract another engine
implements. The key words MUST, SHOULD and MAY are used as in RFC 2119.

## Terms

- **Prompt.** What the server's own chat template and tokenizer make of a chat-completions request body: the token
  ids it would prefill for that request. Every route takes a request body, never token ids, so the client needs no
  tokenizer and a server never has to trust one.
- **Entry.** A cached state of the model after some prefix of a prompt: the server's prompt cache, whatever its
  form (KV rows, recurrent state, both). An entry covers *n* tokens when it is the state after the first *n* tokens.
- **State bytes.** An entry serialized for transfer. Opaque to the client.

## `POST /cache/export`

Request: a chat-completions request body (`model`, `messages`, `tools`, any fields the server's template uses),
`Content-Type: application/json`.

The server renders and tokenizes the body as it would for a real request and finds the entry covering the longest
prefix of that prompt.

- `200`: the state bytes of that entry, `Content-Type: application/octet-stream`, with the header
  `X-Cache-Tokens: <n>`, the number of prompt tokens the entry covers. The body MUST be everything an import needs:
  if the entry is stored as a delta on a base, the bytes carry the base first and the delta after it.
- `404`: no entry covers any prefix of this prompt.
- `400`: the body is not a request the server can render (this includes an empty body; see Discovery).
- `409`: the server has no prompt cache (it runs with it off). This check comes before the body is read, so a
  server without a cache answers `409` even to an empty body.

The export MUST NOT change what the server holds. It SHOULD stream.

## `POST /cache/import`

Request: state bytes exactly as an export of the same model returned them, `Content-Type: application/octet-stream`.

- `200`: every entry in the body is now in the server's prompt cache, as the entry it was. Body:
  `{"status": "ok", "bytes": <n>}`. A later request whose prompt starts with the covered prefix reuses it as if this
  server had computed it.
- `409`: refused as foreign. The bytes come from a server of a different model, weights, quantization or any
  configuration that changes the state, or a delta arrived without its base. The server MUST check this; importing
  a foreign state silently would corrupt every answer that reuses it.
- `400`: the bytes are damaged (a header, token or checksum check failed).

The state bytes MUST be self-describing enough for those checks: a fingerprint of everything the state depends on,
the tokens it covers, and a checksum.

## `POST /cache/probe`

Request: a chat-completions request body, as for export.

- `200`: `{"tokens": n, "prompt_tokens": m, "ram": bool}`: the longest prefix of the prompt (*m* tokens) this server's
  cache holds (*n*, 0 for none), and whether that entry is in memory rather than only on disk. Read-only: nothing
  is loaded or moved.

Probe is OPTIONAL. Without it a client imports a state it holds whenever it places a conversation, even on a server
that already has it.

## Version

This is version 1 of the contract. A server SHOULD send `X-Cache-Routes: 1` on every response of the three routes,
errors included; a response without the header is version 1. A client MUST NOT use the routes of a server that
names a version it does not implement. A change that breaks a version 1 client gets a new number.

## Discovery

A client tells whether a server has the routes by sending `POST /cache/export` with an empty body:

- `400`: the routes exist and the server has a prompt cache (the empty body is not a request).
- `409`: the routes exist but the server has no prompt cache, so there is nothing to move. A server MUST check for
  its cache before it reads the body, so this answer takes precedence over the `400`.
- `404`: the routes do not exist.

Only a `400` whose `X-Cache-Routes` is absent or names a version the client implements means the routes are usable.
FreeFitty probes each node this way once it is up, and turns parking, moves and freezing off for a node where they
are not.

## What the engine has to cache

The routes only move what the prompt cache already holds, so two caching behaviours decide what they are worth:

- **A checkpoint at the start of the last user message.** When a request ends in a user message, the server SHOULD
  keep an entry covering everything before that message. A frozen expert depends on it: the client reads the expert
  (system prompt, its reading, its acknowledgement) followed by a short user message, then exports with that same
  body and gets the state that ends at the acknowledgement, which every thread born from the expert shares.
- **The entry for a finished turn.** A server MAY answer a turn before its entries are in the cache. A client
  exporting right after a turn retries until `X-Cache-Tokens` covers what it expects (FreeFitty waits up to about
  10 s).

A server MAY keep states only at fixed boundaries (StrixFold keeps them at 2,048-token chunk ends). An export then
covers up to one boundary less than the prompt and is still the conversation's state. FreeFitty's `node_meta`
`keep_step` tells it how much shortfall to accept.

## How FreeFitty uses them

| Use | Calls |
|---|---|
| Park a sleeping thread | export with the thread's last request, the bytes written to `<data_dir>/parked/<thread>.kv` |
| Wake it | probe on the node chosen; import the file unless the node already holds at least what it covers |
| Move a thread off a full node | export on the old node streamed straight into import on the new one |
| Freeze an expert | read it once, export with the reading's body, keep `<name>@<version>-<context>.kv` |
| Birth from an expert | import the frozen file into whichever node takes the thread's first turn |

## Security

Both export and import carry conversation contents, and an import changes what a server answers for every request
that shares the imported prefix: an import is a way to poison a cache. A server MUST require authentication on the
three routes on any interface other than loopback; on loopback it MAY leave them open.

The reference build does not meet this yet: the pod's strixite serves the routes without authentication, on a rail
network nothing outside the pod can reach. It fails checklist item 6 until it does. FreeFitty sends a node's `node_meta` `bearer_token`, when set, on these routes as on every other.

## Mapping another engine

An engine whose cache can be saved and restored per slot (llama.cpp's server has slot save and restore) can
implement the routes with an adapter. The adapter finds or computes the slot state for the body's prompt and saves
it for export; for import it restores it into a slot, keyed so the next request with that prefix reuses it. The
adapter MUST add the fingerprint and checksum the import checks require when the engine's own save format lacks
them.

## Conformance checklist

1. Export of a prompt the server has just answered returns `200`, `X-Cache-Tokens` at least the prompt length minus
   one boundary, and bytes that import on a second server of the same configuration.
2. After that import, the same request on the second server reports the covered tokens as cached (its usage
   counts them as cached prompt tokens) and does not prefill them again.
3. Import of those bytes into a server of a different model or configuration returns `409`; truncated or altered
   bytes return `400`.
4. Export of a prompt sharing no prefix with any entry returns `404`; an empty-body export returns `400`, or `409`
   when the server has no prompt cache.
5. Probe (if offered) returns the covered token count without changing what the server holds.
6. Off loopback, every route refuses a request without valid authentication.
