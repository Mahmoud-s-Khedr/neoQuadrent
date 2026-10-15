# neoquadrant

**Qdrant's vector search engine as an in-process library: open a directory, no server.**

Think SQLite, but for vectors: `Database("./data")` inside your Node.js, Python or Rust app.
Search quality is Qdrant's own, because the engine *is* Qdrant: the same segment storage,
HNSW index, payload indexes, quantization, filters, sparse vectors and write-ahead log that
the Qdrant server runs. It comes from [Qdrant Edge](https://github.com/qdrant/qdrant/tree/master/lib/edge),
vendored at a pinned release.

```
┌────────────────── your app process ───────────────────┐
│  Node.js (napi-rs)   Python (PyO3)   Rust             │
│        └───────────────┬───────────────┘              │
│        neoquadrant core (crates/neoquadrant)          │
│   Database of collections · process lock · auto-      │
│   indexing · durability · JSON API · validation        │
│                        │                              │
│        Qdrant engine (vendor/qdrant, v1.19.2)         │
│   lib/edge → lib/shard → lib/segment (HNSW, WAL, …)   │
└──────────────────────────┬────────────────────────────┘
                           ▼
              ./data/collections/<name>/{segments,wal}
```

## Quick start

### Node.js / TypeScript

```js
const { Database } = require('neoquadrant')

const db = new Database('./vectors')            // or new Database() for a temp DB
const docs = db.getOrCreateCollection('docs', { vectors: { size: 384, distance: 'Cosine' } })

docs.createPayloadIndex('lang', 'keyword')       // index fields you filter on
docs.upsert([
  { id: 1, vector: embedding /* number[] | Float32Array */, payload: { lang: 'en', title: 'Hello' } },
])

const hits = docs.search(queryEmbedding, {
  limit: 5,
  filter: { must: [{ key: 'lang', match: { value: 'en' } }] },
})
// [{ id: 1, score: 0.93, payload: { lang: 'en', title: 'Hello' } }]

await docs.searchAsync(queryEmbedding, { limit: 5 })  // runs off the event loop
db.close()
```

### Python

```python
import neoquadrant

with neoquadrant.Database("./vectors") as db:
    docs = db.get_or_create_collection("docs", {"vectors": {"size": 384, "distance": "Cosine"}})
    docs.create_payload_index("lang", "keyword")
    docs.upsert_batch(ids, embeddings_np_matrix, [{"lang": "en"} for _ in ids])
    hits = docs.search(query_np_vector, limit=5,
                       filter={"must": [{"key": "lang", "match": {"value": "en"}}]})
```

Thread-safe: the GIL is released during searches, upserts and optimization.

### Rust

```rust
use neoquadrant::{Database, requests::*};
use serde_json::json;

let db = Database::open("./vectors")?;
let docs = db.get_or_create_collection("docs",
    serde_json::from_value(json!({"vectors": {"size": 384, "distance": "Cosine"}}))?)?;
docs.upsert(serde_json::from_value(json!([{"id": 1, "vector": embedding, "payload": {"lang": "en"}}]))?)?;
let hits = docs.search(query, 5)?;
```

The engine's typed API is re-exported as `neoquadrant::engine`.

## The API is Qdrant's

Requests use the same JSON shapes as Qdrant's REST API, so
[Qdrant's documentation](https://qdrant.tech/documentation/concepts/) applies directly:

| Feature | Example |
|---|---|
| Filters | `{"must": [...], "should": [...], "must_not": [...]}` with `match` (`value`/`any`/`except`/`text`), `range`, `has_id`, `is_empty`, `is_null`, `nested`, geo, `values_count` |
| Named + sparse vectors | `{"vectors": {"dense": {...}}, "sparse_vectors": {"text": {}}}` |
| Hybrid search | `{"prefetch": [{...dense}, {...sparse}], "query": {"fusion": "rrf"}}` |
| Multivectors (ColBERT) | `{"size": 128, "distance": "Cosine", "multivector_config": {"comparator": "max_sim"}}` |
| Quantization | `"quantization_config": {"scalar": {"type": "int8"}}` (also `product`, `binary`) |
| On-disk storage | `"memory": "cold"` per vector, `"on_disk_payload": true` |
| Payload indexes | `keyword`, `integer`, `float`, `bool`, `geo`, `datetime`, `text`, `uuid` |
| Order by / facets | `{"query": {"order_by": "price"}}`, `facet({"key": "color"})` |
| Search tuning | `"params": {"hnsw_ef": 256, "exact": false}` |

Two deliberate differences: results include the payload by default (`with_payload: true`), and an
unknown vector name or malformed filter is an error rather than an empty result.

## What this adds on top of Qdrant Edge

Qdrant Edge is the engine. neoquadrant adds what an embedded database needs:

- **A database of collections.** One directory holds many collections (one Edge shard each).
- **A process lock.** A second *process* opening the same directory gets `NEOQ_LOCKED`/`LockedError`.
  Opening it twice in one process shares a single handle, like SQLite connections.
- **Background indexing.** Edge only builds HNSW when you call `optimize()`. Here a background
  thread runs Qdrant's optimizers once writes settle (`optimizeDebounceMs`, default 500 ms).
- **A durability policy.** `full` (default for directories) flushes on every write call, about
  0.3 ms on an SSD. `periodic` flushes on an interval. `manual` flushes only when you call `flush()`.
- **Crash recovery.** See [patches](#patches-to-qdrant). Edge v1.19.2 never replayed its WAL.
- **Validation.** Qdrant's own request validators, which the server runs at its API layer and Edge skips.
- **Bindings.** Node.js (Edge has none) and Python, with one JSON API across all three languages.

## Patches to Qdrant

Qdrant is a git submodule pinned to `v1.19.2`. Changes are kept small and reviewable in
[`patches/`](patches) and applied by `scripts/apply-patches.sh`:

- **`0001-edge-replay-wal-on-load-and-ack-after-flush.patch`**: Edge wrote every operation to its
  WAL but never replayed it on load, so writes since the last flush were lost on a crash. It also
  never truncated the WAL, so the WAL grew without bound. The patch replays unacknowledged WAL
  entries on load, as the server's `LocalShard` does, and acknowledges the WAL after each confirmed
  flush. Qdrant Edge's own test suite (102 tests) passes with it. It's a candidate to send upstream.

## Performance (100k × 128-d, laptop, release build)

| | |
|---|---|
| Bulk insert (batches of 1000) | ~75k points/s |
| Background HNSW build | ~14–20 s after insert |
| Search, top-10 | ~1,400–1,900 QPS single-threaded |
| Filtered search, indexed field, 10% selectivity | ~530 QPS |
| Single-point write, `full` durability | ~0.3 ms |
| Single-point write, `periodic`/`manual` | ~0.02 ms |

Recall depends on your data and `hnsw_ef`. On synthetic clustered data, recall@10 goes from
0.69 at the default `hnsw_ef` to 0.98 at `hnsw_ef: 512`. Synthetic random data is close to the worst
case for any ANN index, so measure on your own embeddings with
`cargo run --release -p neoquadrant --example bench`.

## Examples

[`examples/`](examples) holds five retrieval projects that use neoquadrant like an outside Node.js app
would, with real datasets and local CPU embedding models: notes search, image similarity and
near-duplicates (CLIP), duplicate-question detection, code search with incremental git sync, and
hybrid (dense + BM25) document search. Each experiment checks its own claims; `npm run e2e` runs
them all. Results are in [examples/README.md](examples/README.md), and the bugs, limitations and
surprises found while building them are in [examples/FINDINGS.md](examples/FINDINGS.md).

## Building from source

```bash
git clone --recurse-submodules <repo> && cd neoquadrant
./scripts/apply-patches.sh

cargo test --release -p neoquadrant                    # Rust: core, conformance, crash recovery
cd bindings/node && npm run build && npm test          # Node.js
cd bindings/python && uv venv && uv pip install maturin pytest numpy \
  && .venv/bin/maturin develop --release && .venv/bin/pytest tests   # Python
```

Requires Rust 1.98+ (pinned in `rust-toolchain.toml`). A cold build compiles Qdrant's engine and
takes a few minutes.

### Tests

- `tests/conformance/cases.json` is one behaviour spec, run identically by the Rust, Node and
  Python test suites, so the three APIs cannot drift apart.
- Crash recovery: a child process writes without flushing and calls `abort()`. After reopening,
  every acknowledged write must be present.
- The background optimizer must build HNSW, and recall is checked against exact search.

### Upgrading Qdrant

```bash
git -C vendor/qdrant fetch --depth 1 origin tag vX.Y.Z && git -C vendor/qdrant checkout vX.Y.Z
./scripts/apply-patches.sh   # rebase the patch if it no longer applies
cargo test --release -p neoquadrant && (cd bindings/node && npm run build && npm test)
```

## Layout

```
crates/neoquadrant/     core: Database, Collection, requests (JSON API), background worker
bindings/node/          napi-rs native module + JS wrapper (index.js) + TypeScript types
bindings/python/        PyO3 native module + Python wrapper + type stubs
tests/conformance/      shared cross-language behaviour spec
examples/               five end-to-end retrieval projects (Node.js), results and FINDINGS.md
patches/                changes to vendored Qdrant
vendor/qdrant/          Qdrant source (git submodule, v1.19.2)
```

## License

Apache-2.0. Embeds Qdrant (Apache-2.0); see [NOTICE](NOTICE).
