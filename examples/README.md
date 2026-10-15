# neoquadrant examples

Five retrieval projects that use neoquadrant the way an outside Node.js app would: real datasets,
local CPU embedding models ([transformers.js](https://huggingface.co/docs/transformers.js)), and
experiments that check their own claims. A failed check is reported, not hidden. Hypotheses that did
not hold are printed as such. Bugs and surprises are logged in [FINDINGS.md](FINDINGS.md).

| # | Project | What it exercises |
|---|---------|-------------------|
| 1 | [Notes search](01-notes-search) | named vectors of two sizes, BM25 sparse vectors, UUID ids, hash-based incremental sync, persistence, the process lock |
| 2 | [Image similarity](02-image-similarity) | CLIP image + text vectors, keyword/bool/integer indexes, near-duplicate thresholds, Cosine vs Dot, 512 vs 256 dims, HNSW recall vs exact |
| 3 | [Duplicate questions](03-duplicate-questions) | `score_threshold`, `has_id` filters, `facet`, threshold sweeps on QQP, look-alikes embeddings cannot separate |
| 4 | [Code search](04-code-search) | tree-sitter chunking, two code/text named vectors, RRF, text index on paths, incremental sync from `git diff` |
| 5 | [Hybrid docs](05-hybrid-docs) | dense + BM25 hybrid (RRF, weighted RRF), cross-encoder rerank, filters inside prefetch |

## Running

```bash
cd bindings/node && npm run build && cd ../../examples   # native module first
npm ci
npm run typecheck
npm run p1            # … p5; each takes --quick for a smaller dataset
npm run e2e           # all five with --quick; exits non-zero on any failure
npm run e2e -- --only p2,p4
```

Models (bge-small, mxbai-embed-large, jina-embeddings-v2-base-code, CLIP ViT-B/32, an ms-marco
cross-encoder) download from huggingface.co on first use into `.data/models`. Datasets go to
`.data/datasets`, databases to `.data/db` and metrics to `.data/results/<project>.json`; everything
under `.data/` is gitignored. `EXAMPLES_DATA_DIR` moves it elsewhere.

Network sources: huggingface.co (models; P3's QQP parquet), s3.amazonaws.com (P2's Imagenette),
github.com (P4's RealWorld clone; P5's Pro Git PDF), raw.githubusercontent.com and arxiv.org (P5).
Behind an HTTPS proxy, run with `NODE_USE_ENV_PROXY=1` (Node ≥ 22.21), because Node's built-in
`fetch` ignores `HTTPS_PROXY` otherwise. `examples/.npmrc` skips onnxruntime-node's optional CUDA
download (`onnxruntime-node-install=skip`); the CPU binaries ship with the package.

## Results

Full-size runs (no `--quick`) on a laptop CPU.

### 1. Notes search: 54 notes, 12 labeled queries

| Retriever | MRR |
|-----------|-----|
| bge-small (384-d) | 0.84 |
| mxbai-embed-large (1024-d) | 0.94 |
| BM25 (sparse, IDF by neoquadrant) | 0.54 |

On "how to reduce server response time", every small model ranks word-sharing off-topic notes
above the answers that share no words (FINDINGS #2).

### 2. Image similarity: Imagenette, 3,000 originals + 300 near-duplicates

**Not measured yet.** The code is complete, typechecks, and every engine-side step was run against
the real engine with stand-in vectors (see "How P2 and P4 were verified" below). The environment
that built it could not reach huggingface.co, so no CLIP numbers exist yet. Run `npm run p2` and fill in:

| Measure | Value |
|---------|-------|
| Similar images, label precision@10 (self and own duplicates excluded) | – |
| Near-duplicate recall@1 of the source | – |
| Best threshold: precision / recall / F1 | – |
| Hardest augmentation (recall@1) | – |
| False duplicates at that threshold: same-class / other-class | – |
| Text → image P@10: class prompts / descriptions | – |
| Cosine vs Dot (raw vectors) P@10 | – |
| 512 / 256-truncated / 256-PCA P@10 | – |
| HNSW recall@10 vs exact (default `hnsw_ef`) | – |

### 3. Duplicate questions: 5,000 QQP pairs + 120 workbook questions

| Measure | Value |
|---------|-------|
| Best threshold on QQP | 0.82 (F1 0.73) |
| Engine score vs local cosine, all pairs | identical |
| Mean cosine: paraphrase / look-alike with a different answer | 0.910 / 0.901 |
| Look-alikes flagged as duplicates at 0.82 | 26 of 30 |
| `numbers` keyword filter: number look-alikes removed / paraphrases kept | 10 of 10 / 26 of 28 |

### 4. Code search: neoquadrant crates + bindings, Qdrant Edge `lib/edge/src`, RealWorld Express app

**Retrieval not measured yet** (same reason as P2: jina-code and bge-small could not be downloaded).
The corpus and sync numbers below come from the engine, which doesn't depend on the models:

| Repository | Files | Chunks |
|------------|-------|--------|
| neoquadrant (`crates/`, `bindings/`, commit `5fcadfe`) | 25 | 363 |
| qdrant-edge (`lib/edge/src` at v1.19.2, tests skipped) | 74 | 460 |
| realworld (`src/`, commit `30b68e1e`) | 38 | 91 |

2.2% of chunks came from the line-window fallback (SQL migrations, TOML, Prisma schema); the rest
come from tree-sitter.

Incremental sync, proven with a real commit in a temporary clone (edit a function, add a file,
delete a file, rename a file, shift a file's lines): the sync read 4 changed files, dropped 2,
**re-embedded 3 chunks out of 91**, kept 5 moved chunks with updated line numbers, and the result
was identical (ids, files, lines) to a full re-index of the new commit.

| Mode | recall@5 | MRR |
|------|----------|-----|
| `nl` (bge-small on code-as-text) | – | – |
| `code` (jina-embeddings-v2-base-code) | – | – |
| RRF of both | – | – |

### 5. Hybrid docs: 2,402 chunks (Pro Git, HNSW paper, Node.js API docs), 30 labeled queries

| Mode | MRR |
|------|-----|
| semantic (bge-small) | 0.63 |
| keyword (BM25) | 0.76 |
| hybrid (RRF) | 0.72 |
| hybrid, BM25 weighted ×3 (chosen after seeing results) | 0.78 |

Two planned hypotheses did not hold: semantic did not beat BM25 on conceptual queries, and plain
RRF landed between the two retrievers (FINDINGS #7).

## How P2 and P4 were verified

`huggingface.co` (and `arxiv.org`) were blocked by the network policy of the environment where P2
and P4 were written, so no model could be loaded and no project could run for real there. `npm run e2e`
there correctly reports all five projects as crashed. To still exercise every code path and
engine call, both experiments were run with `shared/embed.ts` swapped (via a Node module hook kept
outside the repository) for deterministic stand-in vectors. That proves the pipelines, filters,
indexes, sync logic and checks run; it proves nothing about retrieval quality. Their quality
checks (precision, recall, MRR thresholds) are hypotheses until the real run.

## Layout

```
shared/          model loaders, BM25 encoder, metrics, report/check helpers, e2e runner
01-notes-search/ … 05-hybrid-docs/   one directory per project: prepare.ts (data), lib.ts, experiment.ts
FINDINGS.md      bugs, limitations and surprises found along the way
```
