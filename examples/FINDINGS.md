# Findings from end-to-end testing

Bugs, limitations and surprises found while building the example projects. "neoquadrant" items are
in our code; "model" items are about the embedding models, not the database.

| # | Area | Finding | Status |
|---|------|---------|--------|
| 1 | neoquadrant (Node types) | `index.d.ts` exported an interface named `Record`, shadowing TypeScript's built-in `Record<K, V>`, so `Payload`, `CreateCollection` and `VectorInput` were broken types. It went unnoticed because `skipLibCheck` (on in most projects) skips `.d.ts` files. | **Fixed**: renamed to `PointRecord`. `npm run typecheck` in `bindings/node` and `examples/` now checks the declarations without `skipLibCheck`. |
| 2 | model | P1: on "how to reduce server response time", every small model tried ranks off-topic notes that share the query's words (support-ticket *response time*, restaurant *server*, *server* room) above the notes that answer it without sharing words (caching, profiling, DB indexes, N+1). Models tried: bge-small and bge-base, all-MiniLM-L6, e5-small, nomic-embed-v1.5, arctic-embed-m and the ms-marco MiniLM cross-encoder. The answers rank 8th to 33rd. mxbai-embed-large (335M) does better (profiling #4, mean rank 12.5 vs 16) but still lets one trap into its top 5. | Documented. P1 compares bge-small and mxbai-large as two named vectors in one collection. Overall, both still clearly beat BM25 (MRR 0.84 / 0.94 vs 0.54). |
