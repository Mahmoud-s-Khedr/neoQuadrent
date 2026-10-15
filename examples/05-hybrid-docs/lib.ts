// Hybrid document search: one collection with a dense vector (bge-small) and a BM25 sparse
// vector (IDF computed by neoquadrant), queried four ways.
import type { Collection, Database, ScoredPoint } from 'neoquadrant'
import { bm25Document, bm25Query } from '../shared/bm25.ts'
import { chunks as batches } from '../shared/db.ts'
import { embedPassages, embedQuery, rerankScores } from '../shared/embed.ts'
import type { Chunk } from './prepare.ts'

export const COLLECTION = 'docs'
export const MODES = ['semantic', 'keyword', 'hybrid', 'hybrid (bm25 ×3)', 'hybrid+rerank'] as const
export type Mode = (typeof MODES)[number]

export function createCollection(db: Database): Collection {
  const col = db.createCollection(COLLECTION, {
    vectors: { dense: { size: 384, distance: 'Cosine' } },
    sparse_vectors: { bm25: { modifier: 'idf' } },
  })
  col.createPayloadIndex('doc', 'keyword')
  return col
}

export async function ingest(col: Collection, chunks: Chunk[], onProgress?: (done: number) => void) {
  let done = 0
  for (const batch of batches(chunks, 64)) {
    const dense = await embedPassages(batch.map((c) => c.text))
    await col.upsertAsync(
      batch.map((c, i) => ({
        id: c.id,
        vector: { dense: dense[i], bm25: bm25Document(c.text, { avgLen: 160 }) },
        payload: { doc: c.doc, location: c.location, text: c.text },
      })),
    )
    onProgress?.((done += batch.length))
  }
}

const RERANK_DEPTH = 30

export async function search(col: Collection, mode: Mode, query: string, limit = 10, docs?: string[]): Promise<ScoredPoint[]> {
  const filter = docs ? { must: [{ key: 'doc', match: { any: docs } }] } : undefined
  const sparse = bm25Query(query)
  switch (mode) {
    case 'keyword':
      return col.query({ query: sparse, using: 'bm25', limit, filter })
    case 'semantic':
      return col.query({ query: await embedQuery(query), using: 'dense', limit, filter })
    case 'hybrid':
    case 'hybrid (bm25 ×3)':
    case 'hybrid+rerank': {
      const dense = await embedQuery(query)
      const depth = mode === 'hybrid+rerank' ? RERANK_DEPTH : limit
      // Reciprocal Rank Fusion of the two candidate lists, computed inside the engine.
      // The weighted variant gives the BM25 list 3× the influence (Qdrant's position-compressing weights).
      const fused = col.query({
        prefetch: [
          { query: dense, using: 'dense', limit: 50, filter },
          { query: sparse, using: 'bm25', limit: 50, filter },
        ],
        query: mode === 'hybrid (bm25 ×3)' ? { rrf: { weights: [1, 3] } } : { fusion: 'rrf' },
        limit: depth,
      })
      if (mode !== 'hybrid+rerank') return fused
      const scores = await rerankScores(query, fused.map((h) => h.payload!.text as string))
      return fused
        .map((h, i) => ({ ...h, score: scores[i] }))
        .sort((a, b) => b.score - a.score)
        .slice(0, limit)
    }
  }
}
