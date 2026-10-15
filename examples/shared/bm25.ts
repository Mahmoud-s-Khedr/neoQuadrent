// BM25 as sparse vectors, the way Qdrant does it: documents carry saturated term frequencies,
// and the collection's sparse vector is declared with `modifier: "idf"`, so neoquadrant/Qdrant
// computes IDF from its own statistics at query time. Queries are just the set of terms.
import type { SparseVector } from 'neoquadrant'

const STOPWORDS = new Set(
  `a an and are as at be but by for from has have how i if in into is it its of on or that the
   their there these this to was were what when where which who why will with you your can do does`.split(/\s+/),
)

/** Lowercase word tokens. Identifiers keep their dots split off: `fs.createReadStream` → fs, createreadstream. */
export function tokenize(text: string): string[] {
  return (text.toLowerCase().match(/[a-z0-9_]+/g) ?? []).filter((t) => t.length > 1 && !STOPWORDS.has(t))
}

/** 32-bit FNV-1a: stable token → sparse index without a vocabulary file. */
function hash(token: string): number {
  let h = 0x811c9dc5
  for (let i = 0; i < token.length; i++) {
    h ^= token.charCodeAt(i)
    h = Math.imul(h, 0x01000193)
  }
  return h >>> 0
}

export interface Bm25Options {
  k1?: number
  b?: number
  /** Average document length in tokens (fastembed's Qdrant/bm25 uses a fixed 256). */
  avgLen?: number
}

export function bm25Document(text: string, { k1 = 1.2, b = 0.75, avgLen = 256 }: Bm25Options = {}): SparseVector {
  const tokens = tokenize(text)
  const tf = new Map<number, number>()
  for (const t of tokens) tf.set(hash(t), (tf.get(hash(t)) ?? 0) + 1)
  const norm = k1 * (1 - b + (b * tokens.length) / avgLen)
  const indices = [...tf.keys()].sort((x, y) => x - y)
  return { indices, values: indices.map((i) => (tf.get(i)! * (k1 + 1)) / (tf.get(i)! + norm)) }
}

export function bm25Query(text: string): SparseVector {
  const indices = [...new Set(tokenize(text).map(hash))].sort((x, y) => x - y)
  return { indices, values: indices.map(() => 1) }
}
