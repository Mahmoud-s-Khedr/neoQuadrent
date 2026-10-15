// Notes → chunks → neoquadrant, with hash-based incremental sync.
import crypto from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'
import type { Collection, Database, Filter, PointStruct } from 'neoquadrant'
import { bm25Document, bm25Query } from '../shared/bm25.ts'
import { chunks } from '../shared/db.ts'
import { embedPassages, embedPassagesLarge, embedQuery, embedQueryLarge } from '../shared/embed.ts'

export const COLLECTION = 'notes'

export interface Chunk {
  path: string // note slug, e.g. "caching-layer"
  title: string
  heading: string
  tags: string[]
  text: string
  index: number
}

/** UUID v5 (SHA-1, URL namespace) — stable point ids derived from `path#index`. */
export function uuidV5(name: string): string {
  const ns = Buffer.from('6ba7b8119dad11d180b400c04fd430c8', 'hex')
  const h = crypto.createHash('sha1').update(ns).update(name).digest()
  h[6] = (h[6] & 0x0f) | 0x50
  h[8] = (h[8] & 0x3f) | 0x80
  const x = h.subarray(0, 16).toString('hex')
  return `${x.slice(0, 8)}-${x.slice(8, 12)}-${x.slice(12, 16)}-${x.slice(16, 20)}-${x.slice(20)}`
}

export function parseNote(slug: string, markdown: string): Chunk[] {
  const tags = markdown.match(/^tags:\s*\[(.*)\]/m)?.[1].split(',').map((t) => t.trim()).filter(Boolean) ?? []
  const body = markdown.replace(/^---[\s\S]*?---\n/, '')
  const title = body.match(/^# (.+)$/m)?.[1].trim() ?? slug
  const sections = body.split(/^## /m).slice(1)
  const parts = sections.length > 0 ? sections : [body]
  return parts.map((section, index) => {
    const [heading, ...rest] = section.split('\n')
    // Each chunk carries its note title so a section is understandable on its own.
    return { path: slug, title, heading: heading.trim(), tags, index, text: `${title}. ${heading.trim()}. ${rest.join(' ').replace(/\s+/g, ' ').trim()}` }
  })
}

export function createCollection(db: Database): Collection {
  const col = db.getOrCreateCollection(COLLECTION, {
    // Two embedding models side by side, as named vectors of different sizes.
    vectors: { dense: { size: 384, distance: 'Cosine' }, large: { size: 1024, distance: 'Cosine' } },
    sparse_vectors: { bm25: { modifier: 'idf' } },
  })
  col.createPayloadIndex('path', 'keyword')
  col.createPayloadIndex('tags', 'keyword')
  return col
}

export interface SyncStats {
  added: number
  updated: number
  deleted: number
  unchanged: number
  chunks: number
}

/** Bring the collection in line with the notes directory: only changed notes are re-embedded. */
export async function syncNotes(col: Collection, notesDir: string): Promise<SyncStats> {
  // What is indexed now: path → content hash (stored on every chunk of a note).
  const indexed = new Map<string, string>()
  let offset: string | number | undefined
  do {
    const page = col.scroll({ limit: 256, offset, with_payload: ['path', 'hash'] })
    for (const p of page.points) indexed.set(p.payload!.path as string, p.payload!.hash as string)
    offset = page.next_page_offset ?? undefined
  } while (offset !== undefined)

  const stats: SyncStats = { added: 0, updated: 0, deleted: 0, unchanged: 0, chunks: 0 }
  const onDisk = new Set<string>()
  const toIndex: { chunk: Chunk; hash: string }[] = []

  for (const file of fs.readdirSync(notesDir).filter((f) => f.endsWith('.md')).sort()) {
    const slug = file.replace(/\.md$/, '')
    onDisk.add(slug)
    const markdown = fs.readFileSync(path.join(notesDir, file), 'utf8')
    const hash = crypto.createHash('sha256').update(markdown).digest('hex').slice(0, 16)
    const previous = indexed.get(slug)
    if (previous === hash) {
      stats.unchanged++
      continue
    }
    if (previous !== undefined) {
      // A note can shrink to fewer sections: drop all its old chunks first.
      col.delete({ filter: { must: [{ key: 'path', match: { value: slug } }] } })
      stats.updated++
    } else {
      stats.added++
    }
    for (const chunk of parseNote(slug, markdown)) toIndex.push({ chunk, hash })
  }

  for (const slug of indexed.keys()) {
    if (!onDisk.has(slug)) {
      col.delete({ filter: { must: [{ key: 'path', match: { value: slug } }] } })
      stats.deleted++
    }
  }

  for (const batch of chunks(toIndex, 64)) {
    const texts = batch.map((b) => b.chunk.text)
    const [small, large] = [await embedPassages(texts), await embedPassagesLarge(texts)]
    const points: PointStruct[] = batch.map(({ chunk, hash }, i) => ({
      id: uuidV5(`${chunk.path}#${chunk.index}`),
      vector: { dense: small[i], large: large[i], bm25: bm25Document(chunk.text) },
      payload: { ...chunk, hash },
    }))
    await col.upsertAsync(points)
  }
  stats.chunks = toIndex.length
  return stats
}

export interface NoteHit {
  path: string
  score: number
  heading: string
}

/** One result per note (best chunk), in rank order. */
function dedupe(hits: { score: number; payload?: Record<string, unknown> }[], limit: number): NoteHit[] {
  const seen = new Set<string>()
  const out: NoteHit[] = []
  for (const h of hits) {
    const p = h.payload!.path as string
    if (seen.has(p)) continue
    seen.add(p)
    out.push({ path: p, score: h.score, heading: h.payload!.heading as string })
    if (out.length === limit) break
  }
  return out
}

export type Model = 'dense' | 'large'

export async function semanticSearch(col: Collection, query: string, limit = 10, filter?: Filter, model: Model = 'dense'): Promise<NoteHit[]> {
  const vector = model === 'large' ? await embedQueryLarge(query) : await embedQuery(query)
  const hits = col.query({ query: vector, using: model, limit: limit * 4, filter, with_payload: ['path', 'heading'] })
  return dedupe(hits, limit)
}

export function keywordSearch(col: Collection, query: string, limit = 10): NoteHit[] {
  const hits = col.query({ query: bm25Query(query), using: 'bm25', limit: limit * 4, with_payload: ['path', 'heading'] })
  return dedupe(hits, limit)
}
