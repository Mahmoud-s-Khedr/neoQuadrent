// Code search: three git repositories → syntax-aware chunks → one collection with two named
// vectors (`nl`: bge-small on code-as-text, `code`: jina-embeddings-v2-base-code), kept in sync
// incrementally from `git diff` against the last indexed commit (stored in a `meta` collection).
import { execFileSync } from 'node:child_process'
import crypto from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import type { Collection, Database, Filter, PointId, ScoredPoint } from 'neoquadrant'
import { chunks as batches } from '../shared/db.ts'
import { MODELS, embedCode, embedPassages, embedQuery } from '../shared/embed.ts'
import { uuidV5 } from '../shared/ids.ts'
import { DATASETS_DIR, EXAMPLES_DIR, QUICK } from '../shared/paths.ts'
import { type CodeChunk, chunkFile, languageOf } from './chunk.ts'

const HERE = path.dirname(fileURLToPath(import.meta.url))

export const CODE = 'code'
export const META = 'meta'

export interface RepoSpec {
  /** Short name stored in the `repo` payload field. */
  name: string
  dir: string
  /** Pathspecs to index (git ls-tree / git diff). */
  include: string[]
  /** Paths to skip, e.g. large test modules (keeps jina-code's CPU time bounded). */
  exclude?: RegExp
}

const ROOT = path.resolve(EXAMPLES_DIR, '..')

/** RealWorld "Conduit" backend: Express + Prisma, JWT auth via express-jwt, bcrypt (MIT). */
export const REALWORLD = {
  url: 'https://github.com/gothinkster/node-express-realworld-example-app',
  commit: '30b68e1e881462b2f4164ea09ab4c4f5699c7b0b',
  dir: path.join(DATASETS_DIR, 'p4', 'realworld'),
}

export function repos(): RepoSpec[] {
  const all: RepoSpec[] = [
    // This repository at HEAD (committed content only; the working tree is ignored).
    { name: 'neoquadrant', dir: ROOT, include: QUICK ? ['crates/neoquadrant/src', 'bindings/node/index.js', 'bindings/node/index.d.ts'] : ['crates', 'bindings'] },
    // Qdrant Edge at the vendored commit (the submodule's HEAD, i.e. without our patch).
    { name: 'qdrant-edge', dir: path.join(ROOT, 'vendor/qdrant'), include: QUICK ? ['lib/edge/src/edge_shard'] : ['lib/edge/src'], exclude: /(^|\/)tests?(\.rs$|\/)/ },
    { name: 'realworld', dir: REALWORLD.dir, include: ['src'] },
  ]
  return all
}

// ------------------------------------------------------------------ git

export function git(dir: string, ...args: string[]): string {
  return execFileSync('git', ['-C', dir, ...args], { encoding: 'utf8', maxBuffer: 1 << 28, stdio: ['ignore', 'pipe', 'pipe'] })
}

/** Clone the RealWorld app at its pinned commit (once). */
export function ensureRealworld() {
  if (!fs.existsSync(path.join(REALWORLD.dir, '.git'))) {
    fs.mkdirSync(path.dirname(REALWORLD.dir), { recursive: true })
    console.log(`  cloning ${REALWORLD.url}`)
    execFileSync('git', ['clone', '--quiet', REALWORLD.url, REALWORLD.dir], { stdio: 'inherit' })
  }
  if (git(REALWORLD.dir, 'rev-parse', 'HEAD').trim() !== REALWORLD.commit) git(REALWORLD.dir, 'checkout', '--quiet', '--detach', REALWORLD.commit)
}

const head = (repo: RepoSpec) => git(repo.dir, 'rev-parse', 'HEAD').trim()
const indexable = (repo: RepoSpec, file: string) => !!languageOf(file) && !repo.exclude?.test(file)

function listFiles(repo: RepoSpec, commit: string): string[] {
  return git(repo.dir, 'ls-tree', '-r', '-z', '--name-only', commit, '--', ...repo.include)
    .split('\0')
    .filter((f) => f && indexable(repo, f))
}

function readAt(repo: RepoSpec, commit: string, file: string): string {
  return git(repo.dir, 'cat-file', 'blob', `${commit}:${file}`)
}

/** Files added/modified and deleted between two commits (renames as delete + add). */
function diff(repo: RepoSpec, from: string, to: string): { changed: string[]; deleted: string[] } {
  const out = git(repo.dir, 'diff', '--name-status', '-z', '--no-renames', from, to, '--', ...repo.include).split('\0')
  const changed: string[] = []
  const deleted: string[] = []
  for (let i = 0; i + 1 < out.length; i += 2) {
    const [status, file] = [out[i], out[i + 1]]
    if (!indexable(repo, file)) continue
    ;(status === 'D' ? deleted : changed).push(file)
  }
  return { changed, deleted }
}

const commitExists = (repo: RepoSpec, commit: string) => {
  try {
    git(repo.dir, 'cat-file', '-e', `${commit}^{commit}`)
    return true
  } catch {
    return false
  }
}

// ------------------------------------------------------------------ collections

export function createCollections(db: Database): { code: Collection; meta: Collection } {
  const code = db.getOrCreateCollection(CODE, {
    vectors: { nl: { size: 384, distance: 'Cosine' }, code: { size: 768, distance: 'Cosine' } },
  })
  for (const field of ['repo', 'language', 'kind', 'file']) code.createPayloadIndex(field, 'keyword')
  // Word-tokenized full-text index: `{match: {text: "auth"}}` finds src/app/routes/auth/auth.ts.
  code.createPayloadIndex('path', { type: 'text', tokenizer: 'word', lowercase: true })
  // Sync state: one point per repository, payload only. A collection must declare at least one
  // vector, but (as in Qdrant) a point may omit named vectors, so these points carry none.
  const meta = db.getOrCreateCollection(META, { vectors: { unused: { size: 1, distance: 'Dot' } } })
  return { code, meta }
}

/** The text each model sees: the path and symbol give a chunk its context. Capped for CPU cost. */
const MAX_EMBED_CHARS = 2000
export const nlText = (c: CodeChunk) => `${c.path} ${c.kind} ${c.name}\n${c.text}`.slice(0, MAX_EMBED_CHARS)
export const codeText = (c: CodeChunk) => `// ${c.path}\n${c.text}`.slice(0, MAX_EMBED_CHARS)

export interface Embedder {
  nl(texts: string[]): Promise<number[][]>
  code(texts: string[]): Promise<number[][]>
}
export const defaultEmbedder: Embedder = { nl: (t) => embedPassages(t, 16), code: (t) => embedCode(t, 4) }

/**
 * Fingerprint of everything that shapes the stored chunks and vectors: the models, and the source
 * of the chunker and of this file (which defines the embedded text). Any change forces a full
 * rebuild instead of mixing old vectors with a new query embedder or chunker.
 */
let pipeline: string | undefined
export function pipelineVersion(): string {
  if (pipeline) return pipeline
  const h = crypto.createHash('sha1').update(JSON.stringify([MODELS.text, MODELS.code, MAX_EMBED_CHARS]))
  for (const f of ['chunk.ts', 'lib.ts']) h.update(fs.readFileSync(path.join(HERE, f)))
  return (pipeline = h.digest('hex').slice(0, 16))
}

export interface SyncStats {
  repo: string
  mode: 'full' | 'incremental' | 'unchanged'
  from: string | null
  to: string
  files_indexed: number
  files_deleted: number
  chunks_embedded: number
  chunks_kept: number
  chunks_deleted: number
  chunks_total: number
  seconds: number
}

const fileFilter = (repo: string, file: string): Filter => ({ must: [{ key: 'repo', match: { value: repo } }, { key: 'file', match: { value: file } }] })

/**
 * Point id of a chunk: content-addressed, so unchanged chunks keep their id (and embedding).
 * The kind and qualified name are part of it because the embedded text includes them: renaming
 * a class changes its methods' names (`Old.m` → `New.m`) without changing their text.
 */
export function chunkIds(repo: string, chunks: CodeChunk[]): string[] {
  const seen = new Map<string, number>()
  return chunks.map((c) => {
    const h = crypto.createHash('sha1').update(`${c.kind}\0${c.name}\0${c.text}`).digest('hex')
    const n = seen.get(h) ?? 0
    seen.set(h, n + 1)
    return uuidV5(`${repo}\0${c.path}\0${h}\0${n}`)
  })
}

function scrollAll(col: Collection, filter: Filter, with_payload: string[]): { id: PointId; payload: Record<string, unknown> }[] {
  const out: { id: PointId; payload: Record<string, unknown> }[] = []
  let offset: PointId | undefined
  do {
    const page = col.scroll({ filter, limit: 1000, offset, with_payload })
    for (const p of page.points) out.push({ id: p.id, payload: p.payload ?? {} })
    offset = page.next_page_offset ?? undefined
  } while (offset !== undefined)
  return out
}

/**
 * Bring the collection in line with the repository's HEAD. First run: index every file. Later
 * runs: only files in `git diff <last indexed commit> HEAD`, and within a changed file only
 * chunks whose text changed are re-embedded (moved ones get new line numbers).
 */
export async function syncRepo(code: Collection, meta: Collection, repo: RepoSpec, embedder: Embedder = defaultEmbedder): Promise<SyncStats> {
  const t0 = performance.now()
  const metaId = uuidV5(`repo:${repo.name}`)
  const state = meta.retrieve([metaId])[0]?.payload
  // What was indexed, not just where: a commit indexed with other paths (e.g. --quick) is not reusable.
  const scope = JSON.stringify({ include: repo.include, exclude: repo.exclude?.source ?? null, pipeline: pipelineVersion() })
  const last = state?.scope === scope ? ((state.commit as string | undefined) ?? null) : null
  const to = head(repo)
  const stats: SyncStats = { repo: repo.name, mode: 'full', from: last, to, files_indexed: 0, files_deleted: 0, chunks_embedded: 0, chunks_kept: 0, chunks_deleted: 0, chunks_total: 0, seconds: 0 }

  let changed: string[]
  let deleted: string[] = []
  if (last && last === to) {
    stats.mode = 'unchanged'
    changed = []
  } else if (last && commitExists(repo, last)) {
    stats.mode = 'incremental'
    ;({ changed, deleted } = diff(repo, last, to))
  } else {
    // First sync, the indexed paths changed, or the last indexed commit is gone (history
    // rewritten): start over.
    code.delete({ filter: { must: [{ key: 'repo', match: { value: repo.name } }] } })
    changed = listFiles(repo, to)
  }

  for (const file of deleted) {
    stats.chunks_deleted += code.count({ filter: fileFilter(repo.name, file), exact: true })
    code.delete({ filter: fileFilter(repo.name, file) })
    stats.files_deleted++
  }

  const toEmbed: { id: string; chunk: CodeChunk }[] = []
  for (const file of changed) {
    const chunks = await chunkFile(file, readAt(repo, to, file))
    const ids = chunkIds(repo.name, chunks)
    const existing = new Map(scrollAll(code, fileFilter(repo.name, file), ['start_line', 'end_line']).map((p) => [String(p.id), p.payload]))
    const stale = [...existing.keys()].filter((id) => !ids.includes(id))
    if (stale.length) code.delete(stale)
    stats.chunks_deleted += stale.length
    chunks.forEach((chunk, i) => {
      const old = existing.get(ids[i])
      if (!old) return toEmbed.push({ id: ids[i], chunk })
      stats.chunks_kept++
      if (old.start_line !== chunk.start_line || old.end_line !== chunk.end_line) code.setPayload({ start_line: chunk.start_line, end_line: chunk.end_line }, [ids[i]])
    })
    stats.files_indexed++
  }

  for (const batch of batches(toEmbed, 32)) {
    const [nl, cv] = [await embedder.nl(batch.map((b) => nlText(b.chunk))), await embedder.code(batch.map((b) => codeText(b.chunk)))]
    await code.upsertAsync(
      batch.map((b, i) => ({
        id: b.id,
        vector: { nl: nl[i], code: cv[i] },
        payload: { repo: repo.name, file: b.chunk.path, path: b.chunk.path, language: b.chunk.language, kind: b.chunk.kind, name: b.chunk.name, start_line: b.chunk.start_line, end_line: b.chunk.end_line, via: b.chunk.via, text: b.chunk.text },
      })),
    )
    stats.chunks_embedded += batch.length
  }

  stats.chunks_total = code.count({ filter: { must: [{ key: 'repo', match: { value: repo.name } }] }, exact: true })
  // Record the commit only after its chunks are written: a crash before this line re-syncs from `last`.
  meta.upsert([{ id: metaId, vector: {}, payload: { repo: repo.name, commit: to, scope, chunks: stats.chunks_total, synced_at: new Date().toISOString() } }])
  stats.seconds = (performance.now() - t0) / 1000
  return stats
}

// ------------------------------------------------------------------ search

export const MODES = ['nl', 'code', 'rrf'] as const
export type Mode = (typeof MODES)[number]

export async function search(col: Collection, mode: Mode, query: string, limit = 10, filter?: Filter): Promise<ScoredPoint[]> {
  const opts = { limit, filter, with_payload: ['repo', 'path', 'kind', 'name', 'start_line', 'end_line'] }
  switch (mode) {
    case 'nl':
      return col.queryAsync({ query: await embedQuery(query), using: 'nl', ...opts })
    case 'code':
      return col.queryAsync({ query: (await embedCode([query]))[0], using: 'code', ...opts })
    case 'rrf': {
      const [nl, cv] = [await embedQuery(query), (await embedCode([query]))[0]]
      return col.queryAsync({
        prefetch: [
          { query: nl, using: 'nl', limit: 50, filter },
          { query: cv, using: 'code', limit: 50, filter },
        ],
        query: { fusion: 'rrf' },
        ...opts,
      })
    }
  }
}
