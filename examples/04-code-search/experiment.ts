// Project 4 experiment: code search over three repositories with two embedding models (bge-small
// on code-as-text, jina-code), compared alone and fused (RRF); incremental sync from `git diff`,
// proven with a real commit in a temporary clone.
import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import type { ScoredPoint } from 'neoquadrant'
import { openDb } from '../shared/db.ts'
import { mean, recallAtK, reciprocalRank } from '../shared/metrics.ts'
import { DATA_DIR, QUICK } from '../shared/paths.ts'
import { check, finish, heading, saveResults, table, timed } from '../shared/report.ts'
import { chunkFile } from './chunk.ts'
import { MODES, type Mode, REALWORLD, type RepoSpec, type SyncStats, chunkIds, createCollections, ensureRealworld, git, repos, search, syncRepo } from './lib.ts'

const HERE = path.dirname(fileURLToPath(import.meta.url))
const PROJECT = '04-code-search'

interface LabeledQuery {
  query: string
  relevant: { repo: string; path: string; name: string }[]
}

await timed('clone the RealWorld app at its pinned commit', () => ensureRealworld())
let db = openDb(PROJECT, { fresh: true })
let { code, meta } = createCollections(db)

// ------------------------------------------------------------------ ingest
heading(`Ingest: ${repos().map((r) => r.name).join(', ')}${QUICK ? ' (quick mode: subset of each)' : ''}`)
const ingest: SyncStats[] = []
for (const repo of repos()) {
  const s = await syncRepo(code, meta, repo)
  ingest.push(s)
  console.log(`  ${repo.name}: ${s.files_indexed} files → ${s.chunks_total} chunks in ${s.seconds.toFixed(1)} s`)
}
table(ingest.map((s) => ({ repo: s.repo, mode: s.mode, commit: s.to.slice(0, 10), files: s.files_indexed, chunks: s.chunks_total, embedded: s.chunks_embedded, 's (embed both models)': s.seconds })))
const totalChunks = code.count({ exact: true })
const embedSeconds = ingest.reduce((a, s) => a + s.seconds, 0)
console.log(`  ${totalChunks} chunks, ${(totalChunks / embedSeconds).toFixed(1)} chunks/s (bge-small + jina-code on CPU)`)
check('every repository produced chunks', ingest.every((s) => s.mode === 'full' && s.chunks_total > 0))
check('chunk counts add up', totalChunks === ingest.reduce((a, s) => a + s.chunks_total, 0))
table(code.facet({ key: 'kind', limit: 20 }).map((f) => ({ kind: f.value, chunks: f.count })))
table(code.facet({ key: 'language', limit: 20 }).map((f) => ({ language: f.value, chunks: f.count })))
const viaLines = code.count({ filter: { must: [{ key: 'via', match: { value: 'lines' } }] }, exact: true })
console.log(`  ${viaLines} chunks (${((100 * viaLines) / totalChunks).toFixed(1)}%) came from the line-window fallback (no grammar or parse failure)`)
{
  const states = meta.scroll({ limit: 10 }).points
  check('meta stores the indexed commit of every repository', states.length === repos().length && repos().every((r) => states.some((p) => p.payload!.repo === r.name && p.payload!.commit === git(r.dir, 'rev-parse', 'HEAD').trim())))
  const again = await Promise.all(repos().map((r) => syncRepo(code, meta, r)))
  check('re-sync at the same commits embeds nothing', again.every((s) => s.mode === 'unchanged' && s.chunks_embedded === 0))
}

// ------------------------------------------------------------------ payload indexes
heading('Payload indexes: keyword (repo, language, kind) and full-text (path)')
{
  const info = code.info()
  console.log(`  payload schema: ${Object.entries(info.payload_schema).map(([k, v]) => `${k}:${(v as { data_type?: string }).data_type ?? '?'}`).join(', ')}`)
  const byPath = code.scroll({ filter: { must: [{ key: 'path', match: { text: 'auth' } }] }, limit: 1000, with_payload: ['path'] }).points
  const tokens = (p: string) => p.toLowerCase().split(/[^a-z0-9]+/)
  check('text match on path finds files under auth/ and nothing else', byPath.length > 0 && byPath.every((p) => tokens(p.payload!.path as string).includes('auth')), `${byPath.length} chunks in ${new Set(byPath.map((p) => p.payload!.path)).size} files`)
  const rustFns = code.count({ filter: { must: [{ key: 'language', match: { value: 'rust' } }, { key: 'kind', match: { any: ['function', 'method'] } }] }, exact: true })
  console.log(`  rust functions + methods: ${rustFns}`)
  const hits = await search(code, 'rrf', 'verify the token', 5, { must: [{ key: 'repo', match: { value: 'realworld' } }, { key: 'kind', match: { value: 'function' } }] })
  check('filters apply inside both prefetches of the fused query', hits.length > 0 && hits.every((h) => h.payload!.repo === 'realworld' && h.payload!.kind === 'function'))
}

// ------------------------------------------------------------------ evaluation
const baseName = (name: string) => name.replace(/ \(part \d+\/\d+\)$/, '')
const key = (repo: string, p: string, name: string) => `${repo}\0${p}\0${baseName(name)}`
const all: LabeledQuery[] = JSON.parse(fs.readFileSync(path.join(HERE, 'queries.json'), 'utf8'))
const present = new Set(
  code.scroll({ limit: 100_000, with_payload: ['repo', 'path', 'name'] }).points.map((p) => key(p.payload!.repo as string, p.payload!.path as string, p.payload!.name as string)),
)
const queries = all.filter((q) => q.relevant.some((r) => present.has(key(r.repo, r.path, r.name))))
check('every labeled symbol of the full corpus exists as a chunk', QUICK || queries.length === all.length, `${queries.length}/${all.length}`)
const missing = all.flatMap((q) => q.relevant.filter((r) => !present.has(key(r.repo, r.path, r.name))).map((r) => `${r.path}:${r.name}`))
if (missing.length) console.log(`  not in this corpus: ${missing.join(', ')}`)

heading(`Retrieval: nl (bge-small) vs code (jina-code) vs RRF, ${queries.length} labeled queries`)
const rankedKeys = (hits: ScoredPoint[]) => [...new Set(hits.map((h) => key(h.payload!.repo as string, h.payload!.path as string, h.payload!.name as string)))]
const scores = Object.fromEntries(MODES.map((m) => [m, { recall5: [] as number[], rr: [] as number[], ms: [] as number[] }])) as Record<Mode, { recall5: number[]; rr: number[]; ms: number[] }>
const perQuery = []
for (const q of queries) {
  const rel = new Set(q.relevant.map((r) => key(r.repo, r.path, r.name)))
  const row: Record<string, unknown> = { query: q.query.slice(0, 50) }
  for (const m of MODES) {
    const t = performance.now()
    const ranked = rankedKeys(await search(code, m, q.query, 20))
    scores[m].ms.push(performance.now() - t)
    scores[m].recall5.push(recallAtK(ranked, rel, 5))
    scores[m].rr.push(reciprocalRank(ranked, rel))
    row[`${m} rank`] = scores[m].rr.at(-1)! > 0 ? Math.round(1 / scores[m].rr.at(-1)!) : '-'
  }
  perQuery.push(row)
}
table(perQuery)
const summary = Object.fromEntries(MODES.map((m) => [m, { recall_at_5: mean(scores[m].recall5), mrr: mean(scores[m].rr), ms_per_query: mean(scores[m].ms) }])) as Record<Mode, { recall_at_5: number; mrr: number; ms_per_query: number }>
table(MODES.map((m) => ({ mode: m, 'recall@5': summary[m].recall_at_5, MRR: summary[m].mrr, 'ms/query (incl. embedding)': summary[m].ms_per_query })))
const bestSingle = summary.nl.mrr >= summary.code.mrr ? 'nl' : 'code'
check('the best mode finds the labeled code in the top 5 for most queries (recall@5 ≥ 0.6)', Math.max(...MODES.map((m) => summary[m].recall_at_5)) >= 0.6)
const verdict = (ok: boolean) => (ok ? 'held' : 'did NOT hold')
console.log(`  hypothesis "the code model beats the general text model": ${verdict(summary.code.mrr > summary.nl.mrr)} (MRR ${summary.code.mrr.toFixed(3)} vs ${summary.nl.mrr.toFixed(3)})`)
console.log(`  hypothesis "RRF ≥ the best single model": ${verdict(summary.rrf.mrr >= summary[bestSingle].mrr)} (MRR ${summary.rrf.mrr.toFixed(3)} vs ${bestSingle} ${summary[bestSingle].mrr.toFixed(3)})`)

// ------------------------------------------------------------------ incremental sync
heading('Incremental sync: a real commit in a temporary clone')
const demoDir = path.join(DATA_DIR, PROJECT, 'sync-demo')
fs.rmSync(demoDir, { recursive: true, force: true })
fs.mkdirSync(path.dirname(demoDir), { recursive: true })
execFileSync('git', ['clone', '--quiet', '--no-hardlinks', REALWORLD.dir, demoDir])
git(demoDir, 'checkout', '--quiet', '-B', 'demo', REALWORLD.commit)
const demo: RepoSpec = { name: 'demo', dir: demoDir, include: ['src'] }
const sdb = openDb(`${PROJECT}-sync`, { fresh: true })
const s = createCollections(sdb)
const first = await syncRepo(s.code, s.meta, demo)
console.log(`  first sync: ${first.mode}, ${first.files_indexed} files, ${first.chunks_embedded} chunks embedded`)

// The commit: edit a function, add a file, delete a file, rename a file, and shift a file's
// lines without changing its functions.
const at = (p: string) => path.join(demoDir, p)
const tokenUtils = 'src/app/routes/auth/token.utils.ts'
fs.writeFileSync(at(tokenUtils), fs.readFileSync(at(tokenUtils), 'utf8').replace("expiresIn: '60d'", "expiresIn: '7d',\n    issuer: 'conduit'"))
const resetFile = 'src/app/routes/auth/password-reset.ts'
fs.writeFileSync(
  at(resetFile),
  `import * as crypto from 'crypto';

/** One-time token for the "forgot password" email; only its hash is stored. */
export const createPasswordResetToken = (): { token: string; hash: string; expires: Date } => {
  const token = crypto.randomBytes(32).toString('hex');
  const hash = crypto.createHash('sha256').update(token).digest('hex');
  return { token, hash, expires: new Date(Date.now() + 60 * 60 * 1000) };
};
`,
)
git(demoDir, 'rm', '--quiet', 'src/app/routes/tag/tag.service.ts')
git(demoDir, 'mv', 'src/app/routes/profile/profile.utils.ts', 'src/app/routes/profile/profile.mapper.ts')
const authService = 'src/app/routes/auth/auth.service.ts'
fs.writeFileSync(at(authService), `// Authentication service: registration, login, current user.\n\n${fs.readFileSync(at(authService), 'utf8')}`)
git(demoDir, 'add', '-A')
git(demoDir, '-c', 'user.name=neoquadrant examples', '-c', 'user.email=examples@neoquadrant.invalid', 'commit', '--quiet', '-m', 'demo change')
const commit2 = git(demoDir, 'rev-parse', 'HEAD').trim()

const second = await syncRepo(s.code, s.meta, demo)
const changedFiles = git(demoDir, 'diff', '--name-status', `${first.to}`, commit2).trim().split('\n')
console.log(`  git diff: ${changedFiles.map((l) => l.replace('\t', ' ')).join(', ')}`)
table([first, second].map((x) => ({ sync: x.mode, from: x.from?.slice(0, 8) ?? '-', to: x.to.slice(0, 8), 'files indexed': x.files_indexed, 'files deleted': x.files_deleted, embedded: x.chunks_embedded, kept: x.chunks_kept, deleted: x.chunks_deleted, total: x.chunks_total })))
const chunksAt = async (file: string) => chunkFile(file, git(demoDir, 'cat-file', 'blob', `${commit2}:${file}`))
const [resetChunks, mapperChunks, authChunks] = await Promise.all([chunksAt(resetFile), chunksAt('src/app/routes/profile/profile.mapper.ts'), chunksAt(authService)])
check('second sync is incremental from the first commit', second.mode === 'incremental' && second.from === first.to && second.to === commit2)
check('only the 4 changed files were read and the 2 deleted ones dropped', second.files_indexed === 4 && second.files_deleted === 2, `${second.files_indexed} indexed, ${second.files_deleted} deleted`)
const expectEmbedded = 1 /* generateToken */ + resetChunks.length + mapperChunks.length
check('only new or edited chunks were embedded', second.chunks_embedded === expectEmbedded, `${second.chunks_embedded} embedded, expected ${expectEmbedded}`)
check('chunks that only moved kept their embeddings', second.chunks_kept === authChunks.length, `${second.chunks_kept} kept`)
{
  // The incremental result must equal a full re-index of the new commit: same ids, same lines.
  const files = git(demoDir, 'ls-tree', '-r', '--name-only', commit2, '--', 'src').trim().split('\n')
  const expected = new Map<string, { file: string; start: number }>()
  for (const f of files) {
    const cs = await chunksAt(f)
    chunkIds('demo', cs).forEach((id, i) => expected.set(id, { file: f, start: cs[i].start_line }))
  }
  const stored = s.code.scroll({ limit: 100_000, with_payload: ['file', 'start_line'] }).points
  const same = stored.length === expected.size && stored.every((p) => expected.get(String(p.id))?.file === p.payload!.file && expected.get(String(p.id))?.start === p.payload!.start_line)
  check('incremental sync == full re-index of the new commit (ids, files, line numbers)', same, `${stored.length} stored vs ${expected.size} expected`)
  const gone = (file: string) => s.code.count({ filter: { must: [{ key: 'file', match: { value: file } }] }, exact: true }) === 0
  check('deleted and renamed-away files have no chunks left', gone('src/app/routes/tag/tag.service.ts') && gone('src/app/routes/profile/profile.utils.ts'))
  const metaState = s.meta.scroll({ limit: 10 }).points
  check('meta records the new commit', metaState.length === 1 && metaState[0].payload!.commit === commit2)
}
const found = await search(s.code, 'rrf', 'generate a token for resetting a forgotten password', 3)
check('the added function is searchable', found.some((h) => h.payload!.name === 'createPasswordResetToken'), found.map((h) => h.payload!.name).join(', '))
const third = await syncRepo(s.code, s.meta, demo)
check('a sync with no new commit does nothing', third.mode === 'unchanged' && third.chunks_embedded === 0)
sdb.close()

// ------------------------------------------------------------------ persistence
heading('Persistence')
{
  const q = queries[0]?.query ?? 'open the database'
  const before = (await search(code, 'rrf', q, 5)).map((h) => h.id)
  db.close()
  db = openDb(PROJECT)
  ;({ code, meta } = createCollections(db))
  const after = (await search(code, 'rrf', q, 5)).map((h) => h.id)
  check('fused results identical after reopen', JSON.stringify(before) === JSON.stringify(after))
  const resumed = await Promise.all(repos().map((r) => syncRepo(code, meta, r)))
  check('sync state survives reopen (nothing to re-embed)', resumed.every((x) => x.mode === 'unchanged'))
}

saveResults(PROJECT, {
  corpus: { chunks: totalChunks, line_window_chunks: viaLines, repos: ingest.map((x) => ({ repo: x.repo, commit: x.to, files: x.files_indexed, chunks: x.chunks_total, seconds: x.seconds })) },
  queries: queries.length,
  summary,
  per_query: perQuery,
  sync: { first, second, third },
})
db.close()
finish(PROJECT)
