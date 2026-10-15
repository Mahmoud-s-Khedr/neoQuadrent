// Project 1 experiment: semantic vs keyword retrieval on a labeled notes corpus, distance
// metrics, incremental sync (edit / delete / add), persistence, and the process lock.
import { spawnSync } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import type { Database } from 'neoquadrant'
import { dbPath, openDb } from '../shared/db.ts'
import { embedPassages, embedQuery } from '../shared/embed.ts'
import { mean, reciprocalRank, recallAtK } from '../shared/metrics.ts'
import { DATA_DIR } from '../shared/paths.ts'
import { check, finish, heading, saveResults, table, timed } from '../shared/report.ts'
import { COLLECTION, createCollection, keywordSearch, parseNote, semanticSearch, syncNotes, uuidV5 } from './lib.ts'

const HERE = path.dirname(fileURLToPath(import.meta.url))
const PROJECT = '01-notes-search'
const queries: { query: string; relevant: string[] }[] = JSON.parse(fs.readFileSync(path.join(HERE, 'queries.json'), 'utf8'))
const TARGET = queries[0].query

// Work on a copy of the corpus so edit/delete scenarios never touch the committed notes.
const notesDir = path.join(DATA_DIR, PROJECT, 'notes')
fs.rmSync(notesDir, { recursive: true, force: true })
fs.mkdirSync(path.dirname(notesDir), { recursive: true })
fs.cpSync(path.join(HERE, 'notes'), notesDir, { recursive: true })

let db: Database = openDb(PROJECT, { fresh: true })
let col = createCollection(db)

heading('Ingest')
const first = await timed('initial sync (embed + upsert)', () => syncNotes(col, notesDir))
console.log(`  ${first.added} notes, ${first.chunks} chunks`)
check('every note indexed', first.added === fs.readdirSync(notesDir).length)
check('count() matches chunks', col.count() === first.chunks, `count=${col.count()}`)
const again = await syncNotes(col, notesDir)
check('re-sync with no changes re-embeds nothing', again.chunks === 0 && again.unchanged === first.added)

heading('Retrieval quality: bge-small vs mxbai-large vs keyword (BM25)')
type Mode = 'small' | 'large' | 'keyword'
const MODES: Mode[] = ['small', 'large', 'keyword']
async function ranked(mode: Mode, query: string, limit = 10) {
  if (mode === 'keyword') return keywordSearch(col, query, limit)
  return semanticSearch(col, query, limit, undefined, mode === 'small' ? 'dense' : 'large')
}
const recall: Record<Mode, number[]> = { small: [], large: [], keyword: [] }
const rr: Record<Mode, number[]> = { small: [], large: [], keyword: [] }
const rows = []
for (const q of queries) {
  const rel = new Set(q.relevant)
  const row: Record<string, unknown> = { query: q.query.slice(0, 42) }
  for (const m of MODES) {
    const r = (await ranked(m, q.query)).map((h) => h.path)
    recall[m].push(recallAtK(r, rel, 5))
    rr[m].push(reciprocalRank(r, rel))
    row[`${m} R@5`] = recall[m].at(-1)!
  }
  rows.push(row)
}
table(rows)
const summary = Object.fromEntries(MODES.map((m) => [m, { recall_at_5: mean(recall[m]), mrr: mean(rr[m]) }])) as Record<Mode, { recall_at_5: number; mrr: number }>
table(MODES.map((m) => ({ mode: m, 'mean R@5': summary[m].recall_at_5, MRR: summary[m].mrr })))
check('both embedding models beat keyword search on recall@5', summary.small.recall_at_5 > summary.keyword.recall_at_5 && summary.large.recall_at_5 > summary.keyword.recall_at_5)
check('both embedding models beat keyword search on MRR', summary.small.mrr > summary.keyword.mrr && summary.large.mrr > summary.keyword.mrr)

heading(`Target query: "${TARGET}"`)
// The 4 notes that answer the question without sharing any of its words, and 4 off-topic
// "trap" notes that share its words (support response times, server room, restaurant server...).
const core = ['caching-layer', 'profiling-hot-paths', 'database-indexes', 'n-plus-one-queries']
const traps = ['support-response-times', 'server-room-cooling', 'time-management', 'restaurant-server-tips']
const top: Record<Mode, string[]> = { small: [], large: [], keyword: [] }
const full: Record<Mode, string[]> = { small: [], large: [], keyword: [] }
for (const m of MODES) {
  full[m] = (await ranked(m, TARGET, 60)).map((h) => h.path)
  top[m] = full[m].slice(0, 5)
}
table([0, 1, 2, 3, 4].map((i) => ({ rank: i + 1, ...Object.fromEntries(MODES.map((m) => [m, top[m][i] ?? ''])) })))
const coreRank = (m: Mode) => mean(core.map((c) => (full[m].includes(c) ? full[m].indexOf(c) + 1 : 60)))
const target = Object.fromEntries(
  MODES.map((m) => [m, { core_in_top5: top[m].filter((p) => core.includes(p)).length, traps_in_top5: top[m].filter((p) => traps.includes(p)).length, mean_core_rank: coreRank(m) }]),
)
table(MODES.map((m) => ({ mode: m, ...target[m] })))
console.log('  Finding: every small embedding model tried (bge-small/base, MiniLM, e5, nomic, arctic) and the')
console.log('  ms-marco cross-encoder rank the word-sharing traps above the zero-overlap answers. See FINDINGS.md.')
check('keyword search falls for the traps (expected)', target.keyword.traps_in_top5 >= 3)
check('the larger model ranks the zero-overlap answers better than bge-small', coreRank('large') < coreRank('small'), `mean rank ${coreRank('large').toFixed(1)} vs ${coreRank('small').toFixed(1)}`)
check('the larger model gets at least one zero-overlap answer into its top 5', target.large.core_in_top5 >= 1)
const semTop = top.large

heading('Distance metrics on normalized embeddings')
{
  const all = fs.readdirSync(notesDir).flatMap((f) => parseNote(f.replace(/\.md$/, ''), fs.readFileSync(path.join(notesDir, f), 'utf8')))
  const vecs = await embedPassages(all.map((c) => c.text)) // bge-small, unit-normalized
  const metrics = ['Cosine', 'Dot', 'Euclid'] as const
  for (const m of metrics) {
    db.dropCollection(`metric-${m}`)
    const c = db.createCollection(`metric-${m}`, { vectors: { size: 384, distance: m } })
    c.upsert(all.map((ch, i) => ({ id: i, vector: vecs[i], payload: { path: ch.path } })))
  }
  let identical = 0
  for (const q of queries) {
    const qv = await embedQuery(q.query)
    const ranks = metrics.map((m) => db.collection(`metric-${m}`).search(qv, { limit: 10 }).map((h) => h.id).join(','))
    if (ranks[0] === ranks[1] && ranks[1] === ranks[2]) identical++
  }
  const sample = db.collection('metric-Euclid').search(await embedQuery(TARGET), { limit: 1 })[0]
  const cos = db.collection('metric-Cosine').search(await embedQuery(TARGET), { limit: 1 })[0]
  console.log(`  identical top-10 for ${identical}/${queries.length} queries; best hit: cosine=${cos.score.toFixed(4)}, euclid distance=${sample.score.toFixed(4)} (expected √(2−2·cos)=${Math.sqrt(2 - 2 * cos.score).toFixed(4)})`)
  check('Cosine, Dot and Euclid rank identically for unit vectors', identical === queries.length)
  check('Euclid score is a distance: √(2 − 2·cos)', Math.abs(sample.score - Math.sqrt(2 - 2 * cos.score)) < 1e-3)
  for (const m of metrics) db.dropCollection(`metric-${m}`)
}

heading('Incremental sync: edit, delete, add')
{
  // Edit: rewrite the caching note to be about memoization; search must reflect it.
  fs.writeFileSync(
    path.join(notesDir, 'caching-layer.md'),
    '---\ntags: [backend, performance]\n---\n# Memoizing pure functions\n\n## Idea\n\nWrap expensive pure functions so repeated calls with the same arguments return a stored answer.\n',
  )
  // Delete: remove the password note.
  fs.rmSync(path.join(notesDir, 'password-hashing.md'))
  // Add: a new note.
  fs.writeFileSync(
    path.join(notesDir, 'graphql-dataloader.md'),
    '---\ntags: [backend, graphql]\n---\n# Batching resolvers with DataLoader\n\n## Problem\n\nNested GraphQL resolvers fetch related objects one at a time.\n\n## Fix\n\nDataLoader collects keys during a tick and loads them in one batch.\n',
  )
  const before = col.count()
  const s = await syncNotes(col, notesDir)
  console.log(`  sync: ${JSON.stringify(s)}`)
  check('sync detected 1 added, 1 updated, 1 deleted', s.added === 1 && s.updated === 1 && s.deleted === 1)
  check('count reflects changes', col.count() === before - 3 /* caching 3→1 */ - 3 /* password 3 chunks */ + 1 + 2, `count=${col.count()} before=${before}`)

  const pw = await semanticSearch(col, 'keep user passwords safe in storage', 10)
  check('deleted note never returned', !pw.some((h) => h.path === 'password-hashing'))
  check('deleted note gone from retrieve()', col.retrieve([uuidV5('password-hashing#0')]).length === 0)
  const memo = await semanticSearch(col, 'remember results of expensive function calls', 3)
  check('edited note found by its new content', memo[0]?.path === 'caching-layer', memo[0]?.path)
  const dl = await semanticSearch(col, 'n+1 problem in graphql', 3)
  check('added note is searchable', dl.some((h) => h.path === 'graphql-dataloader'), dl.map((h) => h.path).join(','))
  const filtered = await semanticSearch(col, TARGET, 10, { must: [{ key: 'tags', match: { value: 'database' } }] })
  check('tag filter restricts results', filtered.length > 0 && filtered.every((h) => ['database-indexes', 'n-plus-one-queries', 'connection-pooling', 'schema-migrations', 'document-databases', 'sql-vs-nosql', 'sql-injection', 'book-ddia'].includes(h.path)), filtered.map((h) => h.path).join(','))
}

heading('Persistence and locking')
{
  const before = (await semanticSearch(col, TARGET, 5)).map((h) => `${h.path}:${h.score.toFixed(5)}`)
  const count = col.count()
  db.close()
  db = openDb(PROJECT)
  col = db.collection(COLLECTION)
  const after = (await semanticSearch(col, TARGET, 5)).map((h) => `${h.path}:${h.score.toFixed(5)}`)
  check('reopened DB returns identical results', JSON.stringify(before) === JSON.stringify(after))
  check('reopened DB has the same count', col.count() === count)

  const child = spawnSync(
    process.execPath,
    ['-e', `const {Database}=require('neoquadrant');try{new Database(${JSON.stringify(dbPath(PROJECT))});console.log('opened')}catch(e){console.log(e.code)}`],
    { encoding: 'utf8', cwd: path.join(HERE, '..') },
  )
  check('a second process gets NEOQ_LOCKED', child.stdout.trim() === 'NEOQ_LOCKED', child.stdout.trim() || child.stderr.trim())
  const info = col.info()
  console.log(`  info: points=${info.points_count} segments=${info.segments_count} payload indexes=${Object.keys(info.payload_schema).join(',')}`)
  check('payload indexes persisted', 'path' in info.payload_schema && 'tags' in info.payload_schema)
}

saveResults(PROJECT, { modes: summary, target, target_top5: top })
db.close()
finish(PROJECT)
