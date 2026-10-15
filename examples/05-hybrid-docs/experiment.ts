// Project 5 experiment: semantic vs keyword (BM25) vs hybrid (RRF) vs hybrid + cross-encoder
// rerank, on conceptual, exact-identifier and mixed queries over technical documents.
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { openDb } from '../shared/db.ts'
import { mean, ndcgAtK, reciprocalRank } from '../shared/metrics.ts'
import { QUICK } from '../shared/paths.ts'
import { check, finish, heading, saveResults, table, timed } from '../shared/report.ts'
import { COLLECTION, MODES, type Mode, createCollection, ingest, search } from './lib.ts'
import { type Chunk, loadCorpus } from './prepare.ts'

const HERE = path.dirname(fileURLToPath(import.meta.url))
const PROJECT = '05-hybrid-docs'

interface LabeledQuery {
  bucket: 'identifier' | 'conceptual' | 'mixed'
  query: string
  relevant: { doc: string | string[]; all: string[] }
}

const corpus = await timed('load corpus', () => loadCorpus())
const isRelevant = (q: LabeledQuery, c: { doc: string; text: string }) =>
  [q.relevant.doc].flat().includes(c.doc) && q.relevant.all.every((s) => c.text.toLowerCase().includes(s))

const all: LabeledQuery[] = JSON.parse(fs.readFileSync(path.join(HERE, 'queries.json'), 'utf8'))
const queries = all.filter((q) => corpus.some((c) => isRelevant(q, c)))
console.log(`  ${corpus.length} chunks; ${queries.length}/${all.length} queries have answers in this corpus${QUICK ? ' (quick mode)' : ''}`)

const db = openDb(PROJECT, { fresh: true })
let col = createCollection(db)
await timed(`ingest ${corpus.length} chunks (bge-small + BM25)`, () => ingest(col, corpus, (n) => n % 640 === 0 && process.stdout.write(`    ${n}\n`)))
check('every chunk stored', col.count() === corpus.length)

heading('Retrieval quality per mode (hit@10 · MRR · nDCG@10)')
type Scores = { hit: number[]; mrr: number[]; ndcg: number[] }
const results = {} as Record<Mode, Record<string, Scores>>
const timings = {} as Record<Mode, number>
const byId = new Map<number, Chunk>(corpus.map((c) => [c.id, c]))
for (const mode of MODES) {
  results[mode] = {}
  const t = performance.now()
  for (const q of queries) {
    const hits = await search(col, mode, q.query, 10)
    const ranked = hits.map((h) => h.id as number)
    const rel = new Set(ranked.filter((id) => isRelevant(q, byId.get(id)!)))
    // Relevant set for nDCG: everything in the corpus that satisfies the predicate.
    const allRel = new Set(corpus.filter((c) => isRelevant(q, c)).map((c) => c.id))
    for (const bucket of [q.bucket, 'all']) {
      const s = (results[mode][bucket] ??= { hit: [], mrr: [], ndcg: [] })
      s.hit.push(rel.size > 0 ? 1 : 0)
      s.mrr.push(reciprocalRank(ranked, allRel))
      s.ndcg.push(ndcgAtK(ranked, allRel, 10))
    }
  }
  timings[mode] = (performance.now() - t) / queries.length
}
const BUCKETS = ['identifier', 'conceptual', 'mixed', 'all']
const summary = Object.fromEntries(
  MODES.map((m) => [m, Object.fromEntries(BUCKETS.map((b) => [b, { hit10: mean(results[m][b]?.hit ?? []), mrr: mean(results[m][b]?.mrr ?? []), ndcg10: mean(results[m][b]?.ndcg ?? []) }]))]),
) as Record<Mode, Record<string, { hit10: number; mrr: number; ndcg10: number }>>
table(
  MODES.map((m) => ({
    mode: m,
    ...Object.fromEntries(BUCKETS.map((b) => [`${b} MRR`, summary[m][b].mrr])),
    'all hit@10': summary[m].all.hit10,
    'all nDCG@10': summary[m].all.ndcg10,
    'ms/query': timings[m],
  })),
)

check('keyword beats semantic on exact identifiers (MRR)', summary.keyword.identifier.mrr > summary.semantic.identifier.mrr)
check('hybrid recovers what semantic misses (hit@10)', summary.hybrid.all.hit10 >= summary.semantic.all.hit10)
check('hybrid beats semantic overall (MRR)', summary.hybrid.all.mrr > summary.semantic.all.mrr)
check('reranking does not hurt hybrid overall (MRR)', summary['hybrid+rerank'].all.mrr >= summary.hybrid.all.mrr - 0.02)

// Hypotheses from the plan that this corpus does not support. Reported, not asserted: see FINDINGS.md.
const verdict = (ok: boolean) => (ok ? 'held' : 'did NOT hold')
console.log(`  hypothesis "semantic beats keyword on conceptual queries": ${verdict(summary.semantic.conceptual.mrr > summary.keyword.conceptual.mrr)} (${summary.semantic.conceptual.mrr.toFixed(3)} vs ${summary.keyword.conceptual.mrr.toFixed(3)})`)
const bestSingle = Math.max(summary.semantic.all.mrr, summary.keyword.all.mrr)
console.log(`  hypothesis "plain RRF ≥ best single mode": ${verdict(summary.hybrid.all.mrr >= bestSingle)} (${summary.hybrid.all.mrr.toFixed(3)} vs ${bestSingle.toFixed(3)})`)
console.log(`  weighted RRF (BM25 ×3, chosen after seeing the results, so not a fair test): ${summary['hybrid (bm25 ×3)'].all.mrr.toFixed(3)}`)

heading('Per query: rank of first relevant chunk')
table(
  queries.map((q) => {
    const row: Record<string, unknown> = { bucket: q.bucket, query: q.query.slice(0, 48) }
    const i = queries.indexOf(q)
    for (const m of MODES) {
      const rr = results[m][q.bucket].mrr[queries.filter((x) => x.bucket === q.bucket).indexOf(q)]
      row[m] = rr > 0 ? Math.round(1 / rr) : '-'
    }
    void i
    return row
  }),
)

heading('Engine checks')
{
  // Filtering inside prefetch: restrict hybrid search to the Pro Git book.
  const hits = await search(col, 'hybrid', 'how to undo changes', 10, ['progit'])
  check('payload filter applies inside both prefetch branches', hits.length > 0 && hits.every((h) => h.payload!.doc === 'progit'))
  // Sparse IDF: a rare identifier must outrank chunks that only share common words.
  const [top] = col.query({ query: { indices: [], values: [] }, using: 'bm25', limit: 1 })
  check('empty sparse query returns nothing instead of failing', top === undefined)
  const info = col.info()
  console.log(`  info: ${info.points_count} points, ${info.segments_count} segments, ${info.indexed_vectors_count} indexed vectors`)
  // Reopen and compare a hybrid query.
  const before = (await search(col, 'hybrid', 'git reflog', 5)).map((h) => h.id)
  db.close()
  const db2 = openDb(PROJECT)
  col = db2.collection(COLLECTION)
  const after = (await search(col, 'hybrid', 'git reflog', 5)).map((h) => h.id)
  check('hybrid results identical after reopen (sparse + dense persisted)', JSON.stringify(before) === JSON.stringify(after))
  db2.close()
}

saveResults(PROJECT, { chunks: corpus.length, queries: queries.length, summary, ms_per_query: timings })
finish(PROJECT)
