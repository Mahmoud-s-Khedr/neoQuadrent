// Project 3 experiment: exact duplicates via a payload index, paraphrase thresholds on QQP,
// and the embedding blind spot — look-alike questions with different answers.
import crypto from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import type { Collection, Filter } from 'neoquadrant'
import { chunks, openDb } from '../shared/db.ts'
import { embedTexts } from '../shared/embed.ts'
import { mean, range, thresholdSweep } from '../shared/metrics.ts'
import { QUICK } from '../shared/paths.ts'
import { check, finish, heading, saveResults, table, timed } from '../shared/report.ts'
import { loadQqp } from './prepare.ts'

const HERE = path.dirname(fileURLToPath(import.meta.url))
const PROJECT = '03-duplicate-questions'
const N_PAIRS = QUICK ? 1000 : 5000

interface Question {
  id: string
  group: string
  kind: 'original' | 'exact_duplicate' | 'paraphrase' | 'lookalike'
  lookalike_kind?: 'numbers' | 'wording' | 'wording_same_answer'
  concept: string
  text: string
  answer: string
}

/** Case, whitespace and punctuation-insensitive form used for exact-duplicate detection. */
// Math operators must survive: "3a - 4" and "3a + 4" are different questions.
export const normalize = (t: string) => t.toLowerCase().replace(/[^\p{L}\p{N}%$/.²+\-=*×÷<>^]+/gu, ' ').replace(/\s*\.\s*$/, '').trim()
const normHash = (t: string) => crypto.createHash('sha1').update(normalize(t)).digest('hex').slice(0, 16)
/** The numbers a question mentions as a multiset ("mode of 3, 7, 7" ≠ "mode of 3, 3, 7"). */
export const numbersKey = (t: string) => (t.match(/\d+(?:\.\d+)?/g) ?? []).sort().join('|')

const db = openDb(PROJECT, { fresh: true })

// ------------------------------------------------------------------ QQP: thresholds
heading(`QQP: paraphrase detection on ${N_PAIRS} labeled pairs`)
const pairs = await timed('load QQP', () => loadQqp(N_PAIRS))
const texts = [...new Set(pairs.flatMap((p) => [p.question1, p.question2]))]
const idOf = new Map(texts.map((t, i) => [t, i]))
const qqp = db.createCollection('qqp', { vectors: { size: 384, distance: 'Cosine' } })
const vectors = await timed(`embed ${texts.length} unique questions`, () => embedTexts(texts))
await timed('upsert', async () => {
  for (const batch of chunks(texts.map((t, i) => i), 500)) {
    await qqp.upsertAsync(batch.map((i) => ({ id: i, vector: vectors[i], payload: { text: texts[i] } })))
  }
})

// Pair similarity through the engine: search with q1's vector, restricted to q2 by a has_id filter.
const dot = (a: number[], b: number[]) => a.reduce((s, x, i) => s + x * b[i], 0)
let mismatches = 0
const scored = await timed('score pairs via has_id-filtered queries', async () =>
  Promise.all(
    pairs.map(async (p) => {
      const [a, b] = [idOf.get(p.question1)!, idOf.get(p.question2)!]
      const hit = (await qqp.searchAsync(vectors[a], { limit: 1, filter: { must: [{ has_id: [b] }] }, with_payload: false }))[0]
      const score = a === b ? 1 : (hit?.score ?? 0)
      if (Math.abs(score - dot(vectors[a], vectors[b])) > 1e-3) mismatches++
      return { score, positive: p.label === 1 }
    }),
  ),
)
check('engine scores equal the locally computed cosine for every pair', mismatches === 0, `${mismatches} mismatches`)

const sweep = thresholdSweep(scored, range(0.7, 0.98, 0.02))
table(sweep.filter((_, i) => i % 2 === 0).map((r) => ({ threshold: r.threshold, precision: r.precision, recall: r.recall, f1: r.f1 })))
const best = sweep.reduce((a, b) => (b.f1 > a.f1 ? b : a))
console.log(`  best threshold ${best.threshold}: precision ${best.precision.toFixed(3)}, recall ${best.recall.toFixed(3)}, F1 ${best.f1.toFixed(3)}`)
check('best-threshold F1 on QQP ≥ 0.6', best.f1 >= 0.6, best.f1.toFixed(3))

// score_threshold: candidate duplicates for a few questions across the whole corpus.
{
  const probes = [...idOf.values()].slice(0, 100)
  const results = await Promise.all(
    probes.map((i) => qqp.searchAsync(vectors[i], { limit: 5, score_threshold: best.threshold, filter: { must_not: [{ has_id: [i] }] }, with_payload: false })),
  )
  const all = results.flat()
  check('score_threshold drops every hit below the threshold', all.every((h) => h.score >= best.threshold), `${all.length} candidates`)
  check('must_not has_id excludes the probe itself', results.every((r, k) => r.every((h) => h.id !== probes[k])))
}

// ------------------------------------------------------------------ workbook questions
heading('Workbook questions: exact duplicates, paraphrases, look-alikes')
const questions: Question[] = JSON.parse(fs.readFileSync(path.join(HERE, 'workbook.json'), 'utf8'))
const wb = db.createCollection('workbook', { vectors: { size: 384, distance: 'Cosine' } })
for (const field of ['norm_hash', 'numbers', 'concept', 'kind']) wb.createPayloadIndex(field, 'keyword')
const qv = await embedTexts(questions.map((q) => q.text))
wb.upsert(
  questions.map((q, i) => ({
    id: i,
    vector: qv[i],
    payload: { ...q, norm_hash: normHash(q.text), numbers: numbersKey(q.text) },
  })),
)

// 1) Exact duplicates need no vectors: facet over the normalized-text hash.
const dupGroups = wb.facet({ key: 'norm_hash', limit: 1000 }).filter((h) => h.count > 1)
const expectedDup = questions.filter((q) => q.kind === 'exact_duplicate').length
console.log(`  facet(norm_hash): ${dupGroups.length} values occur more than once`)
check('facet finds every exact-duplicate group, and nothing else', dupGroups.length === expectedDup && dupGroups.every((g) => g.count === 2))
{
  const g = dupGroups[0]
  const members = wb.scroll({ filter: { must: [{ key: 'norm_hash', match: { value: g.value as string } }] } }).points
  console.log(`  e.g. ${members.map((m) => JSON.stringify(m.payload!.text)).join('  ==  ')}`)
}

// 2) Near-duplicates: for every original, which other questions score above the QQP threshold?
interface Outcome {
  q: Question
  candidate: Question
  score: number
}
async function candidates(threshold: number, extra?: (q: Question, i: number) => Filter): Promise<Outcome[]> {
  const out: Outcome[] = []
  for (const [i, q] of questions.entries()) {
    if (q.kind !== 'original') continue
    const filter: Filter = { must_not: [{ has_id: [i] }, { key: 'kind', match: { value: 'exact_duplicate' } }], ...(extra?.(q, i) ?? {}) }
    for (const h of await wb.searchAsync(qv[i], { limit: 5, score_threshold: threshold, filter, with_payload: false })) {
      out.push({ q, candidate: questions[h.id as number], score: h.score })
    }
  }
  return out
}
const sameGroup = (o: Outcome) => o.candidate.group === o.q.group
function summarize(label: string, outs: Outcome[]) {
  const para = outs.filter((o) => sameGroup(o) && o.candidate.kind === 'paraphrase').length
  const look = outs.filter((o) => sameGroup(o) && o.candidate.kind === 'lookalike')
  return {
    setup: label,
    'paraphrases found (of 30)': para,
    'look-alikes flagged': look.length,
    'number look-alikes': look.filter((o) => o.candidate.lookalike_kind === 'numbers').length,
    'wording look-alikes': look.filter((o) => o.candidate.lookalike_kind !== 'numbers').length,
    'other false hits': outs.filter((o) => !sameGroup(o)).length,
  }
}

const T = best.threshold
const plain = await candidates(T)
const scores = (kind: Question['kind']) =>
  questions.flatMap((q, i) => {
    if (q.kind !== kind) return []
    const orig = questions.findIndex((o) => o.group === q.group && o.kind === 'original')
    return [dot(qv[i], qv[orig])]
  })
table([
  { pair: 'original ↔ paraphrase (same answer)', 'mean cosine': mean(scores('paraphrase')), min: Math.min(...scores('paraphrase')) },
  { pair: 'original ↔ look-alike (different answer)', 'mean cosine': mean(scores('lookalike')), min: Math.min(...scores('lookalike')) },
])
const numbersFiltered = await candidates(T, (q) => ({ must: [{ key: 'numbers', match: { value: numbersKey(q.text) } }] }))
const rows = [summarize(`cosine ≥ ${T}`, plain), summarize(`cosine ≥ ${T} + same numbers`, numbersFiltered)]
table(rows)

const lookBefore = plain.filter((o) => sameGroup(o) && o.candidate.kind === 'lookalike')
const numLookBefore = lookBefore.filter((o) => o.candidate.lookalike_kind === 'numbers').length
const numLookAfter = rows[1]['number look-alikes']
check('embeddings flag most look-alikes as duplicates (the limitation)', lookBefore.length >= 15, `${lookBefore.length}/30 above threshold`)
check('the numbers filter removes ≥80% of number look-alike false duplicates', numLookBefore > 0 && (numLookBefore - numLookAfter) / numLookBefore >= 0.8, `${numLookBefore} → ${numLookAfter}`)
check('the numbers filter keeps ≥80% of true paraphrases', rows[1]['paraphrases found (of 30)'] >= 0.8 * rows[0]['paraphrases found (of 30)'])
console.log(`  wording look-alikes (largest→smallest, absorb→release, boil→freeze) remain: ${rows[1]['wording look-alikes']}.`)
console.log('  No payload filter can tell those apart: it needs answer-aware checking (see FINDINGS.md).')
const worst = lookBefore.sort((a, b) => b.score - a.score).slice(0, 3)
for (const w of worst) console.log(`    ${w.score.toFixed(3)}  "${w.q.text}" (${w.q.answer})  vs  "${w.candidate.text}" (${w.candidate.answer})`)

// 3) Concepts: facet counts and concept-scoped search.
heading('Concepts')
const concepts = wb.facet({ key: 'concept', limit: 50 })
table(concepts.slice(0, 5).map((c) => ({ concept: c.value, questions: c.count })))
const conceptCount = new Set(questions.map((q) => q.concept)).size
check('facet returns every concept with the right count', concepts.length === conceptCount && concepts.every((c) => c.count === questions.filter((q) => q.concept === c.value).length))
const [query] = await embedTexts(['how big is a shape'])
const inArea = wb.search(query, { limit: 10, filter: { must: [{ key: 'concept', match: { value: 'area' } }] } })
check('concept filter returns only that concept', inArea.length > 0 && inArea.every((h) => h.payload!.concept === 'area'), `${inArea.length} hits`)

saveResults(PROJECT, {
  qqp: { pairs: N_PAIRS, best_threshold: best.threshold, precision: best.precision, recall: best.recall, f1: best.f1 },
  workbook: rows,
})
db.close()
finish(PROJECT)
