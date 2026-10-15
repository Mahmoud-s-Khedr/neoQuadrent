// Project 2 experiment: similar-image search, near-duplicate detection, text→image search,
// Cosine vs Dot on raw CLIP vectors, 512 vs 256 dimensions, and HNSW recall vs exact search,
// on Imagenette (10 ImageNet classes) plus generated near-duplicates with known sources.
import path from 'node:path'
import type { Collection, Database, Distance, ScoredPoint } from 'neoquadrant'
import { chunks, openDb, waitForIndex } from '../shared/db.ts'
import { embedClipTexts } from '../shared/embed.ts'
import { mean, range } from '../shared/metrics.ts'
import { DATASETS_DIR, QUICK } from '../shared/paths.ts'
import { check, finish, heading, saveResults, table, timed } from '../shared/report.ts'
import { fitPca } from './pca.ts'
import { CLASSES, type Image, loadEmbeddings, loadImages } from './prepare.ts'

const PROJECT = '02-image-similarity'
const DIM = 512

const images = await timed('load images', () => loadImages())
const raw = await timed('CLIP image embeddings (cached after the first run)', () => loadEmbeddings(images))
const originals = images.filter((im) => !im.is_augmented)
const dups = images.filter((im) => im.is_augmented)
/** Sources that have a near-duplicate in the collection. */
const hasDup = new Set(dups.map((d) => d.source_id))
const AUGS = [...new Set(dups.map((d) => d.augmentation!))]
console.log(`  ${images.length} images: ${originals.length} originals, ${dups.length} near-duplicates (${AUGS.length} augmentations)${QUICK ? ' (quick mode)' : ''}`)

const norm = (v: number[]) => Math.hypot(...v)
const dot = (a: number[], b: number[]) => a.reduce((s, x, i) => s + x * b[i], 0)
const cosine = (a: number[], b: number[]) => dot(a, b) / (norm(a) * norm(b))

const db: Database = openDb(PROJECT, { fresh: true })

/** A collection with the project's schema. Tiny thresholds so even the quick dataset gets HNSW. */
function createImageCollection(name: string, size: number, distance: Distance): Collection {
  const col = db.createCollection(name, {
    vectors: { image: { size, distance } },
    // full_scan_threshold (KB): below it a segment is searched exactly even when it has an HNSW
    // index. Our 1–7 MB of vectors is under the 10 MB default, which would hide the index.
    hnsw_config: { m: 16, ef_construct: 100, full_scan_threshold: 10 },
    // indexing_threshold (KB): segments smaller than this never get an HNSW index.
    optimizers_config: { indexing_threshold: 100 },
  })
  col.createPayloadIndex('label', 'keyword')
  col.createPayloadIndex('is_augmented', 'bool')
  col.createPayloadIndex('source_id', 'integer')
  return col
}

async function fill(col: Collection, vectors: number[][]) {
  for (const batch of chunks(images, 256)) {
    await col.upsertAsync(
      batch.map((im) => ({
        id: im.id,
        vector: { image: vectors[im.id] },
        payload: {
          label: im.label,
          is_augmented: im.is_augmented,
          source_id: im.source_id,
          augmentation: im.augmentation ?? 'none',
          file: path.relative(DATASETS_DIR, im.file),
        },
      })),
    )
  }
}

// ------------------------------------------------------------------ storage
heading('Storage: named vector `image` (512, Cosine), keyword/bool/integer payload indexes')
const col = createImageCollection('images', DIM, 'Cosine')
await timed('upsert', () => fill(col, raw))
const info = await timed('wait for the background HNSW build', () => waitForIndex(col, 0.9))
console.log(`  points=${info.points_count} indexed=${info.indexed_vectors_count} segments=${info.segments_count}`)
check('every image stored', info.points_count === images.length)
check('background optimizer built HNSW for ≥90% of vectors', info.indexed_vectors_count >= 0.9 * images.length, `${info.indexed_vectors_count}/${images.length}`)
check('payload indexes exist', ['label', 'is_augmented', 'source_id'].every((f) => f in info.payload_schema), Object.keys(info.payload_schema).join(','))
const labelFacet = col.facet({ key: 'label', limit: 20 })
check('facet(label) counts match the dataset', labelFacet.length === 10 && labelFacet.every((f) => f.count === images.filter((im) => im.label === f.value).length))
check('bool filter counts the near-duplicates', col.count({ filter: { must: [{ key: 'is_augmented', match: { value: true } }] } }) === dups.length)
{
  // Qdrant's Cosine normalizes on insert: the stored vector is unit length.
  const [stored] = col.retrieve([originals[0].id], { with_vector: true })
  const v = (stored.vector as Record<string, number[]>).image
  console.log(`  raw |v| = ${norm(raw[originals[0].id]).toFixed(3)}, stored |v| = ${norm(v).toFixed(4)}`)
  check('Cosine collection stores unit-normalized vectors', Math.abs(norm(v) - 1) < 1e-3)
}

// ------------------------------------------------------------------ similar images
/** Top-k for each original, excluding the query and its own near-duplicates (same source_id). */
async function similar(c: Collection, vectors: number[][], queries: Image[], k = 10, exact = false): Promise<ScoredPoint[][]> {
  return Promise.all(
    queries.map((q) =>
      c.searchAsync(vectors[q.id], {
        using: 'image',
        limit: k,
        filter: { must_not: [{ key: 'source_id', match: { value: q.id } }] },
        params: exact ? { exact: true } : undefined,
        with_payload: ['label', 'source_id'],
      }),
    ),
  )
}
const labelPrecision = (queries: Image[], results: ScoredPoint[][], k = 10) =>
  queries.map((q, i) => results[i].slice(0, k).filter((h) => h.payload!.label === q.label).length / k)

heading('Similar images: label precision@10 (query and its own near-duplicates excluded)')
const simResults = await timed(`${originals.length} queries`, () => similar(col, raw, originals))
const simP10 = labelPrecision(originals, simResults)
check('no result is the query or one of its near-duplicates', simResults.every((r, i) => r.every((h) => h.payload!.source_id !== originals[i].id)))
const perClass = Object.values(CLASSES).map((label) => ({
  label,
  'P@10': mean(simP10.filter((_, i) => originals[i].label === label)),
  'worst query': Math.min(...simP10.filter((_, i) => originals[i].label === label)),
}))
table(perClass)
const simP10Mean = mean(simP10)
console.log(`  mean P@10 = ${simP10Mean.toFixed(3)}`)
check('mean label precision@10 ≥ 0.9', simP10Mean >= 0.9, simP10Mean.toFixed(3))

// ------------------------------------------------------------------ near-duplicates
heading('Near-duplicates: is the nearest other image the same photo?')
interface Top1 {
  q: Image
  hit: Image
  score: number
  /** The hit is the query's source or its near-duplicate. */
  correct: boolean
  /** The query has a duplicate in the collection (it is one, or it has one). */
  hasDuplicate: boolean
}
async function nearest(c: Collection, vectors: number[][], exact = false): Promise<Top1[]> {
  const hits = await Promise.all(
    images.map((q) => c.searchAsync(vectors[q.id], { using: 'image', limit: 1, filter: { must_not: [{ has_id: [q.id] }] }, params: exact ? { exact: true } : undefined, with_payload: false })),
  )
  return images.map((q, i) => {
    const hit = images[hits[i][0].id as number]
    return { q, hit, score: hits[i][0].score, correct: hit.source_id === q.source_id, hasDuplicate: q.is_augmented || hasDup.has(q.id) }
  })
}
const top1 = await timed(`${images.length} top-1 queries`, () => nearest(col, raw))
const dupRecall1 = (t: Top1[]) => mean(t.filter((x) => x.q.is_augmented).map((x) => (x.correct ? 1 : 0)))
const recall1 = dupRecall1(top1)
console.log(`  recall@1 of the source image over ${dups.length} near-duplicates: ${recall1.toFixed(3)}`)
check('near-duplicate recall@1 ≥ 0.9', recall1 >= 0.9, recall1.toFixed(3))

/** "Flag as duplicate if the nearest other image scores ≥ t", scored against the known sources. */
function sweep(t1: Top1[], thresholds: number[]) {
  const positives = t1.filter((x) => x.hasDuplicate).length
  return thresholds.map((threshold) => {
    const flagged = t1.filter((x) => x.score >= threshold)
    const tp = flagged.filter((x) => x.correct).length
    const fp = flagged.filter((x) => !x.correct)
    const precision = flagged.length === 0 ? 1 : tp / flagged.length
    const recall = tp / positives
    return {
      threshold,
      precision,
      recall,
      f1: precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall),
      tp,
      'fp same class': fp.filter((x) => x.hit.label === x.q.label).length,
      'fp other class': fp.filter((x) => x.hit.label !== x.q.label).length,
    }
  })
}
const sw = sweep(top1, range(0.8, 0.99, 0.01))
table(sw.filter((_, i) => i % 2 === 0))
const best = sw.reduce((a, b) => (b.f1 > a.f1 ? b : a))
console.log(`  best threshold ${best.threshold}: precision ${best.precision.toFixed(3)}, recall ${best.recall.toFixed(3)}, F1 ${best.f1.toFixed(3)}`)
check('best-threshold F1 ≥ 0.8', best.f1 >= 0.8, best.f1.toFixed(3))
const T = best.threshold

heading(`Per augmentation type at threshold ${T}`)
function row(name: string, group: Top1[]) {
  const withDup = group.filter((x) => x.hasDuplicate)
  const flagged = group.filter((x) => x.score >= T)
  const fp = flagged.filter((x) => !x.correct)
  return {
    queries: name,
    n: group.length,
    'recall@1': withDup.length ? mean(withDup.map((x) => (x.correct ? 1 : 0))) : '-',
    'cos to source': withDup.length && name !== 'original (has duplicate)' ? mean(withDup.map((x) => cosine(raw[x.q.id], raw[x.q.source_id]))) : '-',
    found: flagged.filter((x) => x.correct).length,
    missed: withDup.filter((x) => !(x.correct && x.score >= T)).length,
    'fp same class': fp.filter((x) => x.hit.label === x.q.label).length,
    'fp other class': fp.filter((x) => x.hit.label !== x.q.label).length,
  }
}
const augRows = [
  ...AUGS.map((a) => row(a, top1.filter((x) => x.q.augmentation === a))),
  row('original (has duplicate)', top1.filter((x) => !x.q.is_augmented && hasDup.has(x.q.id))),
  row('original (no duplicate)', top1.filter((x) => !x.q.is_augmented && !hasDup.has(x.q.id))),
]
table(augRows)
const weakest = augRows.slice(0, AUGS.length).reduce((a, b) => ((b['recall@1'] as number) < (a['recall@1'] as number) ? b : a))
console.log(`  hardest augmentation: ${weakest.queries} (recall@1 ${(weakest['recall@1'] as number).toFixed(3)})`)
const fpSame = best['fp same class']
const fpOther = best['fp other class']
console.log(`  hypothesis "false duplicates are mostly same-class look-alikes": ${fpSame >= fpOther ? 'held' : 'did NOT hold'} (${fpSame} same-class vs ${fpOther} other-class)`)
{
  const examples = top1.filter((x) => x.score >= T && !x.correct).sort((a, b) => b.score - a.score).slice(0, 3)
  for (const e of examples) console.log(`    false duplicate ${e.score.toFixed(3)}: ${path.basename(e.q.file)} (${e.q.label}) ↔ ${path.basename(e.hit.file)} (${e.hit.label})`)
}

// ------------------------------------------------------------------ text → image
heading('Text → image search with CLIP text embeddings (originals only, via the bool index)')
const prompts: { query: string; label: string; kind: string }[] = [
  ...Object.values(CLASSES).map((label) => ({ query: `a photo of a ${label}`, label, kind: 'class prompt' })),
  { query: 'a fisherman holding a large fish', label: 'tench', kind: 'description' },
  { query: 'a brass musical instrument', label: 'french horn', kind: 'description' },
  { query: 'a vehicle that collects rubbish', label: 'garbage truck', kind: 'description' },
  { query: 'a building with a steeple', label: 'church', kind: 'description' },
  { query: 'filling up the car with fuel', label: 'gas pump', kind: 'description' },
  { query: 'someone gliding down from the sky', label: 'parachute', kind: 'description' },
  { query: 'a tool for cutting down trees', label: 'chain saw', kind: 'description' },
  { query: 'a dog with floppy ears in a field', label: 'english springer', kind: 'description' },
]
const textVecs = await timed(`embed ${prompts.length} CLIP text queries`, () => embedClipTexts(prompts.map((p) => p.query)))
async function textSearch(c: Collection, vecs: number[][], k = 10) {
  return Promise.all(vecs.map((v) => c.searchAsync(v, { using: 'image', limit: k, filter: { must: [{ key: 'is_augmented', match: { value: false } }] }, with_payload: ['label', 'is_augmented'] })))
}
const textHits = await textSearch(col, textVecs)
const textP10 = prompts.map((p, i) => textHits[i].filter((h) => h.payload!.label === p.label).length / 10)
table(prompts.map((p, i) => ({ kind: p.kind, query: p.query, 'P@10': textP10[i], 'top-1 score': textHits[i][0]?.score ?? 0 })))
const textSummary = Object.fromEntries(['class prompt', 'description'].map((k) => [k, mean(textP10.filter((_, i) => prompts[i].kind === k))]))
check('bool filter: text results are all originals', textHits.every((r) => r.every((h) => h.payload!.is_augmented === false)))
check('class prompts: mean P@10 ≥ 0.9', textSummary['class prompt'] >= 0.9, textSummary['class prompt'].toFixed(3))
console.log(`  descriptions (no class name): mean P@10 = ${textSummary.description.toFixed(3)}`)
const textTop1 = mean(textHits.map((r) => r[0]?.score ?? 0))
const imageTop1 = mean(simResults.map((r) => r[0]?.score ?? 0))
console.log(`  modality gap: mean top-1 cosine is ${textTop1.toFixed(3)} for text queries vs ${imageTop1.toFixed(3)} for image queries,`)
console.log('  so a similarity threshold tuned on image↔image scores cannot be reused for text↔image.')
check('text↔image scores sit far below image↔image scores (modality gap)', textTop1 < imageTop1 - 0.2)

// ------------------------------------------------------------------ distance functions
heading('Distance: Cosine vs Dot on raw (unnormalized) CLIP vectors, exact search')
const norms = images.map((im) => norm(raw[im.id]))
console.log(`  |v| over all images: min ${Math.min(...norms).toFixed(2)}, mean ${mean(norms).toFixed(2)}, max ${Math.max(...norms).toFixed(2)}`)
const dotCol = createImageCollection('images-dot', DIM, 'Dot')
await fill(dotCol, raw)
{
  const q = originals[1]
  const [h] = await dotCol.searchAsync(raw[q.id], { using: 'image', limit: 1, filter: { must_not: [{ has_id: [q.id] }] } })
  check('Dot scores are raw dot products of the unnormalized vectors', Math.abs(h.score - dot(raw[q.id], raw[h.id as number])) < 1e-2 * Math.abs(h.score), `${h.score.toFixed(3)}`)
}
const [cosSim, dotSim] = [await similar(col, raw, originals, 10, true), await similar(dotCol, raw, originals, 10, true)]
const [cosTop1, dotTop1] = [await nearest(col, raw, true), await nearest(dotCol, raw, true)]
const overlap = mean(cosSim.map((r, i) => r.filter((h) => dotSim[i].some((d) => d.id === h.id)).length / 10))
/** How often the most-returned image shows up across all top-10 lists ("hubness"). */
function hub(results: ScoredPoint[][]) {
  const counts = new Map<number, number>()
  for (const r of results) for (const h of r) counts.set(h.id as number, (counts.get(h.id as number) ?? 0) + 1)
  const [id, n] = [...counts.entries()].reduce((a, b) => (b[1] > a[1] ? b : a))
  return { n, norm: norm(raw[id]) }
}
const meanResultNorm = (results: ScoredPoint[][]) => mean(results.flatMap((r) => r.map((h) => norm(raw[h.id as number]))))
const distanceRows = [
  { distance: 'Cosine', 'P@10': mean(labelPrecision(originals, cosSim)), 'dup recall@1': dupRecall1(cosTop1), 'mean |v| of results': meanResultNorm(cosSim), 'top hub (lists)': hub(cosSim).n },
  { distance: 'Dot', 'P@10': mean(labelPrecision(originals, dotSim)), 'dup recall@1': dupRecall1(dotTop1), 'mean |v| of results': meanResultNorm(dotSim), 'top hub (lists)': hub(dotSim).n },
]
table(distanceRows)
console.log(`  top-10 overlap Cosine vs Dot: ${overlap.toFixed(3)}; mean |v| of all images: ${mean(norms).toFixed(2)}`)
console.log(`  hypothesis "Dot on raw vectors favours large-norm images": ${distanceRows[1]['mean |v| of results'] > distanceRows[0]['mean |v| of results'] ? 'held' : 'did NOT hold'}`)
console.log(`  hypothesis "Cosine ≥ Dot on label precision": ${distanceRows[0]['P@10'] >= distanceRows[1]['P@10'] ? 'held' : 'did NOT hold'}`)
db.dropCollection('images-dot')

// ------------------------------------------------------------------ dimensionality
heading('Dimensionality: 512 vs 256 (truncated, PCA), exact search')
const pca = await timed('fit PCA-256 on the originals', () => fitPca(originals.map((im) => raw[im.id]), 256))
console.log(`  PCA-256 keeps ${(pca.explained * 100).toFixed(1)}% of the variance`)
const variants: { name: string; dim: number; map: (v: number[]) => number[] }[] = [
  { name: '512 (full)', dim: 512, map: (v) => v },
  { name: '256 truncated', dim: 256, map: (v) => v.slice(0, 256) },
  { name: '256 PCA', dim: 256, map: (v) => pca.project(v) },
]
const dimRows = []
for (const v of variants) {
  const vecs = raw.map(v.map)
  const c = v.dim === 512 ? col : createImageCollection(`images-${v.dim}-${v.name.split(' ')[1]}`, v.dim, 'Cosine')
  if (c !== col) await fill(c, vecs)
  const t = performance.now()
  const sim = await similar(c, vecs, originals, 10, true)
  const ms = (performance.now() - t) / originals.length
  const t1 = await nearest(c, vecs, true)
  const txt = await textSearch(c, textVecs.map(v.map))
  dimRows.push({
    vectors: v.name,
    'bytes/vector': v.dim * 4,
    'P@10': mean(labelPrecision(originals, sim)),
    'dup recall@1': dupRecall1(t1),
    'text P@10': mean(prompts.map((p, i) => txt[i].filter((h) => h.payload!.label === p.label).length / 10)),
    'ms/query (concurrent)': ms,
  })
  if (c !== col) db.dropCollection(c.name)
}
table(dimRows)
check('PCA-256 keeps ≥95% of the 512-d label precision', dimRows[2]['P@10'] >= 0.95 * dimRows[0]['P@10'], `${dimRows[2]['P@10'].toFixed(3)} vs ${dimRows[0]['P@10'].toFixed(3)}`)
console.log(`  hypothesis "PCA-256 beats plain truncation (CLIP is not Matryoshka-trained)": ${dimRows[2]['P@10'] >= dimRows[1]['P@10'] ? 'held' : 'did NOT hold'}`)

// ------------------------------------------------------------------ index quality
heading('Index quality: HNSW recall@10 vs exact search')
const probes = originals.filter((_, i) => i % Math.max(1, Math.floor(originals.length / 200)) === 0).slice(0, 200)
async function hnswRecall(hnsw_ef?: number, filtered = false) {
  const recalls = await Promise.all(
    probes.map(async (q) => {
      const filter = filtered ? { must: [{ key: 'label', match: { value: q.label === 'church' ? 'tench' : 'church' } }] } : undefined
      const opts = { using: 'image', limit: 10, filter, with_payload: false }
      const [approx, exact] = await Promise.all([
        col.searchAsync(raw[q.id], { ...opts, params: hnsw_ef ? { hnsw_ef } : undefined }),
        col.searchAsync(raw[q.id], { ...opts, params: { exact: true } }),
      ])
      const truth = new Set(exact.map((h) => h.id))
      return approx.filter((h) => truth.has(h.id)).length / Math.max(1, truth.size)
    }),
  )
  return mean(recalls)
}
const indexRows = []
for (const ef of [undefined, 16, 64, 128]) indexRows.push({ hnsw_ef: ef ?? 'default', 'recall@10': await hnswRecall(ef), 'recall@10 (label filter, other class)': await hnswRecall(ef, true) })
table(indexRows)
check('HNSW recall@10 vs exact ≥ 0.95 at the default hnsw_ef', (indexRows[0]['recall@10'] as number) >= 0.95, (indexRows[0]['recall@10'] as number).toFixed(3))

saveResults(PROJECT, {
  images: images.length,
  near_duplicates: dups.length,
  index: { indexed_vectors: info.indexed_vectors_count, segments: info.segments_count, recall: indexRows },
  similar: { p_at_10: simP10Mean, per_class: perClass },
  near_dup: { recall_at_1: recall1, best, per_augmentation: augRows },
  text_to_image: { ...textSummary, top1_text: textTop1, top1_image: imageTop1 },
  distance: { rows: distanceRows, overlap_top10: overlap },
  dimensions: { pca_variance_kept: pca.explained, rows: dimRows },
})
db.close()
finish(PROJECT)
