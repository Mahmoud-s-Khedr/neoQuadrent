// Retrieval and classification metrics. `ranked` is a list of result keys in rank order;
// `relevant` is the set of keys that should be found.

export function recallAtK<T>(ranked: T[], relevant: Set<T>, k: number): number {
  if (relevant.size === 0) return 0
  const top = new Set(ranked.slice(0, k))
  let hit = 0
  for (const r of relevant) if (top.has(r)) hit++
  return hit / relevant.size
}

export function precisionAtK<T>(ranked: T[], relevant: Set<T>, k: number): number {
  const top = ranked.slice(0, k)
  return top.length === 0 ? 0 : top.filter((r) => relevant.has(r)).length / top.length
}

/** Reciprocal rank of the first relevant result (0 if none). */
export function reciprocalRank<T>(ranked: T[], relevant: Set<T>): number {
  const i = ranked.findIndex((r) => relevant.has(r))
  return i < 0 ? 0 : 1 / (i + 1)
}

/** nDCG@k with binary relevance. */
export function ndcgAtK<T>(ranked: T[], relevant: Set<T>, k: number): number {
  let dcg = 0
  ranked.slice(0, k).forEach((r, i) => {
    if (relevant.has(r)) dcg += 1 / Math.log2(i + 2)
  })
  let idcg = 0
  for (let i = 0; i < Math.min(k, relevant.size); i++) idcg += 1 / Math.log2(i + 2)
  return idcg === 0 ? 0 : dcg / idcg
}

export const mean = (xs: number[]) => (xs.length === 0 ? 0 : xs.reduce((a, b) => a + b, 0) / xs.length)

export interface Prf {
  threshold: number
  precision: number
  recall: number
  f1: number
  tp: number
  fp: number
  fn: number
}

/** Binary classification quality of `score >= threshold` against labels, for each threshold. */
export function thresholdSweep(scored: { score: number; positive: boolean }[], thresholds: number[]): Prf[] {
  const totalPos = scored.filter((s) => s.positive).length
  return thresholds.map((threshold) => {
    let tp = 0
    let fp = 0
    for (const s of scored) {
      if (s.score >= threshold) s.positive ? tp++ : fp++
    }
    const fn = totalPos - tp
    const precision = tp + fp === 0 ? 1 : tp / (tp + fp)
    const recall = totalPos === 0 ? 0 : tp / totalPos
    const f1 = precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall)
    return { threshold, precision, recall, f1, tp, fp, fn }
  })
}

export const range = (from: number, to: number, step: number) =>
  Array.from({ length: Math.round((to - from) / step) + 1 }, (_, i) => +(from + i * step).toFixed(4))
