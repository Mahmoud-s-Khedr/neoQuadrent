// Download GLUE QQP (Quora Question Pairs) validation split: 40k labeled question pairs.
import fs from 'node:fs'
import path from 'node:path'
import { asyncBufferFromFile, parquetReadObjects } from 'hyparquet'
import { DATASETS_DIR } from '../shared/paths.ts'

const URL = 'https://huggingface.co/datasets/nyu-mll/glue/resolve/main/qqp/validation-00000-of-00001.parquet'
export const QQP_FILE = path.join(DATASETS_DIR, 'qqp-validation.parquet')

export interface QqpPair {
  question1: string
  question2: string
  label: number // 1 = duplicate
}

export async function ensureQqp(): Promise<void> {
  if (fs.existsSync(QQP_FILE)) return
  fs.mkdirSync(DATASETS_DIR, { recursive: true })
  console.log(`downloading ${URL}`)
  const res = await fetch(URL)
  if (!res.ok) throw new Error(`download failed: ${res.status}`)
  fs.writeFileSync(QQP_FILE + '.part', Buffer.from(await res.arrayBuffer()))
  fs.renameSync(QQP_FILE + '.part', QQP_FILE)
}

/** A deterministic sample of `n` pairs (every k-th row). */
export async function loadQqp(n: number): Promise<QqpPair[]> {
  await ensureQqp()
  const rows = (await parquetReadObjects({ file: await asyncBufferFromFile(QQP_FILE), columns: ['question1', 'question2', 'label'] })) as unknown as QqpPair[]
  const step = Math.max(1, Math.floor(rows.length / n))
  // int64 parquet columns arrive as BigInt; normalize to number.
  return rows.filter((_, i) => i % step === 0).slice(0, n).map((r) => ({ ...r, label: Number(r.label) }))
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const pairs = await loadQqp(5)
  console.log(pairs)
}
