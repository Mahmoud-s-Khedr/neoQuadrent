import fs from 'node:fs'
import path from 'node:path'
import { Database, type DatabaseOptions } from 'neoquadrant'
import { DB_DIR } from './paths.ts'

export function dbPath(project: string): string {
  return path.join(DB_DIR, project)
}

/** Open the project's database. `fresh` deletes any previous one first. */
export function openDb(project: string, { fresh = false, ...options }: { fresh?: boolean } & DatabaseOptions = {}): Database {
  const dir = dbPath(project)
  if (fresh) fs.rmSync(dir, { recursive: true, force: true })
  fs.mkdirSync(dir, { recursive: true })
  return new Database(dir, options)
}

/** Wait until the background optimizer has indexed (HNSW) at least `fraction` of the points. */
export async function waitForIndex<I extends { points_count: number; indexed_vectors_count: number }>(col: { info(): I }, fraction = 0.9, timeoutMs = 300_000): Promise<I> {
  const start = Date.now()
  for (;;) {
    const info = col.info()
    if (info.points_count > 0 && info.indexed_vectors_count >= info.points_count * fraction) return info
    if (Date.now() - start > timeoutMs) return info
    await new Promise((r) => setTimeout(r, 250))
  }
}

/** Batches of `size` from `items`. */
export function* chunks<T>(items: T[], size: number): Generator<T[]> {
  for (let i = 0; i < items.length; i += size) yield items.slice(i, i + size)
}
