// Console output + assertions. A failed `check` marks the run as failed (exit code 1) but
// lets the experiment continue, so one run reports every problem.
import fs from 'node:fs'
import path from 'node:path'
import { DATA_DIR } from './paths.ts'

let failures = 0

export function heading(text: string) {
  console.log(`\n\x1b[1m== ${text}\x1b[0m`)
}

export function check(name: string, ok: boolean, detail = '') {
  if (!ok) failures++
  console.log(`  ${ok ? '\x1b[32m✔' : '\x1b[31m✘'}\x1b[0m ${name}${detail ? `  (${detail})` : ''}`)
}

export function table(rows: Record<string, unknown>[]) {
  if (rows.length === 0) return
  const cols = Object.keys(rows[0])
  const fmt = (v: unknown) => (typeof v === 'number' ? (Number.isInteger(v) ? String(v) : v.toFixed(3)) : String(v))
  const widths = cols.map((c) => Math.max(c.length, ...rows.map((r) => fmt(r[c]).length)))
  const line = (vals: string[]) => '  ' + vals.map((v, i) => v.padEnd(widths[i])).join('  ')
  console.log(line(cols))
  console.log(line(widths.map((w) => '-'.repeat(w))))
  for (const r of rows) console.log(line(cols.map((c) => fmt(r[c]))))
}

export async function timed<T>(label: string, fn: () => Promise<T> | T): Promise<T> {
  const t = performance.now()
  const result = await fn()
  console.log(`  ${label}: ${((performance.now() - t) / 1000).toFixed(2)} s`)
  return result
}

/** Save metrics for the e2e summary and README tables. */
export function saveResults(project: string, results: Record<string, unknown>) {
  const dir = path.join(DATA_DIR, 'results')
  fs.mkdirSync(dir, { recursive: true })
  fs.writeFileSync(path.join(dir, `${project}.json`), JSON.stringify({ project, failures, ...results }, null, 2))
}

/** Call at the end of an experiment. */
export function finish(project: string) {
  heading(`${project}: ${failures === 0 ? 'all checks passed' : `${failures} check(s) failed`}`)
  if (failures > 0) process.exitCode = 1
}
