// Run every project in --quick mode, then summarize .data/results. Exits non-zero if any
// project crashed or had a failed check.
//   npm run e2e                 all projects
//   npm run e2e -- --only p2,p4 a subset
import { spawnSync } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'
import { DATA_DIR, EXAMPLES_DIR, flag } from './paths.ts'
import { heading, table } from './report.ts'

const PROJECTS = [
  { id: 'p1', name: '01-notes-search' },
  { id: 'p2', name: '02-image-similarity' },
  { id: 'p3', name: '03-duplicate-questions' },
  { id: 'p4', name: '04-code-search' },
  { id: 'p5', name: '05-hybrid-docs' },
]
const only = flag('only')?.split(',')
const unknown = only?.filter((o) => !PROJECTS.some((p) => p.id === o || p.name === o)) ?? []
if (unknown.length > 0 || only?.length === 0) {
  // A typo must not turn into "0 projects run, all passed".
  console.error(`unknown project(s) in --only: ${unknown.join(', ') || '(empty)'}; expected ${PROJECTS.map((p) => p.id).join(', ')}`)
  process.exit(2)
}
const selected = PROJECTS.filter((p) => !only || only.includes(p.id) || only.includes(p.name))
const resultsDir = path.join(DATA_DIR, 'results')

type Json = Record<string, any>
const f = (x: unknown) => (typeof x === 'number' ? x.toFixed(3) : '?')

/** The few numbers worth reading per project. */
function highlights(id: string, r: Json): string {
  switch (id) {
    case 'p1':
      return `MRR small ${f(r.modes?.small?.mrr)} · large ${f(r.modes?.large?.mrr)} · BM25 ${f(r.modes?.keyword?.mrr)}`
    case 'p2':
      return `P@10 ${f(r.similar?.p_at_10)} · dup R@1 ${f(r.near_dup?.recall_at_1)} · dup F1 ${f(r.near_dup?.best?.f1)} · text P@10 ${f(r.text_to_image?.['class prompt'])} · HNSW R@10 ${f(r.index?.recall?.[0]?.['recall@10'])}`
    case 'p3':
      return `QQP F1 ${f(r.qqp?.f1)} at ${r.qqp?.best_threshold ?? '?'}`
    case 'p4':
      return `MRR nl ${f(r.summary?.nl?.mrr)} · code ${f(r.summary?.code?.mrr)} · RRF ${f(r.summary?.rrf?.mrr)} · sync re-embedded ${r.sync?.second?.chunks_embedded ?? '?'}/${r.sync?.second?.chunks_total ?? '?'}`
    case 'p5':
      return `MRR semantic ${f(r.summary?.semantic?.all?.mrr)} · BM25 ${f(r.summary?.keyword?.all?.mrr)} · hybrid ${f(r.summary?.hybrid?.all?.mrr)} · +rerank ${f(r.summary?.['hybrid+rerank']?.all?.mrr)}`
    default:
      return ''
  }
}

const rows = []
for (const p of selected) {
  heading(`e2e: ${p.id} ${p.name} --quick`)
  const resultFile = path.join(resultsDir, `${p.name}.json`)
  fs.rmSync(resultFile, { force: true }) // a crash must not look like an old success
  const t = performance.now()
  const run = spawnSync(process.execPath, ['--import', 'tsx', path.join(p.name, 'experiment.ts'), '--quick'], { cwd: EXAMPLES_DIR, stdio: 'inherit' })
  const seconds = (performance.now() - t) / 1000
  const result: Json | undefined = fs.existsSync(resultFile) ? JSON.parse(fs.readFileSync(resultFile, 'utf8')) : undefined
  const status =
    run.status === 0 && result?.failures === 0
      ? 'passed'
      : result
        ? `${result.failures} check(s) failed`
        : `crashed (exit ${run.status ?? run.signal})`
  rows.push({ project: `${p.id} ${p.name}`, status, seconds: Math.round(seconds), highlights: result ? highlights(p.id, result) : '-' })
}

heading('e2e summary (--quick)')
table(rows)
const failed = rows.filter((r) => r.status !== 'passed')
console.log(failed.length === 0 ? '\n  all projects passed' : `\n  ${failed.length} of ${rows.length} project(s) did not pass`)
process.exitCode = failed.length === 0 ? 0 : 1
