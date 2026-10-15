// Search the code index built by `npm run p4` (syncing it to the current commits first).
//   npm run p4:search -- "where is the JWT verified" [--mode nl|code|rrf] [--repo realworld] [--lang rust]
import type { Filter } from 'neoquadrant'
import { openDb } from '../shared/db.ts'
import { args, flag } from '../shared/paths.ts'
import { MODES, type Mode, createCollections, ensureRealworld, repos, search, syncRepo } from './lib.ts'

const query = args.filter((a, i) => !a.startsWith('--') && !args[i - 1]?.startsWith('--')).join(' ')
if (!query) {
  console.error('usage: npm run p4:search -- "query" [--mode nl|code|rrf] [--repo name] [--lang language]')
  process.exit(2)
}
const mode = (flag('mode') ?? 'rrf') as Mode
if (!MODES.includes(mode)) throw new Error(`--mode must be one of ${MODES.join(', ')}`)

ensureRealworld()
const db = openDb('04-code-search')
const { code, meta } = createCollections(db)
for (const repo of repos()) {
  const s = await syncRepo(code, meta, repo)
  if (s.mode !== 'unchanged') console.log(`synced ${repo.name}: ${s.mode}, ${s.chunks_embedded} chunks embedded`)
}

const must = [
  ...(flag('repo') ? [{ key: 'repo', match: { value: flag('repo')! } }] : []),
  ...(flag('lang') ? [{ key: 'language', match: { value: flag('lang')! } }] : []),
]
const filter: Filter | undefined = must.length ? { must } : undefined
for (const h of await search(code, mode, query, 10, filter)) {
  const p = h.payload!
  console.log(`${h.score.toFixed(3)}  ${p.repo}/${p.path}:${p.start_line}-${p.end_line}  ${p.kind} ${p.name}`)
}
db.close()
