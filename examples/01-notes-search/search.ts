// Index a notes folder and search it.
//   npm run p1:search -- "how to reduce server response time" [--notes ./my-notes] [--keyword]
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { openDb } from '../shared/db.ts'
import { args, flag } from '../shared/paths.ts'
import { table } from '../shared/report.ts'
import { createCollection, keywordSearch, semanticSearch, syncNotes } from './lib.ts'

const HERE = path.dirname(fileURLToPath(import.meta.url))
const notesDir = path.resolve(flag('notes') ?? path.join(HERE, 'notes'))
const query = args.find((a, i) => !a.startsWith('--') && args[i - 1] !== '--notes') ?? 'how to reduce server response time'

const db = openDb(`01-notes-search-${path.basename(notesDir)}`)
const col = createCollection(db)
const stats = await syncNotes(col, notesDir)
console.log(`indexed ${notesDir}: ${JSON.stringify(stats)}\n`)

const hits = args.includes('--keyword') ? keywordSearch(col, query, 8) : await semanticSearch(col, query, 8)
console.log(`"${query}"`)
table(hits.map((h, i) => ({ '#': i + 1, note: h.path, section: h.heading, score: h.score })))
db.close()
