// Download the corpus (freely licensed), extract text, and split it into chunks.
//   Pro Git (CC BY-NC-SA 3.0) · HNSW paper (arXiv 1603.09320) · Node.js API docs (MIT)
import fs from 'node:fs'
import path from 'node:path'
import { extractText, getDocumentProxy } from 'unpdf'
import { DATASETS_DIR, QUICK } from '../shared/paths.ts'

const DIR = path.join(DATASETS_DIR, 'p5')
const NODE_DOCS = ['fs', 'errors', 'http', 'stream', 'path', 'events']
const SOURCES = {
  progit: 'https://github.com/progit/progit2/releases/download/2.1.450/progit.pdf',
  hnsw: 'https://arxiv.org/pdf/1603.09320',
  ...Object.fromEntries(NODE_DOCS.map((d) => [`node-${d}`, `https://raw.githubusercontent.com/nodejs/node/v22.21.1/doc/api/${d}.md`])),
} as Record<string, string>

export interface Chunk {
  id: number
  doc: string
  /** Page number (PDFs) or section heading (Markdown). */
  location: string
  text: string
}

async function download(name: string, url: string): Promise<string> {
  const ext = url.endsWith('.md') ? '.md' : '.pdf'
  const file = path.join(DIR, name + ext)
  if (!fs.existsSync(file)) {
    console.log(`downloading ${url}`)
    const res = await fetch(url)
    if (!res.ok) throw new Error(`${url}: ${res.status}`)
    fs.writeFileSync(file, Buffer.from(await res.arrayBuffer()))
  }
  return file
}

/** Sliding windows of `size` words with `overlap`, so a passage cut at a boundary still appears whole. */
function windows(text: string, size = 200, overlap = 40): string[] {
  const words = text.split(/\s+/).filter(Boolean)
  if (words.length <= size) return words.length ? [words.join(' ')] : []
  const out: string[] = []
  for (let i = 0; i < words.length; i += size - overlap) {
    out.push(words.slice(i, i + size).join(' '))
    if (i + size >= words.length) break
  }
  return out
}

async function pdfSections(file: string, maxPages: number): Promise<{ location: string; text: string }[]> {
  const pdf = await getDocumentProxy(new Uint8Array(fs.readFileSync(file)))
  const { text } = await extractText(pdf, { mergePages: false })
  return (text as string[]).slice(0, maxPages).map((t, i) => ({ location: `p.${i + 1}`, text: t.replace(/-\n(\w)/g, '$1').replace(/\s+/g, ' ') }))
}

function markdownSections(file: string): { location: string; text: string }[] {
  const md = fs.readFileSync(file, 'utf8').replace(/<!--[\s\S]*?-->/g, '')
  return md
    .split(/^(?=#{2,4} )/m)
    .map((s) => {
      const [head, ...body] = s.split('\n')
      const heading = head.replace(/^#+\s*/, '').trim()
      // Keep the heading in the text: it carries identifiers like `fs.createReadStream()`.
      return { location: heading, text: `${heading}. ${body.join(' ')}`.replace(/\s+/g, ' ') }
    })
    .filter((s) => s.text.length > 40)
}

export async function loadCorpus(): Promise<Chunk[]> {
  const cache = path.join(DIR, `chunks${QUICK ? '-quick' : ''}.json`)
  if (fs.existsSync(cache)) return JSON.parse(fs.readFileSync(cache, 'utf8'))
  fs.mkdirSync(DIR, { recursive: true })
  const chunks: Chunk[] = []
  for (const [name, url] of Object.entries(SOURCES)) {
    if (QUICK && !['progit', 'hnsw', 'node-fs', 'node-errors'].includes(name)) continue
    const file = await download(name, url)
    const sections = file.endsWith('.pdf') ? await pdfSections(file, QUICK && name === 'progit' ? 120 : Infinity) : markdownSections(file)
    for (const s of sections) for (const text of windows(s.text)) chunks.push({ id: chunks.length, doc: name, location: s.location, text })
  }
  fs.writeFileSync(cache, JSON.stringify(chunks))
  return chunks
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const chunks = await loadCorpus()
  const byDoc = new Map<string, number>()
  for (const c of chunks) byDoc.set(c.doc, (byDoc.get(c.doc) ?? 0) + 1)
  console.log(`${chunks.length} chunks`, Object.fromEntries(byDoc))
}
