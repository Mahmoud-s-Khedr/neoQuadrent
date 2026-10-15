// Loads each model once and reports dimensions + throughput on this machine.
import { embedClipTexts, embedCode, embedImages, embedPassages, embedQuery, rerankScores } from './embed.ts'
import { RawImage } from '@huggingface/transformers'

async function time<T>(label: string, fn: () => Promise<T>): Promise<T> {
  const t = performance.now()
  const r = await fn()
  console.log(`${label.padEnd(40)} ${(performance.now() - t).toFixed(0).padStart(7)} ms`)
  return r
}

const texts = Array.from({ length: 64 }, (_, i) => `Caching database query results reduces latency, note ${i}.`)
const docs = await time('bge-small: load + 64 passages', () => embedPassages(texts))
await time('bge-small: 64 passages (warm)', () => embedPassages(texts))
const q = await embedQuery('how to reduce server response time')
console.log(`  dim=${docs[0].length} |q|=${Math.hypot(...q).toFixed(3)}`)

const code = await time('jina-code: load + 16 snippets', () => embedCode(new Array(16).fill('fn verify_jwt(token: &str) -> bool { decode(token).is_ok() }')))
await time('jina-code: 16 snippets (warm)', () => embedCode(new Array(16).fill('function validate(token) { return jwt.verify(token, secret) }')))
console.log(`  dim=${code[0].length}`)

const img = new RawImage(new Uint8ClampedArray(160 * 160 * 3).fill(128), 160, 160, 3)
const ie = await time('clip: load + 16 images', () => embedImages(new Array(16).fill(img)))
await time('clip: 16 images (warm)', () => embedImages(new Array(16).fill(img)))
const te = await embedClipTexts(['a photo of a dog'])
console.log(`  image dim=${ie[0].length} text dim=${te[0].length} |img|=${Math.hypot(...ie[0]).toFixed(2)}`)

const s = await time('reranker: load + 16 pairs', () =>
  rerankScores('what reduces latency?', ['Caching reduces latency.', 'Bananas are yellow.', ...new Array(14).fill('filler')]),
)
console.log(`  relevant=${s[0].toFixed(2)} irrelevant=${s[1].toFixed(2)}`)
