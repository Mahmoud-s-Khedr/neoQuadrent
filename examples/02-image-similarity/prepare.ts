// Imagenette-160 (10 ImageNet classes, 160 px) + generated near-duplicates with known sources.
import { execFileSync } from 'node:child_process'
import crypto from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import sharp, { type Sharp } from 'sharp'
import { MODELS, TRANSFORMERS_VERSION, embedImages } from '../shared/embed.ts'
import { DATASETS_DIR, EXAMPLES_DIR, QUICK } from '../shared/paths.ts'

const URL = 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz'
const DIR = path.join(DATASETS_DIR, 'p2')
const ROOT = path.join(DIR, 'imagenette2-160')
export const CLASSES: Record<string, string> = {
  n01440764: 'tench',
  n02102040: 'english springer',
  n02979186: 'cassette player',
  n03000684: 'chain saw',
  n03028079: 'church',
  n03394916: 'french horn',
  n03417042: 'garbage truck',
  n03425413: 'gas pump',
  n03445777: 'golf ball',
  n03888257: 'parachute',
}

/**
 * Version of this file's data preparation (class list, sampling, augmentations): derived files
 * (manifest, augmented JPEGs, vector cache) are keyed by it, so changing an augmentation
 * regenerates them instead of silently reusing the old ones.
 */
const PREP_VERSION = crypto.createHash('sha1').update(fs.readFileSync(fileURLToPath(import.meta.url))).digest('hex').slice(0, 8)

export interface Image {
  id: number
  file: string
  label: string
  /** For a near-duplicate: the id of the image it was made from. Originals point to themselves. */
  source_id: number
  is_augmented: boolean
  augmentation?: string
}

async function ensureDataset() {
  if (fs.existsSync(ROOT)) return
  fs.mkdirSync(DIR, { recursive: true })
  const tgz = path.join(DIR, 'imagenette2-160.tgz')
  if (!fs.existsSync(tgz)) {
    console.log(`downloading ${URL} (~99 MB)`)
    const res = await fetch(URL)
    if (!res.ok) throw new Error(`download failed: ${res.status}`)
    fs.writeFileSync(tgz + '.part', Buffer.from(await res.arrayBuffer()))
    fs.renameSync(tgz + '.part', tgz)
  }
  execFileSync('tar', ['-xzf', tgz, '-C', DIR])
}

const AUGMENTATIONS: [string, (img: Sharp, w: number, h: number) => Sharp][] = [
  ['crop 85% + resize', (img, w, h) => img.extract({ left: Math.round(w * 0.075), top: Math.round(h * 0.075), width: Math.round(w * 0.85), height: Math.round(h * 0.85) }).resize(w, h)],
  ['jpeg quality 25', (img) => img.jpeg({ quality: 25 })],
  ['horizontal flip', (img) => img.flop()],
  ['brightness +30%', (img) => img.modulate({ brightness: 1.3 })],
  ['downscale ×0.5', (img, w) => img.resize(Math.round(w / 2))],
  ['grayscale', (img) => img.grayscale()],
]

/** Originals (evenly across classes) + near-duplicates, cached as a manifest. */
export async function loadImages(): Promise<Image[]> {
  const nOriginals = QUICK ? 500 : 3000
  const nDups = QUICK ? 60 : 300
  const manifest = path.join(DIR, `manifest-${nOriginals}-${PREP_VERSION}.json`)
  if (fs.existsSync(manifest)) return JSON.parse(fs.readFileSync(manifest, 'utf8'))
  await ensureDataset()

  const images: Image[] = []
  const perClass = Math.ceil(nOriginals / 10)
  for (const wnid of Object.keys(CLASSES)) {
    const dir = path.join(ROOT, 'train', wnid)
    for (const f of fs.readdirSync(dir).sort().slice(0, perClass)) {
      images.push({ id: images.length, file: path.join(dir, f), label: CLASSES[wnid], source_id: images.length, is_augmented: false })
    }
  }
  images.length = Math.min(images.length, nOriginals)

  const augDir = path.join(DIR, `augmented-${nOriginals}-${PREP_VERSION}`)
  fs.mkdirSync(augDir, { recursive: true })
  const originals = images.length
  for (let k = 0; k < nDups; k++) {
    // Deterministic spread of sources across classes.
    const src = images[(k * 7919) % originals]
    const [name, fn] = AUGMENTATIONS[k % AUGMENTATIONS.length]
    const out = path.join(augDir, `${k}-${crypto.createHash('md5').update(src.file + name).digest('hex').slice(0, 8)}.jpg`)
    if (!fs.existsSync(out)) {
      const meta = await sharp(src.file).metadata()
      await fn(sharp(src.file), meta.width!, meta.height!).jpeg({ quality: name.startsWith('jpeg') ? 25 : 90 }).toFile(out)
    }
    images.push({ id: images.length, file: out, label: src.label, source_id: src.id, is_augmented: true, augmentation: name })
  }
  fs.writeFileSync(manifest, JSON.stringify(images))
  return images
}

/** CLIP embeddings for all images, cached on disk (raw, unnormalized, 512-d). */
export async function loadEmbeddings(images: Image[]): Promise<number[][]> {
  // Keyed by everything that shapes the vectors (model, embedding code, library version) and by
  // the ordered image list, so a change never silently reuses vectors from an older pipeline.
  const key = crypto
    .createHash('sha1')
    .update(JSON.stringify([MODELS.clip, TRANSFORMERS_VERSION, images.map((im) => im.file)]))
    .update(fs.readFileSync(path.join(EXAMPLES_DIR, 'shared', 'embed.ts')))
    .digest('hex')
    .slice(0, 12)
  const cache = path.join(DIR, `clip-${images.length}-${key}.f32`)
  if (fs.existsSync(cache)) {
    const flat = new Float32Array(fs.readFileSync(cache).buffer.slice(0))
    return images.map((_, i) => Array.from(flat.subarray(i * 512, (i + 1) * 512)))
  }
  const vectors: number[][] = []
  for (let i = 0; i < images.length; i += 64) {
    vectors.push(...(await embedImages(images.slice(i, i + 64).map((im) => im.file))))
    process.stdout.write(`\r  CLIP embeddings: ${vectors.length}/${images.length}`)
  }
  process.stdout.write('\n')
  fs.writeFileSync(cache, Buffer.from(Float32Array.from(vectors.flat()).buffer))
  return vectors
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const images = await loadImages()
  console.log(`${images.length} images (${images.filter((i) => i.is_augmented).length} near-duplicates)`)
}
