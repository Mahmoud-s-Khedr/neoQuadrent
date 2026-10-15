// Local CPU embedding models (transformers.js / onnxruntime-node). Models load lazily on first
// use and are cached under examples/.data/models.
import {
  AutoModelForSequenceClassification,
  AutoProcessor,
  AutoTokenizer,
  CLIPTextModelWithProjection,
  CLIPVisionModelWithProjection,
  RawImage,
  env,
  pipeline,
  type FeatureExtractionPipeline,
  type Tensor,
} from '@huggingface/transformers'
import { MODELS_DIR } from './paths.ts'

env.cacheDir = MODELS_DIR

export const MODELS = {
  text: 'Xenova/bge-small-en-v1.5', // 384-d, CLS pooling
  textLarge: 'mixedbread-ai/mxbai-embed-large-v1', // 1024-d, CLS pooling, 335M params
  code: 'jinaai/jina-embeddings-v2-base-code', // 768-d, mean pooling
  clip: 'Xenova/clip-vit-base-patch32', // 512-d image + text
  reranker: 'Xenova/ms-marco-MiniLM-L-6-v2', // cross-encoder
} as const

// bge-v1.5 recommends this prefix for short retrieval queries (not for passages).
const BGE_QUERY_PREFIX = 'Represent this sentence for searching relevant passages: '

const DTYPE = 'q8' as const

function once<T>(load: () => Promise<T>): () => Promise<T> {
  let p: Promise<T> | undefined
  return () => (p ??= load())
}

const textPipe = once(
  () => pipeline('feature-extraction', MODELS.text, { dtype: DTYPE }) as Promise<FeatureExtractionPipeline>,
)
const textLargePipe = once(
  () => pipeline('feature-extraction', MODELS.textLarge, { dtype: DTYPE }) as Promise<FeatureExtractionPipeline>,
)
const codePipe = once(
  () => pipeline('feature-extraction', MODELS.code, { dtype: DTYPE }) as Promise<FeatureExtractionPipeline>,
)

async function runBatched(
  pipe: FeatureExtractionPipeline,
  texts: string[],
  pooling: 'cls' | 'mean',
  batchSize: number,
): Promise<number[][]> {
  const out: number[][] = []
  for (let i = 0; i < texts.length; i += batchSize) {
    const batch = texts.slice(i, i + batchSize)
    const t = await pipe(batch, { pooling, normalize: true })
    out.push(...(t.tolist() as number[][]))
  }
  return out
}

/** Passage embeddings (bge-small, normalized, 384-d). */
export async function embedPassages(texts: string[], batchSize = 32): Promise<number[][]> {
  return runBatched(await textPipe(), texts, 'cls', batchSize)
}

/** Query embeddings (bge-small with its retrieval prefix). */
export async function embedQueries(texts: string[]): Promise<number[][]> {
  return runBatched(await textPipe(), texts.map((t) => BGE_QUERY_PREFIX + t), 'cls', 32)
}

export async function embedQuery(text: string): Promise<number[]> {
  return (await embedQueries([text]))[0]
}

/** Passage embeddings from the larger mxbai-embed-large model (1024-d). Slower; better semantics. */
export async function embedPassagesLarge(texts: string[], batchSize = 16): Promise<number[][]> {
  return runBatched(await textLargePipe(), texts, 'cls', batchSize)
}

/** mxbai uses the same retrieval prompt as bge for queries. */
export async function embedQueryLarge(text: string): Promise<number[]> {
  return (await runBatched(await textLargePipe(), [BGE_QUERY_PREFIX + text], 'cls', 1))[0]
}

/** Symmetric text embeddings without the query prefix (question-vs-question similarity). */
export const embedTexts = embedPassages

/** Code (or natural-language-about-code) embeddings, jina-v2-base-code, 768-d. */
export async function embedCode(texts: string[], batchSize = 8): Promise<number[][]> {
  return runBatched(await codePipe(), texts, 'mean', batchSize)
}

// ---------------------------------------------------------------- CLIP

const clip = once(async () => ({
  processor: await AutoProcessor.from_pretrained(MODELS.clip),
  tokenizer: await AutoTokenizer.from_pretrained(MODELS.clip),
  vision: await CLIPVisionModelWithProjection.from_pretrained(MODELS.clip, { dtype: DTYPE }),
  text: await CLIPTextModelWithProjection.from_pretrained(MODELS.clip, { dtype: DTYPE }),
}))

/** Raw (unnormalized) CLIP image embeddings, 512-d. Accepts file paths or RawImages. */
export async function embedImages(inputs: (string | RawImage)[], batchSize = 16): Promise<number[][]> {
  const { processor, vision } = await clip()
  const out: number[][] = []
  for (let i = 0; i < inputs.length; i += batchSize) {
    const images = await Promise.all(inputs.slice(i, i + batchSize).map((x) => (typeof x === 'string' ? RawImage.read(x) : x)))
    const pixels = await processor(images)
    const { image_embeds } = (await vision(pixels)) as { image_embeds: Tensor }
    out.push(...(image_embeds.tolist() as number[][]))
  }
  return out
}

/** Raw CLIP text embeddings in the same space as images. */
export async function embedClipTexts(texts: string[]): Promise<number[][]> {
  const { tokenizer, text } = await clip()
  const inputs = tokenizer(texts, { padding: true, truncation: true })
  const { text_embeds } = (await text(inputs)) as { text_embeds: Tensor }
  return text_embeds.tolist() as number[][]
}

// ---------------------------------------------------------------- reranker

const reranker = once(async () => ({
  tokenizer: await AutoTokenizer.from_pretrained(MODELS.reranker),
  model: await AutoModelForSequenceClassification.from_pretrained(MODELS.reranker, { dtype: DTYPE }),
}))

/** Cross-encoder relevance scores for (query, passage) pairs. Higher is more relevant. */
export async function rerankScores(query: string, passages: string[], batchSize = 16): Promise<number[]> {
  const { tokenizer, model } = await reranker()
  const scores: number[] = []
  for (let i = 0; i < passages.length; i += batchSize) {
    const batch = passages.slice(i, i + batchSize)
    const inputs = tokenizer(new Array(batch.length).fill(query), { text_pair: batch, padding: true, truncation: true })
    const { logits } = (await model(inputs)) as { logits: Tensor }
    scores.push(...(logits.tolist() as number[][]).map((row) => row[0]))
  }
  return scores
}

export function l2norm(v: number[]): number[] {
  const n = Math.hypot(...v)
  return n > 0 ? v.map((x) => x / n) : v
}
