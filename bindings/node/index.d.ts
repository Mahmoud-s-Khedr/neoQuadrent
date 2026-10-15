// Types follow Qdrant's REST API shapes: https://api.qdrant.tech/api-reference

export type PointId = number | string
export type Payload = Record<string, unknown>
export type DenseVector = number[] | Float32Array
export interface SparseVector {
  indices: number[] | Uint32Array
  values: number[] | Float32Array
}
export type MultiDenseVector = DenseVector[]
export type VectorInput = DenseVector | MultiDenseVector | Record<string, DenseVector | SparseVector | MultiDenseVector>

export interface PointStruct {
  id: PointId
  vector: VectorInput
  payload?: Payload | null
}

export type Distance = 'Cosine' | 'Dot' | 'Euclid' | 'Manhattan'

export interface VectorParams {
  size: number
  distance: Distance
  /** "cached" (RAM, default) | "cold" (mmap on disk) */
  memory?: 'cached' | 'cold' | 'pinned'
  datatype?: 'float32' | 'float16' | 'uint8'
  multivector_config?: { comparator: 'max_sim' }
  hnsw_config?: HnswConfig
  quantization_config?: QuantizationConfig
}

export interface HnswConfig {
  m: number
  ef_construct: number
  full_scan_threshold: number
  max_indexing_threads?: number
  on_disk?: boolean
  payload_m?: number
}

export type QuantizationConfig =
  | { scalar: { type: 'int8'; quantile?: number; always_ram?: boolean } }
  | { product: { compression: 'x4' | 'x8' | 'x16' | 'x32' | 'x64'; always_ram?: boolean } }
  | { binary: { always_ram?: boolean } }

export interface CreateCollection {
  vectors?: VectorParams | Record<string, VectorParams>
  sparse_vectors?: Record<string, { modifier?: 'idf' | 'none'; index?: Record<string, unknown> }>
  hnsw_config?: HnswConfig
  quantization_config?: QuantizationConfig
  optimizers_config?: {
    deleted_threshold?: number
    vacuum_min_vector_number?: number
    default_segment_number?: number
    max_segment_size?: number
    /** KB of vectors in a segment before it gets an HNSW index (smaller segments are searched exactly). */
    indexing_threshold?: number
  }
  on_disk_payload?: boolean
}

// ----- filters (Qdrant filter syntax) -----

export interface Filter {
  must?: Condition | Condition[]
  should?: Condition | Condition[]
  must_not?: Condition | Condition[]
  min_should?: { conditions: Condition[]; min_count: number }
}

export type MatchValue = string | number | boolean
export type Match =
  | { value: MatchValue }
  | { any: MatchValue[] }
  | { except: MatchValue[] }
  | { text: string }
  | { text_any: string }
  | { phrase: string }

export interface Range {
  gt?: number | string
  gte?: number | string
  lt?: number | string
  lte?: number | string
}

export type Condition =
  | { key: string; match?: Match; range?: Range; values_count?: Range; is_empty?: boolean; is_null?: boolean; geo_radius?: unknown; geo_bounding_box?: unknown; geo_polygon?: unknown }
  | { has_id: PointId[] }
  | { has_vector: string }
  | { is_empty: { key: string } }
  | { is_null: { key: string } }
  | { nested: { key: string; filter: Filter } }
  | Filter

export interface SearchParams {
  hnsw_ef?: number
  exact?: boolean
  quantization?: { ignore?: boolean; rescore?: boolean; oversampling?: number }
  indexed_only?: boolean
}

export type WithPayload = boolean | string[] | { include: string[] } | { exclude: string[] }
export type WithVector = boolean | string[]

export type QueryInput =
  | DenseVector
  | MultiDenseVector
  | SparseVector
  | { nearest: DenseVector | MultiDenseVector | SparseVector }
  | { fusion: 'rrf' | 'dbsf' }
  | { order_by: string | { key: string; direction?: 'asc' | 'desc' } }

export interface Prefetch {
  prefetch?: Prefetch | Prefetch[]
  query?: QueryInput
  using?: string
  filter?: Filter
  params?: SearchParams
  score_threshold?: number
  limit?: number
}

export interface QueryRequest extends Prefetch {
  offset?: number
  /** Default `true`. */
  with_payload?: WithPayload
  /** Default `false`. */
  with_vector?: WithVector
}

export type SearchOptions = Omit<QueryRequest, 'query'>

export interface ScoredPoint {
  id: PointId
  score: number
  payload?: Payload
  vector?: VectorInput
  order_value?: number | string
}

export interface Record {
  id: PointId
  payload?: Payload
  vector?: VectorInput
}

export interface ScrollRequest {
  offset?: PointId
  limit?: number
  filter?: Filter
  order_by?: string | { key: string; direction?: 'asc' | 'desc' }
  with_payload?: WithPayload
  with_vector?: WithVector
}

export interface ScrollResponse {
  points: Record[]
  next_page_offset: PointId | null
}

export type PointsSelector = PointId[] | { points: PointId[] } | { filter: Filter }

export type PayloadSchema =
  | 'keyword'
  | 'integer'
  | 'float'
  | 'bool'
  | 'geo'
  | 'datetime'
  | 'text'
  | 'uuid'
  | { type: string; [param: string]: unknown }

export interface CollectionInfo {
  name: string
  points_count: number
  indexed_vectors_count: number
  segments_count: number
  payload_schema: { [field: string]: unknown }
  config: unknown
}

export interface DatabaseOptions {
  /** "full" (default for files): flush every write; "periodic": flush every `flushIntervalMs`; "manual" (default for temp). */
  durability?: 'full' | 'periodic' | 'manual'
  flushIntervalMs?: number
  /** Build HNSW indexes / merge segments in the background after writes settle. Default true. */
  autoOptimize?: boolean
  optimizeDebounceMs?: number
  maxIndexingThreads?: number
}

export class NeoquadrantError extends Error {
  code:
    | 'NEOQ_ENGINE'
    | 'NEOQ_IO'
    | 'NEOQ_LOCKED'
    | 'NEOQ_EXISTS'
    | 'NEOQ_NOT_FOUND'
    | 'NEOQ_CLOSED'
    | 'NEOQ_INVALID_ARGUMENT'
}

export class Collection {
  readonly name: string
  upsert(points: PointStruct[]): void
  upsertAsync(points: PointStruct[]): Promise<void>
  query(request: QueryRequest): ScoredPoint[]
  queryAsync(request: QueryRequest): Promise<ScoredPoint[]>
  /** Nearest-neighbour shorthand for `query({query: vector, ...options})`. */
  search(vector: QueryInput, options?: SearchOptions): ScoredPoint[]
  searchAsync(vector: QueryInput, options?: SearchOptions): Promise<ScoredPoint[]>
  retrieve(ids: PointId[], options?: { with_payload?: WithPayload; with_vector?: WithVector }): Record[]
  scroll(request?: ScrollRequest): ScrollResponse
  count(request?: { filter?: Filter; exact?: boolean }): number
  /** Value counts of an indexed keyword/integer/bool/uuid field. */
  facet(request: { key: string; limit?: number; filter?: Filter; exact?: boolean }): { value: string | number | boolean; count: number }[]
  delete(target: PointsSelector): void
  setPayload(payload: Payload, target: PointsSelector): void
  overwritePayload(payload: Payload, target: PointsSelector): void
  deletePayload(keys: string[], target: PointsSelector): void
  clearPayload(target: PointsSelector): void
  createPayloadIndex(fieldName: string, fieldSchema: PayloadSchema): void
  deletePayloadIndex(fieldName: string): void
  /** Run Qdrant's optimizers now (blocking). Returns whether anything changed. */
  optimize(): boolean
  optimizeAsync(): Promise<boolean>
  flush(): void
  info(): CollectionInfo
}

export class Database {
  /** Open (or create) a database directory. Omit `location` or pass ":memory:" for a temporary database. */
  constructor(location?: string, options?: DatabaseOptions)
  readonly path: string
  createCollection(name: string, config: CreateCollection): Collection
  getOrCreateCollection(name: string, config: CreateCollection): Collection
  /** @throws NeoquadrantError with code NEOQ_NOT_FOUND */
  collection(name: string): Collection
  hasCollection(name: string): boolean
  listCollections(): string[]
  dropCollection(name: string): boolean
  flush(): void
  /** Stop background work, flush, release the directory lock. */
  close(): void
  [Symbol.dispose](): void
}
