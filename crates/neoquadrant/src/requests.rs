//! Request and response types, shaped like Qdrant's REST API so its documentation applies.
//!
//! Every type here is `serde`, so bindings pass plain JSON (`serde_json::Value`) through, and
//! Rust users can build the structs directly. Filters, payload index schemas, search params and
//! `with_payload` selectors are Qdrant's own engine types.

use std::collections::HashMap;

use edge::{
    EdgeConfig, EdgeOptimizersConfig, EdgeSparseVectorParams, EdgeVectorParams, Filter, Fusion,
    HnswIndexConfig, NamedQuery, OrderByInterface, Payload, PayloadFieldSchema, PointId,
    PointStructPersisted, QuantizationConfig, QueryEnum, ScoringQuery, SearchParams,
    VectorInternal, VectorPersisted, VectorStructInternal, VectorStructPersisted,
    WithPayloadInterface, WithVector,
};
use serde::{Deserialize, Serialize};

use crate::{Error, Result};

/// A point to upsert: `{"id": 1, "vector": [..], "payload": {..}}`.
///
/// `vector` is a dense list, a multi-vector (list of lists), or an object of named vectors where
/// each value is dense, multi-dense or sparse (`{"indices": [..], "values": [..]}`).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PointStruct {
    pub id: PointId,
    pub vector: VectorsInput,
    #[serde(default)]
    pub payload: Option<Payload>,
}

impl PointStruct {
    pub fn new(id: impl Into<PointId>, vector: Vec<f32>) -> Self {
        PointStruct {
            id: id.into(),
            vector: VectorsInput::Dense(vector),
            payload: None,
        }
    }

    pub(crate) fn into_persisted(self) -> Result<PointStructPersisted> {
        let vector = match self.vector {
            VectorsInput::Dense(v) => VectorStructPersisted::Single(v),
            VectorsInput::MultiDense(m) => VectorStructPersisted::MultiDense(m),
            VectorsInput::Named(map) => VectorStructPersisted::Named(
                map.into_iter()
                    .map(|(name, v)| Ok((name, v.into_persisted()?)))
                    .collect::<Result<_>>()?,
            ),
        };
        Ok(PointStructPersisted {
            id: self.id,
            vector,
            payload: self.payload,
        })
    }
}

/// The `vector` field of a point.
///
/// Untagged variant order matters: serde can read a struct from a JSON array, so Qdrant's
/// persisted vector type (tried Dense → Sparse → MultiDense) would parse the multivector
/// `[[1, 0], [0, 1]]` as a sparse vector. These input types try sparse last, as objects only.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum VectorsInput {
    Dense(Vec<f32>),
    MultiDense(Vec<Vec<f32>>),
    Named(HashMap<String, VectorInput>),
}

/// One (possibly named) vector: dense `[..]`, multi-dense `[[..], ..]` or sparse
/// `{"indices": [..], "values": [..]}`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum VectorInput {
    Dense(Vec<f32>),
    MultiDense(Vec<Vec<f32>>),
    Sparse(SparseInput),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SparseInput {
    pub indices: Vec<u32>,
    pub values: Vec<f32>,
}

impl VectorInput {
    fn into_persisted(self) -> Result<VectorPersisted> {
        Ok(match self {
            VectorInput::Dense(v) => VectorPersisted::Dense(v),
            VectorInput::MultiDense(m) => VectorPersisted::MultiDense(m),
            VectorInput::Sparse(s) => VectorPersisted::Sparse(
                edge::SparseVector::new(s.indices, s.values)
                    .map_err(|e| Error::InvalidArgument(format!("invalid sparse vector: {e}")))?,
            ),
        })
    }
}

// ---------------------------------------------------------------------------------------------
// Collection config
// ---------------------------------------------------------------------------------------------

/// `create_collection` body: `{"vectors": {"size": 384, "distance": "Cosine"}}`.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CreateCollection {
    #[serde(default)]
    pub vectors: Option<VectorsConfig>,
    #[serde(default)]
    pub sparse_vectors: HashMap<String, EdgeSparseVectorParams>,
    #[serde(default)]
    pub hnsw_config: Option<HnswIndexConfig>,
    #[serde(default)]
    pub quantization_config: Option<QuantizationConfig>,
    #[serde(default)]
    pub optimizers_config: Option<EdgeOptimizersConfig>,
    /// Store payloads on disk (mmap) instead of RAM.
    #[serde(default)]
    pub on_disk_payload: Option<bool>,
}

/// One unnamed vector, or a map of named vectors.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum VectorsConfig {
    Single(EdgeVectorParams),
    Named(HashMap<String, EdgeVectorParams>),
}

impl CreateCollection {
    #[allow(deprecated)]
    pub fn into_edge_config(self) -> Result<EdgeConfig> {
        let vectors = match self.vectors {
            Some(VectorsConfig::Single(p)) => HashMap::from([(String::new(), p)]),
            Some(VectorsConfig::Named(map)) => map,
            None => HashMap::new(),
        };
        if vectors.is_empty() && self.sparse_vectors.is_empty() {
            return Err(Error::InvalidArgument(
                "collection needs at least one dense (`vectors`) or sparse (`sparse_vectors`) vector".into(),
            ));
        }
        Ok(EdgeConfig {
            on_disk_payload: self.on_disk_payload,
            vectors,
            sparse_vectors: self.sparse_vectors,
            hnsw_config: self.hnsw_config,
            quantization_config: self.quantization_config,
            optimizers: self.optimizers_config,
            ..Default::default()
        })
    }
}

// ---------------------------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------------------------

fn default_limit() -> usize {
    10
}

fn default_true() -> bool {
    true
}

fn payload_on() -> WithPayloadInterface {
    WithPayloadInterface::Bool(true)
}

fn vector_off() -> WithVector {
    WithVector::Bool(false)
}

/// What to score candidates by (Qdrant's universal `query` field).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum QueryInput {
    /// `{"nearest": [..]}`
    Nearest { nearest: VectorInput },
    /// `{"fusion": "rrf" | "dbsf"}` — combine the results of `prefetch` stages.
    Fusion { fusion: FusionKind },
    /// `{"order_by": "price"}` or `{"order_by": {"key": "price", "direction": "desc"}}`
    OrderBy { order_by: OrderByInterface },
    /// A bare vector: dense `[..]`, multi-dense `[[..], ..]` or sparse `{"indices", "values"}`.
    Vector(VectorInput),
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FusionKind {
    Rrf,
    Dbsf,
}

impl QueryInput {
    fn into_scoring(self, using: Option<String>) -> Result<ScoringQuery> {
        Ok(match self {
            QueryInput::Nearest { nearest: v } | QueryInput::Vector(v) => {
                ScoringQuery::Vector(QueryEnum::Nearest(NamedQuery {
                    query: VectorInternal::from(v.into_persisted()?),
                    using,
                }))
            }
            QueryInput::Fusion {
                fusion: FusionKind::Rrf,
            } => ScoringQuery::Fusion(Fusion::Rrf {
                k: segment::common::reciprocal_rank_fusion::DEFAULT_RRF_K,
                weights: None,
            }),
            QueryInput::Fusion {
                fusion: FusionKind::Dbsf,
            } => ScoringQuery::Fusion(Fusion::Dbsf),
            QueryInput::OrderBy { order_by } => ScoringQuery::OrderBy(order_by.into()),
        })
    }
}

/// Accept a single object or a list.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OneOrMany<T> {
    One(Box<T>),
    Many(Vec<T>),
}

impl<T> OneOrMany<T> {
    fn into_vec(self) -> Vec<T> {
        match self {
            OneOrMany::One(t) => vec![*t],
            OneOrMany::Many(v) => v,
        }
    }
}

/// Universal query: `{"query": [..], "filter": {..}, "limit": 10}` (Qdrant `/points/query`).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QueryRequest {
    #[serde(default)]
    pub prefetch: Option<OneOrMany<PrefetchRequest>>,
    #[serde(default)]
    pub query: Option<QueryInput>,
    /// Named vector to search with. Omit for the unnamed (default) vector.
    #[serde(default)]
    pub using: Option<String>,
    #[serde(default)]
    pub filter: Option<Filter>,
    #[serde(default)]
    pub params: Option<SearchParams>,
    #[serde(default)]
    pub score_threshold: Option<f32>,
    #[serde(default = "default_limit")]
    pub limit: usize,
    #[serde(default)]
    pub offset: usize,
    /// Default `true` (Qdrant's REST default is `false`; in-process you usually want it).
    #[serde(default = "payload_on")]
    pub with_payload: WithPayloadInterface,
    #[serde(default = "vector_off")]
    pub with_vector: WithVector,
}

impl QueryRequest {
    /// Nearest-neighbour query for a dense vector.
    pub fn nearest(vector: Vec<f32>, limit: usize) -> Self {
        QueryRequest {
            prefetch: None,
            query: Some(QueryInput::Nearest {
                nearest: VectorInput::Dense(vector),
            }),
            using: None,
            filter: None,
            params: None,
            score_threshold: None,
            limit,
            offset: 0,
            with_payload: payload_on(),
            with_vector: vector_off(),
        }
    }

    pub fn filter(mut self, filter: Filter) -> Self {
        self.filter = Some(filter);
        self
    }

    /// Vector names this request scores with (`None` = the unnamed default vector),
    /// including all nested prefetch stages.
    pub(crate) fn vector_names(&self) -> Vec<Option<&str>> {
        fn walk<'a>(
            query: &Option<QueryInput>,
            using: &'a Option<String>,
            prefetch: &'a Option<OneOrMany<PrefetchRequest>>,
            out: &mut Vec<Option<&'a str>>,
        ) {
            if matches!(
                query,
                Some(QueryInput::Nearest { .. } | QueryInput::Vector(_))
            ) {
                out.push(using.as_deref());
            }
            let stages: Vec<&PrefetchRequest> = match prefetch {
                Some(OneOrMany::One(p)) => vec![p],
                Some(OneOrMany::Many(v)) => v.iter().collect(),
                None => vec![],
            };
            for p in stages {
                walk(&p.query, &p.using, &p.prefetch, out);
            }
        }
        let mut out = Vec::new();
        walk(&self.query, &self.using, &self.prefetch, &mut out);
        out
    }

    /// Every filter in the request, including nested prefetch stages.
    pub(crate) fn filters(&self) -> Vec<&Filter> {
        fn walk<'a>(
            filter: &'a Option<Filter>,
            prefetch: &'a Option<OneOrMany<PrefetchRequest>>,
            out: &mut Vec<&'a Filter>,
        ) {
            out.extend(filter.iter());
            let stages: Vec<&PrefetchRequest> = match prefetch {
                Some(OneOrMany::One(p)) => vec![p],
                Some(OneOrMany::Many(v)) => v.iter().collect(),
                None => vec![],
            };
            for p in stages {
                walk(&p.filter, &p.prefetch, out);
            }
        }
        let mut out = Vec::new();
        walk(&self.filter, &self.prefetch, &mut out);
        out
    }

    pub(crate) fn into_edge(self) -> Result<edge::QueryRequest> {
        Ok(edge::QueryRequest {
            prefetches: self
                .prefetch
                .map(OneOrMany::into_vec)
                .unwrap_or_default()
                .into_iter()
                .map(PrefetchRequest::into_edge)
                .collect::<Result<_>>()?,
            query: self.query.map(|q| q.into_scoring(self.using)).transpose()?,
            filter: self.filter,
            score_threshold: self.score_threshold,
            limit: self.limit,
            offset: self.offset,
            params: self.params,
            with_vector: self.with_vector,
            with_payload: self.with_payload,
        })
    }
}

/// One stage of a multi-stage query (hybrid search, re-ranking).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrefetchRequest {
    #[serde(default)]
    pub prefetch: Option<OneOrMany<PrefetchRequest>>,
    #[serde(default)]
    pub query: Option<QueryInput>,
    #[serde(default)]
    pub using: Option<String>,
    #[serde(default)]
    pub filter: Option<Filter>,
    #[serde(default)]
    pub params: Option<SearchParams>,
    #[serde(default)]
    pub score_threshold: Option<f32>,
    #[serde(default = "default_limit")]
    pub limit: usize,
}

impl PrefetchRequest {
    fn into_edge(self) -> Result<edge::Prefetch> {
        Ok(edge::Prefetch {
            prefetches: self
                .prefetch
                .map(OneOrMany::into_vec)
                .unwrap_or_default()
                .into_iter()
                .map(PrefetchRequest::into_edge)
                .collect::<Result<_>>()?,
            query: self.query.map(|q| q.into_scoring(self.using)).transpose()?,
            limit: self.limit,
            params: self.params,
            filter: self.filter,
            score_threshold: self.score_threshold,
        })
    }
}

/// `{"limit": 100, "offset": <id>, "filter": {..}}` — page through points in id order.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScrollRequest {
    #[serde(default)]
    pub offset: Option<PointId>,
    #[serde(default = "default_limit")]
    pub limit: usize,
    #[serde(default)]
    pub filter: Option<Filter>,
    #[serde(default)]
    pub order_by: Option<OrderByInterface>,
    #[serde(default = "payload_on")]
    pub with_payload: WithPayloadInterface,
    #[serde(default = "vector_off")]
    pub with_vector: WithVector,
}

impl Default for ScrollRequest {
    fn default() -> Self {
        serde_json::from_value(serde_json::json!({})).expect("defaults")
    }
}

/// `{"ids": [1, 2], "with_payload": true, "with_vector": false}`
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RetrieveRequest {
    pub ids: Vec<PointId>,
    #[serde(default = "payload_on")]
    pub with_payload: WithPayloadInterface,
    #[serde(default = "vector_off")]
    pub with_vector: WithVector,
}

/// `{"filter": {..}, "exact": true}`
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CountRequest {
    #[serde(default)]
    pub filter: Option<Filter>,
    #[serde(default = "default_true")]
    pub exact: bool,
}

/// `{"key": "city", "limit": 10, "filter": {..}}` — value counts of a payload field.
/// The field needs a keyword/integer/bool/uuid payload index.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FacetRequest {
    pub key: String,
    #[serde(default = "default_limit")]
    pub limit: usize,
    #[serde(default)]
    pub filter: Option<Filter>,
    #[serde(default = "default_true")]
    pub exact: bool,
}

/// Which points an operation targets: `{"points": [..]}` or `{"filter": {..}}`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PointsSelector {
    Ids { points: Vec<PointId> },
    Filter { filter: Filter },
}

/// `{"payload": {..}, "points": [..] | "filter": {..}, "key": "nested.path"}`
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SetPayloadRequest {
    pub payload: Payload,
    #[serde(default)]
    pub points: Option<Vec<PointId>>,
    #[serde(default)]
    pub filter: Option<Filter>,
    /// Set the payload under this nested key instead of the root.
    #[serde(default)]
    pub key: Option<String>,
}

/// `{"keys": ["a", "b"], "points": [..] | "filter": {..}}`
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeletePayloadRequest {
    pub keys: Vec<String>,
    #[serde(default)]
    pub points: Option<Vec<PointId>>,
    #[serde(default)]
    pub filter: Option<Filter>,
}

/// `{"field_name": "city", "field_schema": "keyword"}`
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CreateFieldIndex {
    pub field_name: String,
    pub field_schema: PayloadFieldSchema,
}

// ---------------------------------------------------------------------------------------------
// Responses
// ---------------------------------------------------------------------------------------------

/// A search hit.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScoredPoint {
    pub id: PointId,
    pub score: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub payload: Option<Payload>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vector: Option<VectorStructPersisted>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub order_value: Option<serde_json::Value>,
}

/// A stored point (`retrieve` / `scroll`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Record {
    pub id: PointId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub payload: Option<Payload>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vector: Option<VectorStructPersisted>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScrollResponse {
    pub points: Vec<Record>,
    pub next_page_offset: Option<PointId>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FacetHit {
    pub value: serde_json::Value,
    pub count: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CollectionInfo {
    pub name: String,
    pub points_count: usize,
    /// Vectors already in an HNSW index. Small collections are searched exactly instead.
    pub indexed_vectors_count: usize,
    pub segments_count: usize,
    pub payload_schema: HashMap<String, serde_json::Value>,
    pub config: EdgeConfig,
}

/// Collapse `{"": [..]}` (the unnamed vector) to a plain vector for readability.
pub(crate) fn vector_out(v: Option<VectorStructInternal>) -> Option<VectorStructPersisted> {
    v.map(|v| match VectorStructPersisted::from(v) {
        VectorStructPersisted::Named(mut map) if map.len() == 1 && map.contains_key("") => {
            match map.remove("").expect("checked") {
                VectorPersisted::Dense(d) => VectorStructPersisted::Single(d),
                VectorPersisted::MultiDense(m) => VectorStructPersisted::MultiDense(m),
                sparse => VectorStructPersisted::Named(HashMap::from([(String::new(), sparse)])),
            }
        }
        other => other,
    })
}
