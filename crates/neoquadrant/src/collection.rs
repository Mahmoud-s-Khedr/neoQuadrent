use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use edge::{
    CountRequestBuilder, CreateIndex, DeletePayloadOp, EdgeConfig, EdgeShard, FacetRequestBuilder,
    FacetValue, FieldIndexOperations, JsonPath, PayloadOps, PointId, PointInsertOperations,
    PointOperations, RetrieveRequestBuilder, ScrollRequestBuilder, SetPayloadOp, UpdateOperation,
};
use parking_lot::{Mutex, RwLock, RwLockReadGuard};

use crate::db::Ctx;
use crate::options::Durability;
use crate::requests::*;
use crate::{Error, Result};

/// Handle to one collection. Cheap to clone and safe to share between threads.
///
/// Reads run concurrently; writes are serialized by Qdrant's WAL.
#[derive(Clone)]
pub struct Collection {
    pub(crate) inner: Arc<CollectionInner>,
}

pub(crate) struct CollectionInner {
    pub(crate) name: String,
    pub(crate) path: PathBuf,
    /// `None` once the collection is dropped or the database is closed.
    pub(crate) shard: RwLock<Option<EdgeShard>>,
    pub(crate) ctx: Arc<Ctx>,
    pub(crate) needs_optimize: AtomicBool,
    pub(crate) needs_flush: AtomicBool,
    pub(crate) last_write: Mutex<Instant>,
    /// Serializes optimization passes, so an explicit `optimize()` waits for an in-flight
    /// background pass instead of returning early while the index is still being built.
    optimize_lock: Mutex<()>,
}

impl std::fmt::Debug for Collection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Collection")
            .field("name", &self.inner.name)
            .finish()
    }
}

impl CollectionInner {
    pub(crate) fn new(name: String, path: PathBuf, shard: EdgeShard, ctx: Arc<Ctx>) -> Self {
        CollectionInner {
            name,
            path,
            shard: RwLock::new(Some(shard)),
            ctx,
            needs_optimize: AtomicBool::new(true),
            needs_flush: AtomicBool::new(false),
            last_write: Mutex::new(Instant::now()),
            optimize_lock: Mutex::new(()),
        }
    }

    pub(crate) fn shard(&self) -> Result<ShardGuard<'_>> {
        let guard = self.shard.read();
        if guard.is_none() {
            return Err(Error::Closed(self.name.clone()));
        }
        Ok(ShardGuard(guard))
    }

    /// Close the shard (flushes via `EdgeShard`'s Drop). Waits for in-flight operations.
    pub(crate) fn close(&self) {
        self.shard.write().take();
    }

    pub(crate) fn optimize_now(&self) -> Result<bool> {
        let _serial = self.optimize_lock.lock();
        let shard = self.shard()?;
        self.needs_optimize.store(false, Ordering::Release);
        Ok(shard.optimize_with_budget(self.ctx.budget.clone())?)
    }

    pub(crate) fn flush_now(&self) -> Result<()> {
        let shard = self.shard()?;
        self.needs_flush.store(false, Ordering::Release);
        Ok(shard.flush()?)
    }
}

pub(crate) struct ShardGuard<'a>(RwLockReadGuard<'a, Option<EdgeShard>>);

impl std::ops::Deref for ShardGuard<'_> {
    type Target = EdgeShard;
    fn deref(&self) -> &EdgeShard {
        self.0.as_ref().expect("checked in CollectionInner::shard")
    }
}

impl Collection {
    pub fn name(&self) -> &str {
        &self.inner.name
    }

    /// Directory holding this collection's segments and WAL.
    pub fn path(&self) -> &std::path::Path {
        &self.inner.path
    }

    // ------------------------------------------------------------------ writes

    fn apply(&self, op: UpdateOperation) -> Result<()> {
        let shard = self.inner.shard()?;
        shard.update(op)?;
        let durability = self.inner.ctx.options.durability;
        if durability == Durability::Full {
            shard.flush()?;
        } else {
            self.inner.needs_flush.store(true, Ordering::Release);
        }
        drop(shard);
        *self.inner.last_write.lock() = Instant::now();
        self.inner.needs_optimize.store(true, Ordering::Release);
        self.inner.ctx.notify();
        Ok(())
    }

    /// Insert or replace points.
    pub fn upsert(&self, points: Vec<PointStruct>) -> Result<()> {
        if points.is_empty() {
            return Ok(());
        }
        let points = points
            .into_iter()
            .map(PointStruct::into_persisted)
            .collect::<Result<Vec<_>>>()?;
        points.iter().try_for_each(check)?;
        self.apply(UpdateOperation::PointOperation(
            PointOperations::UpsertPoints(PointInsertOperations::PointsList(points)),
        ))
    }

    /// Delete points by id or by filter.
    pub fn delete(&self, selector: PointsSelector) -> Result<()> {
        if let PointsSelector::Filter { filter } = &selector {
            check(filter)?;
        }
        let op = match selector {
            PointsSelector::Ids { points } => PointOperations::DeletePoints { ids: points },
            PointsSelector::Filter { filter } => PointOperations::DeletePointsByFilter(filter),
        };
        self.apply(UpdateOperation::PointOperation(op))
    }

    /// Merge keys into the payload of the selected points.
    pub fn set_payload(&self, req: SetPayloadRequest) -> Result<()> {
        self.apply(UpdateOperation::PayloadOperation(PayloadOps::SetPayload(
            set_op(req)?,
        )))
    }

    /// Replace the whole payload of the selected points.
    pub fn overwrite_payload(&self, req: SetPayloadRequest) -> Result<()> {
        self.apply(UpdateOperation::PayloadOperation(
            PayloadOps::OverwritePayload(set_op(req)?),
        ))
    }

    /// Remove keys from the payload of the selected points.
    pub fn delete_payload(&self, req: DeletePayloadRequest) -> Result<()> {
        require_selector(&req.points, &req.filter)?;
        check_filter(&req.filter)?;
        let keys = req
            .keys
            .iter()
            .map(|k| parse_path(k))
            .collect::<Result<_>>()?;
        self.apply(UpdateOperation::PayloadOperation(
            PayloadOps::DeletePayload(DeletePayloadOp {
                keys,
                points: req.points,
                filter: req.filter,
            }),
        ))
    }

    /// Remove the whole payload of the selected points.
    pub fn clear_payload(&self, selector: PointsSelector) -> Result<()> {
        if let PointsSelector::Filter { filter } = &selector {
            check(filter)?;
        }
        let op = match selector {
            PointsSelector::Ids { points } => PayloadOps::ClearPayload { points },
            PointsSelector::Filter { filter } => PayloadOps::ClearPayloadByFilter(filter),
        };
        self.apply(UpdateOperation::PayloadOperation(op))
    }

    /// Index a payload field so filters on it are fast (`keyword`, `integer`, `float`, `bool`,
    /// `geo`, `datetime`, `text`, `uuid`). Same semantics as Qdrant's payload indexes.
    pub fn create_payload_index(&self, req: CreateFieldIndex) -> Result<()> {
        self.apply(UpdateOperation::FieldIndexOperation(
            FieldIndexOperations::CreateIndex(CreateIndex {
                field_name: parse_path(&req.field_name)?,
                field_schema: Some(req.field_schema),
            }),
        ))
    }

    pub fn delete_payload_index(&self, field_name: &str) -> Result<()> {
        self.apply(UpdateOperation::FieldIndexOperation(
            FieldIndexOperations::DeleteIndex(parse_path(field_name)?),
        ))
    }

    // ------------------------------------------------------------------ reads

    /// Universal query: nearest neighbours, hybrid prefetch + fusion, order-by.
    pub fn query(&self, req: QueryRequest) -> Result<Vec<ScoredPoint>> {
        let shard = self.inner.shard()?;
        for filter in req.filters() {
            check(filter)?;
        }
        // Qdrant Edge returns no results (instead of an error) for an unknown vector name,
        // which turns a typo into silently empty searches. Reject it like the server does.
        {
            let config = shard.config();
            for name in req.vector_names() {
                let name = name.unwrap_or("");
                if !config.vectors.contains_key(name) && !config.sparse_vectors.contains_key(name) {
                    let mut known: Vec<&str> = config
                        .vectors
                        .keys()
                        .chain(config.sparse_vectors.keys())
                        .map(String::as_str)
                        .collect();
                    known.sort_unstable();
                    let shown = if name.is_empty() { "<default>" } else { name };
                    return Err(Error::InvalidArgument(format!(
                        "vector `{shown}` does not exist in collection `{}` (vectors: {known:?}); set `using`",
                        self.inner.name
                    )));
                }
            }
        }
        let hits = shard.query(req.into_edge()?)?;
        let mut hits: Vec<ScoredPoint> = hits
            .into_iter()
            .map(|p| ScoredPoint {
                id: p.id,
                score: p.score,
                payload: p.payload,
                vector: vector_out(p.vector),
                order_value: p.order_value.and_then(|v| serde_json::to_value(v).ok()),
            })
            .collect();
        order_ties_by_id(&mut hits);
        Ok(hits)
    }

    /// Shorthand for a nearest-neighbour query on the default vector.
    pub fn search(&self, vector: Vec<f32>, limit: usize) -> Result<Vec<ScoredPoint>> {
        self.query(QueryRequest::nearest(vector, limit))
    }

    /// Fetch points by id. Missing ids are skipped.
    pub fn retrieve(&self, req: RetrieveRequest) -> Result<Vec<Record>> {
        let shard = self.inner.shard()?;
        let records = shard.retrieve(
            RetrieveRequestBuilder::new(req.ids)
                .with_payload(req.with_payload)
                .with_vector(req.with_vector)
                .build(),
        )?;
        Ok(records.into_iter().map(record_out).collect())
    }

    /// Page through points (by id, or by `order_by`).
    pub fn scroll(&self, req: ScrollRequest) -> Result<ScrollResponse> {
        check_filter(&req.filter)?;
        let shard = self.inner.shard()?;
        let mut b = ScrollRequestBuilder::new()
            .limit(req.limit)
            .with_payload(req.with_payload)
            .with_vector(req.with_vector);
        if let Some(offset) = req.offset {
            b = b.offset(offset);
        }
        if let Some(filter) = req.filter {
            b = b.filter(filter);
        }
        if let Some(order_by) = req.order_by {
            b = b.order_by(order_by);
        }
        let (points, next) = shard.scroll(b.build())?;
        Ok(ScrollResponse {
            points: points.into_iter().map(record_out).collect(),
            next_page_offset: next,
        })
    }

    pub fn count(&self, req: CountRequest) -> Result<usize> {
        check_filter(&req.filter)?;
        let shard = self.inner.shard()?;
        let mut b = CountRequestBuilder::new().exact(req.exact);
        if let Some(filter) = req.filter {
            b = b.filter(filter);
        }
        Ok(shard.count(b.build())?)
    }

    /// Count distinct values of an indexed payload field.
    pub fn facet(&self, req: FacetRequest) -> Result<Vec<FacetHit>> {
        check_filter(&req.filter)?;
        let shard = self.inner.shard()?;
        let mut b = FacetRequestBuilder::new(parse_path(&req.key)?)
            .limit(req.limit)
            .exact(req.exact);
        if let Some(filter) = req.filter {
            b = b.filter(filter);
        }
        let resp = shard.facet(b.build())?;
        Ok(resp
            .hits
            .into_iter()
            .map(|h| FacetHit {
                value: match h.value {
                    FacetValue::Keyword(s) => s.into(),
                    FacetValue::Int(i) => i.into(),
                    FacetValue::Bool(b) => b.into(),
                    FacetValue::Uuid(u) => {
                        edge::external::uuid::Uuid::from_u128(u).to_string().into()
                    }
                },
                count: h.count,
            })
            .collect())
    }

    pub fn info(&self) -> Result<CollectionInfo> {
        let shard = self.inner.shard()?;
        let info = shard.info()?;
        let config: EdgeConfig = shard.config().clone();
        Ok(CollectionInfo {
            name: self.inner.name.clone(),
            points_count: info.points_count,
            indexed_vectors_count: info.indexed_vectors_count,
            segments_count: info.segments_count,
            payload_schema: info
                .payload_schema
                .into_iter()
                .map(|(k, v)| (k.to_string(), serde_json::to_value(v).unwrap_or_default()))
                .collect(),
            config,
        })
    }

    // ------------------------------------------------------------------ maintenance

    /// Run Qdrant's optimizers now (build HNSW indexes, merge segments, vacuum deletions).
    /// Blocks until done. Returns whether anything changed.
    pub fn optimize(&self) -> Result<bool> {
        self.inner.optimize_now()
    }

    /// Make all writes so far durable on disk.
    pub fn flush(&self) -> Result<()> {
        self.inner.flush_now()
    }
}

/// The engine returns exactly tied results (common with RRF fusion, whose implementation "does not
/// break ties", and with duplicate vectors) in arbitrary order, so the same query can come back in a different order on every call.
/// Order each run of equal (score, order_value) by point id so results are deterministic.
/// Only tied neighbours move; the ranking itself is unchanged.
fn order_ties_by_id(hits: &mut [ScoredPoint]) {
    let mut start = 0;
    while start < hits.len() {
        let mut end = start + 1;
        while end < hits.len()
            && hits[end].score == hits[start].score
            && hits[end].order_value == hits[start].order_value
        {
            end += 1;
        }
        hits[start..end].sort_by_key(|p| p.id);
        start = end;
    }
}

/// Run Qdrant's own request validators (the server runs these at its API layer; Edge does not).
fn check<T: validator::Validate>(value: &T) -> Result<()> {
    value
        .validate()
        .map_err(|e| Error::InvalidArgument(e.to_string().replace('\n', "; ")))
}

fn check_filter(filter: &Option<edge::Filter>) -> Result<()> {
    filter.as_ref().map_or(Ok(()), check)
}

fn record_out(r: edge::Record) -> Record {
    Record {
        id: r.id,
        payload: r.payload,
        vector: vector_out(r.vector),
    }
}

fn parse_path(key: &str) -> Result<JsonPath> {
    key.parse()
        .map_err(|_| Error::InvalidArgument(format!("invalid payload key `{key}`")))
}

fn require_selector(points: &Option<Vec<PointId>>, filter: &Option<edge::Filter>) -> Result<()> {
    if points.is_none() && filter.is_none() {
        return Err(Error::InvalidArgument(
            "specify `points` or `filter`".into(),
        ));
    }
    Ok(())
}

fn set_op(req: SetPayloadRequest) -> Result<SetPayloadOp> {
    require_selector(&req.points, &req.filter)?;
    check_filter(&req.filter)?;
    Ok(SetPayloadOp {
        payload: req.payload,
        points: req.points,
        filter: req.filter,
        key: req.key.as_deref().map(parse_path).transpose()?,
    })
}
