use std::time::{Duration, Instant};

use neoquadrant::requests::*;
use neoquadrant::{Database, Durability, Error, Options};
use serde_json::{Value, json};

fn from<T: serde::de::DeserializeOwned>(v: Value) -> T {
    serde_json::from_value(v).unwrap()
}

/// Deterministic pseudo-random vectors (xorshift), so tests need no extra deps.
fn vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut s = seed.max(1);
    (0..n)
        .map(|_| {
            (0..dim)
                .map(|_| {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    (s % 10_000) as f32 / 5_000.0 - 1.0
                })
                .collect()
        })
        .collect()
}

fn ids(hits: &[ScoredPoint]) -> Vec<Value> {
    hits.iter()
        .map(|h| serde_json::to_value(h.id).unwrap())
        .collect()
}

#[test]
fn crud_filters_and_payloads() {
    let db = Database::open_temp().unwrap();
    let col = db
        .create_collection(
            "docs",
            from(json!({"vectors": {"size": 4, "distance": "Cosine"}})),
        )
        .unwrap();

    col.upsert(from(json!([
        {"id": 1, "vector": [1.0, 0.0, 0.0, 0.0], "payload": {"city": "London", "price": 10, "tags": ["a", "b"]}},
        {"id": 2, "vector": [0.9, 0.1, 0.0, 0.0], "payload": {"city": "Paris", "price": 20}},
        {"id": "5c56c793-69f3-4fbf-87e6-c4bf54c28c26", "vector": [0.0, 1.0, 0.0, 0.0], "payload": {"city": "London", "price": 30}},
    ])))
    .unwrap();

    let hits = col.search(vec![1.0, 0.0, 0.0, 0.0], 10).unwrap();
    assert_eq!(ids(&hits)[..2], [json!(1), json!(2)]);
    assert!((hits[0].score - 1.0).abs() < 1e-5);
    assert_eq!(
        serde_json::to_value(&hits[0].payload).unwrap()["city"],
        "London"
    );

    let hits = col
        .query(from(json!({
            "query": [1.0, 0.0, 0.0, 0.0],
            "filter": {"must": [{"key": "city", "match": {"value": "London"}}, {"key": "price", "range": {"gte": 20}}]},
        })))
        .unwrap();
    assert_eq!(ids(&hits), [json!("5c56c793-69f3-4fbf-87e6-c4bf54c28c26")]);

    assert_eq!(
        col.count(from(
            json!({"filter": {"must": [{"key": "tags", "match": {"any": ["b"]}}]}})
        ))
        .unwrap(),
        1
    );
    assert_eq!(col.count(CountRequest::default()).unwrap(), 3);

    let got = col
        .retrieve(from(json!({"ids": [2, 99], "with_vector": true})))
        .unwrap();
    assert_eq!(got.len(), 1);
    assert_eq!(
        serde_json::to_value(got[0].vector.as_ref().unwrap())
            .unwrap()
            .as_array()
            .unwrap()
            .len(),
        4
    );

    col.set_payload(from(json!({"payload": {"stock": true}, "points": [2]})))
        .unwrap();
    col.delete_payload(from(json!({"keys": ["price"], "points": [2]})))
        .unwrap();
    let p2 = &col.retrieve(from(json!({"ids": [2]}))).unwrap()[0];
    assert_eq!(
        serde_json::to_value(&p2.payload).unwrap(),
        json!({"city": "Paris", "stock": true})
    );

    col.delete(from(
        json!({"filter": {"must": [{"key": "city", "match": {"value": "London"}}]}}),
    ))
    .unwrap();
    assert_eq!(col.count(CountRequest::default()).unwrap(), 1);

    let page = col.scroll(from(json!({"limit": 10}))).unwrap();
    assert_eq!(page.points.len(), 1);
    assert_eq!(page.next_page_offset, None);

    // Wrong dimension is a Qdrant engine error, not a panic.
    let err = col
        .upsert(from(json!([{"id": 9, "vector": [1.0, 2.0]}])))
        .unwrap_err();
    assert!(matches!(err, Error::Engine(_)), "{err:?}");

    // Typos in filters are rejected (Qdrant's deny_unknown_fields).
    assert!(
        serde_json::from_value::<QueryRequest>(
            json!({"query": [1,0,0,0], "filter": {"mustt": []}})
        )
        .is_err()
    );
}

#[test]
fn payload_index_and_facets() {
    let db = Database::open_temp().unwrap();
    let col = db
        .create_collection(
            "f",
            from(json!({"vectors": {"size": 2, "distance": "Dot"}})),
        )
        .unwrap();
    col.create_payload_index(from(
        json!({"field_name": "color", "field_schema": "keyword"}),
    ))
    .unwrap();
    let pts: Vec<Value> = (0..30)
        .map(|i| json!({"id": i, "vector": [1.0, i as f32], "payload": {"color": (["red", "green", "blue"][i % 3]), "n": i}}))
        .collect();
    col.upsert(from(Value::Array(pts))).unwrap();

    let facets = col.facet(from(json!({"key": "color"}))).unwrap();
    assert_eq!(facets.len(), 3);
    assert!(facets.iter().all(|h| h.count == 10));
    assert!(col.info().unwrap().payload_schema.contains_key("color"));

    let ordered = col.scroll(from(
        json!({"limit": 3, "order_by": {"key": "n", "direction": "desc"}}),
    ));
    // order_by needs a range index on `n`
    assert!(ordered.is_err());
    col.create_payload_index(from(json!({"field_name": "n", "field_schema": "integer"})))
        .unwrap();
    let ordered = col
        .scroll(from(
            json!({"limit": 3, "order_by": {"key": "n", "direction": "desc"}}),
        ))
        .unwrap();
    let ns: Vec<_> = ordered
        .points
        .iter()
        .map(|p| serde_json::to_value(&p.payload).unwrap()["n"].clone())
        .collect();
    assert_eq!(ns, [json!(29), json!(28), json!(27)]);
}

#[test]
fn persistence_reopen_and_lock() {
    let dir = tempfile::tempdir().unwrap();
    {
        let db = Database::open(dir.path()).unwrap();
        // Same path in the same process shares the handle instead of failing on the lock.
        let again = Database::open(dir.path()).unwrap();
        let col = db
            .create_collection(
                "a",
                from(json!({"vectors": {"size": 3, "distance": "Euclid"}})),
            )
            .unwrap();
        col.upsert(from(
            json!([{"id": 1, "vector": [1.0, 2.0, 3.0], "payload": {"k": "v"}}]),
        ))
        .unwrap();
        assert!(again.has_collection("a"));

        // Another open file description (= another process) cannot take the lock.
        let f = std::fs::File::open(dir.path().join("LOCK")).unwrap();
        assert!(f.try_lock().is_err());
        drop(again);
        db.close().unwrap();
        assert!(f.try_lock().is_ok());
        f.unlock().unwrap();
        // Handles after close fail cleanly.
        assert!(matches!(
            col.count(CountRequest::default()),
            Err(Error::Closed(_))
        ));
    }
    let db = Database::open(dir.path()).unwrap();
    assert_eq!(db.list_collections(), ["a"]);
    let col = db.collection("a").unwrap();
    let hits = col.search(vec![1.0, 2.0, 4.0], 1).unwrap();
    assert_eq!(ids(&hits), [json!(1)]);
    assert!(
        (hits[0].score - 1.0).abs() < 1e-5,
        "euclid distance, got {}",
        hits[0].score
    );

    assert!(db.drop_collection("a").unwrap());
    assert!(matches!(
        col.count(CountRequest::default()),
        Err(Error::Closed(_))
    ));
    assert!(!dir.path().join("collections/a").exists());
    assert!(matches!(
        db.collection("a"),
        Err(Error::CollectionNotFound(_))
    ));
}

#[test]
fn optimizer_builds_hnsw_and_recall_is_high() {
    let db = Database::open_temp_with(Options {
        auto_optimize: true,
        optimize_debounce: Duration::from_millis(100),
        durability: Durability::Manual,
        ..Options::default()
    })
    .unwrap();
    let dim = 64;
    // Low indexing threshold (KB) so a test-sized collection gets an HNSW index.
    let col = db
        .create_collection(
            "v",
            from(json!({"vectors": {"size": dim, "distance": "Cosine"}, "optimizers_config": {"indexing_threshold": 100}})),
        )
        .unwrap();
    let data = vectors(20_000, dim, 42);
    for (chunk_i, chunk) in data.chunks(1000).enumerate() {
        let pts: Vec<PointStruct> = chunk
            .iter()
            .enumerate()
            .map(|(i, v)| from(json!({"id": chunk_i * 1000 + i, "vector": v})))
            .collect();
        col.upsert(pts).unwrap();
    }

    // Background optimizer should index the collection without any explicit call.
    let start = Instant::now();
    loop {
        let info = col.info().unwrap();
        if info.indexed_vectors_count >= 19_000 {
            break;
        }
        assert!(
            start.elapsed() < Duration::from_secs(120),
            "not indexed: {info:?}"
        );
        std::thread::sleep(Duration::from_millis(200));
    }

    let queries = vectors(100, dim, 7);
    let mut found = 0;
    for q in &queries {
        let approx = ids(&col.query(QueryRequest::nearest(q.clone(), 10)).unwrap());
        let exact = ids(&col
            .query(from(
                json!({"query": q, "limit": 10, "params": {"exact": true}}),
            ))
            .unwrap());
        found += approx.iter().filter(|id| exact.contains(id)).count();
    }
    let recall = found as f64 / (queries.len() * 10) as f64;
    println!("recall@10 = {recall:.3}");
    assert!(recall >= 0.95, "recall@10 = {recall}");
}

#[test]
fn named_sparse_and_hybrid_fusion() {
    let db = Database::open_temp().unwrap();
    let col = db
        .create_collection(
            "h",
            from(json!({
                "vectors": {"dense": {"size": 3, "distance": "Cosine"}},
                "sparse_vectors": {"text": {}}
            })),
        )
        .unwrap();
    col.upsert(from(json!([
        {"id": 1, "vector": {"dense": [1.0, 0.0, 0.0], "text": {"indices": [1, 7], "values": [0.5, 1.0]}}},
        {"id": 2, "vector": {"dense": [0.0, 1.0, 0.0], "text": {"indices": [7], "values": [2.0]}}},
        {"id": 3, "vector": {"dense": [0.7, 0.7, 0.0], "text": {"indices": [3], "values": [1.0]}}},
    ])))
    .unwrap();

    let sparse = col
        .query(from(
            json!({"query": {"indices": [7], "values": [1.0]}, "using": "text"}),
        ))
        .unwrap();
    assert_eq!(ids(&sparse), [json!(2), json!(1)]);

    let fused = col
        .query(from(json!({
            "prefetch": [
                {"query": [1.0, 0.0, 0.0], "using": "dense", "limit": 3},
                {"query": {"indices": [7], "values": [1.0]}, "using": "text", "limit": 3}
            ],
            "query": {"fusion": "rrf"},
            "limit": 3
        })))
        .unwrap();
    assert_eq!(
        ids(&fused)[0],
        json!(1),
        "id 1 ranks well in both prefetches"
    );
}

/// Crash recovery: a child process writes without flushing and aborts; the parent must see
/// every acknowledged write after reopening (WAL replay).
#[test]
fn crash_recovery() {
    if let Ok(dir) = std::env::var("NEOQ_CRASH_CHILD") {
        let db = Database::open_with(
            &dir,
            Options {
                durability: Durability::Manual,
                auto_optimize: false,
                ..Options::default()
            },
        )
        .unwrap();
        let col = db
            .create_collection(
                "c",
                from(json!({"vectors": {"size": 2, "distance": "Dot"}})),
            )
            .unwrap();
        for i in 0..500u64 {
            col.upsert(from(json!([{"id": i, "vector": [1.0, i as f32]}])))
                .unwrap();
        }
        std::process::abort(); // no flush, no Drop
    }

    let dir = tempfile::tempdir().unwrap();
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "crash_recovery", "--nocapture"])
        .env("NEOQ_CRASH_CHILD", dir.path())
        .status()
        .unwrap();
    assert!(!status.success(), "child should have aborted");

    let db = Database::open(dir.path()).unwrap();
    let col = db.collection("c").unwrap();
    assert_eq!(col.count(CountRequest::default()).unwrap(), 500);
}
