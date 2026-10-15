//! `cargo run --release --example bench -- [points] [dim]`
//!
//! Measures single-point write latency per durability mode, bulk insert throughput,
//! time until the background optimizer has indexed everything, query QPS, and recall@10
//! against exact search.

use std::time::{Duration, Instant};

use neoquadrant::requests::*;
use neoquadrant::{Database, Durability, Options};
use serde_json::json;

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

/// Gaussian-ish clusters around random centroids — closer to real embedding distributions.
fn clustered(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let centroids = vectors(n / 100 + 1, dim, seed ^ 0x9e37);
    let noise = vectors(n, dim, seed);
    noise
        .into_iter()
        .enumerate()
        .map(|(i, e)| {
            let c = &centroids[i % centroids.len()];
            c.iter().zip(e).map(|(c, e)| c + 0.35 * e).collect()
        })
        .collect()
}

fn point(id: usize, v: &[f32]) -> PointStruct {
    serde_json::from_value(json!({"id": id, "vector": v, "payload": {"group": id % 10}})).unwrap()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .filter_map(|a| a.parse().ok())
        .collect();
    let n = args.first().copied().unwrap_or(100_000);
    let dim = args.get(1).copied().unwrap_or(128);
    let config: CreateCollection =
        serde_json::from_value(json!({"vectors": {"size": dim, "distance": "Cosine"}}))?;

    println!("== single-point write latency ({dim}d, 1000 writes)");
    let one = vectors(1000, dim, 1);
    for (label, durability) in [
        ("Full (flush every write)", Durability::Full),
        (
            "Periodic(200ms)",
            Durability::Periodic(Duration::from_millis(200)),
        ),
        ("Manual", Durability::Manual),
    ] {
        let dir = tempfile::tempdir()?;
        let db = Database::open_with(
            dir.path(),
            Options {
                durability,
                auto_optimize: false,
                ..Options::default()
            },
        )?;
        let col = db.create_collection("w", config.clone())?;
        let t = Instant::now();
        for (i, v) in one.iter().enumerate() {
            col.upsert(vec![point(i, v)])?;
        }
        let per = t.elapsed() / one.len() as u32;
        println!("  {label:<26} {per:>10.1?} / write");
    }

    println!("== bulk insert {n} x {dim}d (batches of 1000), background optimizer on");
    let db = Database::open_temp_with(Options {
        durability: Durability::Manual,
        ..Options::default()
    })?;
    let col = db.create_collection("bench", config)?;
    col.create_payload_index(serde_json::from_value(
        json!({"field_name": "group", "field_schema": "integer"}),
    )?)?;
    let uniform = std::env::var("BENCH_UNIFORM").is_ok();
    println!(
        "  data: {}",
        if uniform {
            "uniform random (worst case for ANN)"
        } else {
            "clustered"
        }
    );
    let data = if uniform {
        vectors(n, dim, 42)
    } else {
        clustered(n, dim, 42)
    };
    let t = Instant::now();
    for (b, chunk) in data.chunks(1000).enumerate() {
        col.upsert(
            chunk
                .iter()
                .enumerate()
                .map(|(i, v)| point(b * 1000 + i, v))
                .collect(),
        )?;
    }
    let insert = t.elapsed();
    println!(
        "  inserted in {insert:.2?} ({:.0} points/s)",
        n as f64 / insert.as_secs_f64()
    );

    let t = Instant::now();
    loop {
        let info = col.info()?;
        if info.indexed_vectors_count >= n * 95 / 100 {
            println!(
                "  indexed {} vectors in {:.2?} after insert ({} segments)",
                info.indexed_vectors_count,
                t.elapsed(),
                info.segments_count
            );
            break;
        }
        if t.elapsed() > Duration::from_secs(600) {
            println!("  WARNING: still not indexed after 10 min: {info:?}");
            break;
        }
        std::thread::sleep(Duration::from_millis(100));
    }

    let queries = if uniform {
        vectors(500, dim, 7)
    } else {
        clustered(500, dim, 7)
    };
    let t = Instant::now();
    let approx: Vec<_> = queries
        .iter()
        .map(|q| col.query(QueryRequest::nearest(q.clone(), 10)))
        .collect::<Result<_, _>>()?;
    let qps = queries.len() as f64 / t.elapsed().as_secs_f64();

    let t = Instant::now();
    let filtered = queries
        .iter()
        .map(|q| {
            col.query(serde_json::from_value(json!({"query": q, "limit": 10, "filter": {"must": [{"key": "group", "match": {"value": 3}}]}}))?)
        })
        .collect::<Result<Vec<_>, neoquadrant::Error>>()?;
    let fqps = queries.len() as f64 / t.elapsed().as_secs_f64();
    assert!(filtered.iter().all(|hits| hits.len() == 10));

    println!("  search: {qps:.0} QPS, filtered (10% selectivity, indexed field): {fqps:.0} QPS");
    drop(approx);

    let exact: Vec<_> = queries
        .iter()
        .map(|q| {
            col.query(serde_json::from_value(
                json!({"query": q, "limit": 10, "params": {"exact": true}}),
            )?)
        })
        .collect::<Result<_, neoquadrant::Error>>()?;
    for ef in [None, Some(128), Some(256), Some(512)] {
        let params = ef.map_or(json!(null), |ef| json!({"hnsw_ef": ef}));
        let t = Instant::now();
        let mut found = 0;
        for (q, ex) in queries.iter().zip(&exact) {
            let hits = col.query(serde_json::from_value(
                json!({"query": q, "limit": 10, "params": params}),
            )?)?;
            found += hits
                .iter()
                .filter(|h| ex.iter().any(|e| e.id == h.id))
                .count();
        }
        let label = ef.map_or("default".to_string(), |e| e.to_string());
        println!(
            "  hnsw_ef={label:<8} recall@10 {:.4}  ({:.0} QPS)",
            found as f64 / (queries.len() * 10) as f64,
            queries.len() as f64 / t.elapsed().as_secs_f64()
        );
    }
    Ok(())
}
