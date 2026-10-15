//! # neoquadrant
//!
//! Qdrant's vector search engine as an in-process library — open a directory, no server.
//!
//! The engine is Qdrant itself ([Qdrant Edge](https://github.com/qdrant/qdrant/tree/master/lib/edge):
//! the same segment, HNSW, payload index, quantization and WAL code the server runs). This crate
//! adds what an embedded database needs on top: a database of many collections, a process lock,
//! background indexing, a durability policy, and a JSON API shaped like Qdrant's REST API.
//!
//! ```no_run
//! use neoquadrant::{Database, requests::*};
//! use serde_json::json;
//!
//! let db = Database::open("./my-vectors")?;
//! let docs = db.get_or_create_collection(
//!     "docs",
//!     serde_json::from_value(json!({"vectors": {"size": 4, "distance": "Cosine"}}))?,
//! )?;
//! docs.upsert(serde_json::from_value(json!([
//!     {"id": 1, "vector": [0.1, 0.2, 0.3, 0.4], "payload": {"lang": "en"}},
//! ]))?)?;
//! let hits = docs.query(serde_json::from_value(json!({
//!     "query": [0.1, 0.2, 0.3, 0.4],
//!     "filter": {"must": [{"key": "lang", "match": {"value": "en"}}]},
//!     "limit": 5,
//! }))?)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod collection;
mod db;
mod error;
mod options;
pub mod requests;

pub use collection::Collection;
pub use db::Database;
pub use error::{Error, Result};
pub use options::{Durability, Options};

/// The underlying Qdrant Edge engine, for typed access to filters, configs and query types.
pub use edge as engine;

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<edge::EdgeShard>();
    assert_send_sync::<Database>();
    assert_send_sync::<Collection>();
};
