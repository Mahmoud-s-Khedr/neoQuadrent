use std::path::PathBuf;

use edge::OperationError;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Error raised by the Qdrant engine (wrong dimension, missing vector name, storage error, …).
    #[error(transparent)]
    Engine(#[from] OperationError),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("database at {} is locked by another process", .0.display())]
    Locked(PathBuf),

    #[error("collection `{0}` already exists")]
    CollectionExists(String),

    #[error("collection `{0}` not found")]
    CollectionNotFound(String),

    /// The collection was dropped, or its database was closed.
    #[error("collection `{0}` is closed")]
    Closed(String),

    #[error("invalid argument: {0}")]
    InvalidArgument(String),
}

impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Error::InvalidArgument(e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, Error>;
