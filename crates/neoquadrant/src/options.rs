use std::time::Duration;

/// When writes are made durable on disk.
///
/// Every write is appended to Qdrant's write-ahead log before it is applied, so a crashed
/// process always recovers to a consistent state. This setting controls how much of the most
/// recent work can be lost if the *machine* loses power.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Durability {
    /// Flush after every write call (SQLite `synchronous=FULL`). The default. Costs a few hundred
    /// microseconds per call on an SSD, so batch points into one `upsert` when loading data.
    Full,
    /// Flush in the background at this interval (SQLite `synchronous=NORMAL`-like).
    Periodic(Duration),
    /// Only flush on [`Collection::flush`](crate::Collection::flush), optimization, and close.
    Manual,
}

/// Options for [`Database::open`](crate::Database::open).
#[derive(Clone, Debug)]
pub struct Options {
    pub durability: Durability,
    /// Run Qdrant's optimizers (HNSW indexing, segment merging, vacuum) on a background thread
    /// after writes settle. Disable to call [`Collection::optimize`](crate::Collection::optimize)
    /// yourself.
    pub auto_optimize: bool,
    /// How long a collection must be idle (no writes) before background optimization starts.
    pub optimize_debounce: Duration,
    /// CPU threads the background optimizer may use. `0` = all cores but one.
    pub max_indexing_threads: usize,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            durability: Durability::Full,
            auto_optimize: true,
            optimize_debounce: Duration::from_millis(500),
            max_indexing_threads: 0,
        }
    }
}

impl Options {
    pub(crate) fn indexing_threads(&self) -> usize {
        if self.max_indexing_threads > 0 {
            return self.max_indexing_threads;
        }
        std::thread::available_parallelism().map_or(1, |n| n.get().saturating_sub(1).max(1))
    }
}
