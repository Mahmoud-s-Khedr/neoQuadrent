use std::collections::{BTreeMap, HashMap};
use std::fs::{self, File, TryLockError};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::{Arc, LazyLock, Weak};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use edge::{EdgeConfig, EdgeShard, ResourceBudget};
use parking_lot::{Condvar, Mutex, RwLock};

use crate::collection::{Collection, CollectionInner};
use crate::options::{Durability, Options};
use crate::requests::CreateCollection;
use crate::{Error, Result};

const COLLECTIONS_DIR: &str = "collections";
const LOCK_FILE: &str = "LOCK";

/// Open databases in this process, so opening the same path twice shares one handle
/// (like SQLite connections to one file) instead of failing on the file lock.
static REGISTRY: LazyLock<Mutex<HashMap<PathBuf, Weak<DbInner>>>> = LazyLock::new(Default::default);

/// An embedded Qdrant database: a directory holding any number of collections.
///
/// Cheap to clone; all clones share the same state. The database is closed when the last
/// handle is dropped (or explicitly with [`Database::close`]).
#[derive(Clone)]
pub struct Database {
    inner: Arc<DbInner>,
}

/// State shared with collections and the background worker (must not own collections).
pub(crate) struct Ctx {
    pub(crate) options: Options,
    pub(crate) budget: ResourceBudget,
    worker: Mutex<WorkerState>,
    wake: Condvar,
}

#[derive(Default)]
struct WorkerState {
    stop: bool,
    pending: bool,
}

impl Ctx {
    pub(crate) fn notify(&self) {
        let mut st = self.worker.lock();
        st.pending = true;
        self.wake.notify_one();
    }
}

type Collections = Arc<RwLock<BTreeMap<String, Arc<CollectionInner>>>>;

struct DbInner {
    root: PathBuf,
    ctx: Arc<Ctx>,
    collections: Collections,
    worker: Mutex<Option<JoinHandle<()>>>,
    lock: Mutex<Option<File>>,
    _temp: Option<tempfile::TempDir>,
}

impl Database {
    /// Open (or create) a database directory with default [`Options`].
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_with(path, Options::default())
    }

    /// Open (or create) a database directory. If this process already has it open, the existing
    /// handle is returned and `options` are ignored.
    pub fn open_with(path: impl AsRef<Path>, options: Options) -> Result<Self> {
        fs::create_dir_all(path.as_ref())?;
        let root = fs::canonicalize(path.as_ref())?;

        let mut registry = REGISTRY.lock();
        registry.retain(|_, db| db.strong_count() > 0);
        if let Some(inner) = registry.get(&root).and_then(Weak::upgrade) {
            return Ok(Database { inner });
        }
        let inner = Arc::new(DbInner::open(root.clone(), options, None)?);
        registry.insert(root, Arc::downgrade(&inner));
        Ok(Database { inner })
    }

    /// A throwaway database in a temporary directory, deleted when closed (like `:memory:`).
    pub fn open_temp() -> Result<Self> {
        Self::open_temp_with(Options {
            durability: Durability::Manual,
            ..Options::default()
        })
    }

    pub fn open_temp_with(options: Options) -> Result<Self> {
        let temp = tempfile::Builder::new().prefix("neoquadrant-").tempdir()?;
        let root = temp.path().to_path_buf();
        Ok(Database {
            inner: Arc::new(DbInner::open(root, options, Some(temp))?),
        })
    }

    pub fn path(&self) -> &Path {
        &self.inner.root
    }

    /// Create a collection from a Qdrant-style config,
    /// e.g. `{"vectors": {"size": 384, "distance": "Cosine"}}`.
    pub fn create_collection(&self, name: &str, config: CreateCollection) -> Result<Collection> {
        self.create_collection_with(name, config.into_edge_config()?)
    }

    /// Create a collection from a full engine config (named/sparse vectors, quantization, …).
    pub fn create_collection_with(&self, name: &str, config: EdgeConfig) -> Result<Collection> {
        validate_name(name)?;
        let mut cols = self.inner.collections.write();
        self.inner.check_open()?;
        if cols.contains_key(name) {
            return Err(Error::CollectionExists(name.to_owned()));
        }
        let path = self.inner.root.join(COLLECTIONS_DIR).join(name);
        if path.exists() {
            // Leftover from an interrupted drop/create.
            fs::remove_dir_all(&path)?;
        }
        let shard = EdgeShard::new(&path, config)?;
        let inner = Arc::new(CollectionInner::new(
            name.to_owned(),
            path,
            shard,
            self.inner.ctx.clone(),
        ));
        cols.insert(name.to_owned(), inner.clone());
        Ok(Collection { inner })
    }

    pub fn collection(&self, name: &str) -> Result<Collection> {
        self.inner
            .collections
            .read()
            .get(name)
            .map(|inner| Collection {
                inner: inner.clone(),
            })
            .ok_or_else(|| Error::CollectionNotFound(name.to_owned()))
    }

    /// Return the collection, creating it with `config` if it does not exist yet.
    pub fn get_or_create_collection(
        &self,
        name: &str,
        config: CreateCollection,
    ) -> Result<Collection> {
        match self.collection(name) {
            Err(Error::CollectionNotFound(_)) => match self.create_collection(name, config) {
                Err(Error::CollectionExists(_)) => self.collection(name),
                other => other,
            },
            other => other,
        }
    }

    pub fn has_collection(&self, name: &str) -> bool {
        self.inner.collections.read().contains_key(name)
    }

    pub fn list_collections(&self) -> Vec<String> {
        self.inner.collections.read().keys().cloned().collect()
    }

    /// Delete a collection and its files. Returns `false` if it did not exist.
    /// Existing handles to it fail with [`Error::Closed`] afterwards.
    pub fn drop_collection(&self, name: &str) -> Result<bool> {
        let Some(col) = self.inner.collections.write().remove(name) else {
            return Ok(false);
        };
        col.close();
        fs::remove_dir_all(&col.path)?;
        Ok(true)
    }

    /// Flush every collection to disk.
    pub fn flush(&self) -> Result<()> {
        for col in self.inner.collections.read().values() {
            col.flush_now()?;
        }
        Ok(())
    }

    /// Close the database: stop background work, flush, and release the lock.
    /// Every handle (including clones and collections) fails with [`Error::Closed`] afterwards.
    pub fn close(self) -> Result<()> {
        self.inner.shutdown();
        Ok(())
    }
}

impl std::fmt::Debug for Database {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Database")
            .field("path", &self.inner.root)
            .finish()
    }
}

impl DbInner {
    fn open(root: PathBuf, options: Options, temp: Option<tempfile::TempDir>) -> Result<Self> {
        let lock = File::options()
            .create(true)
            .truncate(false)
            .write(true)
            .open(root.join(LOCK_FILE))?;
        match lock.try_lock() {
            Ok(()) => {}
            Err(TryLockError::WouldBlock) => return Err(Error::Locked(root)),
            Err(TryLockError::Error(e)) => return Err(e.into()),
        }

        let threads = options.indexing_threads();
        let ctx = Arc::new(Ctx {
            options,
            budget: ResourceBudget::new(threads, threads),
            worker: Mutex::new(WorkerState::default()),
            wake: Condvar::new(),
        });

        let dir = root.join(COLLECTIONS_DIR);
        fs::create_dir_all(&dir)?;
        let mut collections = BTreeMap::new();
        for entry in fs::read_dir(&dir)? {
            let entry = entry?;
            if !entry.file_type()?.is_dir() {
                continue;
            }
            let name = entry.file_name().to_string_lossy().into_owned();
            if validate_name(&name).is_err() {
                continue;
            }
            let shard = EdgeShard::load(&entry.path(), None)?;
            collections.insert(
                name.clone(),
                Arc::new(CollectionInner::new(name, entry.path(), shard, ctx.clone())),
            );
        }
        let collections: Collections = Arc::new(RwLock::new(collections));

        let worker = {
            let (ctx, cols) = (ctx.clone(), collections.clone());
            std::thread::Builder::new()
                .name("neoquadrant-bg".into())
                .spawn(move || background_worker(&ctx, &cols))?
        };

        Ok(DbInner {
            root,
            ctx,
            collections,
            worker: Mutex::new(Some(worker)),
            lock: Mutex::new(Some(lock)),
            _temp: temp,
        })
    }

    fn check_open(&self) -> Result<()> {
        if self.lock.lock().is_none() {
            return Err(Error::Closed(self.root.display().to_string()));
        }
        Ok(())
    }

    fn shutdown(&self) {
        {
            let mut st = self.ctx.worker.lock();
            st.stop = true;
            self.ctx.wake.notify_all();
        }
        if let Some(handle) = self.worker.lock().take() {
            let _ = handle.join();
        }
        for col in self.collections.read().values() {
            col.close(); // EdgeShard flushes on drop
        }
        if let Some(lock) = self.lock.lock().take() {
            let _ = lock.unlock();
        }
        REGISTRY.lock().remove(&self.root);
    }
}

impl Drop for DbInner {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Runs Qdrant's optimizers once a collection has been idle for `optimize_debounce`, and flushes
/// on the `Durability::Periodic` interval.
fn background_worker(ctx: &Ctx, collections: &RwLock<BTreeMap<String, Arc<CollectionInner>>>) {
    let opts = &ctx.options;
    let flush_every = match opts.durability {
        Durability::Periodic(d) => Some(d),
        _ => None,
    };
    let mut last_flush = Instant::now();

    loop {
        let snapshot: Vec<Arc<CollectionInner>> = collections.read().values().cloned().collect();

        // Next time anything is due; `None` = nothing to do until a write arrives.
        let mut next_due: Option<Duration> = None;
        let mut due = |d: Duration| next_due = Some(next_due.map_or(d, |n| n.min(d)));

        if let Some(every) = flush_every {
            let elapsed = last_flush.elapsed();
            if elapsed >= every {
                for col in snapshot
                    .iter()
                    .filter(|c| c.needs_flush.load(Ordering::Acquire))
                {
                    if let Err(e) = col.flush_now() {
                        log_error(&col.name, "flush", &e);
                    }
                }
                last_flush = Instant::now();
            }
            if snapshot
                .iter()
                .any(|c| c.needs_flush.load(Ordering::Acquire))
            {
                due(every.saturating_sub(last_flush.elapsed()));
            }
        }

        if opts.auto_optimize {
            for col in snapshot
                .iter()
                .filter(|c| c.needs_optimize.load(Ordering::Acquire))
            {
                let idle = col.last_write.lock().elapsed();
                if idle >= opts.optimize_debounce {
                    if ctx.worker.lock().stop {
                        return;
                    }
                    if let Err(e) = col.optimize_now()
                        && !matches!(e, Error::Closed(_))
                    {
                        log_error(&col.name, "optimize", &e);
                    }
                } else {
                    due(opts.optimize_debounce - idle);
                }
            }
        }
        drop(snapshot);

        let mut st = ctx.worker.lock();
        if st.stop {
            return;
        }
        if !st.pending {
            match next_due {
                Some(d) => {
                    ctx.wake.wait_for(&mut st, d.max(Duration::from_millis(1)));
                }
                None => ctx.wake.wait(&mut st),
            }
        }
        st.pending = false;
        if st.stop {
            return;
        }
    }
}

fn log_error(collection: &str, what: &str, e: &Error) {
    eprintln!("neoquadrant: background {what} of collection `{collection}` failed: {e}");
}

fn validate_name(name: &str) -> Result<()> {
    let ok = !name.is_empty()
        && name.len() <= 255
        && !name.starts_with('.')
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.'));
    if ok {
        Ok(())
    } else {
        Err(Error::InvalidArgument(format!(
            "invalid collection name `{name}` (use letters, digits, `_`, `-`, `.`)"
        )))
    }
}
