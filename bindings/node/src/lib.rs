//! Native layer of the Node.js binding. Requests and results cross the boundary as JSON
//! strings: V8's `JSON.stringify`/`JSON.parse` and serde are both fast, there is no per-call
//! node limit (napi's `serde_json::Value` conversion caps at 1M nodes, which a single bulk
//! upsert of embeddings exceeds), and the JS wrapper in `index.js` hides it entirely.

use std::time::Duration;

use napi::bindgen_prelude::*;
use napi_derive::napi;
use neoquadrant::requests::*;
use neoquadrant::{Collection, Database, Durability, Options};
use serde::Serialize;
use serde::de::DeserializeOwned;

fn to_napi(e: neoquadrant::Error) -> napi::Error {
    use neoquadrant::Error::*;
    let code = match &e {
        Engine(_) => "NEOQ_ENGINE",
        Io(_) => "NEOQ_IO",
        Locked(_) => "NEOQ_LOCKED",
        CollectionExists(_) => "NEOQ_EXISTS",
        CollectionNotFound(_) => "NEOQ_NOT_FOUND",
        Closed(_) => "NEOQ_CLOSED",
        InvalidArgument(_) => "NEOQ_INVALID_ARGUMENT",
    };
    napi::Error::from_reason(format!("{code}|{e}"))
}

fn parse<T: DeserializeOwned>(json: &str) -> napi::Result<T> {
    serde_json::from_str(json).map_err(|e| {
        napi::Error::from_reason(format!("NEOQ_INVALID_ARGUMENT|invalid argument: {e}"))
    })
}

fn to_json<T: Serialize>(value: &T) -> napi::Result<String> {
    serde_json::to_string(value).map_err(|e| napi::Error::from_reason(format!("NEOQ_ENGINE|{e}")))
}

trait OrNapi<T> {
    fn or_napi(self) -> napi::Result<T>;
}

impl<T> OrNapi<T> for neoquadrant::Result<T> {
    fn or_napi(self) -> napi::Result<T> {
        self.map_err(to_napi)
    }
}

#[napi(object)]
#[derive(Default)]
pub struct OpenOptions {
    /// "full" (default) | "periodic" | "manual"
    pub durability: Option<String>,
    /// Flush interval for `durability: "periodic"` (default 200 ms).
    pub flush_interval_ms: Option<u32>,
    pub auto_optimize: Option<bool>,
    pub optimize_debounce_ms: Option<u32>,
    pub max_indexing_threads: Option<u32>,
}

impl OpenOptions {
    fn into_options(self, default_durability: Durability) -> napi::Result<Options> {
        let d = Options::default();
        let durability = match self.durability.as_deref() {
            None => default_durability,
            Some("full") => Durability::Full,
            Some("periodic") => Durability::Periodic(Duration::from_millis(
                self.flush_interval_ms.unwrap_or(200).into(),
            )),
            Some("manual") => Durability::Manual,
            Some(other) => {
                return Err(napi::Error::from_reason(format!(
                    "NEOQ_INVALID_ARGUMENT|durability must be \"full\", \"periodic\" or \"manual\", got {other:?}"
                )));
            }
        };
        Ok(Options {
            durability,
            auto_optimize: self.auto_optimize.unwrap_or(d.auto_optimize),
            optimize_debounce: self
                .optimize_debounce_ms
                .map_or(d.optimize_debounce, |ms| Duration::from_millis(ms.into())),
            max_indexing_threads: self
                .max_indexing_threads
                .map_or(d.max_indexing_threads, |n| n as usize),
        })
    }
}

#[napi]
pub struct NativeDatabase {
    db: Option<Database>,
}

impl NativeDatabase {
    fn db(&self) -> napi::Result<&Database> {
        self.db
            .as_ref()
            .ok_or_else(|| napi::Error::from_reason("NEOQ_CLOSED|database is closed"))
    }
}

#[napi]
impl NativeDatabase {
    #[napi(factory)]
    pub fn open(path: String, options: Option<OpenOptions>) -> napi::Result<Self> {
        let options = options.unwrap_or_default().into_options(Durability::Full)?;
        Ok(NativeDatabase {
            db: Some(Database::open_with(path, options).or_napi()?),
        })
    }

    #[napi(factory)]
    pub fn open_temp(options: Option<OpenOptions>) -> napi::Result<Self> {
        let options = options
            .unwrap_or_default()
            .into_options(Durability::Manual)?;
        Ok(NativeDatabase {
            db: Some(Database::open_temp_with(options).or_napi()?),
        })
    }

    #[napi(getter)]
    pub fn path(&self) -> napi::Result<String> {
        Ok(self.db()?.path().to_string_lossy().into_owned())
    }

    #[napi]
    pub fn create_collection(
        &self,
        name: String,
        config: String,
    ) -> napi::Result<NativeCollection> {
        let config: CreateCollection = parse(&config)?;
        Ok(NativeCollection {
            col: self.db()?.create_collection(&name, config).or_napi()?,
        })
    }

    #[napi]
    pub fn get_or_create_collection(
        &self,
        name: String,
        config: String,
    ) -> napi::Result<NativeCollection> {
        let config: CreateCollection = parse(&config)?;
        Ok(NativeCollection {
            col: self
                .db()?
                .get_or_create_collection(&name, config)
                .or_napi()?,
        })
    }

    #[napi]
    pub fn collection(&self, name: String) -> napi::Result<NativeCollection> {
        Ok(NativeCollection {
            col: self.db()?.collection(&name).or_napi()?,
        })
    }

    #[napi]
    pub fn has_collection(&self, name: String) -> napi::Result<bool> {
        Ok(self.db()?.has_collection(&name))
    }

    #[napi]
    pub fn list_collections(&self) -> napi::Result<Vec<String>> {
        Ok(self.db()?.list_collections())
    }

    #[napi]
    pub fn drop_collection(&self, name: String) -> napi::Result<bool> {
        self.db()?.drop_collection(&name).or_napi()
    }

    #[napi]
    pub fn flush(&self) -> napi::Result<()> {
        self.db()?.flush().or_napi()
    }

    #[napi]
    pub fn close(&mut self) -> napi::Result<()> {
        match self.db.take() {
            Some(db) => db.close().or_napi(),
            None => Ok(()),
        }
    }
}

#[napi]
pub struct NativeCollection {
    col: Collection,
}

/// Work that runs on the libuv thread pool so it does not block the event loop.
pub enum Job {
    Upsert(String),
    Query(String),
    Optimize,
}

pub struct CollectionTask {
    col: Collection,
    job: Option<Job>,
}

impl Task for CollectionTask {
    type Output = String;
    type JsValue = String;

    fn compute(&mut self) -> napi::Result<String> {
        match self.job.take().expect("task runs once") {
            Job::Upsert(json) => {
                self.col.upsert(parse(&json)?).or_napi()?;
                Ok("null".into())
            }
            Job::Query(json) => to_json(&self.col.query(parse(&json)?).or_napi()?),
            Job::Optimize => to_json(&self.col.optimize().or_napi()?),
        }
    }

    fn resolve(&mut self, _env: Env, output: String) -> napi::Result<String> {
        Ok(output)
    }
}

#[napi]
impl NativeCollection {
    fn task(&self, job: Job) -> AsyncTask<CollectionTask> {
        AsyncTask::new(CollectionTask {
            col: self.col.clone(),
            job: Some(job),
        })
    }

    #[napi(getter)]
    pub fn name(&self) -> String {
        self.col.name().to_owned()
    }

    #[napi]
    pub fn upsert(&self, points: String) -> napi::Result<()> {
        self.col.upsert(parse(&points)?).or_napi()
    }

    #[napi(ts_return_type = "Promise<string>")]
    pub fn upsert_async(&self, points: String) -> AsyncTask<CollectionTask> {
        self.task(Job::Upsert(points))
    }

    #[napi]
    pub fn query(&self, request: String) -> napi::Result<String> {
        to_json(&self.col.query(parse(&request)?).or_napi()?)
    }

    #[napi(ts_return_type = "Promise<string>")]
    pub fn query_async(&self, request: String) -> AsyncTask<CollectionTask> {
        self.task(Job::Query(request))
    }

    #[napi]
    pub fn retrieve(&self, request: String) -> napi::Result<String> {
        to_json(&self.col.retrieve(parse(&request)?).or_napi()?)
    }

    #[napi]
    pub fn scroll(&self, request: String) -> napi::Result<String> {
        to_json(&self.col.scroll(parse(&request)?).or_napi()?)
    }

    #[napi]
    pub fn count(&self, request: String) -> napi::Result<i64> {
        Ok(self.col.count(parse(&request)?).or_napi()? as i64)
    }

    #[napi]
    pub fn facet(&self, request: String) -> napi::Result<String> {
        to_json(&self.col.facet(parse(&request)?).or_napi()?)
    }

    #[napi]
    pub fn delete(&self, selector: String) -> napi::Result<()> {
        self.col.delete(parse(&selector)?).or_napi()
    }

    #[napi]
    pub fn set_payload(&self, request: String) -> napi::Result<()> {
        self.col.set_payload(parse(&request)?).or_napi()
    }

    #[napi]
    pub fn overwrite_payload(&self, request: String) -> napi::Result<()> {
        self.col.overwrite_payload(parse(&request)?).or_napi()
    }

    #[napi]
    pub fn delete_payload(&self, request: String) -> napi::Result<()> {
        self.col.delete_payload(parse(&request)?).or_napi()
    }

    #[napi]
    pub fn clear_payload(&self, selector: String) -> napi::Result<()> {
        self.col.clear_payload(parse(&selector)?).or_napi()
    }

    #[napi]
    pub fn create_payload_index(&self, request: String) -> napi::Result<()> {
        self.col.create_payload_index(parse(&request)?).or_napi()
    }

    #[napi]
    pub fn delete_payload_index(&self, field_name: String) -> napi::Result<()> {
        self.col.delete_payload_index(&field_name).or_napi()
    }

    #[napi]
    pub fn optimize(&self) -> napi::Result<bool> {
        self.col.optimize().or_napi()
    }

    #[napi(ts_return_type = "Promise<string>")]
    pub fn optimize_async(&self) -> AsyncTask<CollectionTask> {
        self.task(Job::Optimize)
    }

    #[napi]
    pub fn flush(&self) -> napi::Result<()> {
        self.col.flush().or_napi()
    }

    #[napi]
    pub fn info(&self) -> napi::Result<String> {
        to_json(&self.col.info().or_napi()?)
    }
}
