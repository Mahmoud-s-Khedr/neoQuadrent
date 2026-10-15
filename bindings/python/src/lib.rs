//! Native layer of the Python binding (`neoquadrant._native`). The public, documented API is
//! the pure-Python wrapper in `python/neoquadrant/__init__.py`.
//!
//! Requests arrive as plain Python dicts/lists (numpy arrays are converted by the wrapper),
//! are converted with `pythonize`, and the engine call runs with the GIL released so other
//! Python threads keep running during searches, upserts and optimization.

use std::time::Duration;

use neoquadrant::requests::*;
use neoquadrant::{Collection, Database, Durability, Options};
use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pythonize::{depythonize, pythonize};
use serde::Serialize;
use serde::de::DeserializeOwned;

create_exception!(
    _native,
    NeoquadrantError,
    PyException,
    "Base class for neoquadrant errors."
);
create_exception!(
    _native,
    EngineError,
    NeoquadrantError,
    "Error raised by the Qdrant engine (wrong dimension, unknown vector name, ...)."
);
create_exception!(
    _native,
    LockedError,
    NeoquadrantError,
    "The database directory is locked by another process."
);
create_exception!(
    _native,
    CollectionExistsError,
    NeoquadrantError,
    "A collection with this name already exists."
);
create_exception!(
    _native,
    CollectionNotFoundError,
    NeoquadrantError,
    "No collection with this name."
);
create_exception!(
    _native,
    ClosedError,
    NeoquadrantError,
    "The collection was dropped or the database was closed."
);
create_exception!(
    _native,
    InvalidArgumentError,
    NeoquadrantError,
    "A request is malformed."
);

fn to_py(e: neoquadrant::Error) -> PyErr {
    use neoquadrant::Error::*;
    let msg = e.to_string();
    match e {
        Engine(_) => EngineError::new_err(msg),
        Io(_) => NeoquadrantError::new_err(msg),
        Locked(_) => LockedError::new_err(msg),
        CollectionExists(_) => CollectionExistsError::new_err(msg),
        CollectionNotFound(_) => CollectionNotFoundError::new_err(msg),
        Closed(_) => ClosedError::new_err(msg),
        InvalidArgument(_) => InvalidArgumentError::new_err(msg),
    }
}

trait OrPy<T> {
    fn or_py(self) -> PyResult<T>;
}

impl<T> OrPy<T> for neoquadrant::Result<T> {
    fn or_py(self) -> PyResult<T> {
        self.map_err(to_py)
    }
}

fn parse<T: DeserializeOwned>(obj: &Bound<'_, PyAny>) -> PyResult<T> {
    depythonize(obj).map_err(|e| InvalidArgumentError::new_err(format!("invalid argument: {e}")))
}

fn out<T: Serialize>(py: Python<'_>, value: &T) -> PyResult<Py<PyAny>> {
    Ok(pythonize(py, value)
        .map_err(|e| EngineError::new_err(e.to_string()))?
        .unbind())
}

#[pyclass(module = "neoquadrant._native", frozen)]
struct NativeDatabase {
    db: std::sync::Mutex<Option<Database>>,
}

impl NativeDatabase {
    fn db(&self) -> PyResult<Database> {
        self.db
            .lock()
            .unwrap()
            .clone()
            .ok_or_else(|| ClosedError::new_err("database is closed"))
    }
}

fn options(
    durability: &str,
    flush_interval: f64,
    auto_optimize: bool,
    optimize_debounce: f64,
    max_indexing_threads: usize,
) -> PyResult<Options> {
    let durability = match durability {
        "full" => Durability::Full,
        "periodic" => Durability::Periodic(Duration::from_secs_f64(flush_interval)),
        "manual" => Durability::Manual,
        other => {
            return Err(InvalidArgumentError::new_err(format!(
                "durability must be 'full', 'periodic' or 'manual', got {other:?}"
            )));
        }
    };
    Ok(Options {
        durability,
        auto_optimize,
        optimize_debounce: Duration::from_secs_f64(optimize_debounce),
        max_indexing_threads,
    })
}

#[pymethods]
impl NativeDatabase {
    #[staticmethod]
    #[pyo3(signature = (path, durability, flush_interval, auto_optimize, optimize_debounce, max_indexing_threads))]
    fn open(
        py: Python<'_>,
        path: Option<std::path::PathBuf>,
        durability: &str,
        flush_interval: f64,
        auto_optimize: bool,
        optimize_debounce: f64,
        max_indexing_threads: usize,
    ) -> PyResult<Self> {
        let opts = options(
            durability,
            flush_interval,
            auto_optimize,
            optimize_debounce,
            max_indexing_threads,
        )?;
        let db = py.detach(|| match path {
            Some(path) => Database::open_with(path, opts),
            None => Database::open_temp_with(opts),
        });
        Ok(NativeDatabase {
            db: std::sync::Mutex::new(Some(db.or_py()?)),
        })
    }

    #[getter]
    fn path(&self) -> PyResult<std::path::PathBuf> {
        Ok(self.db()?.path().to_path_buf())
    }

    fn create_collection(
        &self,
        name: &str,
        config: &Bound<'_, PyAny>,
    ) -> PyResult<NativeCollection> {
        let config: CreateCollection = parse(config)?;
        Ok(NativeCollection {
            col: self.db()?.create_collection(name, config).or_py()?,
        })
    }

    fn get_or_create_collection(
        &self,
        name: &str,
        config: &Bound<'_, PyAny>,
    ) -> PyResult<NativeCollection> {
        let config: CreateCollection = parse(config)?;
        Ok(NativeCollection {
            col: self.db()?.get_or_create_collection(name, config).or_py()?,
        })
    }

    fn collection(&self, name: &str) -> PyResult<NativeCollection> {
        Ok(NativeCollection {
            col: self.db()?.collection(name).or_py()?,
        })
    }

    fn has_collection(&self, name: &str) -> PyResult<bool> {
        Ok(self.db()?.has_collection(name))
    }

    fn list_collections(&self) -> PyResult<Vec<String>> {
        Ok(self.db()?.list_collections())
    }

    fn drop_collection(&self, py: Python<'_>, name: &str) -> PyResult<bool> {
        let db = self.db()?;
        py.detach(|| db.drop_collection(name)).or_py()
    }

    fn flush(&self, py: Python<'_>) -> PyResult<()> {
        let db = self.db()?;
        py.detach(|| db.flush()).or_py()
    }

    fn close(&self, py: Python<'_>) -> PyResult<()> {
        let db = self.db.lock().unwrap().take();
        match db {
            Some(db) => py.detach(|| db.close()).or_py(),
            None => Ok(()),
        }
    }
}

#[pyclass(module = "neoquadrant._native", frozen)]
struct NativeCollection {
    col: Collection,
}

#[pymethods]
impl NativeCollection {
    #[getter]
    fn name(&self) -> &str {
        self.col.name()
    }

    fn upsert(&self, py: Python<'_>, points: &Bound<'_, PyAny>) -> PyResult<()> {
        let points: Vec<PointStruct> = parse(points)?;
        py.detach(|| self.col.upsert(points)).or_py()
    }

    fn query(&self, py: Python<'_>, request: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let request: QueryRequest = parse(request)?;
        let hits = py.detach(|| self.col.query(request)).or_py()?;
        out(py, &hits)
    }

    fn retrieve(&self, py: Python<'_>, request: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let request: RetrieveRequest = parse(request)?;
        let records = py.detach(|| self.col.retrieve(request)).or_py()?;
        out(py, &records)
    }

    fn scroll(&self, py: Python<'_>, request: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let request: ScrollRequest = parse(request)?;
        let page = py.detach(|| self.col.scroll(request)).or_py()?;
        out(py, &page)
    }

    fn count(&self, py: Python<'_>, request: &Bound<'_, PyAny>) -> PyResult<usize> {
        let request: CountRequest = parse(request)?;
        py.detach(|| self.col.count(request)).or_py()
    }

    fn facet(&self, py: Python<'_>, request: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let request: FacetRequest = parse(request)?;
        let hits = py.detach(|| self.col.facet(request)).or_py()?;
        out(py, &hits)
    }

    fn delete(&self, py: Python<'_>, selector: &Bound<'_, PyAny>) -> PyResult<()> {
        let selector: PointsSelector = parse(selector)?;
        py.detach(|| self.col.delete(selector)).or_py()
    }

    fn set_payload(&self, py: Python<'_>, request: &Bound<'_, PyAny>) -> PyResult<()> {
        let request: SetPayloadRequest = parse(request)?;
        py.detach(|| self.col.set_payload(request)).or_py()
    }

    fn overwrite_payload(&self, py: Python<'_>, request: &Bound<'_, PyAny>) -> PyResult<()> {
        let request: SetPayloadRequest = parse(request)?;
        py.detach(|| self.col.overwrite_payload(request)).or_py()
    }

    fn delete_payload(&self, py: Python<'_>, request: &Bound<'_, PyAny>) -> PyResult<()> {
        let request: DeletePayloadRequest = parse(request)?;
        py.detach(|| self.col.delete_payload(request)).or_py()
    }

    fn clear_payload(&self, py: Python<'_>, selector: &Bound<'_, PyAny>) -> PyResult<()> {
        let selector: PointsSelector = parse(selector)?;
        py.detach(|| self.col.clear_payload(selector)).or_py()
    }

    fn create_payload_index(&self, py: Python<'_>, request: &Bound<'_, PyAny>) -> PyResult<()> {
        let request: CreateFieldIndex = parse(request)?;
        py.detach(|| self.col.create_payload_index(request)).or_py()
    }

    fn delete_payload_index(&self, py: Python<'_>, field_name: &str) -> PyResult<()> {
        py.detach(|| self.col.delete_payload_index(field_name))
            .or_py()
    }

    fn optimize(&self, py: Python<'_>) -> PyResult<bool> {
        py.detach(|| self.col.optimize()).or_py()
    }

    fn flush(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.col.flush()).or_py()
    }

    fn info(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let info = py.detach(|| self.col.info()).or_py()?;
        out(py, &info)
    }
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add_class::<NativeDatabase>()?;
    m.add_class::<NativeCollection>()?;
    m.add("NeoquadrantError", py.get_type::<NeoquadrantError>())?;
    m.add("EngineError", py.get_type::<EngineError>())?;
    m.add("LockedError", py.get_type::<LockedError>())?;
    m.add(
        "CollectionExistsError",
        py.get_type::<CollectionExistsError>(),
    )?;
    m.add(
        "CollectionNotFoundError",
        py.get_type::<CollectionNotFoundError>(),
    )?;
    m.add("ClosedError", py.get_type::<ClosedError>())?;
    m.add(
        "InvalidArgumentError",
        py.get_type::<InvalidArgumentError>(),
    )?;
    Ok(())
}
