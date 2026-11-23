mod catalog;
mod pruning;
mod source;
mod storage;
mod types;

use std::sync::Arc;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use tokio::runtime::Runtime;

use catalog::Catalog;
use source::Source;

pub(crate) type Result<T> = std::result::Result<T, String>;

pub(crate) fn runtime_error(error: String) -> PyErr {
    PyRuntimeError::new_err(error)
}

pub(crate) fn runtime() -> Result<Arc<Runtime>> {
    static RUNTIME: std::sync::OnceLock<std::result::Result<Arc<Runtime>, String>> =
        std::sync::OnceLock::new();
    RUNTIME
        .get_or_init(|| {
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .build()
                .map(Arc::new)
                .map_err(|e| format!("Cannot create async runtime: {e}"))
        })
        .clone()
}

#[pyclass(module = "pldk._pldk")]
struct Connection {
    catalog: Arc<Catalog>,
    runtime: Arc<Runtime>,
}

#[pymethods]
impl Connection {
    #[new]
    #[pyo3(signature = (dsn=None, object_store_key=None))]
    fn new(
        py: Python<'_>,
        dsn: Option<String>,
        object_store_key: Option<String>,
    ) -> PyResult<Self> {
        let dsn = dsn
            .or_else(|| std::env::var("PLDK_DSN").ok())
            .filter(|s| !s.is_empty())
            .ok_or_else(|| PyValueError::new_err("Provide dsn or set PLDK_DSN"))?;
        if !(dsn.starts_with("postgres://") || dsn.starts_with("postgresql://")) {
            // TODO(mysql, sqlite): implement backend-specific connection and catalog SQL.
            return Err(PyValueError::new_err(
                "Only PostgreSQL catalogs are supported",
            ));
        }
        let key = object_store_key
            .or_else(|| std::env::var("PLDK_CONN_KEY").ok())
            .or_else(|| std::env::var("PLDK_SAS_KEY").ok())
            .filter(|s| !s.is_empty())
            .ok_or_else(|| {
                PyValueError::new_err(
                    "Provide object_store_key or set PLDK_CONN_KEY or PLDK_SAS_KEY",
                )
            })?;
        let runtime = runtime().map_err(runtime_error)?;
        let catalog = py
            .detach(|| runtime.block_on(Catalog::connect(&dsn, key)))
            .map_err(runtime_error)?;
        Ok(Self {
            catalog: Arc::new(catalog),
            runtime,
        })
    }

    #[pyo3(signature = (table_name, snapshot_id=None))]
    fn scan_table(
        &self,
        py: Python<'_>,
        table_name: &str,
        snapshot_id: Option<i64>,
    ) -> PyResult<Source> {
        let plan = py
            .detach(|| {
                self.runtime
                    .block_on(self.catalog.plan(table_name, snapshot_id))
            })
            .map_err(runtime_error)?;
        Ok(Source::new(
            Arc::new(plan),
            self.catalog.clone(),
            self.runtime.clone(),
        ))
    }

    fn get_stats(&self, py: Python<'_>, table_name: &str) -> PyResult<PyDataFrame> {
        py.detach(|| self.runtime.block_on(self.catalog.stats(table_name)))
            .map(PyDataFrame)
            .map_err(runtime_error)
    }
}

#[pymodule]
fn _pldk(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Connection>()?;
    m.add_class::<Source>()?;
    m.add_class::<source::BatchIterator>()?;
    Ok(())
}
