use std::collections::HashSet;
use std::sync::Arc;

use polars::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_polars::{PyDataFrame, PySchema};
use tokio::runtime::Runtime;

use crate::catalog::{Catalog, DataFile, Plan};
use crate::{Result, pruning, runtime_error, types};

#[pyclass(module = "pldk._pldk")]
pub(crate) struct Source {
    plan: Arc<Plan>,
    catalog: Arc<Catalog>,
    runtime: Arc<Runtime>,
}

impl Source {
    pub fn new(plan: Arc<Plan>, catalog: Arc<Catalog>, runtime: Arc<Runtime>) -> Self {
        Self {
            plan,
            catalog,
            runtime,
        }
    }
}

#[pymethods]
impl Source {
    fn schema(&self) -> PySchema {
        PySchema(Arc::new(
            self.plan
                .columns
                .iter()
                .map(|c| Field::new(c.name.clone().into(), c.dtype.clone()))
                .collect(),
        ))
    }

    #[pyo3(signature = (with_columns, predicate, n_rows, batch_size))]
    fn __call__(
        &self,
        with_columns: Option<Vec<String>>,
        predicate: Option<Bound<'_, PyAny>>,
        n_rows: Option<usize>,
        batch_size: Option<usize>,
    ) -> PyResult<BatchIterator> {
        let batch_size = batch_size.unwrap_or(65_536);
        if batch_size == 0 {
            return Err(PyValueError::new_err("batch_size must be positive"));
        }
        let output = with_columns
            .unwrap_or_else(|| self.plan.columns.iter().map(|c| c.name.clone()).collect());
        for name in &output {
            if !self.plan.columns.iter().any(|c| &c.name == name) {
                return Err(PyValueError::new_err(format!("Unknown column {name:?}")));
            }
        }
        // Compact serde enum indices depend on Cargo features, even at the same
        // Polars commit. Named JSON variants cannot silently decode as another op.
        let mut residual = None;
        let predicate = predicate
            .map(|p| {
                let kwargs = PyDict::new(p.py());
                kwargs.set_item("format", "json")?;
                let json: String = p
                    .getattr("meta")?
                    .call_method("serialize", (), Some(&kwargs))?
                    .extract()?;
                match Expr::deserialize_json_from_str(&json) {
                    Ok(expr) => Ok(Some(expr)),
                    Err(error) => {
                        p.py().import("warnings")?.call_method1(
                            "warn",
                            (format!(
                                "DuckLake statistics pruning unavailable for this expression; \
                                 filtering with Python Polars instead: {error}"
                            ),),
                        )?;
                        residual = Some(p.unbind());
                        Ok::<_, PyErr>(None)
                    }
                }
            })
            .transpose()?
            .flatten();
        let mut needed: HashSet<String> = output.iter().cloned().collect();
        if let Some(p) = &predicate {
            needed.extend(
                p.clone()
                    .meta()
                    .root_names()
                    .into_iter()
                    .map(|s| s.to_string()),
            );
        }
        if residual.is_some() {
            needed.extend(self.plan.columns.iter().map(|c| c.name.clone()));
        }
        Ok(BatchIterator {
            plan: self.plan.clone(),
            catalog: self.catalog.clone(),
            runtime: self.runtime.clone(),
            output,
            needed,
            predicate,
            residual,
            remaining: n_rows.unwrap_or(usize::MAX),
            batch_size,
            file_index: 0,
            inline_index: 0,
            pending: None,
            offset: 0,
            emitted: false,
            finished: false,
        })
    }
}

#[pyclass(module = "pldk._pldk")]
pub(crate) struct BatchIterator {
    plan: Arc<Plan>,
    catalog: Arc<Catalog>,
    runtime: Arc<Runtime>,
    output: Vec<String>,
    needed: HashSet<String>,
    predicate: Option<Expr>,
    residual: Option<Py<PyAny>>,
    remaining: usize,
    batch_size: usize,
    file_index: usize,
    inline_index: usize,
    pending: Option<DataFrame>,
    offset: usize,
    emitted: bool,
    finished: bool,
}

#[pymethods]
impl BatchIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<PyDataFrame>> {
        py.detach(|| self.next_batch())
            .map(|df| df.map(PyDataFrame))
            .map_err(runtime_error)
    }
}

impl BatchIterator {
    fn next_batch(&mut self) -> Result<Option<DataFrame>> {
        if self.finished {
            return Ok(None);
        }
        loop {
            if self.remaining == 0 {
                self.finished = true;
                break;
            }
            if let Some(frame) = &self.pending {
                let count = self
                    .batch_size
                    .min(self.remaining)
                    .min(frame.height() - self.offset);
                if count > 0 {
                    let batch = frame.slice(self.offset as i64, count);
                    self.offset += count;
                    self.remaining -= count;
                    self.emitted = true;
                    return Ok(Some(batch));
                }
                self.pending = None;
            }
            let frame = if self.file_index < self.plan.files.len() {
                let file = &self.plan.files[self.file_index];
                self.file_index += 1;
                if self
                    .predicate
                    .as_ref()
                    .is_some_and(|p| !pruning::may_match(p, &file.stats))
                {
                    continue;
                }
                self.runtime.block_on(self.read_file(file))?
            } else if self.inline_index < self.plan.inline.len() {
                let frame = self.plan.inline[self.inline_index].clone();
                self.inline_index += 1;
                frame
            } else {
                self.finished = true;
                break;
            };
            let mut lazy = frame.lazy();
            if let Some(predicate) = &self.predicate {
                lazy = lazy.filter(predicate.clone());
            }
            let mut frame = lazy.collect().map_err(|e| e.to_string())?;
            if let Some(predicate) = &self.residual {
                frame = Python::attach(|py| {
                    PyDataFrame(frame)
                        .into_pyobject(py)?
                        .call_method1("filter", (predicate.bind(py),))?
                        .extract::<PyDataFrame>()
                        .map(|df| df.0)
                })
                .map_err(|e| e.to_string())?;
            }
            let frame = if self.output.is_empty() {
                DataFrame::empty_with_height(frame.height())
            } else {
                frame
                    .select(self.output.iter().map(String::as_str))
                    .map_err(|e| e.to_string())?
            };
            self.pending = Some(frame);
            self.offset = 0;
        }
        if !self.emitted {
            self.emitted = true;
            let columns: Vec<_> = self
                .output
                .iter()
                .filter_map(|name| self.plan.columns.iter().find(|c| &c.name == name))
                .cloned()
                .collect();
            return types::empty_frame(&columns).map(Some);
        }
        Ok(None)
    }

    async fn read_file(&self, file: &DataFile) -> Result<DataFrame> {
        let mut reader = self.catalog.storage.parquet(&file.path).await?;
        let schema = reader.schema().map_err(|e| e.to_string())?;
        let mut sources = Vec::new();
        let mut selected = Vec::new();
        for c in self
            .plan
            .columns
            .iter()
            .filter(|c| self.needed.contains(&c.name))
        {
            let source = if let Some(mapping) = &file.mapping {
                mapping.get(&c.id).cloned()
            } else {
                schema.iter().find_map(|(name, f)| {
                    f.metadata
                        .as_ref()
                        .and_then(|m| m.get("PARQUET:field_id"))
                        .and_then(|s| s.parse::<i64>().ok())
                        .filter(|&id| id == c.id)
                        .map(|_| name.to_string())
                })
            };
            if let Some(name) = &source {
                if !schema.contains(name.as_str()) {
                    return Err(format!("Mapped Parquet column {name:?} is missing"));
                }
                selected.push(name.clone());
            }
            sources.push((c, source));
        }
        let partial = file.partial_max.is_some_and(|s| s > self.plan.snapshot);
        if partial {
            selected.push("_ducklake_internal_snapshot_id".into());
        }
        // A physical column preserves height even for zero-column projections.
        if selected.is_empty() {
            if let Some((name, _)) = schema.iter().next() {
                selected.push(name.to_string());
            }
        }
        selected.sort();
        selected.dedup();
        let raw = reader
            .with_columns(Some(selected))
            .finish()
            .map_err(|e| e.to_string())?;
        let mut deleted = file.inline_deletes.clone();
        if let Some(delete) = &file.delete {
            let mut frame = self
                .catalog
                .storage
                .parquet(&delete.path)
                .await?
                .finish()
                .map_err(|e| e.to_string())?;
            if delete.partial_max.is_some_and(|s| s > self.plan.snapshot) {
                frame = frame
                    .lazy()
                    .filter(col("_ducklake_internal_snapshot_id").lt_eq(lit(self.plan.snapshot)))
                    .collect()
                    .map_err(|e| e.to_string())?;
            }
            let positions = frame
                .column("row_id")
                .map_err(|e| e.to_string())?
                .cast(&DataType::Int64)
                .map_err(|e| e.to_string())?;
            for position in positions.i64().map_err(|e| e.to_string())?.iter() {
                let position = position.ok_or("Delete file contains a NULL row position")?;
                if position < 0 || position as usize >= raw.height() {
                    return Err(format!(
                        "Delete file has out-of-range position for file {}",
                        file.id
                    ));
                }
                deleted.insert(position);
            }
        }
        let mut cols = Vec::new();
        for (c, source) in sources {
            let series = match source {
                Some(source) => {
                    let mut series = raw
                        .column(&source)
                        .map_err(|e| e.to_string())?
                        .as_materialized_series()
                        .strict_cast(&c.dtype)
                        .map_err(|e| e.to_string())?;
                    series.rename(c.name.clone().into());
                    series
                }
                None => types::text_series(c, &vec![c.initial_default.clone(); raw.height()])?,
            };
            cols.push(series.into_column());
        }
        let mut frame = if cols.is_empty() {
            DataFrame::empty_with_height(raw.height())
        } else {
            DataFrame::new_infer_height(cols).map_err(|e| e.to_string())?
        };
        let mut mask: Vec<bool> = (0..raw.height())
            .map(|i| !deleted.contains(&(i as i64)))
            .collect();
        if partial {
            let snapshots = raw
                .column("_ducklake_internal_snapshot_id")
                .map_err(|e| e.to_string())?
                .i64()
                .map_err(|e| e.to_string())?;
            for (keep, snapshot) in mask.iter_mut().zip(snapshots.iter()) {
                *keep &= snapshot.ok_or("Partial data file has a NULL snapshot ID")?
                    <= self.plan.snapshot;
            }
        }
        if partial || !deleted.is_empty() {
            frame = frame
                .filter(&BooleanChunked::from_slice("".into(), &mask))
                .map_err(|e| e.to_string())?;
        }
        Ok(frame)
    }
}
