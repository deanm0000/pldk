# pldk

A read-only Rust/PyO3 DuckLake reader for Polars. PostgreSQL metadata is read
through SQLx; Azure blobs are read through Apache `object_store`. Neither the
Python package nor its Rust library depends on DuckDB.

## Compatibility

- Python 3.11+ and **Polars 2.1.0**.
- DuckLake catalog format **1.1** on PostgreSQL.
- Polars and `pyo3-polars` are pinned to the Python release's
  [Git commit](https://github.com/pola-rs/polars/commit/7f902bd3ffd27ff82d14278a09a295ff2f7aed92).
  The matching Rust toolchain and upstream hashbrown patch are pinned too.
- Azure public-cloud Blob Storage/ADLS paths: `https://`, `az://`, `azure://`,
  `abfs://`, and `abfss://`. Container-only paths need a credential containing
  the account name.

This is an initial implementation, not a complete DuckLake implementation.
MySQL, SQLite, other cloud providers, writing, encrypted Parquet, Puffin deletion
vectors, nested types, variant/geometry, interval, timetz and uint128 are not yet
supported. Unsupported catalog features raise errors rather than returning
partial table contents.

## Build

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install 'maturin>=1.9,<2'
maturin develop
```

The first build downloads and compiles the pinned Polars sources.

## Connect and scan

Configure secrets outside source control:

```sh
export PLDK_DSN='postgresql://user@host:5432/database'
# Set PLDK_SAS_KEY to a container SAS URL, or PLDK_CONN_KEY to an Azure
# connection string or account key, using your secret manager.
```

```python
import polars as pl
from pldk import Connection

connection = Connection()
recent = (
    connection.scan_table("main.events")
    .filter(pl.col("event_id") >= 1000)
    .select("event_id", "message")
    .head(100)
    .collect()
)
historical = connection.scan_table("main.events", snapshot_id=42).collect()
statistics = connection.get_stats("main.events")
```

`Connection(dsn=None, object_store_key=None)` resolves each omitted argument
independently. Credentials fall back to `PLDK_CONN_KEY`, then `PLDK_SAS_KEY`.
An explicit credential can be a connection string, container SAS URL, account
key, or SAS query string. A bare account key/SAS query requires paths identifying
the account. PostgreSQL authentication supports SQLx's normal mechanisms,
including `.pgpass`. The DSN's `options=-csearch_path=your_catalog_schema`
selects a non-default metadata schema; this is separate from DuckLake's logical
schemas such as `main`.

Construction is synchronous in Python, but releases the GIL while a shared
Tokio runtime establishes the SQLx connection, validates the catalog version,
and caches inline table mappings. No background service needs to be started.

## Read semantics

`scan_table` resolves and pins a snapshot in a PostgreSQL repeatable-read,
read-only transaction. Recollecting that LazyFrame uses the same snapshot;
calling `scan_table` again resolves the latest snapshot. Expired or unknown
snapshot IDs are rejected.

The registered source and batch iterator are implemented in Rust using
[`register_io_source`](https://docs.pola.rs/user-guide/plugins/io_plugins/).
They support projection, predicates, row limits and batch-size hints, including
predicate columns absent from the requested projection. Deletes and predicates
are applied **before** the output row limit. Each collection has its own iterator.

The reader combines visible Parquet files with snapshot-visible inline rows,
applies positional Parquet and inline deletions, handles partial-file snapshot
IDs, resolves relative paths, and maps top-level fields by DuckLake column ID
or imported-file name mappings. Added columns use initial defaults; renamed
columns retain their identities. Inline mappings are refreshed under the
metadata transaction so new inline schema versions are visible.

Metadata SQL uses explicit column lists. CTEs combine snapshot/table/path
resolution and file/statistics/mapping/delete retrieval. Inline versions are
read with one `UNION ALL` query rather than one query per version; null counts
are computed in a combined query rather than a query per column.

### Statistics pruning

File statistics are checked **before opening Azure blobs**. Scalar comparisons
(including reversed operands), AND/OR, null tests, literal membership and
between predicates can prune files. Integer comparisons preserve precision;
supported typed date/time/timestamp/decimal bounds are decoded before comparison.
Missing, incompatible or malformed bounds retain the file. Conservative bounds
are not assumed to be exact. NaNs and nulls are treated according to Polars'
filter semantics.

Every retained file still receives the exact predicate. Unsupported pruning
expressions simply retain files. If the Rust build cannot deserialize a Polars
function, it emits a warning and applies the original expression with the
installed Python Polars engine. Named JSON expression serialization avoids
Cargo-feature-dependent compact enum indices silently changing a predicate.

Historical pruning never uses mutable latest-table statistics.

### `get_stats(table_name)`

Returns one row per top-level column in catalog column order, with
`column_name` first, followed by `min`, `max`, `null_values`, `contains_null`,
`contains_nan`, `extra_stats`, `min_is_exact`, and `max_is_exact`.

Min/max retain DuckLake's string encoding because columns can have different
types. Bounds describe the **latest** catalog state and can conservatively
include deleted values. Missing statistics remain null.

`null_values` combines complete file null counts with visible inline null
counts in SQL. It is null when metadata alone cannot establish an exact count
(missing file statistics, visible positional deletions, or historical partial
file contents). **No Parquet or Azure access occurs in `get_stats`.**

### Current resource limitations

Blob downloads are lazy by file, not yet byte-range/row-group reads. Each
retained Parquet blob and its projected decoded columns are materialized before
being split into output batches. Snapshot-visible inline rows are materialized
when the scan is created to preserve the same metadata view. Consequently,
`batch_size` bounds output batches, not peak memory or download size. Azure
clients are reused per account/container.

## Tests

```sh
python -m pip install -e '.[test]'
cargo test --lib
python -m pytest -q
```

Live integration tests require `PLDK_TEST_DSN` and `PLDK_TEST_AZURE_URL` (a
container SAS URL with read/write/delete permissions). They use uniquely named
PostgreSQL schemas and blob prefixes and clean up their own objects. Without
these variables integration tests are skipped; configuration tests still run.
PyArrow is a **test-only** dependency used to write Parquet field-ID fixtures.

The tests are original behavioral ports based on these
[DuckLake upstream test families](https://github.com/duckdb/ducklake/tree/main/test/sql):

- `general/attach_at_snapshot.test`: snapshot visibility and unknown snapshots.
- `data_inlining/data_inlining_filter.test`: filtered inline data.
- `data_inlining/postgres_new_inlined_table_keeps_cache.test`: mapping refresh.
- `add_files/add_files_allow_missing_stats.test`: missing-statistics safety.
- `issues/issue_1442_nan_bounded_pruning.test`: NaN-aware bound handling.
- `alter/rename_column.test` and `alter/add_column.test`: stable identity/defaults.
- `delete/test_delete_partial_max_snapshot.test`: snapshot-aware delete visibility.

Fixtures populate isolated PostgreSQL catalog metadata directly; they do not
establish interoperability with a DuckDB-generated catalog. Pruning tests include
nonexistent blobs which must never be opened by a correctly pruned scan.
