use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

use polars::prelude::*;
use serde::Deserialize;
use sqlx::postgres::{PgPoolOptions, PgRow};
use sqlx::{PgConnection, PgPool, Row};

use crate::Result;
use crate::pruning::ColumnStats;
use crate::storage::{Storage, resolve};
use crate::types::{self, Column};

fn db(error: sqlx::Error) -> String {
    format!("PostgreSQL catalog query failed: {error}")
}

fn inline_system_columns(version: FormatVersion) -> (&'static str, &'static str, &'static str) {
    match version {
        FormatVersion::V1_0 => ("row_id", "begin_snapshot", "end_snapshot"),
        FormatVersion::V1_1 => (
            "_ducklake_row_id",
            "_ducklake_begin_snapshot",
            "_ducklake_end_snapshot",
        ),
    }
}

pub(crate) fn quote(name: &str) -> String {
    // TODO(mysql): use backtick identifier quoting.
    format!("\"{}\"", name.replace('"', "\"\""))
}

#[derive(Clone, PartialEq, Eq)]
struct InlineTable {
    table_id: i64,
    name: String,
    version: i64,
    columns: Vec<(i64, String)>,
}

pub(crate) struct DataFile {
    pub id: i64,
    pub path: String,
    pub partial_max: Option<i64>,
    pub mapping: Option<HashMap<i64, String>>,
    pub stats: HashMap<String, ColumnStats>,
    pub delete: Option<DeleteFile>,
    pub inline_deletes: HashSet<i64>,
}

pub(crate) struct DeleteFile {
    pub path: String,
    pub partial_max: Option<i64>,
}

#[derive(Deserialize)]
struct RawStats {
    column_id: i64,
    min_value: Option<String>,
    max_value: Option<String>,
    null_count: Option<i64>,
    value_count: Option<i64>,
    contains_nan: Option<bool>,
}

#[derive(Deserialize)]
struct NameMapping {
    source_name: String,
    target_field_id: i64,
    is_partition: Option<bool>,
}

pub(crate) struct Plan {
    pub snapshot: i64,
    pub columns: Vec<Column>,
    pub files: Vec<DataFile>,
    pub inline: Vec<DataFrame>,
}

pub(crate) struct Catalog {
    pool: PgPool,
    inline_tables: Mutex<InlineCache>,
    pub storage: Storage,
    format_version: FormatVersion,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum FormatVersion {
    V1_0,
    V1_1,
}

struct InlineCache {
    snapshot: Option<i64>,
    tables: Vec<InlineTable>,
}

async fn inline_tables(connection: &mut PgConnection) -> Result<Vec<InlineTable>> {
    sqlx::query(
        "SELECT i.table_id, i.table_name, i.schema_version,
           coalesce(array_agg(c.column_id ORDER BY c.column_order)
             FILTER (WHERE c.column_id IS NOT NULL), ARRAY[]::bigint[]) AS column_ids,
           coalesce(array_agg(c.column_name ORDER BY c.column_order)
             FILTER (WHERE c.column_id IS NOT NULL), ARRAY[]::varchar[]) AS column_names
         FROM ducklake_inlined_data_tables i
         JOIN ducklake_schema_versions v
           ON v.table_id = i.table_id AND v.schema_version = i.schema_version
         LEFT JOIN ducklake_column c ON c.table_id = i.table_id AND c.parent_column IS NULL
           AND c.begin_snapshot <= v.begin_snapshot
           AND (c.end_snapshot > v.begin_snapshot OR c.end_snapshot IS NULL)
         GROUP BY i.table_id, i.table_name, i.schema_version",
    )
    .fetch_all(connection)
    .await
    .map_err(db)?
    .into_iter()
    .map(|r| {
        let ids: Vec<i64> = r.try_get("column_ids").map_err(db)?;
        let names: Vec<String> = r.try_get("column_names").map_err(db)?;
        Ok(InlineTable {
            table_id: r.try_get("table_id").map_err(db)?,
            name: r.try_get("table_name").map_err(db)?,
            version: r.try_get("schema_version").map_err(db)?,
            columns: ids.into_iter().zip(names).collect(),
        })
    })
    .collect()
}

impl Catalog {
    pub async fn connect(dsn: &str, key: String) -> Result<Self> {
        let pool = PgPoolOptions::new()
            .max_connections(4)
            .connect(dsn)
            .await
            .map_err(db)?;
        let mut connection = pool
            .begin_with("BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY")
            .await
            .map_err(db)?;
        let metadata = sqlx::query(
            "SELECT value, (SELECT max(snapshot_id) FROM ducklake_snapshot) AS snapshot
             FROM ducklake_metadata WHERE key = 'version' AND scope IS NULL",
        )
        .fetch_one(&mut *connection)
        .await
        .map_err(db)?;
        let version: String = metadata.try_get("value").map_err(db)?;
        let format_version = match version.as_str() {
            "1.0" => FormatVersion::V1_0,
            "1.1" => FormatVersion::V1_1,
            _ => {
                return Err(format!(
                    "Only DuckLake catalog formats 1.0 and 1.1 are supported (found {version})"
                ));
            }
        };
        let snapshot = metadata.try_get("snapshot").map_err(db)?;
        let inline = inline_tables(&mut connection).await?;
        connection.commit().await.map_err(db)?;
        Ok(Self {
            pool,
            inline_tables: Mutex::new(InlineCache {
                snapshot,
                tables: inline,
            }),
            storage: Storage::new(key),
            format_version,
        })
    }

    async fn inline_mapping(
        &self,
        connection: &mut PgConnection,
        snapshot: i64,
    ) -> Result<Vec<InlineTable>> {
        {
            let cache = self
                .inline_tables
                .lock()
                .map_err(|_| "Inline table cache lock poisoned")?;
            if cache.snapshot == Some(snapshot) {
                return Ok(cache.tables.clone());
            }
        }
        let tables = inline_tables(connection).await?;
        *self
            .inline_tables
            .lock()
            .map_err(|_| "Inline table cache lock poisoned")? = InlineCache {
            snapshot: Some(snapshot),
            tables: tables.clone(),
        };
        Ok(tables)
    }

    pub async fn plan(&self, name: &str, requested: Option<i64>) -> Result<Plan> {
        // TODO(mysql, sqlite): use the backend's equivalent consistent read transaction.
        let mut tx = self
            .pool
            .begin_with("BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY")
            .await
            .map_err(db)?;
        let table = table(&mut tx, name, requested, self.format_version).await?;
        let snapshot: i64 = table.try_get("snapshot_id").map_err(db)?;
        let table_id: i64 = table.try_get("table_id").map_err(db)?;
        let columns = columns(&mut tx, table_id, snapshot).await?;
        let root: Option<String> = table.try_get("data_path").map_err(db)?;
        let schema_path = resolve(
            root.as_deref().unwrap_or(""),
            &table.try_get::<String, _>("schema_path").map_err(db)?,
            table.try_get("schema_relative").map_err(db)?,
        );
        let table_path = resolve(
            &schema_path,
            &table.try_get::<String, _>("path").map_err(db)?,
            table.try_get("path_is_relative").map_err(db)?,
        );
        // TODO(mysql, sqlite): replace PostgreSQL JSONB aggregates and $n parameters.
        let rows = sqlx::query(
            "WITH visible_files AS (
               SELECT data_file_id, table_id, path, path_is_relative, file_format,
                      encryption_key, mapping_id, partial_max, file_order
               FROM ducklake_data_file WHERE table_id = $1 AND begin_snapshot <= $2
                 AND (end_snapshot > $2 OR end_snapshot IS NULL)
             ), stats AS (
               SELECT s.data_file_id, jsonb_agg(jsonb_build_object(
                 'column_id', s.column_id, 'min_value', s.min_value, 'max_value', s.max_value,
                 'null_count', s.null_count, 'value_count', s.value_count,
                 'contains_nan', s.contains_nan)) AS column_stats
               FROM ducklake_file_column_stats s JOIN visible_files f
                 ON f.data_file_id = s.data_file_id AND f.table_id = s.table_id
               GROUP BY s.data_file_id
             ), mappings AS (
               SELECT m.mapping_id, jsonb_agg(jsonb_build_object(
                 'source_name', m.source_name, 'target_field_id', m.target_field_id,
                 'is_partition', m.is_partition)) AS name_mapping
               FROM ducklake_name_mapping m
               WHERE m.parent_column IS NULL
                 AND m.mapping_id IN (SELECT mapping_id FROM visible_files)
               GROUP BY m.mapping_id
             )
             SELECT f.data_file_id, f.path, f.path_is_relative, f.file_format, f.encryption_key,
                    f.mapping_id, f.partial_max, coalesce(s.column_stats, '[]'::jsonb) AS column_stats,
                    coalesce(m.name_mapping, '[]'::jsonb) AS name_mapping,
                    d.path AS delete_path, d.path_is_relative AS delete_relative,
                    d.format AS delete_format, d.encryption_key AS delete_encryption,
                    d.partial_max AS delete_partial_max
             FROM visible_files f LEFT JOIN stats s USING (data_file_id)
             LEFT JOIN mappings m ON m.mapping_id = f.mapping_id
             LEFT JOIN ducklake_delete_file d ON d.data_file_id = f.data_file_id
               AND d.table_id = f.table_id AND d.begin_snapshot <= $2
               AND (d.end_snapshot > $2 OR d.end_snapshot IS NULL)
             ORDER BY f.file_order NULLS LAST, f.data_file_id",
        ).bind(table_id).bind(snapshot).fetch_all(&mut *tx).await.map_err(db)?;
        let delete_table = format!("ducklake_inlined_delete_{table_id}");
        let exists: bool = table.try_get("has_inline_deletes").map_err(db)?;
        let mut inline_deletes: HashMap<i64, HashSet<i64>> = HashMap::new();
        if exists {
            for r in sqlx::query(&format!(
                "SELECT file_id, row_id FROM {} WHERE begin_snapshot <= $1",
                quote(&delete_table),
            ))
            .bind(snapshot)
            .fetch_all(&mut *tx)
            .await
            .map_err(db)?
            {
                inline_deletes
                    .entry(r.try_get("file_id").map_err(db)?)
                    .or_default()
                    .insert(r.try_get("row_id").map_err(db)?);
            }
        }
        let mut files = Vec::new();
        let mut seen = HashSet::new();
        for r in rows {
            if r.try_get::<String, _>("file_format").map_err(db)? != "parquet"
                || r.try_get::<Option<String>, _>("encryption_key")
                    .map_err(db)?
                    .is_some()
            {
                return Err("Only unencrypted Parquet data files are supported".into());
            }
            let id: i64 = r.try_get("data_file_id").map_err(db)?;
            if !seen.insert(id) {
                return Err("Catalog has multiple visible delete files for one data file".into());
            }
            let mapping_id: Option<i64> = r.try_get("mapping_id").map_err(db)?;
            let mapping = if mapping_id.is_some() {
                let mut mapping = HashMap::new();
                let raw: sqlx::types::Json<Vec<NameMapping>> =
                    r.try_get("name_mapping").map_err(db)?;
                for m in raw.0 {
                    if m.is_partition == Some(true) {
                        return Err("Hive partition column reconstruction is not supported".into());
                    }
                    mapping.insert(m.target_field_id, m.source_name);
                }
                Some(mapping)
            } else {
                None
            };
            let raw: sqlx::types::Json<Vec<RawStats>> = r.try_get("column_stats").map_err(db)?;
            let mut stats = HashMap::new();
            for s in raw.0 {
                if let Some(column) = columns.iter().find(|c| c.id == s.column_id) {
                    stats.insert(
                        column.name.clone(),
                        ColumnStats {
                            dtype: column.dtype.clone(),
                            min: s.min_value,
                            max: s.max_value,
                            null_count: s.null_count,
                            value_count: s.value_count,
                            contains_nan: s.contains_nan,
                        },
                    );
                }
            }
            let delete =
                if let Some(path) = r.try_get::<Option<String>, _>("delete_path").map_err(db)? {
                    if r.try_get::<Option<String>, _>("delete_format")
                        .map_err(db)?
                        .as_deref()
                        != Some("parquet")
                        || r.try_get::<Option<String>, _>("delete_encryption")
                            .map_err(db)?
                            .is_some()
                    {
                        return Err("Puffin or encrypted delete files are not supported".into());
                    }
                    Some(DeleteFile {
                        path: resolve(
                            &table_path,
                            &path,
                            r.try_get("delete_relative").map_err(db)?,
                        ),
                        partial_max: r.try_get("delete_partial_max").map_err(db)?,
                    })
                } else {
                    None
                };
            files.push(DataFile {
                id,
                path: resolve(
                    &table_path,
                    &r.try_get::<String, _>("path").map_err(db)?,
                    r.try_get("path_is_relative").map_err(db)?,
                ),
                partial_max: r.try_get("partial_max").map_err(db)?,
                mapping,
                stats,
                delete,
                inline_deletes: inline_deletes.remove(&id).unwrap_or_default(),
            });
        }
        // Refresh under the same metadata view: new schema versions can add inline tables.
        let inline_mapping = self
            .inline_mapping(&mut tx, table.try_get("latest_snapshot").map_err(db)?)
            .await?;
        let (inline_row_id, inline_begin_snapshot, inline_end_snapshot) =
            inline_system_columns(self.format_version);
        let mut inline = Vec::new();
        let mut inline_queries = Vec::new();
        for entry in inline_mapping.iter().filter(|e| e.table_id == table_id) {
            let stored_columns = &entry.columns;
            let expressions = columns
                .iter()
                .enumerate()
                .map(|(i, c)| {
                    match stored_columns.iter().find(|(id, _)| *id == c.id) {
                        Some((_, old_name)) => {
                            // TODO(mysql, sqlite): DuckLake's BYTEA text encoding is PostgreSQL-specific.
                            let name = quote(old_name);
                            let expr = match c.dtype {
                                DataType::String if c.bytea_text => {
                                    format!("encode({name}, 'hex')")
                                }
                                DataType::Binary => {
                                    format!("encode({name}, 'hex')")
                                }
                                _ => format!("{name}::text"),
                            };
                            format!("{expr} AS {}", quote(&c.name))
                        }
                        None if c.bytea_text => format!(
                            "encode(convert_to(${}::text, 'UTF8'), 'hex') AS {}",
                            i + 2,
                            quote(&c.name),
                        ),
                        None => format!("${}::text AS {}", i + 2, quote(&c.name)),
                    }
                })
                .collect::<Vec<_>>()
                .join(", ");
            inline_queries.push(format!(
                "(SELECT {expressions} FROM {} WHERE {inline_begin_snapshot} <= $1
                 AND ({inline_end_snapshot} > $1 OR {inline_end_snapshot} IS NULL)
                 ORDER BY {inline_row_id})",
                quote(&entry.name),
                inline_begin_snapshot = quote(inline_begin_snapshot),
                inline_end_snapshot = quote(inline_end_snapshot),
                inline_row_id = quote(inline_row_id),
            ));
        }
        if !inline_queries.is_empty() {
            // The parameter CTE types defaults even when every inline version has the column.
            let defaults = columns
                .iter()
                .enumerate()
                .map(|(i, _)| format!("${}::text AS d{i}", i + 2))
                .collect::<Vec<_>>()
                .join(",");
            let sql = format!(
                "WITH defaults AS (SELECT {defaults}) {}",
                inline_queries.join(" UNION ALL ")
            );
            let mut query = sqlx::query(&sql).bind(snapshot);
            for c in &columns {
                query = query.bind(&c.initial_default);
            }
            let rows = query.fetch_all(&mut *tx).await.map_err(db)?;
            let mut out = Vec::new();
            for (i, c) in columns.iter().enumerate() {
                let mut values = rows
                    .iter()
                    .map(|r| r.try_get::<Option<String>, _>(i).map_err(db))
                    .collect::<Result<Vec<_>>>()?;
                if c.bytea_text {
                    for value in values.iter_mut().flatten() {
                        *value = String::from_utf8(types::decode_hex(value)?)
                            .map_err(|_| format!("Invalid UTF-8 in inline column {:?}", c.name))?;
                    }
                }
                out.push(types::text_series(c, &values)?.into_column());
            }
            inline.push(DataFrame::new_infer_height(out).map_err(|e| e.to_string())?);
        }
        tx.commit().await.map_err(db)?;
        Ok(Plan {
            snapshot,
            columns,
            files,
            inline,
        })
    }

    pub async fn stats(&self, name: &str) -> Result<DataFrame> {
        let mut tx = self
            .pool
            .begin_with("BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY")
            .await
            .map_err(db)?;
        let t = table(&mut tx, name, None, self.format_version).await?;
        let snapshot: i64 = t.try_get("snapshot_id").map_err(db)?;
        let id: i64 = t.try_get("table_id").map_err(db)?;
        let exactness_columns = match self.format_version {
            FormatVersion::V1_0 => "NULL::boolean AS min_is_exact, NULL::boolean AS max_is_exact",
            FormatVersion::V1_1 => "s.min_is_exact, s.max_is_exact",
        };
        let rows = sqlx::query(&format!(
            "SELECT c.column_id, c.column_name, c.initial_default,
                    s.min_value, s.max_value, s.contains_null, s.contains_nan,
                    s.extra_stats, {exactness_columns}
             FROM ducklake_column c LEFT JOIN ducklake_table_column_stats s
               ON c.table_id = s.table_id AND c.column_id = s.column_id
             WHERE c.table_id = $1 AND c.parent_column IS NULL AND c.begin_snapshot <= $2
               AND (c.end_snapshot > $2 OR c.end_snapshot IS NULL)
             ORDER BY c.column_order"
        ))
        .bind(id)
        .bind(snapshot)
        .fetch_all(&mut *tx)
        .await
        .map_err(db)?;
        let strings = |key| {
            rows.iter()
                .map(|r| r.try_get::<Option<String>, _>(key).map_err(db))
                .collect::<Result<Vec<_>>>()
        };
        let bools = |key| {
            rows.iter()
                .map(|r| r.try_get::<Option<bool>, _>(key).map_err(db))
                .collect::<Result<Vec<_>>>()
        };
        let current_columns: Vec<(i64, Option<String>)> = rows
            .iter()
            .map(|r| {
                Ok((
                    r.try_get("column_id").map_err(db)?,
                    r.try_get("initial_default").map_err(db)?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let delete_table = quote(&format!("ducklake_inlined_delete_{id}"));
        let has_inline_deletes: bool = t.try_get("has_inline_deletes").map_err(db)?;
        let mappings = self
            .inline_mapping(&mut tx, t.try_get("latest_snapshot").map_err(db)?)
            .await?;
        let ids = current_columns
            .iter()
            .map(|(id, _)| id.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let (_, inline_begin_snapshot, inline_end_snapshot) =
            inline_system_columns(self.format_version);
        let inline_queries = mappings
            .iter()
            .filter(|e| e.table_id == id)
            .map(|entry| {
                let counts = current_columns
                    .iter()
                    .map(|(column_id, initial_default)| {
                        let condition = match entry.columns.iter().find(|(id, _)| id == column_id) {
                            Some((_, name)) => format!("{} IS NULL", quote(name)),
                            None if initial_default.is_none() => "true".into(),
                            None => "false".into(),
                        };
                        format!("count(*) FILTER (WHERE {condition})")
                    })
                    .collect::<Vec<_>>()
                    .join(",");
                format!(
                    "SELECT unnest(ARRAY[{ids}]::bigint[]) AS column_id,
                        unnest(ARRAY[{counts}]::bigint[]) AS null_count
                 FROM {} WHERE {inline_begin_snapshot} <= $2
                   AND ({inline_end_snapshot} > $2 OR {inline_end_snapshot} IS NULL)",
                    quote(&entry.name),
                    inline_begin_snapshot = quote(inline_begin_snapshot),
                    inline_end_snapshot = quote(inline_end_snapshot),
                )
            })
            .collect::<Vec<_>>();
        let inline_sql = if inline_queries.is_empty() {
            "SELECT NULL::bigint AS column_id, 0::bigint AS null_count WHERE false".into()
        } else {
            inline_queries.join(" UNION ALL ")
        };
        let no_inline_deletes = if has_inline_deletes {
            format!("NOT EXISTS (SELECT 1 FROM {delete_table} WHERE begin_snapshot <= $2)")
        } else {
            "true".into()
        };
        let null_rows = sqlx::query(&format!(
            "WITH inline_counts AS ({inline_sql}),
             visible_files AS (
               SELECT data_file_id, table_id, partial_max FROM ducklake_data_file
               WHERE table_id = $1 AND begin_snapshot <= $2
                 AND (end_snapshot > $2 OR end_snapshot IS NULL)
             )
             SELECT c.column_id,
               CASE WHEN count(f.data_file_id) = count(s.null_count)
                 AND NOT coalesce(bool_or(f.partial_max > $2), false)
                 AND {no_inline_deletes} AND NOT EXISTS (
                   SELECT 1 FROM ducklake_delete_file d WHERE d.table_id = $1
                     AND d.begin_snapshot <= $2 AND (d.end_snapshot > $2 OR d.end_snapshot IS NULL)
                 )
               THEN (coalesce(sum(s.null_count),0) + coalesce(
                 (SELECT sum(i.null_count) FROM inline_counts i WHERE i.column_id = c.column_id),0
               ))::bigint END AS null_values
             FROM ducklake_column c LEFT JOIN visible_files f ON f.table_id = c.table_id
             LEFT JOIN ducklake_file_column_stats s ON s.data_file_id = f.data_file_id
               AND s.table_id = f.table_id AND s.column_id = c.column_id
             WHERE c.table_id = $1 AND c.parent_column IS NULL AND c.begin_snapshot <= $2
               AND (c.end_snapshot > $2 OR c.end_snapshot IS NULL)
             GROUP BY c.column_id, c.column_order ORDER BY c.column_order"
        ))
        .bind(id)
        .bind(snapshot)
        .fetch_all(&mut *tx)
        .await
        .map_err(db)?;
        let null_values = null_rows
            .iter()
            .map(|r| r.try_get::<Option<i64>, _>("null_values").map_err(db))
            .collect::<Result<Vec<_>>>()?;
        let frame = df!(
            "column_name" => strings("column_name")?,
            "min" => strings("min_value")?,
            "max" => strings("max_value")?,
            "null_values" => null_values,
            "contains_null" => bools("contains_null")?,
            "contains_nan" => bools("contains_nan")?,
            "extra_stats" => strings("extra_stats")?,
            "min_is_exact" => bools("min_is_exact")?,
            "max_is_exact" => bools("max_is_exact")?
        )
        .map_err(|e| e.to_string())?;
        tx.commit().await.map_err(db)?;
        Ok(frame)
    }
}

async fn table(
    connection: &mut PgConnection,
    name: &str,
    snapshot: Option<i64>,
    format_version: FormatVersion,
) -> Result<PgRow> {
    // TODO(mysql, sqlite): replace to_regclass and PostgreSQL parameter/cast syntax.
    let schemas = match format_version {
        FormatVersion::V1_0 => {
            "schemas AS (
               SELECT schema_id, schema_name::text AS full_name, path, path_is_relative
               FROM ducklake_schema CROSS JOIN chosen_snapshot v
               WHERE begin_snapshot <= v.snapshot_id
                 AND (end_snapshot > v.snapshot_id OR end_snapshot IS NULL)
             )"
        }
        FormatVersion::V1_1 => {
            "schemas AS (
               SELECT schema_id, schema_name::text AS full_name, path, path_is_relative
               FROM ducklake_schema CROSS JOIN chosen_snapshot v
               WHERE parent_schema_id IS NULL AND begin_snapshot <= v.snapshot_id
                 AND (end_snapshot > v.snapshot_id OR end_snapshot IS NULL)
               UNION ALL
               SELECT s.schema_id, p.full_name || '.' || s.schema_name, s.path, s.path_is_relative
               FROM ducklake_schema s JOIN schemas p ON s.parent_schema_id = p.schema_id
               CROSS JOIN chosen_snapshot v
               WHERE s.begin_snapshot <= v.snapshot_id
                 AND (s.end_snapshot > v.snapshot_id OR s.end_snapshot IS NULL)
             )"
        }
    };
    let rows = sqlx::query(&format!(
        "WITH RECURSIVE latest AS (
           SELECT max(snapshot_id) AS latest_snapshot FROM ducklake_snapshot
         ), chosen_snapshot AS (
           SELECT snapshot_id, latest_snapshot FROM ducklake_snapshot CROSS JOIN latest
           WHERE snapshot_id = coalesce($2::bigint, latest_snapshot)
         ), {schemas}
         SELECT t.table_id, t.path, t.path_is_relative,
                s.path AS schema_path, s.path_is_relative AS schema_relative,
                v.snapshot_id, v.latest_snapshot,
                to_regclass('ducklake_inlined_delete_' || t.table_id::text) IS NOT NULL
                  AS has_inline_deletes,
                (SELECT value FROM ducklake_metadata
                 WHERE key = 'data_path' AND scope IS NULL) AS data_path
         FROM ducklake_table t JOIN schemas s USING (schema_id)
         CROSS JOIN chosen_snapshot v
         WHERE (t.table_name = $1 OR s.full_name || '.' || t.table_name = $1)
           AND t.begin_snapshot <= v.snapshot_id
           AND (t.end_snapshot > v.snapshot_id OR t.end_snapshot IS NULL)",
    ))
    .bind(name)
    .bind(snapshot)
    .fetch_all(connection)
    .await
    .map_err(db)?;
    if rows.len() != 1 {
        return Err(if rows.is_empty() {
            format!("Table {name:?} or Snapshot {snapshot:?} does not exist or has expired")
        } else {
            format!("Ambiguous table name {name:?}; qualify it with its DuckLake schema")
        });
    }
    Ok(rows.into_iter().next().unwrap())
}

async fn columns(connection: &mut PgConnection, id: i64, snapshot: i64) -> Result<Vec<Column>> {
    columns_at(connection, id, snapshot).await
}

async fn columns_at(connection: &mut PgConnection, id: i64, snapshot: i64) -> Result<Vec<Column>> {
    let rows = sqlx::query(
        "SELECT column_id, column_name, column_type, parent_column, initial_default
         FROM ducklake_column WHERE table_id = $1
         AND begin_snapshot <= $2 AND (end_snapshot > $2 OR end_snapshot IS NULL)
         ORDER BY column_order",
    )
    .bind(id)
    .bind(snapshot)
    .fetch_all(connection)
    .await
    .map_err(db)?;
    let mut result = Vec::new();
    for r in rows {
        if r.try_get::<Option<i64>, _>("parent_column")
            .map_err(db)?
            .is_some()
        {
            return Err("Nested DuckLake columns are not yet supported".into());
        }
        result.push(Column {
            id: r.try_get("column_id").map_err(db)?,
            name: r.try_get("column_name").map_err(db)?,
            dtype: types::dtype(&r.try_get::<String, _>("column_type").map_err(db)?)?,
            initial_default: r.try_get("initial_default").map_err(db)?,
            bytea_text: matches!(
                r.try_get::<String, _>("column_type").map_err(db)?.as_str(),
                "varchar" | "json"
            ),
        });
    }
    Ok(result)
}
