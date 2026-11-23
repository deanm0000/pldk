"""Original behavioral ports of DuckLake read tests; see README for provenance."""

import polars as pl
from polars.testing import assert_frame_equal
import pytest

from pldk import Connection

pytestmark = pytest.mark.integration


def connect(lake):
    return Connection(lake.dsn, lake.credential)


def test_snapshot_and_inline_visibility(lake):
    lake.file(1, [1, 2], end=3)
    lake.file(2, [10, 11], begin=3)
    lake.db.execute(
        "INSERT INTO ducklake_inlined_data_1_1 VALUES "
        "(0,2,3,7,convert_to('old','UTF8')),"
        "(1,3,NULL,8,convert_to('new','UTF8'))"
    )
    conn = connect(lake)
    assert conn.scan_table("items", 1).collect()["id"].to_list() == [1, 2]
    assert conn.scan_table("main.items", 2).collect()["id"].to_list() == [1, 2, 7]
    scan = conn.scan_table("items")
    assert scan.collect()["id"].to_list() == [10, 11, 8]
    assert_frame_equal(scan.collect(), scan.collect(engine="streaming"))
    with pytest.raises(RuntimeError, match="Snapshot"):
        conn.scan_table("items", 999)


def test_catalog_format_1_0(lake):
    lake.db.execute("UPDATE ducklake_metadata SET value='1.0' WHERE key='version'")
    lake.db.execute("ALTER TABLE ducklake_schema DROP COLUMN parent_schema_id")
    lake.db.execute("ALTER TABLE ducklake_table_column_stats DROP COLUMN min_is_exact")
    lake.db.execute("ALTER TABLE ducklake_table_column_stats DROP COLUMN max_is_exact")
    lake.db.execute(
        "ALTER TABLE ducklake_inlined_data_1_1 "
        "RENAME COLUMN _ducklake_row_id TO row_id"
    )
    lake.db.execute(
        "ALTER TABLE ducklake_inlined_data_1_1 "
        "RENAME COLUMN _ducklake_begin_snapshot TO begin_snapshot"
    )
    lake.db.execute(
        "ALTER TABLE ducklake_inlined_data_1_1 "
        "RENAME COLUMN _ducklake_end_snapshot TO end_snapshot"
    )
    lake.db.execute(
        "INSERT INTO ducklake_inlined_data_1_1 VALUES "
        "(1,1,NULL,7,convert_to('inline','UTF8'))"
    )

    conn = connect(lake)
    assert conn.scan_table("items").collect().row(0) == (7, "inline")
    stats = conn.get_stats("items")
    assert stats["min_is_exact"].to_list() == [None, None]
    assert stats["max_is_exact"].to_list() == [None, None]


def test_pruning_proven_by_unreadable_blob(lake):
    lake.file(1, [1, 2], upload=False)
    lake.file(2, [10, 11])
    scan = connect(lake).scan_table("items")
    actual = scan.filter(pl.col("id") >= 10).select("label").collect()
    assert actual["label"].to_list() == ["v10", "v11"]
    with pytest.raises(RuntimeError, match="Azure blob download failed"):
        scan.collect()


@pytest.mark.parametrize(
    "predicate,expected",
    [
        (pl.col("id") == 11, [11]),
        ((pl.col("id") >= 10) & (pl.col("id") < 12), [10, 11]),
        ((pl.col("id") == 2) | (pl.col("id") == 20), [2, 20]),
        (pl.col("id").is_null(), [None]),
        (pl.col("id").is_not_null(), [1, 2, 10, 11, 20]),
        (pl.col("id").is_in([2, 20]), [2, 20]),
        ((pl.col("id") + 1) == 12, [11]),
    ],
)
def test_predicate_matches_eager_polars(lake, predicate, expected):
    lake.file(1, [1, 2])
    lake.file(2, [10, 11, None])
    lake.file(3, [20])
    actual = connect(lake).scan_table("items").filter(predicate).collect()
    assert actual["id"].to_list() == expected


def test_filter_before_limit_and_projection(lake):
    lake.file(1, list(range(20)))
    scan = connect(lake).scan_table("items")
    assert scan.filter(pl.col("id") >= 15).select("label").head(2).collect()[
        "label"
    ].to_list() == ["v15", "v16"]
    assert scan.head(0).collect().schema == {"id": pl.Int64, "label": pl.String}
    assert scan.select(pl.len()).collect().item() == 20
    source = connect(lake)._connection.scan_table("items", None)
    batches = list(source(["id"], pl.col("id") >= 15, 3, 2))
    assert [b.height for b in batches] == [2, 1]
    assert pl.concat(batches)["id"].to_list() == [15, 16, 17]


def test_missing_statistics_never_prune(lake):
    lake.file(1, [100])
    lake.db.execute("DELETE FROM ducklake_file_column_stats")
    assert connect(lake).scan_table("items").filter(pl.col("id") == 100).collect().height == 1


def test_deletions_are_applied_before_filter_and_limit(lake):
    lake.file(1, [10, 11, 12, 13])
    path = lake.parquet("delete.parquet", pl.DataFrame({"row_id": [1]}))
    lake.db.execute(
        "INSERT INTO ducklake_delete_file VALUES (1,1,2,NULL,%s,false,'parquet',NULL,NULL)",
        (path,),
    )
    lake.db.execute("INSERT INTO ducklake_inlined_delete_1 VALUES (1,2,3)")
    conn = connect(lake)
    assert conn.scan_table("items", 1).collect()["id"].to_list() == [10, 11, 12, 13]
    assert conn.scan_table("items", 2).collect()["id"].to_list() == [10, 12, 13]
    assert conn.scan_table("items").head(2).collect()["id"].to_list() == [10, 13]


def test_partial_file_snapshot_filtering(lake):
    lake.file(1, [1, 2])
    path = lake.parquet(
        "partial.parquet",
        pl.DataFrame({"id": [1, 2], "label": ["a", "b"], "_ducklake_internal_snapshot_id": [1, 3]}),
    )
    lake.db.execute("UPDATE ducklake_data_file SET path=%s, partial_max=3", (path,))
    assert connect(lake).scan_table("items", 2).collect()["id"].to_list() == [1]
    assert connect(lake).scan_table("items").collect()["id"].to_list() == [1, 2]


def test_stats_are_catalog_only(lake):
    lake.file(1, [1, 3], upload=False)
    lake.db.execute(
        "INSERT INTO ducklake_table_column_stats VALUES "
        "(1,1,'1','3',false,false,NULL,true,true)"
    )
    stats = connect(lake).get_stats("items")
    assert stats.columns[0] == "column_name"
    assert stats["column_name"].to_list() == ["id", "label"]
    assert stats["min"].to_list() == ["1", None]
    assert stats["max"].to_list() == ["3", None]


def test_environment_fallbacks_are_independent(lake, monkeypatch):
    monkeypatch.setenv("PLDK_DSN", lake.dsn)
    monkeypatch.setenv("PLDK_SAS_KEY", lake.credential)
    monkeypatch.delenv("PLDK_CONN_KEY", raising=False)
    assert Connection().scan_table("items").collect().height == 0
    assert Connection(lake.dsn).get_stats("items").height == 2
    assert Connection(None, lake.credential).get_stats("items").height == 2


def test_cache_refreshes_new_inline_versions(lake):
    conn = connect(lake)
    lake.db.execute("INSERT INTO ducklake_snapshot VALUES (4)")
    lake.db.execute(
        "CREATE TABLE ducklake_inlined_data_1_2 "
        "(LIKE ducklake_inlined_data_1_1 INCLUDING ALL)"
    )
    lake.db.execute("INSERT INTO ducklake_inlined_data_tables VALUES (1,'ducklake_inlined_data_1_2',2)")
    lake.db.execute("INSERT INTO ducklake_schema_versions VALUES (1,2,2)")
    lake.db.execute(
        "INSERT INTO ducklake_inlined_data_1_2 VALUES (1,2,NULL,42,convert_to('yes','UTF8'))"
    )
    assert conn.scan_table("items").collect()["id"].to_list() == [42]


@pytest.mark.parametrize(
    "predicate",
    [pl.col("id").is_between(10, 12), pl.col("id").is_in([10, 11])],
)
def test_range_and_membership_prune_before_opening_blobs(lake, predicate):
    lake.file(1, [1, 2], upload=False)
    lake.file(2, [10, 11])
    assert connect(lake).scan_table("items").filter(predicate).collect()["id"].to_list() == [10, 11]


def test_unsupported_rust_function_uses_exact_python_filter(lake):
    lake.file(1, [0, 1, 2, 3])
    scan = connect(lake).scan_table("items")
    with pytest.warns(UserWarning, match="filtering with Python Polars"):
        actual = scan.filter(pl.col("id").sin() > 0.5).select("label").head(1).collect()
    assert actual["label"].to_list() == ["v1"]


def test_schema_evolution_preserves_column_identity_and_defaults(lake):
    lake.file(1, [1, 2])
    lake.db.execute("UPDATE ducklake_column SET end_snapshot=2 WHERE column_id=2")
    lake.db.execute(
        "INSERT INTO ducklake_column VALUES "
        "(2,1,'renamed','varchar',1,NULL,2,NULL,NULL),"
        "(3,1,'added','int32',2,NULL,2,NULL,'42')"
    )
    lake.db.execute(
        "INSERT INTO ducklake_inlined_data_1_1 VALUES (1,1,NULL,7,convert_to('inline','UTF8'))"
    )
    conn = connect(lake)
    actual = conn.scan_table("items").collect()
    assert actual["renamed"].to_list() == ["v1", "v2", "inline"]
    assert actual["added"].to_list() == [42, 42, 42]
    assert conn.scan_table("items", 1).collect().columns == ["id", "label"]


def test_native_parquet_field_ids_not_physical_names(lake):
    import io
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema(
        [
            pa.field("physical_id", pa.int64(), metadata={b"PARQUET:field_id": b"1"}),
            pa.field("old_label", pa.string(), metadata={b"PARQUET:field_id": b"2"}),
        ]
    )
    table = pa.Table.from_pydict({"physical_id": [7], "old_label": ["native"]}, schema=schema)
    buffer = io.BytesIO()
    pq.write_table(table, buffer)
    path = f"{lake.prefix}/native.parquet"
    lake.container.upload_blob(path, buffer.getvalue())
    lake.blobs.append(path)
    lake.db.execute(
        "INSERT INTO ducklake_data_file VALUES "
        "(1,1,1,NULL,NULL,%s,false,'parquet',NULL,NULL,NULL)",
        (f"{lake.container.url.split('?')[0]}/{path}",),
    )
    assert connect(lake).scan_table("items").collect().row(0) == (7, "native")


def test_stats_null_counts_include_inline_rows_without_parquet(lake):
    lake.file(1, [1, None], upload=False)
    lake.db.execute("INSERT INTO ducklake_inlined_data_1_1 VALUES (1,2,NULL,NULL,NULL)")
    stats = connect(lake).get_stats("items")
    assert stats.filter(pl.col("column_name") == "id")["null_values"].item() == 2
    lake.db.execute("INSERT INTO ducklake_inlined_delete_1 VALUES (1,0,3)")
    stats = connect(lake).get_stats("items")
    assert stats["null_values"].to_list() == [None, None]


def test_scan_pins_snapshot_before_later_commits(lake):
    lake.file(1, [7])
    conn = connect(lake)
    pinned = conn.scan_table("items")
    lake.db.execute("INSERT INTO ducklake_snapshot VALUES (4)")
    lake.db.execute("UPDATE ducklake_data_file SET end_snapshot=4")
    lake.file(2, [99], begin=4)
    assert pinned.collect()["id"].to_list() == [7]
    assert conn.scan_table("items").collect()["id"].to_list() == [99]


def test_inline_strings_preserve_embedded_null_bytes(lake):
    lake.db.execute(
        "INSERT INTO ducklake_inlined_data_1_1 VALUES (1,1,NULL,7,%s)",
        (b"before\x00after",),
    )
    assert connect(lake).scan_table("items").collect()["label"].item() == "before\x00after"


def test_partial_delete_file_uses_requested_snapshot(lake):
    lake.file(1, [10, 11, 12])
    path = lake.parquet(
        "partial-deletes.parquet",
        pl.DataFrame({"row_id": [0, 2], "_ducklake_internal_snapshot_id": [2, 3]}),
    )
    lake.db.execute(
        "INSERT INTO ducklake_delete_file VALUES (1,1,2,NULL,%s,false,'parquet',NULL,3)",
        (path,),
    )
    conn = connect(lake)
    assert conn.scan_table("items", 2).collect()["id"].to_list() == [11, 12]
    assert conn.scan_table("items", 3).collect()["id"].to_list() == [11]


def test_inline_uuid_date_and_decimal_encodings(lake):
    from datetime import date
    from decimal import Decimal

    lake.db.execute(
        "ALTER TABLE ducklake_inlined_data_1_1 "
        "ADD COLUMN token uuid, ADD COLUMN day text, ADD COLUMN amount text"
    )
    lake.db.execute(
        "INSERT INTO ducklake_column VALUES "
        "(3,1,'token','uuid',2,NULL,1,NULL,NULL),"
        "(4,1,'day','date',3,NULL,1,NULL,NULL),"
        "(5,1,'amount','decimal(12,2)',4,NULL,1,NULL,NULL)"
    )
    token = "550e8400-e29b-41d4-a716-446655440000"
    lake.db.execute(
        "INSERT INTO ducklake_inlined_data_1_1 VALUES "
        "(1,1,NULL,7,NULL,%s,'2024-01-15','123.45')",
        (token,),
    )
    frame = connect(lake).scan_table("items").collect()
    assert frame["token"].item() == token
    assert frame["day"].item() == date(2024, 1, 15)
    assert frame["amount"].item() == Decimal("123.45")


@pytest.mark.parametrize("scheme", ["https", "az", "abfss"])
def test_relative_paths_and_azure_credential_forms(lake, scheme):
    from urllib.parse import urlsplit

    url = urlsplit(lake.credential)
    account = url.hostname.split(".")[0]
    container = lake.container.container_name
    lake.parquet("logical/table/data.parquet", pl.DataFrame({"id": [7], "label": ["relative"]}))
    if scheme == "https":
        root = f"https://{url.hostname}/{container}/{lake.prefix}/"
        credential = url.query
    elif scheme == "az":
        root = f"az://{container}/{lake.prefix}/"
        credential = f"AccountName={account};SharedAccessSignature={url.query}"
    else:
        root = f"abfss://{container}@{account}.dfs.core.windows.net/{lake.prefix}/"
        credential = lake.credential
    lake.db.execute("INSERT INTO ducklake_metadata VALUES ('data_path',%s,NULL,NULL)", (root,))
    lake.db.execute("UPDATE ducklake_schema SET path='logical/', path_is_relative=true")
    lake.db.execute("UPDATE ducklake_table SET path='table/', path_is_relative=true")
    lake.db.execute(
        "INSERT INTO ducklake_data_file VALUES "
        "(1,1,1,NULL,NULL,'data.parquet',true,'parquet',NULL,1,NULL)"
    )
    assert Connection(lake.dsn, credential).scan_table("items").collect().row(0) == (7, "relative")


def test_stats_do_not_require_supported_reader_types(lake):
    lake.db.execute("UPDATE ducklake_column SET column_type='variant' WHERE column_id=1")
    conn = connect(lake)
    assert conn.get_stats("items")["column_name"].to_list() == ["id", "label"]
    with pytest.raises(RuntimeError, match="Unsupported DuckLake type"):
        conn.scan_table("items")
