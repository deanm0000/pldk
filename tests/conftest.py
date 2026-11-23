from __future__ import annotations

import io
import os
from dataclasses import dataclass, field
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

import polars as pl
import psycopg
from psycopg import sql
import pytest
from azure.storage.blob import ContainerClient


DDL = """
CREATE TABLE ducklake_metadata (key text, value text, scope text, scope_id bigint);
CREATE TABLE ducklake_snapshot (snapshot_id bigint PRIMARY KEY);
CREATE TABLE ducklake_schema (
    schema_id bigint, schema_name text, path text, path_is_relative boolean,
    parent_schema_id bigint, begin_snapshot bigint, end_snapshot bigint);
CREATE TABLE ducklake_table (
    table_id bigint, schema_id bigint, table_name text, path text,
    path_is_relative boolean, begin_snapshot bigint, end_snapshot bigint);
CREATE TABLE ducklake_column (
    column_id bigint, table_id bigint, column_name text, column_type text,
    column_order bigint, parent_column bigint, begin_snapshot bigint,
    end_snapshot bigint, initial_default text);
CREATE TABLE ducklake_data_file (
    data_file_id bigint, table_id bigint, begin_snapshot bigint, end_snapshot bigint,
    file_order bigint, path text, path_is_relative boolean, file_format text,
    encryption_key text, mapping_id bigint, partial_max bigint);
CREATE TABLE ducklake_file_column_stats (
    data_file_id bigint, table_id bigint, column_id bigint, min_value text,
    max_value text, null_count bigint, value_count bigint, contains_nan boolean);
CREATE TABLE ducklake_delete_file (
    data_file_id bigint, table_id bigint, begin_snapshot bigint, end_snapshot bigint,
    path text, path_is_relative boolean, format text, encryption_key text, partial_max bigint);
CREATE TABLE ducklake_inlined_data_tables (
    table_id bigint, table_name text, schema_version bigint);
CREATE TABLE ducklake_schema_versions (
    table_id bigint, begin_snapshot bigint, schema_version bigint);
CREATE TABLE ducklake_name_mapping (
    mapping_id bigint, parent_column bigint, source_name text,
    target_field_id bigint, is_partition boolean);
CREATE TABLE ducklake_table_column_stats (
    table_id bigint, column_id bigint, min_value text, max_value text,
    contains_null boolean, contains_nan boolean, extra_stats text,
    min_is_exact boolean, max_is_exact boolean);
CREATE TABLE ducklake_inlined_data_1_1 (
    _ducklake_row_id bigint, _ducklake_begin_snapshot bigint,
    _ducklake_end_snapshot bigint, id bigint, label bytea);
CREATE TABLE ducklake_inlined_delete_1 (
    file_id bigint, row_id bigint, begin_snapshot bigint);
"""


@dataclass
class Lake:
    db: psycopg.Connection
    dsn: str = field(repr=False)
    credential: str = field(repr=False)
    container: ContainerClient = field(repr=False)
    prefix: str
    blobs: list[str] = field(default_factory=list)

    def parquet(self, name: str, data: pl.DataFrame) -> str:
        buffer = io.BytesIO()
        data.write_parquet(buffer)
        path = f"{self.prefix}/{name}"
        self.container.upload_blob(path, buffer.getvalue(), overwrite=False)
        self.blobs.append(path)
        return f"{self.container.url.split('?')[0]}/{path}"

    def file(
        self,
        file_id: int,
        values: list[int | None],
        *,
        begin: int = 1,
        end: int | None = None,
        upload: bool = True,
    ) -> None:
        data = pl.DataFrame(
            {"id": values, "label": [f"v{x}" if x is not None else None for x in values]},
            schema={"id": pl.Int64, "label": pl.String},
        )
        path = (
            self.parquet(f"{file_id}.parquet", data)
            if upload
            else f"{self.container.url.split('?')[0]}/{self.prefix}/missing-{file_id}.parquet"
        )
        self.db.execute(
            "INSERT INTO ducklake_data_file VALUES (%s,1,%s,%s,NULL,%s,false,'parquet',NULL,1,NULL)",
            (file_id, begin, end, path),
        )
        non_null = [v for v in values if v is not None]
        self.db.execute(
            "INSERT INTO ducklake_file_column_stats VALUES (%s,1,1,%s,%s,%s,%s,false)",
            (
                file_id,
                str(min(non_null)) if non_null else None,
                str(max(non_null)) if non_null else None,
                len(values) - len(non_null),
                len(non_null),
            ),
        )


@pytest.fixture
def lake():
    dsn = os.environ.get("PLDK_TEST_DSN")
    credential = os.environ.get("PLDK_TEST_AZURE_URL")
    if not dsn or not credential:
        pytest.skip("Set PLDK_TEST_DSN and PLDK_TEST_AZURE_URL for live service tests")
    schema = f"pldk_test_{uuid4().hex}"
    container = ContainerClient.from_container_url(credential)
    with psycopg.connect(dsn, autocommit=True) as db:
        db.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        db.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(schema)))
        parts = urlsplit(dsn)
        query = parse_qsl(parts.query)
        query.append(("options", f"-csearch_path={schema}"))
        scoped_dsn = urlunsplit(parts._replace(query=urlencode(query)))
        fixture = Lake(db, scoped_dsn, credential, container, schema)
        try:
            db.execute(DDL)
            db.execute("INSERT INTO ducklake_metadata VALUES ('version','1.1',NULL,NULL)")
            db.execute("INSERT INTO ducklake_snapshot VALUES (1),(2),(3)")
            db.execute("INSERT INTO ducklake_schema VALUES (1,'main','',false,NULL,1,NULL)")
            db.execute("INSERT INTO ducklake_table VALUES (1,1,'items','',false,1,NULL)")
            db.execute(
                "INSERT INTO ducklake_column VALUES "
                "(1,1,'id','int64',0,NULL,1,NULL,NULL),"
                "(2,1,'label','varchar',1,NULL,1,NULL,NULL)"
            )
            db.execute(
                "INSERT INTO ducklake_name_mapping VALUES "
                "(1,NULL,'id',1,false),(1,NULL,'label',2,false)"
            )
            db.execute("INSERT INTO ducklake_schema_versions VALUES (1,1,1)")
            db.execute(
                "INSERT INTO ducklake_inlined_data_tables VALUES (1,'ducklake_inlined_data_1_1',1)"
            )
            yield fixture
        finally:
            try:
                for path in fixture.blobs:
                    container.delete_blob(path)
            finally:
                try:
                    db.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
                finally:
                    container.close()
