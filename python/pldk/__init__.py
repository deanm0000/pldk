"""Read-only DuckLake access through the Polars IO plugin interface."""

from __future__ import annotations

import polars as pl
from polars.io.plugins import register_io_source

from ._pldk import Connection as _Connection

__all__ = ["Connection"]


class Connection:
    """A PostgreSQL DuckLake catalog with Azure object storage.

    Each omitted argument is independently resolved from the environment:
    PLDK_DSN and PLDK_CONN_KEY (or PLDK_SAS_KEY). Azure credentials may be a
    connection string, a container SAS URL, or a SAS query string when paths
    identify the storage account. PostgreSQL's search_path selects the catalog.
    """

    def __init__(
        self, dsn: str | None = None, object_store_key: str | None = None
    ) -> None:
        self._connection = _Connection(dsn, object_store_key)

    def scan_table(
        self, table_name: str, snapshot_id: int | None = None
    ) -> pl.LazyFrame:
        """Pin a catalog snapshot and return a reusable lazy scan."""
        source = self._connection.scan_table(table_name, snapshot_id)
        return register_io_source(io_source=source, schema=source.schema())

    def get_stats(self, table_name: str) -> pl.DataFrame:
        """Return current catalog column statistics, without opening Parquet."""
        return self._connection.get_stats(table_name)
