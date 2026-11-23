import pytest

from pldk import Connection


def test_missing_environment_is_reported(monkeypatch):
    for key in ("PLDK_DSN", "PLDK_CONN_KEY", "PLDK_SAS_KEY"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(ValueError, match="PLDK_DSN"):
        Connection()
    with pytest.raises(ValueError, match="PLDK_CONN_KEY"):
        Connection("postgresql://localhost/nonexistent")


@pytest.mark.parametrize("dsn", ["duckdb://test", "mysql://localhost/db", "sqlite://test.db"])
def test_unsupported_catalog_backends(dsn):
    with pytest.raises(ValueError, match="Only PostgreSQL"):
        Connection(dsn, "unused")
