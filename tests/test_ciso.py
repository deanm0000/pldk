import os

import polars as pl
from pldk import Connection


def test_ciso():
    conn = Connection(os.environ["CISO_DSN"], os.environ["Synblob"])
