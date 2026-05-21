"""Postgres connection management with pgvector registered."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import psycopg
from pgvector.psycopg import register_vector

from . import config

_SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schema.sql"


def connect(dsn: Optional[str] = None) -> psycopg.Connection:
    """Open a connection with the vector type adapter registered."""
    conn = psycopg.connect(dsn or config.PG_DSN, autocommit=True)
    register_vector(conn)
    return conn


def init_db(dsn: Optional[str] = None) -> None:
    """Create the extension, tables, and indexes from schema.sql."""
    sql = _SCHEMA_PATH.read_text()
    with connect(dsn) as conn:
        conn.execute(sql)
