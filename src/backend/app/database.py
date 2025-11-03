"""Database helper utilities for the backend API."""
from __future__ import annotations

import os
from typing import Optional, Sequence

import psycopg2
from pgvector.psycopg2 import register_vector
from psycopg2.extras import RealDictCursor, register_default_jsonb
from psycopg2.extensions import connection


DB_NAME = os.getenv("DB_NAME", "proyectogpi1")
DB_USER = os.getenv("DB_USER", "admin")
DB_PASSWORD = os.getenv("DB_PASSWORD", "nariga")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "15432")


class Database:
    """Simple helper around psycopg2 connections."""

    def __init__(self) -> None:
        self._conn: Optional[connection] = None

    def connect(self) -> connection:
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(
                dbname=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD,
                host=DB_HOST,
                port=DB_PORT,
            )
            self._conn.autocommit = True
            register_vector(self._conn)
            register_default_jsonb(self._conn)
        return self._conn

    def query(self, sql: str, params: Optional[Sequence[object]] = None):
        conn = self.connect()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params or [])
            if cur.description:
                return cur.fetchall()
        return []

    def execute(self, sql: str, params: Optional[Sequence[object]] = None) -> None:
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(sql, params or [])


_db = Database()


def db() -> Database:
    return _db


__all__ = ["db", "Database"]
