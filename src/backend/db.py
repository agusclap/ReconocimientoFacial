import os
import psycopg2
from psycopg2.extras import RealDictCursor, register_default_jsonb
from pgvector.psycopg2 import register_vector


DB_NAME = os.getenv('DB_NAME', 'proyectogpi1')
DB_USER = os.getenv('DB_USER', 'admin')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'nariga')
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '15432')


class DB:
    def __init__(self):
        self.conn = None


    def connect(self):
        if self.conn is None or self.conn.closed:
            self.conn = psycopg2.connect(
            dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD,
            host=DB_HOST, port=DB_PORT
        )
        register_vector(self.conn)
        register_default_jsonb(self.conn)
        self.conn.autocommit = True
        return self.conn


def query(self, sql, params=None):
    conn = self.connect()
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, params or [])
    if cur.description:
        return cur.fetchall()
    return []


def execute(self, sql, params=None):
    conn = self.connect()
    with conn.cursor() as cur:
        cur.execute(sql, params or [])


_db = DB()


def db():
    return _db