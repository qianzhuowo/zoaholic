"""
Cloudflare D1 工具封装。

提供：
- D1 prepare/bind/all/first/run/exec 的 Python 包装
- ensure_schema()：初始化/补齐核心表结构
- 常用时间格式转换工具
"""

from __future__ import annotations

import inspect
from datetime import datetime, timezone
from typing import Any, Optional

from core.log_config import logger
from core.cf_env import get_d1


class D1Error(RuntimeError):
    """D1 操作异常。"""


def d1_datetime(value: datetime | str | None) -> Optional[str]:
    """将 datetime 转成 D1(SQLite) 可比较的 UTC 字符串。"""
    if value is None:
        return None

    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        # 兼容 ISO 格式
        try:
            if s.endswith("Z"):
                s = s[:-1] +"+00:00"
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return s

    dt = value
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def parse_d1_datetime(value: Any) -> Optional[datetime]:
    """将 D1 返回的时间值转换为带 UTC 时区的 datetime。"""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)

    s = str(value).strip()
    if not s:
        return None

    try:
        # SQLite 默认 CURRENT_TIMESTAMP: YYYY-MM-DD HH:MM:SS
        if "T" not in s and "+" not in s and "Z" not in s:
            dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
            return dt.replace(tzinfo=timezone.utc)

        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def _call(obj: Any, method_name: str, *args) -> Any:
    fn = getattr(obj, method_name, None)
    if fn is None or not callable(fn):
        raise D1Error(f"D1 object missing callable method: {method_name}")
    return await _maybe_await(fn(*args))


class D1Statement:
    def __init__(self, stmt: Any):
        self._stmt = stmt

    def bind(self, *params) -> "D1Statement":
        if not params:
            return self
        bound = self._stmt.bind(*params)
        if bound is None:
            return self
        return D1Statement(bound)

    async def all(self) -> list[dict[str, Any]]:
        result = await _call(self._stmt, "all")
        if isinstance(result, dict):
            rows = result.get("results")
            if isinstance(rows, list):
                return rows
        if isinstance(result, list):
            return result
        return []

    async def first(self) -> Optional[dict[str, Any]]:
        # D1 的 first() 通常直接返回一行对象，也兼容 all()[0]
        result = await _call(self._stmt, "first")
        if isinstance(result, dict):
            return result
        if isinstance(result, list) and result:
            first = result[0]
            return first if isinstance(first, dict) else None
        return None

    async def run(self) -> Any:
        return await _call(self._stmt, "run")


class D1Database:
    def __init__(self, db: Any):
        self._db = db

    @classmethod
    def from_binding(cls, binding_name: Optional[str] = None, *, required: bool = True) -> Optional["D1Database"]:
        db = get_d1(binding_name=binding_name, required=required)
        if db is None:
            return None
        return cls(db)

    async def prepare(self, sql: str) -> D1Statement:
        stmt = await _call(self._db, "prepare", sql)
        return D1Statement(stmt)

    async def statement(self, sql: str, *params) -> D1Statement:
        stmt = await self.prepare(sql)
        return stmt.bind(*params)

    async def all(self, sql: str, *params) -> list[dict[str, Any]]:
        stmt = await self.statement(sql, *params)
        return await stmt.all()

    async def first(self, sql: str, *params) -> Optional[dict[str, Any]]:
        stmt = await self.statement(sql, *params)
        return await stmt.first()

    async def run(self, sql: str, *params) -> Any:
        stmt = await self.statement(sql, *params)
        return await stmt.run()

    async def exec(self, sql: str) -> Any:
        if not sql or not sql.strip():
            return None
        return await _call(self._db, "exec", sql)


async def get_d1_database(binding_name: Optional[str] = None, *, required: bool = True) -> Optional[D1Database]:
    db = D1Database.from_binding(binding_name=binding_name, required=required)
    return db


DEFAULT_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS request_stats (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  request_id TEXT,
  endpoint TEXT,
  client_ip TEXT,
  process_time REAL,
  first_response_time REAL,
  content_start_time REAL,
  provider TEXT,
  model TEXT,
  api_key TEXT,
  success INTEGER DEFAULT 0,
  status_code INTEGER,
  is_flagged INTEGER DEFAULT 0,
  text TEXT,
  prompt_tokens INTEGER DEFAULT 0,
  completion_tokens INTEGER DEFAULT 0,
  total_tokens INTEGER DEFAULT 0,
  prompt_price REAL DEFAULT 0.0,
  completion_price REAL DEFAULT 0.0,
  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
  provider_id TEXT,
  provider_key_index INTEGER,
  api_key_name TEXT,
  api_key_group TEXT,
  retry_count INTEGER DEFAULT 0,
  retry_path TEXT,
  request_headers TEXT,
  request_body TEXT,
  upstream_request_headers TEXT,
  upstream_request_body TEXT,
  upstream_response_body TEXT,
  response_body TEXT,
  raw_data_expires_at DATETIME
);

CREATE TABLE IF NOT EXISTS channel_stats (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  request_id TEXT,
  provider TEXT,
  model TEXT,
  api_key TEXT,
  provider_api_key TEXT,
  success INTEGER DEFAULT 0,
  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS admin_user (
  id INTEGER PRIMARY KEY,
  username TEXT NOT NULL,
  password_hash TEXT NOT NULL,
  jwt_secret TEXT
);

CREATE TABLE IF NOT EXISTS app_config (
  id INTEGER PRIMARY KEY,
  config_json TEXT,
  config_yaml TEXT,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_request_stats_provider ON request_stats(provider);
CREATE INDEX IF NOT EXISTS idx_request_stats_model ON request_stats(model);
CREATE INDEX IF NOT EXISTS idx_request_stats_api_key ON request_stats(api_key);
CREATE INDEX IF NOT EXISTS idx_request_stats_success ON request_stats(success);
CREATE INDEX IF NOT EXISTS idx_request_stats_status_code ON request_stats(status_code);
CREATE INDEX IF NOT EXISTS idx_request_stats_timestamp ON request_stats(timestamp);
CREATE INDEX IF NOT EXISTS idx_request_stats_provider_id ON request_stats(provider_id);

CREATE INDEX IF NOT EXISTS idx_channel_stats_provider ON channel_stats(provider);
CREATE INDEX IF NOT EXISTS idx_channel_stats_model ON channel_stats(model);
CREATE INDEX IF NOT EXISTS idx_channel_stats_provider_api_key ON channel_stats(provider_api_key);
CREATE INDEX IF NOT EXISTS idx_channel_stats_timestamp ON channel_stats(timestamp);

CREATE INDEX IF NOT EXISTS idx_admin_user_username ON admin_user(username);
CREATE INDEX IF NOT EXISTS idx_app_config_updated_at ON app_config(updated_at);
""".strip()


_COLUMN_PATCHES: dict[str, dict[str, str]] = {
    "request_stats": {
        "content_start_time": "content_start_time REAL",
        "success": "success INTEGER DEFAULT 0",
        "status_code": "status_code INTEGER",
        "prompt_price": "prompt_price REAL DEFAULT 0.0",
        "completion_price": "completion_price REAL DEFAULT 0.0",
        "provider_id": "provider_id TEXT",
        "provider_key_index": "provider_key_index INTEGER",
        "api_key_name": "api_key_name TEXT",
        "api_key_group": "api_key_group TEXT",
        "retry_count": "retry_count INTEGER DEFAULT 0",
        "retry_path":"retry_path TEXT",
        "request_headers": "request_headers TEXT",
        "request_body": "request_body TEXT",
        "upstream_request_headers": "upstream_request_headers TEXT",
        "upstream_request_body": "upstream_request_body TEXT",
        "upstream_response_body": "upstream_response_body TEXT",
        "response_body": "response_body TEXT",
        "raw_data_expires_at": "raw_data_expires_at DATETIME",
    },
    "channel_stats": {
        "provider_api_key": "provider_api_key TEXT",
    },
    "admin_user": {
        "jwt_secret": "jwt_secret TEXT",
    },
    "app_config": {
        "config_json": "config_json TEXT",
        "config_yaml": "config_yaml TEXT",
        "updated_at": "updated_at DATETIME DEFAULT CURRENT_TIMESTAMP",
    },
}


async def _table_columns(db: D1Database, table_name: str) -> set[str]:
    rows = await db.all(f"PRAGMA table_info({table_name})")
    names: set[str] = set()
    for row in rows:
        if isinstance(row, dict):
            name = row.get("name")
            if isinstance(name, str) and name:
                names.add(name)
    return names


async def _ensure_columns(db: D1Database, table_name: str, patches: dict[str, str]) -> None:
    existing = await _table_columns(db, table_name)
    for col_name, ddl in patches.items():
        if col_name in existing:
            continue
        try:
            await db.exec(f"ALTER TABLE {table_name} ADD COLUMN {ddl}")
            logger.info("D1 schema patch: added column '%s.%s'", table_name, col_name)
        except Exception as e:
            logger.warning("D1 schema patch failed for %s.%s: %s", table_name, col_name, e)


async def ensure_schema(db: Optional[D1Database] = None, schema_sql: Optional[str] = None) -> None:
    """确保 D1 核心表结构存在，并尽力补齐历史缺失列。"""
    database = db or await get_d1_database(required=True)
    if database is None:
        raise D1Error("D1 database binding not found")

    await database.exec(schema_sql or DEFAULT_SCHEMA_SQL)

    for table_name, patch in _COLUMN_PATCHES.items():
        await _ensure_columns(database, table_name, patch)
