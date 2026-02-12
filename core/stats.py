"""
数据库统计模块

负责：
- 数据库表初始化和迁移
- 请求统计写入 (RequestStat)
- 渠道统计写入 (ChannelStat)
- Token 使用量查询和聚合
- 成本计算
"""

from __future__ import annotations

import json
import asyncio
from asyncio import Semaphore
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, field_serializer

from core.env import env_bool
from core.log_config import logger
from db import (
    Base,
    RequestStat,
    ChannelStat,
    AppConfig,
    AdminUser,
    db_engine,
    async_session,
    DISABLE_DATABASE,
    DB_TYPE,
    IS_D1_MODE,
)
from core.d1 import get_d1_database, ensure_schema, d1_datetime

if not IS_D1_MODE:
    from sqlalchemy import inspect, text, func, select
    from sqlalchemy.sql import sqltypes
else:
    inspect = None
    text = None
    func = None
    select = None
    sqltypes = None

# SQLite 写入重试配置
SQLITE_MAX_RETRIES = 3
SQLITE_RETRY_DELAY = 0.5  # 初始重试延迟（秒）

is_debug = env_bool("DEBUG", False)

_db_kind = (DB_TYPE or "sqlite").lower()
if _db_kind == "sqlite":
    db_semaphore = Semaphore(1)
    logger.info("Database semaphore configured for SQLite (1 concurrent writer).")
else:
    # PostgreSQL / D1 都允许较高并发
    db_semaphore = Semaphore(50)
    logger.info("Database semaphore configured for %s (50 concurrent writers).", _db_kind)


# ============== Pydantic Models ==============

class TokenUsageEntry(BaseModel):
    api_key_prefix: str
    model: str
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    request_count: int


class QueryDetails(BaseModel):
    model_config = {"protected_namespaces": ()}

    start_datetime: Optional[str] = None
    end_datetime: Optional[str] = None
    api_key_filter: Optional[str] = None
    model_filter: Optional[str] = None
    credits: Optional[str] = None
    total_cost: Optional[str] = None
    balance: Optional[str] = None


class TokenUsageResponse(BaseModel):
    usage: List[TokenUsageEntry]
    query_details: QueryDetails


class ChannelKeyRanking(BaseModel):
    api_key: str
    success_count: int
    total_requests: int
    success_rate: float


class ChannelKeyRankingsResponse(BaseModel):
    rankings: List[ChannelKeyRanking]
    query_details: QueryDetails


class TokenInfo(BaseModel):
    api_key_prefix: str
    model: str
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    request_count: int


class ApiKeyState(BaseModel):
    credits: float
    created_at: datetime
    all_tokens_info: List[Dict[str, Any]]
    total_cost: float
    enabled: bool

    @field_serializer("created_at")
    def serialize_dt(self, dt: datetime):
        return dt.isoformat()


class ApiKeysStatesResponse(BaseModel):
    api_keys_states: Dict[str, ApiKeyState]


# ============== 数据库初始化 ==============

def _map_sa_type_to_sql_type(sa_type):
    """将 SQLAlchemy 类型映射到 SQL 类型字符串"""
    if not sqltypes:
        return "TEXT"
    type_map = {
        sqltypes.Integer: "INTEGER",
        sqltypes.String: "TEXT",
        sqltypes.Float: "REAL",
        sqltypes.Boolean: "BOOLEAN",
        sqltypes.DateTime: "DATETIME",
        sqltypes.Text: "TEXT",
    }
    return type_map.get(type(sa_type), "TEXT")


def _get_default_sql(default):
    """生成列默认值的 SQL 片段"""
    if default is None:
        return ""
    if isinstance(default.arg, bool):
        return f" DEFAULT {str(default.arg).upper()}"
    if isinstance(default.arg, (int, float)):
        return f" DEFAULT {default.arg}"
    if isinstance(default.arg, str):
        return f" DEFAULT '{default.arg}'"
    return ""


async def create_tables():
    """创建数据库表并执行简易列迁移"""
    if DISABLE_DATABASE:
        return

    if IS_D1_MODE:
        db = await get_d1_database(required=False)
        if db is None:
            logger.warning("D1 schema init skipped: D1 binding not found in current context")
            return
        await ensure_schema(db)
        return

    async with db_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

        # 检查并添加缺失的列 - 扩展此简易迁移以支持 SQLite 和 PostgreSQL
        db_type = (DB_TYPE or "sqlite").lower()
        if db_type in ["sqlite", "postgres"]:

            def check_and_add_columns(connection):
                inspector_obj = inspect(connection)
                for table in [RequestStat, ChannelStat, AppConfig, AdminUser]:
                    table_name = table.__tablename__
                    existing_columns = {col["name"] for col in inspector_obj.get_columns(table_name)}

                    for column_name, column in table.__table__.columns.items():
                        if column_name not in existing_columns:
                            col_type = column.type.compile(connection.dialect)
                            default = _get_default_sql(column.default) if db_type == "sqlite" else ""

                            connection.execute(
                                text(
                                    f'ALTER TABLE "{table_name}" ADD COLUMN "{column_name}" {col_type}{default}'
                                )
                            )
                            logger.info(
                                "Added column '%s' (%s) to table '%s'.",
                                column_name,
                                col_type,
                                table_name,
                            )

            await conn.run_sync(check_and_add_columns)


# ============== 成本计算 ==============

def get_current_model_prices(app, model_name: str):
    """
    根据当前配置偏好，返回指定模型的 prompt_price 和 completion_price（单位：$/M tokens）
    """
    from utils import safe_get

    try:
        model_price = safe_get(app.state.config, "preferences", "model_price", default={})
        price_str = next(
            (model_price[k] for k in model_price.keys() if model_name and model_name.startswith(k)),
            model_price.get("default", "0.3,1"),
        )
        parts = [p.strip() for p in str(price_str).split(",")]
        prompt_price = float(parts[0]) if len(parts) > 0 and parts[0] != "" else 0.3
        completion_price = float(parts[1]) if len(parts) > 1 and parts[1] != "" else 1.0
        return prompt_price, completion_price
    except Exception:
        return 0.3, 1.0


async def compute_total_cost_from_db(
    filter_api_key: Optional[str] = None,
    start_dt_obj: Optional[datetime] = None,
) -> float:
    """
    直接从数据库历史记录累计成本：
    sum((prompt_tokens*prompt_price + completion_tokens*completion_price)/1e6)
    """
    if DISABLE_DATABASE:
        return 0.0

    if IS_D1_MODE:
        db = await get_d1_database(required=False)
        if db is None:
            return 0.0

        sql = (
            "SELECT COALESCE(SUM((COALESCE(prompt_tokens, 0) * COALESCE(prompt_price, 0.3) "
            "+ COALESCE(completion_tokens, 0) * COALESCE(completion_price, 1.0)) / 1000000.0), 0.0) AS total_cost "
            "FROM request_stats WHERE 1=1"
        )
        params: list[Any] = []
        if filter_api_key:
            sql += " AND api_key = ?"
            params.append(filter_api_key)
        if start_dt_obj:
            sql += " AND timestamp >= ?"
            params.append(d1_datetime(start_dt_obj))

        row = await db.first(sql, *params)
        try:
            return float((row or {}).get("total_cost") or 0.0)
        except Exception:
            return 0.0

    async with async_session() as session:
        expr = (
            (
                func.coalesce(RequestStat.prompt_tokens, 0) * func.coalesce(RequestStat.prompt_price, 0.3)
                + func.coalesce(RequestStat.completion_tokens, 0) * func.coalesce(RequestStat.completion_price, 1.0)
            )
            / 1000000.0
        )
        query = select(func.coalesce(func.sum(expr), 0.0))
        if filter_api_key:
            query = query.where(RequestStat.api_key == filter_api_key)
        if start_dt_obj:
            query = query.where(RequestStat.timestamp >= start_dt_obj)
        result = await session.execute(query)
        total_cost = result.scalar_one() or 0.0
        try:
            total_cost = float(total_cost)
        except Exception:
            total_cost = 0.0
        return total_cost


# ============== 统计写入 ==============

_D1_REQUEST_STAT_COLUMNS = {
    "request_id",
    "endpoint",
    "client_ip",
    "process_time",
    "first_response_time",
    "content_start_time",
    "provider",
    "model",
    "api_key",
    "success",
    "status_code",
    "is_flagged",
    "text",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "prompt_price",
    "completion_price",
    "provider_id",
    "provider_key_index",
    "api_key_name",
    "api_key_group",
    "retry_count",
    "retry_path",
    "request_headers",
    "request_body",
    "upstream_request_headers",
    "upstream_request_body",
    "upstream_response_body",
    "response_body",
    "raw_data_expires_at",
}

_D1_CHANNEL_STAT_COLUMNS = {
    "request_id",
    "provider",
    "model",
    "api_key",
    "provider_api_key",
    "success",
}


def _to_d1_value(value: Any) -> Any:
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, datetime):
        return d1_datetime(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, str):
        return value.replace("\x00", "")
    return value


async def _insert_d1(table: str, payload: dict[str, Any]) -> None:
    db = await get_d1_database(required=False)
    if db is None:
        raise RuntimeError("D1 database binding not found in current request context")

    columns = list(payload.keys())
    placeholders = ", ".join(["?"] * len(columns))
    columns_sql = ", ".join(columns)
    sql = f"INSERT INTO {table} ({columns_sql}) VALUES ({placeholders})"
    values = [_to_d1_value(payload[c]) for c in columns]
    await db.run(sql, *values)


async def update_stats(current_info: dict, app=None, get_model_prices_func=None):
    """更新请求统计到数据库"""
    if DISABLE_DATABASE:
        return

    # 在成功请求时，快照当前价格，写入数据库
    try:
        if current_info.get("success") and current_info.get("model"):
            if get_model_prices_func:
                prompt_price, completion_price = get_model_prices_func(current_info["model"])
            elif app:
                prompt_price, completion_price = get_current_model_prices(app, current_info["model"])
            else:
                prompt_price, completion_price = 0.3, 1.0
            current_info["prompt_price"] = prompt_price
            current_info["completion_price"] = completion_price
    except Exception:
        pass

    if IS_D1_MODE:
        for attempt in range(SQLITE_MAX_RETRIES):
            try:
                async with db_semaphore:
                    filtered_info = {k: v for k, v in current_info.items() if k in _D1_REQUEST_STAT_COLUMNS}
                    await _insert_d1("request_stats", filtered_info)

                check_key = current_info.get("api_key")
                if app and check_key and hasattr(app.state, "paid_api_keys_states"):
                    if check_key in app.state.paid_api_keys_states and current_info.get("total_tokens", 0) > 0:
                        await update_paid_api_keys_states(app, check_key)
                return
            except Exception as e:
                if attempt < SQLITE_MAX_RETRIES - 1:
                    delay = SQLITE_RETRY_DELAY * (2 ** attempt)
                    await asyncio.sleep(delay)
                else:
                    logger.error("Error updating stats (d1): %s", str(e))
                    if is_debug:
                        import traceback

                        traceback.print_exc()
        return

    # SQLAlchemy 路径
    for attempt in range(SQLITE_MAX_RETRIES):
        try:
            async with db_semaphore:
                async with async_session() as session:
                    async with session.begin():
                        columns = [column.key for column in RequestStat.__table__.columns]
                        filtered_info = {k: v for k, v in current_info.items() if k in columns}

                        for key, value in filtered_info.items():
                            if isinstance(value, str):
                                filtered_info[key] = value.replace("\x00", "")

                        new_request_stat = RequestStat(**filtered_info)
                        session.add(new_request_stat)
                        await session.commit()

            check_key = current_info.get("api_key")
            if app and check_key and hasattr(app.state, "paid_api_keys_states"):
                if check_key in app.state.paid_api_keys_states and current_info.get("total_tokens", 0) > 0:
                    await update_paid_api_keys_states(app, check_key)
            return

        except Exception as e:
            error_str = str(e).lower()
            is_lock_error = "database is locked" in error_str or "busy" in error_str

            if is_lock_error and attempt < SQLITE_MAX_RETRIES - 1:
                delay = SQLITE_RETRY_DELAY * (2 ** attempt)
                logger.warning(
                    "Database locked, retrying in %ss (attempt %s/%s)",
                    delay,
                    attempt + 1,
                    SQLITE_MAX_RETRIES,
                )
                await asyncio.sleep(delay)
            else:
                logger.error("Error updating stats: %s", str(e))
                if is_debug:
                    import traceback

                    traceback.print_exc()
                break


async def update_channel_stats(request_id, provider, model, api_key, success, provider_api_key: str = None):
    """更新渠道统计到数据库"""
    if DISABLE_DATABASE:
        return

    if IS_D1_MODE:
        payload = {
            "request_id": request_id,
            "provider": provider,
            "model": model,
            "api_key": api_key,
            "provider_api_key": provider_api_key,
            "success": success,
        }
        payload = {k: v for k, v in payload.items() if k in _D1_CHANNEL_STAT_COLUMNS}

        for attempt in range(SQLITE_MAX_RETRIES):
            try:
                async with db_semaphore:
                    await _insert_d1("channel_stats", payload)
                return
            except Exception as e:
                if attempt < SQLITE_MAX_RETRIES - 1:
                    delay = SQLITE_RETRY_DELAY * (2 ** attempt)
                    await asyncio.sleep(delay)
                else:
                    logger.error("Error updating channel stats (d1): %s", str(e))
                    if is_debug:
                        import traceback

                        traceback.print_exc()
        return

    for attempt in range(SQLITE_MAX_RETRIES):
        try:
            async with db_semaphore:
                async with async_session() as session:
                    async with session.begin():
                        channel_stat = ChannelStat(
                            request_id=request_id,
                            provider=provider,
                            model=model,
                            api_key=api_key,
                            provider_api_key=provider_api_key,
                            success=success,
                        )
                        session.add(channel_stat)
                        await session.commit()
            return

        except Exception as e:
            error_str = str(e).lower()
            is_lock_error = "database is locked" in error_str or "busy" in error_str

            if is_lock_error and attempt < SQLITE_MAX_RETRIES - 1:
                delay = SQLITE_RETRY_DELAY * (2 ** attempt)
                logger.warning(
                    "Database locked (channel stats), retrying in %ss (attempt %s/%s)",
                    delay,
                    attempt + 1,
                    SQLITE_MAX_RETRIES,
                )
                await asyncio.sleep(delay)
            else:
                logger.error("Error updating channel stats: %s", str(e))
                if is_debug:
                    import traceback

                    traceback.print_exc()
                break


# ============== Token 使用量查询 ==============

async def query_token_usage(
    session: Optional[Any] = None,
    filter_api_key: Optional[str] = None,
    filter_model: Optional[str] = None,
    start_dt: Optional[datetime] = None,
    end_dt: Optional[datetime] = None,
) -> List[Dict]:
    """查询 RequestStat 表获取聚合 token 使用量。"""

    if IS_D1_MODE:
        db = await get_d1_database(required=False)
        if db is None:
            return []

        sql = (
            "SELECT api_key, model, "
            "COALESCE(SUM(prompt_tokens), 0) AS total_prompt_tokens, "
            "COALESCE(SUM(completion_tokens), 0) AS total_completion_tokens, "
            "COALESCE(SUM(total_tokens), 0) AS total_tokens, "
            "COUNT(id) AS request_count "
            "FROM request_stats WHERE 1=1"
        )
        params: list[Any] = []

        if filter_api_key:
            sql += " AND api_key = ?"
            params.append(filter_api_key)
        if filter_model:
            sql += " AND model = ?"
            params.append(filter_model)
        if start_dt:
            sql += " AND timestamp >= ?"
            params.append(d1_datetime(start_dt))
        if end_dt:
            sql += " AND timestamp < ?"
            params.append(d1_datetime(end_dt + timedelta(days=1)))
        if not filter_model:
            sql += " AND model IS NOT NULL AND model != ''"

        sql += " GROUP BY api_key, model"
        rows = await db.all(sql, *params)

        processed_usage = []
        for row in rows:
            usage_dict = {
                "model": row.get("model"),
                "total_prompt_tokens": int(row.get("total_prompt_tokens") or 0),
                "total_completion_tokens": int(row.get("total_completion_tokens") or 0),
                "total_tokens": int(row.get("total_tokens") or 0),
                "request_count": int(row.get("request_count") or 0),
            }
            api_key = row.get("api_key") or ""
            if api_key and len(api_key) > 7:
                usage_dict["api_key_prefix"] = f"{api_key[:7]}...{api_key[-4:]}"
            else:
                usage_dict["api_key_prefix"] = api_key
            processed_usage.append(usage_dict)

        return processed_usage

    query = select(
        RequestStat.api_key,
        RequestStat.model,
        func.sum(RequestStat.prompt_tokens).label("total_prompt_tokens"),
        func.sum(RequestStat.completion_tokens).label("total_completion_tokens"),
        func.sum(RequestStat.total_tokens).label("total_tokens"),
        func.count(RequestStat.id).label("request_count"),
    ).group_by(RequestStat.api_key, RequestStat.model)

    if filter_api_key:
        query = query.where(RequestStat.api_key == filter_api_key)
    if filter_model:
        query = query.where(RequestStat.model == filter_model)
    if start_dt:
        query = query.where(RequestStat.timestamp >= start_dt)
    if end_dt:
        query = query.where(RequestStat.timestamp < end_dt + timedelta(days=1))

    if not filter_model:
        query = query.where(RequestStat.model.isnot(None) & (RequestStat.model != ""))

    result = await session.execute(query)
    rows = result.mappings().all()

    processed_usage = []
    for row in rows:
        usage_dict = dict(row)
        api_key = usage_dict.get("api_key", "")
        if api_key and len(api_key) > 7:
            usage_dict["api_key_prefix"] = f"{api_key[:7]}...{api_key[-4:]}"
        else:
            usage_dict["api_key_prefix"] = api_key
        del usage_dict["api_key"]
        processed_usage.append(usage_dict)

    return processed_usage


async def get_usage_data(
    filter_api_key: Optional[str] = None,
    filter_model: Optional[str] = None,
    start_dt_obj: Optional[datetime] = None,
    end_dt_obj: Optional[datetime] = None,
) -> List[Dict]:
    """查询数据库并获取令牌使用数据。"""
    if IS_D1_MODE:
        return await query_token_usage(
            session=None,
            filter_api_key=filter_api_key,
            filter_model=filter_model,
            start_dt=start_dt_obj,
            end_dt=end_dt_obj,
        )

    async with async_session() as session:
        usage_data = await query_token_usage(
            session=session,
            filter_api_key=filter_api_key,
            filter_model=filter_model,
            start_dt=start_dt_obj,
            end_dt=end_dt_obj,
        )
    return usage_data


# ============== 付费 API 密钥状态 ==============

async def update_paid_api_keys_states(app, paid_key: str):
    """更新付费 API 密钥状态。"""
    from utils import safe_get

    check_index = app.state.api_list.index(paid_key)
    credits = safe_get(app.state.config, "api_keys", check_index, "preferences", "credits", default=-1)
    created_at = safe_get(
        app.state.config,
        "api_keys",
        check_index,
        "preferences",
        "created_at",
        default=datetime.now(timezone.utc) - timedelta(days=30),
    )
    created_at = created_at.astimezone(timezone.utc)

    if credits != -1:
        all_tokens_info = await get_usage_data(filter_api_key=paid_key, start_dt_obj=created_at)
        total_cost = await compute_total_cost_from_db(filter_api_key=paid_key, start_dt_obj=created_at)

        app.state.paid_api_keys_states[paid_key] = {
            "credits": credits,
            "created_at": created_at,
            "all_tokens_info": all_tokens_info,
            "total_cost": total_cost,
            "enabled": True if total_cost <= credits else False,
        }
        return credits, total_cost

    return credits, 0
