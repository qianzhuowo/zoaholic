"""
Stats 统计和使用量路由
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, Request, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_serializer

from db import RequestStat, ChannelStat, async_session, DISABLE_DATABASE, IS_D1_MODE
from core.d1 import get_d1_database, d1_datetime, parse_d1_datetime
from utils import safe_get, query_channel_key_stats
from routes.deps import rate_limit_dependency, verify_api_key, verify_admin_api_key, get_app

if not IS_D1_MODE:
    from sqlalchemy import select, case, func, desc, or_
    from sqlalchemy.ext.asyncio import AsyncSession
else:
    select = None
    case = None
    func = None
    desc = None
    or_ = None
    AsyncSession = Any


router = APIRouter()


# ============ Pydantic Models ============

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


class LogEntry(BaseModel):
    id: int
    timestamp: datetime
    endpoint: Optional[str] = None
    client_ip: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    api_key_prefix: Optional[str] = None
    process_time: Optional[float] = None
    first_response_time: Optional[float] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    success: bool = False
    status_code: Optional[int] = None
    is_flagged: bool = False

    # 扩展日志字段
    provider_id: Optional[str] = None
    provider_key_index: Optional[int] = None
    api_key_name: Optional[str] = None
    api_key_group: Optional[str] = None
    retry_count: Optional[int] = None
    retry_path: Optional[str] = None  # JSON 格式的重试路径
    request_headers: Optional[str] = None  # 用户请求头
    request_body: Optional[str] = None  # 用户请求体
    upstream_request_body: Optional[str] = None  # 发送到上游的请求体
    upstream_response_body: Optional[str] = None  # 上游返回的响应体
    response_body: Optional[str] = None  # 返回给用户的响应体
    raw_data_expires_at: Optional[datetime] = None  # 原始数据过期时间

    @field_serializer("timestamp")
    def serialize_dt(self, dt: datetime):
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()

    @field_serializer("raw_data_expires_at")
    def serialize_expires_at(self, dt: Optional[datetime]):
        if dt is None:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()


class LogsPage(BaseModel):
    items: List[LogEntry]
    total: int
    page: int
    page_size: int
    total_pages: int


# ============ Helper Functions ============

def _row_get(row: Any, key: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(key, default)
    return getattr(row, key, default)


def _to_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    try:
        return int(v) == 1
    except Exception:
        return bool(v)


def _mask_api_key(api_key: str) -> str:
    if api_key and len(api_key) > 7:
        return f"{api_key[:7]}...{api_key[-4:]}"
    return api_key


async def query_token_usage(
    session: Optional[Any],
    filter_api_key: Optional[str] = None,
    filter_model: Optional[str] = None,
    start_dt: Optional[datetime] = None,
    end_dt: Optional[datetime] = None,
) -> List[Dict]:
    """查询 RequestStat 表获取聚合的 token 使用量"""
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
    else:
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
        usage_dict = {
            "api_key": _row_get(row, "api_key", ""),
            "model": _row_get(row, "model"),
            "total_prompt_tokens": int(_row_get(row, "total_prompt_tokens", 0) or 0),
            "total_completion_tokens": int(_row_get(row, "total_completion_tokens", 0) or 0),
            "total_tokens": int(_row_get(row, "total_tokens", 0) or 0),
            "request_count": int(_row_get(row, "request_count", 0) or 0),
        }

        usage_dict["api_key_prefix"] = _mask_api_key(str(usage_dict.get("api_key") or ""))
        usage_dict.pop("api_key", None)
        processed_usage.append(usage_dict)

    return processed_usage


async def get_usage_data(
    filter_api_key: Optional[str] = None,
    filter_model: Optional[str] = None,
    start_dt_obj: Optional[datetime] = None,
    end_dt_obj: Optional[datetime] = None,
) -> List[Dict]:
    """查询数据库并获取令牌使用数据"""
    if IS_D1_MODE:
        return await query_token_usage(
            session=None,
            filter_api_key=filter_api_key,
            filter_model=filter_model,
            start_dt=start_dt_obj,
            end_dt=end_dt_obj,
        )

    async with async_session() as session:
        return await query_token_usage(
            session=session,
            filter_api_key=filter_api_key,
            filter_model=filter_model,
            start_dt=start_dt_obj,
            end_dt=end_dt_obj,
        )


def parse_datetime_input(dt_input: str) -> datetime:
    """解析 ISO 8601 字符串或 Unix 时间戳"""
    try:
        return datetime.fromtimestamp(float(dt_input), tz=timezone.utc)
    except ValueError:
        try:
            if dt_input.endswith("Z"):
                dt_input = dt_input[:-1] + "+00:00"
            dt_obj = datetime.fromisoformat(dt_input)
            if dt_obj.tzinfo is None:
                dt_obj = dt_obj.replace(tzinfo=timezone.utc)
            return dt_obj.astimezone(timezone.utc)
        except ValueError:
            raise ValueError(
                f"Invalid datetime format: {dt_input}. "
                "Use ISO 8601 (YYYY-MM-DDTHH:MM:SSZ) or Unix timestamp."
            )


# ============ Routes ============

@router.get("/v1/stats", dependencies=[Depends(rate_limit_dependency)])
async def get_stats(
    request: Request,
    token: str = Depends(verify_admin_api_key),
    hours: int = Query(default=24, ge=1, le=720, description="Number of hours to look back for stats (1-720)"),
):
    """获取统计数据"""
    if DISABLE_DATABASE:
        return JSONResponse(content={"stats": {}})

    start_time = datetime.now(timezone.utc) - timedelta(hours=hours)

    if IS_D1_MODE:
        db = await get_d1_database(required=False)
        if db is None:
            return JSONResponse(content={"stats": {}})

        start_time_str = d1_datetime(start_time)
        channel_model_stats = await db.all(
            "SELECT provider, model, COUNT(*) AS total, "
            "COALESCE(SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END), 0) AS success_count "
            "FROM channel_stats WHERE timestamp >= ? "
            "GROUP BY provider, model",
            start_time_str,
        )
        channel_stats = await db.all(
            "SELECT provider, COUNT(*) AS total, "
            "COALESCE(SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END), 0) AS success_count "
            "FROM channel_stats WHERE timestamp >= ? "
            "GROUP BY provider",
            start_time_str,
        )
        model_stats = await db.all(
            "SELECT model, COUNT(*) AS count FROM request_stats "
            "WHERE timestamp >= ? GROUP BY model ORDER BY count DESC",
            start_time_str,
        )
        endpoint_stats = await db.all(
            "SELECT endpoint, COUNT(*) AS count FROM request_stats "
            "WHERE timestamp >= ? GROUP BY endpoint ORDER BY count DESC",
            start_time_str,
        )
        ip_stats = await db.all(
            "SELECT client_ip, COUNT(*) AS count FROM request_stats "
            "WHERE timestamp >= ? GROUP BY client_ip ORDER BY count DESC",
            start_time_str,
        )
    else:
        async with async_session() as session:
            channel_model_stats = await session.execute(
                select(
                    ChannelStat.provider,
                    ChannelStat.model,
                    func.count().label("total"),
                    func.sum(case((ChannelStat.success, 1), else_=0)).label("success_count"),
                )
                .where(ChannelStat.timestamp >= start_time)
                .group_by(ChannelStat.provider, ChannelStat.model)
            )
            channel_model_stats = channel_model_stats.fetchall()

            channel_stats = await session.execute(
                select(
                    ChannelStat.provider,
                    func.count().label("total"),
                    func.sum(case((ChannelStat.success, 1), else_=0)).label("success_count"),
                )
                .where(ChannelStat.timestamp >= start_time)
                .group_by(ChannelStat.provider)
            )
            channel_stats = channel_stats.fetchall()

            model_stats = await session.execute(
                select(RequestStat.model, func.count().label("count"))
                .where(RequestStat.timestamp >= start_time)
                .group_by(RequestStat.model)
                .order_by(desc("count"))
            )
            model_stats = model_stats.fetchall()

            endpoint_stats = await session.execute(
                select(RequestStat.endpoint, func.count().label("count"))
                .where(RequestStat.timestamp >= start_time)
                .group_by(RequestStat.endpoint)
                .order_by(desc("count"))
            )
            endpoint_stats = endpoint_stats.fetchall()

            ip_stats = await session.execute(
                select(RequestStat.client_ip, func.count().label("count"))
                .where(RequestStat.timestamp >= start_time)
                .group_by(RequestStat.client_ip)
                .order_by(desc("count"))
            )
            ip_stats = ip_stats.fetchall()

    def _success_rate(item: Any) -> float:
        total = int(_row_get(item, "total", 0) or 0)
        success_count = int(_row_get(item, "success_count", 0) or 0)
        return success_count / total if total > 0 else 0

    stats = {
        "time_range": f"Last {hours} hours",
        "channel_model_success_rates": [
            {
                "provider": _row_get(stat, "provider"),
                "model": _row_get(stat, "model"),
                "success_rate": _success_rate(stat),
                "total_requests": int(_row_get(stat, "total", 0) or 0),
            }
            for stat in sorted(channel_model_stats, key=_success_rate, reverse=True)
        ],
        "channel_success_rates": [
            {
                "provider": _row_get(stat, "provider"),
                "success_rate": _success_rate(stat),
                "total_requests": int(_row_get(stat, "total", 0) or 0),
            }
            for stat in sorted(channel_stats, key=_success_rate, reverse=True)
        ],
        "model_request_counts": [
            {
                "model": _row_get(stat, "model"),
                "count": int(_row_get(stat, "count", 0) or 0),
            }
            for stat in model_stats
        ],
        "endpoint_request_counts": [
            {
                "endpoint": _row_get(stat, "endpoint"),
                "count": int(_row_get(stat, "count", 0) or 0),
            }
            for stat in endpoint_stats
        ],
        "ip_request_counts": [
            {
                "ip": _row_get(stat, "client_ip"),
                "count": int(_row_get(stat, "count", 0) or 0),
            }
            for stat in ip_stats
        ],
    }

    return JSONResponse(content=stats)


@router.get("/v1/token_usage", response_model=TokenUsageResponse, dependencies=[Depends(rate_limit_dependency)])
async def get_token_usage(
    request: Request,
    api_key_param: Optional[str] = None,
    model: Optional[str] = None,
    start_datetime: Optional[str] = None,
    end_datetime: Optional[str] = None,
    last_n_days: Optional[int] = None,
    api_index: int = Depends(verify_api_key),
):
    """获取聚合 token 使用统计"""
    if DISABLE_DATABASE:
        raise HTTPException(status_code=503, detail="Database is disabled.")

    app = get_app()
    requesting_token = safe_get(app.state.config, "api_keys", api_index, "api", default="")

    # 判断是否为管理员
    is_admin = False
    if hasattr(app.state, "admin_api_key") and requesting_token in app.state.admin_api_key:
        is_admin = True

    # 确定 API key 过滤器
    filter_api_key = None
    api_key_filter_detail = "all"
    if is_admin:
        if api_key_param:
            filter_api_key = api_key_param
            api_key_filter_detail = api_key_param
    else:
        filter_api_key = requesting_token
        api_key_filter_detail = "self"

    # 确定时间范围
    end_dt_obj = None
    start_dt_obj = None
    start_datetime_detail = None
    end_datetime_detail = None
    now = datetime.now(timezone.utc)

    if last_n_days is not None:
        if start_datetime or end_datetime:
            raise HTTPException(
                status_code=400,
                detail="Cannot use last_n_days with start_datetime or end_datetime.",
            )
        if last_n_days <= 0:
            raise HTTPException(status_code=400, detail="last_n_days must be positive.")
        start_dt_obj = now - timedelta(days=last_n_days)
        end_dt_obj = now
        start_datetime_detail = start_dt_obj.isoformat(timespec="seconds")
        end_datetime_detail = end_dt_obj.isoformat(timespec="seconds")
    elif start_datetime or end_datetime:
        try:
            if start_datetime:
                start_dt_obj = parse_datetime_input(start_datetime)
                start_datetime_detail = start_dt_obj.isoformat(timespec="seconds")
            if end_datetime:
                end_dt_obj = parse_datetime_input(end_datetime)
                end_datetime_detail = end_dt_obj.isoformat(timespec="seconds")
            if start_dt_obj and end_dt_obj and end_dt_obj < start_dt_obj:
                raise HTTPException(
                    status_code=400,
                    detail="end_datetime cannot be before start_datetime.",
                )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    else:
        start_dt_obj = now - timedelta(days=30)
        end_dt_obj = now
        start_datetime_detail = start_dt_obj.isoformat(timespec="seconds")
        end_datetime_detail = end_dt_obj.isoformat(timespec="seconds")

    usage_data = await get_usage_data(
        filter_api_key=filter_api_key,
        filter_model=model,
        start_dt_obj=start_dt_obj,
        end_dt_obj=end_dt_obj,
    )

    # 获取付费 API key 状态
    if filter_api_key:
        from main import update_paid_api_keys_states

        credits, total_cost = await update_paid_api_keys_states(app, filter_api_key)
    else:
        credits, total_cost = None, None

    query_details = QueryDetails(
        start_datetime=start_datetime_detail,
        end_datetime=end_datetime_detail,
        api_key_filter=api_key_filter_detail,
        model_filter=model if model else "all",
        credits="$" + str(credits) if credits is not None else None,
        total_cost="$" + str(total_cost) if total_cost is not None else None,
        balance="$" + str(float(credits) - float(total_cost)) if credits and total_cost else None,
    )

    response_data = TokenUsageResponse(
        usage=[TokenUsageEntry(**item) for item in usage_data],
        query_details=query_details,
    )

    return response_data


@router.get(
    "/v1/channel_key_rankings",
    response_model=ChannelKeyRankingsResponse,
    dependencies=[Depends(rate_limit_dependency)],
)
async def get_channel_key_rankings(
    request: Request,
    provider_name: str,
    start_datetime: Optional[str] = None,
    end_datetime: Optional[str] = None,
    last_n_days: Optional[int] = None,
    token: str = Depends(verify_admin_api_key),
):
    """获取特定渠道的 API key 成功率排名，可按时间范围过滤。"""
    if DISABLE_DATABASE:
        raise HTTPException(status_code=503, detail="Database is disabled.")

    end_dt_obj = None
    start_dt_obj = None
    start_datetime_detail = None
    end_datetime_detail = None
    now = datetime.now(timezone.utc)

    if last_n_days is not None:
        if start_datetime or end_datetime:
            raise HTTPException(
                status_code=400,
                detail="Cannot use last_n_days with start_datetime or end_datetime.",
            )
        if last_n_days <= 0:
            raise HTTPException(status_code=400, detail="last_n_days must be positive.")
        start_dt_obj = now - timedelta(days=last_n_days)
        end_dt_obj = now
        start_datetime_detail = start_dt_obj.isoformat(timespec="seconds")
        end_datetime_detail = end_dt_obj.isoformat(timespec="seconds")
    elif start_datetime or end_datetime:
        try:
            if start_datetime:
                start_dt_obj = parse_datetime_input(start_datetime)
                start_datetime_detail = start_dt_obj.isoformat(timespec="seconds")
            if end_datetime:
                end_dt_obj = parse_datetime_input(end_datetime)
                end_datetime_detail = end_dt_obj.isoformat(timespec="seconds")
            if start_dt_obj and end_dt_obj and end_dt_obj < start_dt_obj:
                raise HTTPException(
                    status_code=400,
                    detail="end_datetime cannot be before start_datetime.",
                )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    else:
        start_dt_obj = now - timedelta(days=1)
        end_dt_obj = now
        start_datetime_detail = start_dt_obj.isoformat(timespec="seconds")
        end_datetime_detail = end_dt_obj.isoformat(timespec="seconds")

    rankings_data = await query_channel_key_stats(
        provider_name=provider_name,
        start_dt=start_dt_obj,
        end_dt=end_dt_obj,
    )

    query_details = QueryDetails(
        start_datetime=start_datetime_detail,
        end_datetime=end_datetime_detail,
        api_key_filter=provider_name,
    )

    response_data = ChannelKeyRankingsResponse(
        rankings=[ChannelKeyRanking(**item) for item in rankings_data],
        query_details=query_details,
    )

    return response_data


@router.get("/v1/api_keys_states", dependencies=[Depends(rate_limit_dependency)])
async def api_keys_states(token: str = Depends(verify_admin_api_key)):
    """获取所有付费 API key 的状态"""
    app = get_app()

    states_dict = {}
    for key, state in app.state.paid_api_keys_states.items():
        states_dict[key] = ApiKeyState(
            credits=state["credits"],
            created_at=state["created_at"],
            all_tokens_info=state["all_tokens_info"],
            total_cost=state["total_cost"],
            enabled=state["enabled"],
        )

    response = ApiKeysStatesResponse(api_keys_states=states_dict)
    return response


@router.post("/v1/add_credits", dependencies=[Depends(rate_limit_dependency)])
async def add_credits_to_api_key(
    request: Request,
    paid_key: str = Query(..., description="The API key to add credits to"),
    amount: float = Query(..., description="The amount of credits to add. Must be positive.", gt=0),
    token: str = Depends(verify_admin_api_key),
):
    """为指定的 API key 添加额度"""
    from core.log_config import logger

    app = get_app()

    if paid_key not in app.state.paid_api_keys_states:
        raise HTTPException(
            status_code=404,
            detail=f"API key '{paid_key}' not found in paid API keys states.",
        )

    app.state.paid_api_keys_states[paid_key]["credits"] += float(amount)

    current_credits = app.state.paid_api_keys_states[paid_key]["credits"]
    total_cost = app.state.paid_api_keys_states[paid_key]["total_cost"]
    app.state.paid_api_keys_states[paid_key]["enabled"] = current_credits >= total_cost

    logger.info(
        f"Credits for API key '{paid_key}' updated. "
        f"Amount added: {amount}, New credits: {current_credits}, "
        f"Enabled: {app.state.paid_api_keys_states[paid_key]['enabled']}"
    )

    return JSONResponse(
        content={
            "message": f"Successfully added {amount} credits to API key '{paid_key}'.",
            "paid_key": paid_key,
            "new_credits": current_credits,
            "enabled": app.state.paid_api_keys_states[paid_key]["enabled"],
        }
    )


@router.get("/v1/logs", response_model=LogsPage, dependencies=[Depends(rate_limit_dependency)])
async def get_logs(
    request: Request,
    page: int = Query(1, ge=1, description="Page number (starting from 1)"),
    page_size: int = Query(20, ge=1, le=200, description="Number of items per page"),
    start_time: Optional[str] = Query(None, description="Start time filter (ISO 8601 or Unix timestamp)"),
    end_time: Optional[str] = Query(None, description="End time filter (ISO 8601 or Unix timestamp)"),
    provider: Optional[str] = Query(None, description="Provider/channel filter (fuzzy match)"),
    api_key: Optional[str] = Query(None, description="API key/token filter (fuzzy match)"),
    model: Optional[str] = Query(None, description="Model name filter (fuzzy match)"),
    success: Optional[bool] = Query(None, description="Filter by success status"),
    token: str = Depends(verify_admin_api_key),
):
    """获取请求日志（RequestStat）分页列表，仅管理员可访问。"""
    if DISABLE_DATABASE:
        raise HTTPException(status_code=503, detail="Database is disabled.")

    if IS_D1_MODE:
        db = await get_d1_database(required=False)
        if db is None:
            raise HTTPException(status_code=503, detail="D1 binding is not available.")

        where_parts = ["1=1"]
        params: list[Any] = []

        if start_time:
            try:
                start_dt = parse_datetime_input(start_time)
                where_parts.append("timestamp >= ?")
                params.append(d1_datetime(start_dt))
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid start_time: {e}")

        if end_time:
            try:
                end_dt = parse_datetime_input(end_time)
                where_parts.append("timestamp <= ?")
                params.append(d1_datetime(end_dt))
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid end_time: {e}")

        if provider:
            fuzzy = f"%{provider}%"
            where_parts.append("(provider_id LIKE ? OR provider LIKE ?)")
            params.extend([fuzzy, fuzzy])

        if api_key:
            fuzzy = f"%{api_key}%"
            where_parts.append("(api_key_name LIKE ? OR api_key_group LIKE ? OR api_key LIKE ?)")
            params.extend([fuzzy, fuzzy, fuzzy])

        if model:
            where_parts.append("model LIKE ?")
            params.append(f"%{model}%")

        if success is not None:
            where_parts.append("success = ?")
            params.append(1 if success else 0)

        where_sql = " AND ".join(where_parts)

        count_row = await db.first(
            f"SELECT COUNT(id) AS total FROM request_stats WHERE {where_sql}",
            *params,
        )
        total = int((count_row or {}).get("total") or 0)

        if total == 0:
            return LogsPage(
                items=[],
                total=0,
                page=page,
                page_size=page_size,
                total_pages=0,
            )

        total_pages = (total + page_size - 1) // page_size
        if page > total_pages:
            page = total_pages

        offset = (page - 1) * page_size
        rows = await db.all(
            f"SELECT * FROM request_stats WHERE {where_sql} ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            *params,
            page_size,
            offset,
        )
    else:
        async with async_session() as session:
            conditions = []

            if start_time:
                try:
                    start_dt = parse_datetime_input(start_time)
                    conditions.append(RequestStat.timestamp >= start_dt)
                except ValueError as e:
                    raise HTTPException(status_code=400, detail=f"Invalid start_time: {e}")

            if end_time:
                try:
                    end_dt = parse_datetime_input(end_time)
                    conditions.append(RequestStat.timestamp <= end_dt)
                except ValueError as e:
                    raise HTTPException(status_code=400, detail=f"Invalid end_time: {e}")

            if provider:
                conditions.append(
                    or_(
                        RequestStat.provider_id.ilike(f"%{provider}%"),
                        RequestStat.provider.ilike(f"%{provider}%"),
                    )
                )

            if api_key:
                conditions.append(
                    or_(
                        RequestStat.api_key_name.ilike(f"%{api_key}%"),
                        RequestStat.api_key_group.ilike(f"%{api_key}%"),
                        RequestStat.api_key.ilike(f"%{api_key}%"),
                    )
                )

            if model:
                conditions.append(RequestStat.model.ilike(f"%{model}%"))

            if success is not None:
                conditions.append(RequestStat.success == success)

            count_query = select(func.count(RequestStat.id)).where(*conditions)
            result = await session.execute(count_query)
            total = result.scalar() or 0

            if total == 0:
                return LogsPage(
                    items=[],
                    total=0,
                    page=page,
                    page_size=page_size,
                    total_pages=0,
                )

            total_pages = (total + page_size - 1) // page_size
            if page > total_pages:
                page = total_pages

            offset = (page - 1) * page_size
            query = (
                select(RequestStat)
                .where(*conditions)
                .order_by(RequestStat.timestamp.desc())
                .offset(offset)
                .limit(page_size)
            )
            rows_result = await session.execute(query)
            rows = rows_result.scalars().all()

    items: List[LogEntry] = []
    now = datetime.now(timezone.utc)

    for row in rows:
        api_key_raw = str(_row_get(row, "api_key", "") or "")
        api_key_prefix = _mask_api_key(api_key_raw) if len(api_key_raw) > 11 else api_key_raw

        raw_data_expires_at_value = _row_get(row, "raw_data_expires_at")
        raw_data_expires_at = parse_d1_datetime(raw_data_expires_at_value) if IS_D1_MODE else raw_data_expires_at_value

        raw_data_expired = False
        if raw_data_expires_at:
            expires_at = raw_data_expires_at
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
            raw_data_expired = expires_at < now

        timestamp_value = parse_d1_datetime(_row_get(row, "timestamp")) if IS_D1_MODE else _row_get(row, "timestamp")
        if timestamp_value is None:
            timestamp_value = now

        items.append(
            LogEntry(
                id=int(_row_get(row, "id", 0) or 0),
                timestamp=timestamp_value,
                endpoint=_row_get(row, "endpoint"),
                client_ip=_row_get(row, "client_ip"),
                provider=_row_get(row, "provider"),
                model=_row_get(row, "model"),
                api_key_prefix=api_key_prefix,
                process_time=_row_get(row, "process_time"),
                first_response_time=_row_get(row, "first_response_time"),
                prompt_tokens=_row_get(row, "prompt_tokens"),
                completion_tokens=_row_get(row, "completion_tokens"),
                total_tokens=_row_get(row, "total_tokens"),
                success=_to_bool(_row_get(row, "success", False)),
                status_code=_row_get(row, "status_code"),
                is_flagged=_to_bool(_row_get(row, "is_flagged", False)),
                provider_id=_row_get(row, "provider_id"),
                provider_key_index=_row_get(row, "provider_key_index"),
                api_key_name=_row_get(row, "api_key_name"),
                api_key_group=_row_get(row, "api_key_group"),
                retry_count=_row_get(row, "retry_count"),
                retry_path=_row_get(row, "retry_path") if not raw_data_expired else None,
                request_headers=_row_get(row, "request_headers") if not raw_data_expired else None,
                request_body=_row_get(row, "request_body") if not raw_data_expired else None,
                upstream_request_body=_row_get(row, "upstream_request_body") if not raw_data_expired else None,
                upstream_response_body=_row_get(row, "upstream_response_body") if not raw_data_expired else None,
                response_body=_row_get(row, "response_body") if not raw_data_expired else None,
                raw_data_expires_at=raw_data_expires_at,
            )
        )

    total_pages = (total + page_size - 1) // page_size if total > 0 else 0
    return LogsPage(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )
