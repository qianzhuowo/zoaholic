"""统一反代错误插件（proxy_error_shield）

目标：
- 当某个 provider 显式启用本插件后，将上游返回的错误统一改写为“反向代理错误”。
- 避免把上游渠道名称、原始错误结构、供应商特征字段直接暴露给客户端。
- 对非流式 / 流式首包错误 / 主流程抛出的 HTTPException 同时生效。
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple

from core.error_response import build_proxy_error_dict, normalize_proxy_error_policy
from core.log_config import logger
from core.middleware import request_info
from core.plugins import register_request_interceptor, register_response_interceptor
from core.plugins import unregister_request_interceptor, unregister_response_interceptor
from core.utils import safe_get


PLUGIN_INFO = {
    "name": "proxy_error_shield",
    "version": "1.0.0",
    "description": "统一反代错误返回，避免泄露上游渠道信息",
    "author": "Zoaholic Team",
    "dependencies": [],
    "metadata": {
        "category": "interceptors",
        "tags": ["proxy", "error", "security"],
        "params_hint": "配置在 provider.preferences.proxy_error_shield；插件需在 enabled_plugins 中启用。",
        "provider_config": {
            "key": "proxy_error_shield",
            "type": "json",
            "title": "统一反代错误返回",
            "description": "将该渠道的上游错误统一改写为固定的反代错误响应，避免暴露供应商/渠道细节。",
            "example": {
                "message": "Reverse proxy request failed.",
                "status_code": 502,
                "code": "reverse_proxy_error",
                "error_type": "api_error"
            },
        },
    },
}

EXTENSIONS = [
    "interceptors:proxy_error_shield_request",
    "interceptors:proxy_error_shield_response",
]


def _get_policy(provider: Dict[str, Any]) -> Optional[dict[str, Any]]:
    raw_policy = safe_get(provider, "preferences", "proxy_error_shield", default={})
    return normalize_proxy_error_policy(raw_policy)


def _looks_like_error_payload(payload: Any) -> bool:
    return isinstance(payload, dict) and (
        "error" in payload
        or safe_get(payload, "choices", 0, "error", default=None) is not None
        or safe_get(payload, "base_resp", "status_code", default=200) != 200
    )


def _mask_sse_chunk(chunk: str, policy: dict[str, Any]) -> str:
    if not isinstance(chunk, str) or not chunk.strip():
        return chunk

    lines = chunk.splitlines(keepends=True)
    changed = False
    masked_payload = json.dumps(build_proxy_error_dict(policy), ensure_ascii=False)
    new_lines = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("data:"):
            data_str = stripped[len("data:"):].strip()
            if not data_str or data_str == "[DONE]":
                new_lines.append(line)
                continue
            try:
                payload = json.loads(data_str)
            except Exception:
                new_lines.append(line)
                continue
            if _looks_like_error_payload(payload):
                newline = "\n" if line.endswith("\n") else ""
                new_lines.append(f"data: {masked_payload}{newline}")
                changed = True
                continue
        new_lines.append(line)

    if changed:
        return "".join(new_lines)

    stripped_chunk = chunk.strip()
    try:
        payload = json.loads(stripped_chunk)
    except Exception:
        return chunk

    if _looks_like_error_payload(payload):
        return masked_payload
    return chunk


async def proxy_error_shield_request_interceptor(
    request: Any,
    engine: str,
    provider: Dict[str, Any],
    api_key: Optional[str],
    url: str,
    headers: Dict[str, Any],
    payload: Dict[str, Any],
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    policy = _get_policy(provider)
    if not policy:
        return url, headers, payload

    current_info = request_info.get()
    current_info["proxy_error_policy"] = policy
    logger.debug("[proxy_error_shield] enabled for provider=%s engine=%s", provider.get("provider"), engine)
    return url, headers, payload


async def proxy_error_shield_response_interceptor(
    response_chunk: Any,
    engine: str,
    model: str,
    is_stream: bool,
):
    current_info = request_info.get()
    policy = normalize_proxy_error_policy(current_info.get("proxy_error_policy")) if isinstance(current_info, dict) else None
    if not policy:
        return response_chunk

    if _looks_like_error_payload(response_chunk):
        return build_proxy_error_dict(policy)

    if isinstance(response_chunk, str):
        return _mask_sse_chunk(response_chunk, policy)

    return response_chunk


def setup(manager):
    logger.info(f"[{PLUGIN_INFO['name']}] 正在初始化...")

    register_request_interceptor(
        interceptor_id="proxy_error_shield_request",
        callback=proxy_error_shield_request_interceptor,
        priority=30,
        plugin_name=PLUGIN_INFO["name"],
        metadata={"description": "为当前请求注入统一反代错误策略"},
    )
    register_response_interceptor(
        interceptor_id="proxy_error_shield_response",
        callback=proxy_error_shield_response_interceptor,
        priority=30,
        plugin_name=PLUGIN_INFO["name"],
        metadata={"description": "将响应中的上游错误改写为统一反代错误"},
    )

    logger.info(f"[{PLUGIN_INFO['name']}] 已注册请求/响应拦截器")


def teardown(manager):
    logger.info(f"[{PLUGIN_INFO['name']}] 正在清理...")
    unregister_request_interceptor("proxy_error_shield_request")
    unregister_response_interceptor("proxy_error_shield_response")
    logger.info(f"[{PLUGIN_INFO['name']}] 已清理完成")
