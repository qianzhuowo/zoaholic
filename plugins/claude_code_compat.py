"""
Claude Code 兼容插件

功能：
1. 将 Anthropic 风格的 x-api-key 认证改写为 Authorization: Bearer。
2. 为请求补充 Claude Code 风格的 User-Agent。
3. 为 Claude /messages 请求在 system 最前面注入
   x-anthropic-billing-header 文本块。

用途：
适用于需要模拟 Claude Code 请求特征的 Anthropic 兼容上游。

使用方式：
preferences:
  enabled_plugins:
    - claude_code_compat

简单参数格式：
- claude_code_compat
- claude_code_compat:2.1.76
- claude_code_compat:2.1.76,cli
- claude_code_compat:2.1.76,,59cf53e54c78
- claude_code_compat:2.1.76,cli,59cf53e54c78

说明：
- 不带参数时，默认版本为 2.1.76，默认 entrypoint 为 cli。
- 第一个参数为 Claude Code 版本。
- 第二个参数为 entrypoint。
- 第三个参数为 billing salt。
- User-Agent 固定根据版本自动生成。
- 不传时 billing salt 使用内置固定值。
"""

from __future__ import annotations

import hashlib
import os
from typing import Any, Dict, Optional, Tuple

from core.log_config import logger
from core.plugins import (
    get_plugin_options,
    register_request_interceptor,
    unregister_request_interceptor,
)


PLUGIN_INFO = {
    "name": "claude_code_compat",
    "version": "1.0.1",
    "description": "Claude Code 兼容插件，支持 Bearer 认证、User-Agent 和 billing header 注入",
    "author": "Zoaholic",
    "dependencies": [],
    "metadata": {
        "category": "interceptor",
        "tags": ["claude", "anthropic", "claude-code", "auth", "bearer", "billing"],
        "params_hint": "格式：2.1.76 或 2.1.76,cli 或 2.1.76,cli,59cf53e54c78。留空使用默认值。",
    },
}

EXTENSIONS = [
    "interceptors:claude_code_compat_request",
]

DEFAULT_CLAUDE_CODE_VERSION = "2.1.76"
DEFAULT_BILLING_SALT = "59cf53e54c78"
DEFAULT_ENTRYPOINT_ENV = "CLAUDE_CODE_ENTRYPOINT"
DEFAULT_ENTRYPOINT = "cli"
BILLING_HEADER_PREFIX = "x-anthropic-billing-header:"
BILLING_SAMPLE_INDEXES = (4, 7, 20)


def get_header_case_insensitive(headers: Dict[str, Any], name: str) -> Optional[Any]:
    """按大小写不敏感方式读取请求头。"""
    target = name.lower()
    for key, value in headers.items():
        if str(key).lower() == target:
            return value
    return None


def pop_header_case_insensitive(headers: Dict[str, Any], name: str) -> Optional[Any]:
    """按大小写不敏感方式移除请求头并返回其值。"""
    target = name.lower()
    for key in list(headers.keys()):
        if str(key).lower() == target:
            return headers.pop(key)
    return None


def set_header_case_insensitive(headers: Dict[str, Any], name: str, value: Any) -> None:
    """按大小写不敏感方式设置请求头。"""
    pop_header_case_insensitive(headers, name)
    headers[name] = value


def resolve_plugin_config(provider: Dict[str, Any]) -> Dict[str, Any]:
    """只从 enabled_plugins 的简单参数中解析插件配置。"""
    version = DEFAULT_CLAUDE_CODE_VERSION
    entrypoint = os.getenv(DEFAULT_ENTRYPOINT_ENV) or DEFAULT_ENTRYPOINT
    billing_salt = DEFAULT_BILLING_SALT

    plugin_options = get_plugin_options(PLUGIN_INFO["name"], provider)
    if plugin_options:
        parts = [part.strip() for part in str(plugin_options).split(",")]
        if parts and parts[0]:
            version = parts[0]
        if len(parts) > 1 and parts[1]:
            entrypoint = parts[1]
        if len(parts) > 2 and parts[2]:
            billing_salt = parts[2]

    return {
        "version": version,
        "entrypoint": entrypoint,
        "user_agent": f"claude-code/{version}",
        "billing_salt": billing_salt,
    }


def is_claude_request(engine: str, url: str, headers: Dict[str, Any]) -> bool:
    """判断是否为 Claude/Anthropic 请求。"""
    engine_lower = str(engine or "").lower()
    if engine_lower in {"claude", "anthropic"}:
        return True

    if get_header_case_insensitive(headers, "anthropic-version") is not None:
        return True

    if get_header_case_insensitive(headers, "x-api-key") is None:
        return False

    url_lower = str(url or "").lower()
    return "/messages" in url_lower or "/models" in url_lower


def is_message_request(url: str, payload: Dict[str, Any]) -> bool:
    """判断当前是否为 Claude messages 请求。"""
    if not isinstance(payload, dict) or not isinstance(payload.get("messages"), list):
        return False

    url_lower = str(url or "").lower()
    return url_lower.endswith("/messages") or "/messages?" in url_lower or "/messages/" in url_lower


def first_user_message_text(messages: Any) -> str:
    """提取第一条 user 消息中的首个文本内容。"""
    if not isinstance(messages, list):
        return ""

    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue

        content = message.get("content")
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    return item["text"]
        return ""

    return ""


def sample_js_code_unit(text: str, idx: int) -> str:
    """按 JavaScript UTF-16 code unit 语义采样单个字符。"""
    if not isinstance(text, str) or idx < 0:
        return "0"

    utf16_le = text.encode("utf-16-le")
    start = idx * 2
    end = start + 2
    if end > len(utf16_le):
        return "0"

    return utf16_le[start:end].decode("utf-16-le", errors="replace")


def build_billing_header_text(messages: Any, version: str, entrypoint: str, billing_salt: str) -> str:
    """构造 x-anthropic-billing-header 文本。"""
    sampled = "".join(sample_js_code_unit(first_user_message_text(messages), idx) for idx in BILLING_SAMPLE_INDEXES)
    digest = hashlib.sha256(f"{billing_salt}{sampled}{version}".encode("utf-8")).hexdigest()[:3]
    return (
        f"{BILLING_HEADER_PREFIX} cc_version={version}.{digest}; "
        f"cc_entrypoint={entrypoint}; cch=00000;"
    )


def has_billing_header(system: Any) -> bool:
    """检查 system 中是否已经存在 billing header。"""
    if isinstance(system, str):
        return system.strip().startswith(BILLING_HEADER_PREFIX)

    if isinstance(system, list) and system:
        first_item = system[0]
        if isinstance(first_item, dict):
            text = first_item.get("text")
            return isinstance(text, str) and text.strip().startswith(BILLING_HEADER_PREFIX)
        if isinstance(first_item, str):
            return first_item.strip().startswith(BILLING_HEADER_PREFIX)

    return False


def prepend_system_text_block(payload: Dict[str, Any], text: str) -> None:
    """将文本块插入到 system 最前面。"""
    block = {"type": "text", "text": text}
    current = payload.get("system")

    if current is None:
        payload["system"] = [block]
        return

    if isinstance(current, str):
        systems = [block]
        if current.strip():
            systems.append({"type": "text", "text": current})
        payload["system"] = systems
        return

    if isinstance(current, list):
        payload["system"] = [block, *current]
        return

    payload["system"] = [block, current]


async def claude_code_compat_request_interceptor(
    request: Any,
    engine: str,
    provider: Dict[str, Any],
    api_key: Optional[str],
    url: str,
    headers: Dict[str, Any],
    payload: Dict[str, Any],
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """Claude Code 兼容请求拦截器。"""
    if not is_claude_request(engine, url, headers):
        return url, headers, payload

    config = resolve_plugin_config(provider)
    modified = False

    api_key_value = pop_header_case_insensitive(headers, "x-api-key")
    if api_key_value:
        if get_header_case_insensitive(headers, "Authorization") is None:
            set_header_case_insensitive(headers, "Authorization", f"Bearer {api_key_value}")
        modified = True

    set_header_case_insensitive(headers, "User-Agent", config["user_agent"])
    modified = True

    if is_message_request(url, payload):
        system = payload.get("system")
        if not has_billing_header(system):
            billing_text = build_billing_header_text(
                payload.get("messages", []),
                version=config["version"],
                entrypoint=config["entrypoint"],
                billing_salt=config["billing_salt"],
            )
            prepend_system_text_block(payload, billing_text)
            modified = True

    if modified:
        logger.debug(
            "[claude_code_compat] applied. url=%s, version=%s, user_agent=%s",
            url,
            config["version"],
            config["user_agent"],
        )

    return url, headers, payload


def setup(manager):
    """插件初始化。"""
    register_request_interceptor(
        interceptor_id="claude_code_compat_request",
        callback=claude_code_compat_request_interceptor,
        priority=10,
        plugin_name=PLUGIN_INFO["name"],
        overwrite=True,
        metadata={"description": "Claude Code 兼容请求处理"},
    )


def teardown(manager):
    """插件卸载。"""
    unregister_request_interceptor("claude_code_compat_request")
