"""
屏蔽词过滤插件（content_word_filter）

定位：
- 作为“入站拦截器插件”运行，在请求进入上游前检查消息内容。
- 仅当某个渠道在 provider.preferences.enabled_plugins 显式启用本插件时生效。

配置方式（声明式 UI，推荐）：
- 在渠道流水线 / 插件配置面板里填写参数，前端会通过 enabled_plugins 的 key=value 写入。
- 支持的参数见 PLUGIN_INFO["metadata"]["params_schema"]。

处理模式（mode）：
- replace：将匹配到的屏蔽词替换为 replacement 指定的文本
- reject：直接拒绝请求，返回 400 并附带提示
- strip：从消息内容中移除所有匹配到的屏蔽词；若消息因此变为空，则整条消息被移除

匹配范围（scope）：
- all：所有 role 的消息都检查（默认）
- system / user / assistant：仅检查对应 role

匹配类型（match_type）：
- word：把屏蔽词列表视为独立词集合，逐词边界匹配
- substring：把屏蔽词列表视为子串，直接 in 检查
- pattern：把屏蔽词列表视为正则表达式列表
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from fastapi import HTTPException

from core.log_config import logger
from core.plugins.interceptors import (
    register_inbound_interceptor,
    unregister_inbound_interceptor,
)


PARAMS_SCHEMA = [
    {
        "key": "mode",
        "label": "处理模式",
        "type": "select",
        "default": "replace",
        "options": [
            {"value": "replace", "label": "replace（替换）"},
            {"value": "reject", "label": "reject（拒绝）"},
            {"value": "strip", "label": "strip（移除）"},
        ],
        "serialize": "key_value",
    },
    {
        "key": "words",
        "label": "屏蔽词列表（每行一个）",
        "type": "textarea",
        "default": "",
        "placeholder": "badword1\nbadword2\nbadword3",
        "serialize": "key_value",
    },
    {
        "key": "replacement",
        "label": "替换文本（replace 模式生效）",
        "type": "text",
        "default": "***",
        "serialize": "key_value",
    },
    {
        "key": "scope",
        "label": "检查范围",
        "type": "select",
        "default": "all",
        "options": [
            {"value": "all", "label": "all（全部角色）"},
            {"value": "system", "label": "system"},
            {"value": "user", "label": "user"},
            {"value": "assistant", "label": "assistant"},
        ],
        "serialize": "key_value",
    },
    {
        "key": "match_type",
        "label": "匹配类型",
        "type": "select",
        "default": "substring",
        "options": [
            {"value": "word", "label": "word（独立词）"},
            {"value": "substring", "label": "substring（子串）"},
            {"value": "pattern", "label": "pattern（正则）"},
        ],
        "serialize": "key_value",
    },
    {
        "key": "case_sensitive",
        "label": "区分大小写",
        "type": "toggle",
        "default": False,
        "serialize": "key_value",
    },
    {
        "key": "strip_empty_message",
        "label": "strip 模式：移除变为空的消息",
        "type": "toggle",
        "default": True,
        "serialize": "key_value",
    },
]

PLUGIN_INFO = {
    "name": "content_word_filter",
    "version": "1.0.0",
    "description": "屏蔽词过滤插件：基于入站请求消息内容，对消息中的屏蔽词执行替换、拒绝或移除。",
    "author": "Zoaholic",
    "dependencies": [],
    "metadata": {
        "category": "interceptors",
        "params_hint": (
            "mode=replace|reject|strip; words=每行一个屏蔽词; "
            "scope=all|system|user|assistant; match_type=word|substring|pattern"
        ),
        "params_schema": PARAMS_SCHEMA,
    },
}


def _normalize_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off"):
        return False
    return default


def _split_lines(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        items: List[str] = []
        for item in value:
            items.extend(_split_lines(item))
        return items
    text = str(value)
    return [line.strip() for line in text.splitlines() if line.strip()]


def _parse_plugin_options(options: Optional[Any]) -> Dict[str, Any]:
    if not options:
        return {}

    if isinstance(options, dict):
        raw = dict(options)
    else:
        raw = {}
        text = str(options)
        for line in text.splitlines():
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            raw[key.strip()] = value.strip()

    result: Dict[str, Any] = {}
    if "mode" in raw:
        result["mode"] = str(raw["mode"]).strip().lower()
    if "match_type" in raw:
        result["match_type"] = str(raw["match_type"]).strip().lower()
    if "scope" in raw:
        result["scope"] = str(raw["scope"]).strip().lower()
    if "replacement" in raw:
        result["replacement"] = raw["replacement"]
    if "words" in raw:
        result["words"] = _split_lines(raw["words"])
    result["case_sensitive"] = _normalize_bool(raw.get("case_sensitive"), default=False)
    result["strip_empty_message"] = _normalize_bool(
        raw.get("strip_empty_message"), default=True
    )
    return result


def _extract_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if hasattr(item, "type") and item.type == "text" and getattr(item, "text", None):
                parts.append(str(item.text))
            elif isinstance(item, dict) and item.get("type") == "text" and item.get("text"):
                parts.append(str(item["text"]))
        return "".join(parts)
    return str(content)


def _set_text(content: Any, new_text: str) -> Any:
    if content is None:
        return new_text
    if isinstance(content, str):
        return new_text
    if isinstance(content, list):
        updated = []
        for item in content:
            if hasattr(item, "type") and item.type == "text":
                updated.append(
                    item.__class__(**{**item.model_dump(), "text": new_text})
                    if hasattr(item, "model_dump")
                    else item.__class__(type="text", text=new_text)
                )
            else:
                updated.append(item)
        return updated
    return new_text


def _compile_matcher(keywords: List[str], case_sensitive: bool, match_type: str):
    if match_type == "substring":
        flags = 0 if case_sensitive else re.IGNORECASE
        return [re.compile(re.escape(word), flags) for word in keywords]

    if match_type == "pattern":
        flags = 0 if case_sensitive else re.IGNORECASE
        compiled: List[re.Pattern[str]] = []
        for word in keywords:
            try:
                compiled.append(re.compile(word, flags))
            except re.error as exc:  # pragma: no cover - defensive
                logger.warning(f"[content_word_filter] 正则表达式编译失败，已跳过: {word!r} ({exc})")
        return compiled

    # word
    if not case_sensitive:
        lowered = [word.lower() for word in keywords]
        return [re.compile(rf"(?<!\w){re.escape(word)}(?!\w)", re.IGNORECASE) for word in keywords]
    return [re.compile(rf"(?<!\w){re.escape(word)}(?!\w)") for word in keywords]


def _match_text(text: str, matchers: List[re.Pattern[str]]) -> bool:
    return any(pattern.search(text) for pattern in matchers)


def _replace_text(text: str, matchers: List[re.Pattern[str]], replacement: str) -> str:
    for pattern in matchers:
        text = pattern.sub(replacement, text)
    return text


def _check_and_filter_text(
    text: str,
    matchers: List[re.Pattern[str]],
    mode: str,
    replacement: str,
) -> Tuple[str, bool]:
    if not matchers:
        return text, False

    if not _match_text(text, matchers):
        return text, False

    if mode == "reject":
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": "请求包含屏蔽词，已被 content_word_filter 拒绝。",
                    "type": "content_filter",
                    "code": "content_word_filter",
                }
            },
        )

    if mode == "replace":
        return _replace_text(text, matchers, replacement), True

    if mode == "strip":
        for pattern in matchers:
            text = pattern.sub("", text)
        return text, True

    return text, False


async def content_word_filter_interceptor(
    request_data: Any,
    request: Any,
    api_key_info: Optional[Dict[str, Any]],
    enabled_plugins: Optional[List[Any]],
) -> Any:
    if request_data is None:
        return request_data

    plugin_options = parse_enabled_plugins(enabled_plugins or []).get("content_word_filter")
    if not plugin_options:
        return request_data

    cfg = _parse_plugin_options(plugin_options)
    words = cfg.get("words") or []
    if not words:
        return request_data

    mode = cfg.get("mode", "replace")
    scope = cfg.get("scope", "all")
    match_type = cfg.get("match_type", "substring")
    case_sensitive = bool(cfg.get("case_sensitive", False))
    replacement = cfg.get("replacement", "***")
    strip_empty_message = bool(cfg.get("strip_empty_message", True))

    if mode not in {"replace", "reject", "strip"}:
        logger.warning(f"[content_word_filter] 未知 mode={mode!r}，已回退为 replace。")
        mode = "replace"
    if match_type not in {"word", "substring", "pattern"}:
        logger.warning(f"[content_word_filter] 未知 match_type={match_type!r}，已回退为 substring。")
        match_type = "substring"
    if scope not in {"all", "system", "user", "assistant"}:
        logger.warning(f"[content_word_filter] 未知 scope={scope!r}，已回退为 all。")
        scope = "all"

    matchers = _compile_matcher(words, case_sensitive, match_type)
    if not matchers:
        return request_data

    request_data = dict(request_data) if isinstance(request_data, dict) else request_data
    messages = ((request_data or {}).get("messages") or []) if isinstance(request_data, dict) else None
    if not messages:
        return request_data

    filtered_messages = []
    blocked = False
    for message in messages:
        role = message.get("role") if isinstance(message, dict) else getattr(message, "role", None)
        if scope != "all" and role != scope:
            filtered_messages.append(message)
            continue

        content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
        text = _extract_text(content)
        if not text:
            filtered_messages.append(message)
            continue

        try:
            new_text, changed = _check_and_filter_text(
                text=text,
                matchers=matchers,
                mode=mode,
                replacement=replacement,
            )
        except HTTPException:
            blocked = True
            raise

        if not changed:
            filtered_messages.append(message)
            continue

        if mode == "strip" and strip_empty_message:
            cleaned = new_text.strip()
            if not cleaned:
                continue

        if isinstance(message, dict):
            message = dict(message)
            message["content"] = _set_text(message.get("content"), new_text)
        else:
            try:
                message = message.__class__(**{**message.model_dump(), "content": new_text})
            except Exception:
                message = dict(getattr(message, "model_dump", lambda: message)())
                message["content"] = new_text

        filtered_messages.append(message)

    if blocked:
        return request_data

    if isinstance(request_data, dict):
        request_data = dict(request_data)
        request_data["messages"] = filtered_messages

    logger.debug(
        f"[content_word_filter] mode={mode}, match_type={match_type}, scope={scope}, "
        f"words={len(words)}, messages={len(filtered_messages)}"
    )
    return request_data


def setup(manager: Any) -> None:
    logger.info(f"[{PLUGIN_INFO['name']}] 正在初始化...")
    register_inbound_interceptor(
        interceptor_id="content_word_filter",
        callback=content_word_filter_interceptor,
        priority=100,
        plugin_name=PLUGIN_INFO["name"],
        metadata={
            "description": PLUGIN_INFO["description"],
            "stage": "inbound_interceptors",
            "params_hint": PLUGIN_INFO["metadata"]["params_hint"],
            "params_schema": PLUGIN_INFO["metadata"]["params_schema"],
        },
    )
    logger.info(f"[{PLUGIN_INFO['name']}] 已注册入站拦截器")


def teardown(manager: Any) -> None:
    logger.info(f"[{PLUGIN_INFO['name']}] 正在清理...")
    unregister_inbound_interceptor("content_word_filter")
    logger.info(f"[{PLUGIN_INFO['name']}] 已清理完成")


def unload() -> None:
    logger.info(f"[{PLUGIN_INFO['name']}] Plugin unloading...")
