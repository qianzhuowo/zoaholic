"""
OpenAI Responses API 方言

实现 OpenAI Responses API 格式的输入输出转换：
- input (Responses API) -> messages (Chat Completions)
- 支持 /v1/responses 端点
- 支持流式和非流式响应
"""

from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

from core.json_utils import json_dumps_text
from core.models import RequestModel, Message, ContentItem
from core.stream_convert import encode_sse, iter_sse_events
from core.stream_errors import extract_stream_error
from core.usage import extract_cache_usage

from .registry import DialectDefinition, EndpointDefinition, register_dialect

if TYPE_CHECKING:
    from fastapi import Request, BackgroundTasks


# ============================================================
# Responses API -> Chat Completions 转换
# ============================================================


def convert_responses_input_to_messages(input_data: Union[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    将 Responses API 的 input 转换为 Chat Completions 的 messages 格式

    Responses API input 格式：
    - 字符串: 直接作为 user 消息
    - 数组: 消息列表，每个消息包含 role 和 content
      - content 可以是字符串或数组
      - 数组中的 item 类型: input_text, input_image, input_audio 等

    Args:
        input_data: Responses API 的 input 字段

    Returns:
        Chat Completions 格式的 messages 列表
    """
    messages = []

    # 如果是字符串，直接作为 user 消息
    if isinstance(input_data, str):
        return [{"role": "user", "content": input_data}]

    # 如果是列表，逐个转换
    if isinstance(input_data, list):
        for item in input_data:
            if isinstance(item, str):
                # 字符串直接作为 user 消息
                messages.append({"role": "user", "content": item})
            elif isinstance(item, dict):
                # Responses represents calls/results as separate input items, not messages.
                if item.get("type") == "function_call":
                    call = {"id": item.get("call_id") or item.get("id"), "type": "function",
                            "function": {"name": item.get("name", ""),
                                         "arguments": item.get("arguments", "{}")}}
                    if messages and messages[-1].get("role") == "assistant":
                        messages[-1].setdefault("tool_calls", []).append(call)
                    else:
                        messages.append({"role": "assistant", "content": None, "tool_calls": [call]})
                    continue
                if item.get("type") == "function_call_output":
                    output = item.get("output", "")
                    messages.append({"role": "tool", "tool_call_id": item.get("call_id"),
                                     "content": output if isinstance(output, str) else json_dumps_text(output)})
                    continue
                role = item.get("role", "user")
                content = item.get("content")

                # 处理 content
                if content is None:
                    continue

                if isinstance(content, str):
                    # 字符串 content 直接使用
                    messages.append({"role": role, "content": content})
                elif isinstance(content, list):
                    # 数组 content，需要转换 type
                    converted_content = []
                    for content_item in content:
                        if isinstance(content_item, str):
                            converted_content.append({"type": "text", "text": content_item})
                        elif isinstance(content_item, dict):
                            item_type = content_item.get("type", "")

                            # input_text -> text
                            if item_type in ("input_text", "output_text"):
                                converted_content.append({
                                    "type": "text",
                                    "text": content_item.get("text", "")
                                })
                            # input_image -> image_url
                            elif item_type == "input_image":
                                image_url = content_item.get("image_url") or content_item.get("url")
                                if image_url:
                                    converted_content.append({
                                        "type": "image_url",
                                        "image_url": {"url": image_url}
                                    })
                            # input_audio -> file
                            elif item_type == "input_audio":
                                data = content_item.get("input_audio", {}).get("data", "")
                                if data:
                                    converted_content.append({
                                        "type": "file",
                                        "file": {"mime_type": "audio/wav", "data": data}
                                    })
                            # input_file -> file
                            elif item_type == "input_file":
                                file_obj = {}
                                for field in ["file_id", "filename", "file_url", "file_data"]:
                                    val = content_item.get(field)
                                    if val is not None:
                                        if field == "file_url" or (field == "file_data" and str(val).startswith("data:")):
                                            file_obj["url"] = val
                                        elif field == "file_data":
                                            file_obj["data"] = val
                                        else:
                                            file_obj[field] = val
                                converted_content.append({
                                    "type": "file",
                                    "file": file_obj
                                })
                            # 其他类型原样传递
                            else:
                                # 尝试保留原有格式（如 text, image_url）
                                if item_type == "text":
                                    converted_content.append(content_item)
                                elif item_type == "image_url":
                                    converted_content.append(content_item)
                                else:
                                    converted_content.append(content_item)

                    if converted_content:
                        messages.append({"role": role, "content": converted_content})
                else:
                    # 其他类型尝试转为字符串
                    messages.append({"role": role, "content": str(content)})

    return messages


def convert_responses_tools(tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
    """
    将 Responses API 的 tools 转换为 Chat Completions 格式

    Responses API tools 格式更扁平化：
    - type: "custom" / "function" / "web_search" / "file_search" 等
    - name: 工具名称
    - description: 工具描述

    Chat Completions tools 格式：
    - type: "function"
    - function: { name, description, parameters }
    """
    if not tools:
        return None

    converted_tools = []
    for tool in tools:
        tool_type = tool.get("type", "")

        # 内置工具类型，不需要转换
        if tool_type in ("web_search", "file_search", "code_interpreter", "image_generation"):
            converted_tools.append(tool)
            continue

        # custom 类型转换为 function
        if tool_type == "custom":
            converted_tools.append({
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {"type": "object", "properties": {}})
                }
            })
        # function 类型
        elif tool_type == "function":
            # 可能已经是 Chat Completions 格式
            if "function" in tool:
                converted_tools.append(tool)
            else:
                converted_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.get("name", ""),
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters", {"type": "object", "properties": {}})
                    }
                })
        else:
            # 其他类型原样传递
            converted_tools.append(tool)

    for source, converted in zip(tools, converted_tools):
        if "strict" in source and isinstance(converted.get("function"), dict):
            converted["function"]["strict"] = source["strict"]
    return converted_tools if converted_tools else None


async def parse_responses_request(
    native_body: Dict[str, Any],
    path_params: Dict[str, str],
    headers: Dict[str, str],
) -> RequestModel:
    """
    将 Responses API 请求转换为 Canonical (RequestModel) 格式

    主要转换：
    - input -> messages
    - reasoning.effort -> (通过模型后缀处理)
    - tools 格式转换
    """
    # 获取基本字段
    model = native_body.get("model", "")
    input_data = native_body.get("input", [])
    stream = native_body.get("stream", False)

    # 转换 input -> messages
    messages = convert_responses_input_to_messages(input_data)

    # 处理顶层 instructions -> system message（Responses API 公开语义）
    instructions = native_body.get("instructions")
    if instructions and isinstance(instructions, str) and instructions.strip():
        # 插入到 messages 最前面作为 system 消息
        messages.insert(0, {"role": "system", "content": instructions.strip()})

    # 构建基本请求
    request_data = {
        "model": model,
        "messages": messages,
        "stream": stream,
    }

    # 转换 tools
    tools = convert_responses_tools(native_body.get("tools"))
    if tools:
        request_data["tools"] = tools

    # 透传 reasoning 参数（如 {"effort": "high"}），由 channel 层决定如何处理
    reasoning = native_body.get("reasoning")
    if reasoning is not None:
        request_data["reasoning"] = reasoning

    # 可选参数映射
    optional_fields = [
        "temperature", "top_p", "max_tokens", "max_completion_tokens", "max_output_tokens",
        "presence_penalty", "frequency_penalty", "n", "user",
        "tool_choice", "response_format", "stream_options"
    ]

    for field in optional_fields:
        if field in native_body and native_body[field] is not None:
            # max_output_tokens -> max_tokens
            if field == "max_output_tokens":
                request_data["max_tokens"] = native_body[field]
            else:
                request_data[field] = native_body[field]

    if isinstance(request_data.get("tool_choice"), dict):
        choice = request_data["tool_choice"]
        if choice.get("type") == "function" and "name" in choice:
            request_data["tool_choice"] = {"type": "function", "function": {"name": choice["name"]}}

    # text.format -> response_format
    text_format = native_body.get("text", {}).get("format")
    if text_format:
        format_type = text_format.get("type")
        if format_type == "json_object":
            request_data["response_format"] = {"type": "json_object"}
        elif format_type == "json_schema":
            request_data["response_format"] = text_format

    return RequestModel(**request_data)


def _responses_usage_from_canonical(usage: Any) -> Dict[str, Any]:
    """将内核 OpenAI usage 转为 Responses API usage。

    Responses API 的缓存命中字段位于 input_tokens_details.cached_tokens；集中转换可以同时覆盖
    非流式响应和 usage-only 流式事件。
    """
    usage = usage if isinstance(usage, dict) else {}
    response_usage: Dict[str, Any] = {
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }
    cache_usage = extract_cache_usage(usage)
    if cache_usage["cached_tokens"] > 0:
        response_usage["input_tokens_details"] = {"cached_tokens": cache_usage["cached_tokens"]}
    details = usage.get("completion_tokens_details") or {}
    if "reasoning_tokens" in details:
        response_usage["output_tokens_details"] = {"reasoning_tokens": details["reasoning_tokens"]}
    return response_usage


async def render_responses_response(
    canonical_response: Dict[str, Any],
    model: str,
) -> Dict[str, Any]:
    """
    将 Canonical (OpenAI Chat Completions) 响应转换为 Responses API 格式

    Chat Completions 响应:
    {
        "id": "chatcmpl-xxx",
        "choices": [{"message": {"role": "assistant", "content": "..."}}],
        "usage": {...}
    }

    Responses API 响应:
    {
        "id": "resp_xxx",
        "object": "response",
        "output": [
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "..."}]}
        ]
    }
    """
    # Make explicit re-rendering safe, including native failed Responses objects.
    if canonical_response.get("object") == "response" or canonical_response.get("error"):
        return canonical_response
    from .responses_stream import ResponsesStreamRenderer
    renderer = ResponsesStreamRenderer(model)
    choices = canonical_response.get("choices") or []
    if not choices:
        renderer.finish(error={"code": "empty_response", "message": "Upstream returned no choices"})
        return renderer.response
    choice = choices[0]
    message = dict(choice.get("message") or {})
    message["tool_calls"] = [dict(call, index=index)
                             for index, call in enumerate(message.get("tool_calls") or [])]
    renderer.feed({"choices": [{"index": 0, "delta": message,
                                "finish_reason": choice.get("finish_reason") or "stop"}],
                   "usage": canonical_response.get("usage")})
    renderer.finish()
    return renderer.response


def parse_responses_usage(data: Any) -> Optional[Dict[str, int]]:
    """从 Responses API 格式中提取 usage"""
    if not isinstance(data, dict):
        return None

    # Responses API 流式完成事件
    if data.get("type") == "response.completed":
        usage = data.get("response", {}).get("usage", {})
        if usage:
            prompt = usage.get("input_tokens", 0)
            completion = usage.get("output_tokens", 0)
            total = usage.get("total_tokens", 0) or (prompt + completion)
            # Responses API 的缓存命中字段位于 input_tokens_details，需要和普通 token 同步回传。
            cache_usage = extract_cache_usage(usage)
            return {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": total,
                **cache_usage,
            }

    # 非流式响应
    usage = data.get("usage")
    if usage:
        prompt = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
        completion = usage.get("output_tokens") or usage.get("completion_tokens") or 0
        total = usage.get("total_tokens") or (prompt + completion)
        # 非流式 Responses API 同样携带 input_tokens_details.cached_tokens，这里避免只解析 token 总数。
        cache_usage = extract_cache_usage(usage)
        if prompt or completion or cache_usage["cached_tokens"] or cache_usage["cache_creation_tokens"]:
            return {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": total,
                **cache_usage,
            }

    return None


# ============================================================
# 端点处理函数
# ============================================================


async def responses_handler(
    request: "Request",
    background_tasks: "BackgroundTasks",
    api_index: int,
    **kwargs,
):
    """
    OpenAI Responses API 端点 - POST /v1/responses

    将 Responses API 请求转换为 Chat Completions 格式，调用内部处理器
    """
    from routes.deps import get_model_handler

    native_body: Dict[str, Any] = await request.json()

    # 转换为 RequestModel
    request_model = await parse_responses_request(native_body, {}, {})

    model_handler = get_model_handler()

    # 调用内部处理器，传递原始 payload 用于透传判断
    return await model_handler.request_model(
        request_model,
        api_index,
        background_tasks,
        dialect_id="openai-responses",
        original_payload=native_body,
        original_headers=dict(request.headers),
        raw_request=request,
    )


async def responses_subpath_handler(
    request: "Request",
    background_tasks: "BackgroundTasks",
    api_index: int,
    **kwargs,
):
    """
    OpenAI Responses compact endpoint - POST /v1/responses/compact

    Compact is a native Responses auxiliary endpoint.  Keep the original payload
    for upstream passthrough and only build a minimal RequestModel for auth,
    rate limit, model routing, and provider selection.
    """
    from core.error_response import openai_error_response
    from routes.deps import get_model_handler

    try:
        native_body: Dict[str, Any] = await request.json()
    except Exception:
        native_body = {}

    model = native_body.get("model")
    if not isinstance(model, str) or not model.strip():
        return openai_error_response("Missing required parameter: model", 400)

    request_model = RequestModel(
        model=model.strip(),
        messages=[Message(role="user", content="")],
        stream=False,
    )

    model_handler = get_model_handler()
    return await model_handler.request_model(
        request_model,
        api_index,
        background_tasks,
        endpoint=request.url.path,
        dialect_id="openai-responses",
        original_payload=native_body,
        original_headers=dict(request.headers),
        raw_request=request,
        passthrough_only=True,
    )


# ============================================================
# 注册
# ============================================================


def _create_responses_stream_renderer():
    from .responses_stream import create_responses_stream_renderer
    return create_responses_stream_renderer()


# ============================================================
# 透传 stream_mode：Responses 原生流式 <-> 非流式
# ============================================================

_RESPONSES_TERMINAL_EVENTS = ("response.completed", "response.incomplete", "response.failed")


async def assemble_responses_stream(stream_generator) -> Dict[str, Any]:
    """Responses 原生 SSE -> Response 对象（force_stream 透传）。

    以终态事件携带的 response 为准；部分上游的终态 response.output 为空，
    此时用 response.output_item.done 收集到的输出项补齐。
    """
    final: Optional[Dict[str, Any]] = None
    items: Dict[int, Dict[str, Any]] = {}

    async for event_type, data in iter_sse_events(stream_generator):
        if not isinstance(data, dict):
            continue
        kind = data.get("type") or event_type
        if kind == "error" or extract_stream_error(data):
            return data
        if kind == "response.output_item.done" and isinstance(data.get("item"), dict):
            items[data.get("output_index", len(items))] = data["item"]
        elif kind in _RESPONSES_TERMINAL_EVENTS and isinstance(data.get("response"), dict):
            final = data["response"]

    if final is None:
        return {"error": {"type": "upstream_error",
                          "message": "Responses stream ended without a terminal response event"},
                "status_code": 502}
    if not final.get("output") and items:
        final = {**final, "output": [items[i] for i in sorted(items)]}
    return final


def _responses_item_events(index: int, item: Dict[str, Any], event):
    item_id = item.get("id")
    item_type = item.get("type")
    pending = {**item, "status": "in_progress"} if "status" in item else dict(item)

    if item_type == "message":
        yield event("response.output_item.added", output_index=index, item={**pending, "content": []})
        for content_index, part in enumerate(item.get("content") or []):
            where = {"item_id": item_id, "output_index": index, "content_index": content_index}
            part_type = part.get("type")
            if part_type in ("output_text", "refusal"):
                key = "text" if part_type == "output_text" else "refusal"
                empty = {**part, key: ""}
                if part_type == "output_text":
                    empty["annotations"] = []
                yield event("response.content_part.added", **where, part=empty)
                if part.get(key):
                    yield event(f"response.{part_type}.delta", **where, delta=part[key])
                yield event(f"response.{part_type}.done", **where, **{key: part.get(key) or ""})
            else:
                yield event("response.content_part.added", **where, part=part)
            yield event("response.content_part.done", **where, part=part)
    elif item_type == "reasoning":
        yield event("response.output_item.added", output_index=index, item={**pending, "summary": []})
        for summary_index, part in enumerate(item.get("summary") or []):
            where = {"item_id": item_id, "output_index": index, "summary_index": summary_index}
            yield event("response.reasoning_summary_part.added", **where, part={**part, "text": ""})
            if part.get("text"):
                yield event("response.reasoning_summary_text.delta", **where, delta=part["text"])
            yield event("response.reasoning_summary_text.done", **where, text=part.get("text") or "")
            yield event("response.reasoning_summary_part.done", **where, part=part)
    elif item_type == "function_call":
        arguments = item.get("arguments") or ""
        yield event("response.output_item.added", output_index=index, item={**pending, "arguments": ""})
        where = {"item_id": item_id, "output_index": index}
        if arguments:
            yield event("response.function_call_arguments.delta", **where, delta=arguments)
        yield event("response.function_call_arguments.done", **where, arguments=arguments)
    else:
        yield event("response.output_item.added", output_index=index, item=item)
    yield event("response.output_item.done", output_index=index, item=item)


async def split_responses_response(response: Dict[str, Any], model: str = ""):
    """Response 对象 -> Responses 原生 SSE 事件序列（force_non_stream 透传）。"""
    if not isinstance(response, dict):
        raise TypeError(f"split_responses_response expects a dict, got {type(response).__name__}")

    sequence = 0

    def event(kind: str, **fields) -> str:
        nonlocal sequence
        frame = encode_sse({"type": kind, "sequence_number": sequence, **fields}, kind)
        sequence += 1
        return frame

    status = response.get("status") or "completed"
    if status == "failed":
        yield event("response.failed", response=response)
        return

    pending = {**response, "status": "in_progress", "output": [], "usage": None}
    pending.setdefault("model", model)
    yield event("response.created", response=pending)
    yield event("response.in_progress", response=pending)
    for index, item in enumerate(response.get("output") or []):
        if isinstance(item, dict):
            for frame in _responses_item_events(index, item, event):
                yield frame
    terminal = "response.incomplete" if status == "incomplete" else "response.completed"
    yield event(terminal, response=response)


def register() -> None:
    """注册 OpenAI Responses API 方言"""
    register_dialect(
        DialectDefinition(
            id="openai-responses",
            name="OpenAI Responses API",
            description="OpenAI Responses API 格式（GPT-5/o1/o3 等新模型专用）",
            parse_request=parse_responses_request,
            render_response=render_responses_response,
            render_stream_factory=_create_responses_stream_renderer,
            parse_usage=parse_responses_usage,
            stream_to_response=assemble_responses_stream,
            response_to_stream=split_responses_response,
            target_engine="openai-responses",
            endpoints=[
                EndpointDefinition(
                    path="/v1/responses",
                    methods=["POST"],
                    handler=responses_handler,
                    tags=["Responses"],
                    summary="Create Response",
                    description="创建响应请求，兼容 OpenAI Responses API 格式（GPT-5/o1/o3 等新模型）",
                ),
                # POST /v1/responses/* - Responses 子端点通配透传入口
                EndpointDefinition(
                    path="/v1/responses/{subpath:path}",
                    passthrough_root="/v1/responses",
                    methods=["POST"],
                    handler=responses_subpath_handler,
                    tags=["Responses"],
                    summary="Responses Passthrough Subpaths",
                    description="OpenAI Responses API 子端点透传入口（仅在上游为 openai-responses 时可用）",
                    passthrough_only=True,
                ),
            ],
        )
    )
