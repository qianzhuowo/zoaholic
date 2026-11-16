"""
流式/非流式转换工具

核心功能：
- force_stream: 客户端请求非流 → 内部走流式打上游 → 拼装成非流式 JSON 返回
- force_non_stream: 客户端请求流式 → 内部走非流打上游 → 拆成 SSE 事件流返回

本模块处理 OpenAI Chat（即 Canonical）格式，并提供各方言共用的 SSE 解析/编码工具。
Claude / Gemini / Responses 的原生转换由各自方言模块实现并注册到 DialectDefinition。
"""

import codecs
import json
import random
import string
import time

from .log_config import logger
from .stream_errors import extract_stream_error


def encode_sse(data, event=None):
    """编码一个 SSE 事件；带 event 时输出 `event:` 行（Claude / Responses 风格）。"""
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    if event:
        return f"event: {event}\ndata: {payload}\n\n"
    return f"data: {payload}\n\n"


async def iter_sse_events(stream_generator):
    """
    从 SSE 文本流中逐个取出 JSON 事件。

    按行缓冲，不假设上游 chunk 与 SSE 事件边界对齐；忽略注释（keepalive）、[DONE]
    与无法解析的 data 行。内部错误信封（dict）原样产出。

    Yields:
        (event_type, data): event_type 为最近的 `event:` 行（可能为空），data 为解析后的 JSON。
    """
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    buffer = ""
    event_type = ""

    def parse_line(line):
        nonlocal event_type
        line = line.rstrip("\r")
        if not line:
            event_type = ""
            return None
        if line.startswith(":"):
            return None
        if line.startswith("event:"):
            event_type = line[6:].strip()
            return None
        if not line.startswith("data:"):
            return None
        data_str = line[5:].strip()
        if not data_str or data_str == "[DONE]":
            return None
        try:
            return json.loads(data_str)
        except json.JSONDecodeError:
            return None

    async for chunk in stream_generator:
        if isinstance(chunk, dict):
            yield "", chunk
            continue
        if isinstance(chunk, (bytes, bytearray)):
            chunk = decoder.decode(bytes(chunk))
        elif not isinstance(chunk, str):
            continue
        buffer += chunk
        while True:
            newline = buffer.find("\n")
            if newline < 0:
                break
            line, buffer = buffer[:newline], buffer[newline + 1:]
            data = parse_line(line)
            if data is not None:
                yield event_type, data

    buffer += decoder.decode(b"", final=True)
    for line in buffer.split("\n"):
        data = parse_line(line)
        if data is not None:
            yield event_type, data


def parse_json_body(chunks):
    """将非流式响应的全部 chunk 拼接并解析为 JSON；已是 dict 的单个 chunk 直接返回。"""
    if len(chunks) == 1 and isinstance(chunks[0], dict):
        return chunks[0]
    raw = bytearray()
    for chunk in chunks:
        if isinstance(chunk, (bytes, bytearray)):
            raw.extend(chunk)
        elif isinstance(chunk, str):
            raw.extend(chunk.encode("utf-8"))
        else:
            raise ValueError(f"Unexpected response chunk type: {type(chunk).__name__}")
    return json.loads(bytes(raw))


async def assemble_stream_to_json(stream_generator):
    """
    收集流式 SSE 响应，拼装成标准 OAI 非流式 JSON 响应。
    
    Args:
        stream_generator: 产出 SSE 字符串或 error dict 的异步生成器
    
    Returns:
        dict: 标准 OpenAI chat.completion 格式的响应
    """
    msg_id = ""
    msg_model = ""
    created = 0
    role = "assistant"
    content = ""
    reasoning_content = ""
    refusal = None
    tool_calls_accum = {}  # index → {id, type, function: {name, arguments}}
    finish_reason = None
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0

    try:
        async for _event, data in iter_sse_events(stream_generator):
            if not isinstance(data, dict):
                continue
            # 错误响应直接返回
            if extract_stream_error(data):
                return data

            if data.get("id") and not msg_id:
                msg_id = data["id"]
            if data.get("model"):
                msg_model = data["model"]
            if data.get("created"):
                created = data["created"]

            choices = data.get("choices", [])
            if choices:
                choice = choices[0]
                delta = choice.get("delta", {})

                if delta.get("role"):
                    role = delta["role"]
                if "content" in delta and isinstance(delta["content"], str):
                    content += delta["content"]
                if "reasoning_content" in delta and isinstance(delta["reasoning_content"], str):
                    reasoning_content += delta["reasoning_content"]
                if "refusal" in delta and isinstance(delta["refusal"], str):
                    refusal = (refusal or "") + delta["refusal"]

                if delta.get("tool_calls"):
                    for tc in delta["tool_calls"]:
                        idx = tc.get("index", 0)
                        if idx not in tool_calls_accum:
                            tool_calls_accum[idx] = {
                                "id": tc.get("id", ""),
                                "type": tc.get("type", "function"),
                                "function": {
                                    "name": tc.get("function", {}).get("name", ""),
                                    "arguments": "",
                                },
                            }
                        else:
                            if tc.get("id"):
                                tool_calls_accum[idx]["id"] = tc["id"]
                            if tc.get("type"):
                                tool_calls_accum[idx]["type"] = tc["type"]
                        fn = tc.get("function", {})
                        if fn.get("name"):
                            tool_calls_accum[idx]["function"]["name"] = fn["name"]
                        if "arguments" in fn:
                            tool_calls_accum[idx]["function"]["arguments"] += fn["arguments"]

                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]

            usage = data.get("usage")
            if usage and isinstance(usage, dict):
                pt = usage.get("prompt_tokens")
                ct = usage.get("completion_tokens")
                tt = usage.get("total_tokens")
                if pt and pt > 0:
                    prompt_tokens = pt
                if ct and ct > 0:
                    completion_tokens = ct
                if tt and tt > 0:
                    total_tokens = tt

    except Exception as e:
        logger.error(f"[stream_convert] Error assembling stream: {e}")
        return {
            "error": f"Stream assembly error: {e}",
            "status_code": 502,
            "details": str(e),
        }

    # 组装 message
    message = {"role": role, "refusal": refusal}
    if tool_calls_accum:
        message["content"] = content if content else None
        message["tool_calls"] = [tool_calls_accum[i] for i in sorted(tool_calls_accum.keys())]
    else:
        message["content"] = content
    if reasoning_content:
        message["reasoning_content"] = reasoning_content

    if not total_tokens and (prompt_tokens or completion_tokens):
        total_tokens = prompt_tokens + completion_tokens
    if not msg_id:
        random_str = "".join(random.choices(string.ascii_letters + string.digits, k=29))
        msg_id = f"chatcmpl-{random_str}"

    assembled = {
        "id": msg_id,
        "object": "chat.completion",
        "created": created or int(time.time()),
        "model": msg_model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "logprobs": None,
                "finish_reason": finish_reason or "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
        "system_fingerprint": None,
    }

    logger.info(
        f"[stream_convert] Assembled: model={msg_model}, "
        f"content_len={len(content)}, reasoning_len={len(reasoning_content)}, "
        f"tool_calls={len(tool_calls_accum)}, "
        f"tokens={prompt_tokens}+{completion_tokens}={total_tokens}"
    )

    return assembled


async def convert_json_to_sse(json_response, model=""):
    """
    将非流式 JSON 响应拆成 SSE 事件流。
    
    Args:
        json_response: dict, 标准 OAI chat.completion 格式
        model: 模型名
    
    Yields:
        str: SSE 格式的字符串（始终是字符串，可直接交给 StreamingResponse）
    """
    if not isinstance(json_response, dict):
        raise TypeError(f"convert_json_to_sse expects a dict, got {type(json_response).__name__}")

    if "error" in json_response:
        yield encode_sse(json_response)
        return

    msg_id = json_response.get("id", "")
    msg_model = json_response.get("model", model)
    created = json_response.get("created", int(time.time()))

    choices = json_response.get("choices", [])
    if not choices:
        yield encode_sse(json_response)
        yield "data: [DONE]\n\n"
        return

    choice = choices[0]
    message = choice.get("message", {})
    role = message.get("role", "assistant")
    content = message.get("content", "")
    reasoning_content = message.get("reasoning_content", "")
    tool_calls = message.get("tool_calls", [])
    finish_reason = choice.get("finish_reason", "stop")
    usage = json_response.get("usage")

    # Build canonical chunks directly: generate_sse_response is async and accepts
    # a timestamp, not a Chat response ID or a created keyword.
    def content_chunk(delta):
        return "data: " + json.dumps({
            "id": msg_id, "object": "chat.completion.chunk", "created": created,
            "model": msg_model, "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        }, ensure_ascii=False) + "\n\n"

    yield content_chunk({"role": role})

    # 发送 reasoning_content（如果有）
    if reasoning_content:
        # 分块发送避免单条 SSE 过大
        chunk_size = 200
        for i in range(0, len(reasoning_content), chunk_size):
            chunk = reasoning_content[i:i + chunk_size]
            delta = {"reasoning_content": chunk}
            sse_data = {
                "id": msg_id, "object": "chat.completion.chunk",
                "created": created, "model": msg_model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            }
            yield f"data: {json.dumps(sse_data, ensure_ascii=False)}\n\n"

    # 发送 content
    if content:
        chunk_size = 200
        for i in range(0, len(content), chunk_size):
            chunk = content[i:i + chunk_size]
            yield content_chunk({"content": chunk})

    if message.get("refusal"):
        yield content_chunk({"refusal": message["refusal"]})

    # 发送 tool_calls（如果有）
    if tool_calls:
        for idx, tc in enumerate(tool_calls):
            # 第一个 chunk: id + type + function.name
            delta = {
                "tool_calls": [{
                    "index": idx,
                    "id": tc.get("id", ""),
                    "type": tc.get("type", "function"),
                    "function": {"name": tc.get("function", {}).get("name", ""), "arguments": ""},
                }]
            }
            sse_data = {
                "id": msg_id, "object": "chat.completion.chunk",
                "created": created, "model": msg_model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            }
            yield f"data: {json.dumps(sse_data, ensure_ascii=False)}\n\n"

            # arguments 分块
            args = tc.get("function", {}).get("arguments", "")
            if args:
                for i in range(0, len(args), 200):
                    delta = {"tool_calls": [{"index": idx, "function": {"arguments": args[i:i + 200]}}]}
                    sse_data = {
                        "id": msg_id, "object": "chat.completion.chunk",
                        "created": created, "model": msg_model,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(sse_data, ensure_ascii=False)}\n\n"

    # 发送 finish + usage
    finish_data = {
        "id": msg_id, "object": "chat.completion.chunk",
        "created": created, "model": msg_model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
    }
    if usage:
        finish_data["usage"] = usage
    yield f"data: {json.dumps(finish_data, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"
