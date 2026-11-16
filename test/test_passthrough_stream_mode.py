"""Offline regressions for stream_mode on same-dialect passthrough (issue #2).

Each dialect converts between its own native stream and non-stream shapes;
no production requests are made.
"""
import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from fastapi import BackgroundTasks, FastAPI

import core.dialects  # noqa: F401  registers built-in dialects
from core.dialects.passthrough import PassthroughContext
from core.dialects.registry import get_dialect
from core.models import RequestModel

OPENAI = {
    "id": "chatcmpl-1", "object": "chat.completion", "created": 1, "model": "m",
    "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
        "role": "assistant", "content": "你好", "reasoning_content": "think",
        "tool_calls": [{"id": "call_1", "type": "function",
                        "function": {"name": "read", "arguments": "{\"path\":\"a\"}"}}]}}],
    "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
}
CLAUDE = {
    "id": "msg_1", "type": "message", "role": "assistant", "model": "m",
    "content": [
        {"type": "thinking", "thinking": "think", "signature": "sig"},
        {"type": "text", "text": "你好"},
        {"type": "tool_use", "id": "toolu_1", "name": "read", "input": {"path": "a"}},
    ],
    "stop_reason": "tool_use", "stop_sequence": None,
    "usage": {"input_tokens": 3, "output_tokens": 4, "cache_read_input_tokens": 1},
}
GEMINI = {
    "candidates": [{"index": 0, "finishReason": "STOP", "content": {"role": "model", "parts": [
        {"text": "think", "thought": True},
        {"text": "你好", "thoughtSignature": "sig"},
        {"functionCall": {"name": "read", "args": {"path": "a"}}},
    ]}}],
    "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 4, "totalTokenCount": 7},
    "modelVersion": "m",
}
RESPONSES = {
    "id": "resp_1", "object": "response", "created_at": 1, "model": "m", "status": "completed",
    "output": [
        {"id": "rs_1", "type": "reasoning", "summary": [{"type": "summary_text", "text": "think"}]},
        {"id": "msg_1", "type": "message", "role": "assistant", "status": "completed",
         "content": [{"type": "output_text", "text": "你好", "annotations": []}]},
        {"id": "fc_1", "type": "function_call", "status": "completed", "call_id": "call_1",
         "name": "read", "arguments": "{\"path\":\"a\"}"},
    ],
    "usage": {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7},
}
NATIVE = {"openai": OPENAI, "claude": CLAUDE, "gemini": GEMINI, "openai-responses": RESPONSES}


def split(dialect_id, response):
    async def run():
        return [frame async for frame in get_dialect(dialect_id).response_to_stream(response, "m")]
    return asyncio.run(run())


def assemble(dialect_id, parts):
    async def source():
        for part in parts:
            yield part
    return asyncio.run(get_dialect(dialect_id).stream_to_response(source()))


@pytest.mark.parametrize("dialect_id", list(NATIVE))
def test_native_round_trip_survives_arbitrary_chunk_boundaries(dialect_id):
    frames = split(dialect_id, NATIVE[dialect_id])
    assert all(isinstance(frame, str) for frame in frames)
    wire = "".join(frames).encode()
    # Byte-sized chunks split multi-byte UTF-8 and SSE lines.
    assembled = assemble(dialect_id, [wire[i:i + 1] for i in range(len(wire))])
    if dialect_id == "openai":
        message = assembled["choices"][0]["message"]
        assert message["content"] == "你好" and message["reasoning_content"] == "think"
        assert message["tool_calls"][0]["function"] == OPENAI["choices"][0]["message"]["tool_calls"][0]["function"]
        assert assembled["choices"][0]["finish_reason"] == "tool_calls"
        assert assembled["usage"] == OPENAI["usage"]
    else:
        assert assembled == NATIVE[dialect_id]


def test_claude_stream_events_follow_protocol_order():
    kinds = [json.loads(f.split("data: ", 1)[1])["type"] for f in split("claude", CLAUDE)]
    assert kinds[0] == "message_start" and kinds[-2:] == ["message_delta", "message_stop"]
    assert kinds.count("content_block_start") == kinds.count("content_block_stop") == 3


def test_gemini_assembles_fragmented_text_parts():
    chunks = [
        {"candidates": [{"content": {"role": "model", "parts": [{"text": "你"}]}}]},
        {"candidates": [{"content": {"role": "model", "parts": [{"text": "好"}]}, "finishReason": "STOP"}],
         "usageMetadata": {"totalTokenCount": 2}},
    ]
    result = assemble("gemini", ["data: " + json.dumps(c) + "\n\n" for c in chunks])
    assert result["candidates"][0]["content"]["parts"] == [{"text": "你好"}]
    assert result["candidates"][0]["finishReason"] == "STOP"
    assert result["usageMetadata"] == {"totalTokenCount": 2}


def test_responses_fills_empty_terminal_output_from_items():
    item = RESPONSES["output"][1]
    wire = [
        "event: response.output_item.done\ndata: " + json.dumps(
            {"type": "response.output_item.done", "output_index": 0, "item": item}) + "\n\n",
        "event: response.completed\ndata: " + json.dumps(
            {"type": "response.completed", "response": {**RESPONSES, "output": []}}) + "\n\n",
    ]
    assert assemble("openai-responses", wire)["output"] == [item]


@pytest.mark.parametrize("dialect_id,event", [
    ("claude", {"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}),
    ("gemini", {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "quota"}}),
    ("openai-responses", {"type": "response.failed", "response": {"status": "failed",
                                                                 "error": {"message": "bad"}}}),
])
def test_stream_errors_surface_as_error_envelopes(dialect_id, event):
    from core.stream_errors import extract_stream_error
    assert extract_stream_error(assemble(dialect_id, ["data: " + json.dumps(event) + "\n\n"]))


ENGINES = {"openai": "openai", "claude": "claude", "gemini": "gemini", "openai-responses": "openai-responses"}
URLS = {"openai": "https://offline.invalid/v1/chat/completions",
        "claude": "https://offline.invalid/v1/messages",
        "gemini": "https://offline.invalid/v1beta/models/m:generateContent",
        "openai-responses": "https://offline.invalid/v1/responses"}


def run_passthrough(monkeypatch, dialect_id, *, client_stream, mode, endpoint=None):
    module = __import__("core.passthrough", fromlist=["x"])
    handler = __import__("core.handler", fromlist=["x"])
    seen = {}

    async def adapter(request, engine, provider, api_key):
        seen["adapter_stream"] = request.stream
        url = URLS[dialect_id]
        if dialect_id == "gemini" and request.stream:
            url = url.replace("generateContent", "streamGenerateContent")
        return url, {}, None

    async def interceptors(request, engine, provider, api_key, url, headers, payload, plugins):
        return url, headers, payload

    async def fetch_response(client, url, headers, payload, timeout, **kwargs):
        seen.update(url=url, payload=payload, upstream="non-stream")
        text = json.dumps(NATIVE[dialect_id], ensure_ascii=False)
        yield text[:7]
        yield text[7:]

    stream_frames = split(dialect_id, NATIVE[dialect_id])

    async def fetch_stream(client, url, headers, payload, timeout, **kwargs):
        seen.update(url=url, payload=payload, upstream="stream")
        for frame in stream_frames:
            yield frame

    async def resolve(app, key, **kwargs):
        return key

    @asynccontextmanager
    async def client(*args):
        yield object()

    channel = SimpleNamespace(passthrough_adapter=adapter, request_adapter=None,
                              passthrough_payload_adapter=None, passthrough_stream_adapter=None,
                              passthrough_response_adapter=None)
    monkeypatch.setattr(module, "get_engine", lambda *a: (ENGINES[dialect_id], None, mode))
    monkeypatch.setattr("core.channels.get_channel", lambda engine: channel)
    monkeypatch.setattr("core.plugins.interceptors.apply_request_interceptors", interceptors)
    monkeypatch.setattr(module, "_fetch_passthrough_response", fetch_response)
    monkeypatch.setattr(module, "_fetch_passthrough_stream", fetch_stream)
    monkeypatch.setattr(handler, "_resolve_oauth_api_key", resolve)
    monkeypatch.setattr(handler, "_fire_and_forget_channel_stats", lambda *a, **k: None)

    app = FastAPI()
    app.state.config = {}
    app.state.client_manager = SimpleNamespace(get_client=client)
    info = {"request_id": "test", "api_key": "test"}
    payload = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    if dialect_id != "gemini":
        payload["stream"] = client_stream
    ctx = PassthroughContext(enabled=True, dialect_id=dialect_id, original_payload=payload,
                             original_headers={}, modifications={})
    request = RequestModel(model="m", messages=[{"role": "user", "content": "hi"}], stream=client_stream)

    async def run():
        response = await module.process_request_passthrough(
            request, {"provider": "offline-test", "_model_dict_cache": {"m": "m"}}, BackgroundTasks(), app,
            lambda: info, lambda *a, **k: None, passthrough_ctx=ctx, endpoint=endpoint)
        body = []
        async for chunk in response.body_iterator:
            body.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
        return "".join(body)

    return asyncio.run(run()), seen


@pytest.mark.parametrize("dialect_id", list(NATIVE))
def test_force_non_stream_passthrough_returns_native_sse(monkeypatch, dialect_id):
    body, seen = run_passthrough(monkeypatch, dialect_id, client_stream=True, mode="force_non_stream")
    assert seen["upstream"] == "non-stream" and seen["adapter_stream"] is False
    if dialect_id == "gemini":
        assert "stream" not in seen["payload"] and ":generateContent" in seen["url"]
    else:
        assert seen["payload"]["stream"] is False
    assembled = assemble(dialect_id, [body])
    if dialect_id == "openai":
        assert assembled["choices"][0]["message"]["content"] == "你好"
    else:
        assert assembled == NATIVE[dialect_id]


@pytest.mark.parametrize("dialect_id", list(NATIVE))
def test_force_stream_passthrough_returns_native_json(monkeypatch, dialect_id):
    body, seen = run_passthrough(monkeypatch, dialect_id, client_stream=False, mode="force_stream")
    assert seen["upstream"] == "stream" and seen["adapter_stream"] is True
    result = json.loads(body)
    if dialect_id == "openai":
        assert result["choices"][0]["message"]["content"] == "你好"
    else:
        assert result == NATIVE[dialect_id]


def test_subpath_ignores_stream_mode(monkeypatch):
    body, seen = run_passthrough(monkeypatch, "claude", client_stream=False, mode="force_stream",
                                 endpoint="/v1/messages/count_tokens")
    assert seen["upstream"] == "non-stream" and seen["url"].endswith("/v1/messages/count_tokens")
    assert json.loads(body) == CLAUDE
