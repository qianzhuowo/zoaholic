import asyncio
import sys
import types
from time import time


if "utils" not in sys.modules:
    utils_stub = types.ModuleType("utils")

    def safe_get(data, *keys, default=None):
        current = data
        for key in keys:
            try:
                current = current[key]
            except (KeyError, IndexError, TypeError):
                return default
        return current if current is not None else default

    utils_stub.safe_get = safe_get
    sys.modules["utils"] = utils_stub

if "core.stats" not in sys.modules:
    stats_stub = types.ModuleType("core.stats")

    async def update_stats(*args, **kwargs):
        return None

    stats_stub.update_stats = update_stats
    sys.modules["core.stats"] = stats_stub

if "core.utils" not in sys.modules:
    core_utils_stub = types.ModuleType("core.utils")

    def truncate_for_logging(value, *args, **kwargs):
        return value

    core_utils_stub.truncate_for_logging = truncate_for_logging
    sys.modules["core.utils"] = core_utils_stub

from core.dialects.claude import parse_claude_usage
from core.streaming import LoggingStreamingResponse


async def _collect_stream_chunks(response: LoggingStreamingResponse):
    chunks = []
    async for chunk in response._logging_iterator():
        chunks.append(chunk)
    return chunks


def test_parse_claude_usage_supports_message_start_nested_usage():
    event = {
        "type": "message_start",
        "message": {
            "id": "msg_1",
            "usage": {
                "input_tokens": 123,
            },
        },
    }

    usage = parse_claude_usage(event)

    assert usage == {
        "prompt_tokens": 123,
        "completion_tokens": 0,
        "total_tokens": 123,
    }


def test_logging_streaming_response_buffers_split_sse_and_merges_usage_non_zero():
    raw_chunks = [
        b'data: {"choices":[{"delta":{"content":"Hel',
        b'lo"}}]}\n',
        b'data: {"message":{"usage":{"input_tokens":12}}}\n',
        b'data: {"usage":{"output_tokens":5}}',
    ]

    async def body_iter():
        for chunk in raw_chunks:
            yield chunk

    current_info = {
        "start_time": time(),
        "endpoint": "/v1/messages",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    response = LoggingStreamingResponse(
        body_iter(),
        media_type="text/event-stream",
        current_info=current_info,
        dialect_id="claude",
    )

    streamed_chunks = asyncio.run(_collect_stream_chunks(response))

    assert b"".join(streamed_chunks) == b"".join(raw_chunks)
    assert current_info["prompt_tokens"] == 12
    assert current_info["completion_tokens"] == 5
    assert current_info["total_tokens"] == 17
    assert current_info["content_start_time"] >= 0
