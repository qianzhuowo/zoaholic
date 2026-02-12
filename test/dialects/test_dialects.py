import asyncio

from core.dialects.gemini import parse_gemini_request, render_gemini_response
from core.dialects.claude import parse_claude_request
from core.dialects.passthrough import detect_passthrough, apply_passthrough_modifications


def run(coro):
    return asyncio.run(coro)


def test_gemini_parse_simple_text():
    native = {
        "contents": [
            {"role": "user", "parts": [{"text": "Hello"}]}
        ]
    }
    canonical = run(parse_gemini_request(native, {"model": "gemini-pro", "action": "generateContent"}, {}))
    assert canonical.model == "gemini-pro"
    assert canonical.messages[0].role == "user"
    assert canonical.messages[0].content == "Hello"
    assert canonical.stream is False or canonical.stream is None


def test_gemini_parse_system_instruction():
    native = {
        "systemInstruction": {"parts": [{"text": "SYS"}]},
        "contents": [{"role": "user", "parts": [{"text": "Hi"}]}],
        "generationConfig": {"temperature": 0.7},
    }
    canonical = run(parse_gemini_request(native, {"model": "gemini-pro", "action": "generateContent"}, {}))
    assert canonical.messages[0].role == "system"
    assert canonical.messages[0].content == "SYS"
    assert canonical.messages[1].role == "user"
    assert canonical.temperature == 0.7


def test_gemini_render_response():
    canonical_resp = {
        "choices": [{"message": {"role": "assistant", "content": "OK"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    }
    gemini_resp = run(render_gemini_response(canonical_resp, "gemini-pro"))
    assert gemini_resp["candidates"][0]["content"]["parts"][0]["text"] == "OK"
    assert gemini_resp["usageMetadata"]["promptTokenCount"] == 1
    assert gemini_resp["usageMetadata"]["candidatesTokenCount"] == 2
    assert gemini_resp["usageMetadata"]["totalTokenCount"] == 3


def test_claude_parse_basic_blocks_and_system():
    native = {
        "model": "claude-3-5-sonnet",
        "system": "SYS",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Hi"}]},
        ],
        "max_tokens": 123,
    }
    canonical = run(parse_claude_request(native, {}, {}))
    assert canonical.model == "claude-3-5-sonnet"
    assert canonical.messages[0].role == "system"
    assert canonical.messages[0].content == "SYS"
    assert canonical.messages[1].role == "user"
    assert canonical.messages[1].content == "Hello"
    assert canonical.messages[2].role == "assistant"
    assert canonical.messages[2].content == "Hi"
    assert canonical.max_tokens == 123


def test_detect_passthrough_registry_and_fallback():
    assert detect_passthrough("gemini", "gemini") is True
    assert detect_passthrough("gemini", "openai") is False
    # 未注册 dialect 的回退规则：dialect_id == engine
    assert detect_passthrough("foo", "foo") is True
    assert detect_passthrough("foo", "bar") is False


def test_apply_passthrough_system_prompt_openai():
    payload = {"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]}
    mods = {"system_prompt": "SYS"}
    new_payload = apply_passthrough_modifications(
        payload, mods, "openai", request_model="gpt-4o", original_model="gpt-4o"
    )
    assert new_payload["messages"][0]["role"] == "system"
    assert "SYS" in new_payload["messages"][0]["content"]


def test_apply_passthrough_overrides_deep_merge():
    payload = {"generationConfig": {"temperature": 0.1}, "foo": 1}
    mods = {
        "overrides": {
            "all": {"generationConfig": {"topP": 0.9}},
            "gpt-4o": {"foo": 2},
            "bar": 3,
        }
    }
    new_payload = apply_passthrough_modifications(
        payload, mods, "gemini", request_model="gpt-4o", original_model="gpt-4o"
    )
    assert new_payload["generationConfig"]["temperature"] == 0.1
    assert new_payload["generationConfig"]["topP"] == 0.9
    assert new_payload["foo"] == 2
    assert new_payload["bar"] == 3