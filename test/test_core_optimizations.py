import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.channels.claude_channel import get_claude_payload
from core.channels.openai_responses_channel import _normalize_responses_base_url
from core.models import RequestModel
from core.request import _prepend_system_prompt
from utils import is_local_api_key


CLAUDE_PROVIDER = {
    "provider": "anthropic",
    "base_url": "https://api.anthropic.com/v1",
    "model": ["claude-3-5-sonnet"],
}


def test_prepend_system_prompt_preserves_unset_request_fields():
    request = RequestModel(
        model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": "hello"}],
    )

    patched = _prepend_system_prompt(request, "system prompt")
    dumped = patched.model_dump(exclude_unset=True)

    assert dumped["messages"][0]["role"] == "system"
    assert "temperature" not in dumped
    assert "top_p" not in dumped
    assert "presence_penalty" not in dumped
    assert "frequency_penalty" not in dumped
    assert "n" not in dumped


def test_is_local_api_key_supports_sk_and_zk_prefixes():
    assert is_local_api_key("sk-abc") is True
    assert is_local_api_key("zk-abc") is True
    assert is_local_api_key("pk-abc") is False
    assert is_local_api_key(None) is False


def test_claude_thinking_payload_only_includes_non_empty_fields_and_enforces_constraints():
    async def _run():
        request = RequestModel(
            model="claude-3-5-sonnet",
            messages=[{"role": "user", "content": "hello"}],
            thinking={"budget_tokens": 2048},
            top_p=0.9,
            top_k=40,
        )
        return await get_claude_payload(request, "anthropic", CLAUDE_PROVIDER, api_key="test-key")

    _, _, payload = asyncio.run(_run())

    assert payload["thinking"] == {"budget_tokens": 2048}
    assert payload["temperature"] == 1
    assert "top_p" not in payload
    assert "top_k" not in payload


def test_empty_claude_thinking_config_is_ignored():
    async def _run():
        request = RequestModel(
            model="claude-3-5-sonnet",
            messages=[{"role": "user", "content": "hello"}],
            thinking={},
            top_p=0.9,
        )
        return await get_claude_payload(request, "anthropic", CLAUDE_PROVIDER, api_key="test-key")

    _, _, payload = asyncio.run(_run())

    assert "thinking" not in payload
    assert payload["top_p"] == 0.9
    assert "temperature" not in payload


def test_normalize_responses_base_url_supports_v1_and_responses_inputs():
    v1_url, responses_url, models_url = _normalize_responses_base_url("https://example.com/proxy/v1")
    assert v1_url == "https://example.com/proxy/v1"
    assert responses_url == "https://example.com/proxy/v1/responses"
    assert models_url == "https://example.com/proxy/v1/models"

    v1_url, responses_url, models_url = _normalize_responses_base_url("https://example.com/proxy/v1/responses")
    assert v1_url == "https://example.com/proxy/v1"
    assert responses_url == "https://example.com/proxy/v1/responses"
    assert models_url == "https://example.com/proxy/v1/models"
