"""
Claude Bearer Auth 插件

将 Anthropic Claude 渠道的 x-api-key 认证方式转换为 Authorization: Bearer 格式。
适用于兼容 Anthropic API 但使用 Bearer token 认证的第三方服务。

使用方式：在渠道配置的 preferences 中启用此插件：
  preferences:
    enabled_plugins:
      - claude_bearer_auth
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.plugins import PluginManager


PLUGIN_INFO = {
    "name": "claude_bearer_auth",
    "version": "1.0.0",
    "description": "将 Claude 渠道的 x-api-key 转换为 Authorization: Bearer 格式",
    "author": "Zoaholic",
    "dependencies": [],
    "metadata": {
        "category": "interceptor",
        "tags": ["claude", "auth", "bearer"],
    },
}


async def claude_bearer_auth_interceptor(request, engine, provider, api_key, url, headers, payload):
    """
    请求拦截器：将 x-api-key 替换为 Authorization: Bearer

    仅在 headers 中存在 x-api-key 时生效，将其取出并改为 Bearer 格式。
    """
    if "x-api-key" in headers:
        api_key_value = headers.pop("x-api-key")
        headers["Authorization"] = f"Bearer {api_key_value}"
    return url, headers, payload


def setup(manager: "PluginManager"):
    """插件初始化，注册请求拦截器"""
    from core.plugins.interceptors import register_request_interceptor

    register_request_interceptor(
        interceptor_id="claude_bearer_auth",
        callback=claude_bearer_auth_interceptor,
        priority=10,
        plugin_name=PLUGIN_INFO["name"],
        overwrite=True,
    )


def teardown(manager: "PluginManager"):
    """插件卸载，注销请求拦截器"""
    from core.plugins.interceptors import unregister_request_interceptor
    unregister_request_interceptor("claude_bearer_auth")
