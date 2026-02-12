"""
Cloudflare Workers Python 入口。

职责：
1) 将当前请求的 Cloudflare `env` 绑定到 contextvars（供 D1/KV/R2 访问）
2) 将请求转发给 FastAPI ASGI 应用（main.app）

说明：
- 需要在 wrangler.toml 中将 `main` 指向 worker.py
- 运行时依赖 Cloudflare Python Workers 提供的 `asgi` 适配器
"""

from __future__ import annotations

from core.cf_env import cf_env_context
from main import app


try:
    from asgi import fetch as asgi_fetch
    _ASGI_FETCH_AVAILABLE = True
except Exception as e:  # pragma: no cover
    asgi_fetch = None
    _ASGI_FETCH_AVAILABLE = False

    _ASGI_FETCH_IMPORT_ERROR = e


async def on_fetch(request, env):
    """Cloudflare Workers 入口函数。"""
    if not _ASGI_FETCH_AVAILABLE:
        from starlette.responses import JSONResponse

        return JSONResponse(
            {
                "error": "Cloudflare Python runtime adapter 'asgi' is required for Workers deployment.",
                "detail": str(_ASGI_FETCH_IMPORT_ERROR),
            },
            status_code=500,
        )
    with cf_env_context(env):
        return await asgi_fetch(app, request, env)
