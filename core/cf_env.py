"""
Cloudflare Workers 运行时上下文工具。

- 使用 contextvars 在请求作用域保存当前 Cloudflare env
- 提供 D1/KV/R2 等 binding 读取辅助
"""

from __future__ import annotations

import os
import contextvars
from contextlib import contextmanager
from typing import Any, Optional


_cf_env_var: contextvars.ContextVar[Any] = contextvars.ContextVar("cf_env", default=None)


def set_cf_env(env: Any) -> contextvars.Token:
    """设置当前请求的 Cloudflare env，返回 token 供 reset 使用。"""
    return _cf_env_var.set(env)


def reset_cf_env(token: contextvars.Token) -> None:
    """恢复到 set_cf_env 之前的上下文。"""
    _cf_env_var.reset(token)


def clear_cf_env() -> None:
    """清空当前请求作用域中的 Cloudflare env。"""
    _cf_env_var.set(None)


def get_cf_env(default: Any = None) -> Any:
    """获取当前请求作用域中的 Cloudflare env。"""
    env = _cf_env_var.get()
    return default if env is None else env


def _read_binding(env: Any, name: str) -> Any:
    """从 Cloudflare env 对象中读取 binding。"""
    if env is None:
        return None

    # dict-like
    try:
        if isinstance(env, dict):
            return env.get(name)
    except Exception:
        pass

    # mapping-like
    try:
        getter = getattr(env, "get", None)
        if callable(getter):
            value = getter(name)
            if value is not None:
                return value
    except Exception:
        pass

    # attribute-like
    try:
        return getattr(env, name)
    except Exception:
        return None


def get_binding(name: str, *, required: bool = False) -> Any:
    """按 binding 名称读取当前请求 env 中的对象。"""
    value = _read_binding(get_cf_env(), name)
    if value is None and required:
        raise RuntimeError(f"Cloudflare binding '{name}' not found in current request env")
    return value


def get_d1(binding_name: Optional[str] = None, *, required: bool = True) -> Any:
    """获取 D1 binding。

    默认 binding 名称：
    - D1_BINDING 环境变量
    - 否则使用 "DB"
    """
    name = (binding_name or os.getenv("D1_BINDING") or "DB").strip()
    return get_binding(name, required=required)


def get_kv(binding_name: str, *, required: bool = True) -> Any:
    """获取 KV binding。"""
    return get_binding(binding_name, required=required)


def get_r2(binding_name: str, *, required: bool = True) -> Any:
    """获取 R2 binding。"""
    return get_binding(binding_name, required=required)


@contextmanager
def cf_env_context(env: Any):
    """在上下文中临时设置 Cloudflare env。"""
    token = set_cf_env(env)
    try:
        yield
    finally:
        reset_cf_env(token)
