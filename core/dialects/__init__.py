"""
方言（Dialect）模块

自动导入并注册所有内置方言实现（openai / gemini / claude）。
方言用于描述“外部 API 输入/输出格式”，与 channels（上游适配）相互独立。

用法：
- 通过 registry 访问已注册方言
- 导入 dialect_router 并注册到 FastAPI 应用
- 在方言模块中定义 endpoints 即可自动注册路由
"""

from .registry import (
    DialectDefinition,
    EndpointDefinition,
    ParseRequest,
    RenderResponse,
    RenderStream,
    DetectPassthrough,
    register_dialect,
    unregister_dialect,
    get_dialect,
    list_dialects,
    list_dialect_ids,
)

# 导入内置方言模块以触发注册
from . import openai as openai_dialect
from . import openai_responses as openai_responses_dialect
from . import gemini as gemini_dialect
from . import claude as claude_dialect

# 调用 register() 完成注册
openai_dialect.register()
openai_responses_dialect.register()
gemini_dialect.register()
claude_dialect.register()

# 导入路由模块并注册路由
from .router import dialect_router, register_dialect_routes

# 自动注册所有方言路由
register_dialect_routes()

__all__ = [
    "DialectDefinition",
    "EndpointDefinition",
    "ParseRequest",
    "RenderResponse",
    "RenderStream",
    "DetectPassthrough",
    "register_dialect",
    "unregister_dialect",
    "get_dialect",
    "list_dialects",
    "list_dialect_ids",
    "dialect_router",
    "register_dialect_routes",
]