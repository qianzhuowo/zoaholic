# Zoaholic 多端点格式（Multi-Dialect API）支持规划

> 版本：v1.0
> 日期：2025-12-12
> 状态：已实现（基础版）

## 1. 背景与目标

### 1.1 现状

当前 Zoaholic 仅对外暴露 OpenAI 兼容格式的 `/v1/...` 端点：
- 用户输入：OpenAI Chat Completions 格式
- 内部路由：基于 `api.yaml` 配置选择 provider
- 上游适配：通过 channel adapters 把 OpenAI 格式转为各上游的原生格式（Gemini、Claude、Azure 等）

```
用户 → [OpenAI 格式] → Zoaholic → [Gemini/Claude/...格式] → 上游 Provider
```

### 1.2 需求

- 支持用户使用 **Gemini 原生格式**（`/v1beta/models/{model}:generateContent`）调用网关
- 支持用户使用 **Claude 原生格式**（`/v1/messages`）调用网关
- 后续可扩展更多厂商格式
- 保持现有 OpenAI 端点完全兼容

### 1.3 设计目标

| 目标 | 说明 |
|------|------|
| **统一管线** | 所有格式共享调度、重试、统计、插件等核心逻辑 |
| **透传优化** | 同格式请求（如 Gemini→Gemini）可透传，避免双重转换 |
| **最小侵入** | 不影响现有 OpenAI 端点，不重构核心调度模块 |
| **易于扩展** | 新增格式只需实现转换器，无需理解全系统 |

---

## 2. 整体架构

### 2.1 架构层次

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         External API Layer                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐ │
│  │ OpenAI 端点  │  │ Gemini 端点  │  │ Claude 端点  │  │ 其他端点... │ │
│  │ /v1/chat/... │  │ /v1beta/...  │  │ /v1/messages │  │             │ │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬──────┘ │
└─────────┼─────────────────┼─────────────────┼─────────────────┼────────┘
          │                 │                 │                 │
          ▼                 ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      Dialect Layer (方言层)                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │
│  │ OpenAI       │  │ Gemini       │  │ Claude       │                  │
│  │ Dialect      │  │ Dialect      │  │ Dialect      │                  │
│  │ (passthrough)│  │ (converter)  │  │ (converter)  │                  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                  │
│         │                 │                 │                          │
│         │    ┌────────────┴────────────┐    │                          │
│         │    │    透传检测 (Passthrough │    │                          │
│         │    │    Detection)           │    │                          │
│         │    └────────────┬────────────┘    │                          │
│         ▼                 ▼                 ▼                          │
│  ┌─────────────────────────────────────────────────────────────┐       │
│  │              Canonical Request (OpenAI 风格)                │       │
│  │              RequestModel / UnifiedRequest                  │       │
│  └──────────────────────────┬──────────────────────────────────┘       │
└─────────────────────────────┼──────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     Core Pipeline (核心管线)                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────────┐  │
│  │ 路由调度    │→ │ 权重/重试   │→ │ 统计日志    │→ │ 插件拦截器   │  │
│  │ (routing)   │  │ (handler)   │  │ (stats)     │  │ (interceptors)│  │
│  └─────────────┘  └─────────────┘  └─────────────┘  └───────────────┘  │
└─────────────────────────────┬──────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     Channel Layer (渠道层) - 已有                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐ │
│  │ openai       │  │ gemini       │  │ claude       │  │ vertex/...  │ │
│  │ channel      │  │ channel      │  │ channel      │  │ channel     │ │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬──────┘ │
└─────────┼─────────────────┼─────────────────┼─────────────────┼────────┘
          │                 │                 │                 │
          ▼                 ▼                 ▼                 ▼
       上游 OpenAI       上游 Gemini       上游 Claude       其他上游
```

### 2.2 透传优化路径

当入口格式与目标上游格式相同时，启用透传：

```
用户 [Gemini 格式] 
    │
    ▼
┌───────────────────────────────────┐
│ Gemini Dialect                    │
│ 检测：入口格式 == 目标 engine？    │
│   → 是：标记 passthrough=true     │
│   → 否：转换为 Canonical          │
└───────────────────────────────────┘
    │
    ▼ (passthrough=true)
┌───────────────────────────────────┐
│ Core Pipeline                     │
│ - 路由选择目标 provider           │
│ - 确认 engine=gemini              │
│ - 跳过 Canonical→Gemini 转换      │
│ - 直接透传原始 payload            │
└───────────────────────────────────┘
    │
    ▼
上游 Gemini（原样请求）
```

**透传策略（宽松模式）**：

透传是**默认行为**，只要入口格式与上游 engine 匹配即可。以下情况会在透传基础上做轻量处理：

| 场景 | 处理方式 | 是否仍为透传 |
|------|----------|--------------|
| 入口格式 == 上游 engine | 直接透传 | ✅ 是 |
| 模型需要重命名 | 透传 + 替换 model 字段 | ✅ 是 |
| 需要注入 system_prompt | 透传 + 追加 systemInstruction/system | ✅ 是 |
| 需要应用 post_body_parameter_overrides | 透传 + 合并覆写字段 | ✅ 是 |
| 请求拦截器需要修改 payload | 透传 + 应用拦截器修改 | ✅ 是 |
| 入口格式 ≠ 上游 engine | 完整转换（Canonical 中转） | ❌ 否 |

**透传本质**：跳过 `parse_request` → Canonical → `get_*_payload` 的双重转换，直接把原生 payload 发往上游。透传时仍可对 payload 做轻量级修改（字段替换/追加），但不重构整个消息结构。

---

## 3. 关键模块设计

### 3.1 Dialect Registry（方言注册中心）

**文件**: `core/dialects/registry.py`

```python
from dataclasses import dataclass
from typing import Callable, Optional, Dict, Any, AsyncIterator

@dataclass
class DialectDefinition:
    """
    方言定义：描述一种外部 API 格式
    
    Attributes:
        id: 方言唯一标识 (openai, gemini, claude)
        name: 显示名称
        description: 描述
        parse_request: 将原生请求转为 Canonical 的函数
        render_response: 将 Canonical 响应转为原生格式的函数
        render_stream: 将 Canonical SSE 流转为原生流格式的函数
        detect_passthrough: 检测是否可透传的函数
        target_engine: 透传时对应的 engine (如 gemini dialect → gemini engine)
    """
    id: str
    name: str
    description: Optional[str] = None
    
    # 请求转换：native → Canonical
    parse_request: Optional[Callable] = None
    
    # 响应转换：Canonical → native (非流式)
    render_response: Optional[Callable] = None
    
    # 流式转换：Canonical SSE → native stream
    render_stream: Optional[Callable] = None
    
    # 透传检测
    detect_passthrough: Optional[Callable] = None
    
    # 对应的上游 engine（用于透传匹配）
    target_engine: Optional[str] = None

# 全局注册表
_DIALECT_REGISTRY: Dict[str, DialectDefinition] = {}

def register_dialect(dialect: DialectDefinition) -> None:
    """注册方言"""
    _DIALECT_REGISTRY[dialect.id] = dialect

def get_dialect(dialect_id: str) -> Optional[DialectDefinition]:
    """获取方言定义"""
    return _DIALECT_REGISTRY.get(dialect_id)

def list_dialects() -> list:
    """列出所有已注册方言"""
    return list(_DIALECT_REGISTRY.values())
```

### 3.2 Gemini Dialect 实现示例

**文件**: `core/dialects/gemini.py`

```python
from typing import Dict, Any, Optional
from core.models import RequestModel, Message, ContentItem

async def parse_gemini_request(
    native_body: Dict[str, Any],
    path_params: Dict[str, str],
    headers: Dict[str, str],
) -> RequestModel:
    """
    将 Gemini 原生请求转为 Canonical (OpenAI) 格式
    
    Gemini 格式:
    {
        "contents": [
            {"role": "user", "parts": [{"text": "..."}]}
        ],
        "systemInstruction": {"parts": [{"text": "..."}]},
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 1024,
            "topP": 0.9
        }
    }
    """
    messages = []
    
    # 处理 systemInstruction
    if "systemInstruction" in native_body:
        sys_parts = native_body["systemInstruction"].get("parts", [])
        sys_text = "".join(p.get("text", "") for p in sys_parts)
        if sys_text:
            messages.append(Message(role="system", content=sys_text))
    
    # 处理 contents
    for content in native_body.get("contents", []):
        role = content.get("role", "user")
        if role == "model":
            role = "assistant"
        
        parts = content.get("parts", [])
        # 简化处理：只取 text
        text_parts = [p.get("text", "") for p in parts if "text" in p]
        content_text = "".join(text_parts)
        
        # 处理图片
        content_items = []
        for p in parts:
            if "text" in p:
                content_items.append(ContentItem(type="text", text=p["text"]))
            elif "inlineData" in p:
                # 转换为 OpenAI 格式的 image_url
                mime_type = p["inlineData"].get("mimeType", "image/png")
                data = p["inlineData"].get("data", "")
                content_items.append(ContentItem(
                    type="image_url",
                    image_url={"url": f"data:{mime_type};base64,{data}"}
                ))
        
        if len(content_items) > 1 or any(c.type != "text" for c in content_items):
            messages.append(Message(role=role, content=content_items))
        else:
            messages.append(Message(role=role, content=content_text))
    
    # 从 path 获取模型名
    model = path_params.get("model", "")
    
    # 处理 generationConfig
    gen_config = native_body.get("generationConfig", {})
    
    return RequestModel(
        model=model,
        messages=messages,
        temperature=gen_config.get("temperature"),
        max_tokens=gen_config.get("maxOutputTokens"),
        top_p=gen_config.get("topP"),
        stream=":streamGenerateContent" in path_params.get("action", ""),
    )


async def render_gemini_response(
    canonical_response: Dict[str, Any],
    model: str,
) -> Dict[str, Any]:
    """
    将 Canonical (OpenAI) 响应转为 Gemini 格式
    
    OpenAI 格式:
    {
        "choices": [{"message": {"role": "assistant", "content": "..."}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
    }
    
    Gemini 格式:
    {
        "candidates": [{"content": {"role": "model", "parts": [{"text": "..."}]}}],
        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 20, "totalTokenCount": 30}
    }
    """
    choices = canonical_response.get("choices", [])
    content = ""
    if choices:
        message = choices[0].get("message", {})
        content = message.get("content", "")
    
    usage = canonical_response.get("usage", {})
    
    return {
        "candidates": [{
            "content": {
                "role": "model",
                "parts": [{"text": content}]
            },
            "finishReason": "STOP"
        }],
        "usageMetadata": {
            "promptTokenCount": usage.get("prompt_tokens", 0),
            "candidatesTokenCount": usage.get("completion_tokens", 0),
            "totalTokenCount": usage.get("total_tokens", 0)
        }
    }


async def render_gemini_stream(canonical_sse_chunk: str) -> str:
    """
    将 Canonical SSE chunk 转为 Gemini SSE 格式
    
    输入: "data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n"
    输出: "data: {"candidates":[{"content":{"parts":[{"text":"Hello"}]}}]}\n\n"
    """
    import json
    
    if not canonical_sse_chunk.startswith("data: "):
        return canonical_sse_chunk
    
    data_str = canonical_sse_chunk[6:].strip()
    if data_str == "[DONE]":
        # Gemini 没有 [DONE]，返回空或最后一个带 finishReason 的块
        return ""
    
    try:
        canonical = json.loads(data_str)
        choices = canonical.get("choices", [])
        
        if not choices:
            return ""
        
        delta = choices[0].get("delta", {})
        content = delta.get("content", "")
        reasoning = delta.get("reasoning_content", "")
        
        gemini_chunk = {
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": []
                }
            }]
        }
        
        if reasoning:
            gemini_chunk["candidates"][0]["content"]["parts"].append({
                "thought": True,
                "text": reasoning
            })
        if content:
            gemini_chunk["candidates"][0]["content"]["parts"].append({
                "text": content
            })
        
        finish_reason = choices[0].get("finish_reason")
        if finish_reason:
            gemini_chunk["candidates"][0]["finishReason"] = "STOP"
        
        # 处理 usage
        usage = canonical.get("usage")
        if usage:
            gemini_chunk["usageMetadata"] = {
                "promptTokenCount": usage.get("prompt_tokens", 0),
                "candidatesTokenCount": usage.get("completion_tokens", 0),
                "totalTokenCount": usage.get("total_tokens", 0)
            }
        
        return f"data: {json.dumps(gemini_chunk)}\n\n"
    
    except json.JSONDecodeError:
        return canonical_sse_chunk


def detect_passthrough(dialect_id: str, target_engine: str) -> bool:
    """
    检测是否可以透传（宽松模式）
    
    唯一条件：入口方言与目标 engine 匹配
    其他差异（模型重命名、system_prompt、overrides）通过轻量级修改处理
    """
    dialect_to_engine = {"gemini": "gemini", "claude": "claude", "openai": "openai"}
    return dialect_to_engine.get(dialect_id) == target_engine


def register():
    """注册 Gemini 方言"""
    from .registry import register_dialect, DialectDefinition
    
    register_dialect(DialectDefinition(
        id="gemini",
        name="Google Gemini",
        description="Google Gemini API 原生格式",
        parse_request=parse_gemini_request,
        render_response=render_gemini_response,
        render_stream=render_gemini_stream,
        detect_passthrough=detect_gemini_passthrough,
        target_engine="gemini",
    ))
```

### 3.3 透传处理流程

**文件**: `core/dialects/passthrough.py`

```python
from typing import Dict, Any, Optional, Tuple
from core.models import RequestModel

class PassthroughContext:
    """透传上下文，携带原始请求信息和轻量级修改"""
    
    def __init__(
        self,
        enabled: bool,
        dialect_id: str,
        original_payload: Dict[str, Any],
        original_headers: Dict[str, str],
        modifications: Dict[str, Any] = None,  # 透传时需要的轻量级修改
    ):
        self.enabled = enabled
        self.dialect_id = dialect_id
        self.original_payload = original_payload
        self.original_headers = original_headers
        self.modifications = modifications or {}


async def evaluate_passthrough(
    dialect_id: str,
    original_payload: Dict[str, Any],
    original_headers: Dict[str, str],
    target_provider: Dict[str, Any],
    request_model: str,
) -> PassthroughContext:
    """
    评估是否可以透传（宽松模式）
    
    唯一条件：格式匹配。其他修改在透传时轻量处理。
    """
    target_engine = target_provider.get("engine")
    can_passthrough = detect_passthrough(dialect_id, target_engine)
    
    modifications = {}
    if can_passthrough:
        # 收集需要的轻量级修改（不影响透传决策）
        model_dict = target_provider.get("_model_dict_cache", {})
        if request_model in model_dict and model_dict[request_model] != request_model:
            modifications["model_rename"] = model_dict[request_model]
        
        if target_provider.get("preferences", {}).get("system_prompt"):
            modifications["system_prompt"] = target_provider["preferences"]["system_prompt"]
        
        if target_provider.get("preferences", {}).get("post_body_parameter_overrides"):
            modifications["overrides"] = target_provider["preferences"]["post_body_parameter_overrides"]
    
    return PassthroughContext(
        enabled=can_passthrough,
        dialect_id=dialect_id,
        original_payload=original_payload,
        original_headers=original_headers,
        modifications=modifications,
    )
```

### 3.4 路由集成

**文件**: `routes/dialects/gemini.py`

```python
from fastapi import APIRouter, Request, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse

from routes.deps import rate_limit_dependency, verify_api_key, get_model_handler
from core.dialects.registry import get_dialect
from core.dialects.passthrough import evaluate_passthrough, PassthroughContext

router = APIRouter(prefix="/v1beta", tags=["Gemini Dialect"])


@router.post("/models/{model}:generateContent", dependencies=[Depends(rate_limit_dependency)])
@router.post("/models/{model}:streamGenerateContent", dependencies=[Depends(rate_limit_dependency)])
async def gemini_generate_content(
    model: str,
    request: Request,
    background_tasks: BackgroundTasks,
    api_index: int = Depends(verify_api_key),
):
    """
    Gemini 原生格式端点
    
    支持:
    - POST /v1beta/models/{model}:generateContent (非流式)
    - POST /v1beta/models/{model}:streamGenerateContent (流式)
    """
    # 获取方言
    dialect = get_dialect("gemini")
    
    # 解析原始请求
    native_body = await request.json()
    headers = dict(request.headers)
    path_params = {
        "model": model,
        "action": request.url.path.split(":")[-1],
    }
    
    # 转换为 Canonical
    canonical_request = await dialect.parse_request(native_body, path_params, headers)
    
    # 获取 model handler
    model_handler = get_model_handler()
    
    # 这里需要扩展 request_model 以支持透传上下文
    # 详见下方"核心管线改造"
    
    response = await model_handler.request_model(
        request_data=canonical_request,
        api_index=api_index,
        background_tasks=background_tasks,
        dialect_id="gemini",
        original_payload=native_body,
        original_headers=headers,
    )
    
    # 如果是透传，response 已经是 Gemini 格式
    # 如果不是透传，需要转换响应格式
    if isinstance(response, StreamingResponse):
        # 流式：包装生成器进行格式转换
        async def convert_stream():
            async for chunk in response.body_iterator:
                converted = await dialect.render_stream(chunk)
                if converted:
                    yield converted
        
        return StreamingResponse(
            convert_stream(),
            media_type="text/event-stream",
        )
    else:
        # 非流式：转换 JSON
        content = response.body
        import json
        canonical_json = json.loads(content)
        gemini_json = await dialect.render_response(canonical_json, model)
        return JSONResponse(content=gemini_json)
```

---

## 4. 核心管线改造

### 4.1 ModelRequestHandler 扩展

需要在 `core/handler.py` 的 `request_model` 方法中支持透传：

```python
async def request_model(
    self,
    request_data: Union[RequestModel, ...],
    api_index: int,
    background_tasks: BackgroundTasks,
    endpoint: Optional[str] = None,
    # 新增参数
    dialect_id: Optional[str] = None,
    original_payload: Optional[Dict[str, Any]] = None,
    original_headers: Optional[Dict[str, str]] = None,
) -> Response:
    """
    处理模型请求
    
    新增透传支持:
    - dialect_id: 入口方言 ID
    - original_payload: 原始请求体（用于透传）
    - original_headers: 原始请求头（用于透传）
    """
    ...
    
    # 在选择 provider 后，评估透传
    if dialect_id and original_payload:
        passthrough_ctx = await evaluate_passthrough(
            dialect_id=dialect_id,
            canonical_request=request_data,
            original_payload=original_payload,
            original_headers=original_headers or {},
            target_provider=provider,
        )
        
        if passthrough_ctx.enabled:
            # 透传模式：直接使用原始 payload
            return await process_request_passthrough(
                passthrough_ctx=passthrough_ctx,
                provider=provider,
                background_tasks=background_tasks,
                ...
            )
    
    # 非透传：走原有逻辑
    ...
```

### 4.2 process_request 透传版本

```python
async def process_request_passthrough(
    passthrough_ctx: PassthroughContext,
    provider: Dict[str, Any],
    background_tasks: BackgroundTasks,
    app: "FastAPI",
    ...
) -> Response:
    """
    透传模式的请求处理
    
    直接把原始 payload 发送到上游，跳过 Canonical→Native 转换
    """
    # 构建 URL（使用 channel 的 URL 构建逻辑）
    from .channels import get_channel
    channel = get_channel(provider.get("engine"))
    
    # 获取 API key
    if provider.get("api"):
        api_key = await provider_api_circular_list[provider['provider']].next(original_model)
    else:
        api_key = None
    
    # 直接使用原始 payload，只替换认证信息
    url = ...  # 根据 channel 构建
    headers = {**passthrough_ctx.original_headers}
    
    # 添加认证
    if api_key and channel.auth_header:
        auth_header = channel.auth_header.replace("{api_key}", api_key)
        header_name, header_value = auth_header.split(": ", 1)
        headers[header_name] = header_value
    
    payload = passthrough_ctx.original_payload
    
    # 发送请求（复用现有的 client_manager 和错误处理）
    async with app.state.client_manager.get_client(url) as client:
        if request.stream:
            # 流式透传：直接转发上游 SSE
            generator = channel.stream_adapter(client, url, headers, payload, model, timeout)
            return LoggingStreamingResponse(generator, ...)
        else:
            # 非流式透传
            response = await client.post(url, headers=headers, json=payload, timeout=timeout)
            return Response(content=response.content, media_type="application/json")
```

---

## 5. 实现计划

### 5.1 已完成功能

| 功能 | 状态 | 说明 |
|------|------|------|
| 方言注册中心 | ✅ 已完成 | `core/dialects/registry.py` |
| 端点自动注册 | ✅ 已完成 | `core/dialects/router.py` |
| Gemini 方言 | ✅ 已完成 | 支持 generateContent / streamGenerateContent |
| Claude 方言 | ✅ 已完成 | 支持 /v1/messages |
| 透传机制 | ✅ 已完成 | `core/dialects/passthrough.py` |
| OpenAI 方言 | ✅ 已完成 | 作为默认 Canonical 格式 |

### 5.2 文件结构

```
core/dialects/
├── __init__.py          # 自动加载所有方言并注册路由
├── registry.py          # 注册中心（DialectDefinition + EndpointDefinition）
├── router.py            # 路由自动注册逻辑
├── passthrough.py       # 透传检测与上下文
├── openai.py            # OpenAI 方言（Canonical）
├── gemini.py            # Gemini 方言（含端点定义）
└── claude.py            # Claude 方言（含端点定义）
```

### 5.3 新增方言指南

只需 **一个文件** 即可完成新方言的添加：

#### 步骤 1：创建方言模块

```python
# core/dialects/xxx.py

from typing import Any, Dict
from core.models import RequestModel
from .registry import DialectDefinition, EndpointDefinition, register_dialect


async def parse_xxx_request(native_body, path_params, headers) -> RequestModel:
    """Native -> Canonical 转换"""
    # 实现转换逻辑
    pass


async def render_xxx_response(canonical_response, model) -> Dict[str, Any]:
    """Canonical -> Native 响应转换"""
    # 实现转换逻辑
    pass


async def render_xxx_stream(canonical_sse_chunk: str) -> str:
    """Canonical SSE -> Native SSE 流式转换"""
    # 实现转换逻辑
    pass


def register() -> None:
    """注册方言"""
    register_dialect(DialectDefinition(
        id="xxx",
        name="XXX API",
        description="XXX API 原生格式",
        parse_request=parse_xxx_request,
        render_response=render_xxx_response,
        render_stream=render_xxx_stream,
        target_engine="xxx",  # 用于透传匹配
        endpoints=[
            # 定义端点，路由会自动注册
            EndpointDefinition(
                path="/v1/xxx/chat",
                methods=["POST"],
                tags=["XXX Dialect"],
            ),
            # 支持路径参数
            EndpointDefinition(
                prefix="/v1",
                path="/models/{model}:generate",
                methods=["POST"],
            ),
            # 自定义处理函数
            EndpointDefinition(
                path="/v1/xxx/models",
                methods=["GET"],
                handler=list_xxx_models,  # 自定义函数
            ),
        ],
    ))
```

#### 步骤 2：在 `__init__.py` 中注册

```python
# core/dialects/__init__.py

from . import xxx as xxx_dialect
xxx_dialect.register()
```

#### 完成！

路由会自动注册，无需修改其他文件。

### 5.4 EndpointDefinition 说明

| 属性 | 类型 | 说明 |
|------|------|------|
| `path` | str | 路由路径，支持 FastAPI 参数语法如 `/models/{model}` |
| `methods` | List[str] | HTTP 方法，默认 `["POST"]` |
| `prefix` | str | 路由前缀，如 `/v1beta` |
| `tags` | List[str] | OpenAPI tags |
| `handler` | Callable | 自定义处理函数（可选） |
| `summary` | str | 端点摘要 |
| `description` | str | 端点描述 |

### 5.5 透传机制

透传检测逻辑（宽松模式）：

```python
# 唯一条件：格式匹配
def detect_passthrough(dialect_id, target_engine):
    return dialect_id == target_engine

# 透传时的轻量级修改（不阻止透传）
def apply_passthrough_modifications(payload, modifications, dialect_id):
    # 模型重命名
    # system_prompt 注入
    # overrides 深度合并
```

---

## 6. 测试策略

### 6.1 单元测试

每个方言需要覆盖：

```python
# test/dialects/test_gemini.py

class TestGeminiDialect:
    
    async def test_parse_simple_request(self):
        """测试简单请求转换"""
        native = {
            "contents": [{"role": "user", "parts": [{"text": "Hello"}]}]
        }
        canonical = await parse_gemini_request(native, {"model": "gemini-pro"}, {})
        assert canonical.model == "gemini-pro"
        assert canonical.messages[0].content == "Hello"
    
    async def test_parse_with_system(self):
        """测试 systemInstruction 转换"""
        ...
    
    async def test_render_response(self):
        """测试响应转换"""
        ...
    
    async def test_render_stream(self):
        """测试流式响应转换"""
        ...
    
    async def test_passthrough_detection(self):
        """测试透传条件检测"""
        ...
```

### 6.2 集成测试

```python
# test/dialects/test_integration.py

class TestDialectIntegration:
    
    async def test_gemini_to_gemini_passthrough(self):
        """Gemini 格式 → Gemini 上游：应该透传"""
        ...
    
    async def test_gemini_to_openai_convert(self):
        """Gemini 格式 → OpenAI 上游：应该转换"""
        ...
    
    async def test_roundtrip_consistency(self):
        """往返转换一致性：Gemini → Canonical → Gemini"""
        ...
```

---

## 7. 配置示例

### 7.1 api.yaml 配置

方言层对 `api.yaml` 透明，用户仍然按原有方式配置 providers：

```yaml
providers:
  # Gemini 渠道 - 用户可通过 Gemini 原生格式或 OpenAI 格式访问
  - provider: gemini-main
    base_url: https://generativelanguage.googleapis.com/v1beta
    api: YOUR_API_KEY
    engine: gemini
    model:
      - gemini-2.5-pro
      - gemini-2.5-flash

api_keys:
  - api: sk-xxx
    model:
      - gemini-main/*
```

### 7.2 使用方式

**OpenAI 格式调用**（现有方式）:
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-xxx" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-2.5-pro",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

**Gemini 原生格式调用**（新增）:
```bash
curl -X POST http://localhost:8000/v1beta/models/gemini-2.5-pro:generateContent \
  -H "Authorization: Bearer sk-xxx" \
  -H "Content-Type: application/json" \
  -d '{
    "contents": [{"role": "user", "parts": [{"text": "Hello"}]}]
  }'
```

**Claude 原生格式调用**（新增）:
```bash
curl -X POST http://localhost:8000/v1/messages \
  -H "Authorization: Bearer sk-xxx" \
  -H "Content-Type: application/json" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "claude-3-5-sonnet",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 1024
  }'
```

---

## 8. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 透传条件判断错误 | 请求格式不兼容导致上游报错 | 严格的透传条件检查；fallback 到转换模式 |
| 格式转换丢失字段 | 特殊功能不可用 | 完善转换器；记录不支持的字段 |
| 性能开销 | 转换增加延迟 | 透传优化；异步转换 |
| 插件兼容性 | 插件假设 OpenAI 格式 | 在 Canonical 层应用插件，透传时跳过 |

---

## 9. 未来扩展

### 9.1 更多方言支持

- **Azure OpenAI**: 与 OpenAI 格式接近，主要处理 URL 和认证差异
- **AWS Bedrock**: 支持 invoke-model 格式
- **Cohere**: 支持 chat 和 generate 格式
- **Ollama**: 兼容 OpenAI，可透传

### 9.2 能力扩展

- **Embeddings**: 各家 embedding API 格式适配
- **Image Generation**: DALL-E、Imagen 等格式
- **Speech**: TTS/STT 格式适配

### 9.3 管理功能

- **方言启用/禁用**: 通过配置控制暴露哪些方言端点
- **方言级统计**: 按入口格式统计请求量
- **方言级限流**: 不同格式不同限流策略

---

## 附录 A：字段映射表

### A.1 Gemini ↔ OpenAI 字段映射

| Gemini | OpenAI | 说明 |
|--------|--------|------|
| `contents[].role` | `messages[].role` | `model` → `assistant` |
| `contents[].parts[].text` | `messages[].content` | 文本内容 |
| `contents[].parts[].inlineData` | `messages[].content[].image_url` | 图片 |
| `systemInstruction` | `messages[0]` (role=system) | 系统提示 |
| `generationConfig.temperature` | `temperature` | 温度 |
| `generationConfig.maxOutputTokens` | `max_tokens` | 最大输出 |
| `generationConfig.topP` | `top_p` | Top-P |
| `generationConfig.topK` | `top_k` | Top-K |
| `tools[].function_declarations` | `tools[].function` | 工具定义 |
| `usageMetadata.promptTokenCount` | `usage.prompt_tokens` | 输入 token |
| `usageMetadata.candidatesTokenCount` | `usage.completion_tokens` | 输出 token |

### A.2 Claude ↔ OpenAI 字段映射

| Claude | OpenAI | 说明 |
|--------|--------|------|
| `messages[].role` | `messages[].role` | 角色相同 |
| `messages[].content` | `messages[].content` | 内容（string 或 array） |
| `system` | `messages[0]` (role=system) | 系统提示 |
| `max_tokens` | `max_tokens` | 最大输出 |
| `temperature` | `temperature` | 温度 |
| `top_p` | `top_p` | Top-P |
| `tools` | `tools` | 工具定义（格式略有差异） |
| `tool_choice` | `tool_choice` | 工具选择 |
| `thinking` | `thinking` | 思考模式（扩展字段） |
| `usage.input_tokens` | `usage.prompt_tokens` | 输入 token |
| `usage.output_tokens` | `usage.completion_tokens` | 输出 token |

---

## 附录 B：目录结构

```
zoaholic/
├── core/
│   ├── dialects/                    # 方言层
│   │   ├── __init__.py              # 自动加载方言并注册路由
│   │   ├── registry.py              # 注册中心（含 EndpointDefinition）
│   │   ├── router.py                # 路由自动注册逻辑
│   │   ├── passthrough.py           # 透传机制
│   │   ├── openai.py                # OpenAI 方言（Canonical）
│   │   ├── gemini.py                # Gemini 方言（含端点定义）
│   │   └── claude.py                # Claude 方言（含端点定义）
│   ├── channels/                    # 渠道层（上游适配）
│   ├── handler.py                   # 支持透传参数
│   ├── request.py
│   ├── response.py
│   └── ...
├── routes/
│   ├── __init__.py                  # 引入 dialect_router
│   ├── chat.py                      # OpenAI 端点
│   └── ...
├── test/
│   └── dialects/
│       └── test_dialects.py
└── docs/
    └── multi-dialect-api-plan.md    # 本文档
```

注意：方言端点路由已集成到方言模块中（`core/dialects/xxx.py`），
无需在 `routes/dialects/` 目录下创建单独的路由文件。