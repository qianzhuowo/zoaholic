"""
示例渠道插件

这是一个完整的渠道插件示例，展示如何创建自定义渠道。
您可以将此文件作为模板来创建自己的渠道插件。

插件规范：
1. 定义 PLUGIN_INFO 字典提供插件元信息
2. 实现 setup(manager) 函数用于初始化
3. 可选实现 teardown(manager) 函数用于清理
"""

import json
import asyncio
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.plugins import PluginManager

# ==================== 插件元信息 ====================

PLUGIN_INFO = {
    "name": "example_channel",
    "version": "1.0.0",
    "description": "示例渠道插件，展示如何创建自定义渠道",
    "author": "Your Name",
    "dependencies": [],  # 依赖的其他插件
    "metadata": {
        "category": "channel",
        "tags": ["example", "demo"],
    },
}

# 声明此插件提供的扩展
EXTENSIONS = ["channels:example"]


# ==================== 渠道适配器实现 ====================

async def get_example_payload(request, engine, provider, api_key=None):
    """
    构建请求 payload
    
    这是 RequestAdapter 的实现，用于将统一请求格式转换为目标 API 的格式。
    
    Args:
        request: 请求对象 (RequestModel)
        engine: 引擎类型
        provider: 渠道提供者配置
        api_key: API 密钥（可选）
        
    Returns:
        tuple: (url, headers, payload)
    """
    # 从 provider 配置中获取基础 URL
    url = provider.get('base_url', 'https://api.example.com/v1/chat/completions')
    
    # 构建请求头
    headers = {
        'Content-Type': 'application/json',
    }
    if api_key:
        headers['Authorization'] = f"Bearer {api_key}"
    
    # 获取模型映射
    model_dict = provider.get('_model_dict_cache', {})
    original_model = model_dict.get(request.model, request.model)
    
    # 构建消息列表
    messages = []
    for msg in request.messages:
        if isinstance(msg.content, list):
            # 处理多模态消息
            content = []
            for item in msg.content:
                if item.type == "text":
                    content.append({"type": "text", "text": item.text})
                elif item.type == "image_url":
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": item.image_url.url}
                    })
            messages.append({"role": msg.role, "content": content})
        else:
            messages.append({"role": msg.role, "content": msg.content})
    
    # 构建 payload
    payload = {
        "model": original_model,
        "messages": messages,
    }
    
    # 添加可选参数
    if request.temperature is not None:
        payload["temperature"] = request.temperature
    if request.max_tokens is not None:
        payload["max_tokens"] = request.max_tokens
    if request.stream is not None:
        payload["stream"] = request.stream
    
    return url, headers, payload


async def fetch_example_response_stream(client, url, headers, payload, engine, model, timeout):
    """
    处理流式响应
    
    这是 StreamAdapter 的实现，用于处理流式 SSE 响应。
    
    Args:
        client: httpx.AsyncClient 实例
        url: 请求 URL
        headers: 请求头
        payload: 请求体
        engine: 引擎类型
        model: 模型名称
        timeout: 超时时间
        
    Yields:
        str: SSE 格式的响应数据
    """
    timestamp = int(datetime.timestamp(datetime.now()))
    
    # 发送请求
    json_payload = await asyncio.to_thread(json.dumps, payload)
    async with client.stream('POST', url, headers=headers, content=json_payload, timeout=timeout) as response:
        # 检查响应状态
        if response.status_code != 200:
            error_text = await response.aread()
            yield f'data: {{"error": "API error: {response.status_code}", "details": {json.dumps(error_text.decode())}}}\n\n'
            return
        
        # 处理 SSE 流
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                data = line[6:]
                if data == "[DONE]":
                    break
                
                try:
                    chunk = json.loads(data)
                    # 转换为统一格式（如果需要）
                    yield f"data: {json.dumps(chunk)}\n\n"
                except json.JSONDecodeError:
                    continue
    
    yield "data: [DONE]\n\n"


async def fetch_example_response(client, url, headers, payload, engine, model, timeout):
    """
    处理非流式响应
    
    这是 ResponseAdapter 的实现，用于处理普通 JSON 响应。
    
    Args:
        client: httpx.AsyncClient 实例
        url: 请求 URL
        headers: 请求头
        payload: 请求体
        engine: 引擎类型
        model: 模型名称
        timeout: 超时时间
        
    Yields:
        str: JSON 格式的响应数据
    """
    # 确保不是流式请求
    payload["stream"] = False
    
    json_payload = await asyncio.to_thread(json.dumps, payload)
    response = await client.post(url, headers=headers, content=json_payload, timeout=timeout)
    
    if response.status_code != 200:
        error_data = {
            "error": {
                "message": f"API error: {response.status_code}",
                "type": "api_error",
                "code": response.status_code
            }
        }
        yield f"data: {json.dumps(error_data)}\n\n"
        return
    
    result = response.json()
    yield f"data: {json.dumps(result)}\n\n"


# ==================== 渠道定义 ====================

class ExampleChannelAdapter:
    """
    示例渠道适配器类
    
    将所有适配器函数封装在一个类中，便于管理。
    """
    
    id = "example"
    type_name = "example_provider"
    
    # 适配器函数
    request_adapter = staticmethod(get_example_payload)
    stream_adapter = staticmethod(fetch_example_response_stream)
    response_adapter = staticmethod(fetch_example_response)


# ==================== 插件生命周期函数 ====================

def setup(manager: "PluginManager"):
    """
    插件初始化
    
    当插件被加载时调用。在这里注册扩展到插件系统。
    
    Args:
        manager: 插件管理器实例
    """
    # 方式 1: 通过插件系统注册扩展
    manager.register_extension(
        extension_point="channels",
        extension_id="example",
        implementation=ExampleChannelAdapter,
        priority=100,
        metadata={
            "description": "示例渠道适配器",
            "supported_features": ["chat", "stream"],
        },
        plugin_name=PLUGIN_INFO["name"],
    )
    
    # 方式 2: 同时注册到渠道注册表（保持向后兼容）
    from core.channels.registry import register_channel
    
    try:
        register_channel(
            id=ExampleChannelAdapter.id,
            type_name=ExampleChannelAdapter.type_name,
            request_adapter=ExampleChannelAdapter.request_adapter,
            stream_adapter=ExampleChannelAdapter.stream_adapter,
            response_adapter=ExampleChannelAdapter.response_adapter,
        )
        print(f"[{PLUGIN_INFO['name']}] Channel 'example' registered successfully!")
    except ValueError as e:
        # 渠道已存在，可能是重载
        print(f"[{PLUGIN_INFO['name']}] Channel registration skipped: {e}")


def teardown(manager: "PluginManager"):
    """
    插件清理
    
    当插件被卸载时调用。在这里清理资源和注销扩展。
    
    Args:
        manager: 插件管理器实例
    """
    # 注销扩展
    manager.unregister_extension("channels", "example")
    
    # 同时从渠道注册表注销
    from core.channels.registry import unregister_channel
    unregister_channel("example")
    
    print(f"[{PLUGIN_INFO['name']}] Channel 'example' unregistered!")


def unload():
    """
    插件卸载回调（可选）
    
    当插件模块被从内存中移除前调用。
    用于清理全局状态或关闭连接等。
    """
    print(f"[{PLUGIN_INFO['name']}] Plugin unloading...")


# ==================== 直接运行时显示插件信息 ====================

if __name__ == "__main__":
    print(f"Plugin: {PLUGIN_INFO['name']} v{PLUGIN_INFO['version']}")
    print(f"Description: {PLUGIN_INFO['description']}")
    print(f"Author: {PLUGIN_INFO['author']}")
    print(f"Extensions: {EXTENSIONS}")