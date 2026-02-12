# 插件开发指南

本文档介绍如何为 Zoaholic 开发和使用插件。

## 插件系统概述

Zoaholic 的插件系统基于**扩展点（Extension Point）**机制，支持：

- **渠道适配器**（channels）：添加新的 API 渠道
- **拦截器**（interceptors）：拦截和处理请求/响应
- **中间件**（middlewares）：处理请求和响应
- **处理器**（processors）：自定义数据处理
- **格式转换器**（formatters）：格式转换
- **验证器**（validators）：数据验证

## 快速开始

### 1. 创建插件文件

在 `plugins/` 目录下创建 Python 文件：

```python
# plugins/my_plugin.py

PLUGIN_INFO = {
    "name": "my_plugin",
    "version": "1.0.0",
    "description": "我的第一个插件",
    "author": "Your Name",
}

def setup(manager):
    """插件初始化"""
    print(f"[{PLUGIN_INFO['name']}] 插件已加载!")

def teardown(manager):
    """插件清理"""
    print(f"[{PLUGIN_INFO['name']}] 插件已卸载!")
```

### 2. 加载插件

插件会在应用启动时自动加载，也可以手动加载：

```python
from core.plugins import get_plugin_manager

manager = get_plugin_manager()

# 加载所有插件
manager.load_all()

# 或加载单个插件
manager.load_plugin("plugins/my_plugin.py")
```

## 插件结构

### PLUGIN_INFO（推荐）

定义插件元信息：

```python
PLUGIN_INFO = {
    "name": "plugin_name",           # 插件名称（必需）
    "version": "1.0.0",              # 版本号
    "description": "插件描述",        # 描述
    "author": "作者",                 # 作者
    "dependencies": ["other_plugin"], # 依赖的其他插件
    "metadata": {                     # 自定义元数据
        "category": "channel",
        "tags": ["example"],
    },
}
```

### EXTENSIONS（可选）

声明插件提供的扩展：

```python
EXTENSIONS = [
    "channels:my_channel",      # 渠道扩展
    "middlewares:my_middleware", # 中间件扩展
]
```

### 生命周期函数

```python
def setup(manager):
    """
    插件初始化（推荐）
    
    当插件被加载并激活时调用。
    在这里注册扩展到插件系统。
    
    Args:
        manager: PluginManager 实例
    """
    pass

def teardown(manager):
    """
    插件清理（可选）
    
    当插件被卸载时调用。
    在这里清理资源和注销扩展。
    
    Args:
        manager: PluginManager 实例
    """
    pass

def unload():
    """
    插件卸载回调（可选）
    
    当插件模块被从内存中移除前调用。
    用于清理全局状态或关闭连接等。
    """
    pass
```

## 扩展点详解

### 渠道扩展（channels）

用于添加新的 API 渠道适配器：

```python
def setup(manager):
    manager.register_extension(
        extension_point="channels",
        extension_id="my_channel",
        implementation=MyChannelAdapter,
        priority=100,
        metadata={"description": "我的渠道"},
        plugin_name=PLUGIN_INFO["name"],
    )
```

渠道适配器需要实现以下接口：

```python
class MyChannelAdapter:
    id = "my_channel"
    type_name = "my_provider"
    
    @staticmethod
    async def request_adapter(request, engine, provider, api_key=None):
        """构建请求"""
        url = provider.get('base_url')
        headers = {'Authorization': f'Bearer {api_key}'}
        payload = {...}
        return url, headers, payload
    
    @staticmethod
    async def stream_adapter(client, url, headers, payload, engine, model, timeout):
        """处理流式响应"""
        async with client.stream('POST', url, ...) as response:
            async for line in response.aiter_lines():
                yield f"data: {line}\n\n"
        yield "data: [DONE]\n\n"
    
    @staticmethod
    async def response_adapter(client, url, headers, payload, engine, model, timeout):
        """处理非流式响应"""
        response = await client.post(url, ...)
        yield f"data: {response.json()}\n\n"
```

同时注册到渠道注册表（保持兼容性）：

```python
from core.channels.registry import register_channel

register_channel(
    id="my_channel",
    type_name="my_provider",
    request_adapter=MyChannelAdapter.request_adapter,
    stream_adapter=MyChannelAdapter.stream_adapter,
    response_adapter=MyChannelAdapter.response_adapter,
)
```

### 请求/响应拦截器（推荐）

这是最简单的插件扩展方式，允许在请求发送前和响应返回后进行拦截和处理：

```python
from core.plugins import (
    register_request_interceptor,
    unregister_request_interceptor,
    register_response_interceptor,
    unregister_response_interceptor,
)

# 请求拦截器：在请求发送到渠道前调用
async def my_request_interceptor(request, engine, provider, api_key, url, headers, payload):
    """
    拦截并修改请求参数
    
    Args:
        request: 原始请求对象
        engine: 引擎类型 (openai, gemini, claude, etc.)
        provider: 提供商配置
        api_key: API 密钥
        url: 请求 URL
        headers: 请求头
        payload: 请求体
        
    Returns:
        (url, headers, payload) 修改后的请求参数
    """
    # 添加自定义 header
    headers["X-Custom-Header"] = "value"
    
    # 修改 payload
    payload["custom_param"] = "value"
    
    return url, headers, payload

# 响应拦截器：在响应返回时调用
async def my_response_interceptor(response_chunk, engine, model, is_stream):
    """
    拦截并处理响应数据
    
    Args:
        response_chunk: 响应数据（流式时为单个 chunk，非流式时为完整响应）
        engine: 引擎类型
        model: 模型名称
        is_stream: 是否为流式响应
        
    Returns:
        修改后的响应数据
    """
    # 可以记录日志、修改响应等
    return response_chunk

def setup(manager):
    # 注册拦截器
    register_request_interceptor(
        interceptor_id="my_request_interceptor",
        callback=my_request_interceptor,
        priority=100,  # 数值越小越先执行
        plugin_name="my_plugin",
    )
    
    register_response_interceptor(
        interceptor_id="my_response_interceptor",
        callback=my_response_interceptor,
        priority=100,
        plugin_name="my_plugin",
    )

def teardown(manager):
    # 注销拦截器
    unregister_request_interceptor("my_request_interceptor")
    unregister_response_interceptor("my_response_interceptor")
```

#### 拦截器执行顺序

1. **请求拦截器**：在 `get_payload()` 构建完请求后、发送前调用
2. **响应拦截器**：在渠道返回响应后、返回给客户端前调用

拦截器按 `priority` 排序执行，数值越小越先执行。

#### 按渠道控制拦截器

拦截器支持按渠道启用/禁用。在渠道配置的 `preferences.enabled_plugins` 中指定要启用的插件列表：

```yaml
providers:
  - provider: my_provider
    base_url: https://api.example.com
    preferences:
      enabled_plugins:
        - claude_thinking
        - my_custom_plugin
```

只有在 `enabled_plugins` 列表中的插件的拦截器才会被执行。如果未配置 `enabled_plugins`，则所有启用的拦截器都会执行。

在前端渠道编辑界面中，可以通过"插件拦截器"部分配置要启用的插件。

#### 拦截器管理 API

```python
from core.plugins import get_interceptor_registry

registry = get_interceptor_registry()

# 获取所有拦截器
request_interceptors = registry.get_request_interceptors()
response_interceptors = registry.get_response_interceptors()

# 启用/禁用拦截器
registry.enable_request_interceptor("my_interceptor")
registry.disable_request_interceptor("my_interceptor")

# 按插件注销所有拦截器
registry.unregister_plugin_interceptors("my_plugin")

# 获取统计信息
stats = registry.get_stats()

# 获取所有注册了拦截器的插件列表
interceptor_plugins = registry.get_interceptor_plugins()
```

### 中间件扩展（middlewares）

用于处理请求和响应：

```python
class MyMiddleware:
    async def process_request(self, request):
        """处理请求"""
        # 修改或验证请求
        return request
    
    async def process_response(self, response):
        """处理响应"""
        # 修改或处理响应
        return response
    
    async def on_error(self, error):
        """错误处理"""
        pass

def setup(manager):
    manager.register_extension(
        extension_point="middlewares",
        extension_id="my_middleware",
        implementation=MyMiddleware(),
        priority=50,  # 优先级越小越先执行
    )
```

### 生命周期钩子扩展（hooks）

用于监听应用生命周期事件（注意：这与请求/响应拦截器不同）：

```python
class MyLifecycleHooks:
    async def on_startup(self):
        """应用启动时"""
        pass
    
    async def on_shutdown(self):
        """应用关闭时"""
        pass
    
    async def before_request(self, request):
        """请求处理前"""
        pass
    
    async def after_request(self, request, response):
        """请求处理后"""
        pass
    
    async def on_error(self, error):
        """发生错误时"""
        pass

def setup(manager):
    manager.register_extension(
        extension_point="hooks",
        extension_id="my_lifecycle_hooks",
        implementation=MyLifecycleHooks(),
    )
```

## 插件管理器 API

### 加载插件

```python
from core.plugins import get_plugin_manager

manager = get_plugin_manager()

# 加载所有插件
result = manager.load_all()

# 加载单个插件（文件路径）
info = manager.load_plugin("plugins/my_plugin.py")

# 加载单个插件（模块路径）
info = manager.load_plugin("my_package.my_plugin")
```

### 卸载和重载

```python
# 卸载插件
manager.unload_plugin("my_plugin")

# 重载插件（热更新）
manager.reload_plugin("my_plugin")
```

### 扩展管理

```python
# 获取扩展
extensions = manager.get_extensions("channels")

# 获取实现
implementations = manager.get_implementations("channels")

# 启用/禁用扩展
manager.enable_extension("channels", "my_channel")
manager.disable_extension("channels", "my_channel")

# 注销扩展
manager.unregister_extension("channels", "my_channel")
```

### 状态查询

```python
# 获取系统状态
status = manager.get_status()
print(status)

# 获取所有插件
plugins = manager.plugins

# 获取扩展点列表
extension_points = manager.list_extension_points()
```

## 通过 pip 安装的插件

插件也可以打包为 Python 包并通过 pip 安装。

### 创建插件包

```
my_zoaholic_plugin/
├── pyproject.toml
├── my_plugin/
│   ├── __init__.py
│   └── channel.py
```

### pyproject.toml

```toml
[project]
name = "my-zoaholic-plugin"
version = "1.0.0"

[project.entry-points."zoaholic.plugins"]
my_plugin = "my_plugin"
```

### 安装和使用

```bash
pip install my-zoaholic-plugin
```

插件会在启动时自动通过 entry points 加载。

## 自定义扩展点

您可以创建自己的扩展点：

```python
from core.plugins import ExtensionPoint, ExtensionPointType

def setup(manager):
    # 定义新的扩展点
    my_extension_point = ExtensionPoint(
        name="my_custom_point",
        type=ExtensionPointType.CUSTOM,
        description="我的自定义扩展点",
        required_methods=["process"],
        optional_methods=["validate"],
        singleton=False,
        priority_support=True,
    )
    
    # 注册扩展点
    manager.register_extension_point(my_extension_point)
```

## 最佳实践

### 1. 错误处理

```python
def setup(manager):
    try:
        # 注册扩展
        manager.register_extension(...)
    except Exception as e:
        print(f"[{PLUGIN_INFO['name']}] 初始化失败: {e}")
        raise
```

### 2. 资源清理

```python
_resources = []

def setup(manager):
    # 初始化资源
    _resources.append(create_resource())

def teardown(manager):
    # 清理资源
    for resource in _resources:
        resource.close()
    _resources.clear()
```

### 3. 配置管理

```python
PLUGIN_CONFIG = {
    "api_url": "https://api.example.com",
    "timeout": 30,
}

def setup(manager):
    # 从环境变量覆盖配置
    import os
    if url := os.getenv("MY_PLUGIN_API_URL"):
        PLUGIN_CONFIG["api_url"] = url
```

### 4. 日志记录

```python
from core.log_config import logger

def setup(manager):
    logger.info(f"[{PLUGIN_INFO['name']}] 正在初始化...")
```

## 完整示例

查看 `plugins/example_channel.py` 获取完整的渠道插件示例。

## 常见问题

### Q: 插件加载顺序？

A: 插件按文件名字母顺序加载。如需控制顺序，可以使用数字前缀命名，如 `01_first_plugin.py`。

### Q: 如何处理插件依赖？

A: 在 `PLUGIN_INFO["dependencies"]` 中声明依赖，系统会检查依赖是否满足。

### Q: 插件可以访问应用状态吗？

A: 可以，通过 `manager` 参数可以访问插件系统，但建议避免直接修改应用状态。

### Q: 如何调试插件？

A: 设置环境变量 `DEBUG=True`，查看详细日志输出。