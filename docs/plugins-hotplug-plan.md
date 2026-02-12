# 前端导入/下载 + 热插拔插件系统规划

> 本文档基于现有 `core/plugins/` 子系统（`extension.py`、`loader.py`、`manager.py`、`registry.py`）与 `docs/plugin-development.md` 进行扩展，规划"前端导入/下载 + 热插拔"能力。

---

## 1. 目标与范围

| 目标 | 说明 |
|------|------|
| **前端上传** | 管理员可在 Web UI 上传 `.zip` 或单文件 `.py` 插件包 |
| **远程下载** | 支持从 URL 或官方插件仓库拉取插件 |
| **热插拔** | 运行时加载/卸载/重载/启用/禁用，无需重启服务 |
| **安全隔离** | 路径沙箱、导入白名单、可选签名校验 |
| **版本兼容** | `engine_version` 约束，语义化版本检查 |
| **持久化** | 安装状态、启用状态、配置持久化到 `installed_plugins.json` |
| **回滚** | 安装/卸载失败时自动回滚 |

---

## 2. 插件包规范

### 2.1 目录结构

```
my_plugin/
├── plugin.json          # 必需：元数据
├── __init__.py          # 必需：入口模块
├── channel.py           # 可选：渠道实现
├── middleware.py        # 可选：中间件
├── assets/              # 可选：静态资源
└── requirements.txt     # 可选：依赖声明
```

打包为 `my_plugin.zip`（根目录为 `my_plugin/`）。

### 2.2 plugin.json 规范

```json
{
  "name": "my_plugin",
  "version": "1.0.0",
  "description": "示例插件",
  "author": "Your Name",
  "license": "MIT",
  "engine_version": ">=2.0.0,<3.0.0",
  "entry": "__init__.py",
  "extensions": [
    "channels:my_channel",
    "middlewares:my_middleware"
  ],
  "dependencies": [],
  "permissions": ["network", "filesystem:read"],
  "checksum": "sha256:abc123...",
  "signature": "base64..."
}
```

| 字段 | 必需 | 说明 |
|------|------|------|
| `name` | ✅ | 唯一标识，与目录名一致 |
| `version` | ✅ | 语义化版本 |
| `engine_version` | ✅ | 兼容的 Zoaholic 版本范围 |
| `entry` | ✅ | 入口模块相对路径 |
| `extensions` | ❌ | 声明提供的扩展点 |
| `dependencies` | ❌ | 依赖的其他插件 |
| `permissions` | ❌ | 权限声明（用于安全审计） |
| `checksum` | ❌ | 包完整性校验 |
| `signature` | ❌ | 可选签名（用于官方仓库） |

### 2.3 单文件插件

对于简单插件，可直接上传 `.py` 文件，系统自动生成默认 `plugin.json`。

---

## 3. 后端 REST API 设计

### 3.1 路由前缀

`/v1/plugins`（需 admin 权限）

### 3.2 接口列表

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/v1/plugins` | 列出所有已安装插件 |
| `GET` | `/v1/plugins/{name}` | 获取单个插件详情 |
| `POST` | `/v1/plugins/upload` | 上传 `.zip` 或 `.py` 文件 |
| `POST` | `/v1/plugins/install` | 从 URL 下载并安装 |
| `POST` | `/v1/plugins/{name}/enable` | 启用插件 |
| `POST` | `/v1/plugins/{name}/disable` | 禁用插件 |
| `POST` | `/v1/plugins/{name}/reload` | 热重载插件 |
| `DELETE` | `/v1/plugins/{name}` | 卸载并删除插件 |
| `GET` | `/v1/plugins/status` | 插件系统状态 |
| `GET` | `/v1/plugins/extensions` | 列出所有扩展 |
| `GET` | `/v1/plugins/extension-points` | 列出所有扩展点 |
| `GET` | `/v1/plugins/repository` | 远程仓库索引（可选） |

### 3.3 请求/响应示例

#### 上传插件

```http
POST /v1/plugins/upload
Content-Type: multipart/form-data
Authorization: Bearer sk-admin-xxx

file: my_plugin.zip
```

响应：

```json
{
  "success": true,
  "plugin": {
    "name": "my_plugin",
    "version": "1.0.0",
    "enabled": true,
    "extensions": ["channels:my_channel"]
  }
}
```

#### 从 URL 安装

```http
POST /v1/plugins/install
Content-Type: application/json
Authorization: Bearer sk-admin-xxx

{
  "url": "https://example.com/plugins/my_plugin.zip",
  "checksum": "sha256:abc123..."
}
```

---

## 4. 存储与安装流程

### 4.1 目录布局

```
plugins/
├── __init__.py
├── installed_plugins.json   # 安装状态持久化
├── my_plugin/               # 已安装插件目录
│   ├── plugin.json
│   └── __init__.py
└── .staging/                # 临时解压目录（安装中）
```

### 4.2 安装流程

```
┌─────────────────────────────────────────────────────────────┐
│                      安装流程                                │
├─────────────────────────────────────────────────────────────┤
│ 1. 接收文件/URL                                              │
│ 2. 保存到 .staging/{uuid}/                                   │
│ 3. 解压（如为 zip）                                          │
│ 4. 校验 plugin.json 存在性与格式                             │
│ 5. 校验 engine_version 兼容性                                │
│ 6. 校验 checksum（如提供）                                   │
│ 7. 校验签名（如启用）                                        │
│ 8. 检测冲突（同名插件已存在）                                │
│ 9. 移动到 plugins/{name}/                                    │
│ 10. 调用 PluginLoader.load_from_file()                       │
│ 11. 调用 PluginManager._activate_plugin()                    │
│ 12. 更新 installed_plugins.json                              │
│ 13. 清理 .staging/                                           │
└─────────────────────────────────────────────────────────────┘
```

### 4.3 回滚策略

- 任何步骤失败，删除 `.staging/` 临时目录
- 如已移动到 `plugins/{name}/`，则删除该目录
- 如已加载模块，调用 `unload_plugin()` 清理

---

## 5. 热插拔集成

### 5.1 现有能力

| 方法 | 位置 | 说明 |
|------|------|------|
| `load_from_file()` | `PluginLoader` | 从文件加载模块 |
| `unload_plugin()` | `PluginLoader` | 卸载模块并清理 `sys.modules` |
| `reload_plugin()` | `PluginLoader` | 卸载后重新加载 |
| `_activate_plugin()` | `PluginManager` | 调用 `setup(manager)` |
| `_deactivate_plugin()` | `PluginManager` | 调用 `teardown(manager)` 并注销扩展 |

### 5.2 扩展点

新增 `PluginService` 类（`core/plugins/service.py`）封装安装/卸载/启用/禁用逻辑：

```python
class PluginService:
    def __init__(self, manager: PluginManager):
        self.manager = manager
        self.install_dir = Path("plugins")
        self.staging_dir = self.install_dir / ".staging"
        self.index_file = self.install_dir / "installed_plugins.json"

    async def install_from_file(self, file: UploadFile) -> PluginInfo: ...
    async def install_from_url(self, url: str, checksum: str = None) -> PluginInfo: ...
    async def uninstall(self, name: str) -> bool: ...
    async def enable(self, name: str) -> bool: ...
    async def disable(self, name: str) -> bool: ...
    async def reload(self, name: str) -> PluginInfo: ...
    def list_installed(self) -> List[dict]: ...
    def get_status(self) -> dict: ...
```

---

## 6. 前端管理页面

### 6.1 入口

在 `AppConfig.navItems` 中添加：

```javascript
{ id: "plugins", label: "插件管理", icon: "extension" }
```

或复用 `tools` 视图，新增 Tab。

### 6.2 页面结构

```
┌─────────────────────────────────────────────────────────────┐
│ 插件管理                                    [上传] [刷新]    │
├─────────────────────────────────────────────────────────────┤
│ [已安装] [仓库]                                              │
├─────────────────────────────────────────────────────────────┤
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ my_plugin v1.0.0                          [启用] [卸载] │ │
│ │ 示例插件 - 作者: Your Name                              │ │
│ │ 扩展: channels:my_channel                               │ │
│ └─────────────────────────────────────────────────────────┘ │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ another_plugin v2.1.0                     [禁用] [重载] │ │
│ │ ...                                                     │ │
│ └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 6.3 交互

| 操作 | 说明 |
|------|------|
| 上传 | 弹出文件选择器，支持 `.zip` / `.py` |
| 从 URL 安装 | 弹出对话框输入 URL |
| 启用/禁用 | 调用 API，刷新列表 |
| 重载 | 调用 API，显示结果 |
| 卸载 | 确认对话框，调用 API |
| 详情 | 展开显示扩展、依赖、权限等 |

### 6.4 复用组件

- `UI.card()`、`UI.btn()`、`UI.iconBtn()`、`UI.snackbar()`
- `UI.dialog()` 用于确认/输入
- `UI.sideSheet()` 用于详情展示
- 文件上传参考 `ToolsView` 中的拖拽区域

---

## 7. 远程仓库（可选）

### 7.1 索引格式

```json
{
  "version": 1,
  "plugins": [
    {
      "name": "official_channel",
      "version": "1.2.0",
      "description": "官方渠道适配器",
      "download_url": "https://repo.zoaholic.io/plugins/official_channel-1.2.0.zip",
      "checksum": "sha256:...",
      "signature": "..."
    }
  ]
}
```

### 7.2 流程

1. 前端请求 `/v1/plugins/repository`
2. 后端拉取远程索引并缓存
3. 前端展示可安装列表
4. 用户点击安装，调用 `/v1/plugins/install`

---

## 8. 安全与隔离

### 8.1 路径沙箱

- 所有插件文件必须位于 `plugins/` 目录下
- 禁止 `..` 路径穿越
- 解压时校验文件路径

### 8.2 导入限制

- 可选：使用 `importlib` 钩子限制可导入模块
- 可选：禁止 `os.system`、`subprocess` 等危险调用

### 8.3 权限声明

- `plugin.json` 中声明 `permissions`
- 安装时提示用户审核
- 运行时可根据权限限制行为

### 8.4 签名校验

- 官方仓库插件必须签名
- 本地上传可选校验
- 使用 RSA/Ed25519 签名

---

## 9. 版本兼容策略

### 9.1 engine_version

- 使用 PEP 440 版本规范
- 安装时校验当前 Zoaholic 版本是否满足
- 不满足则拒绝安装并提示

### 9.2 插件升级

- 检测同名插件已存在
- 比较版本号，提示升级/降级
- 升级时先卸载旧版本，再安装新版本

---

## 10. 运行时状态与日志

### 10.1 状态查询

`GET /v1/plugins/status` 返回：

```json
{
  "initialized": true,
  "total_plugins": 5,
  "enabled_plugins": 4,
  "extension_points": 6,
  "total_extensions": 12
}
```

### 10.2 操作日志

- 所有安装/卸载/启用/禁用操作记录到日志
- 可选：写入 `plugin_audit.log`

### 10.3 实时通知（可选）

- 使用 SSE 或 WebSocket 推送操作结果
- 前端订阅并显示 Snackbar

---

## 11. 实现步骤

| 阶段 | 任务 | 优先级 |
|------|------|--------|
| **Phase 1** | 后端 API 骨架 + PluginService | P0 |
| **Phase 2** | 上传/安装/卸载流程 | P0 |
| **Phase 3** | 前端管理页面 | P0 |
| **Phase 4** | 热重载/启用/禁用 | P1 |
| **Phase 5** | 远程仓库支持 | P2 |
| **Phase 6** | 签名校验 | P2 |
| **Phase 7** | 权限沙箱 | P3 |
| **Phase 8** | 自动化测试 | P1 |

---

## 12. 文件清单

### 新增文件

| 文件 | 说明 |
|------|------|
| `core/plugins/service.py` | 安装/卸载/启用/禁用服务 |
| `routes/plugins.py` | 插件管理 API 路由 |
| `static/js/views/plugins.js` | 前端插件管理视图 |
| `plugins/installed_plugins.json` | 安装状态持久化 |

### 修改文件

| 文件 | 修改内容 |
|------|----------|
| `main.py` | 注册 `/v1/plugins` 路由 |
| `static/js/config.js` | 添加 `plugins` 导航项 |
| `static/js/views/index.js` | 注册 `PluginsView` |
| `core/plugins/__init__.py` | 导出 `PluginService` |

---

## 13. 示例代码骨架

### routes/plugins.py

```python
from fastapi import APIRouter, Depends, UploadFile, File, Body
from fastapi.responses import JSONResponse
from routes.deps import verify_admin_api_key
from core.plugins.service import PluginService, get_plugin_service

router = APIRouter(prefix="/v1/plugins", tags=["plugins"])

@router.get("")
async def list_plugins(service: PluginService = Depends(get_plugin_service)):
    return JSONResponse(content={"plugins": service.list_installed()})

@router.post("/upload")
async def upload_plugin(
    file: UploadFile = File(...),
    _: int = Depends(verify_admin_api_key),
    service: PluginService = Depends(get_plugin_service),
):
    info = await service.install_from_file(file)
    return JSONResponse(content={"success": True, "plugin": info})

@router.post("/install")
async def install_from_url(
    url: str = Body(..., embed=True),
    checksum: str = Body(None, embed=True),
    _: int = Depends(verify_admin_api_key),
    service: PluginService = Depends(get_plugin_service),
):
    info = await service.install_from_url(url, checksum)
    return JSONResponse(content={"success": True, "plugin": info})

@router.post("/{name}/enable")
async def enable_plugin(name: str, ...): ...

@router.post("/{name}/disable")
async def disable_plugin(name: str, ...): ...

@router.post("/{name}/reload")
async def reload_plugin(name: str, ...): ...

@router.delete("/{name}")
async def uninstall_plugin(name: str, ...): ...

@router.get("/status")
async def plugin_status(...): ...
```

### static/js/views/plugins.js

```javascript
const PluginsView = {
    render(container) {
        const header = UI.el("div", "flex justify-between items-center mb-6");
        header.appendChild(UI.el("h2", "text-display-small text-md-on-surface", "插件管理"));
        const actions = UI.el("div", "flex gap-2");
        actions.appendChild(UI.btn("上传插件", () => PluginsView._openUploadDialog(), "filled", "upload"));
        actions.appendChild(UI.iconBtn("refresh", () => PluginsView._refresh(container), "standard"));
        header.appendChild(actions);
        container.appendChild(header);

        const list = UI.el("div", "flex flex-col gap-4");
        list.id = "plugins-list";
        container.appendChild(list);

        PluginsView._loadPlugins(list);
    },

    async _loadPlugins(listEl) { /* fetch /v1/plugins */ },
    _openUploadDialog() { /* UI.dialog with file input */ },
    _refresh(container) { /* re-render */ },
};
```

---

## 14. 参考资料

- [docs/plugin-development.md](plugin-development.md)
- [core/plugins/loader.py](../core/plugins/loader.py)
- [core/plugins/manager.py](../core/plugins/manager.py)
- [core/plugins/registry.py](../core/plugins/registry.py)
- [core/plugins/extension.py](../core/plugins/extension.py)