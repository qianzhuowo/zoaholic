"""
Zoaholic Plugins

将自定义插件放在此目录下，系统会自动加载。

插件规范：
1. 必须定义 PLUGIN_INFO 字典提供插件元信息（推荐）
2. 可选实现 setup(manager) 函数用于初始化
3. 可选实现 teardown(manager) 函数用于清理
4. 可选实现 unload() 函数用于卸载

详见 docs/plugin-development.md
"""