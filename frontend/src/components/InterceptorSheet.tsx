import { useEffect, useMemo, useState } from 'react';
import * as Dialog from '@radix-ui/react-dialog';
import { 
  Puzzle, 
  Settings2, 
  ChevronDown, 
  ChevronRight, 
  Check, 
  X,
} from 'lucide-react';

interface PluginOption {
  plugin_name: string;
  version: string;
  description: string;
  enabled: boolean;
  request_interceptors: any[];
  response_interceptors: any[];
  metadata?: {
    params_hint?: string;
    provider_config?: {
      key: string;
      type?: 'json' | 'text';
      title?: string;
      description?: string;
      example?: any;
    };
  };
}

interface InterceptorSheetProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  allPlugins: PluginOption[];
  enabledPlugins: string[]; // ["pluginA:config", "pluginB"]
  providerPreferences: Record<string, any>;
  availableModels?: string[];
  onUpdate: (payload: { enabled_plugins: string[]; preferences_patch: Record<string, any>; preferences_delete: string[] }) => void;
}

const COMMON_PAYLOAD_FIELDS = [
  'thinking',
  'min_p',
  'top_k',
  'seed',
  'response_format',
  'stream_options.include_usage',
  'parallel_tool_calls',
  'max_completion_tokens',
  'reasoning',
  'tools',
  'tool_choice',
];

type FilterMode = 'deny' | 'allow';

function normalizeStringList(value: any): string[] {
  if (!value) return [];
  const arr = Array.isArray(value) ? value : [value];
  return Array.from(new Set(arr.map(v => String(v).trim()).filter(Boolean)));
}

function parsePostBodyParameterFilter(text: string): {
  mode: FilterMode;
  use_defaults: boolean;
  globalFields: string[];
  rules: Array<{ model: string; modeOverride: '' | FilterMode; fields: string[] }>;
} {
  const defaults = { mode: 'deny' as FilterMode, use_defaults: true, globalFields: [] as string[], rules: [] as Array<{ model: string; modeOverride: '' | FilterMode; fields: string[] }> };

  const t = (text || '').trim();
  if (!t) return defaults;

  const obj = JSON.parse(t);

  // list => deny list
  if (Array.isArray(obj)) {
    return { ...defaults, globalFields: normalizeStringList(obj) };
  }

  if (obj && typeof obj === 'object') {
    const hasDirect = ['deny', 'allow', 'mode', 'enabled', 'use_defaults'].some(k => Object.prototype.hasOwnProperty.call(obj, k));
    if (hasDirect) {
      const mode = (String((obj as any).mode || 'deny').toLowerCase() === 'allow' ? 'allow' : 'deny') as FilterMode;
      const use_defaults = (obj as any).use_defaults !== undefined ? Boolean((obj as any).use_defaults) : true;
      const globalFields = normalizeStringList(mode === 'allow' ? (obj as any).allow : (obj as any).deny);
      return { mode, use_defaults, globalFields, rules: [] };
    }

    // { all: {...}, modelA: {...} }
    const globalObj = (obj as any).all ?? (obj as any)['*'] ?? {};
    const mode = (String(globalObj?.mode || 'deny').toLowerCase() === 'allow' ? 'allow' : 'deny') as FilterMode;
    const use_defaults = globalObj?.use_defaults !== undefined ? Boolean(globalObj.use_defaults) : true;
    const globalFields = normalizeStringList(mode === 'allow' ? globalObj?.allow : globalObj?.deny);

    const rules: Array<{ model: string; modeOverride: '' | FilterMode; fields: string[] }> = [];
    Object.keys(obj as any).forEach((k) => {
      if (k === 'all' || k === '*') return;
      const r = (obj as any)[k];
      if (!r || typeof r !== 'object') return;
      const rMode = r.mode ? ((String(r.mode).toLowerCase() === 'allow' ? 'allow' : 'deny') as FilterMode) : '';
      const effectiveMode = (rMode || mode) as FilterMode;
      const fields = normalizeStringList(effectiveMode === 'allow' ? (r as any).allow : (r as any).deny);
      if (fields.length === 0 && !rMode) return;
      rules.push({ model: k, modeOverride: rMode, fields });
    });

    return { mode, use_defaults, globalFields, rules };
  }

  return defaults;
}

function buildPostBodyParameterFilterConfig(state: {
  mode: FilterMode;
  use_defaults: boolean;
  globalFields: string[];
  rules: Array<{ model: string; modeOverride: '' | FilterMode; fields: string[] }>;
}): any {
  const all: any = {
    mode: state.mode,
    use_defaults: state.use_defaults,
  };
  if (state.mode === 'allow') all.allow = state.globalFields;
  else all.deny = state.globalFields;

  const cfg: any = { all };

  for (const rule of state.rules) {
    const model = String(rule.model || '').trim();
    if (!model) continue;
    const ruleMode = (rule.modeOverride || state.mode) as FilterMode;
    const obj: any = {};
    if (rule.modeOverride) obj.mode = rule.modeOverride;
    if (ruleMode === 'allow') obj.allow = rule.fields;
    else obj.deny = rule.fields;
    cfg[model] = obj;
  }

  return cfg;
}

function PostBodyParameterFilterEditor(props: {
  valueText: string;
  onChangeText: (text: string) => void;
  availableModels: string[];
}) {
  const { valueText, onChangeText, availableModels } = props;

  const { parsed, parseError } = useMemo(() => {
    try {
      return {
        parsed: parsePostBodyParameterFilter(valueText),
        parseError: null as string | null,
      };
    } catch (e: any) {
      return {
        parsed: {
          mode: 'deny' as FilterMode,
          use_defaults: true,
          globalFields: [],
          rules: [],
        },
        parseError: e?.message || 'invalid json',
      };
    }
  }, [valueText]);

  const update = (patch: Partial<typeof parsed>) => {
    const next = { ...parsed, ...patch };
    const cfg = buildPostBodyParameterFilterConfig(next);
    onChangeText(JSON.stringify(cfg, null, 2));
  };

  const addGlobalField = (field: string) => {
    const f = String(field || '').trim();
    if (!f) return;
    update({ globalFields: Array.from(new Set([...parsed.globalFields, f])) });
  };

  const removeGlobalField = (field: string) => {
    update({ globalFields: parsed.globalFields.filter(x => x !== field) });
  };

  const addRule = () => {
    update({ rules: [...parsed.rules, { model: '', modeOverride: '', fields: [] }] });
  };

  const updateRule = (idx: number, patch: Partial<(typeof parsed.rules)[number]>) => {
    const next = [...parsed.rules];
    next[idx] = { ...next[idx], ...patch };
    update({ rules: next });
  };

  const removeRule = (idx: number) => {
    const next = parsed.rules.filter((_, i) => i !== idx);
    update({ rules: next });
  };

  const addRuleField = (idx: number, field: string) => {
    const f = String(field || '').trim();
    if (!f) return;
    const next = [...parsed.rules];
    next[idx] = { ...next[idx], fields: Array.from(new Set([...(next[idx].fields || []), f])) };
    update({ rules: next });
  };

  const removeRuleField = (idx: number, field: string) => {
    const next = [...parsed.rules];
    next[idx] = { ...next[idx], fields: (next[idx].fields || []).filter(x => x !== field) };
    update({ rules: next });
  };

  return (
    <div className="space-y-4">
      {parseError && (
        <div className="text-xs text-red-600 dark:text-red-400">
          当前 JSON 解析失败：{parseError}（将基于默认值继续编辑）
        </div>
      )}

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        <div className="space-y-1">
          <label className="text-xs font-medium text-muted-foreground">模式</label>
          <select
            value={parsed.mode}
            onChange={(e) => update({ mode: (e.target.value === 'allow' ? 'allow' : 'deny') as FilterMode })}
            className="w-full bg-background border border-border text-foreground focus:border-emerald-500 px-3 py-2 rounded-md text-sm"
          >
            <option value="deny">deny（移除字段）</option>
            <option value="allow">allow（仅保留字段）</option>
          </select>
        </div>

        <div className="space-y-1">
          <label className="text-xs font-medium text-muted-foreground">内置默认过滤</label>
          <div className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={parsed.use_defaults}
              onChange={(e) => update({ use_defaults: e.target.checked })}
              className="w-4 h-4"
            />
            <span className="text-xs text-muted-foreground">叠加 Zoaholic 内置的兼容性 deny 列表</span>
          </div>
        </div>
      </div>

      {/* Global */}
      <div className="space-y-2">
        <label className="text-xs font-medium text-muted-foreground">全局规则（all）</label>

        <div className="flex flex-wrap gap-2">
          {parsed.globalFields.length === 0 ? (
            <span className="text-xs text-muted-foreground italic">暂无字段</span>
          ) : (
            parsed.globalFields.map((f) => (
              <span key={f} className="text-xs bg-muted px-2 py-1 rounded flex items-center gap-1 font-mono">
                {f}
                <button onClick={() => removeGlobalField(f)} className="text-muted-foreground hover:text-red-500">
                  <X className="w-3 h-3" />
                </button>
              </span>
            ))
          )}
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
          <select
            onChange={(e) => {
              if (e.target.value) addGlobalField(e.target.value);
              e.currentTarget.value = '';
            }}
            className="w-full bg-background border border-border text-foreground focus:border-emerald-500 px-3 py-2 rounded-md text-sm font-mono"
            defaultValue=""
          >
            <option value="" disabled>+ 常用字段</option>
            {COMMON_PAYLOAD_FIELDS.map(f => (
              <option key={f} value={f}>{f}</option>
            ))}
          </select>

          <div className="flex gap-2">
            <input
              type="text"
              list="post_body_parameter_filter_common_fields"
              placeholder="自定义字段（可含 dot-path）"
              className="flex-1 bg-background border border-border text-foreground focus:border-emerald-500 px-3 py-2 rounded-md text-sm font-mono outline-none"
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  e.preventDefault();
                  addGlobalField((e.target as HTMLInputElement).value);
                  (e.target as HTMLInputElement).value = '';
                }
              }}
            />
            <button
              type="button"
              onClick={(e) => {
                const input = (e.currentTarget.parentElement?.querySelector('input') as HTMLInputElement | null);
                if (!input) return;
                addGlobalField(input.value);
                input.value = '';
              }}
              className="px-3 py-2 text-sm font-medium text-emerald-600 dark:text-emerald-500 bg-emerald-500/10 rounded-md"
            >
              添加
            </button>
          </div>
        </div>

        <datalist id="post_body_parameter_filter_common_fields">
          {COMMON_PAYLOAD_FIELDS.map(f => <option key={f} value={f} />)}
        </datalist>
      </div>

      {/* Per-model */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <label className="text-xs font-medium text-muted-foreground">按模型单独配置</label>
          <button
            type="button"
            onClick={addRule}
            className="text-xs font-medium text-emerald-600 dark:text-emerald-500 hover:text-emerald-500 px-2 py-1 bg-emerald-500/10 rounded"
          >
            + 新增规则
          </button>
        </div>

        {parsed.rules.length === 0 ? (
          <div className="text-xs text-muted-foreground italic">暂无模型规则</div>
        ) : (
          <div className="space-y-3">
            {parsed.rules.map((r, idx) => (
              <div key={idx} className="border border-border rounded-lg p-3 bg-background space-y-2">
                <div className="flex items-center gap-2">
                  <select
                    value={r.model}
                    onChange={(e) => updateRule(idx, { model: e.target.value })}
                    className="flex-1 bg-background border border-border text-foreground focus:border-emerald-500 px-3 py-2 rounded-md text-sm font-mono"
                  >
                    <option value="">选择模型...</option>
                    {availableModels.map(m => <option key={m} value={m}>{m}</option>)}
                  </select>

                  <select
                    value={r.modeOverride}
                    onChange={(e) => updateRule(idx, { modeOverride: (e.target.value as any) || '' })}
                    className="bg-background border border-border text-foreground focus:border-emerald-500 px-3 py-2 rounded-md text-sm"
                    title="可选：覆盖全局模式"
                  >
                    <option value="">继承全局</option>
                    <option value="deny">deny</option>
                    <option value="allow">allow</option>
                  </select>

                  <button
                    type="button"
                    onClick={() => removeRule(idx)}
                    className="text-red-600 dark:text-red-400 hover:text-red-500"
                    title="删除规则"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>

                <div className="flex flex-wrap gap-2">
                  {(r.fields || []).length === 0 ? (
                    <span className="text-xs text-muted-foreground italic">暂无字段</span>
                  ) : (
                    (r.fields || []).map((f) => (
                      <span key={f} className="text-xs bg-muted px-2 py-1 rounded flex items-center gap-1 font-mono">
                        {f}
                        <button onClick={() => removeRuleField(idx, f)} className="text-muted-foreground hover:text-red-500">
                          <X className="w-3 h-3" />
                        </button>
                      </span>
                    ))
                  )}
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                  <select
                    onChange={(e) => {
                      if (e.target.value) addRuleField(idx, e.target.value);
                      e.currentTarget.value = '';
                    }}
                    className="w-full bg-background border border-border text-foreground focus:border-emerald-500 px-3 py-2 rounded-md text-sm font-mono"
                    defaultValue=""
                  >
                    <option value="" disabled>+ 常用字段</option>
                    {COMMON_PAYLOAD_FIELDS.map(f => (
                      <option key={f} value={f}>{f}</option>
                    ))}
                  </select>

                  <div className="flex gap-2">
                    <input
                      type="text"
                      list="post_body_parameter_filter_common_fields"
                      placeholder="自定义字段"
                      className="flex-1 bg-background border border-border text-foreground focus:border-emerald-500 px-3 py-2 rounded-md text-sm font-mono outline-none"
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') {
                          e.preventDefault();
                          addRuleField(idx, (e.target as HTMLInputElement).value);
                          (e.target as HTMLInputElement).value = '';
                        }
                      }}
                    />
                    <button
                      type="button"
                      onClick={(e) => {
                        const input = (e.currentTarget.parentElement?.querySelector('input') as HTMLInputElement | null);
                        if (!input) return;
                        addRuleField(idx, input.value);
                        input.value = '';
                      }}
                      className="px-3 py-2 text-sm font-medium text-emerald-600 dark:text-emerald-500 bg-emerald-500/10 rounded-md"
                    >
                      添加
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export function InterceptorSheet({ open, onOpenChange, allPlugins, enabledPlugins, providerPreferences, availableModels, onUpdate }: InterceptorSheetProps) {
  // Parsing helpers
  const parseEntry = (entry: string) => {
    // 约定：enabled_plugins 的单条配置使用“第一个冒号”分隔 name 与 options。
    // 这样 options 中可以包含冒号（例如 URL、JSON 等）。
    const colonIdx = entry.indexOf(':');
    if (colonIdx === -1) return { name: entry.trim(), options: '' };
    return { 
      name: entry.substring(0, colonIdx).trim(), 
      options: entry.substring(colonIdx + 1).trim() 
    };
  };

  // State
  const [selected, setSelected] = useState<Map<string, string>>(new Map());
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [providerConfigText, setProviderConfigText] = useState<Map<string, string>>(new Map());

  // Re-init when opening (important: same sheet instance is reused across different providers)
  useEffect(() => {
    if (!open) return;

    const m = new Map<string, string>();
    enabledPlugins.forEach(entry => {
      const { name, options } = parseEntry(entry);
      if (name) m.set(name, options);
    });
    setSelected(m);
    setExpanded(new Set());

    const cfgMap = new Map<string, string>();
    allPlugins.forEach(p => {
      const meta = p.metadata?.provider_config;
      if (!meta?.key) return;

      const raw = (providerPreferences || {})[meta.key];
      if (raw === undefined || raw === null) {
        cfgMap.set(p.plugin_name, '');
      } else {
        try {
          cfgMap.set(p.plugin_name, JSON.stringify(raw, null, 2));
        } catch {
          cfgMap.set(p.plugin_name, String(raw));
        }
      }
    });
    setProviderConfigText(cfgMap);
  }, [open, enabledPlugins, allPlugins, providerPreferences]);

  // Handlers
  const toggleSelect = (pluginName: string) => {
    const isSelecting = !selected.has(pluginName);

    setSelected(prev => {
      const next = new Map(prev);
      const wasSelected = next.has(pluginName);
      if (wasSelected) next.delete(pluginName);
      else next.set(pluginName, '');
      return next;
    });

    // 对 post_body_parameter_filter：首次启用时自动填入一个最小默认配置，避免“启用但未生效”
    if (isSelecting && pluginName === 'post_body_parameter_filter') {
      setProviderConfigText(prev => {
        const next = new Map(prev);
        const cur = (next.get(pluginName) || '').trim();
        if (!cur) {
          const cfg = buildPostBodyParameterFilterConfig({
            mode: 'deny',
            use_defaults: true,
            globalFields: [],
            rules: [],
          });
          next.set(pluginName, JSON.stringify(cfg, null, 2));
        }
        return next;
      });
    }
  };

  const updateOptions = (pluginName: string, options: string) => {
    setSelected(prev => {
      const next = new Map(prev);
      if (next.has(pluginName)) next.set(pluginName, options);
      return next;
    });
  };

  const toggleExpand = (pluginName: string) => {
    setExpanded(prev => {
      const next = new Set(prev);
      if (next.has(pluginName)) next.delete(pluginName);
      else next.add(pluginName);
      return next;
    });
  };

  const selectAll = () => {
    const next = new Map(selected);
    allPlugins.forEach(p => {
      if (!next.has(p.plugin_name)) next.set(p.plugin_name, '');
    });
    setSelected(next);
  };

  const clearAll = () => {
    setSelected(new Map());
  };

  const updateProviderConfigText = (pluginName: string, text: string) => {
    setProviderConfigText(prev => {
      const next = new Map(prev);
      next.set(pluginName, text);
      return next;
    });
  };

  const formatJsonText = (text: string): string => {
    if (!text.trim()) return '';
    const obj = JSON.parse(text);
    return JSON.stringify(obj, null, 2);
  };

  const handleSave = () => {
    const result: string[] = [];
    selected.forEach((options, name) => {
      result.push(options ? `${name}:${options}` : name);
    });

    const preferences_patch: Record<string, any> = {};
    const preferences_delete: string[] = [];

    for (const plugin of allPlugins) {
      const meta = plugin.metadata?.provider_config;
      if (!meta?.key) continue;

      const text = providerConfigText.get(plugin.plugin_name) || '';
      const t = text.trim();

      // empty => delete config key
      if (!t) {
        preferences_delete.push(meta.key);
        continue;
      }

      const configType = meta.type || 'json';
      if (configType === 'json') {
        try {
          preferences_patch[meta.key] = JSON.parse(t);
        } catch (e: any) {
          alert(`插件 ${plugin.plugin_name} 配置 JSON 格式错误：${e?.message || 'invalid json'}`);
          return;
        }
      } else {
        preferences_patch[meta.key] = t;
      }
    }

    onUpdate({ enabled_plugins: result, preferences_patch, preferences_delete });
    onOpenChange(false);
  };

  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 bg-black/50 backdrop-blur-sm z-[60] animate-in fade-in" />
        <Dialog.Content className="fixed right-0 top-0 h-full w-[500px] bg-background border-l border-border shadow-2xl z-[70] flex flex-col animate-in slide-in-from-right duration-200">
          
          <div className="flex items-center justify-between px-6 py-4 border-b border-border bg-muted/30">
            <Dialog.Title className="text-lg font-semibold text-foreground flex items-center gap-2">
              <Puzzle className="w-5 h-5 text-emerald-500" />
              配置插件拦截器
            </Dialog.Title>
            <Dialog.Close className="text-muted-foreground hover:text-foreground p-1 rounded-full hover:bg-muted transition-colors">
              <X className="w-5 h-5" />
            </Dialog.Close>
          </div>

          <div className="flex-1 overflow-y-auto p-6 space-y-4">
            <p className="text-sm text-muted-foreground">
              勾选要在本渠道启用的插件拦截器。可为每个插件配置参数（格式：plugin:options）。
            </p>

            {/* Toolbar */}
            <div className="flex items-center justify-between p-3 bg-muted/40 border border-border rounded-lg">
              <span className="text-sm text-muted-foreground">
                共 {allPlugins.length} 个插件，已选 <span className="text-foreground font-medium">{selected.size}</span> 个
              </span>
              <div className="flex gap-2">
                <button onClick={selectAll} className="text-xs font-medium text-emerald-500 hover:text-emerald-400 px-2 py-1 bg-emerald-500/10 rounded">全选</button>
                <button onClick={clearAll} className="text-xs font-medium text-red-500 hover:text-red-400 px-2 py-1 bg-red-500/10 rounded">全不选</button>
              </div>
            </div>

            {/* Plugin List (Accordion) */}
            <div className="space-y-2.5">
              {allPlugins.map(plugin => {
                const isSelected = selected.has(plugin.plugin_name);
                const isExpanded = expanded.has(plugin.plugin_name);
                const options = selected.get(plugin.plugin_name) || '';

                return (
                  <div key={plugin.plugin_name} className={`border rounded-lg transition-colors ${isSelected ? 'border-emerald-500/30 bg-emerald-500/5' : 'border-border bg-card'}`}>
                    {/* Header */}
                    <div className="flex items-center justify-between p-3 cursor-pointer select-none" onClick={() => toggleExpand(plugin.plugin_name)}>
                      <div className="flex items-center gap-3">
                        <button 
                          onClick={(e) => { e.stopPropagation(); toggleSelect(plugin.plugin_name); }}
                          className={`w-5 h-5 rounded border flex items-center justify-center transition-colors ${isSelected ? 'bg-emerald-500 border-emerald-500 text-white' : 'bg-background border-border'}`}
                        >
                          {isSelected && <Check className="w-3.5 h-3.5" />}
                        </button>
                        <div>
                          <div className="flex items-center gap-2">
                            <span className={`text-sm font-medium ${isSelected ? 'text-foreground' : 'text-muted-foreground'}`}>{plugin.plugin_name}</span>
                            <span className="text-xs bg-muted text-muted-foreground px-1.5 py-0.5 rounded font-mono">v{plugin.version}</span>
        </div>
                          {options && <div className="text-xs text-muted-foreground font-mono mt-0.5 max-w-[200px] truncate">{options}</div>}
                        </div>
                      </div>
                      <div className="text-muted-foreground">{isExpanded ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}</div>
                    </div>

                    {/* Details */}
                    {isExpanded && (
                      <div className="px-3 pb-3 pt-1 border-t border-border bg-muted/20">
                        <p className="text-xs text-muted-foreground mb-3 mt-2">{plugin.description}</p>
                        
                        <div className="space-y-1.5">
                          <label className="text-xs font-medium text-muted-foreground flex items-center gap-1"><Settings2 className="w-3.5 h-3.5" /> 插件参数</label>
                          <input
                            type="text"
                            value={options}
                            onChange={(e) => updateOptions(plugin.plugin_name, e.target.value)}
                            disabled={!isSelected}
                            placeholder={plugin.metadata?.params_hint || "留空使用默认值"}
                            className="w-full bg-background border border-border text-foreground focus:border-emerald-500 px-3 py-2 rounded-md text-sm font-mono disabled:opacity-50 outline-none"
                          />
                        </div>

                        {plugin.metadata?.provider_config?.key && isSelected && (
                          <div className="space-y-2 mt-4">
                            <label className="text-xs font-medium text-muted-foreground flex items-center gap-1">
                              <Settings2 className="w-3.5 h-3.5" />
                              {plugin.metadata?.provider_config?.title || '渠道配置（JSON）'}
                            </label>

                            {plugin.metadata?.provider_config?.description && (
                              <p className="text-xs text-muted-foreground">{plugin.metadata.provider_config.description}</p>
                            )}

                            {plugin.plugin_name === 'post_body_parameter_filter' ? (
                              <PostBodyParameterFilterEditor
                                valueText={providerConfigText.get(plugin.plugin_name) || ''}
                                onChangeText={(text) => updateProviderConfigText(plugin.plugin_name, text)}
                                availableModels={Array.from(new Set((availableModels || []).map(x => String(x).trim()).filter(Boolean)))}
                              />
                            ) : (
                              <textarea
                                value={providerConfigText.get(plugin.plugin_name) || ''}
                                onChange={(e) => updateProviderConfigText(plugin.plugin_name, e.target.value)}
                                disabled={!isSelected}
                                rows={6}
                                placeholder={
                                  plugin.metadata?.provider_config?.example
                                    ? JSON.stringify(plugin.metadata.provider_config.example, null, 2)
                                    : '请输入 JSON'
                                }
                                className="w-full bg-background border border-border text-foreground focus:border-emerald-500 px-3 py-2 rounded-md text-sm font-mono disabled:opacity-50 outline-none"
                              />
                            )}

                            <div className="flex items-center gap-2">
                              <button
                                type="button"
                                disabled={!isSelected}
                                onClick={() => {
                                  try {
                                    updateProviderConfigText(plugin.plugin_name, formatJsonText(providerConfigText.get(plugin.plugin_name) || ''));
                                  } catch (e: any) {
                                    alert(`格式化失败：${e?.message || 'invalid json'}`);
                                  }
                                }}
                                className="text-xs font-medium text-muted-foreground hover:text-foreground px-2 py-1 bg-muted rounded disabled:opacity-50"
                              >
                                格式化
                              </button>

                              {plugin.metadata?.provider_config?.example && (
                                <button
                                  type="button"
                                  disabled={!isSelected}
                                  onClick={() => updateProviderConfigText(plugin.plugin_name, JSON.stringify(plugin.metadata?.provider_config?.example, null, 2))}
                                  className="text-xs font-medium text-emerald-600 dark:text-emerald-500 hover:text-emerald-500 px-2 py-1 bg-emerald-500/10 rounded disabled:opacity-50"
                                >
                                  填入示例
                                </button>
                              )}

                              <button
                                type="button"
                                disabled={!isSelected}
                                onClick={() => updateProviderConfigText(plugin.plugin_name, '')}
                                className="text-xs font-medium text-red-600 dark:text-red-400 hover:text-red-500 px-2 py-1 bg-red-500/10 rounded disabled:opacity-50"
                              >
                                清空
                              </button>
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>

          </div>

          <div className="p-4 bg-muted/30 border-t border-border flex justify-end gap-3">
            <Dialog.Close className="px-4 py-2 text-sm font-medium text-foreground bg-muted hover:bg-muted/80 rounded-lg">取消</Dialog.Close>
            <button onClick={handleSave} className="px-4 py-2 text-sm font-medium text-primary-foreground bg-primary hover:bg-primary/90 rounded-lg">
              保存插件配置
            </button>
          </div>

        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}