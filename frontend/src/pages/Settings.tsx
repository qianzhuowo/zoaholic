import { useState, useEffect } from 'react';
import { useAuthStore } from '../store/authStore';
import { apiFetch } from '../lib/api';
import {
  Settings2, Save, RefreshCw, AlertCircle, Clock, Zap, Shield,
  Timer, Database, Server, Blocks, Plus, Trash2, Edit2, Link, DollarSign
} from 'lucide-react';
import { RateLimitEditor } from '../components/RateLimitEditor';

type CleanupAction = 'clear_fields' | 'delete_rows';
type CleanupTimeMode = 'older_than_hours' | 'custom_range' | 'all';
type CleanupSuccessMode = 'ALL' | 'SUCCESS' | 'FAILED';

interface LogsCleanupResponse {
  dry_run: boolean;
  action: CleanupAction;
  matched_rows: number;
  affected_rows: number;
  selected_fields: string[];
  non_null_counts: Record<string, number>;
  filters: Record<string, unknown>;
  message: string;
}

interface ModelPriceRow {
  id: string;
  provider: string;
  model: string;
  prompt: number;
  completion: number;
}

const LOG_CLEANUP_FIELD_OPTIONS: { key: string; label: string }[] = [
  { key: 'request_headers', label: '用户请求头' },
  { key: 'request_body', label: '用户请求体' },
  { key: 'upstream_request_headers', label: '上游请求头' },
  { key: 'upstream_request_body', label: '上游请求体' },
  { key: 'upstream_response_body', label: '上游响应体' },
  { key: 'response_body', label: '返回给用户的响应体' },
  { key: 'retry_path', label: '重试路径' },
  { key: 'text', label: '文本摘要' },
];

const DEFAULT_CLEANUP_FIELDS = LOG_CLEANUP_FIELD_OPTIONS
  .filter(item => item.key !== 'text')
  .map(item => item.key);

const GLOBAL_PROMPT_PRICE_KEY = 'usage_analysis_default_prompt_price';
const GLOBAL_COMPLETION_PRICE_KEY = 'usage_analysis_default_completion_price';
const GLOBAL_MODEL_PRICES_KEY = 'usage_analysis_model_prices';

const asFiniteNumber = (value: unknown, fallback: number) => {
  const parsed = typeof value === 'number' ? value : Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
};

const buildModelPriceKey = (provider: string, model: string) => `${provider}:${model}`;

const parseModelPriceRows = (value: unknown): ModelPriceRow[] => {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return [];
  }

  return Object.entries(value as Record<string, unknown>).flatMap(([key, rawValue]) => {
    if (!rawValue || typeof rawValue !== 'object' || Array.isArray(rawValue)) return [];
    const [provider = '', ...modelParts] = key.split(':');
    const model = modelParts.join(':');
    if (!provider || !model) return [];

    const candidate = rawValue as Record<string, unknown>;
    return [{
      id: key,
      provider,
      model,
      prompt: asFiniteNumber(candidate.prompt, 0.3),
      completion: asFiniteNumber(candidate.completion, 1.0),
    }];
  });
};

const serializeModelPriceRows = (rows: ModelPriceRow[]) => {
  const result: Record<string, { prompt: number; completion: number }> = {};
  rows.forEach(row => {
    const provider = row.provider.trim();
    const model = row.model.trim();
    if (!provider || !model) return;
    result[buildModelPriceKey(provider, model)] = {
      prompt: asFiniteNumber(row.prompt, 0),
      completion: asFiniteNumber(row.completion, 0),
    };
  });
  return result;
};

export default function Settings() {
  const { token } = useAuthStore();
  const [preferences, setPreferences] = useState<any>({});
  const [modelPriceRows, setModelPriceRows] = useState<ModelPriceRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  // 数据库清理状态
  const [cleanupAction, setCleanupAction] = useState<CleanupAction>('clear_fields');
  const [cleanupTimeMode, setCleanupTimeMode] = useState<CleanupTimeMode>('older_than_hours');
  const [cleanupOlderThanHours, setCleanupOlderThanHours] = useState(168);
  const [cleanupStartTime, setCleanupStartTime] = useState('');
  const [cleanupEndTime, setCleanupEndTime] = useState('');
  const [cleanupProvider, setCleanupProvider] = useState('');
  const [cleanupModel, setCleanupModel] = useState('');
  const [cleanupApiKey, setCleanupApiKey] = useState('');
  const [cleanupStatusCodes, setCleanupStatusCodes] = useState('');
  const [cleanupSuccessMode, setCleanupSuccessMode] = useState<CleanupSuccessMode>('ALL');
  const [cleanupFlaggedOnly, setCleanupFlaggedOnly] = useState(false);
  const [cleanupFields, setCleanupFields] = useState<string[]>(DEFAULT_CLEANUP_FIELDS);
  const [cleanupRunning, setCleanupRunning] = useState(false);
  const [cleanupResult, setCleanupResult] = useState<LogsCleanupResponse | null>(null);
  const [cleanupConfirmText, setCleanupConfirmText] = useState('');
  const [cleanupMessage, setCleanupMessage] = useState('');

  // Load configuration
  useEffect(() => {
    const fetchConfig = async () => {
      if (!token) return;
      setLoading(true);
      try {
        const res = await apiFetch('/v1/api_config', {
          headers: { Authorization: `Bearer ${token}` }
        });
        if (res.ok) {
          const data = await res.json();
          const loadedPreferences = data.api_config?.preferences || data.preferences || {};

          if (!loadedPreferences.external_clients) {
            loadedPreferences.external_clients = [
              { name: 'IdoFront', icon: '🌚', link: 'https://idofront.pages.dev/?baseurl={address}/v1&key={key}' }
            ];
          }
          if (loadedPreferences[GLOBAL_PROMPT_PRICE_KEY] == null) {
            loadedPreferences[GLOBAL_PROMPT_PRICE_KEY] = 0.3;
          }
          if (loadedPreferences[GLOBAL_COMPLETION_PRICE_KEY] == null) {
            loadedPreferences[GLOBAL_COMPLETION_PRICE_KEY] = 1.0;
          }

          setModelPriceRows(parseModelPriceRows(loadedPreferences[GLOBAL_MODEL_PRICES_KEY]));
          setPreferences(loadedPreferences);
        }
      } catch (err) {
        console.error('Failed to load settings:', err);
      } finally {
        setLoading(false);
      }
    };
    fetchConfig();
  }, [token]);

  const updatePreference = (key: string, value: any) => {
    setPreferences((prev: any) => ({ ...prev, [key]: value }));
  };

  const updateModelPriceRow = (id: string, field: keyof Omit<ModelPriceRow, 'id'>, value: string | number) => {
    setModelPriceRows(prev => prev.map(row => {
      if (row.id !== id) return row;
      if (field === 'prompt' || field === 'completion') {
        return { ...row, [field]: typeof value === 'number' ? value : parseFloat(String(value)) || 0 };
      }
      const nextValue = String(value);
      const provider = field === 'provider' ? nextValue : row.provider;
      const model = field === 'model' ? nextValue : row.model;
      return {
        ...row,
        [field]: nextValue,
        id: buildModelPriceKey(provider, model),
      };
    }));
  };

  const addModelPriceRow = () => {
    const seed = `new-provider:${Date.now()}`;
    setModelPriceRows(prev => ([
      ...prev,
      { id: seed, provider: '', model: '', prompt: 0.3, completion: 1.0 },
    ]));
  };

  const removeModelPriceRow = (id: string) => {
    setModelPriceRows(prev => prev.filter(row => row.id !== id));
  };

  const parseErrorMessage = async (res: Response) => {
    try {
      const data = await res.json();
      return data?.detail || data?.message || `HTTP ${res.status}`;
    } catch {
      return `HTTP ${res.status}`;
    }
  };

  const toIsoStringOrUndefined = (localDateTime: string) => {
    if (!localDateTime) return undefined;
    const dt = new Date(localDateTime);
    if (Number.isNaN(dt.getTime())) return undefined;
    return dt.toISOString();
  };

  const toggleCleanupField = (field: string) => {
    setCleanupFields(prev => (
      prev.includes(field) ? prev.filter(item => item !== field) : [...prev, field]
    ));
  };

  const buildCleanupPayload = (dryRun: boolean) => {
    const payload: Record<string, unknown> = {
      dry_run: dryRun,
      action: cleanupAction,
      success_mode: cleanupSuccessMode,
      flagged_only: cleanupFlaggedOnly,
    };

    if (cleanupAction === 'clear_fields') {
      payload.fields = cleanupFields;
    }

    if (cleanupTimeMode === 'older_than_hours') {
      payload.older_than_hours = cleanupOlderThanHours;
    } else if (cleanupTimeMode === 'custom_range') {
      payload.start_time = toIsoStringOrUndefined(cleanupStartTime);
      payload.end_time = toIsoStringOrUndefined(cleanupEndTime);
    }

    if (cleanupProvider.trim()) payload.provider = cleanupProvider.trim();
    if (cleanupModel.trim()) payload.model = cleanupModel.trim();
    if (cleanupApiKey.trim()) payload.api_key = cleanupApiKey.trim();
    if (cleanupStatusCodes.trim()) {
      payload.status_codes = cleanupStatusCodes
        .split(',')
        .map(item => parseInt(item.trim(), 10))
        .filter(item => Number.isFinite(item));
    }

    return payload;
  };

  const handleCleanupPreview = async () => {
    if (!token) return;
    setCleanupRunning(true);
    setCleanupMessage('');
    try {
      const res = await apiFetch('/v1/logs/cleanup', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify(buildCleanupPayload(true)),
      });

      if (!res.ok) {
        const msg = await parseErrorMessage(res);
        setCleanupMessage(`预览失败：${msg}`);
        return;
      }

      const data: LogsCleanupResponse = await res.json();
      setCleanupResult(data);
      setCleanupMessage(`预览完成：匹配 ${data.matched_rows} 条记录`);
    } catch {
      setCleanupMessage('预览失败：网络错误');
    } finally {
      setCleanupRunning(false);
    }
  };

  const handleCleanupExecute = async () => {
    const requiredConfirmPhrase = cleanupAction === 'delete_rows' ? 'DELETE' : 'CLEAR';
    if (!token) return;

    if (cleanupAction === 'clear_fields' && cleanupFields.length === 0) {
      alert('请至少选择一个要清空的字段');
      return;
    }

    if (cleanupConfirmText.trim().toUpperCase() !== requiredConfirmPhrase) {
      alert(`请输入确认词 ${requiredConfirmPhrase} 后再执行`);
      return;
    }

    if (!window.confirm('该操作会修改数据库，是否确认执行？')) {
      return;
    }

    setCleanupRunning(true);
    setCleanupMessage('');
    try {
      const res = await apiFetch('/v1/logs/cleanup', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify(buildCleanupPayload(false)),
      });

      if (!res.ok) {
        const msg = await parseErrorMessage(res);
        setCleanupMessage(`执行失败：${msg}`);
        return;
      }

      const data: LogsCleanupResponse = await res.json();
      setCleanupResult(data);
      setCleanupConfirmText('');
      setCleanupMessage(`执行完成：影响 ${data.affected_rows} 条记录`);
    } catch (err) {
      setCleanupMessage('执行失败：网络错误');
    } finally {
      setCleanupRunning(false);
    }
  };

  const handleSave = async () => {
    if (!token) return;

    const duplicateKeys = new Set<string>();
    const seenKeys = new Set<string>();
    for (const row of modelPriceRows) {
      const provider = row.provider.trim();
      const model = row.model.trim();
      if (!provider && !model) continue;
      if (!provider || !model) {
        alert('模型价格表中的 provider 和 model 必须同时填写');
        return;
      }
      const key = buildModelPriceKey(provider, model);
      if (seenKeys.has(key)) {
        duplicateKeys.add(key);
      }
      seenKeys.add(key);
    }

    if (duplicateKeys.size > 0) {
      alert(`模型价格表存在重复项：${Array.from(duplicateKeys).join('、')}`);
      return;
    }

    setSaving(true);
    try {
      const nextPreferences = {
        ...preferences,
        [GLOBAL_MODEL_PRICES_KEY]: serializeModelPriceRows(modelPriceRows),
      };
      const res = await apiFetch('/v1/api_config/update', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify({ preferences: nextPreferences })
      });
      if (res.ok) {
        setPreferences(nextPreferences);
        alert('配置已保存成功');
      } else {
        const msg = await parseErrorMessage(res);
        alert(`保存失败：${msg}`);
      }
    } catch (err) {
      alert('网络错误');
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-muted-foreground">
        <RefreshCw className="w-8 h-8 animate-spin mb-4" />
        <p>加载配置中...</p>
      </div>
    );
  }

  return (
    <div className="space-y-6 animate-in fade-in duration-500 font-sans max-w-4xl mx-auto pb-12">
      <div className="flex justify-between items-center border-b border-border pb-6">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-foreground">系统设置</h1>
          <p className="text-muted-foreground mt-1">管理全局配置和系统首选项</p>
        </div>
        <button
          onClick={handleSave}
          disabled={saving}
          className="bg-primary hover:bg-primary/90 text-primary-foreground px-4 py-2 rounded-lg flex items-center gap-2 font-medium transition-colors disabled:opacity-50"
        >
          {saving ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Save className="w-4 h-4" />}
          保存配置
        </button>
      </div>

      <div className="space-y-8">
        <section className="bg-card border border-border rounded-xl overflow-hidden">
          <div className="p-4 border-b border-border bg-muted/30 flex items-center gap-2 font-medium text-foreground">
            <Zap className="w-5 h-5 text-amber-500" /> 高可用性与调度
          </div>
          <div className="p-6 space-y-6">
            <div className="grid grid-cols-2 gap-6">
              <div>
                <label className="text-sm font-medium text-foreground mb-1.5 block">最大重试次数</label>
                <input
                  type="number" min="1" max="100"
                  value={preferences.max_retry_count ?? 10}
                  onChange={e => updatePreference('max_retry_count', parseInt(e.target.value))}
                  className="w-full bg-background border border-border px-3 py-2 rounded-lg text-sm text-foreground"
                />
                <p className="text-xs text-muted-foreground mt-1">多渠道场景下的最大重试次数上限（1-100）</p>
              </div>
              <div>
                <label className="text-sm font-medium text-foreground mb-1.5 block">渠道冷却时间 (秒)</label>
                <input
                  type="number" min="0"
                  value={preferences.cooldown_period ?? 300}
                  onChange={e => updatePreference('cooldown_period', parseInt(e.target.value))}
                  className="w-full bg-background border border-border px-3 py-2 rounded-lg text-sm text-foreground"
                />
                <p className="text-xs text-muted-foreground mt-1">失败渠道的冷却时间，设为 0 禁用</p>
              </div>
            </div>

            <div>
              <label className="text-sm font-medium text-foreground mb-1.5 block">全局调度算法</label>
              <select
                value={preferences.SCHEDULING_ALGORITHM || 'fixed_priority'}
                onChange={e => updatePreference('SCHEDULING_ALGORITHM', e.target.value)}
                className="w-full bg-background border border-border px-3 py-2 rounded-lg text-sm text-foreground"
              >
                <option value="fixed_priority">固定优先级 (fixed_priority) - 始终使用第一个可用渠道</option>
                <option value="round_robin">轮询 (round_robin) - 按顺序依次请求</option>
                <option value="weighted_round_robin">加权轮询 (weighted_round_robin) - 按渠道权重分配</option>
                <option value="lottery">抽奖 (lottery) - 按权重随机选择</option>
                <option value="random">随机 (random) - 完全随机</option>
                <option value="smart_round_robin">智能轮询 (smart_round_robin) - 基于历史成功率</option>
              </select>
            </div>
          </div>
        </section>

        <section className="bg-card border border-border rounded-xl overflow-hidden">
          <div className="p-4 border-b border-border bg-muted/30 flex items-center gap-2 font-medium text-foreground">
            <DollarSign className="w-5 h-5 text-amber-500" /> 仪表盘费用模拟默认价格
          </div>
          <div className="p-6 space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="text-sm font-medium text-foreground mb-1.5 block">默认输入价格 ($/M)</label>
                <input
                  type="number"
                  min="0"
                  step="0.01"
                  value={preferences[GLOBAL_PROMPT_PRICE_KEY] ?? 0.3}
                  onChange={e => updatePreference(GLOBAL_PROMPT_PRICE_KEY, parseFloat(e.target.value) || 0)}
                  className="w-full bg-background border border-border px-3 py-2 rounded-lg text-sm text-foreground"
                />
                <p className="text-xs text-muted-foreground mt-1">当不存在专属模型价格时，回退使用该默认输入 Token 单价。</p>
              </div>
              <div>
                <label className="text-sm font-medium text-foreground mb-1.5 block">默认输出价格 ($/M)</label>
                <input
                  type="number"
                  min="0"
                  step="0.01"
                  value={preferences[GLOBAL_COMPLETION_PRICE_KEY] ?? 1.0}
                  onChange={e => updatePreference(GLOBAL_COMPLETION_PRICE_KEY, parseFloat(e.target.value) || 0)}
                  className="w-full bg-background border border-border px-3 py-2 rounded-lg text-sm text-foreground"
                />
                <p className="text-xs text-muted-foreground mt-1">当不存在专属模型价格时，回退使用该默认输出 Token 单价。</p>
              </div>
            </div>
            <div className="rounded-lg border border-amber-500/20 bg-amber-500/5 px-4 py-3 text-xs text-muted-foreground">
              该配置为系统全局配置。保存后，所有管理员在仪表盘中看到的默认价格都会同步更新。
            </div>
          </div>
        </section>

        <section className="bg-card border border-border rounded-xl overflow-hidden">
          <div className="p-4 border-b border-border bg-muted/30 flex items-center justify-between gap-2 font-medium text-foreground">
            <div className="flex items-center gap-2">
              <DollarSign className="w-5 h-5 text-emerald-500" /> provider + model 专属价格表
            </div>
            <button
              type="button"
              onClick={addModelPriceRow}
              className="inline-flex items-center gap-1 rounded-lg border border-border bg-background px-3 py-1.5 text-xs hover:bg-muted"
            >
              <Plus className="w-3.5 h-3.5" /> 新增一行
            </button>
          </div>
          <div className="p-6 space-y-4">
            <div className="rounded-lg border border-emerald-500/20 bg-emerald-500/5 px-4 py-3 text-xs text-muted-foreground">
              专属价格优先级高于全局默认价格。键规则为 <code className="px-1 rounded bg-muted">provider:model</code>。
            </div>

            <div className="overflow-x-auto border border-border rounded-lg">
              <table className="w-full text-left text-sm">
                <thead className="bg-muted text-muted-foreground font-medium">
                  <tr>
                    <th className="px-4 py-3">渠道 provider</th>
                    <th className="px-4 py-3">模型 model</th>
                    <th className="px-4 py-3 text-center">输入价格 ($/M)</th>
                    <th className="px-4 py-3 text-center">输出价格 ($/M)</th>
                    <th className="px-4 py-3 text-center">操作</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border">
                  {modelPriceRows.length === 0 ? (
                    <tr>
                      <td colSpan={5} className="px-4 py-8 text-center text-muted-foreground">暂无专属价格，新增后保存即可生效</td>
                    </tr>
                  ) : (
                    modelPriceRows.map(row => (
                      <tr key={row.id} className="hover:bg-muted/50 transition-colors">
                        <td className="px-4 py-3">
                          <input
                            type="text"
                            value={row.provider}
                            onChange={e => updateModelPriceRow(row.id, 'provider', e.target.value)}
                            placeholder="例如 openai"
                            className="w-full bg-background border border-border px-3 py-2 rounded-lg text-sm text-foreground"
                          />
                        </td>
                        <td className="px-4 py-3">
                          <input
                            type="text"
                            value={row.model}
                            onChange={e => updateModelPriceRow(row.id, 'model', e.target.value)}
                            placeholder="例如 gpt-4o"
                            className="w-full bg-background border border-border px-3 py-2 rounded-lg text-sm text-foreground"
                          />
                        </td>
                        <td className="px-4 py-3 text-center">
                          <input
                            type="number"
                            min="0"
                            step="0.01"
                            value={row.prompt}
                            onChange={e => updateModelPriceRow(row.id, 'prompt', e.target.value)}
                            className="w-28 bg-background border border-border px-3 py-2 rounded-lg text-sm text-foreground text-center"
                          />
                        </td>
                        <td className="px-4 py-3 text-center">
                          <input
                            type="number"
                            min="0"
                            step="0.01"
                            value={row.completion}
                            onChange={e => updateModelPriceRow(row.id, 'completion', e.target.value)}
                            className="w-28 bg-background border border-border px-3 py-2 rounded-lg text-sm text-foreground text-center"
                          />
                        </td>
                        <td className="px-4 py-3 text-center">
                          <button
                            type="button"
                            onClick={() => removeModelPriceRow(row.id)}
                            className="inline-flex items-center gap-1 rounded-lg border border-rose-500/20 bg-rose-500/10 px-3 py-1.5 text-xs text-rose-600 hover:bg-rose-500/20"
                          >
                            <Trash2 className="w-3.5 h-3.5" /> 删除
                          </button>
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </section>

        <section className="bg-card border border-border rounded-xl overflow-hidden">
          <div className="p-4 border-b border-border bg-muted/30 flex items-center gap-2 font-medium text-foreground">
            <Shield className="w-5 h-5 text-emerald-500" /> 安全与速率限制
          </div>
          <div className="p-6 space-y-4">
            <RateLimitEditor
              value={preferences.rate_limit}
              onChange={value => updatePreference('rate_limit', value)}
              title="全局速率限制"
              description="作用于整个网关入口。留空则回退到默认无限制配置（999999/min）。支持组合多个时间窗口。"
            />
            <div className="rounded-lg border border-emerald-500/20 bg-emerald-500/5 px-4 py-3 text-xs text-muted-foreground">
              推荐做法：先设置短窗口保护突发流量（如 60/min），再叠加长窗口保护（如 5000/day）。
            </div>
          </div>
        </section>

        <section className="bg-card border border-border rounded-xl overflow-hidden">
          <div className="p-4 border-b border-border bg-muted/30 flex items-center gap-2 font-medium text-foreground">
            <Timer className="w-5 h-5 text-blue-500" /> 超时与心跳配置
          </div>
          <div className="p-6 space-y-6">
            <div className="grid grid-cols-2 gap-6">
              <div>
                <label className="text-sm font-medium text-foreground mb-1.5 block">默认模型超时时间 (秒)</label>
                <input
                  type="number" min="30" max="3600"
                  value={preferences.model_timeout?.default ?? 600}
                  onChange={e => updatePreference('model_timeout', { ...preferences.model_timeout, default: parseInt(e.target.value) })}
                  className="w-full bg-background border border-border px-3 py-2 rounded-lg text-sm text-foreground"
                />
              </div>
              <div>
                <label className="text-sm font-medium text-foreground mb-1.5 block">Keepalive 心跳间隔 (秒)</label>
                <input
                  type="number" min="0" max="300"
                  value={preferences.keepalive_interval?.default ?? 25}
                  onChange={e => updatePreference('keepalive_interval', { ...preferences.keepalive_interval, default: parseInt(e.target.value) })}
                  className="w-full bg-background border border-border px-3 py-2 rounded-lg text-sm text-foreground"
                />
              </div>
            </div>

            <div className="p-4 bg-blue-500/10 border border-blue-500/20 rounded-lg flex gap-3 text-sm">
              <AlertCircle className="w-5 h-5 text-blue-500 flex-shrink-0" />
              <div>
                <div className="font-medium text-blue-700 dark:text-blue-400 mb-1">长思考模型配置建议</div>
                <ul className="list-disc pl-4 space-y-1 text-blue-600 dark:text-blue-300/80">
                  <li>Nginx 反向代理请设置 <code className="bg-blue-500/20 px-1 rounded">proxy_read_timeout 600s;</code></li>
                  <li>对于 DeepSeek R1 / Claude Thinking，建议心跳间隔设为 20-30 秒</li>
                  <li>Keepalive 可以有效防止 CDN 因空闲时间过长断开连接</li>
                </ul>
              </div>
            </div>
          </div>
        </section>

        <section className="bg-card border border-border rounded-xl overflow-hidden">
          <div className="p-4 border-b border-border bg-muted/30 flex items-center gap-2 font-medium text-foreground">
            <Database className="w-5 h-5 text-purple-500" /> 数据保留策略
          </div>
          <div className="p-6 space-y-6">
            <div>
              <label className="text-sm font-medium text-foreground mb-1.5 block">日志原始数据保留时间 (小时)</label>
              <input
                type="number" min="0"
                value={preferences.log_raw_data_retention_hours ?? 1}
                onChange={e => updatePreference('log_raw_data_retention_hours', parseInt(e.target.value))}
                className="w-full bg-background border border-border px-3 py-2 rounded-lg text-sm text-foreground"
              />
              <p className="text-xs text-muted-foreground mt-2">设为 0 表示不保存请求/响应原始数据，减少存储占用</p>
            </div>

            <div>
              <label className="text-sm font-medium text-foreground mb-1.5 block">日志保留策略</label>
              <select
                value={preferences.log_retention_mode ?? 'keep'}
                onChange={e => updatePreference('log_retention_mode', e.target.value)}
                className="w-full bg-background border border-border px-3 py-2 rounded-lg text-sm text-foreground"
              >
                <option value="keep">不自动清理（永久保留）</option>
                <option value="manual">仅手动清理</option>
                <option value="auto_delete">自动清理（删除过期日志）</option>
              </select>
            </div>

            {(preferences.log_retention_mode ?? 'keep') === 'auto_delete' && (
              <div>
                <label className="text-sm font-medium text-foreground mb-1.5 block">保留天数</label>
                <div className="flex flex-wrap gap-2 items-center">
                  <input
                    type="number" min="1" max="3650"
                    value={preferences.log_retention_days ?? 30}
                    onChange={e => {
                      const v = parseInt(e.target.value, 10);
                      updatePreference('log_retention_days', Number.isFinite(v) ? v : 30);
                    }}
                    className="w-40 bg-background border border-border px-3 py-2 rounded-lg text-sm text-foreground"
                  />
                  <button type="button" onClick={() => updatePreference('log_retention_days', 7)} className="text-xs bg-muted hover:bg-muted/80 border border-border px-2 py-1 rounded">7 天</button>
                  <button type="button" onClick={() => updatePreference('log_retention_days', 30)} className="text-xs bg-muted hover:bg-muted/80 border border-border px-2 py-1 rounded">30 天</button>
                  <button type="button" onClick={() => updatePreference('log_retention_days', 90)} className="text-xs bg-muted hover:bg-muted/80 border border-border px-2 py-1 rounded">90 天</button>
                </div>
                <p className="text-xs text-muted-foreground mt-2">后台任务每天在指定时间执行一次：删除早于 N 天的 request_stats / channel_stats</p>
              </div>
            )}

            {(preferences.log_retention_mode ?? 'keep') === 'auto_delete' && (
              <div>
                <label className="text-sm font-medium text-foreground mb-1.5 block">每天执行时间（按服务器时区；容器环境默认可能是 UTC）</label>
                <div className="flex items-center gap-2">
                  <input
                    type="time"
                    value={preferences.log_retention_run_at ?? '03:00'}
                    onChange={e => updatePreference('log_retention_run_at', e.target.value)}
                    className="w-40 bg-background border border-border px-3 py-2 rounded-lg text-sm text-foreground"
                  />
                  <span className="text-xs text-muted-foreground">默认 03:00</span>
                </div>
                <p className="text-xs text-muted-foreground mt-2">
                  若不设置时区，系统会使用服务器本地时区（在容器中通常为 UTC）。如需指定时区（例如 Asia/Shanghai），可在配置中设置{' '}
                  <code className="px-1 rounded bg-muted">log_retention_timezone</code>
                </p>
              </div>
            )}
          </div>
        </section>

        <section className="bg-card border border-border rounded-xl overflow-hidden">
          <div className="p-4 border-b border-border bg-muted/30 flex items-center gap-2 font-medium text-foreground">
            <Database className="w-5 h-5 text-rose-500" /> 数据库清理工具
          </div>
          <div className="p-6 space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="text-sm font-medium text-foreground mb-1.5 block">操作类型</label>
                <select
                  value={cleanupAction}
                  onChange={e => setCleanupAction(e.target.value as CleanupAction)}
                  className="w-full bg-background border border-border px-3 py-2 rounded-lg text-sm text-foreground"
                >
                  <option value="clear_fields">清空字段内容</option>
                  <option value="delete_rows">删除整行日志</option>
                </select>
              </div>
              <div>
                <label className="text-sm font-medium text-foreground mb-1.5 block">时间范围</label>
                <select
                  value={cleanupTimeMode}
                  onChange={e => setCleanupTimeMode(e.target.value as CleanupTimeMode)}
                  className="w-full bg-background border border-border px-3 py-2 rounded-lg text-sm text-foreground"
                >
                  <option value="older_than_hours">早于指定小时</option>
                  <option value="custom_range">自定义时间范围</option>
                  <option value="all">全部时间</option>
                </select>
              </div>
            </div>

            {cleanupTimeMode === 'older_than_hours' && (
              <div>
                <label className="text-sm font-medium text-foreground mb-1.5 block">早于多少小时</label>
                <input
                  type="number"
                  min="1"
                  value={cleanupOlderThanHours}
                  onChange={e => setCleanupOlderThanHours(parseInt(e.target.value, 10) || 168)}
                  className="w-full md:w-60 bg-background border border-border px-3 py-2 rounded-lg text-sm text-foreground"
                />
              </div>
            )}

            {cleanupTimeMode === 'custom_range' && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <label className="text-sm font-medium text-foreground mb-1.5 block">开始时间</label>
                  <input
                    type="datetime-local"
                    value={cleanupStartTime}
                    onChange={e => setCleanupStartTime(e.target.value)}
                    className="w-full bg-background border border-border px-3 py-2 rounded-lg text-sm text-foreground"
                  />
                </div>
                <div>
                  <label className="text-sm font-medium text-foreground mb-1.5 block">结束时间</label>
                  <input
                    type="datetime-local"
                    value={cleanupEndTime}
                    onChange={e => setCleanupEndTime(e.target.value)}
                    className="w-full bg-background border border-border px-3 py-2 rounded-lg text-sm text-foreground"
                  />
                </div>
              </div>
            )}

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="text-sm font-medium text-foreground mb-1.5 block">渠道 provider（可选）</label>
                <input
                  type="text"
                  value={cleanupProvider}
                  onChange={e => setCleanupProvider(e.target.value)}
                  className="w-full bg-background border border-border px-3 py-2 rounded-lg text-sm text-foreground"
                />
              </div>
              <div>
                <label className="text-sm font-medium text-foreground mb-1.5 block">模型 model（可选）</label>
                <input
                  type="text"
                  value={cleanupModel}
                  onChange={e => setCleanupModel(e.target.value)}
                  className="w-full bg-background border border-border px-3 py-2 rounded-lg text-sm text-foreground"
                />
              </div>
              <div>
                <label className="text-sm font-medium text-foreground mb-1.5 block">API Key 前缀（可选）</label>
                <input
                  type="text"
                  value={cleanupApiKey}
                  onChange={e => setCleanupApiKey(e.target.value)}
                  className="w-full bg-background border border-border px-3 py-2 rounded-lg text-sm text-foreground"
                />
              </div>
              <div>
                <label className="text-sm font-medium text-foreground mb-1.5 block">状态码（可选，逗号分隔）</label>
                <input
                  type="text"
                  value={cleanupStatusCodes}
                  onChange={e => setCleanupStatusCodes(e.target.value)}
                  className="w-full bg-background border border-border px-3 py-2 rounded-lg text-sm text-foreground"
                />
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="text-sm font-medium text-foreground mb-1.5 block">成功状态筛选</label>
                <select
                  value={cleanupSuccessMode}
                  onChange={e => setCleanupSuccessMode(e.target.value as CleanupSuccessMode)}
                  className="w-full bg-background border border-border px-3 py-2 rounded-lg text-sm text-foreground"
                >
                  <option value="ALL">全部</option>
                  <option value="SUCCESS">仅成功</option>
                  <option value="FAILED">仅失败</option>
                </select>
              </div>
              <div className="flex items-center gap-3 pt-7">
                <input
                  id="cleanup-flagged-only"
                  type="checkbox"
                  checked={cleanupFlaggedOnly}
                  onChange={e => setCleanupFlaggedOnly(e.target.checked)}
                  className="h-4 w-4"
                />
                <label htmlFor="cleanup-flagged-only" className="text-sm text-foreground">仅处理已标记日志</label>
              </div>
            </div>

            {cleanupAction === 'clear_fields' && (
              <div>
                <label className="text-sm font-medium text-foreground mb-3 block">要清空的字段</label>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {LOG_CLEANUP_FIELD_OPTIONS.map(option => (
                    <label key={option.key} className="flex items-center gap-3 rounded-lg border border-border px-3 py-2 text-sm text-foreground bg-background">
                      <input
                        type="checkbox"
                        checked={cleanupFields.includes(option.key)}
                        onChange={() => toggleCleanupField(option.key)}
                        className="h-4 w-4"
                      />
                      <span>{option.label}</span>
                    </label>
                  ))}
                </div>
              </div>
            )}

            <div className="rounded-lg border border-amber-500/20 bg-amber-500/5 px-4 py-3 text-sm text-muted-foreground">
              先执行“预览”确认命中范围，再执行正式操作。删除行操作不可恢复，请务必谨慎。
            </div>

            <div className="flex flex-wrap gap-3">
              <button
                onClick={handleCleanupPreview}
                disabled={cleanupRunning}
                className="px-4 py-2 rounded-lg border border-border bg-background hover:bg-muted text-sm font-medium text-foreground disabled:opacity-50"
              >
                {cleanupRunning ? '处理中...' : '预览'}
              </button>
              <button
                onClick={handleCleanupExecute}
                disabled={cleanupRunning}
                className="px-4 py-2 rounded-lg bg-rose-600 hover:bg-rose-700 text-white text-sm font-medium disabled:opacity-50"
              >
                {cleanupRunning ? '处理中...' : '执行'}
              </button>
              <input
                type="text"
                value={cleanupConfirmText}
                onChange={e => setCleanupConfirmText(e.target.value)}
                placeholder={cleanupAction === 'delete_rows' ? '输入 DELETE 确认' : '输入 CLEAR 确认'}
                className="px-3 py-2 rounded-lg border border-border bg-background text-sm text-foreground"
              />
            </div>

            {cleanupMessage && (
              <div className="text-sm text-muted-foreground">{cleanupMessage}</div>
            )}

            {cleanupResult && (
              <div className="rounded-lg border border-border bg-background p-4 text-sm text-foreground space-y-2">
                <div>匹配行数：{cleanupResult.matched_rows}</div>
                <div>影响行数：{cleanupResult.affected_rows}</div>
                <div>操作说明：{cleanupResult.message}</div>
              </div>
            )}
          </div>
        </section>
      </div>
    </div>
  );
}
