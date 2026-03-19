import { useEffect, useMemo, useState } from 'react';
import { BarChart3, CheckCircle2, Cpu, RefreshCw, Zap } from 'lucide-react';
import { useAuthStore } from '../store/authStore';
import { apiFetch } from '../lib/api';
import {
  type AnalysisEntry,
  type RowPrice,
  type StatData,
  type TrendPoint,
  DashboardHeader,
  PrimaryChartsSection,
  SecondaryChartsSection,
  StatsCardsGrid,
  TIME_RANGES,
  UsageAnalysisPanel,
  buildFormattedChannelStats,
  buildFormattedEndpointStats,
  formatNumber,
} from '../components/dashboard/DashboardShared';

interface UsageAnalysisResponse {
  usage: AnalysisEntry[];
  summary?: {
    provider_count: number;
    model_count: number;
    request_count: number;
    total_prompt_tokens: number;
    total_completion_tokens: number;
    total_tokens: number;
  };
}

interface TrendResponse {
  data?: TrendPoint[];
  models?: string[];
}

interface ApiConfigResponse {
  api_config?: {
    preferences?: Record<string, unknown>;
  };
  preferences?: Record<string, unknown>;
}

interface StoredModelPriceMapValue {
  prompt: number;
  completion: number;
}

type StoredModelPriceMap = Record<string, StoredModelPriceMapValue>;

const DEFAULT_PRICE_STORAGE_KEY = 'dashboard:usage-analysis-default-prices';
const GLOBAL_PROMPT_PRICE_KEY = 'usage_analysis_default_prompt_price';
const GLOBAL_COMPLETION_PRICE_KEY = 'usage_analysis_default_completion_price';
const GLOBAL_MODEL_PRICES_KEY = 'usage_analysis_model_prices';
const FALLBACK_DEFAULT_PRICES: RowPrice = { prompt: 0.3, completion: 1.0 };

const formatDatetimeLocal = (d: Date) =>
  `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}T${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;

const computeTimeRangeWindow = (hours: number): { start: string; end: string } => {
  const now = new Date();
  return { start: formatDatetimeLocal(new Date(now.getTime() - hours * 3600_000)), end: formatDatetimeLocal(now) };
};

const asFiniteNumber = (value: unknown, fallback: number) => {
  const parsed = typeof value === 'number' ? value : Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
};

const getProviderModelKey = (provider: string, model: string) => `${provider}:${model}`;

const normalizeStoredModelPrices = (value: unknown): StoredModelPriceMap => {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return {};
  }

  const result: StoredModelPriceMap = {};
  for (const [key, rawValue] of Object.entries(value as Record<string, unknown>)) {
    if (!rawValue || typeof rawValue !== 'object' || Array.isArray(rawValue)) continue;
    const candidate = rawValue as Partial<StoredModelPriceMapValue>;
    result[key] = {
      prompt: asFiniteNumber(candidate.prompt, FALLBACK_DEFAULT_PRICES.prompt),
      completion: asFiniteNumber(candidate.completion, FALLBACK_DEFAULT_PRICES.completion),
    };
  }
  return result;
};

const readLegacyStoredDefaultPrices = (): RowPrice | null => {
  if (typeof window === 'undefined') {
    return null;
  }

  try {
    const raw = window.localStorage.getItem(DEFAULT_PRICE_STORAGE_KEY);
    if (!raw) {
      return null;
    }

    const parsed = JSON.parse(raw) as Partial<RowPrice>;
    return {
      prompt: asFiniteNumber(parsed.prompt, FALLBACK_DEFAULT_PRICES.prompt),
      completion: asFiniteNumber(parsed.completion, FALLBACK_DEFAULT_PRICES.completion),
    };
  } catch {
    return null;
  }
};

const removeLegacyStoredDefaultPrices = () => {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.removeItem(DEFAULT_PRICE_STORAGE_KEY);
  } catch {
    // ignore
  }
};

export default function Dashboard() {
  const [stats, setStats] = useState<StatData | null>(null);
  const [totalTokens, setTotalTokens] = useState(0);
  const [loading, setLoading] = useState(true);
  const [timeRange, setTimeRange] = useState(24);
  const { token } = useAuthStore();

  const [analysisOpen, setAnalysisOpen] = useState(false);
  const [analysisLoading, setAnalysisLoading] = useState(false);
  const [analysisData, setAnalysisData] = useState<AnalysisEntry[]>([]);
  const [analysisProviders, setAnalysisProviders] = useState<string[]>([]);
  const [analysisModels, setAnalysisModels] = useState<string[]>([]);
  const [analysisStart, setAnalysisStart] = useState(() => computeTimeRangeWindow(24).start);
  const [analysisEnd, setAnalysisEnd] = useState(() => computeTimeRangeWindow(24).end);
  const [defaultPromptPrice, setDefaultPromptPrice] = useState(FALLBACK_DEFAULT_PRICES.prompt);
  const [defaultCompletionPrice, setDefaultCompletionPrice] = useState(FALLBACK_DEFAULT_PRICES.completion);
  const [storedModelPrices, setStoredModelPrices] = useState<StoredModelPriceMap>({});
  const [savingModelPriceKey, setSavingModelPriceKey] = useState<string | null>(null);
  const [rowPrices, setRowPrices] = useState<Record<number, RowPrice>>({});
  const [analysisQueried, setAnalysisQueried] = useState(false);
  const [trendData, setTrendData] = useState<TrendPoint[]>([]);
  const [trendModels, setTrendModels] = useState<string[]>([]);
  const [trendLoading, setTrendLoading] = useState(false);
  const [originalRowPrices, setOriginalRowPrices] = useState<Record<number, RowPrice>>({});
  const [savingAll, setSavingAll] = useState(false);

  const tooltipStyle = {
    backgroundColor: 'hsl(var(--popover))',
    borderColor: 'hsl(var(--border))',
    color: 'hsl(var(--popover-foreground))',
    borderRadius: '8px',
  };

  const fetchGlobalPriceConfig = async () => {
    if (!token) return;

    const legacyPrices = readLegacyStoredDefaultPrices();

    try {
      const res = await apiFetch('/v1/api_config', {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!res.ok) return;

      const data: ApiConfigResponse = await res.json();
      const preferences = data.api_config?.preferences || data.preferences || {};
      const promptFromConfig = preferences[GLOBAL_PROMPT_PRICE_KEY];
      const completionFromConfig = preferences[GLOBAL_COMPLETION_PRICE_KEY];
      const modelPricesFromConfig = normalizeStoredModelPrices(preferences[GLOBAL_MODEL_PRICES_KEY]);

      const promptExists = promptFromConfig !== undefined && promptFromConfig !== null && promptFromConfig !== '';
      const completionExists = completionFromConfig !== undefined && completionFromConfig !== null && completionFromConfig !== '';

      setStoredModelPrices(modelPricesFromConfig);

      if (promptExists || completionExists) {
        setDefaultPromptPrice(asFiniteNumber(promptFromConfig, FALLBACK_DEFAULT_PRICES.prompt));
        setDefaultCompletionPrice(asFiniteNumber(completionFromConfig, FALLBACK_DEFAULT_PRICES.completion));
        removeLegacyStoredDefaultPrices();
        return;
      }

      if (legacyPrices) {
        setDefaultPromptPrice(legacyPrices.prompt);
        setDefaultCompletionPrice(legacyPrices.completion);

        await apiFetch('/v1/api_config/update', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
          body: JSON.stringify({
            preferences: {
              [GLOBAL_PROMPT_PRICE_KEY]: legacyPrices.prompt,
              [GLOBAL_COMPLETION_PRICE_KEY]: legacyPrices.completion,
            },
          }),
        });
        removeLegacyStoredDefaultPrices();
        return;
      }

      setDefaultPromptPrice(FALLBACK_DEFAULT_PRICES.prompt);
      setDefaultCompletionPrice(FALLBACK_DEFAULT_PRICES.completion);
    } catch (err) {
      console.error('Failed to load global usage analysis price config:', err);
      if (legacyPrices) {
        setDefaultPromptPrice(legacyPrices.prompt);
        setDefaultCompletionPrice(legacyPrices.completion);
      }
    }
  };

  const fetchData = async () => {
    if (!token) return;
    setLoading(true);
    try {
      const headers = { Authorization: `Bearer ${token}` };
      const [statsRes, usageRes] = await Promise.all([
        fetch(`/v1/stats?hours=${timeRange}`, { headers }),
        fetch(`/v1/stats/usage_analysis?hours=${timeRange}`, { headers }),
      ]);

      if (statsRes.ok) {
        const data = await statsRes.json();
        setStats(data.stats || data);
      }

      if (usageRes.ok) {
        const data: UsageAnalysisResponse = await usageRes.json();
        setTotalTokens(data.summary?.total_tokens || 0);
      } else {
        setTotalTokens(0);
      }
    } catch (err) {
      console.error('Failed to fetch stats:', err);
      setTotalTokens(0);
    } finally {
      setLoading(false);
    }
  };

  const fetchAnalysis = async () => {
    if (!token) return;
    setAnalysisLoading(true);
    setAnalysisQueried(true);
    setTrendData([]);

    try {
      const headers = { Authorization: `Bearer ${token}` };
      const params = new URLSearchParams();

      if (analysisStart) {
        params.set('start_datetime', new Date(analysisStart).toISOString());
      }
      if (analysisEnd) {
        params.set('end_datetime', new Date(analysisEnd).toISOString());
      }
      if (!analysisStart && !analysisEnd) {
        params.set('hours', String(timeRange));
      }
      if (analysisProviders.length > 0) {
        params.set('provider', analysisProviders.join(','));
      }
      if (analysisModels.length > 0) {
        params.set('model', analysisModels.join(','));
      }

      const queryString = params.toString();
      const [res, trendRes] = await Promise.all([
        fetch(`/v1/stats/usage_analysis?${queryString}`, { headers }),
        fetch(`/v1/stats/model_trend?${queryString}`, { headers }),
      ]);

      if (res.ok) {
        const result: UsageAnalysisResponse = await res.json();
        const data = result.usage || [];
        setAnalysisData(prevData => {
          const newPrices: Record<number, RowPrice> = {};
          data.forEach((entry, index) => {
            const key = getProviderModelKey(entry.provider, entry.model);
            const oldIndex = prevData.findIndex(item => getProviderModelKey(item.provider, item.model) === key);
            if (oldIndex !== -1 && rowPrices[oldIndex]) {
              newPrices[index] = rowPrices[oldIndex];
            } else if (storedModelPrices[key]) {
              newPrices[index] = storedModelPrices[key];
            } else {
              newPrices[index] = { prompt: defaultPromptPrice, completion: defaultCompletionPrice };
            }
          });
          setRowPrices(newPrices);
          setOriginalRowPrices(structuredClone(newPrices));
          return data;
        });
      } else {
        setAnalysisData([]);
        setRowPrices({});
        setOriginalRowPrices({});
      }

      setTrendLoading(true);
      if (trendRes.ok) {
        const trendResult: TrendResponse = await trendRes.json();
        setTrendData(trendResult.data || []);
        setTrendModels(trendResult.models || []);
      } else {
        const text = await trendRes.text().catch(() => '');
        console.error('Trend API request failed:', trendRes.status, text.slice(0, 200));
        setTrendData([]);
        setTrendModels([]);
      }
    } catch (err) {
      console.error('Failed to fetch analysis:', err);
      setAnalysisData([]);
      setTrendData([]);
      setTrendModels([]);
    } finally {
      setAnalysisLoading(false);
      setTrendLoading(false);
    }
  };

  const applyDefaultPricesToAll = () => {
    const prices: Record<number, RowPrice> = {};
    analysisData.forEach((entry, index) => {
      const storedPrice = storedModelPrices[getProviderModelKey(entry.provider, entry.model)];
      prices[index] = storedPrice || { prompt: defaultPromptPrice, completion: defaultCompletionPrice };
    });
    setRowPrices(prices);
  };

  const updateRowPrice = (index: number, field: 'prompt' | 'completion', value: number) => {
    setRowPrices(prev => ({
      ...prev,
      [index]: { ...prev[index], [field]: value },
    }));
  };

  const saveModelPrice = async (entry: AnalysisEntry, index: number) => {
    if (!token) return;

    const price = rowPrices[index] || { prompt: defaultPromptPrice, completion: defaultCompletionPrice };
    const modelKey = getProviderModelKey(entry.provider, entry.model);
    setSavingModelPriceKey(modelKey);

    try {
      const nextStoredModelPrices: StoredModelPriceMap = {
        ...storedModelPrices,
        [modelKey]: {
          prompt: price.prompt,
          completion: price.completion,
        },
      };

      const res = await apiFetch('/v1/api_config/update', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify({
          preferences: {
            [GLOBAL_MODEL_PRICES_KEY]: nextStoredModelPrices,
          },
        }),
      });

      if (!res.ok) {
        const text = await res.text().catch(() => '');
        alert(`保存模型默认价格失败：${text || `HTTP ${res.status}`}`);
        return;
      }

      setStoredModelPrices(nextStoredModelPrices);
    } catch (err) {
      console.error('Failed to save model price:', err);
      alert('保存模型默认价格失败：网络错误');
    } finally {
      setSavingModelPriceKey(null);
    }
  };

  useEffect(() => {
    fetchData();
  }, [token, timeRange]);

  // 当 timeRange 变化时，自动更新 analysisStart / analysisEnd
  useEffect(() => {
    const { start, end } = computeTimeRangeWindow(timeRange);
    setAnalysisStart(start);
    setAnalysisEnd(end);
  }, [timeRange]);

  useEffect(() => {
    fetchGlobalPriceConfig();
  }, [token]);

  useEffect(() => {
    setRowPrices(prev => {
      if (Object.keys(prev).length === 0) return prev;
      const next: Record<number, RowPrice> = {};
      let changed = false;

      analysisData.forEach((entry, index) => {
        const current = prev[index];
        const fallback = storedModelPrices[getProviderModelKey(entry.provider, entry.model)] || {
          prompt: defaultPromptPrice,
          completion: defaultCompletionPrice,
        };

        if (current) {
          next[index] = current;
          return;
        }

        changed = true;
        next[index] = fallback;
      });

      return changed ? next : prev;
    });
  }, [storedModelPrices, defaultPromptPrice, defaultCompletionPrice, analysisData]);

  const channelStats = stats?.channel_success_rates || [];
  const modelStats = stats?.model_request_counts || [];
  const endpointStats = stats?.endpoint_request_counts || [];

  // 计算有哪些行价格被修改了（与原始快照对比）
  const unsavedIndices = useMemo(() => {
    const indices: number[] = [];
    for (const [idxStr, price] of Object.entries(rowPrices)) {
      const idx = Number(idxStr);
      const orig = originalRowPrices[idx];
      if (!orig) continue;
      if (price.prompt !== orig.prompt || price.completion !== orig.completion) {
        indices.push(idx);
      }
    }
    return indices;
  }, [rowPrices, originalRowPrices]);

  const hasUnsavedChanges = unsavedIndices.length > 0;

  const saveAllModelPrices = async () => {
    if (!token || unsavedIndices.length === 0) return;
    setSavingAll(true);
    try {
      const nextStoredModelPrices: StoredModelPriceMap = { ...storedModelPrices };
      for (const idx of unsavedIndices) {
        const entry = analysisData[idx];
        const price = rowPrices[idx];
        if (!entry || !price) continue;
        const modelKey = getProviderModelKey(entry.provider, entry.model);
        nextStoredModelPrices[modelKey] = { prompt: price.prompt, completion: price.completion };
      }

      const res = await apiFetch('/v1/api_config/update', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify({
          preferences: { [GLOBAL_MODEL_PRICES_KEY]: nextStoredModelPrices },
        }),
      });

      if (!res.ok) {
        const text = await res.text().catch(() => '');
        alert(`批量保存模型价格失败：${text || `HTTP ${res.status}`}`);
        return;
      }

      setStoredModelPrices(nextStoredModelPrices);
      setOriginalRowPrices(structuredClone(rowPrices));
    } catch (err) {
      console.error('Failed to save all model prices:', err);
      alert('批量保存模型价格失败：网络错误');
    } finally {
      setSavingAll(false);
    }
  };

  const resetAllModelPrices = () => {
    setRowPrices(structuredClone(originalRowPrices));
  };

  const totalRequests = channelStats.reduce((sum, item) => sum + item.total_requests, 0) || 0;
  const avgSuccessRate = totalRequests > 0
    ? channelStats.reduce((sum, item) => sum + item.success_rate * item.total_requests, 0) / totalRequests
    : 0;
  const activeChannels = channelStats.length || 0;

  const timeRangeLabel = TIME_RANGES.find(range => range.value === timeRange)?.label ?? `${timeRange} 小时`;
  const availableProviders = Array.from(new Set(channelStats.map(item => item.provider).filter(Boolean))).sort();
  const availableModels = Array.from(new Set(modelStats.map(item => item.model).filter(Boolean))).sort();

  const analysisTotalRequests = analysisData.reduce((sum, item) => sum + item.request_count, 0);
  const analysisTotalPrompt = analysisData.reduce((sum, item) => sum + item.total_prompt_tokens, 0);
  const analysisTotalCompletion = analysisData.reduce((sum, item) => sum + item.total_completion_tokens, 0);
  const analysisTotalTokensAll = analysisData.reduce((sum, item) => sum + item.total_tokens, 0);
  const analysisTotalCost = analysisData.reduce((sum, entry, index) => {
    const price = rowPrices[index] || storedModelPrices[getProviderModelKey(entry.provider, entry.model)] || {
      prompt: defaultPromptPrice,
      completion: defaultCompletionPrice,
    };
    return sum + (entry.total_prompt_tokens * price.prompt + entry.total_completion_tokens * price.completion) / 1_000_000;
  }, 0);

  const topCards = [
    { label: '总请求量', value: formatNumber(totalRequests), icon: Zap, color: 'text-amber-500', bg: 'bg-amber-500/10' },
    { label: `Token 消耗 (${timeRangeLabel})`, value: formatNumber(totalTokens), icon: BarChart3, color: 'text-blue-500', bg: 'bg-blue-500/10' },
    { label: '平均成功率', value: `${(avgSuccessRate * 100).toFixed(1)}%`, icon: CheckCircle2, color: 'text-emerald-500', bg: 'bg-emerald-500/10' },
    { label: '活跃渠道', value: activeChannels.toString(), icon: Cpu, color: 'text-purple-500', bg: 'bg-purple-500/10' },
  ];

  const savedModelPriceKeys = useMemo(() => Object.keys(storedModelPrices), [storedModelPrices]);

  if (loading && !stats) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-muted-foreground">
        <RefreshCw className="w-8 h-8 animate-spin mb-3" />
        <p>加载数据中...</p>
      </div>
    );
  }

  return (
    <div className="space-y-6 animate-in fade-in duration-500 font-sans pb-12">
      <DashboardHeader
        timeRange={timeRange}
        onTimeRangeChange={setTimeRange}
        onRefresh={fetchData}
        loading={loading}
      />

      <StatsCardsGrid cards={topCards} />

      <PrimaryChartsSection
        formattedChannelStats={buildFormattedChannelStats(channelStats)}
        modelStats={modelStats}
        tooltipStyle={tooltipStyle}
      />

      <SecondaryChartsSection
        formattedEndpointStats={buildFormattedEndpointStats(endpointStats)}
        channelStats={channelStats}
        tooltipStyle={tooltipStyle}
      />

      <UsageAnalysisPanel
        open={analysisOpen}
        onToggle={() => setAnalysisOpen(prev => !prev)}
        analysisStart={analysisStart}
        setAnalysisStart={setAnalysisStart}
        analysisEnd={analysisEnd}
        setAnalysisEnd={setAnalysisEnd}
        timeRangeLabel={timeRangeLabel}
        availableProviders={availableProviders}
        analysisProviders={analysisProviders}
        setAnalysisProviders={setAnalysisProviders}
        availableModels={availableModels}
        analysisModels={analysisModels}
        setAnalysisModels={setAnalysisModels}
        defaultPromptPrice={defaultPromptPrice}
        setDefaultPromptPrice={setDefaultPromptPrice}
        defaultCompletionPrice={defaultCompletionPrice}
        setDefaultCompletionPrice={setDefaultCompletionPrice}
        onQuery={fetchAnalysis}
        analysisLoading={analysisLoading}
        analysisQueried={analysisQueried}
        trendLoading={trendLoading}
        trendData={trendData}
        trendModels={trendModels}
        analysisData={analysisData}
        rowPrices={rowPrices}
        onApplyDefaultPricesToAll={applyDefaultPricesToAll}
        onUpdateRowPrice={updateRowPrice}
        analysisTotalRequests={analysisTotalRequests}
        analysisTotalPrompt={analysisTotalPrompt}
        analysisTotalCompletion={analysisTotalCompletion}
        analysisTotalTokensAll={analysisTotalTokensAll}
        analysisTotalCost={analysisTotalCost}
        tooltipStyle={tooltipStyle}
        defaultPriceDescription="默认价格来自系统全局配置；若存在 provider + model 专属价格，会优先自动套用。"
        savedModelPriceKeys={savedModelPriceKeys}
        savingModelPriceKey={savingModelPriceKey}
        getModelPriceKey={(entry) => getProviderModelKey(entry.provider, entry.model)}
        onSaveModelPrice={saveModelPrice}
        onSaveAllModelPrices={saveAllModelPrices}
        onResetAllModelPrices={resetAllModelPrices}
        hasUnsavedChanges={hasUnsavedChanges}
        unsavedCount={unsavedIndices.length}
      />
    </div>
  );
}
