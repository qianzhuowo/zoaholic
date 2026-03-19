import type { CSSProperties, Dispatch, SetStateAction } from 'react';
import {
  Activity,
  AlertCircle,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Cpu,
  DollarSign,
  RefreshCw,
  RotateCcw,
  Save,
  Search,
  Server,
  X,
  type LucideIcon,
} from 'lucide-react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

export interface StatData {
  time_range: string;
  channel_success_rates: { provider: string; success_rate: number; total_requests: number }[];
  model_request_counts: { model: string; count: number }[];
  endpoint_request_counts: { endpoint: string; count: number }[];
}

export interface AnalysisEntry {
  provider: string;
  model: string;
  request_count: number;
  total_prompt_tokens: number;
  total_completion_tokens: number;
  total_tokens: number;
}

export interface RowPrice {
  prompt: number;
  completion: number;
}

export type TrendPoint = Record<string, string | number | null>;

export interface DashboardCard {
  label: string;
  value: string;
  icon: LucideIcon;
  color: string;
  bg: string;
}

export const TIME_RANGES = [
  { label: '1 小时', value: 1 },
  { label: '6 小时', value: 6 },
  { label: '24 小时', value: 24 },
  { label: '7 天', value: 168 },
  { label: '30 天', value: 720 },
];

export const CHART_COLORS = [
  'hsl(var(--primary))',
  'hsl(var(--ring))',
  'hsl(160 84% 39%)',
  'hsl(38 92% 50%)',
  'hsl(var(--destructive))',
  'hsl(var(--secondary-foreground))',
];

export const LINE_COLORS = [
  '#3b82f6',
  '#ef4444',
  '#22c55e',
  '#f59e0b',
  '#8b5cf6',
  '#ec4899',
  '#06b6d4',
  '#84cc16',
];

export const AXIS_COLOR = 'hsl(var(--muted-foreground))';
export const SUCCESS_COLOR = 'hsl(160 84% 39%)';
export const WARNING_COLOR = 'hsl(38 92% 50%)';
export const ERROR_COLOR = 'hsl(var(--destructive))';

export const formatTokens = (n: number) => {
  if (n >= 1_000_000_000) return `${(n / 1_000_000_000).toFixed(1)}B`;
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return n.toString();
};

export const formatNumber = (n: number) => (n || 0).toLocaleString();

export const formatCost = (n: number) => {
  if (n === 0) return '$0.00';
  if (n >= 1) return `$${n.toFixed(2)}`;
  if (n >= 0.01) return `$${n.toFixed(4)}`;
  return `$${n.toFixed(6)}`;
};

export const buildFormattedEndpointStats = (endpointStats: StatData['endpoint_request_counts']) => (
  endpointStats.slice(0, 5).map(item => ({
    name: item.endpoint.replace('POST ', '').replace('GET ', ''),
    value: item.count,
  }))
);

export const buildFormattedChannelStats = (channelStats: StatData['channel_success_rates']) => (
  channelStats.slice(0, 6).map(item => ({
    name: item.provider,
    success_rate: item.success_rate * 100,
    requests: item.total_requests,
  }))
);

function MultiSelect({
  label,
  options,
  selected,
  onChange,
  placeholder,
}: {
  label: string;
  options: string[];
  selected: string[];
  onChange: (val: string[]) => void;
  placeholder: string;
}) {
  const toggle = (val: string) => {
    if (selected.includes(val)) {
      onChange(selected.filter(v => v !== val));
    } else {
      onChange([...selected, val]);
    }
  };

  return (
    <div>
      <div className="flex items-center justify-between mb-1.5">
        <label className="text-xs font-medium text-muted-foreground">{label}</label>
        <div className="flex items-center gap-2">
          {options.length > 0 && (
            <button
              type="button"
              onClick={() => onChange(selected.length === options.length ? [] : [...options])}
              className="text-xs text-primary hover:underline"
            >
              {selected.length === options.length ? '清空' : '全选'}
            </button>
          )}
          {selected.length > 0 && selected.length < options.length && (
            <button
              type="button"
              onClick={() => onChange([])}
              className="text-xs text-muted-foreground hover:text-foreground"
            >
              清空
            </button>
          )}
        </div>
      </div>
      <div className="flex flex-wrap gap-1.5 p-2 bg-background border border-border rounded-lg min-h-[36px] max-h-32 overflow-y-auto">
        {options.length === 0 ? (
          <span className="text-xs text-muted-foreground py-0.5">{placeholder}</span>
        ) : (
          options.map(opt => (
            <button
              key={opt}
              type="button"
              onClick={() => toggle(opt)}
              className={`inline-flex items-center gap-1 px-2 py-1 text-xs rounded-md border transition-colors ${
                selected.includes(opt)
                  ? 'bg-primary text-primary-foreground border-primary'
                  : 'bg-muted/50 text-muted-foreground border-border hover:bg-muted hover:text-foreground'
              }`}
            >
              <span className="truncate max-w-[150px]">{opt}</span>
              {selected.includes(opt) && <X className="w-3 h-3 shrink-0" />}
            </button>
          ))
        )}
      </div>
    </div>
  );
}

export function DashboardHeader({
  timeRange,
  onTimeRangeChange,
  onRefresh,
  loading,
}: {
  timeRange: number;
  onTimeRangeChange: (value: number) => void;
  onRefresh: () => void;
  loading: boolean;
}) {
  return (
    <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
      <div>
        <h1 className="text-3xl font-bold tracking-tight text-foreground">数据看板</h1>
        <p className="text-muted-foreground mt-1">系统网关的实时监控与数据分析。</p>
      </div>

      <div className="flex items-center gap-2">
        <div className="flex items-center bg-card border border-border rounded-lg p-1">
          {TIME_RANGES.map(range => (
            <button
              key={range.value}
              onClick={() => onTimeRangeChange(range.value)}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all ${timeRange === range.value
                ? 'bg-primary text-primary-foreground shadow-sm'
                : 'text-muted-foreground hover:text-foreground hover:bg-muted/50'
              }`}
            >
              {range.label}
            </button>
          ))}
        </div>
        <button onClick={onRefresh} className="p-2 text-muted-foreground hover:text-foreground bg-card border border-border rounded-lg transition-colors">
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>
    </div>
  );
}

export function StatsCardsGrid({ cards }: { cards: DashboardCard[] }) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      {cards.map((stat) => {
        const Icon = stat.icon;
        return (
          <div key={stat.label} className="bg-card border border-border p-6 rounded-xl shadow-sm">
            <div className="flex justify-between items-start">
              <div>
                <p className="text-sm font-medium text-muted-foreground">{stat.label}</p>
                <h3 className="text-3xl font-bold text-foreground mt-2">{stat.value}</h3>
              </div>
              <div className={`p-2 rounded-lg ${stat.bg}`}>
                <Icon className={`w-5 h-5 ${stat.color}`} />
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

export function PrimaryChartsSection({
  formattedChannelStats,
  modelStats,
  tooltipStyle,
}: {
  formattedChannelStats: Array<{ name: string; success_rate: number; requests: number }>;
  modelStats: StatData['model_request_counts'];
  tooltipStyle: CSSProperties;
}) {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <div className="bg-card border border-border rounded-xl p-6 shadow-sm">
        <h3 className="text-base font-semibold text-foreground mb-6 flex items-center gap-2">
          <Cpu className="w-4 h-4 text-emerald-500" />
          渠道成功率 (%)
        </h3>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={formattedChannelStats} margin={{ top: 0, right: 0, left: 0, bottom: 0 }}>
              <XAxis dataKey="name" stroke={AXIS_COLOR} fontSize={12} tickLine={false} axisLine={false} />
              <YAxis stroke={AXIS_COLOR} fontSize={12} tickLine={false} axisLine={false} domain={[0, 100]} />
              <Tooltip
                cursor={{ fill: 'hsl(var(--muted) / 0.5)' }}
                contentStyle={tooltipStyle}
                itemStyle={{ color: tooltipStyle.color }}
              />
              <Bar dataKey="success_rate" name="成功率" radius={[4, 4, 0, 0]}>
                {formattedChannelStats.map((entry, index) => (
                  <Cell key={`channel-success-${entry.name}-${index}`} fill={entry.success_rate >= 95 ? SUCCESS_COLOR : entry.success_rate >= 80 ? WARNING_COLOR : ERROR_COLOR} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="bg-card border border-border rounded-xl p-6 shadow-sm">
        <h3 className="text-base font-semibold text-foreground mb-6 flex items-center gap-2">
          <Activity className="w-4 h-4 text-blue-500" />
          模型请求量分布
        </h3>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={modelStats.slice(0, 5)}
                cx="35%"
                innerRadius={50}
                outerRadius={80}
                paddingAngle={3}
                dataKey="count"
                nameKey="model"
              >
                {modelStats.slice(0, 5).map((item, index) => (
                  <Cell key={`model-distribution-${item.model}-${index}`} fill={CHART_COLORS[index % CHART_COLORS.length]} />
                ))}
              </Pie>
              <Tooltip contentStyle={tooltipStyle} itemStyle={{ color: tooltipStyle.color }} />
              <Legend
                layout="vertical"
                align="right"
                verticalAlign="middle"
                wrapperStyle={{ paddingLeft: '10px', fontSize: '12px', maxWidth: '45%' }}
                formatter={(value: string) => <span className="text-foreground truncate block max-w-[120px]" title={value}>{value}</span>}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

export function SecondaryChartsSection({
  formattedEndpointStats,
  channelStats,
  tooltipStyle,
}: {
  formattedEndpointStats: Array<{ name: string; value: number }>;
  channelStats: StatData['channel_success_rates'];
  tooltipStyle: CSSProperties;
}) {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <div className="bg-card border border-border rounded-xl p-6 shadow-sm flex flex-col">
        <h3 className="text-base font-semibold text-foreground mb-6 flex items-center gap-2">
          <Server className="w-4 h-4 text-purple-500" />
          接口访问分布 (Endpoint)
        </h3>
        <div className="flex-1 min-h-[250px]">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={formattedEndpointStats}
                outerRadius={100}
                dataKey="value"
                nameKey="name"
                label={({ name, percent }) => `${name} (${(percent * 100).toFixed(0)}%)`}
                labelLine={false}
              >
                {formattedEndpointStats.map((item, index) => (
                  <Cell key={`endpoint-distribution-${item.name}-${index}`} fill={CHART_COLORS[(index + 2) % CHART_COLORS.length]} />
                ))}
              </Pie>
              <Tooltip contentStyle={tooltipStyle} itemStyle={{ color: tooltipStyle.color }} />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="lg:col-span-2 bg-card border border-border rounded-xl shadow-sm overflow-hidden flex flex-col">
        <div className="p-6 border-b border-border bg-muted/30">
          <h3 className="text-base font-semibold text-foreground flex items-center gap-2">
            <Cpu className="w-4 h-4 text-primary" />
            渠道健康状况详细
          </h3>
        </div>
        <div className="overflow-x-auto flex-1">
          <table className="w-full text-left text-sm">
            <thead className="bg-muted text-muted-foreground font-medium">
              <tr>
                <th className="px-6 py-4">渠道名称</th>
                <th className="px-6 py-4">健康状态</th>
                <th className="px-6 py-4">请求数</th>
                <th className="px-6 py-4">成功率</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {channelStats.length === 0 ? (
                <tr>
                  <td colSpan={4} className="px-6 py-8 text-center text-muted-foreground">暂无渠道数据</td>
                </tr>
              ) : (
                channelStats.map((channel, index) => {
                  const isHealthy = channel.success_rate >= 0.95;
                  const isWarning = channel.success_rate < 0.95 && channel.success_rate >= 0.8;
                  return (
                    <tr key={`${channel.provider}-${index}`} className="hover:bg-muted/50 transition-colors">
                      <td className="px-6 py-4 font-medium text-foreground">{channel.provider}</td>
                      <td className="px-6 py-4">
                        <span className={`inline-flex items-center gap-1.5 px-2 py-1 rounded-full text-xs font-medium border ${isHealthy ? 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-500 border-emerald-500/20' : isWarning ? 'bg-amber-500/10 text-amber-600 dark:text-amber-500 border-amber-500/20' : 'bg-red-500/10 text-red-600 dark:text-red-500 border-red-500/20'}`}>
                          {isHealthy ? <CheckCircle2 className="w-3.5 h-3.5" /> : <AlertCircle className="w-3.5 h-3.5" />}
                          {isHealthy ? '良好' : isWarning ? '警告' : '异常'}
                        </span>
                      </td>
                      <td className="px-6 py-4 text-muted-foreground">{channel.total_requests.toLocaleString()}</td>
                      <td className="px-6 py-4 font-mono font-bold">
                        <span className={isHealthy ? 'text-emerald-600 dark:text-emerald-500' : isWarning ? 'text-amber-600 dark:text-amber-500' : 'text-red-600 dark:text-red-500'}>
                          {(channel.success_rate * 100).toFixed(1)}%
                        </span>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

export function UsageAnalysisPanel({
  open,
  onToggle,
  analysisStart,
  setAnalysisStart,
  analysisEnd,
  setAnalysisEnd,
  timeRangeLabel,
  availableProviders,
  analysisProviders,
  setAnalysisProviders,
  availableModels,
  analysisModels,
  setAnalysisModels,
  defaultPromptPrice,
  setDefaultPromptPrice,
  defaultCompletionPrice,
  setDefaultCompletionPrice,
  onQuery,
  analysisLoading,
  analysisQueried,
  trendLoading,
  trendData,
  trendModels,
  analysisData,
  rowPrices,
  onApplyDefaultPricesToAll,
  onUpdateRowPrice,
  analysisTotalRequests,
  analysisTotalPrompt,
  analysisTotalCompletion,
  analysisTotalTokensAll,
  analysisTotalCost,
  tooltipStyle,
  trendHint,
  defaultPriceDescription,
  savedModelPriceKeys,
  savingModelPriceKey,
  getModelPriceKey,
  onSaveModelPrice,
  onSaveAllModelPrices,
  onResetAllModelPrices,
  hasUnsavedChanges,
  unsavedCount,
}: {
  open: boolean;
  onToggle: () => void;
  analysisStart: string;
  setAnalysisStart: Dispatch<SetStateAction<string>>;
  analysisEnd: string;
  setAnalysisEnd: Dispatch<SetStateAction<string>>;
  timeRangeLabel: string;
  availableProviders: string[];
  analysisProviders: string[];
  setAnalysisProviders: Dispatch<SetStateAction<string[]>>;
  availableModels: string[];
  analysisModels: string[];
  setAnalysisModels: Dispatch<SetStateAction<string[]>>;
  defaultPromptPrice: number;
  setDefaultPromptPrice: Dispatch<SetStateAction<number>>;
  defaultCompletionPrice: number;
  setDefaultCompletionPrice: Dispatch<SetStateAction<number>>;
  onQuery: () => void;
  analysisLoading: boolean;
  analysisQueried: boolean;
  trendLoading: boolean;
  trendData: TrendPoint[];
  trendModels: string[];
  analysisData: AnalysisEntry[];
  rowPrices: Record<number, RowPrice>;
  onApplyDefaultPricesToAll: () => void;
  onUpdateRowPrice: (index: number, field: 'prompt' | 'completion', value: number) => void;
  analysisTotalRequests: number;
  analysisTotalPrompt: number;
  analysisTotalCompletion: number;
  analysisTotalTokensAll: number;
  analysisTotalCost: number;
  tooltipStyle: CSSProperties;
  trendHint?: string;
  defaultPriceDescription?: string;
  savedModelPriceKeys?: string[];
  savingModelPriceKey?: string | null;
  getModelPriceKey?: (entry: AnalysisEntry) => string;
  onSaveModelPrice?: (entry: AnalysisEntry, index: number) => void;
  onSaveAllModelPrices?: () => Promise<void> | void;
  onResetAllModelPrices?: () => void;
  hasUnsavedChanges?: boolean;
  unsavedCount?: number;
}) {
  return (
    <div className="bg-card border border-border rounded-xl shadow-sm overflow-hidden">
      <button
        onClick={onToggle}
        className="w-full p-6 flex items-center justify-between hover:bg-muted/30 transition-colors"
      >
        <h3 className="text-base font-semibold text-foreground flex items-center gap-2">
          <DollarSign className="w-4 h-4 text-amber-500" />
          用量分析与费用模拟
        </h3>
        {open ? <ChevronUp className="w-5 h-5 text-muted-foreground" /> : <ChevronDown className="w-5 h-5 text-muted-foreground" />}
      </button>

      {open && (
        <div className="px-6 pb-6 space-y-5 border-t border-border pt-5">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            <div>
              <label className="block text-xs font-medium text-muted-foreground mb-1.5">开始时间</label>
              <input
                type="datetime-local"
                value={analysisStart}
                onChange={e => setAnalysisStart(e.target.value)}
                className="w-full px-3 py-2 text-sm bg-background border border-border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary/50 text-foreground"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-muted-foreground mb-1.5">结束时间</label>
              <input
                type="datetime-local"
                value={analysisEnd}
                onChange={e => setAnalysisEnd(e.target.value)}
                className="w-full px-3 py-2 text-sm bg-background border border-border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary/50 text-foreground"
              />
            </div>
            <div className="flex items-end">
              <button
                onClick={onQuery}
                disabled={analysisLoading}
                className="px-4 py-2 text-sm font-medium bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors flex items-center justify-center gap-2 disabled:opacity-50"
              >
                {analysisLoading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Search className="w-4 h-4" />}
                查询
              </button>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <MultiSelect
              label="渠道（多选）"
              options={availableProviders}
              selected={analysisProviders}
              onChange={setAnalysisProviders}
              placeholder="全部渠道"
            />
            <MultiSelect
              label="模型（多选）"
              options={availableModels}
              selected={analysisModels}
              onChange={setAnalysisModels}
              placeholder="全部模型"
            />
          </div>

          {defaultPriceDescription && (
            <div className="rounded-lg border border-amber-500/20 bg-amber-500/5 px-4 py-3 text-xs text-muted-foreground">
              {defaultPriceDescription}
            </div>
          )}

          {analysisQueried && (
            <div className="space-y-4">
              <div className="bg-muted/30 rounded-xl p-6 border border-border">
                <h4 className="text-sm font-semibold text-foreground mb-4 flex items-center gap-2">
                  <Activity className="w-4 h-4 text-primary" />
                  所选模型请求频率趋势（按小时）
                </h4>
                {trendHint && (
                  <p className="text-xs text-muted-foreground mb-4">{trendHint}</p>
                )}
                {trendLoading ? (
                  <div className="h-64 flex items-center justify-center text-sm text-muted-foreground">
                    <RefreshCw className="w-4 h-4 animate-spin mr-2" />
                    正在加载趋势数据
                  </div>
                ) : trendData.length > 0 && trendModels.length > 0 ? (
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={trendData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--muted))" vertical={false} />
                        <XAxis
                          dataKey="hour"
                          stroke={AXIS_COLOR}
                          fontSize={10}
                          tickFormatter={(str) => {
                            const parts = String(str).split(' ');
                            return parts.length > 1 ? parts[1].slice(0, 5) : String(str);
                          }}
                        />
                        <YAxis stroke={AXIS_COLOR} fontSize={10} />
                        <Tooltip
                          contentStyle={tooltipStyle}
                          itemStyle={{ fontSize: '12px' }}
                          labelStyle={{ fontSize: '12px', fontWeight: 'bold', marginBottom: '4px' }}
                        />
                        <Legend iconType="circle" wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }} />
                        {trendModels.map((model, index) => (
                          <Line
                            key={model}
                            type="monotone"
                            dataKey={model}
                            name={model}
                            stroke={LINE_COLORS[index % LINE_COLORS.length]}
                            strokeWidth={2}
                            dot={false}
                            connectNulls
                            activeDot={{ r: 4 }}
                          />
                        ))}
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                ) : (
                  <div className="h-64 flex items-center justify-center text-sm text-muted-foreground">
                    当前筛选条件下暂无趋势数据
                  </div>
                )}
              </div>

              {analysisData.length > 0 && (
                <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                  <div className="bg-muted/50 rounded-lg p-3 text-center">
                    <p className="text-xs text-muted-foreground">总请求次数</p>
                    <p className="text-lg font-bold text-foreground mt-1">{formatNumber(analysisTotalRequests)}</p>
                  </div>
                  <div className="bg-muted/50 rounded-lg p-3 text-center">
                    <p className="text-xs text-muted-foreground">输入 Token</p>
                    <p className="text-lg font-bold text-foreground mt-1">{formatTokens(analysisTotalPrompt)}</p>
                  </div>
                  <div className="bg-muted/50 rounded-lg p-3 text-center">
                    <p className="text-xs text-muted-foreground">输出 Token</p>
                    <p className="text-lg font-bold text-foreground mt-1">{formatTokens(analysisTotalCompletion)}</p>
                  </div>
                  <div className="bg-muted/50 rounded-lg p-3 text-center">
                    <p className="text-xs text-muted-foreground">总 Token</p>
                    <p className="text-lg font-bold text-foreground mt-1">{formatTokens(analysisTotalTokensAll)}</p>
                  </div>
                  <div className="bg-amber-500/10 rounded-lg p-3 text-center border border-amber-500/20">
                    <p className="text-xs text-amber-600 dark:text-amber-400">模拟总费用</p>
                    <p className="text-lg font-bold text-amber-600 dark:text-amber-400 mt-1">{formatCost(analysisTotalCost)}</p>
                  </div>
                </div>
              )}

              {analysisData.length > 0 && (
                <div className="flex items-center gap-3 flex-wrap">
                  <div className="flex items-center gap-1.5">
                    <label className="text-xs font-medium text-muted-foreground whitespace-nowrap">默认输入价格 ($/M)</label>
                    <input
                      type="number"
                      step="0.01"
                      min="0"
                      value={defaultPromptPrice}
                      onChange={e => setDefaultPromptPrice(parseFloat(e.target.value) || 0)}
                      className="w-24 px-2 py-1.5 text-sm bg-background border border-border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary/50 text-foreground"
                    />
                  </div>
                  <div className="flex items-center gap-1.5">
                    <label className="text-xs font-medium text-muted-foreground whitespace-nowrap">默认输出价格 ($/M)</label>
                    <input
                      type="number"
                      step="0.01"
                      min="0"
                      value={defaultCompletionPrice}
                      onChange={e => setDefaultCompletionPrice(parseFloat(e.target.value) || 0)}
                      className="w-24 px-2 py-1.5 text-sm bg-background border border-border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary/50 text-foreground"
                    />
                  </div>

                  <div className="h-5 w-px bg-border mx-1" />

                  {onSaveAllModelPrices && (
                    <button
                      onClick={onSaveAllModelPrices}
                      disabled={!hasUnsavedChanges}
                      className="px-3 py-1.5 text-xs font-medium bg-primary hover:bg-primary/90 text-primary-foreground rounded-lg transition-colors flex items-center gap-1.5 disabled:opacity-50"
                    >
                      <Save className="w-3.5 h-3.5" />
                      保存更改{(unsavedCount ?? 0) > 0 ? ` (${unsavedCount})` : ''}
                    </button>
                  )}
                  {onResetAllModelPrices && (
                    <button
                      onClick={onResetAllModelPrices}
                      disabled={!hasUnsavedChanges}
                      className="px-3 py-1.5 text-xs font-medium bg-muted hover:bg-muted/80 text-foreground border border-border rounded-lg transition-colors flex items-center gap-1.5 disabled:opacity-50"
                    >
                      <RotateCcw className="w-3.5 h-3.5" />
                      重置
                    </button>
                  )}
                  <button
                    onClick={onApplyDefaultPricesToAll}
                    className="px-3 py-1.5 text-xs font-medium bg-muted hover:bg-muted/80 text-foreground border border-border rounded-lg transition-colors"
                  >
                    将默认价格应用到所有行
                  </button>
                </div>
              )}

              <div className="overflow-x-auto border border-border rounded-lg">
                <table className="w-full text-left text-sm">
                  <thead className="bg-muted text-muted-foreground font-medium">
                    <tr>
                      <th className="px-4 py-3">渠道</th>
                      <th className="px-4 py-3">模型</th>
                      <th className="px-4 py-3 text-right">请求次数</th>
                      <th className="px-4 py-3 text-right">输入 Token</th>
                      <th className="px-4 py-3 text-right">输出 Token</th>
                      <th className="px-4 py-3 text-center">输入价格 ($/M)</th>
                      <th className="px-4 py-3 text-center">输出价格 ($/M)</th>
                      <th className="px-4 py-3 text-right">模拟费用</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border">
                    {analysisData.length === 0 ? (
                      <tr>
                        <td colSpan={8} className="px-4 py-8 text-center text-muted-foreground">
                          {analysisLoading ? '查询中...' : '暂无数据'}
                        </td>
                      </tr>
                    ) : (
                      analysisData.map((entry, index) => {
                        const price = rowPrices[index] || { prompt: defaultPromptPrice, completion: defaultCompletionPrice };
                        const rowCost = (entry.total_prompt_tokens * price.prompt + entry.total_completion_tokens * price.completion) / 1_000_000;
                        return (
                          <tr key={`${entry.provider}-${entry.model}-${index}`} className="hover:bg-muted/50 transition-colors">
                            <td className="px-4 py-3 font-medium text-foreground">{entry.provider}</td>
                            <td className="px-4 py-3 text-foreground font-mono text-xs">{entry.model}</td>
                            <td className="px-4 py-3 text-right text-muted-foreground">{formatNumber(entry.request_count)}</td>
                            <td className="px-4 py-3 text-right text-muted-foreground">{formatNumber(entry.total_prompt_tokens)}</td>
                            <td className="px-4 py-3 text-right text-muted-foreground">{formatNumber(entry.total_completion_tokens)}</td>
                            <td className="px-2 py-1 text-center">
                              <input
                                type="number"
                                step="0.01"
                                min="0"
                                value={price.prompt}
                                onChange={e => onUpdateRowPrice(index, 'prompt', parseFloat(e.target.value) || 0)}
                                className="w-20 px-2 py-1 text-xs text-center bg-background border border-border rounded focus:outline-none focus:ring-1 focus:ring-primary/50 text-foreground"
                              />
                            </td>
                            <td className="px-2 py-1 text-center">
                              <input
                                type="number"
                                step="0.01"
                                min="0"
                                value={price.completion}
                                onChange={e => onUpdateRowPrice(index, 'completion', parseFloat(e.target.value) || 0)}
                                className="w-20 px-2 py-1 text-xs text-center bg-background border border-border rounded focus:outline-none focus:ring-1 focus:ring-primary/50 text-foreground"
                              />
                            </td>
                            <td className="px-4 py-3 text-right font-mono font-bold text-amber-600 dark:text-amber-400">{formatCost(rowCost)}</td>
                          </tr>
                        );
                      })
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
