import { useEffect, useState } from 'react';
import {
  Activity, Cpu, Zap, BarChart3, AlertCircle, CheckCircle2,
  RefreshCw, Server
} from 'lucide-react';
import { useAuthStore } from '../store/authStore';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
  PieChart, Pie, Legend
} from 'recharts';

interface StatData {
  time_range: string;
  channel_success_rates: { provider: string; success_rate: number; total_requests: number }[];
  model_request_counts: { model: string; count: number }[];
  endpoint_request_counts: { endpoint: string; count: number }[];
}

interface UsageAnalysisEntry {
  provider: string;
  model: string;
  request_count: number;
  total_prompt_tokens: number;
  total_completion_tokens: number;
  total_tokens: number;
}

interface UsageAnalysisResponse {
  usage: UsageAnalysisEntry[];
  summary: {
    provider_count: number;
    model_count: number;
    request_count: number;
    total_prompt_tokens: number;
    total_completion_tokens: number;
    total_tokens: number;
  };
  query_details: {
    start_datetime?: string | null;
    end_datetime?: string | null;
    provider_filter?: string | null;
    model_filter?: string | null;
  };
}

const TIME_RANGES = [
  { label: '1 小时', value: 1 },
  { label: '6 小时', value: 6 },
  { label: '24 小时', value: 24 },
  { label: '7 天', value: 168 },
  { label: '30 天', value: 720 }
];

const CHART_COLORS = [
  'hsl(var(--primary))',
  'hsl(var(--ring))',
  'hsl(160 84% 39%)',
  'hsl(38 92% 50%)',
  'hsl(var(--destructive))',
  'hsl(var(--secondary-foreground))'
];

const AXIS_COLOR = 'hsl(var(--muted-foreground))';
const SUCCESS_COLOR = 'hsl(160 84% 39%)';
const WARNING_COLOR = 'hsl(38 92% 50%)';
const ERROR_COLOR = 'hsl(var(--destructive))';

export default function Dashboard() {
  const [stats, setStats] = useState<StatData | null>(null);
  const [usageAnalysis, setUsageAnalysis] = useState<UsageAnalysisResponse | null>(null);
  const [totalTokens, setTotalTokens] = useState(0);
  const [loading, setLoading] = useState(true);
  const [timeRange, setTimeRange] = useState(24);
  const [usageProviderFilter, setUsageProviderFilter] = useState('all');
  const [usageModelFilter, setUsageModelFilter] = useState('all');
  const { token } = useAuthStore();

  const tooltipStyle = {
    backgroundColor: 'hsl(var(--popover))',
    borderColor: 'hsl(var(--border))',
    color: 'hsl(var(--popover-foreground))',
    borderRadius: '8px'
  };

  const formatNumber = (value: number) => (value || 0).toLocaleString();

  const fetchData = async () => {
    if (!token) return;
    setLoading(true);
    try {
      const headers = { Authorization: `Bearer ${token}` };
      const [statsRes, usageRes] = await Promise.all([
        fetch(`/v1/stats?hours=${timeRange}`, { headers }),
        fetch(`/v1/stats/usage_analysis?hours=${timeRange}`, { headers })
      ]);

      if (statsRes.ok) {
        const data = await statsRes.json();
        setStats(data.stats || data);
      }

      if (usageRes.ok) {
        const data: UsageAnalysisResponse = await usageRes.json();
        setUsageAnalysis(data);
        setTotalTokens(data.summary?.total_tokens || 0);
      } else {
        setUsageAnalysis(null);
        setTotalTokens(0);
      }
    } catch (err) {
      console.error('Failed to fetch stats:', err);
      setUsageAnalysis(null);
      setTotalTokens(0);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, [token, timeRange]);

  const channelStats = stats?.channel_success_rates || [];
  const modelStats = stats?.model_request_counts || [];
  const endpointStats = stats?.endpoint_request_counts || [];
  const usageRows = usageAnalysis?.usage || [];

  const providerOptions = Array.from(new Set(usageRows.map(item => item.provider).filter(Boolean))).sort((a, b) => a.localeCompare(b));
  const modelOptions = Array.from(new Set(
    usageRows
      .filter(item => usageProviderFilter === 'all' || item.provider === usageProviderFilter)
      .map(item => item.model)
      .filter(Boolean)
  )).sort((a, b) => a.localeCompare(b));

  useEffect(() => {
    if (usageProviderFilter !== 'all' && !providerOptions.includes(usageProviderFilter)) {
      setUsageProviderFilter('all');
    }
  }, [usageProviderFilter, usageAnalysis]);

  useEffect(() => {
    if (usageModelFilter !== 'all' && !modelOptions.includes(usageModelFilter)) {
      setUsageModelFilter('all');
    }
  }, [usageModelFilter, usageProviderFilter, usageAnalysis]);

  const filteredUsageRows = usageRows.filter(item => {
    const providerMatched = usageProviderFilter === 'all' || item.provider === usageProviderFilter;
    const modelMatched = usageModelFilter === 'all' || item.model === usageModelFilter;
    return providerMatched && modelMatched;
  });

  const usageSummary = filteredUsageRows.reduce((acc, item) => ({
    request_count: acc.request_count + (item.request_count || 0),
    total_prompt_tokens: acc.total_prompt_tokens + (item.total_prompt_tokens || 0),
    total_completion_tokens: acc.total_completion_tokens + (item.total_completion_tokens || 0),
    total_tokens: acc.total_tokens + (item.total_tokens || 0),
  }), {
    request_count: 0,
    total_prompt_tokens: 0,
    total_completion_tokens: 0,
    total_tokens: 0,
  });

  const filteredProviderCount = new Set(filteredUsageRows.map(item => item.provider).filter(Boolean)).size;
  const filteredModelCount = new Set(filteredUsageRows.map(item => item.model).filter(Boolean)).size;

  const totalRequests = channelStats.reduce((sum, item) => sum + item.total_requests, 0) || 0;
  const avgSuccessRate = totalRequests > 0
    ? channelStats.reduce((sum, item) => sum + item.success_rate * item.total_requests, 0) / totalRequests
    : 0;
  const activeChannels = channelStats.length || 0;

  const timeRangeLabel = TIME_RANGES.find(r => r.value === timeRange)?.label ?? `${timeRange} 小时`;

  const topCards = [
    { label: '总请求量', value: formatNumber(totalRequests), icon: Zap, color: 'text-amber-500', bg: 'bg-amber-500/10' },
    { label: `Token 消耗 (${timeRangeLabel})`, value: formatNumber(totalTokens), icon: BarChart3, color: 'text-blue-500', bg: 'bg-blue-500/10' },
    { label: '平均成功率', value: `${(avgSuccessRate * 100).toFixed(1)}%`, icon: CheckCircle2, color: 'text-emerald-500', bg: 'bg-emerald-500/10' },
    { label: '活跃渠道', value: activeChannels.toString(), icon: Cpu, color: 'text-purple-500', bg: 'bg-purple-500/10' },
  ];

  const formattedEndpointStats = endpointStats.slice(0, 5).map(item => ({
    name: item.endpoint.replace('POST ', '').replace('GET ', ''),
    value: item.count
  }));

  const formattedChannelStats = channelStats.slice(0, 6).map(item => ({
    name: item.provider,
    success_rate: item.success_rate * 100,
    requests: item.total_requests
  }));

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
                onClick={() => setTimeRange(range.value)}
                className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all ${timeRange === range.value
                  ? 'bg-primary text-primary-foreground shadow-sm'
                  : 'text-muted-foreground hover:text-foreground hover:bg-muted/50'
                  }`}
              >
                {range.label}
              </button>
            ))}
          </div>
          <button onClick={fetchData} className="p-2 text-muted-foreground hover:text-foreground bg-card border border-border rounded-lg transition-colors">
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {topCards.map((stat, i) => (
          <div key={i} className="bg-card border border-border p-6 rounded-xl shadow-sm">
            <div className="flex justify-between items-start">
              <div>
                <p className="text-sm font-medium text-muted-foreground">{stat.label}</p>
                <h3 className="text-3xl font-bold text-foreground mt-2">{stat.value}</h3>
              </div>
              <div className={`p-2 rounded-lg ${stat.bg}`}>
                <stat.icon className={`w-5 h-5 ${stat.color}`} />
              </div>
            </div>
          </div>
        ))}
      </div>

      <section className="bg-card border border-border rounded-xl shadow-sm overflow-hidden">
        <div className="p-6 border-b border-border bg-muted/30 flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <h3 className="text-base font-semibold text-foreground flex items-center gap-2">
              <BarChart3 className="w-4 h-4 text-blue-500" />
              用量分析
            </h3>
            <p className="text-sm text-muted-foreground mt-1">按渠道与模型查看 {timeRangeLabel} 内的请求量与 Token 消耗。</p>
          </div>

          <div className="flex flex-col sm:flex-row gap-3 w-full lg:w-auto">
            <select
              value={usageProviderFilter}
              onChange={(e) => {
                setUsageProviderFilter(e.target.value);
                setUsageModelFilter('all');
              }}
              className="w-full sm:w-52 bg-background border border-border px-3 py-2 rounded-lg text-sm text-foreground"
            >
              <option value="all">全部渠道</option>
              {providerOptions.map(provider => (
                <option key={provider} value={provider}>{provider}</option>
              ))}
            </select>
            <select
              value={usageModelFilter}
              onChange={(e) => setUsageModelFilter(e.target.value)}
              className="w-full sm:w-64 bg-background border border-border px-3 py-2 rounded-lg text-sm text-foreground"
            >
              <option value="all">全部模型</option>
              {modelOptions.map(model => (
                <option key={model} value={model}>{model}</option>
              ))}
            </select>
          </div>
        </div>

        <div className="p-6 space-y-6">
          <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-5 gap-4">
            {[
              { label: '渠道数', value: formatNumber(filteredProviderCount) },
              { label: '模型数', value: formatNumber(filteredModelCount) },
              { label: '请求数', value: formatNumber(usageSummary.request_count) },
              { label: '输入 Token', value: formatNumber(usageSummary.total_prompt_tokens) },
              { label: '输出 Token', value: formatNumber(usageSummary.total_completion_tokens) },
            ].map((item) => (
              <div key={item.label} className="bg-background border border-border rounded-lg p-4">
                <p className="text-sm text-muted-foreground">{item.label}</p>
                <p className="text-2xl font-semibold text-foreground mt-2">{item.value}</p>
              </div>
            ))}
          </div>

          <div className="bg-background border border-border rounded-lg p-4">
            <p className="text-sm text-muted-foreground">总 Token</p>
            <p className="text-3xl font-bold text-foreground mt-2">{formatNumber(usageSummary.total_tokens)}</p>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="bg-muted text-muted-foreground font-medium">
                <tr>
                  <th className="px-4 py-3">渠道</th>
                  <th className="px-4 py-3">模型</th>
                  <th className="px-4 py-3 text-right">请求数</th>
                  <th className="px-4 py-3 text-right">输入 Token</th>
                  <th className="px-4 py-3 text-right">输出 Token</th>
                  <th className="px-4 py-3 text-right">总 Token</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {filteredUsageRows.length === 0 ? (
                  <tr>
                    <td colSpan={6} className="px-4 py-10 text-center text-muted-foreground">当前筛选条件下暂无用量数据</td>
                  </tr>
                ) : (
                  filteredUsageRows.map((item) => (
                    <tr key={`${item.provider}-${item.model}`} className="hover:bg-muted/50 transition-colors">
                      <td className="px-4 py-3 font-medium text-foreground">{item.provider}</td>
                      <td className="px-4 py-3 text-foreground max-w-[320px] truncate" title={item.model}>{item.model}</td>
                      <td className="px-4 py-3 text-right font-mono text-muted-foreground">{formatNumber(item.request_count)}</td>
                      <td className="px-4 py-3 text-right font-mono text-muted-foreground">{formatNumber(item.total_prompt_tokens)}</td>
                      <td className="px-4 py-3 text-right font-mono text-muted-foreground">{formatNumber(item.total_completion_tokens)}</td>
                      <td className="px-4 py-3 text-right font-mono font-semibold text-foreground">{formatNumber(item.total_tokens)}</td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      </section>

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
                    <Cell key={`cell-${index}`} fill={entry.success_rate >= 95 ? SUCCESS_COLOR : entry.success_rate >= 80 ? WARNING_COLOR : ERROR_COLOR} />
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
                  {modelStats.slice(0, 5).map((_, index) => (
                    <Cell key={`cell-${index}`} fill={CHART_COLORS[index % CHART_COLORS.length]} />
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
                  {formattedEndpointStats.map((_, index) => (
                    <Cell key={`cell-${index}`} fill={CHART_COLORS[(index + 2) % CHART_COLORS.length]} />
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
                  channelStats.map((channel, i) => {
                    const isHealthy = channel.success_rate >= 0.95;
                    const isWarning = channel.success_rate < 0.95 && channel.success_rate >= 0.8;
                    return (
                      <tr key={i} className="hover:bg-muted/50 transition-colors">
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
    </div>
  );
}
