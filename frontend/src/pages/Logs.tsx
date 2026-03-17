import { useEffect, useState } from 'react';
import { useAuthStore } from '../store/authStore';
import { apiFetch } from '../lib/api';
import {
  RefreshCw,
  Filter,
  ChevronDown,
  ChevronRight,
  FileText,
  Clock,
  ArrowDownToLine,
  CheckCircle2,
  XCircle,
  Globe,
  Key,
  Server,
  RotateCcw,
  Eye,
  EyeOff,
  Flag,
  Users,
  Zap,
  AlertTriangle,
  X,
} from 'lucide-react';

interface LogEntry {
  id: number;
  timestamp: string;
  endpoint?: string;
  client_ip?: string;
  provider?: string;
  model?: string;
  api_key_prefix?: string;
  process_time?: number;
  first_response_time?: number;
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
  success: boolean;
  status_code?: number;
  is_flagged: boolean;
  provider_id?: string;
  provider_key_index?: number;
  api_key_name?: string;
  api_key_group?: string;
  retry_count?: number;
  retry_path?: string;
  request_headers?: string;
  request_body?: string;
  upstream_request_body?: string;
  upstream_response_body?: string;
  response_body?: string;
  raw_data_expires_at?: string;
}

interface QuickOption {
  label: string;
  value: string;
}

function normalizeText(value: unknown): string {
  return String(value ?? '').trim();
}

function sortStrings(values: string[]) {
  return [...values].sort((a, b) => a.localeCompare(b, 'zh-CN'));
}

function mergeStringOptions(current: string[], incoming: string[]) {
  const set = new Set(current);
  incoming.forEach(item => {
    const normalized = normalizeText(item);
    if (normalized) set.add(normalized);
  });
  return sortStrings(Array.from(set));
}

function mergeQuickOptions(current: QuickOption[], incoming: QuickOption[]) {
  const map = new Map<string, QuickOption>();
  current.forEach(item => {
    const key = normalizeText(item.value);
    if (key) map.set(key, { label: item.label, value: key });
  });
  incoming.forEach(item => {
    const key = normalizeText(item.value);
    if (!key) return;
    if (!map.has(key)) {
      map.set(key, { label: normalizeText(item.label) || key, value: key });
    }
  });
  return Array.from(map.values()).sort((a, b) => a.label.localeCompare(b.label, 'zh-CN'));
}

function maskApiKey(apiKey: string) {
  const text = normalizeText(apiKey);
  if (!text) return '';
  if (text.length <= 12) return text;
  return `${text.slice(0, 7)}...${text.slice(-4)}`;
}

function getApiKeySearchValue(apiKey: unknown) {
  const text = normalizeText(apiKey);
  if (!text || text === '-') return '';
  const maskedHead = text.split('...')[0]?.trim();
  if (maskedHead && maskedHead !== '-') return maskedHead;
  return text.length > 7 ? text.slice(0, 7) : text;
}

function collectProviderModels(provider: any) {
  const values: string[] = [];
  const rawModels = Array.isArray(provider?.model)
    ? provider.model
    : Array.isArray(provider?.models)
      ? provider.models
      : [];

  rawModels.forEach((item: any) => {
    if (typeof item === 'string') {
      values.push(item);
      return;
    }
    if (item && typeof item === 'object') {
      Object.entries(item).forEach(([upstream, alias]) => {
        values.push(upstream);
        if (typeof alias === 'string') values.push(alias);
      });
    }
  });

  return values.map(normalizeText).filter(Boolean);
}

function extractConfigQuickOptions(config: any) {
  const providers = Array.isArray(config?.providers) ? config.providers : [];
  const apiKeys = Array.isArray(config?.api_keys) ? config.api_keys : [];

  const modelValues = new Set<string>();
  const providerOptions: QuickOption[] = [];
  const apiKeyOptions: QuickOption[] = [];

  providers.forEach((provider: any) => {
    const providerName = normalizeText(provider?.provider);
    const providerLabel = normalizeText(provider?.provider || provider?.name || provider?.id);
    if (providerName) {
      providerOptions.push({ label: providerLabel || providerName, value: providerName });
    }

    collectProviderModels(provider).forEach(model => modelValues.add(model));
  });

  apiKeys.forEach((item: any) => {
    if (typeof item === 'string') {
      const searchValue = getApiKeySearchValue(item);
      if (searchValue) {
        apiKeyOptions.push({
          label: maskApiKey(item),
          value: searchValue,
        });
      }
      return;
    }

    if (!item || typeof item !== 'object') return;

    const name = normalizeText(item.name || item.preferences?.name);
    const primaryGroup = normalizeText(
      item.group
      || (Array.isArray(item.groups) ? item.groups[0] : '')
      || item.preferences?.group
    );
    const api = normalizeText(item.api || item.key);
    const masked = maskApiKey(api);
    const searchValue = name || getApiKeySearchValue(api) || primaryGroup;

    if (!searchValue) return;

    const labelParts = [name || masked || searchValue];
    if (primaryGroup) labelParts.push(`分组: ${primaryGroup}`);
    if (masked && name) labelParts.push(masked);

    apiKeyOptions.push({
      label: labelParts.join(' · '),
      value: searchValue,
    });
  });

  return {
    models: sortStrings(Array.from(modelValues)),
    providers: mergeQuickOptions([], providerOptions),
    apiKeys: mergeQuickOptions([], apiKeyOptions),
  };
}

function extractLogQuickOptions(items: LogEntry[]) {
  const models = items.map(item => normalizeText(item.model)).filter(Boolean);
  const providers: QuickOption[] = [];
  const apiKeys: QuickOption[] = [];

  items.forEach(item => {
    const provider = normalizeText(item.provider);
    const providerId = normalizeText(item.provider_id);
    if (provider) providers.push({ label: provider, value: provider });
    if (providerId && providerId !== provider) {
      providers.push({ label: `${providerId}（渠道 ID）`, value: providerId });
    }

    const apiKeyName = normalizeText(item.api_key_name);
    const apiKeyGroup = normalizeText(item.api_key_group);
    const apiKeyPrefix = normalizeText(item.api_key_prefix);

    if (apiKeyName) {
      const labelParts = [apiKeyName];
      if (apiKeyGroup) labelParts.push(`分组: ${apiKeyGroup}`);
      if (apiKeyPrefix) labelParts.push(apiKeyPrefix);
      apiKeys.push({
        label: labelParts.join(' · '),
        value: apiKeyName,
      });
      return;
    }

    const searchValue = getApiKeySearchValue(apiKeyPrefix);
    if (searchValue) {
      apiKeys.push({
        label: apiKeyPrefix || searchValue,
        value: searchValue,
      });
    }
  });

  return {
    models,
    providers,
    apiKeys,
  };
}

function QuickFilterField({
  label,
  placeholder,
  value,
  onChange,
  options,
}: {
  label: string;
  placeholder: string;
  value: string;
  onChange: (value: string) => void;
  options: QuickOption[];
}) {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-2">
        <span className="text-xs font-medium text-muted-foreground">{label}</span>
        {value && (
          <button
            type="button"
            onClick={() => onChange('')}
            className="inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
          >
            <X className="w-3.5 h-3.5" /> 清空
          </button>
        )}
      </div>

      <input
        type="text"
        placeholder={placeholder}
        value={value}
        onChange={e => onChange(e.target.value)}
        className="w-full bg-background border border-border text-sm px-3 py-2 rounded-lg text-foreground"
      />

      <select
        value=""
        onChange={e => {
          const nextValue = e.target.value;
          if (!nextValue) return;
          onChange(nextValue);
        }}
        className="w-full bg-background border border-border text-xs px-3 py-2 rounded-lg text-muted-foreground"
      >
        <option value="">快速选择现有配置</option>
        {options.map(option => (
          <option key={`${label}-${option.value}`} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </div>
  );
}

export default function Logs() {
  const { token } = useAuthStore();
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const [hasMore, setHasMore] = useState(true);
  const [totalCount, setTotalCount] = useState(0);

  const [page, setPage] = useState(1);
  const [pageSize] = useState(50);

  const [filterModel, setFilterModel] = useState('');
  const [filterProvider, setFilterProvider] = useState('');
  const [filterApiKey, setFilterApiKey] = useState('');
  const [filterSuccess, setFilterSuccess] = useState<string>('ALL');
  const [showFilters, setShowFilters] = useState(false);

  const [quickModelOptions, setQuickModelOptions] = useState<string[]>([]);
  const [quickProviderOptions, setQuickProviderOptions] = useState<QuickOption[]>([]);
  const [quickApiKeyOptions, setQuickApiKeyOptions] = useState<QuickOption[]>([]);

  const [expandedIds, setExpandedIds] = useState<Set<number>>(new Set());

  const mergeObservedOptions = (items: LogEntry[]) => {
    const observed = extractLogQuickOptions(items);
    setQuickModelOptions(prev => mergeStringOptions(prev, observed.models));
    setQuickProviderOptions(prev => mergeQuickOptions(prev, observed.providers));
    setQuickApiKeyOptions(prev => mergeQuickOptions(prev, observed.apiKeys));
  };

  const fetchFilterOptions = async () => {
    if (!token) return;

    try {
      const res = await apiFetch('/v1/api_config');
      if (!res.ok) return;

      const data = await res.json();
      const config = data.api_config || data;
      const extracted = extractConfigQuickOptions(config);
      setQuickModelOptions(extracted.models);
      setQuickProviderOptions(extracted.providers);
      setQuickApiKeyOptions(extracted.apiKeys);
    } catch (err) {
      console.error('Failed to load quick filter options:', err);
    }
  };

  const fetchLogs = async (resetPage = false) => {
    if (!token) return;
    setLoading(true);

    const currentPage = resetPage ? 1 : page;
    if (resetPage) {
      setPage(1);
      setExpandedIds(new Set());
    }

    try {
      const queryParams = new URLSearchParams({
        page: currentPage.toString(),
        page_size: pageSize.toString(),
      });

      if (filterModel.trim()) queryParams.append('model', filterModel.trim());
      if (filterProvider.trim()) queryParams.append('provider', filterProvider.trim());
      if (filterApiKey.trim()) queryParams.append('api_key', filterApiKey.trim());
      if (filterSuccess === 'SUCCESS') queryParams.append('success', 'true');
      if (filterSuccess === 'FAILED') queryParams.append('success', 'false');

      const res = await apiFetch(`/v1/logs?${queryParams.toString()}`);
      if (!res.ok) {
        throw new Error(`Failed to fetch logs: ${res.status}`);
      }

      const data = await res.json();
      const fetchedLogs: LogEntry[] = data.items || [];
      mergeObservedOptions(fetchedLogs);

      if (resetPage) {
        setLogs(fetchedLogs);
      } else {
        setLogs(prev => {
          const seen = new Set(prev.map(item => item.id));
          const appended = fetchedLogs.filter(item => !seen.has(item.id));
          return [...prev, ...appended];
        });
      }

      const total = data.total || 0;
      setTotalCount(total);
      setHasMore(currentPage * pageSize < total);
    } catch (err) {
      console.error('Failed to fetch logs:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchFilterOptions();
  }, [token]);

  useEffect(() => {
    fetchLogs(true);
  }, [token, filterModel, filterProvider, filterApiKey, filterSuccess]);

  const loadMore = () => {
    setPage(prev => prev + 1);
  };

  useEffect(() => {
    if (page > 1) {
      fetchLogs(false);
    }
  }, [page]);

  const toggleExpand = (id: number) => {
    setExpandedIds(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const getStatusColor = (success: boolean, code?: number) => {
    if (success) return 'text-emerald-600 dark:text-emerald-400 bg-emerald-500/10 border-emerald-500/20';
    if (code && code >= 400 && code < 500) return 'text-yellow-600 dark:text-yellow-500 bg-yellow-500/10 border-yellow-500/20';
    return 'text-red-600 dark:text-red-500 bg-red-500/10 border-red-500/20';
  };

  const calculateSpeed = (log: LogEntry) => {
    if (!log.completion_tokens || !log.process_time) return null;

    const startTime = log.first_response_time || 0;
    const genTime = log.process_time - startTime;
    if (genTime <= 0) return null;

    const speed = log.completion_tokens / genTime;
    let color = 'text-muted-foreground';
    if (speed >= 80) color = 'text-purple-600 dark:text-purple-400';
    else if (speed >= 40) color = 'text-emerald-600 dark:text-emerald-400';
    else if (speed < 15) color = 'text-yellow-600 dark:text-yellow-500';

    return { speed: speed.toFixed(1), color };
  };

  const formatTimestamp = (ts: string) => {
    try {
      return new Date(ts).toLocaleString('zh-CN', {
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
      });
    } catch {
      return ts;
    }
  };

  const formatFullTimestamp = (ts: string) => {
    try {
      return new Date(ts).toLocaleString('zh-CN');
    } catch {
      return ts;
    }
  };

  const formatJsonBestEffort = (raw: string) => {
    const input = String(raw ?? '').trim();
    if (!input) return { formatted: '', isJson: false };

    try {
      let parsed: any = JSON.parse(input);
      if (typeof parsed === 'string') {
        const inner = parsed.trim();
        try {
          parsed = JSON.parse(inner);
        } catch {
          return { formatted: parsed, isJson: false };
        }
      }

      if (parsed === null) return { formatted: 'null', isJson: true };
      if (typeof parsed === 'object') return { formatted: JSON.stringify(parsed, null, 2), isJson: true };
      return { formatted: String(parsed), isJson: false };
    } catch {
      return { formatted: raw, isJson: false };
    }
  };

  const getHttpCodeColor = (code?: number | null) => {
    if (code == null) return 'text-muted-foreground bg-muted/30 border-border';
    if (code >= 200 && code < 300) return 'text-emerald-600 dark:text-emerald-400 bg-emerald-500/10 border-emerald-500/20';
    if (code >= 400 && code < 500) return 'text-yellow-600 dark:text-yellow-500 bg-yellow-500/10 border-yellow-500/20';
    return 'text-red-600 dark:text-red-500 bg-red-500/10 border-red-500/20';
  };

  type RetryHop = { provider?: string; status_code?: number | null; error?: string };

  const RetryPathView = ({ retryPathJson }: { retryPathJson: string }) => {
    const [openIndex, setOpenIndex] = useState<number | null>(null);

    let items: RetryHop[] | null = null;
    try {
      const parsed = JSON.parse(retryPathJson);
      if (Array.isArray(parsed)) items = parsed;
    } catch {
      items = null;
    }

    if (!items) {
      return (
        <pre className="bg-background border border-border p-3 rounded-lg text-xs font-mono text-foreground overflow-x-auto whitespace-pre-wrap">
          {retryPathJson}
        </pre>
      );
    }

    if (items.length === 0) {
      return <div className="text-sm text-muted-foreground">无重试记录</div>;
    }

    return (
      <div className="space-y-2">
        {items.map((hop, idx) => {
          const provider = hop.provider || '-';
          const code = hop.status_code;
          const error = hop.error || '';
          const isOpen = openIndex === idx;
          const preview = error.length > 120 ? `${error.slice(0, 120)}...` : error;

          return (
            <div key={`${provider}-${idx}`} className="border border-border rounded-lg overflow-hidden bg-background">
              <button
                type="button"
                onClick={() => setOpenIndex(prev => (prev === idx ? null : idx))}
                className="w-full text-left px-3 py-2 flex items-start gap-2 hover:bg-muted/50 transition-colors"
              >
                <div className="flex-shrink-0 text-xs font-mono text-muted-foreground mt-0.5">
                  #{idx + 1}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="text-sm font-medium text-foreground truncate" title={provider}>{provider}</span>
                    <span className={`px-2 py-0.5 rounded text-xs font-mono border ${getHttpCodeColor(code)}`}>
                      {code ?? '-'}
                    </span>
                    {error && (
                      <span className="text-xs text-muted-foreground flex items-center gap-1">
                        <AlertTriangle className="w-3.5 h-3.5" />
                        <span className="truncate max-w-[520px]">{isOpen ? '展开查看错误详情' : preview}</span>
                      </span>
                    )}
                  </div>
                </div>
                <div className="flex-shrink-0 text-muted-foreground mt-0.5">
                  {isOpen ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
                </div>
              </button>
              {isOpen && error && (
                <pre className="border-t border-border p-3 text-xs font-mono text-foreground whitespace-pre-wrap max-h-72 overflow-y-auto">
                  {error}
                </pre>
              )}
            </div>
          );
        })}
      </div>
    );
  };

  const LogAccordionItem = ({ log }: { log: LogEntry }) => {
    const isExpanded = expandedIds.has(log.id);
    const speedInfo = calculateSpeed(log);

    return (
      <div className="bg-card border border-border rounded-xl overflow-hidden">
        <div className="cursor-pointer hover:bg-muted/50 transition-colors" onClick={() => toggleExpand(log.id)}>
          <div className="flex items-center gap-2 sm:gap-3 p-3 sm:p-4">
            <div className="flex-shrink-0 text-muted-foreground">
              {isExpanded ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
            </div>

            <div className="flex-shrink-0">
              {log.success ? (
                <CheckCircle2 className="w-5 h-5 text-emerald-500" />
              ) : (
                <XCircle className="w-5 h-5 text-red-500" />
              )}
            </div>

            <div className="flex-shrink-0 text-xs sm:text-sm font-mono text-muted-foreground w-[85px] sm:w-[100px]">
              {formatTimestamp(log.timestamp)}
            </div>

            <div className="flex-1 min-w-0">
              <div className="font-medium text-foreground text-sm truncate" title={log.model || '-'}>
                {log.model || '-'}
              </div>
              <div className="text-xs text-muted-foreground truncate">
                {log.provider || '未知'}
                {log.provider_key_index !== undefined && <span className="opacity-60"> [{log.provider_key_index}]</span>}
              </div>
            </div>

            <div className="hidden sm:flex items-center gap-1.5 flex-shrink-0">
              {log.is_flagged && (
                <span className="text-yellow-500" title="已标记">
                  <Flag className="w-4 h-4" />
                </span>
              )}
              {(log.retry_count ?? 0) > 0 && (
                <span className="text-orange-500 flex items-center gap-0.5 text-xs" title={`重试 ${log.retry_count} 次`}>
                  <RotateCcw className="w-3.5 h-3.5" />
                  {log.retry_count}
                </span>
              )}
            </div>

            <div className="flex-shrink-0">
              <span className={`px-1.5 sm:px-2 py-0.5 sm:py-1 rounded text-xs font-mono font-medium border ${getStatusColor(log.success, log.status_code)}`}>
                {log.status_code || '-'}
              </span>
            </div>
          </div>

          <div className="flex items-center gap-2 sm:gap-4 px-3 sm:px-4 pb-3 sm:pb-4 pt-0 text-xs flex-wrap">
            <div className="flex items-center gap-1 text-muted-foreground" title={`API Key: ${log.api_key_name || log.api_key_prefix || '-'}`}>
              <Key className="w-3.5 h-3.5" />
              <span className="max-w-[80px] sm:max-w-[120px] truncate">{log.api_key_name || log.api_key_prefix || '-'}</span>
            </div>

            {log.api_key_group && (
              <div className="hidden sm:flex items-center gap-1 text-muted-foreground">
                <Users className="w-3.5 h-3.5" />
                <span>{log.api_key_group}</span>
              </div>
            )}

            <div className="hidden lg:flex items-center gap-1 text-muted-foreground font-mono">
              <Globe className="w-3.5 h-3.5" />
              <span>{log.client_ip || '-'}</span>
            </div>

            <div className="flex-1" />

            <div className="flex items-center gap-1 font-mono">
              <span className="text-muted-foreground">{log.prompt_tokens || 0}</span>
              <span className="text-muted-foreground/50">+</span>
              <span className="text-blue-600 dark:text-blue-400">{log.completion_tokens || 0}</span>
              <span className="text-muted-foreground/50">=</span>
              <span className="text-foreground">{log.total_tokens || 0}</span>
            </div>

            <div className="flex items-center gap-1 text-muted-foreground" title={`总耗时: ${log.process_time?.toFixed(2)}s, 首响: ${log.first_response_time?.toFixed(2) || '-'}s`}>
              <Clock className="w-3.5 h-3.5" />
              <span className="font-mono">{log.process_time?.toFixed(2) || '-'}s</span>
              {log.first_response_time !== undefined && (
                <span className="text-muted-foreground/60 hidden sm:inline">(首响 {log.first_response_time.toFixed(2)}s)</span>
              )}
            </div>

            {speedInfo && (
              <div className={`flex items-center gap-1 font-mono ${speedInfo.color}`} title="生成速度">
                <Zap className="w-3.5 h-3.5" />
                <span>{speedInfo.speed} t/s</span>
              </div>
            )}
          </div>
        </div>

        {isExpanded && (
          <div className="border-t border-border bg-muted/30 p-4 space-y-4">
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3 text-sm">
              <InfoItem label="日志 ID" value={String(log.id)} mono />
              <InfoItem label="完整时间" value={formatFullTimestamp(log.timestamp)} />
              <InfoItem label="Endpoint" value={log.endpoint || '-'} mono />
              <InfoItem label="客户端 IP" value={log.client_ip || '-'} mono />
              <InfoItem label="Provider ID" value={log.provider_id || '-'} />
              {log.raw_data_expires_at && (
                <InfoItem label="数据过期" value={formatFullTimestamp(log.raw_data_expires_at)} />
              )}
            </div>

            {log.retry_path && (
              <div className="space-y-1">
                <div className="text-xs font-medium text-muted-foreground flex items-center gap-1">
                  <RotateCcw className="w-3.5 h-3.5" /> 重试路径
                </div>
                <RetryPathView retryPathJson={log.retry_path} />
              </div>
            )}

            <div className="space-y-2">
              <JsonAccordion title="请求头" data={log.request_headers} icon={<FileText className="w-4 h-4" />} />
              <JsonAccordion title="用户请求体" data={log.request_body} icon={<Eye className="w-4 h-4" />} />
              <JsonAccordion title="上游请求体" data={log.upstream_request_body} icon={<Server className="w-4 h-4" />} />
              <JsonAccordion title="上游响应体" data={log.upstream_response_body} icon={<Server className="w-4 h-4" />} />
              <JsonAccordion title="用户响应体" data={log.response_body} icon={<EyeOff className="w-4 h-4" />} />
            </div>
          </div>
        )}
      </div>
    );
  };

  const InfoItem = ({ label, value, mono }: { label: string; value: string; mono?: boolean }) => (
    <div className="space-y-0.5">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className={`text-sm text-foreground truncate ${mono ? 'font-mono' : ''}`} title={value}>
        {value}
      </div>
    </div>
  );

  const JsonAccordion = ({ title, data, icon, defaultOpen = false }: { title: string; data?: string; icon?: import('react').ReactNode; defaultOpen?: boolean }) => {
    const [isOpen, setIsOpen] = useState(defaultOpen);

    if (!data) return null;

    const { formatted } = formatJsonBestEffort(data);
    const previewText = formatted.length > 80 ? `${formatted.substring(0, 80)}...` : formatted;

    return (
      <div className="border border-border rounded-lg overflow-hidden">
        <div
          className="flex items-center gap-2 px-3 py-2 bg-muted/50 cursor-pointer hover:bg-muted transition-colors"
          onClick={() => setIsOpen(!isOpen)}
        >
          <div className="flex-shrink-0 text-muted-foreground">
            {isOpen ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
          </div>
          <div className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
            {icon}
            {title}
          </div>
          {!isOpen && (
            <div className="flex-1 text-xs font-mono text-muted-foreground/60 truncate ml-2">
              {previewText.replace(/\n/g, ' ')}
            </div>
          )}
        </div>
        {isOpen && (
          <pre className="bg-background p-3 text-xs font-mono text-foreground overflow-x-auto whitespace-pre-wrap max-h-80 overflow-y-auto border-t border-border">
            {formatted}
          </pre>
        )}
      </div>
    );
  };

  return (
    <div className="space-y-4 sm:space-y-6 animate-in fade-in duration-500 font-sans pb-12 h-full flex flex-col">
      <div className="flex justify-between items-center flex-shrink-0">
        <div>
          <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-foreground">系统日志</h1>
          <p className="text-muted-foreground mt-1 text-sm sm:text-base">监控 API 请求详情与性能，并支持按现有配置快速筛选</p>
        </div>
        <button
          onClick={() => fetchLogs(true)}
          className="p-2 text-muted-foreground hover:text-foreground bg-card border border-border rounded-lg transition-colors"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      <div className="bg-card border border-border p-3 sm:p-4 rounded-xl shadow-sm space-y-3 flex-shrink-0">
        <button
          onClick={() => setShowFilters(!showFilters)}
          className="flex items-center gap-2 text-sm text-muted-foreground md:hidden w-full justify-center py-1"
        >
          <Filter className="w-4 h-4" />
          {showFilters ? '收起筛选' : '展开筛选'}
          <ChevronDown className={`w-4 h-4 transition-transform ${showFilters ? 'rotate-180' : ''}`} />
        </button>

        <div className={`space-y-3 ${showFilters ? 'block' : 'hidden md:block'}`}>
          <div className="grid grid-cols-1 lg:grid-cols-4 gap-3">
            <div className="space-y-2">
              <span className="text-xs font-medium text-muted-foreground">请求状态</span>
              <select
                value={filterSuccess}
                onChange={e => setFilterSuccess(e.target.value)}
                className="w-full bg-background border border-border text-sm px-3 py-2 rounded-lg text-foreground"
              >
                <option value="ALL">所有状态</option>
                <option value="SUCCESS">成功</option>
                <option value="FAILED">失败</option>
              </select>
            </div>

            <QuickFilterField
              label="模型过滤"
              placeholder="支持模糊搜索模型名"
              value={filterModel}
              onChange={setFilterModel}
              options={quickModelOptions.map(item => ({ label: item, value: item }))}
            />

            <QuickFilterField
              label="渠道过滤"
              placeholder="支持模糊搜索渠道 / 渠道 ID"
              value={filterProvider}
              onChange={setFilterProvider}
              options={quickProviderOptions}
            />

            <QuickFilterField
              label="密钥过滤"
              placeholder="支持筛选密钥名称 / 前缀"
              value={filterApiKey}
              onChange={setFilterApiKey}
              options={quickApiKeyOptions}
            />
          </div>

          <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
            <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full bg-muted border border-border">
              共 {totalCount} 条记录
            </span>
            <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full bg-muted border border-border">
              模型选项 {quickModelOptions.length}
            </span>
            <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full bg-muted border border-border">
              渠道选项 {quickProviderOptions.length}
            </span>
            <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full bg-muted border border-border">
              密钥选项 {quickApiKeyOptions.length}
            </span>
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-auto space-y-2">
        {logs.length === 0 && !loading ? (
          <div className="flex flex-col items-center justify-center p-16 text-muted-foreground bg-card border border-border rounded-xl">
            <FileText className="w-12 h-12 mb-4 opacity-50" />
            <p>未找到匹配的日志</p>
          </div>
        ) : (
          logs.map(log => <LogAccordionItem key={log.id} log={log} />)
        )}

        {hasMore && logs.length > 0 && (
          <button
            onClick={loadMore}
            disabled={loading}
            className="w-full text-sm text-muted-foreground hover:text-foreground font-medium flex items-center justify-center gap-1.5 py-4 bg-card border border-border rounded-xl disabled:opacity-50 transition-colors"
          >
            <ArrowDownToLine className="w-4 h-4" />
            {loading ? '加载中...' : `加载更多 (${logs.length}/${totalCount})`}
          </button>
        )}
      </div>
    </div>
  );
}
