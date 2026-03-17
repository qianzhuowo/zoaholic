import { useEffect, useRef, useState } from 'react';
import { useAuthStore } from '../store/authStore';
import { apiFetch } from '../lib/api';
import {
  RefreshCw,
  Search,
  Server,
  Play,
  Pause,
  Terminal,
  AlertTriangle,
  ArrowDown,
  Save,
  Settings2,
  ChevronDown,
  ChevronRight,
} from 'lucide-react';

interface BackendLogEntry {
  id: number;
  captured_at: string;
  stream: 'stdout' | 'stderr';
  message: string;
}

interface BackendLogsResponse {
  items: BackendLogEntry[];
  total: number;
  limit: number;
  max_id: number;
  buffer_size: number;
}

const POLL_INTERVAL_MS = 3000;
const DEFAULT_PAGE_SIZE = 200;
const DEFAULT_BUFFER_SIZE = 200;
const PAGE_SIZE_OPTIONS = [50, 100, 200, 400, 800];
const BUFFER_SIZE_OPTIONS = [200, 400, 800, 1200, 2000, 5000];

function formatTime(ts: string) {
  try {
    return new Date(ts).toLocaleString('zh-CN', {
      hour12: false,
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    });
  } catch {
    return ts;
  }
}

function normalizePositiveInt(value: unknown, fallback: number) {
  const parsed = Number.parseInt(String(value ?? ''), 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

export default function BackendLogs() {
  const { token } = useAuthStore();
  const [items, setItems] = useState<BackendLogEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const [savingConfig, setSavingConfig] = useState(false);
  const [error, setError] = useState('');
  const [configMessage, setConfigMessage] = useState('');
  const [search, setSearch] = useState('');
  const [streamFilter, setStreamFilter] = useState<'ALL' | 'stdout' | 'stderr'>('ALL');
  const [pageSize, setPageSize] = useState(DEFAULT_PAGE_SIZE);
  const [bufferSize, setBufferSize] = useState(DEFAULT_BUFFER_SIZE);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [showConfig, setShowConfig] = useState(false);
  const [total, setTotal] = useState(0);
  const [maxId, setMaxId] = useState(0);
  const [effectiveBufferSize, setEffectiveBufferSize] = useState(DEFAULT_BUFFER_SIZE);
  const scrollerRef = useRef<HTMLDivElement | null>(null);
  const pendingScrollToBottomRef = useRef(false);

  const scrollToBottom = () => {
    if (!scrollerRef.current) return;
    scrollerRef.current.scrollTop = scrollerRef.current.scrollHeight;
  };

  const fetchBackendLogSettings = async () => {
    if (!token) return;

    try {
      const res = await apiFetch('/v1/api_config');
      if (!res.ok) return;

      const data = await res.json();
      const preferences = data.api_config?.preferences || data.preferences || {};
      const serverPageSize = normalizePositiveInt(preferences.backend_logs_page_size, DEFAULT_PAGE_SIZE);
      const serverBufferSize = normalizePositiveInt(preferences.backend_log_buffer_size, DEFAULT_BUFFER_SIZE);

      setPageSize(serverPageSize);
      setBufferSize(serverBufferSize);
    } catch (err) {
      console.error('Failed to load backend log settings:', err);
    }
  };

  const fetchBackendLogs = async ({ scrollAfterLoad = false }: { scrollAfterLoad?: boolean } = {}) => {
    if (!token) return;
    setLoading(true);
    setError('');
    pendingScrollToBottomRef.current = scrollAfterLoad;

    try {
      const queryParams = new URLSearchParams({
        limit: String(pageSize),
      });

      if (search.trim()) queryParams.set('search', search.trim());
      if (streamFilter !== 'ALL') queryParams.set('stream', streamFilter);

      const res = await apiFetch(`/v1/backend_logs?${queryParams.toString()}`);
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || `请求失败 (${res.status})`);
      }

      const data: BackendLogsResponse = await res.json();
      setItems(data.items || []);
      setTotal(data.total || 0);
      setMaxId(data.max_id || 0);
      setEffectiveBufferSize(data.buffer_size || bufferSize);
    } catch (err) {
      pendingScrollToBottomRef.current = false;
      setError(err instanceof Error ? err.message : '获取后台日志失败');
    } finally {
      setLoading(false);
    }
  };

  const handlePageSizeChange = (value: number) => {
    setPageSize(value);
    setConfigMessage('');
  };

  const handleBufferSizeChange = (value: number) => {
    setBufferSize(value);
    setConfigMessage('');
  };

  const saveBackendLogSettings = async () => {
    if (!token) return;

    const normalizedPageSize = pageSize;
    const normalizedBufferSize = bufferSize;

    setSavingConfig(true);
    setConfigMessage('');
    setError('');

    try {
      const res = await apiFetch('/v1/api_config/update', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          preferences: {
            backend_logs_page_size: normalizedPageSize,
            backend_log_buffer_size: normalizedBufferSize,
          },
        }),
      });

      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || `保存失败 (${res.status})`);
      }

      setConfigMessage('显示配置已保存到服务端');
      await fetchBackendLogs();
    } catch (err) {
      setError(err instanceof Error ? err.message : '保存显示配置失败');
    } finally {
      setSavingConfig(false);
    }
  };

  useEffect(() => {
    fetchBackendLogSettings();
  }, [token]);

  useEffect(() => {
    fetchBackendLogs({ scrollAfterLoad: true });
  }, [token, search, streamFilter, pageSize]);

  useEffect(() => {
    if (!autoRefresh || !token) return;
    const timer = window.setInterval(() => {
      fetchBackendLogs({ scrollAfterLoad: false });
    }, POLL_INTERVAL_MS);

    return () => window.clearInterval(timer);
  }, [autoRefresh, token, search, streamFilter, pageSize]);

  useEffect(() => {
    if (!pendingScrollToBottomRef.current) return;
    scrollToBottom();
    pendingScrollToBottomRef.current = false;
  }, [items]);

  const isBufferSmallerThanPage = bufferSize < pageSize;

  return (
    <div className="h-full min-h-0 min-w-0 flex flex-col gap-4 sm:gap-6 animate-in fade-in duration-500">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between flex-shrink-0">
        <div className="min-w-0">
          <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-foreground">后台日志</h1>
          <p className="text-muted-foreground mt-1 text-sm sm:text-base">
            查看当前实例的 stdout / stderr 输出，适合 Render、Railway 等平台快速排查部署日志。
          </p>
        </div>

        <div className="flex items-center gap-2 self-start sm:self-auto">
          <button
            type="button"
            onClick={() => setAutoRefresh(prev => !prev)}
            className={`inline-flex items-center justify-center gap-2 rounded-lg px-3 py-2 text-sm border transition-colors ${
              autoRefresh
                ? 'bg-primary text-primary-foreground border-primary'
                : 'bg-card text-foreground border-border hover:bg-muted'
            }`}
          >
            {autoRefresh ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            {autoRefresh ? '自动刷新中' : '自动刷新已暂停'}
          </button>

          <button
            type="button"
            onClick={() => fetchBackendLogs({ scrollAfterLoad: true })}
            title="立即刷新"
            className="inline-flex items-center justify-center rounded-lg p-2 bg-card border border-border text-muted-foreground hover:text-foreground transition-colors"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </div>

      {showConfig && (
        <div className="rounded-lg border border-border bg-muted/30 px-3 py-3 space-y-3 flex-shrink-0">
          <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-3">
            <div>
              <div className="text-xs font-medium text-muted-foreground flex items-center gap-1.5">
                <Settings2 className="w-3.5 h-3.5" /> 显示配置
              </div>
              <div className="text-[11px] text-muted-foreground/80 mt-1">
                配置保存在服务端，对所有管理员生效，不再跟随当前浏览器。
              </div>
            </div>

            <button
              type="button"
              onClick={saveBackendLogSettings}
              disabled={savingConfig}
              className="inline-flex items-center justify-center gap-2 rounded-lg px-3 py-2 text-sm border bg-primary text-primary-foreground border-primary hover:bg-primary/90 disabled:opacity-60"
            >
              {savingConfig ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Save className="w-4 h-4" />}
              保存配置
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <label className="space-y-1">
              <span className="text-xs font-medium text-muted-foreground">一页显示多少行</span>
              <select
                value={pageSize}
                onChange={e => handlePageSizeChange(parseInt(e.target.value, 10))}
                className="w-full bg-background border border-border rounded-lg px-3 py-2 text-sm text-foreground"
              >
                {PAGE_SIZE_OPTIONS.map(option => (
                  <option key={option} value={option}>{`${option} 行 / 页`}</option>
                ))}
              </select>
            </label>

            <label className="space-y-1">
              <span className="text-xs font-medium text-muted-foreground">后端保留多少行</span>
              <select
                value={bufferSize}
                onChange={e => handleBufferSizeChange(parseInt(e.target.value, 10))}
                className="w-full bg-background border border-border rounded-lg px-3 py-2 text-sm text-foreground"
              >
                {BUFFER_SIZE_OPTIONS.map(option => (
                  <option key={option} value={option}>{`保留最近 ${option} 行`}</option>
                ))}
              </select>
            </label>
          </div>

          {isBufferSmallerThanPage && (
            <div className="rounded-lg border border-yellow-500/20 bg-yellow-500/10 px-3 py-2 text-sm text-yellow-700 dark:text-yellow-400">
              当前“后端保留多少行”小于“一页显示多少行”。这意味着即使单页设置为 {pageSize} 行，
              服务端内存里也只保留最近 {bufferSize} 行，因此页面实际最多只能拿到 {bufferSize} 行日志。
            </div>
          )}

          {configMessage && (
            <div className="rounded-lg border border-emerald-500/20 bg-emerald-500/10 px-3 py-2 text-sm text-emerald-600 dark:text-emerald-400">
              {configMessage}
            </div>
          )}
        </div>
      )}

      <div className="bg-card border border-border rounded-xl shadow-sm p-3 sm:p-4 space-y-3 flex-shrink-0">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <label className="space-y-1 md:col-span-2">
            <span className="text-xs font-medium text-muted-foreground">关键字筛选</span>
            <div className="relative">
              <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" />
              <input
                type="text"
                value={search}
                onChange={e => setSearch(e.target.value)}
                placeholder="搜索日志内容"
                className="w-full bg-background border border-border rounded-lg pl-9 pr-3 py-2 text-sm text-foreground"
              />
            </div>
          </label>

          <label className="space-y-1">
            <span className="text-xs font-medium text-muted-foreground">输出流</span>
            <select
              value={streamFilter}
              onChange={e => setStreamFilter(e.target.value as 'ALL' | 'stdout' | 'stderr')}
              className="w-full bg-background border border-border rounded-lg px-3 py-2 text-sm text-foreground"
            >
              <option value="ALL">全部输出</option>
              <option value="stdout">仅 stdout</option>
              <option value="stderr">仅 stderr</option>
            </select>
          </label>
        </div>

        <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
          <button
            type="button"
            onClick={() => setShowConfig(prev => !prev)}
            className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full bg-muted border border-border hover:bg-muted/80 transition-colors"
          >
            {showConfig ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronRight className="w-3.5 h-3.5" />}
            显示配置
          </button>
          <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full bg-muted border border-border">
            <Server className="w-3.5 h-3.5" /> 当前结果 {items.length} 行
          </span>
          <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full bg-muted border border-border">
            <Terminal className="w-3.5 h-3.5" /> 匹配总数 {total}
          </span>
          <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full bg-muted border border-border">
            <ArrowDown className="w-3.5 h-3.5" /> 最新 ID {maxId}
          </span>
          <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full bg-muted border border-border">
            <Server className="w-3.5 h-3.5" /> 后端当前保留 {effectiveBufferSize} 行
          </span>
          <span className="text-[11px] opacity-80">
            自动刷新不会自动跳到最新一行，避免打断阅读。
          </span>
        </div>

        {error && (
          <div className="flex items-start gap-2 rounded-lg border border-red-500/20 bg-red-500/10 px-3 py-2 text-sm text-red-600 dark:text-red-400">
            <AlertTriangle className="w-4 h-4 mt-0.5 flex-shrink-0" />
            <span>{error}</span>
          </div>
        )}
      </div>

      <div className="flex-1 min-h-0 min-w-0 bg-card border border-border rounded-xl overflow-hidden flex flex-col">
        {items.length === 0 && !loading ? (
          <div className="h-full flex flex-col items-center justify-center text-muted-foreground px-6 py-16 text-center">
            <Terminal className="w-12 h-12 mb-4 opacity-50" />
            <p className="text-sm">当前没有可展示的后台日志</p>
            <p className="text-xs mt-2 opacity-80">可尝试切换筛选条件，或等待应用产生新的 stdout / stderr 输出。</p>
          </div>
        ) : (
          <div ref={scrollerRef} className="flex-1 min-h-0 overflow-auto bg-background/60">
            <div className="w-full min-w-0 font-mono text-xs">
              {items.map(item => {
                const isErrorStream = item.stream === 'stderr';
                return (
                  <div
                    key={item.id}
                    className={`grid grid-cols-1 sm:grid-cols-[120px_78px_minmax(0,1fr)] gap-2 sm:gap-3 px-3 py-2 border-b border-border/60 hover:bg-muted/40 ${
                      isErrorStream ? 'bg-red-500/5' : ''
                    }`}
                  >
                    <div className="text-muted-foreground whitespace-nowrap">{formatTime(item.captured_at)}</div>
                    <div>
                      <span
                        className={`inline-flex items-center rounded px-1.5 py-0.5 border ${
                          isErrorStream
                            ? 'text-red-600 dark:text-red-400 bg-red-500/10 border-red-500/20'
                            : 'text-blue-600 dark:text-blue-400 bg-blue-500/10 border-blue-500/20'
                        }`}
                      >
                        {item.stream}
                      </span>
                    </div>
                    <div className="text-foreground whitespace-pre-wrap break-words leading-5 min-w-0">{item.message}</div>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
