import logging
import os
import sys
from collections import deque
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Dict, List, Optional

DEFAULT_BACKEND_LOG_PAGE_SIZE = max(1, int(os.getenv("BACKEND_LOG_PAGE_SIZE", "200")))
DEFAULT_BACKEND_LOG_BUFFER_SIZE = max(50, int(os.getenv("BACKEND_LOG_BUFFER_SIZE", "200")))
MAX_BACKEND_LOG_PAGE_SIZE = 2000
MAX_BACKEND_LOG_BUFFER_SIZE = 50000

_BACKEND_LOG_BUFFER_SIZE = DEFAULT_BACKEND_LOG_BUFFER_SIZE
_backend_log_buffer: deque[Dict[str, Any]] = deque(maxlen=_BACKEND_LOG_BUFFER_SIZE)
_backend_log_lock = RLock()
_backend_log_next_id = 1


def _coerce_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


class TeeStream:
    """将 stdout/stderr 同时写入原始流与内存缓冲区。"""

    def __init__(self, original_stream, stream_name: str):
        self.original_stream = original_stream
        self.stream_name = stream_name
        self._pending = ""

    def write(self, data):
        if data is None:
            return 0
        if isinstance(data, bytes):
            text = data.decode("utf-8", errors="replace")
        else:
            text = str(data)

        written = self.original_stream.write(text)
        self._capture(text)
        return written if written is not None else len(text)

    def flush(self):
        self._flush_pending()
        return self.original_stream.flush()

    def isatty(self):
        return getattr(self.original_stream, "isatty", lambda: False)()

    def fileno(self):
        return self.original_stream.fileno()

    def _capture(self, text: str):
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        self._pending += normalized

        while "\n" in self._pending:
            line, self._pending = self._pending.split("\n", 1)
            _append_backend_log_line(line, self.stream_name)

    def _flush_pending(self):
        if self._pending:
            _append_backend_log_line(self._pending, self.stream_name)
            self._pending = ""

    def __getattr__(self, name):
        return getattr(self.original_stream, name)


def _append_backend_log_line(message: str, stream: str):
    text = str(message or "")
    if not text.strip():
        return

    global _backend_log_next_id
    entry = {
        "id": _backend_log_next_id,
        "captured_at": datetime.now(timezone.utc),
        "stream": stream,
        "message": text,
    }

    with _backend_log_lock:
        _backend_log_buffer.append(entry)
        _backend_log_next_id += 1


def _install_backend_log_capture():
    if getattr(sys, "_zoaholic_backend_log_capture_installed", False):
        return

    if not isinstance(sys.stdout, TeeStream):
        sys.stdout = TeeStream(sys.stdout, "stdout")
    if not isinstance(sys.stderr, TeeStream):
        sys.stderr = TeeStream(sys.stderr, "stderr")

    sys._zoaholic_backend_log_capture_installed = True


_install_backend_log_capture()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("Zoaholic")

logging.getLogger("httpx").setLevel(logging.CRITICAL)
logging.getLogger("watchfiles.main").setLevel(logging.CRITICAL)


def get_backend_log_settings(preferences: Optional[Dict[str, Any]] = None) -> Dict[str, int]:
    prefs = preferences if isinstance(preferences, dict) else {}
    page_size = _coerce_int(
        prefs.get("backend_logs_page_size"),
        DEFAULT_BACKEND_LOG_PAGE_SIZE,
        1,
        MAX_BACKEND_LOG_PAGE_SIZE,
    )
    buffer_size = _coerce_int(
        prefs.get("backend_log_buffer_size"),
        _BACKEND_LOG_BUFFER_SIZE,
        50,
        MAX_BACKEND_LOG_BUFFER_SIZE,
    )

    return {
        "page_size": page_size,
        "buffer_size": buffer_size,
    }


def set_backend_log_buffer_size(size: int) -> int:
    normalized_size = _coerce_int(size, DEFAULT_BACKEND_LOG_BUFFER_SIZE, 50, MAX_BACKEND_LOG_BUFFER_SIZE)

    global _BACKEND_LOG_BUFFER_SIZE, _backend_log_buffer
    with _backend_log_lock:
        snapshot = list(_backend_log_buffer)[-normalized_size:]
        _BACKEND_LOG_BUFFER_SIZE = normalized_size
        _backend_log_buffer = deque(snapshot, maxlen=normalized_size)

    return normalized_size


def apply_backend_log_preferences(preferences: Optional[Dict[str, Any]] = None) -> Dict[str, int]:
    settings = get_backend_log_settings(preferences)
    settings["buffer_size"] = set_backend_log_buffer_size(settings["buffer_size"])
    return settings


def get_backend_log_entries(
    *,
    since_id: Optional[int] = None,
    limit: int = DEFAULT_BACKEND_LOG_PAGE_SIZE,
    search: Optional[str] = None,
    stream: Optional[str] = None,
) -> Dict[str, Any]:
    """返回当前进程最近的后台日志快照。"""

    normalized_stream = (stream or "").strip().lower() or None
    normalized_search = (search or "").strip().lower()

    with _backend_log_lock:
        snapshot: List[Dict[str, Any]] = list(_backend_log_buffer)
        max_id = _backend_log_next_id - 1

    filtered: List[Dict[str, Any]] = []
    for entry in snapshot:
        if since_id is not None and entry["id"] <= since_id:
            continue
        if normalized_stream and entry["stream"] != normalized_stream:
            continue
        if normalized_search and normalized_search not in entry["message"].lower():
            continue
        filtered.append(entry)

    total = len(filtered)
    if limit > 0:
        if since_id is not None:
            filtered = filtered[:limit]
        else:
            filtered = filtered[-limit:]

    return {
        "items": filtered,
        "total": total,
        "max_id": max_id,
        "buffer_size": _BACKEND_LOG_BUFFER_SIZE,
    }
