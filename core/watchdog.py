from __future__ import annotations

import asyncio
import os
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Deque, Optional

from .log_config import logger


DEFAULT_CHECK_INTERVAL_MS = 500
DEFAULT_WARN_THRESHOLD_MS = 300
DEFAULT_CRITICAL_THRESHOLD_MS = 1500
DEFAULT_RECOVERY_WINDOW_MS = 5000
DEFAULT_HISTORY_SIZE = 20


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _isoformat(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat()


def _read_env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except Exception:
        return default


@dataclass(frozen=True)
class EventLoopBlockRecord:
    detected_at: str
    blocked_ms: int
    severity: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "detected_at": self.detected_at,
            "blocked_ms": self.blocked_ms,
            "severity": self.severity,
        }


class EventLoopBlockWatchdog:
    """基于 sleep 漂移检测事件循环阻塞情况。"""

    def __init__(
        self,
        *,
        enabled: bool = True,
        check_interval_ms: int = DEFAULT_CHECK_INTERVAL_MS,
        warn_threshold_ms: int = DEFAULT_WARN_THRESHOLD_MS,
        critical_threshold_ms: int = DEFAULT_CRITICAL_THRESHOLD_MS,
        recovery_window_ms: int = DEFAULT_RECOVERY_WINDOW_MS,
        history_size: int = DEFAULT_HISTORY_SIZE,
    ) -> None:
        self.enabled = bool(enabled)
        self.check_interval_ms = max(50, int(check_interval_ms))
        self.warn_threshold_ms = max(1, int(warn_threshold_ms))
        self.critical_threshold_ms = max(self.warn_threshold_ms, int(critical_threshold_ms))
        self.recovery_window_ms = max(self.check_interval_ms, int(recovery_window_ms))
        self.history_size = max(1, int(history_size))

        self._task: Optional[asyncio.Task] = None
        self._stopping = False
        self._started_at: Optional[datetime] = None
        self._last_check_at: Optional[datetime] = None
        self._last_healthy_at: Optional[datetime] = None
        self._last_blocked_at: Optional[datetime] = None
        self._last_blocked_ms: int = 0
        self._max_blocked_ms: int = 0
        self._block_count: int = 0
        self._sample_count: int = 0
        self._recent_blocks: Deque[EventLoopBlockRecord] = deque(maxlen=self.history_size)

    @classmethod
    def from_env(cls) -> "EventLoopBlockWatchdog":
        enabled_raw = str(os.getenv("EVENT_LOOP_WATCHDOG_ENABLED", "true")).strip().lower()
        enabled = enabled_raw not in {"0", "false", "no", "off"}
        return cls(
            enabled=enabled,
            check_interval_ms=_read_env_int("EVENT_LOOP_WATCHDOG_INTERVAL_MS", DEFAULT_CHECK_INTERVAL_MS),
            warn_threshold_ms=_read_env_int("EVENT_LOOP_WATCHDOG_WARN_MS", DEFAULT_WARN_THRESHOLD_MS),
            critical_threshold_ms=_read_env_int("EVENT_LOOP_WATCHDOG_CRITICAL_MS", DEFAULT_CRITICAL_THRESHOLD_MS),
            recovery_window_ms=_read_env_int("EVENT_LOOP_WATCHDOG_RECOVERY_MS", DEFAULT_RECOVERY_WINDOW_MS),
            history_size=_read_env_int("EVENT_LOOP_WATCHDOG_HISTORY_SIZE", DEFAULT_HISTORY_SIZE),
        )

    async def start(self) -> None:
        if not self.enabled:
            logger.info("EventLoopBlockWatchdog is disabled")
            return
        if self._task and not self._task.done():
            return

        self._stopping = False
        self._started_at = _utcnow()
        self._last_healthy_at = self._started_at
        self._task = asyncio.create_task(self._run(), name="event-loop-block-watchdog")
        logger.info(
            "EventLoopBlockWatchdog started: interval=%sms warn=%sms critical=%sms recovery=%sms",
            self.check_interval_ms,
            self.warn_threshold_ms,
            self.critical_threshold_ms,
            self.recovery_window_ms,
        )

    async def stop(self) -> None:
        self._stopping = True
        if not self._task:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        finally:
            self._task = None

    async def _run(self) -> None:
        interval_seconds = self.check_interval_ms / 1000.0
        loop = asyncio.get_running_loop()
        last_tick = loop.time()

        try:
            while True:
                await asyncio.sleep(interval_seconds)
                now_loop = loop.time()
                blocked_seconds = max(0.0, now_loop - last_tick - interval_seconds)
                blocked_ms = int(round(blocked_seconds * 1000))
                now = _utcnow()

                self._sample_count += 1
                self._last_check_at = now

                if blocked_ms >= self.warn_threshold_ms:
                    self._block_count += 1
                    self._last_blocked_at = now
                    self._last_blocked_ms = blocked_ms
                    self._max_blocked_ms = max(self._max_blocked_ms, blocked_ms)
                    severity = "critical" if blocked_ms >= self.critical_threshold_ms else "warning"
                    self._recent_blocks.append(
                        EventLoopBlockRecord(
                            detected_at=_isoformat(now) or "",
                            blocked_ms=blocked_ms,
                            severity=severity,
                        )
                    )
                    log_func = logger.error if severity == "critical" else logger.warning
                    log_func(
                        "Event loop block detected: blocked=%sms severity=%s interval=%sms",
                        blocked_ms,
                        severity,
                        self.check_interval_ms,
                    )
                else:
                    self._last_healthy_at = now

                last_tick = now_loop
        except asyncio.CancelledError:
            logger.info("EventLoopBlockWatchdog stopped")
            raise
        except Exception as exc:
            logger.exception("EventLoopBlockWatchdog crashed: %s", exc)
            raise

    def snapshot(self) -> dict[str, Any]:
        if not self.enabled:
            return {
                "enabled": False,
                "status": "disabled",
                "healthy": True,
            }

        now = _utcnow()
        task_running = bool(self._task and not self._task.done() and not self._stopping)
        age_since_last_block_ms: Optional[int] = None
        if self._last_blocked_at is not None:
            age_since_last_block_ms = int((now - self._last_blocked_at).total_seconds() * 1000)

        status = "ok"
        healthy = task_running
        if not task_running:
            status = "stopped"
            healthy = False
        elif (
            age_since_last_block_ms is not None
            and age_since_last_block_ms <= self.recovery_window_ms
            and self._last_blocked_ms >= self.critical_threshold_ms
        ):
            status = "critical"
            healthy = False
        elif (
            age_since_last_block_ms is not None
            and age_since_last_block_ms <= self.recovery_window_ms
            and self._last_blocked_ms >= self.warn_threshold_ms
        ):
            status = "warning"

        return {
            "enabled": True,
            "status": status,
            "healthy": healthy,
            "task_running": task_running,
            "check_interval_ms": self.check_interval_ms,
            "warn_threshold_ms": self.warn_threshold_ms,
            "critical_threshold_ms": self.critical_threshold_ms,
            "recovery_window_ms": self.recovery_window_ms,
            "started_at": _isoformat(self._started_at),
            "last_check_at": _isoformat(self._last_check_at),
            "last_healthy_at": _isoformat(self._last_healthy_at),
            "last_blocked_at": _isoformat(self._last_blocked_at),
            "last_blocked_ms": self._last_blocked_ms,
            "max_blocked_ms": self._max_blocked_ms,
            "block_count": self._block_count,
            "sample_count": self._sample_count,
            "age_since_last_block_ms": age_since_last_block_ms,
            "recent_blocks": [item.to_dict() for item in self._recent_blocks],
        }
