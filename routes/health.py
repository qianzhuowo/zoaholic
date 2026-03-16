from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from db import DISABLE_DATABASE

router = APIRouter()


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_health_payload(app, *, readiness: bool) -> tuple[dict[str, Any], int]:
    state = getattr(app, "state", None)
    startup_completed = bool(getattr(state, "startup_completed", False))
    started_at = getattr(state, "started_at", None)
    if isinstance(started_at, datetime):
        uptime_seconds = max(0.0, (datetime.now(timezone.utc) - started_at).total_seconds())
        started_at_iso = started_at.astimezone(timezone.utc).isoformat()
    else:
        uptime_seconds = None
        started_at_iso = None

    version = getattr(state, "version", "unknown")
    needs_setup = bool(getattr(state, "needs_setup", False))
    config = getattr(state, "config", None)
    providers = len((config or {}).get("providers") or []) if isinstance(config, dict) else 0
    api_keys = len((config or {}).get("api_keys") or []) if isinstance(config, dict) else 0

    watchdog = getattr(state, "event_loop_watchdog", None)
    event_loop = watchdog.snapshot() if watchdog else {
        "enabled": False,
        "status": "missing",
        "healthy": True,
    }

    checks = {
        "startup": {
            "status": "ok" if startup_completed else "error",
            "startup_completed": startup_completed,
            "started_at": started_at_iso,
            "uptime_seconds": uptime_seconds,
        },
        "config": {
            "status": "ok" if config is not None else "error",
            "loaded": config is not None,
            "needs_setup": needs_setup,
            "provider_count": providers,
            "api_key_count": api_keys,
        },
        "client_manager": {
            "status": "ok" if hasattr(state, "client_manager") else "error",
            "initialized": hasattr(state, "client_manager"),
        },
        "channel_manager": {
            "status": "ok" if hasattr(state, "channel_manager") else "error",
            "initialized": hasattr(state, "channel_manager"),
        },
        "database": {
            "status": "disabled" if DISABLE_DATABASE else "ok",
            "enabled": not DISABLE_DATABASE,
        },
        "event_loop": event_loop,
    }

    blocking_error = event_loop.get("status") == "critical"
    blocking_warning = event_loop.get("status") == "warning"
    missing_runtime = not startup_completed or config is None or not hasattr(state, "client_manager") or not hasattr(state, "channel_manager")

    if blocking_error or missing_runtime:
        overall_status = "error"
        status_code = 503
    elif blocking_warning or needs_setup:
        overall_status = "degraded"
        status_code = 200
    else:
        overall_status = "ok"
        status_code = 200

    payload = {
        "status": overall_status,
        "service": "zoaholic",
        "version": version,
        "timestamp": _utcnow_iso(),
        "checks": checks,
    }
    return payload, status_code


@router.get("/healthz")
async def healthz(request: Request):
    payload, status_code = _build_health_payload(request.app, readiness=False)
    return JSONResponse(status_code=status_code, content=payload)


@router.get("/readyz")
async def readyz(request: Request):
    payload, status_code = _build_health_payload(request.app, readiness=True)
    return JSONResponse(status_code=status_code, content=payload)
