"""Lightweight system performance monitoring and safe response for Hermes.

Samples CPU, memory, swap, disk, battery, load, process count, and hot
processes. It writes a baseline, writes process-pressure audits when pressure is
high, and takes only conservative Hermes-self actions by default.

Config: ``logging.performance_monitor`` in ``config.yaml``.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_monitor_thread: Optional[threading.Thread] = None
_stop_event: Optional[threading.Event] = None
_start_time: Optional[float] = None
_interval_seconds: float = 60.0
_auto_actions_enabled: bool = True
_last_gc_action_at: float = 0.0
_self_niced: bool = False
_lock = threading.Lock()

_BYTES_TO_MB = 1024 * 1024
_BASELINE_FILENAME = "performance-baseline.json"
_AUDIT_FILENAME = "performance-process-audit.json"
_ACTION_COOLDOWN_SECONDS = 300.0


@dataclass(frozen=True)
class PerformanceSnapshot:
    cpu_percent: float
    memory_percent: float
    memory_available_mb: int
    swap_percent: float
    disk_percent: float
    load_1m: Optional[float]
    load_per_core: Optional[float]
    cpu_count: int
    process_count: int
    battery_percent: Optional[float]
    power_plugged: Optional[bool]
    top_cpu: str
    top_memory: str
    process_audit_summary: str
    pressure: str
    score: int
    checks: tuple[str, ...]
    recommendation: str
    action: str
    uptime_seconds: int
    sampled_at: str


@dataclass(frozen=True)
class ProcessAuditItem:
    pid: int
    name: str
    username: str
    cpu_percent: float
    memory_percent: float
    status: str
    category: str
    risk: str
    recommended_action: str
    safe_auto_action: bool
    cmdline: str = ""


@dataclass(frozen=True)
class ProcessAudit:
    sampled_at: str
    process_count: int
    category_counts: dict[str, int] = field(default_factory=dict)
    heavy_cpu: list[ProcessAuditItem] = field(default_factory=list)
    heavy_memory: list[ProcessAuditItem] = field(default_factory=list)
    candidates: list[ProcessAuditItem] = field(default_factory=list)
    summary: str = "no audit collected"


def _import_psutil():
    try:
        import psutil  # type: ignore

        return psutil
    except Exception:
        return None


def _hermes_home() -> Path:
    try:
        from hermes_constants import get_hermes_home

        return Path(get_hermes_home())
    except Exception:
        return Path.home() / ".hermes"


def _baseline_path() -> Path:
    return _hermes_home() / _BASELINE_FILENAME


def _audit_path() -> Path:
    return _hermes_home() / _AUDIT_FILENAME


def _safe_str(value: object, limit: int = 220) -> str:
    return str(value or "").replace("\n", " ").strip()[:limit]


def _fmt_process(proc: object, metric: str) -> str:
    try:
        info = getattr(proc, "info", {}) or {}
        name = _safe_str(info.get("name") or "unknown", 28).replace(" ", "_")
        pid = info.get("pid")
        value = float(info.get(metric) or 0.0)
        suffix = "%mem" if metric == "memory_percent" else "%cpu"
        return f"{name}:{pid}:{value:.1f}{suffix}"
    except Exception:
        return "unknown"


def _top_processes(psutil, metric: str, limit: int = 5) -> str:
    try:
        procs = []
        for proc in psutil.process_iter(attrs=["pid", "name", metric]):
            try:
                value = float((proc.info or {}).get(metric) or 0.0)
                if value > 0:
                    procs.append((value, proc))
            except Exception:
                continue
        procs.sort(key=lambda item: item[0], reverse=True)
        return ",".join(_fmt_process(proc, metric) for _, proc in procs[:limit]) or "none"
    except Exception:
        return "unavailable"


def _categorize_process(info: dict) -> str:
    name = str(info.get("name") or "").lower()
    cmdline = " ".join(info.get("cmdline") or []).lower()
    blob = f"{name} {cmdline}"
    if "hermes" in blob or "cartha-voice-listener" in blob:
        return "hermes"
    if any(x in blob for x in ("google chrome helper", "webkit.webcontent", "codex helper", "electron", "renderer")):
        return "browser-helper"
    if any(x in blob for x in ("google drive", "drivefs", "dropbox", "onedrive", "icloud")):
        return "sync-service"
    if any(x in blob for x in ("xcodebuild", "node", "npm", "pnpm", "yarn", "python", "dart", "flutter", "gradle", "java")):
        return "developer-job"
    if any(x in blob for x in ("launchd", "agent", "daemon", "helper")):
        return "background-service"
    return "user-process"


def _process_recommendation(category: str, cpu: float, mem: float, status: str) -> tuple[str, str, bool]:
    if status.lower() == "zombie":
        return "high", "inspect parent process; zombie cannot be killed directly", False
    if category == "hermes" and (cpu >= 50 or mem >= 1.0):
        return "high", "Hermes is costly; lower priority, collect garbage, or restart Hermes gateway if leak persists", True
    if category == "browser-helper" and (cpu >= 35 or mem >= 1.0):
        return "medium", "browser helper is heavy; consider closing unused tabs/windows", False
    if category == "sync-service" and (cpu >= 25 or mem >= 1.0):
        return "medium", "sync service is heavy; consider pausing sync during builds or local model work", False
    if category == "developer-job" and cpu >= 60:
        return "medium", "developer job is CPU-heavy; let it finish or cancel the specific build/test if stale", False
    if cpu >= 90 or mem >= 4.0:
        return "high", "very heavy process; inspect before stopping", False
    if cpu >= 35 or mem >= 1.0:
        return "medium", "notable resource usage; monitor trend", False
    return "low", "observe", False


def _audit_processes(psutil, limit: int = 10) -> ProcessAudit:
    items: list[ProcessAuditItem] = []
    category_counts: dict[str, int] = {}
    attrs = ["pid", "name", "username", "cpu_percent", "memory_percent", "status", "cmdline"]
    try:
        iterator = psutil.process_iter(attrs=attrs)
    except Exception:
        iterator = []
    for proc in iterator:
        try:
            info = proc.info or {}
            category = _categorize_process(info)
            category_counts[category] = category_counts.get(category, 0) + 1
            cpu = float(info.get("cpu_percent") or 0.0)
            mem = float(info.get("memory_percent") or 0.0)
            status = str(info.get("status") or "unknown")
            risk, recommendation, safe = _process_recommendation(category, cpu, mem, status)
            items.append(ProcessAuditItem(
                pid=int(info.get("pid") or 0),
                name=_safe_str(info.get("name") or "unknown", 80),
                username=_safe_str(info.get("username") or "unknown", 80),
                cpu_percent=round(cpu, 2),
                memory_percent=round(mem, 2),
                status=status,
                category=category,
                risk=risk,
                recommended_action=recommendation,
                safe_auto_action=safe,
                cmdline=_safe_str(" ".join(info.get("cmdline") or [])),
            ))
        except Exception:
            continue
    heavy_cpu = sorted(items, key=lambda i: i.cpu_percent, reverse=True)[:limit]
    heavy_memory = sorted(items, key=lambda i: i.memory_percent, reverse=True)[:limit]
    candidates = sorted(
        [i for i in items if i.risk in {"medium", "high"}],
        key=lambda i: (i.risk == "high", i.cpu_percent + i.memory_percent),
        reverse=True,
    )[:limit]
    top_categories = sorted(category_counts.items(), key=lambda kv: kv[1], reverse=True)[:4]
    summary = ";".join(f"{name}={count}" for name, count in top_categories) or "no-processes"
    return ProcessAudit(
        sampled_at=datetime.now(timezone.utc).isoformat(),
        process_count=len(items),
        category_counts=category_counts,
        heavy_cpu=heavy_cpu,
        heavy_memory=heavy_memory,
        candidates=candidates,
        summary=summary,
    )


def _score_from_pressure(cpu: float, mem: float, swap: float, disk: float, available_mb: int, load_per_core: Optional[float], process_count: int = 0) -> int:
    score = 100
    score -= max(0, int((cpu - 55) * 0.6))
    score -= max(0, int((mem - 65) * 0.9))
    score -= max(0, int((swap - 10) * 0.8))
    score -= max(0, int((disk - 80) * 0.7))
    if available_mb < 2048:
        score -= int((2048 - available_mb) / 128)
    if load_per_core is not None and load_per_core > 1.25:
        score -= int((load_per_core - 1.25) * 18)
    if process_count > 1000:
        score -= min(15, int((process_count - 1000) / 75) + 4)
    return max(0, min(100, score))


def _classify_pressure(cpu: float, mem: float, swap: float, disk: float, available_mb: int, load_per_core: Optional[float] = None, process_count: int = 0) -> tuple[str, int, tuple[str, ...], str]:
    checks: list[str] = []
    if available_mb < 750 or mem >= 92 or swap >= 70:
        checks.append("memory-critical")
    elif mem >= 82 or swap >= 40 or available_mb < 1536:
        checks.append("memory-high")
    else:
        checks.append("memory-ok")
    if cpu >= 92 or (load_per_core is not None and load_per_core >= 2.0):
        checks.append("cpu-critical")
    elif cpu >= 80 or (load_per_core is not None and load_per_core >= 1.35):
        checks.append("cpu-high")
    else:
        checks.append("cpu-ok")
    if disk >= 95:
        checks.append("disk-critical")
    elif disk >= 88:
        checks.append("disk-high")
    else:
        checks.append("disk-ok")
    if process_count >= 1200:
        checks.append("process-critical")
    elif process_count >= 900:
        checks.append("process-high")
    else:
        checks.append("process-ok")

    score = _score_from_pressure(cpu, mem, swap, disk, available_mb, load_per_core, process_count)
    pressure = "critical" if any(c.endswith("critical") for c in checks) else "high" if any(c.endswith("high") for c in checks) else "normal"
    if "memory-critical" in checks:
        recommendation = "free-memory-swap: collect Hermes garbage, pause new local-model runs, close memory-heavy apps"
    elif "cpu-critical" in checks:
        recommendation = "reduce-cpu: lower Hermes priority, pause runaway jobs, defer indexing/builds"
    elif "disk-critical" in checks:
        recommendation = "free-disk: clear caches/downloads before builds or media work"
    elif "memory-high" in checks:
        recommendation = "watch-memory: close idle heavy apps or restart leaking background services soon"
    elif "cpu-high" in checks:
        recommendation = "watch-cpu: let current jobs finish before starting more parallel work"
    elif "disk-high" in checks:
        recommendation = "watch-disk: cleanup recommended soon"
    elif "process-high" in checks or "process-critical" in checks:
        recommendation = "watch-processes: too many processes; inspect launch agents, browser helpers, and background jobs"
    else:
        recommendation = "healthy: no action needed"
    return pressure, score, tuple(checks), recommendation


def _safe_action_for_snapshot(snapshot: PerformanceSnapshot) -> str:
    global _last_gc_action_at, _self_niced
    if not _auto_actions_enabled:
        return "disabled"
    actions: list[str] = []
    now = time.monotonic()
    if snapshot.pressure in {"high", "critical"} and now - _last_gc_action_at >= _ACTION_COOLDOWN_SECONDS:
        collected = gc.collect()
        _last_gc_action_at = now
        actions.append(f"gc.collect:{collected}")
    if "cpu-critical" in snapshot.checks and not _self_niced:
        try:
            os.nice(5)
            _self_niced = True
            actions.append("lowered-hermes-priority")
        except Exception as exc:
            actions.append(f"lower-priority-failed:{type(exc).__name__}")
    if not actions:
        return "observe"
    action = "+".join(actions)
    logger.warning("[PERF_ACTION] pressure=%s score=%d action=%s", snapshot.pressure, snapshot.score, action)
    return action


def collect_performance_snapshot(apply_actions: bool = False) -> Optional[PerformanceSnapshot]:
    psutil = _import_psutil()
    if psutil is None:
        return None
    try:
        cpu_count = int(psutil.cpu_count() or 1)
        cpu_percent = float(psutil.cpu_percent(interval=0.1))
        vm = psutil.virtual_memory()
        swap = psutil.swap_memory()
        disk = psutil.disk_usage("/")
        try:
            load_1m = float(psutil.getloadavg()[0])
            load_per_core = load_1m / max(cpu_count, 1)
        except Exception:
            load_1m = None
            load_per_core = None
        try:
            battery = psutil.sensors_battery()
            battery_percent = None if battery is None else float(battery.percent)
            power_plugged = None if battery is None else bool(battery.power_plugged)
        except Exception:
            battery_percent = None
            power_plugged = None
        try:
            process_count = len(psutil.pids())
        except Exception:
            process_count = 0
        available_mb = int(getattr(vm, "available", 0) / _BYTES_TO_MB)
        pressure, score, checks, recommendation = _classify_pressure(
            cpu=cpu_percent,
            mem=float(vm.percent),
            swap=float(swap.percent),
            disk=float(disk.percent),
            available_mb=available_mb,
            load_per_core=load_per_core,
            process_count=process_count,
        )
        snapshot = PerformanceSnapshot(
            cpu_percent=cpu_percent,
            memory_percent=float(vm.percent),
            memory_available_mb=available_mb,
            swap_percent=float(swap.percent),
            disk_percent=float(disk.percent),
            load_1m=load_1m,
            load_per_core=load_per_core,
            cpu_count=cpu_count,
            process_count=process_count,
            battery_percent=battery_percent,
            power_plugged=power_plugged,
            top_cpu=_top_processes(psutil, "cpu_percent"),
            top_memory=_top_processes(psutil, "memory_percent"),
            process_audit_summary="not-collected",
            pressure=pressure,
            score=score,
            checks=checks,
            recommendation=recommendation,
            action="pending" if apply_actions else "observe",
            uptime_seconds=int(time.monotonic() - _start_time) if _start_time else 0,
            sampled_at=datetime.now(timezone.utc).isoformat(),
        )
        if apply_actions:
            snapshot = PerformanceSnapshot(**{**asdict(snapshot), "action": _safe_action_for_snapshot(snapshot)})
        return snapshot
    except Exception as exc:
        logger.debug("Performance snapshot failed: %s", exc)
        return None


def _write_baseline(snapshot: PerformanceSnapshot) -> None:
    payload = {
        "schema_version": 1,
        "purpose": "Initial Hermes gateway system-performance baseline for future tuning.",
        "baseline": asdict(snapshot),
        "interpretation": {
            "score_scale": "0-100, higher is better",
            "pressure_levels": ["normal", "high", "critical"],
            "safe_default_actions": [
                "observe and log full telemetry",
                "collect Hermes Python garbage during high/critical pressure",
                "lower Hermes process priority during critical CPU pressure",
                "write process-pressure audit evidence for high/critical pressure",
            ],
            "not_done_by_default": [
                "kill user applications",
                "delete files or caches",
                "pause sync services",
                "run expensive local-model reasoning on every sample",
            ],
        },
    }
    try:
        path = _baseline_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        logger.info("[PERF_BASELINE] wrote=%s score=%d pressure=%s", path, snapshot.score, snapshot.pressure)
    except Exception as exc:
        logger.debug("Failed to write performance baseline: %s", exc)


def _write_process_audit(audit: ProcessAudit, snapshot: PerformanceSnapshot) -> None:
    payload = {
        "schema_version": 1,
        "purpose": "Hermes process-pressure audit for safe performance tuning.",
        "snapshot": asdict(snapshot),
        "audit": asdict(audit),
        "policy": {
            "safe_auto_actions": [
                "lower Hermes priority under critical CPU pressure",
                "collect Hermes Python garbage under high/critical pressure",
                "write audit evidence for later model/user review",
            ],
            "requires_user_or_policy_approval": [
                "kill or quit non-Hermes applications",
                "pause Google Drive, Dropbox, OneDrive, iCloud, or browser sync",
                "delete caches or files",
                "terminate build/test/developer jobs",
            ],
        },
    }
    try:
        path = _audit_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        logger.warning(
            "[PERF_AUDIT] wrote=%s pressure=%s score=%d process_count=%d candidates=%d summary=%s",
            path, snapshot.pressure, snapshot.score, audit.process_count, len(audit.candidates), audit.summary,
        )
    except Exception as exc:
        logger.debug("Failed to write process audit: %s", exc)


def log_performance_snapshot(prefix: str = "", apply_actions: bool = True) -> Optional[PerformanceSnapshot]:
    snapshot = collect_performance_snapshot(apply_actions=apply_actions)
    tag = f"{prefix} " if prefix else ""
    if snapshot is None:
        logger.info("[PERF] %sunavailable", tag)
        return None

    if snapshot.pressure in {"high", "critical"} or "process-high" in snapshot.checks or "process-critical" in snapshot.checks:
        psutil = _import_psutil()
        if psutil is not None:
            audit = _audit_processes(psutil)
            snapshot = PerformanceSnapshot(**{**asdict(snapshot), "process_audit_summary": audit.summary})
            _write_process_audit(audit, snapshot)

    load = "unavailable" if snapshot.load_1m is None else f"{snapshot.load_1m:.2f}"
    load_core = "unavailable" if snapshot.load_per_core is None else f"{snapshot.load_per_core:.2f}"
    battery = "unavailable" if snapshot.battery_percent is None else f"{snapshot.battery_percent:.0f}%"
    plugged = "unavailable" if snapshot.power_plugged is None else str(snapshot.power_plugged).lower()
    logger.info(
        "[PERF] %spressure=%s score=%d checks=%s cpu=%.1f%% load1=%s load_per_core=%s cores=%d "
        "mem=%.1f%% avail=%dMB swap=%.1f%% disk=%.1f%% processes=%d battery=%s plugged=%s "
        "top_cpu=%s top_mem=%s audit=%s action=%s recommendation=%s uptime=%ds",
        tag, snapshot.pressure, snapshot.score, ",".join(snapshot.checks), snapshot.cpu_percent,
        load, load_core, snapshot.cpu_count, snapshot.memory_percent, snapshot.memory_available_mb,
        snapshot.swap_percent, snapshot.disk_percent, snapshot.process_count, battery, plugged,
        snapshot.top_cpu, snapshot.top_memory, snapshot.process_audit_summary, snapshot.action,
        snapshot.recommendation, snapshot.uptime_seconds,
    )
    if prefix == "baseline":
        _write_baseline(snapshot)
    return snapshot


def _monitor_loop(stop_event: threading.Event, interval: float) -> None:
    while not stop_event.wait(interval):
        try:
            log_performance_snapshot(apply_actions=True)
        except Exception as exc:
            logger.debug("Performance monitor iteration failed: %s", exc)


def start_performance_monitoring(interval_seconds: float = 60.0, auto_actions: bool = True) -> bool:
    global _monitor_thread, _stop_event, _start_time, _interval_seconds, _auto_actions_enabled
    with _lock:
        if _monitor_thread is not None and _monitor_thread.is_alive():
            return False
        if _import_psutil() is None:
            logger.warning("[PERF] Performance monitoring unavailable: psutil is not installed")
            return False
        _start_time = time.monotonic()
        _interval_seconds = max(15.0, float(interval_seconds))
        _auto_actions_enabled = bool(auto_actions)
        _stop_event = threading.Event()
        log_performance_snapshot(prefix="baseline", apply_actions=True)
        _monitor_thread = threading.Thread(
            target=_monitor_loop,
            args=(_stop_event, _interval_seconds),
            name="gateway-performance-monitor",
            daemon=True,
        )
        _monitor_thread.start()
        logger.info(
            "[PERF] Periodic performance monitoring started (interval: %ds, auto_actions=%s)",
            int(_interval_seconds), str(_auto_actions_enabled).lower(),
        )
        return True


def stop_performance_monitoring(timeout: float = 2.0) -> None:
    global _monitor_thread, _stop_event
    with _lock:
        if _stop_event is None or _monitor_thread is None:
            return
        try:
            log_performance_snapshot(prefix="shutdown", apply_actions=False)
        except Exception:
            pass
        _stop_event.set()
        thread = _monitor_thread
        _monitor_thread = None
        _stop_event = None
    try:
        thread.join(timeout=timeout)
    except Exception:
        pass
    logger.info("[PERF] Periodic performance monitoring stopped")


def is_running() -> bool:
    with _lock:
        return _monitor_thread is not None and _monitor_thread.is_alive()
