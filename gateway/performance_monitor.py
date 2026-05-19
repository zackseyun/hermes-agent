"""Lightweight system performance monitoring and safe response for Hermes.

This monitor is intentionally cheap: it samples local CPU, memory, swap, disk,
battery, load, process count, and hottest processes in a daemon thread. It then
emits grep-friendly ``[PERF] ...`` log lines, writes a persistent baseline file,
and performs only conservative self-protection actions by default.

Why not have the local model reason every few seconds? Because that can become
the performance problem. This monitor does frequent telemetry cheaply and leaves
heavier agent reasoning for pressure events or explicit user workflows.

Config: ``logging.performance_monitor`` in ``config.yaml``.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass
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
    pressure: str
    score: int
    checks: tuple[str, ...]
    recommendation: str
    action: str
    uptime_seconds: int
    sampled_at: str


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


def _fmt_process(proc: object, metric: str) -> str:
    """Format a psutil process for compact log output."""
    try:
        info = getattr(proc, "info", {}) or {}
        name = str(info.get("name") or "unknown").replace(" ", "_")[:28]
        pid = info.get("pid")
        value = info.get(metric)
        if value is None:
            value = 0.0
        if metric == "memory_percent":
            return f"{name}:{pid}:{float(value):.1f}%mem"
        return f"{name}:{pid}:{float(value):.1f}%cpu"
    except Exception:
        return "unknown"


def _top_processes(psutil, metric: str, limit: int = 5) -> str:
    attrs = ["pid", "name", metric]
    try:
        procs = []
        for proc in psutil.process_iter(attrs=attrs):
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


def _score_from_pressure(cpu: float, mem: float, swap: float, disk: float, available_mb: int, load_per_core: Optional[float], process_count: int = 0) -> int:
    """Return a 0-100 health score, where 100 is healthier."""
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


def _classify_pressure(
    cpu: float,
    mem: float,
    swap: float,
    disk: float,
    available_mb: int,
    load_per_core: Optional[float] = None,
    process_count: int = 0,
) -> tuple[str, int, tuple[str, ...], str]:
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
    if any(c.endswith("critical") for c in checks):
        pressure = "critical"
    elif any(c.endswith("high") for c in checks):
        pressure = "high"
    else:
        pressure = "normal"

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
    """Take conservative action that cannot close user apps or delete files."""
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
            os.nice(5)  # Lower Hermes priority so user-facing apps stay responsive.
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
    """Collect a cheap point-in-time system performance snapshot.

    Returns ``None`` when psutil is unavailable rather than raising; callers can
    disable the monitor gracefully.
    """
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
        uptime = int(time.monotonic() - _start_time) if _start_time else 0
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
            pressure=pressure,
            score=score,
            checks=checks,
            recommendation=recommendation,
            action="pending" if apply_actions else "observe",
            uptime_seconds=uptime,
            sampled_at=datetime.now(timezone.utc).isoformat(),
        )
        if apply_actions:
            action = _safe_action_for_snapshot(snapshot)
            snapshot = PerformanceSnapshot(**{**asdict(snapshot), "action": action})
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
            ],
            "not_done_by_default": [
                "kill user applications",
                "delete files or caches",
                "run expensive local-model reasoning on every sample",
            ],
        },
    }
    path = _baseline_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        logger.info("[PERF_BASELINE] wrote=%s score=%d pressure=%s", path, snapshot.score, snapshot.pressure)
    except Exception as exc:
        logger.debug("Failed to write performance baseline %s: %s", path, exc)


def log_performance_snapshot(prefix: str = "", apply_actions: bool = True) -> Optional[PerformanceSnapshot]:
    """Log a single structured ``[PERF]`` line and return the snapshot."""
    snapshot = collect_performance_snapshot(apply_actions=apply_actions)
    tag = f"{prefix} " if prefix else ""
    if snapshot is None:
        logger.info("[PERF] %sunavailable", tag)
        return None

    load = "unavailable" if snapshot.load_1m is None else f"{snapshot.load_1m:.2f}"
    load_core = "unavailable" if snapshot.load_per_core is None else f"{snapshot.load_per_core:.2f}"
    battery = "unavailable" if snapshot.battery_percent is None else f"{snapshot.battery_percent:.0f}%"
    plugged = "unavailable" if snapshot.power_plugged is None else str(snapshot.power_plugged).lower()
    logger.info(
        "[PERF] %spressure=%s score=%d checks=%s cpu=%.1f%% load1=%s load_per_core=%s cores=%d "
        "mem=%.1f%% avail=%dMB swap=%.1f%% disk=%.1f%% processes=%d battery=%s plugged=%s "
        "top_cpu=%s top_mem=%s action=%s recommendation=%s uptime=%ds",
        tag,
        snapshot.pressure,
        snapshot.score,
        ",".join(snapshot.checks),
        snapshot.cpu_percent,
        load,
        load_core,
        snapshot.cpu_count,
        snapshot.memory_percent,
        snapshot.memory_available_mb,
        snapshot.swap_percent,
        snapshot.disk_percent,
        snapshot.process_count,
        battery,
        plugged,
        snapshot.top_cpu,
        snapshot.top_memory,
        snapshot.action,
        snapshot.recommendation,
        snapshot.uptime_seconds,
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
    """Start lightweight system performance monitoring in a daemon thread."""
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
            int(_interval_seconds),
            str(_auto_actions_enabled).lower(),
        )
        return True


def stop_performance_monitoring(timeout: float = 2.0) -> None:
    """Stop the monitor thread and log one final snapshot."""
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
