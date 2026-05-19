"""Lightweight system performance monitoring for the Hermes gateway.

This monitor is intentionally cheap: it samples local CPU, memory, swap, disk,
and hottest processes in a daemon thread, then emits a grep-friendly
``[PERF] ...`` log line.  The goal is to keep Hermes aware of machine pressure
without making the local model itself run constantly and become the problem.

Config: ``logging.performance_monitor`` in ``config.yaml``.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

_monitor_thread: Optional[threading.Thread] = None
_stop_event: Optional[threading.Event] = None
_start_time: Optional[float] = None
_interval_seconds: float = 60.0
_lock = threading.Lock()


@dataclass(frozen=True)
class PerformanceSnapshot:
    cpu_percent: float
    memory_percent: float
    memory_available_mb: int
    swap_percent: float
    disk_percent: float
    load_1m: Optional[float]
    cpu_count: int
    top_cpu: str
    top_memory: str
    pressure: str
    recommendation: str
    uptime_seconds: int


def _import_psutil():
    try:
        import psutil  # type: ignore

        return psutil
    except Exception:
        return None


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


def _top_processes(psutil, metric: str, limit: int = 3) -> str:
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


def _classify_pressure(cpu: float, mem: float, swap: float, disk: float, available_mb: int) -> tuple[str, str]:
    if available_mb < 750 or mem >= 92 or swap >= 70:
        return "critical", "free-memory: quit memory-heavy apps, pause browser tabs, avoid new local-model runs"
    if cpu >= 92:
        return "critical", "reduce-cpu: pause runaway jobs, defer indexing/builds, avoid model inference until load drops"
    if disk >= 95:
        return "critical", "free-disk: clear caches/downloads before builds or media work"
    if mem >= 82 or swap >= 40:
        return "high", "watch-memory: close idle heavy apps or restart leaking background services soon"
    if cpu >= 80:
        return "high", "watch-cpu: let current jobs finish before starting more parallel work"
    if disk >= 88:
        return "high", "watch-disk: cleanup recommended soon"
    return "normal", "healthy: no action needed"


def collect_performance_snapshot() -> Optional[PerformanceSnapshot]:
    """Collect a cheap point-in-time system performance snapshot.

    Returns ``None`` when psutil is unavailable rather than raising; callers can
    disable the monitor gracefully.
    """
    psutil = _import_psutil()
    if psutil is None:
        return None

    try:
        cpu_count = int(psutil.cpu_count() or 1)
        cpu_percent = float(psutil.cpu_percent(interval=None))
        vm = psutil.virtual_memory()
        swap = psutil.swap_memory()
        disk = psutil.disk_usage("/")
        try:
            load_1m = float(psutil.getloadavg()[0])
        except Exception:
            load_1m = None

        available_mb = int(getattr(vm, "available", 0) / (1024 * 1024))
        pressure, recommendation = _classify_pressure(
            cpu=cpu_percent,
            mem=float(vm.percent),
            swap=float(swap.percent),
            disk=float(disk.percent),
            available_mb=available_mb,
        )
        uptime = int(time.monotonic() - _start_time) if _start_time else 0
        return PerformanceSnapshot(
            cpu_percent=cpu_percent,
            memory_percent=float(vm.percent),
            memory_available_mb=available_mb,
            swap_percent=float(swap.percent),
            disk_percent=float(disk.percent),
            load_1m=load_1m,
            cpu_count=cpu_count,
            top_cpu=_top_processes(psutil, "cpu_percent"),
            top_memory=_top_processes(psutil, "memory_percent"),
            pressure=pressure,
            recommendation=recommendation,
            uptime_seconds=uptime,
        )
    except Exception as exc:
        logger.debug("Performance snapshot failed: %s", exc)
        return None


def log_performance_snapshot(prefix: str = "") -> Optional[PerformanceSnapshot]:
    """Log a single structured ``[PERF]`` line and return the snapshot."""
    snapshot = collect_performance_snapshot()
    tag = f"{prefix} " if prefix else ""
    if snapshot is None:
        logger.info("[PERF] %sunavailable", tag)
        return None

    load = "unavailable" if snapshot.load_1m is None else f"{snapshot.load_1m:.2f}"
    logger.info(
        "[PERF] %spressure=%s cpu=%.1f%% load1=%s cores=%d mem=%.1f%% avail=%dMB "
        "swap=%.1f%% disk=%.1f%% top_cpu=%s top_mem=%s recommendation=%s uptime=%ds",
        tag,
        snapshot.pressure,
        snapshot.cpu_percent,
        load,
        snapshot.cpu_count,
        snapshot.memory_percent,
        snapshot.memory_available_mb,
        snapshot.swap_percent,
        snapshot.disk_percent,
        snapshot.top_cpu,
        snapshot.top_memory,
        snapshot.recommendation,
        snapshot.uptime_seconds,
    )
    return snapshot


def _monitor_loop(stop_event: threading.Event, interval: float) -> None:
    while not stop_event.wait(interval):
        try:
            log_performance_snapshot()
        except Exception as exc:
            logger.debug("Performance monitor iteration failed: %s", exc)


def start_performance_monitoring(interval_seconds: float = 60.0) -> bool:
    """Start lightweight system performance monitoring in a daemon thread."""
    global _monitor_thread, _stop_event, _start_time, _interval_seconds

    with _lock:
        if _monitor_thread is not None and _monitor_thread.is_alive():
            return False
        if _import_psutil() is None:
            logger.warning("[PERF] Performance monitoring unavailable: psutil is not installed")
            return False

        _start_time = time.monotonic()
        _interval_seconds = max(15.0, float(interval_seconds))
        _stop_event = threading.Event()
        log_performance_snapshot(prefix="baseline")
        _monitor_thread = threading.Thread(
            target=_monitor_loop,
            args=(_stop_event, _interval_seconds),
            name="gateway-performance-monitor",
            daemon=True,
        )
        _monitor_thread.start()
        logger.info("[PERF] Periodic performance monitoring started (interval: %ds)", int(_interval_seconds))
        return True


def stop_performance_monitoring(timeout: float = 2.0) -> None:
    """Stop the monitor thread and log one final snapshot."""
    global _monitor_thread, _stop_event

    with _lock:
        if _stop_event is None or _monitor_thread is None:
            return
        try:
            log_performance_snapshot(prefix="shutdown")
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
