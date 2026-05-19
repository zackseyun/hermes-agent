"""Tests for gateway.performance_monitor system telemetry."""

from __future__ import annotations

import logging
import time
from collections import namedtuple

import pytest

from gateway import performance_monitor as pm


@pytest.fixture(autouse=True)
def _ensure_monitor_stopped():
    pm.stop_performance_monitoring(timeout=1.0)
    yield
    pm.stop_performance_monitoring(timeout=1.0)


def test_log_performance_snapshot_emits_perf_line(caplog):
    caplog.set_level(logging.INFO, logger="gateway.performance_monitor")
    pm.log_performance_snapshot()
    assert any("[PERF]" in r.getMessage() for r in caplog.records)


def test_pressure_classifier_recommends_memory_action():
    pressure, recommendation = pm._classify_pressure(
        cpu=20.0,
        mem=94.0,
        swap=5.0,
        disk=30.0,
        available_mb=500,
    )
    assert pressure == "critical"
    assert "free-memory" in recommendation


def test_start_logs_baseline_and_returns_true(caplog):
    if pm._import_psutil() is None:
        pytest.skip("psutil unavailable")
    caplog.set_level(logging.INFO, logger="gateway.performance_monitor")
    assert pm.start_performance_monitoring(interval_seconds=3600.0) is True
    assert pm.is_running() is True
    messages = [r.getMessage() for r in caplog.records]
    assert any("[PERF] baseline " in m for m in messages), messages
    assert any("Periodic performance monitoring started" in m for m in messages), messages


def test_double_start_is_noop():
    if pm._import_psutil() is None:
        pytest.skip("psutil unavailable")
    assert pm.start_performance_monitoring(interval_seconds=3600.0) is True
    assert pm.start_performance_monitoring(interval_seconds=3600.0) is False


def test_stop_logs_shutdown_snapshot(caplog):
    if pm._import_psutil() is None:
        pytest.skip("psutil unavailable")
    pm.start_performance_monitoring(interval_seconds=3600.0)
    caplog.clear()
    caplog.set_level(logging.INFO, logger="gateway.performance_monitor")
    pm.stop_performance_monitoring(timeout=1.0)
    assert pm.is_running() is False
    messages = [r.getMessage() for r in caplog.records]
    assert any("[PERF] shutdown " in m for m in messages), messages
    assert any("Periodic performance monitoring stopped" in m for m in messages), messages


def test_interval_has_15_second_floor():
    if pm._import_psutil() is None:
        pytest.skip("psutil unavailable")
    assert pm.start_performance_monitoring(interval_seconds=1.0) is True
    assert pm._interval_seconds == 15.0


def test_unavailable_psutil_warns_and_does_not_start(caplog, monkeypatch):
    monkeypatch.setattr(pm, "_import_psutil", lambda: None)
    caplog.set_level(logging.WARNING, logger="gateway.performance_monitor")
    assert pm.start_performance_monitoring(interval_seconds=3600.0) is False
    assert pm.is_running() is False
    assert any("Performance monitoring unavailable" in r.getMessage() for r in caplog.records)
