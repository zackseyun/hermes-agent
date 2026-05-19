"""Tests for gateway.performance_monitor system telemetry and safe actions."""

from __future__ import annotations

import json
import logging

import pytest

from gateway import performance_monitor as pm


@pytest.fixture(autouse=True)
def _ensure_monitor_stopped():
    pm.stop_performance_monitoring(timeout=1.0)
    yield
    pm.stop_performance_monitoring(timeout=1.0)


def test_log_performance_snapshot_emits_perf_line(caplog):
    caplog.set_level(logging.INFO, logger="gateway.performance_monitor")
    pm.log_performance_snapshot(apply_actions=False)
    assert any("[PERF]" in r.getMessage() for r in caplog.records)


def test_pressure_classifier_recommends_memory_action():
    pressure, score, checks, recommendation = pm._classify_pressure(
        cpu=20.0,
        mem=94.0,
        swap=5.0,
        disk=30.0,
        available_mb=500,
    )
    assert pressure == "critical"
    assert score < 100
    assert "memory-critical" in checks
    assert "free-memory-swap" in recommendation


def test_pressure_classifier_includes_cpu_load_and_disk_checks():
    pressure, _score, checks, recommendation = pm._classify_pressure(
        cpu=50.0,
        mem=50.0,
        swap=0.0,
        disk=96.0,
        available_mb=4096,
        load_per_core=2.5,
    )
    assert pressure == "critical"
    assert "cpu-critical" in checks
    assert "disk-critical" in checks
    assert recommendation.startswith("reduce-cpu") or recommendation.startswith("free-disk")


def test_start_logs_baseline_and_returns_true(caplog, tmp_path, monkeypatch):
    if pm._import_psutil() is None:
        pytest.skip("psutil unavailable")
    monkeypatch.setattr(pm, "_baseline_path", lambda: tmp_path / "performance-baseline.json")
    caplog.set_level(logging.INFO, logger="gateway.performance_monitor")
    assert pm.start_performance_monitoring(interval_seconds=3600.0) is True
    assert pm.is_running() is True
    messages = [r.getMessage() for r in caplog.records]
    assert any("[PERF] baseline " in m for m in messages), messages
    assert any("Periodic performance monitoring started" in m for m in messages), messages
    baseline = tmp_path / "performance-baseline.json"
    assert baseline.exists()
    payload = json.loads(baseline.read_text())
    assert payload["baseline"]["score"] >= 0
    assert "safe_default_actions" in payload["interpretation"]


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


def test_auto_actions_can_be_disabled(monkeypatch):
    snapshot = pm.PerformanceSnapshot(
        cpu_percent=99.0,
        memory_percent=50.0,
        memory_available_mb=4096,
        swap_percent=0.0,
        disk_percent=30.0,
        load_1m=20.0,
        load_per_core=2.5,
        cpu_count=8,
        process_count=100,
        battery_percent=None,
        power_plugged=None,
        top_cpu="x:1:99.0%cpu",
        top_memory="x:1:1.0%mem",
        process_audit_summary="not-collected",
        pressure="critical",
        score=50,
        checks=("memory-ok", "cpu-critical", "disk-ok"),
        recommendation="reduce-cpu",
        action="pending",
        uptime_seconds=0,
        sampled_at="2026-05-19T00:00:00+00:00",
    )
    monkeypatch.setattr(pm, "_auto_actions_enabled", False)
    assert pm._safe_action_for_snapshot(snapshot) == "disabled"


def test_unavailable_psutil_warns_and_does_not_start(caplog, monkeypatch):
    monkeypatch.setattr(pm, "_import_psutil", lambda: None)
    caplog.set_level(logging.WARNING, logger="gateway.performance_monitor")
    assert pm.start_performance_monitoring(interval_seconds=3600.0) is False
    assert pm.is_running() is False
    assert any("Performance monitoring unavailable" in r.getMessage() for r in caplog.records)


def test_process_auditor_classifies_browser_and_sync_candidates():
    assert pm._categorize_process({"name": "Google Chrome Helper (Renderer)", "cmdline": []}) == "browser-helper"
    assert pm._categorize_process({"name": "Google Drive", "cmdline": []}) == "sync-service"
    risk, recommendation, safe = pm._process_recommendation("browser-helper", cpu=45.0, mem=1.2, status="running")
    assert risk == "medium"
    assert "tabs" in recommendation
    assert safe is False


def test_process_auditor_marks_hermes_safe_auto_action():
    risk, recommendation, safe = pm._process_recommendation("hermes", cpu=55.0, mem=1.2, status="running")
    assert risk == "high"
    assert "Hermes" in recommendation
    assert safe is True
