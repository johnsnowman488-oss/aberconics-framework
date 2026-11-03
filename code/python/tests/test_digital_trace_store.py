"""Tests for digital F0 contiguous trace storage."""

from __future__ import annotations

import pytest

from d2c.runtime.digital_trace_store import DigitalRuntimeRecord, DigitalTraceStore


def record(step: int, *, continuity: int = 0, session: str = "s") -> DigitalRuntimeRecord:
    return DigitalRuntimeRecord(
        step_index=step,
        t=float(step),
        session_id=session,
        continuity_id=continuity,
        episode_id=f"e{step}",
        phase="EXPLORE",
        u=[float(step), 0.0],
        chi=[[float(step), 0.0]],
        forcing=[1.0, 0.0],
        token="x",
    )


def test_circular_buffer_retains_newest_records_in_order() -> None:
    store = DigitalTraceStore(capacity=3)
    for step in range(5):
        store.append(record(step))
    assert [row.step_index for row in store.records] == [2, 3, 4]
    assert len(store) == 3


def test_windows_do_not_cross_memory_resets() -> None:
    store = DigitalTraceStore(capacity=20)
    store.append(record(0, continuity=0))
    store.append(record(1, continuity=0))
    store.append(record(2, continuity=1))
    store.append(record(3, continuity=1))
    assert store.build_windows(3) == []
    windows = store.build_windows(2)
    assert [(window.start_step, window.end_step) for window in windows] == [(0, 1), (2, 3)]


def test_windows_reject_step_gaps_and_can_filter_session() -> None:
    store = DigitalTraceStore(capacity=20)
    store.append(record(0, session="a"))
    store.append(record(2, session="a"))
    store.append(record(3, session="b"))
    store.append(record(4, session="b"))
    assert store.build_windows(2, session_id="a") == []
    windows = store.build_windows(2, session_id="b")
    assert len(windows) == 1
    assert windows[0].records[0].step_index == 3


def test_sample_contiguous_is_seed_deterministic() -> None:
    store = DigitalTraceStore(capacity=20)
    for step in range(8):
        store.append(record(step))
    first = store.sample_contiguous(3, seed=17)
    second = store.sample_contiguous(3, seed=17)
    assert [row.step_index for row in first.records] == [row.step_index for row in second.records]


def test_invalid_record_is_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        DigitalRuntimeRecord(
            step_index=0,
            t=0.0,
            session_id="s",
            continuity_id=0,
            episode_id="e",
            phase="EXPLORE",
            u=[float("nan")],
            chi=[[0.0]],
            forcing=[0.0],
        )
