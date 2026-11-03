"""Regression and lifecycle tests for the F0 digital runtime."""

from __future__ import annotations

import pytest

from d2c.digital.bridge import DigitalMemoryConfig, seed_memory_state, step_memory
from d2c.digital.director import DigitalEpisode
from d2c.runtime.digital_runtime import (
    DigitalRuntime,
    DigitalRuntimeConfig,
    DigitalStabilityMonitor,
)
from d2c.runtime.phase_controller import DigitalPhaseConfig, DigitalPhaseController, DigitalRuntimePhase


def memory_config() -> DigitalMemoryConfig:
    return DigitalMemoryConfig(
        gamma=[2.0, 0.5, 0.1],
        w=[0.04, 0.08, 0.12],
        dt=0.01,
        leak_rate=1.0,
        form="input_driven",
    )


def episode() -> DigitalEpisode:
    return DigitalEpisode(
        forcings=[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
        tokens=["a", "b", None],
        target="b",
        query=[0.0, 1.0],
        metadata={"task": 1},
    )


def close_state(left, right) -> None:
    assert left.t == pytest.approx(right.t)
    assert left.u == pytest.approx(right.u)
    assert len(left.chi) == len(right.chi)
    for lhs, rhs in zip(left.chi, right.chi):
        assert lhs == pytest.approx(rhs)


def test_runtime_is_numerically_equivalent_to_existing_stepper() -> None:
    config = memory_config()
    task = episode()
    expected = seed_memory_state(state_dim=2, channel_count=config.channel_count)
    for forcing in task.forcings:
        expected = step_memory(expected, forcing, config)

    runtime = DigitalRuntime(
        memory_config=config,
        state_dim=2,
        config=DigitalRuntimeConfig(
            phase=DigitalPhaseConfig(
                warmup_steps=100,
                explore_steps=100,
                consolidate_steps=2,
                evaluation_interval=0,
            )
        ),
    )
    result = runtime.run_episode(task, reset_memory=True)
    close_state(result.final_state, expected)
    assert result.trace.steps[-1].u == pytest.approx(expected.u)
    assert runtime.stability_violations == 0
    assert runtime.stability_checks == len(task.forcings)


def test_isolated_reset_reproduces_episode_and_continuous_mode_carries_state() -> None:
    runtime = DigitalRuntime(memory_config(), state_dim=2)
    task = episode()
    first = runtime.run_episode(task, episode_id="first", reset_memory=True)
    second = runtime.run_episode(task, episode_id="second", reset_memory=True)
    close_state(first.final_state, second.final_state)

    carried = runtime.run_episode(task, episode_id="third", reset_memory=False)
    assert carried.final_state.u != pytest.approx(first.final_state.u)
    assert carried.reset_before_episode is False
    assert runtime.episode_count == 3


def test_phase_controller_cycles_warmup_explore_consolidate() -> None:
    controller = DigitalPhaseController(
        DigitalPhaseConfig(
            warmup_steps=1,
            explore_steps=2,
            consolidate_steps=1,
            evaluation_interval=0,
        )
    )
    assert controller.phase == DigitalRuntimePhase.WARMUP
    controller.complete_step()
    assert controller.phase == DigitalRuntimePhase.EXPLORE
    controller.complete_step()
    assert controller.phase == DigitalRuntimePhase.EXPLORE
    controller.complete_step()
    assert controller.phase == DigitalRuntimePhase.CONSOLIDATE
    controller.complete_step()
    assert controller.phase == DigitalRuntimePhase.EXPLORE


def test_evaluate_and_maintenance_are_observational_transient_phases() -> None:
    controller = DigitalPhaseController(
        DigitalPhaseConfig(
            warmup_steps=0,
            explore_steps=100,
            consolidate_steps=2,
            evaluation_interval=2,
        )
    )
    controller.complete_step()
    assert controller.phase == DigitalRuntimePhase.EXPLORE
    controller.complete_step()
    assert controller.phase == DigitalRuntimePhase.EVALUATE
    controller.complete_step()
    assert controller.phase == DigitalRuntimePhase.EXPLORE

    controller.request_maintenance("PRUNE", reason="unit-test proposal")
    controller.complete_step()
    assert controller.phase == DigitalRuntimePhase.PRUNE
    controller.complete_step()
    assert controller.phase == DigitalRuntimePhase.EXPLORE
    assert any(event.current == "PRUNE" for event in controller.events)


def test_trace_windows_never_bridge_episode_memory_resets() -> None:
    runtime = DigitalRuntime(memory_config(), state_dim=2)
    runtime.run_episode(episode(), episode_id="a", reset_memory=True)
    runtime.run_episode(episode(), episode_id="b", reset_memory=True)
    assert runtime.trace_store.build_windows(4) == []
    windows = runtime.trace_store.build_windows(3)
    assert all(len({row.continuity_id for row in window.records}) == 1 for window in windows)


def test_snapshot_restore_preserves_dynamics_and_phase_position() -> None:
    config = DigitalRuntimeConfig(
        phase=DigitalPhaseConfig(
            warmup_steps=1,
            explore_steps=10,
            consolidate_steps=2,
            evaluation_interval=0,
        )
    )
    left = DigitalRuntime(memory_config(), state_dim=2, config=config, session_id="checkpoint")
    left.run_episode(episode(), reset_memory=True)
    snapshot = left.snapshot()

    right = DigitalRuntime(memory_config(), state_dim=2, config=config)
    right.restore(snapshot)
    assert right.step_count == left.step_count
    assert right.phase == left.phase
    close_state(right.memory_state, left.memory_state)

    next_task = DigitalEpisode(
        forcings=[[0.25, -0.5], [0.0, 0.0]],
        tokens=["c", None],
    )
    left_result = left.run_episode(next_task, reset_memory=False)
    right_result = right.run_episode(next_task, reset_memory=False)
    close_state(left_result.final_state, right_result.final_state)
    assert right.phase_counts == left.phase_counts


def test_restore_rejects_incompatible_checkpoint() -> None:
    runtime = DigitalRuntime(memory_config(), state_dim=2)
    with pytest.raises(ValueError, match="state_dim"):
        runtime.restore({"schema_version": 1, "state_dim": 3})


def test_stability_monitor_does_not_apply_form_b_margin_to_input_driven_form() -> None:
    config = DigitalMemoryConfig(
        gamma=[2.0, 0.5, 0.1],
        w=[0.04, 0.08, 0.12],
        dt=0.01,
        leak_rate=1.0,
        form="input_driven",
    )
    state = seed_memory_state(state_dim=2, channel_count=3)
    assessment = DigitalStabilityMonitor().assess(state, config)
    assert assessment.passed
    assert not assessment.analytical_margin_applicable
    assert assessment.stability_ratio > 0.9


def test_stability_monitor_enforces_negative_feedback_margin_and_euler_bound() -> None:
    monitor = DigitalStabilityMonitor(stability_margin=0.9)
    bad_margin = DigitalMemoryConfig(
        gamma=[0.1],
        w=[0.2],
        dt=0.01,
        leak_rate=1.0,
        form="negative_feedback",
    )
    state = seed_memory_state(state_dim=1, channel_count=1)
    assert not monitor.assess(state, bad_margin).passed

    bad_euler = DigitalMemoryConfig(
        gamma=[0.5],
        w=[0.01],
        dt=2.1,
        leak_rate=1.0,
        form="input_driven",
    )
    assert not monitor.assess(state, bad_euler).passed


def test_maintenance_requests_do_not_change_kernel_parameters() -> None:
    config = memory_config()
    before = config.to_mapping()
    runtime = DigitalRuntime(config, state_dim=2)
    runtime.run_episode(episode(), reset_memory=True)
    runtime.request_grow(reason="capacity diagnostic")
    runtime.run_episode(
        DigitalEpisode(forcings=[[0.0, 0.0]], tokens=[None]),
        reset_memory=False,
    )
    assert runtime.memory_config.to_mapping() == before


def test_evaluation_callback_runs_inside_phase_and_restores_runtime() -> None:
    runtime = DigitalRuntime(
        memory_config(),
        state_dim=2,
        config=DigitalRuntimeConfig(
            phase=DigitalPhaseConfig(
                warmup_steps=5,
                explore_steps=20,
                consolidate_steps=2,
                evaluation_interval=0,
            )
        ),
    )
    observed = runtime.evaluate(
        lambda: runtime.phase.value,
        reason="unit test readout evaluation",
    )
    assert observed == "EVALUATE"
    assert runtime.phase == DigitalRuntimePhase.WARMUP
    assert runtime.phase_controller.events[-1].current == "WARMUP"


def test_scheduled_evaluation_phase_is_consumed_without_nested_transition() -> None:
    runtime = DigitalRuntime(
        memory_config(),
        state_dim=2,
        config=DigitalRuntimeConfig(
            phase=DigitalPhaseConfig(
                warmup_steps=1,
                explore_steps=20,
                consolidate_steps=2,
                evaluation_interval=2,
            )
        ),
    )
    runtime.step([1.0, 0.0])
    runtime.step([0.0, 1.0])
    assert runtime.phase == DigitalRuntimePhase.EVALUATE
    observed = runtime.evaluate(lambda: runtime.phase.value)
    assert observed == "EVALUATE"
    assert runtime.phase == DigitalRuntimePhase.EVALUATE
    runtime.step([0.0, 0.0])
    assert runtime.phase == DigitalRuntimePhase.EXPLORE


def test_negative_feedback_runtime_matches_stepper_and_passes_discrete_stability() -> None:
    config = DigitalMemoryConfig(
        gamma=[0.1, 0.5],
        w=[0.001, 0.002],
        dt=0.01,
        leak_rate=1.0,
        form="negative_feedback",
    )
    forcings = [[1.0, 0.0], [0.0, 0.5], [0.0, 0.0]]
    expected = seed_memory_state(state_dim=2, channel_count=config.channel_count)
    for forcing in forcings:
        expected = step_memory(expected, forcing, config)

    runtime = DigitalRuntime(config, state_dim=2)
    for forcing in forcings:
        runtime.step(forcing)
    close_state(runtime.memory_state, expected)
    assert runtime.stability_violations == 0


def test_checkpoint_restores_runtime_configuration_and_monitor() -> None:
    original = DigitalRuntime(
        memory_config(),
        state_dim=2,
        config=DigitalRuntimeConfig(
            trace_capacity=64,
            prediction_error_alpha=0.25,
            phase=DigitalPhaseConfig(
                warmup_steps=3,
                explore_steps=12,
                consolidate_steps=3,
                evaluation_interval=0,
            ),
        ),
    )
    original.run_episode(episode(), reset_memory=True)
    snapshot = original.snapshot()
    restored = DigitalRuntime(memory_config(), state_dim=2)
    restored.restore(snapshot)
    assert restored.config.trace_capacity == 64
    assert restored.config.prediction_error_alpha == 0.25
    assert restored.config.phase == original.config.phase
    assert restored.monitor.stability_margin == original.monitor.stability_margin
