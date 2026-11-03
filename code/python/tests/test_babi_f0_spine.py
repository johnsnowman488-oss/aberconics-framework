"""Integration checks for the deterministic F0 runtime spine audit."""

from __future__ import annotations

from d2c.digital.bridge import DigitalMemoryConfig
from d2c.experiments.babi_f0 import BabiF0Config, exercise_f0_spine
from d2c.runtime.digital_runtime import DigitalRuntimeConfig
from d2c.runtime.phase_controller import DigitalPhaseConfig


def test_f0_spine_audit_exercises_all_phases_and_runtime_components() -> None:
    memory = DigitalMemoryConfig(
        gamma=[2.0, 0.5, 0.1],
        w=[0.04, 0.08, 0.12],
        dt=0.01,
        leak_rate=1.0,
        form="input_driven",
    )
    config = BabiF0Config(
        dim=2,
        runtime=DigitalRuntimeConfig(
            trace_capacity=128,
            phase=DigitalPhaseConfig(
                warmup_steps=1,
                explore_steps=3,
                consolidate_steps=1,
                evaluation_interval=0,
            ),
        ),
    )

    report = exercise_f0_spine(config, memory, seed=7)

    assert report["status"] == "passed"
    assert all(report["phase_coverage"].values())
    assert report["maintenance"]["prune_proposal_steps"] >= 1
    assert report["maintenance"]["grow_proposal_steps"] >= 1
    assert report["maintenance"]["kernel_unchanged"] is True
    assert report["evaluation"]["callback_phase"] == "EVALUATE"
    assert report["trace_store"]["sample_is_contiguous"] is True
    assert report["checkpoint_restore"]["snapshot_round_trip_equal"] is True
    assert report["checkpoint_restore"]["deterministic_continuation_equal"] is True
    assert report["diagnostics"]["stability_violations"] == 0
    assert report["diagnostics"]["prediction_error_samples"] > 0
