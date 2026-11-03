"""10,000-step deterministic F0 runtime smoke test."""

from d2c.digital.bridge import DigitalMemoryConfig
from d2c.runtime.digital_runtime import DigitalRuntime, DigitalRuntimeConfig
from d2c.runtime.phase_controller import DigitalPhaseConfig


def test_f0_runtime_completes_10000_steps_with_zero_stability_violations() -> None:
    runtime = DigitalRuntime(
        memory_config=DigitalMemoryConfig(
            gamma=[2.0, 0.5, 0.1],
            w=[0.04, 0.08, 0.12],
            dt=0.01,
            leak_rate=1.0,
            form="input_driven",
        ),
        state_dim=4,
        config=DigitalRuntimeConfig(
            trace_capacity=256,
            phase=DigitalPhaseConfig(
                warmup_steps=10,
                explore_steps=32,
                consolidate_steps=4,
                evaluation_interval=128,
            ),
        ),
    )
    for step in range(10_000):
        forcing = [1.0, 0.0, -0.5, 0.25] if step % 7 == 0 else [0.0] * 4
        runtime.step(forcing, token="event" if step % 7 == 0 else None)

    summary = runtime.runtime_summary()
    assert summary["step_count"] == 10_000
    assert summary["stability_checks"] == 10_000
    assert summary["stability_violations"] == 0
    assert summary["trace_records_retained"] == 256
    assert summary["phase_counts"]
    assert any(event["current"] == "EVALUATE" for event in summary["phase_events"])
