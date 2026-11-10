from __future__ import annotations
from pathlib import Path
import random
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from d2c.digital.bridge import DigitalMemoryConfig, DigitalMemoryState, step_memory
from d2c.digital.native_stepper import CppVectorizedMemoryStepper
from d2c.runtime.digital_runtime import DigitalRuntime


def _stepper(config):
    try:
        return CppVectorizedMemoryStepper(config)
    except OSError as exc:
        pytest.skip(f"shared gfe_core library unavailable: {exc}")


@pytest.mark.parametrize("form", ["input_driven", "negative_feedback"])
@pytest.mark.parametrize(("dim", "channels"), [(1, 1), (4, 3), (13, 7), (64, 3)])
def test_cpp_step_matches_python_reference(form, dim, channels):
    config = DigitalMemoryConfig(
        gamma=[0.1 + 0.2 * i for i in range(channels)],
        w=[0.03 + 0.01 * i for i in range(channels)],
        dt=0.017, leak_rate=1.2, form=form,
    )
    rng = random.Random(dim + channels)
    state = DigitalMemoryState(
        [rng.uniform(-0.5, 0.5) for _ in range(dim)],
        [[rng.uniform(-0.1, 0.1) for _ in range(dim)] for _ in range(channels)], 0.3,
    )
    forcing = [rng.uniform(-1.0, 1.0) for _ in range(dim)]
    expected = step_memory(state, forcing, config)
    actual = _stepper(config)(state, forcing, config)
    assert actual.u == pytest.approx(expected.u, rel=1e-13, abs=1e-13)
    assert np.asarray(actual.chi) == pytest.approx(np.asarray(expected.chi), rel=1e-13, abs=1e-13)
    assert actual.t == pytest.approx(expected.t, abs=1e-15)


def test_cpp_sequence_matches_repeated_python_steps():
    config = DigitalMemoryConfig(gamma=[2.0, 0.5, 0.1], w=[0.04, 0.08, 0.12], dt=0.05)
    initial = DigitalMemoryState([0.1] * 8, [[0.0] * 8 for _ in range(3)], 0.25)
    forcings = [[0.1 * (i % 5) for _ in range(8)] for i in range(12)]
    expected = initial
    for forcing in forcings:
        expected = step_memory(expected, forcing, config)
    actual = _stepper(config).run_sequence(initial, forcings, config)
    assert actual["final_state"].u == pytest.approx(expected.u, rel=1e-13, abs=1e-13)
    assert np.asarray(actual["final_state"].chi) == pytest.approx(np.asarray(expected.chi), rel=1e-13, abs=1e-13)
    assert len(actual["times"]) == len(forcings) + 1
    assert len(actual["u"]) == len(forcings) + 1
    assert len(actual["chi"]) == len(forcings) + 1
    assert actual["diagnostics"]["steps_executed"] == len(forcings)


def test_digital_runtime_native_backend_matches_reference():
    config = DigitalMemoryConfig(gamma=[2.0, 0.5, 0.1], w=[0.04, 0.08, 0.12], dt=0.05)
    native = _stepper(config)
    reference = DigitalRuntime(memory_config=config, state_dim=8)
    accelerated = DigitalRuntime(memory_config=config, state_dim=8,
                                 memory_backend="cpp_vectorized", memory_stepper=native)
    for i in range(20):
        forcing = [0.05 * ((i + d) % 3 - 1) for d in range(8)]
        a, b = reference.step(forcing), accelerated.step(forcing)
        assert b.u == pytest.approx(a.u, rel=1e-13, abs=1e-13)
        assert np.asarray(b.chi) == pytest.approx(np.asarray(a.chi), rel=1e-13, abs=1e-13)
        assert b.t == pytest.approx(a.t, abs=1e-15)
    assert accelerated.phase_counts == reference.phase_counts
    assert accelerated.stability_violations == reference.stability_violations == 0
