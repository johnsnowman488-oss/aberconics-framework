from __future__ import annotations

import math
import os

import pytest

from d2c.digital import DigitalEpisode, DigitalMemoryConfig
from d2c.digital.vectorized_hierarchy_director import (
    VectorizedHierarchicalDigitalDirector,
    vectorized_top_down_kernel,
)


def test_top_down_normalizer_preserves_positive_unit_sum_weights():
    base = DigitalMemoryConfig(
        gamma=[2.0, 0.45, 0.035],
        w=[0.16, 0.108, 0.0448],
        dt=0.05,
        leak_rate=1.0,
        form="negative_feedback",
    )
    active = vectorized_top_down_kernel(
        base, [0.1, -0.2, 0.3, -0.4], gain=0.5,
        normalize_weights=True,
    )
    assert math.isclose(sum(active.w), 1.0, rel_tol=0.0, abs_tol=1e-12)
    assert all(value > 0.0 and math.isfinite(value) for value in active.w)
    assert all(value > 0.0 and math.isfinite(value) for value in active.gamma)
    assert active.gamma != base.gamma
    assert active.form == "negative_feedback"


def test_vectorized_hierarchy_uses_vector_memory_at_both_levels():
    library = os.environ.get("GFE_CORE_LIB")
    if not library:
        pytest.skip("GFE_CORE_LIB is required for native hierarchy integration test")

    memory = DigitalMemoryConfig(
        gamma=[2.0, 0.45, 0.035],
        w=[0.16, 0.108, 0.0448],
        dt=0.05,
        leak_rate=1.0,
        form="negative_feedback",
    )
    director = VectorizedHierarchicalDigitalDirector(
        memory_config=memory,
        state_dim=8,
        bottom_up_gain=0.4,
        top_down_gain=0.5,
        context_gamma=[0.35, 0.08],
        context_w=[0.05, 0.02],
        context_leak=0.5,
        normalize_top_down_weights=True,
        library_path=library,
    )
    forcings = []
    for index in range(6):
        row = [0.0] * 8
        row[index % 8] = 0.5
        forcings.append(row)
    episode = DigitalEpisode(
        forcings=forcings,
        tokens=[f"t{index}" for index in range(len(forcings))],
        target="x",
        query=[0.0] * 8,
    )
    result = director.run_episode(episode)

    assert len(result.level_traces) == 2
    assert len(result.active_kernels) == len(forcings)
    assert len(result.final_state.u) == 8
    assert len(result.final_state.chi) == 3
    assert all(len(channel) == 8 for channel in result.final_state.chi)
    assert all(len(step.u) == 8 for trace in result.level_traces for step in trace.steps)
    assert all(len(channel) == 8 for trace in result.level_traces
               for step in trace.steps for channel in step.chi)
    assert result.mean_state_norms[1] > 0.0
    assert result.top_down_gamma_changed
    assert math.isclose(result.top_down_weight_sum_min, 1.0, abs_tol=1e-9)
    assert math.isclose(result.top_down_weight_sum_max, 1.0, abs_tol=1e-9)
    assert all(math.isfinite(value) for row in result.stability_ratios for value in row)
    assert director.step_count == len(forcings)
