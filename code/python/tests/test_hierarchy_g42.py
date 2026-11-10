from __future__ import annotations
import math
import pytest
from d2c.digital.bridge import DigitalMemoryConfig
from d2c.digital.hierarchy_g42 import prune_grow_kernel

@pytest.mark.parametrize("config,activity", [
    (DigitalMemoryConfig(gamma=[2.0, 0.45, 0.035], w=[0.16, 0.108, 0.0448],
        dt=0.05, leak_rate=2.0, form="negative_feedback"), [0.3, 0.2, 0.01]),
    (DigitalMemoryConfig(gamma=[0.35, 0.08], w=[0.05, 0.02],
        dt=0.05, leak_rate=0.5, form="negative_feedback"), [0.2, 0.01]),
])
def test_prune_grow_replaces_slowest_channel_and_preserves_mass_stability(config, activity):
    updated, event = prune_grow_kernel(config, activity)
    assert updated.channel_count == config.channel_count
    assert min(updated.gamma) < min(config.gamma)
    assert math.isclose(sum(updated.w), sum(config.w), abs_tol=1e-10)
    assert math.isclose(updated.stability_ratio(), config.stability_ratio(), abs_tol=1e-10)
    assert all(value > 0.0 and math.isfinite(value) for value in updated.w)
    assert event["operation"] == "prune_then_grow"
    assert event["channel_count_before"] == event["channel_count_after"]

def test_prune_grow_rejects_invalid_activity_shape():
    config = DigitalMemoryConfig(gamma=[1.0, 0.1], w=[0.1, 0.02],
        dt=0.05, leak_rate=2.0, form="negative_feedback")
    with pytest.raises(ValueError):
        prune_grow_kernel(config, [0.1])

def test_h3_director_runs_three_simultaneous_levels_when_native_library_is_available():
    import os
    from d2c.digital.director import DigitalEpisode
    from d2c.digital.hierarchy_g42 import NLevelVectorizedHierarchyDirector
    if not os.environ.get("GFE_CORE_LIB"):
        pytest.skip("GFE_CORE_LIB is required for native hierarchy integration")
    fast = DigitalMemoryConfig(
        gamma=[2.0, 0.45, 0.035], w=[0.16, 0.108, 0.0448],
        dt=0.05, leak_rate=2.0, form="negative_feedback")
    context = DigitalMemoryConfig(
        gamma=[0.35, 0.08], w=[0.05, 0.02],
        dt=0.05, leak_rate=0.5, form="negative_feedback")
    director = NLevelVectorizedHierarchyDirector(
        level_configs=[fast, context, context], state_dim=8)
    forcings = []
    for i in range(5):
        row = [0.0] * 8
        row[i % 8] = 0.25
        forcings.append(row)
    episode = DigitalEpisode(
        forcings=forcings, tokens=[f"t{i}" for i in range(5)],
        target="x", query=[0.0] * 8)
    result = director.run_episode(episode)
    assert len(result.level_traces) == 3
    assert all(len(trace.steps) == 5 for trace in result.level_traces)
    assert len(result.final_state.u) == 8
    assert all(math.isfinite(v) for row in result.stability_ratios for v in row)
