from __future__ import annotations

import math
import os

import pytest

from d2c.digital.bridge import DigitalMemoryConfig, seed_memory_state
from d2c.digital.director import DigitalEpisode
from d2c.digital.hierarchy_g42 import (
    NLevelVectorizedHierarchyDirector, adaptive_capacity_update, prune_grow_kernel,
)
from d2c.experiments.babi_g42 import _channel_counts, _level_configs, _multi_level_features, _orders, G42Config


@pytest.mark.parametrize("config,activity", [
    (DigitalMemoryConfig(gamma=[2.0, 0.45, 0.035], w=[0.16, 0.108, 0.0448],
        dt=0.05, leak_rate=2.0, form="negative_feedback"), [0.3, 0.2, 0.01]),
    (DigitalMemoryConfig(gamma=[0.35, 0.08], w=[0.05, 0.02],
        dt=0.05, leak_rate=0.5, form="negative_feedback"), [0.2, 0.01]),
])
def test_legacy_replacement_primitive_preserves_mass_and_stability(config, activity):
    updated, event = prune_grow_kernel(config, activity)
    assert updated.channel_count == config.channel_count
    assert min(updated.gamma) < min(config.gamma)
    assert math.isclose(sum(updated.w), sum(config.w), abs_tol=1e-10)
    assert math.isclose(updated.stability_ratio(), config.stability_ratio(), abs_tol=1e-10)
    assert all(value > 0.0 and math.isfinite(value) for value in updated.w)
    assert event["operation"] == "prune_then_grow"


def test_duration_checkpoints_accept_3_5_8_12_epochs():
    config = G42Config(
        depth=4, capacity=6, dim=64, epochs=12,
        checkpoint_epochs=(3, 5, 8, 12),
    )
    assert config.epochs == 12
    assert config.checkpoint_epochs == (3, 5, 8, 12)


def test_duration_orders_preserve_the_original_three_epoch_prefix():
    baseline = _orders(count=31, seeds=3, epochs=3)
    extended = _orders(count=31, seeds=3, epochs=12)
    assert [orders[:3] for orders in extended] == baseline


def test_duration_checkpoints_must_include_final_epoch():
    with pytest.raises(ValueError, match="include the final epoch"):
        G42Config(
            depth=2, capacity=3, dim=64, epochs=12,
            checkpoint_epochs=(3, 5, 8),
        )


def test_capacity_adaptation_can_grow_a_real_channel_and_preserve_invariants():
    config = DigitalMemoryConfig(
        gamma=[2.0, 0.45, 0.035], w=[0.16, 0.108, 0.0448],
        dt=0.05, leak_rate=2.0, form="negative_feedback",
    )
    updated, event = adaptive_capacity_update(config, [0.4, 0.3, 0.3])
    assert event["operation"] == "grow"
    assert updated.channel_count == config.channel_count + 1
    assert event["channel_births"] == 1 and event["channel_deaths"] == 0
    assert math.isclose(sum(updated.w), sum(config.w), abs_tol=1e-10)
    assert math.isclose(updated.stability_ratio(), config.stability_ratio(), abs_tol=1e-9)
    assert min(updated.gamma) < min(config.gamma)


def test_capacity_adaptation_can_prune_a_real_channel_and_preserve_invariants():
    config = DigitalMemoryConfig(
        gamma=[2.0, 0.45, 0.035], w=[0.16, 0.108, 0.0448],
        dt=0.05, leak_rate=2.0, form="negative_feedback",
    )
    updated, event = adaptive_capacity_update(config, [0.5, 0.001, 0.4])
    assert event["operation"] == "prune"
    assert updated.channel_count == config.channel_count - 1
    assert event["channel_deaths"] == 1 and event["channel_births"] == 0
    assert math.isclose(sum(updated.w), sum(config.w), abs_tol=1e-10)
    assert math.isclose(updated.stability_ratio(), config.stability_ratio(), abs_tol=1e-9)


@pytest.mark.parametrize("depth", [2, 3, 4])
@pytest.mark.parametrize("capacity", [3, 4, 6])
def test_channel_budget_is_matched_across_depth(depth, capacity):
    counts = _channel_counts(depth, capacity)
    assert sum(counts) == capacity * 5
    assert len(counts) == depth
    assert min(counts) >= 2


def _episode(dim=8):
    forcings = []
    for index in range(8):
        row = [0.0] * dim
        row[index % dim] = 0.5 + index * 0.01
        row[(index + 1) % dim] = 0.15
        forcings.append(row)
    return DigitalEpisode(
        forcings=forcings, tokens=[f"t{index}" for index in range(len(forcings))],
        target="x", query=[0.1] * dim,
    )


def _small_director():
    fast = DigitalMemoryConfig(
        gamma=[2.0, 0.45, 0.035], w=[0.16, 0.108, 0.0448],
        dt=0.05, leak_rate=2.0, form="negative_feedback",
    )
    context = DigitalMemoryConfig(
        gamma=[0.35, 0.08], w=[0.05, 0.02],
        dt=0.05, leak_rate=0.5, form="negative_feedback",
    )
    return NLevelVectorizedHierarchyDirector(
        level_configs=[fast, context, context], state_dim=8,
    )


def _require_native():
    if not os.environ.get("GFE_CORE_LIB"):
        pytest.skip("GFE_CORE_LIB is required for native hierarchy integration")


def test_perturbing_upper_level_vector_changes_lower_level_computation():
    _require_native()
    director = _small_director()
    initial = [
        seed_memory_state(state_dim=8, channel_count=config.channel_count)
        for config in director.level_configs
    ]
    initial[1].u[0] = 0.8
    initial[1].chi[0][1] = 0.35
    with_context = director.run_episode(
        _episode(), initial_states=initial, enable_bottom_up=False, enable_top_down=True,
    )
    without_context = director.run_episode(
        _episode(), initial_states=initial, enable_bottom_up=False, enable_top_down=False,
    )
    delta = sum(
        abs(a - b) for a, b in zip(
            with_context.final_states[0].u, without_context.final_states[0].u,
        )
    )
    assert delta > 1e-8
    assert with_context.top_down_input_norms[0] > 0.0


def test_removing_top_down_coupling_changes_lower_level_output():
    _require_native()
    with_top = _small_director().run_episode(
        _episode(), enable_bottom_up=True, enable_top_down=True,
    )
    without_top = _small_director().run_episode(
        _episode(), enable_bottom_up=True, enable_top_down=False,
    )
    delta = sum(abs(a - b) for a, b in zip(
        with_top.final_states[0].u, without_top.final_states[0].u,
    ))
    assert delta > 1e-8


def test_removing_bottom_up_propagation_changes_upper_representation():
    _require_native()
    with_bottom = _small_director().run_episode(
        _episode(), enable_bottom_up=True, enable_top_down=False,
    )
    without_bottom = _small_director().run_episode(
        _episode(), enable_bottom_up=False, enable_top_down=False,
    )
    delta = sum(abs(a - b) for a, b in zip(
        with_bottom.final_states[1].u, without_bottom.final_states[1].u,
    ))
    assert delta > 1e-8
    assert with_bottom.bottom_up_input_norms[1] > 0.0


def test_all_levels_have_distinct_readout_blocks():
    _require_native()
    cfg = G42Config(depth=3, capacity=3, dim=64)
    configs = _level_configs(cfg)
    director = NLevelVectorizedHierarchyDirector(level_configs=configs, state_dim=64)
    from d2c.digital.director import DigitalEpisode
    forcings = []
    for index in range(8):
        row = [0.0] * 64
        row[index] = 0.25
        forcings.append(row)
    episode = DigitalEpisode(
        forcings=forcings, tokens=[f"r{index}" for index in range(8)],
        target="x", query=[0.1] * 64,
    )
    result = director.run_episode(episode)
    features = _multi_level_features(
        result, director.level_configs, episode.query, feature_scale=10.0,
    )
    block_dim = 64 * 64 + 2 * 64
    assert len(features) == 3 * block_dim
    assert all(any(abs(value) > 1e-12 for value in features[level * block_dim:(level + 1) * block_dim])
               for level in range(3))
    # Bare query is not repeated into the upper-level blocks.
    for level in (1, 2):
        start = level * block_dim + 64
        assert all(value == 0.0 for value in features[start:start + 64])
