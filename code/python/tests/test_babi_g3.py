"""Focused G3 configuration, controls and ABI hierarchy diagnostics tests."""
from __future__ import annotations

import math

import pytest

from d2c.digital import DigitalEpisode, DigitalMemoryState, HierarchicalDigitalDirector
from d2c.experiments.babi_g3 import (
    ARM_FLAT,
    ARM_HIER,
    G3Config,
    _episode_hierarchy_diagnostics,
    _paired_effect,
    _readout_state,
    build_g3_chain,
)
from d2c.experiments.symbolic_induction import memory_config_for_variant
from d2c.ffi import load_library
from gfe_ctypes import GFE_C_HIERARCHICAL_RELATION_BOTTOM_UP, GFE_C_HIERARCHICAL_RELATION_TOP_DOWN


def test_g3_config_requires_full_qa3_qa5_and_matched_seed_count():
    with pytest.raises(ValueError, match="complete QA3 and QA5"):
        G3Config(tasks=(3,))
    with pytest.raises(ValueError, match="3–20 matched seeds"):
        G3Config(seeds=2)
    with pytest.raises(ValueError, match="linear readout"):
        G3Config(readout="mlp")
    with pytest.raises(ValueError, match="three-epoch"):
        G3Config(epochs=4)
    with pytest.raises(ValueError, match="learning rate"):
        G3Config(learning_rate=0.1)
    with pytest.raises(ValueError, match="feature scale"):
        G3Config(feature_scale=100.0)


def test_g3_flat_arm_is_existing_ef0_path_and_abi_builder_is_hierarchy_only():
    config = G3Config(dim=8, seeds=3)
    memory = memory_config_for_variant("full")
    with pytest.raises(ValueError, match="existing G2 E-F0 vectorized runner"):
        build_g3_chain(memory, config, ARM_FLAT)
    hier_levels, hier_edges = build_g3_chain(memory, config, ARM_HIER)

    assert ARM_FLAT == "ef0_vectorized"
    assert len(hier_levels) == 2
    assert len(hier_edges) == 2
    assert {edge["relation"] for edge in hier_edges} == {
        GFE_C_HIERARCHICAL_RELATION_BOTTOM_UP,
        GFE_C_HIERARCHICAL_RELATION_TOP_DOWN,
    }
    bottom_up = next(edge for edge in hier_edges if edge["relation"] == GFE_C_HIERARCHICAL_RELATION_BOTTOM_UP)
    top_down = next(edge for edge in hier_edges if edge["relation"] == GFE_C_HIERARCHICAL_RELATION_TOP_DOWN)
    assert bottom_up["normalize_weights"] is True
    assert top_down["normalize_weights"] is False


def test_g3_readout_broadcasts_scalar_abi_chi_to_feature_dimension():
    state = DigitalMemoryState(
        u=[1.0, 2.0, 3.0],
        chi=[[0.25], [0.5]],
        t=0.1,
    )
    compatible = _readout_state(state, 3)
    assert compatible.u == [1.0, 2.0, 3.0]
    assert compatible.chi == [[0.25, 0.25, 0.25], [0.5, 0.5, 0.5]]


def test_g3_paired_effect_reports_seed_level_interval_and_sign_test():
    effect = _paired_effect([0.02, 0.01, 0.00, -0.01, 0.03], "accuracy")
    assert effect["n"] == 5
    assert effect["mean"] == pytest.approx(0.01)
    assert effect["ci95_two_sided_t"][0] < effect["mean"] < effect["ci95_two_sided_t"][1]
    assert effect["positive_seed_differences"] == 3
    assert effect["negative_seed_differences"] == 1
    assert effect["zero_seed_differences"] == 1


def test_full_hierarchy_executes_both_couplings_and_reports_finite_state():
    try:
        lib = load_library(None)
    except OSError as exc:
        pytest.skip(f"shared C++ core unavailable: {exc}")
    config = G3Config(dim=8, seeds=3)
    memory = memory_config_for_variant("full")
    levels, edges = build_g3_chain(memory, config, ARM_HIER)
    director = HierarchicalDigitalDirector(
        lib=lib, levels=levels, edges=edges, state_dim=config.dim,
        memory_config=memory,
    )
    episode = DigitalEpisode(
        forcings=[
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0] * 8,
        ],
        tokens=["Mary", "moved", None],
        target="hallway",
        query=[0.0] * 8,
    )
    result = director.run_episode(episode, experiment_name="g3_test")
    report = _episode_hierarchy_diagnostics(
        result, ARM_HIER, sum(memory.w), [memory.leak_rate, config.context_leak]
    )

    assert len(result.level_traces) == 2
    assert report["level_count"] == 2
    assert report["bottom_up_context_observed"]
    assert report["top_down_coupling_configured"]
    assert report["finite_state_and_spectral_evidence"]
    assert report["fast_level_active_kernel"]["all_positive"]
    assert report["positive_all_active_kernels"]
    assert report["fast_active_stability_pass"]
    assert report["all_active_stability_pass"]
    assert report["max_hierarchy_active_stability_ratio"] <= report["fast_active_stability_margin"]
    assert len(report["stability_by_level"]) == 2
    assert all(row["passed"] for row in report["stability_by_level"])
    assert report["base_fast_weight_sum"] == pytest.approx(sum(memory.w))
    assert report["first_active_fast_weight_sum"] == pytest.approx(sum(memory.w), rel=1e-9, abs=1e-9)
    assert all(math.isfinite(row["mean_state_norm"]) for row in report["levels"])


def test_flat_arm_calls_existing_g2_ef0_runner(monkeypatch, tmp_path):
    import hashlib
    import json
    import d2c.experiments.babi_g3 as g3

    cfg = G3Config(dim=8, seeds=3)
    orders = {seed: [[0, 1], [1, 0]] for seed in range(3)}
    order_hash = lambda rows: hashlib.sha256(
        json.dumps(rows, separators=(",", ":")).encode()
    ).hexdigest()
    fingerprints = {}

    def fake_runner(config, *, data_dir, eval_split):
        fingerprints["config"] = config
        seeds = []
        for seed in range(3):
            seeds.append({
                "seed": seed,
                "correct": 1,
                "total": 1,
                "accuracy": 1.0,
                "training": {"epochs": 3},
                "initial_readout_sha256": "a" * 64,
                "training_order_sha256": order_hash(orders[seed]),
                "predictions": [{
                    "story_index": 0, "task": 3, "raw_passage": "Mary moved.",
                    "question": "Where?", "target": "hallway",
                    "prediction": "hallway", "correct": True,
                    "probability_vector": [0.9, 0.1],
                }],
                "spine_exercise": {"status": "passed"},
                "train_runtime": {"step_count": 12},
                "test_runtime": {"step_count": 4},
            })
        return {"seed_results": seeds, "train_stories": 2, "evaluation_stories": 1}

    monkeypatch.setattr(g3, "run_babi_f0_experiment", fake_runner)
    rows = g3._f0_vectorized_seed_rows(
        task=3, train_orders=orders, answer_set=["hallway", "kitchen"],
        config=cfg, data_dir=tmp_path, eval_split="valid",
    )

    assert fingerprints["config"].mode == "isolated"
    assert fingerprints["config"].exercise_spine is True
    assert fingerprints["config"].epochs == 3
    assert fingerprints["config"].learning_rate == 0.05
    assert fingerprints["config"].feature_scale == 10.0
    assert [row["seed"] for row in rows] == [0, 1, 2]
    assert rows[0]["runtime_diagnostics"]["backend"].startswith("G2 E-F0")
    assert rows[0]["f0_spine_audit"]["status"] == "passed"
    assert rows[0]["hierarchy_diagnostics"]["level_count"] == 0


def test_g3_pair_rejects_readout_or_training_order_mismatch():
    from d2c.experiments.babi_g3 import _verify_pair

    row = {
        "seed": 2, "total": 1,
        "initial_readout_sha256": "same-readout",
        "training_order_sha256": "same-order",
        "predictions": [{
            "story_index": 0, "task": 3, "raw_passage": "Mary moved.",
            "question": "Where?", "target": "hallway",
        }],
    }
    assert _verify_pair(row, dict(row))
    assert not _verify_pair(row, dict(row, training_order_sha256="different-order"))
    assert not _verify_pair(row, dict(row, initial_readout_sha256="different-readout"))
