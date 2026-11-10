"""G3: G2 E-F0 vectorized runtime versus full scalar-ABI hierarchy.

The flat arm directly reuses the established G2 E-F0 DigitalRuntime runner.
The hierarchy arm uses the C++ ABI scalar-chi substrate with bottom-up and
top-down coupling. Both arms share stories, readout initialization, seed and
training-order identities. The F0 spine is active in the flat arm and is also
qualified with the existing per-seed companion audit. No static
weight-strengthening arm is included in G3.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..digital import DigitalMemoryConfig, DigitalMemoryState, HierarchicalDigitalDirector
from ..digital.babi import (
    BabiDataset, babi_story_to_episode, build_babi_answer_set,
    build_babi_bridge, build_babi_vocabulary,
)
from ..digital.readout import LinearReadout, query_conditioned_features
from ..ffi import load_library
from .babi_f0 import BabiF0Config, run_babi_f0_experiment
from .confidence_metrics import summarize_confidence
from .g0_diagnostics import build_prediction_record, summarize_training_losses
from .symbolic_induction import memory_config_for_variant
from .babi_g2 import run_fixture_suite

try:
    from gfe_ctypes import (
        GFE_C_COUPLING_FORM_B,
        GFE_C_HIERARCHICAL_RELATION_BOTTOM_UP,
        GFE_C_HIERARCHICAL_RELATION_TOP_DOWN,
    )
except ImportError:  # pragma: no cover - import path is set by the project runner
    from gfe_ctypes import (
        GFE_C_COUPLING_FORM_B,
        GFE_C_HIERARCHICAL_RELATION_BOTTOM_UP,
        GFE_C_HIERARCHICAL_RELATION_TOP_DOWN,
    )

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DATA_DIR = _REPO_ROOT / "code" / "python" / "d2c" / "progress" / "babi_data"
_FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "g2_relational_cases.json"
_T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
    7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
    13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
    19: 2.093, 20: 2.086,
}
ARM_FLAT = "ef0_vectorized"
ARM_HIER = "hier_bu_td"
ARMS = (ARM_FLAT, ARM_HIER)


@dataclass(frozen=True, slots=True)
class G3Config:
    tasks: tuple[int, ...] = (3, 5)
    dim: int = 64
    seeds: int = 3
    epochs: int = 3
    learning_rate: float = 0.05
    feature_scale: float = 10.0
    readout: str = "linear"
    bootstrap_samples: int = 400
    bottom_up_gain: float = 0.4
    top_down_gain: float = 0.5
    context_gamma: tuple[float, ...] = (0.35, 0.08)
    context_w: tuple[float, ...] = (0.05, 0.02)
    context_leak: float = 0.5
    normalize_top_down_weights: bool = True
    f0_spine_audit: bool = True

    def __post_init__(self) -> None:
        if self.tasks != (3, 5):
            raise ValueError("G3 requires complete QA3 and QA5 coverage in that order")
        if not 8 <= self.dim <= 512:
            raise ValueError("G3 dimension must be between 8 and 512")
        if not 3 <= self.seeds <= 20:
            raise ValueError("G3 requires 3–20 matched seeds")
        if self.epochs != 3:
            raise ValueError("G3 uses the fixed three-epoch checkpoint")
        if self.learning_rate != 0.05:
            raise ValueError("G3 fixes the learning rate at 0.05")
        if self.feature_scale != 10.0:
            raise ValueError("G3 fixes feature scale at 10.0")
        if self.readout != "linear":
            raise ValueError("G3 is registered with the linear readout only")
        if not 200 <= self.bootstrap_samples <= 5000:
            raise ValueError("bootstrap samples must be between 200 and 5000")
        if len(self.context_gamma) != len(self.context_w) or not self.context_gamma:
            raise ValueError("context gamma and weights must have matching non-empty lengths")
        if any(x <= 0 or not math.isfinite(x) for x in (*self.context_gamma, *self.context_w)):
            raise ValueError("context kernel values must be finite and positive")
        if any(
            not math.isfinite(value) or value <= 0
            for value in (self.bottom_up_gain, self.top_down_gain, self.context_leak)
        ):
            raise ValueError("hierarchy gains and context leak must be finite and positive")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _revision() -> str:
    if os.environ.get("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT,
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build_g3_chain(
    memory: DigitalMemoryConfig,
    config: G3Config,
    arm: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Build the common ABI flat control or the full two-way hierarchy."""
    if arm not in ARMS:
        raise ValueError(f"unknown G3 arm: {arm}")
    fast = {
        "name": "fast_event_memory",
        "gamma": list(memory.gamma),
        "w": list(memory.w),
        "u": [0.0] * config.dim,
        "chi": [0.0] * memory.channel_count,
        "dt": memory.dt,
        "linear_decay": [memory.leak_rate] * config.dim,
        "forcing_bias": [0.0] * config.dim,
        "form": GFE_C_COUPLING_FORM_B,
        "coupling_index": 0,
    }
    if arm != ARM_HIER:
        raise ValueError(
            "the flat G3 arm is the existing G2 E-F0 vectorized runner, not an ABI chain"
        )

    context = {
        "name": "slow_relational_context",
        "gamma": list(config.context_gamma),
        "w": list(config.context_w),
        "u": [0.0] * config.dim,
        "chi": [0.0] * len(config.context_gamma),
        "dt": memory.dt,
        "linear_decay": [config.context_leak] * config.dim,
        "forcing_bias": [0.0] * config.dim,
        "form": GFE_C_COUPLING_FORM_B,
        "coupling_index": 0,
    }
    edges = [
        {
            "source_level": 0,
            "target_level": 1,
            "relation": GFE_C_HIERARCHICAL_RELATION_BOTTOM_UP,
            "gain": config.bottom_up_gain,
            "normalize_weights": True,
        },
        {
            "source_level": 1,
            "target_level": 0,
            "relation": GFE_C_HIERARCHICAL_RELATION_TOP_DOWN,
            "gain": config.top_down_gain,
            # G3-N explicitly enables the top-down positive-kernel
            # normalizer; active-kernel sums and stability diagnostics record
            # its effect rather than silently folding it into the gain.
            "normalize_weights": config.normalize_top_down_weights,
        },
    ]
    return [fast, context], edges


def _readout_state(state: DigitalMemoryState, dim: int) -> DigitalMemoryState:
    """Broadcast ABI scalar-per-channel chi to the readout's vector contract."""
    return DigitalMemoryState(
        u=list(state.u),
        chi=[[float(channel[0])] * dim for channel in state.chi],
        t=float(state.t),
    )


def _new_readout(feature_dim: int, answer_count: int, seed: int):
    return LinearReadout(input_dim=feature_dim, output_dim=answer_count, seed=seed)


def _kernel_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"samples": 0}
    gamma_rows = [row.get("gamma", []) for row in rows]
    weight_rows = [row.get("w", []) for row in rows]
    width = min((len(row) for row in gamma_rows), default=0)
    gamma_means = [statistics.mean(float(row[i]) for row in gamma_rows) for i in range(width)]
    weight_means = [statistics.mean(float(row[i]) for row in weight_rows) for i in range(width)]
    sums = [sum(float(value) for value in row) for row in weight_rows]
    return {
        "samples": len(rows),
        "mean_gamma": gamma_means,
        "mean_w": weight_means,
        "mean_weight_sum": statistics.mean(sums),
        "min_weight_sum": min(sums),
        "max_weight_sum": max(sums),
        "all_positive": all(
            math.isfinite(float(value)) and float(value) > 0
            for row in (*gamma_rows, *weight_rows) for value in row
        ),
        "gamma_range": [
            [min(float(row[i]) for row in gamma_rows), max(float(row[i]) for row in gamma_rows)]
            for i in range(width)
        ],
        "weight_range": [
            [min(float(row[i]) for row in weight_rows), max(float(row[i]) for row in weight_rows)]
            for i in range(width)
        ],
    }


def _episode_hierarchy_diagnostics(
    result, arm: str, base_weight_sum: float, level_leak_rates: list[float],
) -> dict[str, Any]:
    levels = []
    for index, trace in enumerate(result.level_traces):
        states = [step.u for step in trace.steps]
        norms = [math.sqrt(sum(float(v) ** 2 for v in row)) for row in states]
        spectral_rows = [
            step.metadata.get("spectral", {})
            for step in trace.steps
            if isinstance(step.metadata.get("spectral", {}), dict)
        ]
        spectral = {}
        for key in ("deff", "mcap", "mscale", "hmem"):
            values = [float(row[key]) for row in spectral_rows if key in row and math.isfinite(float(row[key]))]
            if values:
                spectral[key] = {"mean": statistics.mean(values), "max": max(values)}
        levels.append({
            "level": index,
            "steps": len(trace.steps),
            "mean_state_norm": statistics.mean(norms) if norms else 0.0,
            "max_state_norm": max(norms, default=0.0),
            "spectral": spectral,
            "finite": all(math.isfinite(float(v)) for row in states for v in row),
        })

    per_level_kernels = [
        [step[level_index] for step in result.active_kernels if len(step) > level_index]
        for level_index in range(len(result.level_traces))
    ]
    flat_kernels = per_level_kernels[0] if per_level_kernels else []
    kernel = _kernel_summary(flat_kernels)
    kernel_by_level = [_kernel_summary(rows) for rows in per_level_kernels]
    context_norm = levels[1]["mean_state_norm"] if len(levels) > 1 else 0.0
    first_active_w_sum = sum(float(v) for v in flat_kernels[0].get("w", [])) if flat_kernels else 0.0
    dynamic_gamma = any(
        len(row.get("gamma", [])) == len(flat_kernels[0].get("gamma", []))
        and any(abs(float(a) - float(b)) > 1e-12
                for a, b in zip(row["gamma"], flat_kernels[0]["gamma"]))
        for row in flat_kernels[1:]
    ) if flat_kernels else False
    stability_by_level = []
    for level_index, kernels in enumerate(per_level_kernels):
        leak_rate = level_leak_rates[level_index]
        ratios = [
            sum(float(weight) / (float(rate) * leak_rate)
                for weight, rate in zip(row.get("w", []), row.get("gamma", [])))
            if len(row.get("w", [])) == len(row.get("gamma", []))
            and row.get("gamma")
            and all(math.isfinite(float(rate)) and float(rate) > 0 for rate in row["gamma"])
            and all(math.isfinite(float(weight)) and float(weight) > 0 for weight in row["w"])
            else math.inf
            for row in kernels
        ]
        finite_ratios = [value for value in ratios if math.isfinite(value)]
        stability_by_level.append({
            "level": level_index,
            "leak_rate": leak_rate,
            "max_ratio": max(ratios, default=math.inf),
            "mean_ratio": statistics.mean(finite_ratios) if finite_ratios else None,
            "passed": bool(ratios) and all(math.isfinite(value) and value <= 0.9 for value in ratios),
        })
    fast_stability = stability_by_level[0] if stability_by_level else {
        "max_ratio": math.inf, "mean_ratio": None, "passed": False
    }
    max_stability_ratio = max(
        (row["max_ratio"] for row in stability_by_level), default=math.inf
    )
    all_active_stability_pass = (
        bool(stability_by_level) and all(row["passed"] for row in stability_by_level)
    )
    positive_all_active_kernels = (
        bool(kernel_by_level) and all(row.get("all_positive", False) for row in kernel_by_level)
    )
    return {
        "arm": arm,
        "level_count": len(levels),
        "levels": levels,
        "fast_level_active_kernel": kernel,
        "active_kernels_by_level": kernel_by_level,
        "base_fast_weight_sum": float(base_weight_sum),
        "first_active_fast_weight_sum": first_active_w_sum,
        "mean_active_fast_weight_sum": kernel.get("mean_weight_sum"),
        "fast_active_stability_ratio_max": fast_stability["max_ratio"],
        "fast_active_stability_ratio_mean": fast_stability["mean_ratio"],
        "fast_active_stability_margin": 0.9,
        "fast_active_stability_pass": fast_stability["passed"],
        "stability_by_level": stability_by_level,
        "max_hierarchy_active_stability_ratio": max_stability_ratio,
        "all_active_stability_pass": all_active_stability_pass,
        "positive_all_active_kernels": positive_all_active_kernels,
        "top_down_gamma_changed_over_episode": dynamic_gamma if arm == ARM_HIER else False,
        "context_level_activity": context_norm,
        "bottom_up_context_observed": len(levels) > 1 and context_norm > 0.0,
        "top_down_coupling_configured": arm == ARM_HIER,
        "finite_state_and_spectral_evidence": all(row["finite"] for row in levels)
            and all(
                math.isfinite(float(metric_value))
                for row in levels
                for metric_group in row["spectral"].values()
                for metric_value in metric_group.values()
            ),
    }


def _orders(story_count: int, seeds: int, epochs: int) -> dict[int, list[list[int]]]:
    result: dict[int, list[list[int]]] = {}
    for seed in range(seeds):
        rng = random.Random(seed)
        epoch_orders = []
        for _ in range(epochs):
            indices = list(range(story_count))
            rng.shuffle(indices)
            epoch_orders.append(indices)
        result[seed] = epoch_orders
    return result


def _run_arm(
    *,
    arm: str,
    task: int,
    seed: int,
    train_stories,
    eval_stories,
    answer_set: list[str],
    bridge,
    memory: DigitalMemoryConfig,
    config: G3Config,
    orders: list[list[int]],
    lib,
) -> dict[str, Any]:
    if arm != ARM_HIER:
        raise ValueError("_run_arm is reserved for the scalar-ABI hierarchy")
    levels, edges = build_g3_chain(memory, config, arm)
    director = HierarchicalDigitalDirector(
        lib=lib, levels=levels, edges=edges, state_dim=config.dim,
        memory_config=memory,
    )
    feature_dim = config.dim * 2 + config.dim * config.dim
    readout = _new_readout(feature_dim, len(answer_set), seed)
    initial_readout_sha256 = hashlib.sha256(
        readout.W.tobytes() + readout.b.tobytes()
    ).hexdigest()
    epoch_losses: list[float] = []

    def features_for(story):
        episode = babi_story_to_episode(story, bridge, answer_set)
        result = director.run_episode(
            episode, experiment_name=f"g3-{arm}-qa{task}-seed{seed}",
        )
        features = query_conditioned_features(
            _readout_state(result.final_state, config.dim),
            memory, result.query, feature_scale=config.feature_scale,
        )
        return features, result

    for order in orders:
        losses = []
        for story_index in order:
            features, _ = features_for(train_stories[story_index])
            target_index = answer_set.index(train_stories[story_index].answer)
            losses.append(readout.train_step(
                features, target_index, learning_rate=config.learning_rate,
            ))
        epoch_losses.append(statistics.mean(losses) if losses else 0.0)

    predictions = []
    hierarchy_rows = []
    correct = 0
    for story_index, story in enumerate(eval_stories):
        features, episode_result = features_for(story)
        probabilities = readout.probabilities(features)
        prediction = build_prediction_record(story, story_index, answer_set, probabilities)
        predictions.append(prediction)
        correct += int(prediction["correct"])
        # Capture the flat ABI baseline as well as the hierarchy arm. The
        # aggregator reports one level for flat_abi and two for hier_bu_td.
        hierarchy_rows.append(
            _episode_hierarchy_diagnostics(
                episode_result,
                arm,
                base_weight_sum=sum(memory.w),
                level_leak_rates=(
                    [memory.leak_rate]
                    if arm == ARM_FLAT
                    else [memory.leak_rate, config.context_leak]
                ),
            )
        )

    accuracy = correct / len(eval_stories) if eval_stories else 0.0
    confidence = summarize_confidence(
        predictions, answer_set, calibration_bins=10,
        bootstrap_samples=config.bootstrap_samples,
        bootstrap_seed=task * 10_000 + seed * 101 + (0 if arm == ARM_FLAT else 1),
    )
    return {
        "seed": seed,
        "arm": arm,
        "correct": correct,
        "total": len(eval_stories),
        "accuracy": accuracy,
        "training": summarize_training_losses(epoch_losses),
        "initial_readout_sha256": initial_readout_sha256,
        "predictions": predictions,
        "confidence": confidence,
        "hierarchy_diagnostics": {
            "evaluation_examples": len(hierarchy_rows),
            "level_count": len(levels),
            "levels": [
                {
                    "level": level,
                    "mean_state_norm": statistics.mean(row["levels"][level]["mean_state_norm"] for row in hierarchy_rows),
                    "max_state_norm": max(row["levels"][level]["max_state_norm"] for row in hierarchy_rows),
                    "spectral_means": {
                        key: statistics.mean(row["levels"][level]["spectral"][key]["mean"]
                                             for row in hierarchy_rows
                                             if key in row["levels"][level]["spectral"])
                        for key in ("deff", "mcap", "mscale", "hmem")
                        if all(key in row["levels"][level]["spectral"] for row in hierarchy_rows)
                    },
                }
                for level in range(2 if arm == ARM_HIER else 1)
            ],
            "fast_active_kernel": _kernel_summary([
                {
                    "gamma": item["fast_level_active_kernel"]["mean_gamma"],
                    "w": item["fast_level_active_kernel"]["mean_w"],
                }
                for item in hierarchy_rows
            ]) if hierarchy_rows else None,
            "bottom_up_context_activity": statistics.mean(
                row["context_level_activity"] for row in hierarchy_rows
            ) if hierarchy_rows else 0.0,
            "top_down_coupling_configured": (
                bool(hierarchy_rows) and all(
                    row["top_down_coupling_configured"] for row in hierarchy_rows
                )
            ),
            "top_down_dynamic_gamma_fraction": (
                sum(bool(row["top_down_gamma_changed_over_episode"]) for row in hierarchy_rows) / len(hierarchy_rows)
                if hierarchy_rows else 0.0
            ),
            "finite_state_and_spectral_pass": all(
                row["finite_state_and_spectral_evidence"] for row in hierarchy_rows
            ) if hierarchy_rows else True,
            "positive_active_kernel_pass": all(
                row["positive_all_active_kernels"] for row in hierarchy_rows
            ) if hierarchy_rows else True,
            "base_fast_weight_sum": (
                statistics.mean(row["base_fast_weight_sum"] for row in hierarchy_rows)
                if hierarchy_rows else sum(memory.w)
            ),
            "first_active_fast_weight_sum": (
                statistics.mean(row["first_active_fast_weight_sum"] for row in hierarchy_rows)
                if hierarchy_rows else None
            ),
            "mean_active_fast_weight_sum": (
                statistics.mean(row["mean_active_fast_weight_sum"] for row in hierarchy_rows)
                if hierarchy_rows else None
            ),
            "weight_normalization_observed": (
                statistics.mean(row["fast_level_active_kernel"]["mean_weight_sum"] for row in hierarchy_rows)
                if hierarchy_rows else None
            ),
            "max_fast_active_stability_ratio": max(
                row["fast_active_stability_ratio_max"] for row in hierarchy_rows
            ) if hierarchy_rows else None,
            # Preserve the frozen stability bound in the seed-level summary;
            # the run-level acceptance reporter consumes this aggregated field.
            "fast_active_stability_margin": min(
                row["fast_active_stability_margin"] for row in hierarchy_rows
            ) if hierarchy_rows else None,
            "fast_active_stability_pass": all(
                row["fast_active_stability_pass"] for row in hierarchy_rows
            ) if hierarchy_rows else True,
            "max_hierarchy_active_stability_ratio": max(
                row["max_hierarchy_active_stability_ratio"] for row in hierarchy_rows
            ) if hierarchy_rows else None,
            "all_active_stability_pass": all(
                row["all_active_stability_pass"] for row in hierarchy_rows
            ) if hierarchy_rows else True,
        },
    }


def _paired_effect(differences: list[float], metric: str) -> dict[str, Any]:
    n = len(differences)
    if not n:
        return {"n": 0, "mean": None, "ci95_two_sided_t": None, "exact_two_sided_sign_test_p": None}
    mean = statistics.mean(differences)
    sd = statistics.stdev(differences) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else None
    ci = None if n < 2 else [
        mean - _T95.get(n - 1, 1.96) * se,
        mean + _T95.get(n - 1, 1.96) * se,
    ]
    pos, neg = sum(x > 0 for x in differences), sum(x < 0 for x in differences)
    nonzero = pos + neg
    minority = min(pos, neg)
    sign_p = (
        min(1.0, 2.0 * sum(math.comb(nonzero, k) for k in range(minority + 1)) / (2 ** nonzero))
        if nonzero else 1.0
    )
    return {
        "metric": metric, "n": n, "mean": mean,
        "standard_deviation": sd, "standard_error": se,
        "ci95_two_sided_t": ci,
        "exact_two_sided_sign_test_p": sign_p,
        "sign_test_nonzero_pairs": nonzero,
        "positive_seed_differences": pos,
        "negative_seed_differences": neg,
        "zero_seed_differences": n - nonzero,
        "differences_hierarchy_minus_flat": differences,
        "interpretation": "Paired seed-level effect. Story-bootstrap uncertainty is reported separately in each seed's confidence record.",
    }


def _arm_summary(seed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = {
        "accuracy": [row["accuracy"] for row in seed_rows],
        "nll": [row["confidence"]["negative_log_likelihood"] for row in seed_rows],
        "brier": [row["confidence"]["multiclass_brier_score"] for row in seed_rows],
        "fixed_ece": [row["confidence"]["calibration"]["fixed_bin_top_label_ece"] for row in seed_rows],
        "adaptive_ece": [row["confidence"]["calibration"]["adaptive_equal_mass_top_label_ece"] for row in seed_rows],
        "aurc": [row["confidence"]["selective_prediction"]["area_under_risk_coverage_curve"] for row in seed_rows],
        "overconfident_error_fraction_0_9": [
            row["confidence"]["overconfident_error_fraction"]["0.9"] or 0.0 for row in seed_rows
        ],
    }
    return {
        "mean_accuracy_over_seeds": statistics.mean(metrics["accuracy"]),
        "standard_error_over_seeds": statistics.stdev(metrics["accuracy"]) / math.sqrt(len(seed_rows))
            if len(seed_rows) > 1 else 0.0,
        "mean_confidence_metrics_over_seeds": {
            key: statistics.mean(values) for key, values in metrics.items() if key != "accuracy"
        },
        "seed_results": seed_rows,
    }


def _f0_vectorized_seed_rows(
    *,
    task: int,
    train_orders: dict[int, list[list[int]]],
    answer_set: list[str],
    config: G3Config,
    data_dir: Path,
    eval_split: str,
) -> list[dict[str, Any]]:
    """Adapt the existing G2 E-F0 runner without rebuilding its stepping path."""
    f0_config = BabiF0Config(
        task=task, variant="full", dim=config.dim, seeds=config.seeds,
        epochs=config.epochs, hidden_dim=32,
        learning_rate=config.learning_rate, readout_type=config.readout,
        feature_scale=config.feature_scale, mode="isolated",
        exercise_spine=config.f0_spine_audit,
    )
    result = run_babi_f0_experiment(
        f0_config, data_dir=data_dir, eval_split=eval_split
    )
    if len(result["seed_results"]) != config.seeds:
        raise AssertionError("G2 E-F0 runner returned an unexpected number of seeds")
    if result["train_stories"] <= 0 or result["evaluation_stories"] <= 0:
        raise AssertionError("G2 E-F0 runner returned an empty task split")

    rows: list[dict[str, Any]] = []
    for seed_row in result["seed_results"]:
        seed = int(seed_row["seed"])
        expected_order_hash = hashlib.sha256(
            json.dumps(train_orders[seed], separators=(",", ":")).encode()
        ).hexdigest()
        if seed_row.get("training_order_sha256") != expected_order_hash:
            raise AssertionError(f"G2 E-F0 training order mismatch for QA{task}, seed {seed}")
        if not seed_row.get("initial_readout_sha256"):
            raise AssertionError(f"G2 E-F0 readout fingerprint missing for QA{task}, seed {seed}")
        predictions = seed_row["predictions"]
        confidence = summarize_confidence(
            predictions, answer_set, calibration_bins=10,
            bootstrap_samples=config.bootstrap_samples,
            bootstrap_seed=task * 10_000 + seed * 101,
        )
        if confidence.get("correct") != int(seed_row["correct"]):
            raise AssertionError("G2 E-F0 confidence diagnostics disagree with runner correctness")
        rows.append({
            "seed": seed, "arm": ARM_FLAT,
            "correct": int(seed_row["correct"]), "total": int(seed_row["total"]),
            "accuracy": float(seed_row["accuracy"]),
            "training": seed_row["training"],
            "initial_readout_sha256": seed_row["initial_readout_sha256"],
            "training_order_sha256": seed_row["training_order_sha256"],
            "predictions": predictions, "confidence": confidence,
            "f0_spine_audit": seed_row.get("spine_exercise"),
            "runtime_diagnostics": {
                "backend": "G2 E-F0 DigitalRuntime -> Python vectorized step_memory",
                "train": seed_row.get("train_runtime", {}),
                "evaluation": seed_row.get("test_runtime", {}),
            },
            "hierarchy_diagnostics": {
                "backend": "python_vectorized_f0_flat",
                "evaluation_examples": len(predictions), "level_count": 0,
                "levels": [], "bottom_up_context_activity": 0.0,
                "top_down_coupling_configured": False,
                "top_down_dynamic_gamma_fraction": 0.0,
                "base_fast_weight_sum": None, "mean_active_fast_weight_sum": None,
                "max_fast_active_stability_ratio": None,
                "max_hierarchy_active_stability_ratio": None,
                "fast_active_stability_pass": True, "all_active_stability_pass": True,
            },
        })
    if [row["seed"] for row in rows] != list(range(config.seeds)):
        raise AssertionError("G2 E-F0 seed identities are not the registered sequence")
    return rows


def _verify_pair(flat: dict[str, Any], hierarchy: dict[str, Any]) -> bool:
    if flat["seed"] != hierarchy["seed"] or flat["total"] != hierarchy["total"]:
        return False
    if flat.get("initial_readout_sha256") != hierarchy.get("initial_readout_sha256"):
        return False
    if flat.get("training_order_sha256") != hierarchy.get("training_order_sha256"):
        return False
    fields = ("story_index", "task", "raw_passage", "question", "target")
    flat_predictions = flat.get("predictions", [])
    hierarchy_predictions = hierarchy.get("predictions", [])
    return (
        len(flat_predictions) == flat["total"]
        and len(hierarchy_predictions) == hierarchy["total"]
        and len(flat_predictions) == len(hierarchy_predictions)
        and all(
            all(a.get(field) == b.get(field) for field in fields)
            for a, b in zip(flat_predictions, hierarchy_predictions)
        )
    )


def run_g3_experiment(
    *,
    config: G3Config | None = None,
    data_dir: Path = _DATA_DIR,
    eval_split: str = "test",
    lib=None,
) -> dict[str, Any]:
    cfg = config or G3Config()
    if eval_split not in {"valid", "test"}:
        raise ValueError("G3 evaluation split must be valid or test")
    data_dir = Path(data_dir)
    for split in ("train", eval_split):
        path = data_dir / f"babi_{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"G3 requires {path.name}")

    dataset = {
        split: {"sha256": _sha256(data_dir / f"babi_{split}.jsonl"),
                "bytes": (data_dir / f"babi_{split}.jsonl").stat().st_size}
        for split in ("train", eval_split)
        if (data_dir / f"babi_{split}.jsonl").is_file()
    }
    training_dataset = BabiDataset.from_jsonl(data_dir / "babi_train.jsonl")
    evaluation_dataset = BabiDataset.from_jsonl(data_dir / f"babi_{eval_split}.jsonl")
    fixture_control = run_fixture_suite(_FIXTURE_PATH)
    runtime_library = lib or load_library(None)
    memory = memory_config_for_variant("full")
    orders_by_seed: dict[str, Any] = {}
    all_task_results: dict[str, Any] = {}
    started_at = datetime.now(timezone.utc).isoformat()

    for task in cfg.tasks:
        train_stories = training_dataset.filter_task(task).stories
        eval_stories = evaluation_dataset.filter_task(task).stories
        if not train_stories or not eval_stories:
            raise ValueError(f"G3 task QA{task} needs non-empty train and {eval_split} splits")
        vocabulary = build_babi_vocabulary(train_stories + eval_stories)
        answer_set = build_babi_answer_set(train_stories)
        if len(answer_set) < 2:
            raise ValueError(f"G3 task QA{task} needs at least two training answer classes")
        bridge = build_babi_bridge(vocabulary, dim=cfg.dim)
        training_orders = _orders(len(train_stories), cfg.seeds, cfg.epochs)
        arm_rows: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
        training_order_hashes = {
            seed: hashlib.sha256(
                json.dumps(order, separators=(",", ":")).encode()
            ).hexdigest()
            for seed, order in training_orders.items()
        }
        task_seed_orders = {
            str(seed): digest for seed, digest in training_order_hashes.items()
        }

        # The flat arm is the existing G2 E-F0 vectorized runtime runner.
        arm_rows[ARM_FLAT] = _f0_vectorized_seed_rows(
            task=task, train_orders=training_orders, answer_set=answer_set,
            config=cfg, data_dir=data_dir, eval_split=eval_split,
        )
        flat_by_seed = {row["seed"]: row for row in arm_rows[ARM_FLAT]}
        paired_ok = True
        for seed in range(cfg.seeds):
            hierarchy_row = _run_arm(
                arm=ARM_HIER, task=task, seed=seed,
                train_stories=train_stories, eval_stories=eval_stories,
                answer_set=answer_set, bridge=bridge, memory=memory,
                config=cfg, orders=training_orders[seed], lib=runtime_library,
            )
            hierarchy_row["training_order_sha256"] = training_order_hashes[seed]
            arm_rows[ARM_HIER].append(hierarchy_row)
            paired_ok = paired_ok and _verify_pair(flat_by_seed[seed], hierarchy_row)

        flat = _arm_summary(arm_rows[ARM_FLAT])
        hierarchy = _arm_summary(arm_rows[ARM_HIER])
        flat_by_seed = {row["seed"]: row for row in flat["seed_results"]}
        hier_by_seed = {row["seed"]: row for row in hierarchy["seed_results"]}
        metrics = ("accuracy", "nll", "brier", "fixed_ece", "adaptive_ece", "aurc")
        paired = {}
        for metric in metrics:
            if metric == "accuracy":
                left = [flat_by_seed[s]["accuracy"] for s in sorted(flat_by_seed)]
                right = [hier_by_seed[s]["accuracy"] for s in sorted(hier_by_seed)]
            else:
                key = {
                    "nll": "negative_log_likelihood",
                    "brier": "multiclass_brier_score",
                    "fixed_ece": "fixed_bin_top_label_ece",
                    "adaptive_ece": "adaptive_equal_mass_top_label_ece",
                    "aurc": "area_under_risk_coverage_curve",
                }[metric]
                left = [flat_by_seed[s]["confidence"][key] if metric in ("nll", "brier")
                        else flat_by_seed[s]["confidence"]["calibration"][key]
                        if metric in ("fixed_ece", "adaptive_ece")
                        else flat_by_seed[s]["confidence"]["selective_prediction"][key]
                        for s in sorted(flat_by_seed)]
                right = [hier_by_seed[s]["confidence"][key] if metric in ("nll", "brier")
                         else hier_by_seed[s]["confidence"]["calibration"][key]
                         if metric in ("fixed_ece", "adaptive_ece")
                         else hier_by_seed[s]["confidence"]["selective_prediction"][key]
                         for s in sorted(hier_by_seed)]
            paired[metric] = _paired_effect(
                [float(b) - float(a) for a, b in zip(left, right)],
                f"{metric}: hierarchy minus flat (negative favours hierarchy for loss/calibration/risk)",
            )
        f0_spines = [row.get("f0_spine_audit") or {} for row in flat["seed_results"]]
        spine_passed = all(
            row.get("status") == "passed"
            and all(row.get("phase_coverage", {}).values())
            and row.get("maintenance", {}).get("kernel_unchanged") is True
            and row.get("trace_store", {}).get("sample_is_contiguous") is True
            and row.get("checkpoint_restore", {}).get("snapshot_round_trip_equal") is True
            and row.get("checkpoint_restore", {}).get("deterministic_continuation_equal") is True
            and row.get("diagnostics", {}).get("stability_violations") == 0
            for row in f0_spines
        )
        # Mechanism execution and stability qualification are separate.
        # G3-N intentionally probes unit-sum top-down normalization, which
        # may violate the unchanged runtime stability margin. That is a
        # reportable experimental outcome, not a failed execution.
        hierarchy_audit_passed = all(
            row["hierarchy_diagnostics"]["finite_state_and_spectral_pass"]
            and row["hierarchy_diagnostics"]["positive_active_kernel_pass"]
            and row["hierarchy_diagnostics"]["bottom_up_context_activity"] > 0
            and row["hierarchy_diagnostics"]["top_down_coupling_configured"] is True
            and row["hierarchy_diagnostics"]["top_down_dynamic_gamma_fraction"] > 0
            for row in hierarchy["seed_results"]
        )
        hierarchy_stability_passed = all(
            row["hierarchy_diagnostics"]["fast_active_stability_pass"]
            and row["hierarchy_diagnostics"]["all_active_stability_pass"]
            for row in hierarchy["seed_results"]
        )
        hierarchy_stability_evidence_recorded = all(
            row["hierarchy_diagnostics"]["max_hierarchy_active_stability_ratio"] is not None
            and row["hierarchy_diagnostics"]["fast_active_stability_margin"] is not None
            for row in hierarchy["seed_results"]
        )
        max_hierarchy_stability_ratio = max(
            row["hierarchy_diagnostics"]["max_hierarchy_active_stability_ratio"]
            for row in hierarchy["seed_results"]
        )
        stability_margin = max(
            row["hierarchy_diagnostics"]["fast_active_stability_margin"]
            for row in hierarchy["seed_results"]
        )
        all_confidence_complete = all(
            row["confidence"].get("negative_log_likelihood") is not None
            and row["confidence"].get("multiclass_brier_score") is not None
            and row["confidence"].get("calibration", {}).get("fixed_bin_top_label_ece") is not None
            and row["confidence"].get("calibration", {}).get("adaptive_equal_mass_top_label_ece") is not None
            and row["confidence"].get("selective_prediction", {}).get("area_under_risk_coverage_curve") is not None
            and row["confidence"].get("accuracy_wilson_95_ci") is not None
            for arm in (flat, hierarchy) for row in arm["seed_results"]
        )
        all_task_results[str(task)] = {
            "task": task,
            "task_label": f"QA{task}",
            "evaluation_stories": len(eval_stories),
            "answer_classes": answer_set,
            "training_story_count": len(train_stories),
            "training_order_hashes_by_seed": task_seed_orders,
            "arms": {ARM_FLAT: flat, ARM_HIER: hierarchy},
            "paired_effects": paired,
            "paired_seed_and_example_identity_verified": paired_ok,
            "f0_spine_audit": {
                "passed": spine_passed,
                "audits_required": cfg.seeds,
                "audits_passed": sum(row.get("status") == "passed" for row in f0_spines),
                "interpretation": "The existing G2 E-F0 six-phase runtime is the vectorized flat benchmark arm; its per-task/per-seed spine audit is retained.",
            },
            "hierarchy_mechanism_audit": {
                "passed": hierarchy_audit_passed,
                "per_seed": [
                    {"seed": row["seed"], **row["hierarchy_diagnostics"]}
                    for row in hierarchy["seed_results"]
                ],
                "weight_normalization_control": {
                    "top_down_normalize_weights": cfg.normalize_top_down_weights,
                    "interpretation": "Top-down positive-kernel weights are normalized to unit sum when enabled; active-kernel sums and stability are recorded per seed.",
                },
            },
            "confidence_evidence": {
                "passed": all_confidence_complete,
                "bootstrap_samples_per_seed": cfg.bootstrap_samples,
                "flat": flat["mean_confidence_metrics_over_seeds"],
                "hierarchy": hierarchy["mean_confidence_metrics_over_seeds"],
            },
        }

    accuracy_ps = [
        task["paired_effects"]["accuracy"]["exact_two_sided_sign_test_p"]
        for task in all_task_results.values()
    ]
    ordered_ps = sorted(enumerate(accuracy_ps), key=lambda item: item[1] if item[1] is not None else 1.0)
    holm_by_index = {}
    running = 0.0
    for rank, (index, p_value) in enumerate(ordered_ps):
        adjusted = min(1.0, max(running, (len(ordered_ps) - rank) * float(p_value)))
        holm_by_index[index] = adjusted
        running = adjusted
    for index, task in enumerate(all_task_results.values()):
        task["paired_effects"]["accuracy"]["holm_adjusted_exact_sign_test_p_across_tasks"] = holm_by_index[index]

    full_coverage = set(all_task_results) == {"3", "5"} and all(
        task["evaluation_stories"] > 0 and task["training_story_count"] > 0
        for task in all_task_results.values()
    )
    pairing = all(task["paired_seed_and_example_identity_verified"] for task in all_task_results.values())
    hierarchy_audit = all(task["hierarchy_mechanism_audit"]["passed"] for task in all_task_results.values())
    spine_audit = all(task["f0_spine_audit"]["passed"] for task in all_task_results.values())
    confidence_complete = all(task["confidence_evidence"]["passed"] for task in all_task_results.values())
    fixture_passed = fixture_control["failed"] == 0 and fixture_control["total"] == 12
    acceptance = {
        "qa3_qa5_coverage": {"passed": full_coverage, "tasks": sorted(all_task_results)},
        "matched_seed_example_and_training_order": {"passed": pairing, "arms": list(ARMS)},
        "full_bottom_up_top_down_mechanism": {"passed": hierarchy_audit},
        "hierarchy_stability_qualification": {
            "passed": hierarchy_stability_passed,
            "evidence_recorded": hierarchy_stability_evidence_recorded,
            "max_observed_stability_ratio": max_hierarchy_stability_ratio,
            "stability_margin": stability_margin,
            "interpretation": (
                "Stable within the frozen margin."
                if hierarchy_stability_passed
                else "Executed and measured, but not stability-qualified; the frozen runtime margin is intentionally unchanged."
            ),
        },
        "f0_six_phase_spine_audit": {"passed": spine_audit, "audits_per_task": cfg.seeds},
        "confidence_evidence": {"passed": confidence_complete},
        "structured_relational_fixtures": {"passed": fixture_passed, "passed_count": fixture_control["passed"], "total": fixture_control["total"]},
    }
    return {
        "schema_version": 1,
        "experiment": "g3_hierarchical_relational",
        "experiment_id": "G3-FULL-HIERARCHY",
        "status": "completed",
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_revision": _revision(),
        "protocol": {
            **asdict(cfg),
            "arms": list(ARMS),
            "arm_descriptions": {
                ARM_FLAT: "Existing G2 E-F0 Python-vectorized DigitalRuntime/step_memory path, isolated mode, six-phase runtime spine active.",
                ARM_HIER: "Two-level C++ ABI scalar-chi runtime with full bottom-up and top-down coupling.",
            },
            "substrate": "Two packages: G2 E-F0 Python-vectorized step_memory flat arm; C++ hierarchy chain-spec ABI with scalar chi and pre-step Form-B feedback at coupling_index 0 for hierarchy.",
            "flat_baseline_source": "d2c.experiments.babi_f0.run_babi_f0_experiment; build_babi_memory_config('full'); isolated episodes; default G2 DigitalRuntime configuration.",
            "hierarchy_kernel_source": "d2c.experiments.symbolic_induction.memory_config_for_variant('full'); C++ ABI Form-B hierarchy with frozen two-level gains and context kernel.",
            "f0_spine": "Existing six-phase E-F0 runtime is used directly as the vectorized flat benchmark arm; the existing per-seed F0 spine exercise is also retained as a companion qualification check.",
            "hierarchy_control_boundary": "No static weight-strengthening arm in G3. This compares complete G2 E-F0 vectorized and scalar-ABI bidirectional hierarchy packages; it does not isolate hierarchy alone because backend and memory feedback formulation differ. Active hierarchy kernels are fully reported.",
            "evaluation_split": eval_split,
            "test_split_opened": eval_split == "test",
            "locked_test_policy": "The test split is evaluated once after implementation and validation checks; it is not used to tune the hierarchy.",
        },
        "dataset": {
            "profile_id": "babi_canonical",
            "directory": "code/python/d2c/progress/babi_data",
            "split_fingerprints": dataset,
        },
        "fixture_control": fixture_control,
        "tasks": all_task_results,
        "acceptance": acceptance,
        "claim_boundary": (
            "G3 compares the established G2 E-F0 vectorized Python runtime, including its six-phase F0 spine, "
            "against a full bidirectional two-level scalar-ABI hierarchy on bAbI QA3/QA5. This is a package-level "
            "comparison, not a pure hierarchy-only causal ablation: the backend and memory feedback formulation "
            "differ. Per-task/per-seed F0 spine audits qualify the existing flat runtime. Static weight strengthening "
            "and runtime ablation remain deferred. Passing G3 does not establish general language reasoning or "
            "close the later F-series learning mechanisms."
        ),
    }


def save_g3_bundle(result: dict[str, Any], output_dir: Path) -> Path:
    bundle = Path(output_dir) / f"g3_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    bundle.mkdir(parents=True, exist_ok=False)
    (bundle / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (bundle / "config.json").write_text(json.dumps(result["protocol"], indent=2), encoding="utf-8")
    rows = []
    examples = []
    for task_key, task in result["tasks"].items():
        for arm_name, arm in task["arms"].items():
            for seed in arm["seed_results"]:
                rows.append({
                    "task": int(task_key), "arm": arm_name, "seed": seed["seed"],
                    "correct": seed["correct"], "total": seed["total"],
                    "accuracy": seed["accuracy"],
                    "nll": seed["confidence"]["negative_log_likelihood"],
                    "brier": seed["confidence"]["multiclass_brier_score"],
                    "fixed_ece": seed["confidence"]["calibration"]["fixed_bin_top_label_ece"],
                    "adaptive_ece": seed["confidence"]["calibration"]["adaptive_equal_mass_top_label_ece"],
                    "aurc": seed["confidence"]["selective_prediction"]["area_under_risk_coverage_curve"],
                    "f0_spine": seed.get("f0_spine_audit", {}).get("status", "not_run"),
                    "runtime_backend": seed.get("runtime_diagnostics", {}).get(
                        "backend", seed.get("hierarchy_diagnostics", {}).get("backend", "scalar_abi_hierarchy")
                    ),
                    "hierarchy_levels": seed["hierarchy_diagnostics"]["level_count"],
                    "context_activity": seed["hierarchy_diagnostics"]["bottom_up_context_activity"],
                    "top_down_dynamic_gamma_fraction": seed["hierarchy_diagnostics"]["top_down_dynamic_gamma_fraction"],
                    "base_fast_weight_sum": seed["hierarchy_diagnostics"]["base_fast_weight_sum"],
                    "mean_active_fast_weight_sum": seed["hierarchy_diagnostics"]["mean_active_fast_weight_sum"],
                    "max_fast_active_stability_ratio": seed["hierarchy_diagnostics"]["max_fast_active_stability_ratio"],
                    "max_hierarchy_active_stability_ratio": seed["hierarchy_diagnostics"]["max_hierarchy_active_stability_ratio"],
                    "fast_active_stability_pass": seed["hierarchy_diagnostics"]["fast_active_stability_pass"],
                    "all_active_stability_pass": seed["hierarchy_diagnostics"]["all_active_stability_pass"],
                })
                for prediction in seed["predictions"]:
                    examples.append({
                        "task": int(task_key), "arm": arm_name, "seed": seed["seed"],
                        **prediction,
                    })
    with (bundle / "per_seed_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    with (bundle / "per_example.jsonl").open("w", encoding="utf-8") as handle:
        for row in examples:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (bundle / "per_example.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["task", "arm", "seed", "story_index", "question", "target", "prediction", "correct", "confidence", "top_two_margin"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in examples:
            writer.writerow({key: row.get(key) for key in fields})
    register = {
        "schema_version": 1,
        "experiment_id": result["experiment_id"],
        "status": result["status"],
        "acceptance": result["acceptance"],
        "task_results": {
            key: {
                "paired_effects": task["paired_effects"],
                "f0_spine_audit": task["f0_spine_audit"],
                "hierarchy_mechanism_audit": task["hierarchy_mechanism_audit"],
                "confidence_evidence": task["confidence_evidence"],
            }
            for key, task in result["tasks"].items()
        },
        "claim_boundary": result["claim_boundary"],
    }
    (bundle / "experiment_lab_register.json").write_text(json.dumps(register, indent=2), encoding="utf-8")
    report_lines = [
        "# G3 Full Hierarchy Relational Report",
        "",
        f"- Evaluation split: {result['protocol']['evaluation_split']}",
        f"- Matched seeds: {result['protocol']['seeds']}",
        f"- Epochs: {result['protocol']['epochs']}",
        f"- Dimension / feature scale: {result['protocol']['dim']} / {result['protocol']['feature_scale']}",
        f"- Fixture control: {result['fixture_control']['passed']}/{result['fixture_control']['total']}",
        "",
        "## Acceptance",
    ]
    for key, value in result["acceptance"].items():
        report_lines.append(f"- {key}: {'PASS' if value['passed'] else 'FAIL'}")
    for task_key, task in result["tasks"].items():
        report_lines.extend(["", f"## QA{task_key}", "", "| Arm | Mean accuracy | NLL | Brier | Fixed ECE | Adaptive ECE | AURC |",
                             "|---|---:|---:|---:|---:|---:|---:|"])
        for arm_name, arm in task["arms"].items():
            metrics = arm["mean_confidence_metrics_over_seeds"]
            report_lines.append(
                f"| {arm_name} | {arm['mean_accuracy_over_seeds']:.4f} | "
                f"{metrics['nll']:.4f} | {metrics['brier']:.4f} | "
                f"{metrics['fixed_ece']:.4f} | {metrics['adaptive_ece']:.4f} | "
                f"{metrics['aurc']:.4f} |"
            )
        effect = task["paired_effects"]["accuracy"]
        hierarchy_seeds = task["arms"][ARM_HIER]["seed_results"]
        hierarchy_diags = [row["hierarchy_diagnostics"] for row in hierarchy_seeds]
        mean_context = statistics.mean(row["bottom_up_context_activity"] for row in hierarchy_diags)
        mean_gamma_fraction = statistics.mean(row["top_down_dynamic_gamma_fraction"] for row in hierarchy_diags)
        mean_active_weight = statistics.mean(row["weight_normalization_observed"] for row in hierarchy_diags)
        base_weight = statistics.mean(row["base_fast_weight_sum"] for row in hierarchy_diags)
        max_stability = max(row["max_hierarchy_active_stability_ratio"] for row in hierarchy_diags)
        stability_pass = all(row["all_active_stability_pass"] for row in hierarchy_diags)
        report_lines.extend([
            "",
            "### Hierarchy mechanism audit",
            f"- Mean bottom-up context activity: {mean_context:.6g}",
            f"- Mean fraction of evaluation episodes with top-down gamma variation: {mean_gamma_fraction:.3f}",
            f"- Base / mean active fast-level weight sum: {base_weight:.6g} / {mean_active_weight:.6g}",
            f"- Maximum active fast-level stability ratio: {max_stability:.4f} (margin 0.9; {'PASS' if stability_pass else 'FAIL'})",
            "",
            f"- Paired accuracy delta (hierarchy − flat): {effect['mean']:.4f}",
            f"- 95% paired-seed t interval: {effect['ci95_two_sided_t']}",
            f"- Exact sign-test p: {effect['exact_two_sided_sign_test_p']}",
            f"- Positive / negative / zero seeds: {effect['positive_seed_differences']} / {effect['negative_seed_differences']} / {effect['zero_seed_differences']}",
        ])
    report_lines.extend(["", "## Claim boundary", "", result["claim_boundary"], ""])
    (bundle / "summary.md").write_text("\n".join(report_lines), encoding="utf-8")
    manifest = {
        "manifest_version": 1,
        "experiment_id": result["experiment_id"],
        "run_id": bundle.name,
        "code_revision": result["code_revision"],
        "python_version": __import__("platform").python_version(),
        "created_at_utc": result["started_at_utc"],
        "completed_at_utc": result["completed_at_utc"],
        "config": result["protocol"],
        "dataset_split_fingerprints": result["dataset"]["split_fingerprints"],
        "result_files": sorted(path.name for path in bundle.iterdir()),
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (bundle / "metadata.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return bundle


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run G3 G2 E-F0 vectorized flat versus full scalar-ABI hierarchy.")
    parser.add_argument("--tasks", default="3,5")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--feature-scale", type=float, default=10.0)
    parser.add_argument("--bootstrap-samples", type=int, default=400)
    parser.add_argument("--eval-split", choices=["valid", "test"], default="test")
    parser.add_argument("--data-dir", type=Path, default=_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/experiment-lab/g3"))
    parser.add_argument("--skip-f0-spine-audit", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    tasks = tuple(int(value.strip()) for value in args.tasks.split(",") if value.strip())
    config = G3Config(
        tasks=tasks, dim=args.dim, seeds=args.seeds, epochs=args.epochs,
        learning_rate=args.lr, feature_scale=args.feature_scale,
        bootstrap_samples=args.bootstrap_samples,
        f0_spine_audit=not args.skip_f0_spine_audit,
    )
    result = run_g3_experiment(config=config, data_dir=args.data_dir, eval_split=args.eval_split)
    bundle = save_g3_bundle(result, args.output_dir)
    print(f"G3 {args.eval_split}: tasks={','.join(map(str, config.tasks))} seeds={config.seeds}")
    for task_key, task in result["tasks"].items():
        effect = task["paired_effects"]["accuracy"]
        print(
            f"QA{task_key}: flat={task['arms'][ARM_FLAT]['mean_accuracy_over_seeds']:.4f} "
            f"hier={task['arms'][ARM_HIER]['mean_accuracy_over_seeds']:.4f} "
            f"paired_delta={effect['mean']:.4f} sign_p={effect['exact_two_sided_sign_test_p']:.4f}"
        )
    print(f"acceptance={json.dumps(result['acceptance'], sort_keys=True)}")
    print(f"bundle={bundle}")
    return result


if __name__ == "__main__":
    main()
