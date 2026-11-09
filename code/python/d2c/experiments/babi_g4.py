"""G4: G3 hierarchy with vector-valued C++ memory at both levels.

This is validation-only. It re-runs the scalar G3 hierarchy on the same
QA3/QA5 validation split and five seed identities, then compares the new
vectorized hierarchy to that matched reference. The locked test split is not
accepted by this module or its workflow.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..digital import (
    BabiDataset, DigitalMemoryConfig, DigitalMemoryState,
    babi_story_to_episode, build_babi_answer_set, build_babi_bridge,
    build_babi_vocabulary,
)
from ..digital.readout import LinearReadout, query_conditioned_features
from ..digital.vectorized_hierarchy_director import VectorizedHierarchicalDigitalDirector
from .babi_g3 import (
    G3Config, _orders, _paired_effect, _sha256, run_g3_experiment,
)
from .confidence_metrics import summarize_confidence
from .g0_diagnostics import build_prediction_record, summarize_training_losses
from .symbolic_induction import memory_config_for_variant


_REPO_ROOT = Path(__file__).resolve().parents[4]
_DATA_DIR = _REPO_ROOT / "code" / "python" / "d2c" / "progress" / "babi_data"
_OUTPUT_DIR = _REPO_ROOT / "runs" / "experiment-lab" / "g4"
_BASELINE_ARM = "hier_bu_td"
_EXPECTED_SEEDS = 5
_EXPECTED_EPOCHS = 3
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}


@dataclass(frozen=True, slots=True)
class G4Config:
    dim: int = 64
    seeds: int = _EXPECTED_SEEDS
    epochs: int = _EXPECTED_EPOCHS
    learning_rate: float = 0.05
    feature_scale: float = 10.0
    bootstrap_samples: int = 400
    bottom_up_gain: float = 0.4
    top_down_gain: float = 0.5
    context_gamma: tuple[float, ...] = (0.35, 0.08)
    context_w: tuple[float, ...] = (0.05, 0.02)
    context_leak: float = 0.5
    normalize_top_down_weights: bool = True

    def __post_init__(self) -> None:
        if self.dim != 64:
            raise ValueError("G4 holds dimension fixed at 64 for the G3 comparison")
        if self.seeds != _EXPECTED_SEEDS:
            raise ValueError("G4 validation uses exactly five matched seeds")
        if self.epochs != _EXPECTED_EPOCHS:
            raise ValueError("G4 uses the fixed three-epoch checkpoint")
        if self.learning_rate != 0.05 or self.feature_scale != 10.0:
            raise ValueError("G4 holds learning rate 0.05 and feature scale 10.0")
        if self.bootstrap_samples < 200:
            raise ValueError("G4 requires at least 200 bootstrap samples")


def _fingerprint(value: object) -> str:
    raw = json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _readout_hash(readout: LinearReadout) -> str:
    return hashlib.sha256(readout.W.tobytes() + readout.b.tobytes()).hexdigest()


def _confidence_value(row: dict[str, Any], metric: str) -> float:
    if metric in {"nll", "brier"}:
        key = "negative_log_likelihood" if metric == "nll" else "multiclass_brier_score"
        return float(row["confidence"][key])
    if metric in {"fixed_ece", "adaptive_ece"}:
        key = (
            "fixed_bin_top_label_ece" if metric == "fixed_ece"
            else "adaptive_equal_mass_top_label_ece"
        )
        return float(row["confidence"]["calibration"][key])
    key = "area_under_risk_coverage_curve"
    return float(row["confidence"]["selective_prediction"][key])


def _summary_for_arm(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [float(row["accuracy"]) for row in rows]
    mean = statistics.mean(values)
    se = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return {
        "mean_accuracy_over_seeds": mean,
        "standard_error_over_seeds": se,
        "seed_results": rows,
    }


def _run_vectorized_seed(
    *,
    task: int,
    seed: int,
    train_stories,
    eval_stories,
    answer_set: list[str],
    bridge,
    memory: DigitalMemoryConfig,
    cfg: G4Config,
    epoch_orders: list[list[int]],
    data_order_hash: str,
) -> dict[str, Any]:
    feature_dim = cfg.dim * 2 + cfg.dim * cfg.dim
    readout = LinearReadout(
        input_dim=feature_dim, output_dim=len(answer_set), seed=seed
    )
    initial_hash = _readout_hash(readout)
    director = VectorizedHierarchicalDigitalDirector(
        memory_config=memory,
        state_dim=cfg.dim,
        bottom_up_gain=cfg.bottom_up_gain,
        top_down_gain=cfg.top_down_gain,
        context_gamma=cfg.context_gamma,
        context_w=cfg.context_w,
        context_leak=cfg.context_leak,
        normalize_top_down_weights=cfg.normalize_top_down_weights,
    )
    epoch_losses: list[float] = []
    train_runtime_steps = 0

    def features_for(story, label: str):
        episode = babi_story_to_episode(story, bridge, answer_set)
        result = director.run_episode(
            episode, experiment_name=f"g4-vector-qa{task}-seed{seed}-{label}"
        )
        features = query_conditioned_features(
            result.final_state, memory, episode.query,
            feature_scale=cfg.feature_scale,
        )
        return features, result

    for epoch, order in enumerate(epoch_orders):
        losses: list[float] = []
        for story_index in order:
            features, result = features_for(
                train_stories[story_index], f"train-e{epoch}-s{story_index}"
            )
            train_runtime_steps += len(result.trace.steps)
            target = train_stories[story_index].answer
            losses.append(readout.train_step(
                features, answer_set.index(target), learning_rate=cfg.learning_rate
            ))
        epoch_losses.append(statistics.mean(losses) if losses else 0.0)

    predictions: list[dict[str, Any]] = []
    correct = 0
    eval_runtime_steps = 0
    eval_max_state_norms = [0.0, 0.0]
    eval_norm_sums = [0.0, 0.0]
    eval_episode_count = 0
    max_stability = [0.0, 0.0]
    all_finite = True
    top_down_dynamic_episodes = 0
    top_down_drive_sum = 0.0
    weight_sum_min = math.inf
    weight_sum_max = -math.inf
    bottom_up_context_activity = 0.0

    for story_index, story in enumerate(eval_stories):
        features, result = features_for(story, f"eval-s{story_index}")
        eval_runtime_steps += len(result.trace.steps)
        probabilities = readout.probabilities(features)
        prediction = build_prediction_record(
            story, story_index, answer_set, probabilities
        )
        predictions.append(prediction)
        correct += int(prediction["correct"])
        eval_episode_count += 1
        top_down_dynamic_episodes += int(result.top_down_gamma_changed)
        top_down_drive_sum += result.top_down_drive_mean
        bottom_up_context_activity += result.mean_state_norms[1]
        for level in range(2):
            eval_max_state_norms[level] = max(
                eval_max_state_norms[level], result.max_state_norms[level]
            )
            eval_norm_sums[level] += result.mean_state_norms[level]
            max_stability[level] = max(
                max_stability[level], max(result.stability_ratios[level], default=0.0)
            )
        weight_sum_min = min(weight_sum_min, result.top_down_weight_sum_min)
        weight_sum_max = max(weight_sum_max, result.top_down_weight_sum_max)
        all_finite = all_finite and all(
            math.isfinite(value)
            for trace in result.level_traces
            for step in trace.steps
            for value in (*step.u, *(v for channel in step.chi for v in channel))
        )

    total = len(eval_stories)
    accuracy = correct / total if total else 0.0
    confidence = summarize_confidence(
        predictions, answer_set, calibration_bins=10,
        bootstrap_samples=cfg.bootstrap_samples,
        bootstrap_seed=task * 10_000 + seed * 101 + 7,
    )
    episode_denominator = max(1, eval_episode_count)
    return {
        "seed": seed,
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "initial_readout_sha256": initial_hash,
        "training_order_sha256": data_order_hash,
        "training": summarize_training_losses(epoch_losses),
        "predictions": predictions,
        "confidence": confidence,
        "runtime": {
            "backend": "cpp_vectorized_at_both_hierarchy_levels",
            "train_steps": train_runtime_steps,
            "evaluation_steps": eval_runtime_steps,
            "native_step_calls": train_runtime_steps + eval_runtime_steps,
            "state_dimension_per_level": cfg.dim,
            "memory_channels_per_level": [memory.channel_count, len(cfg.context_gamma)],
        },
        "hierarchy_diagnostics": {
            "level_count": 2,
            "bottom_up_coupling_configured": True,
            "top_down_coupling_configured": True,
            "bottom_up_context_activity_mean_state_norm": (
                bottom_up_context_activity / episode_denominator
            ),
            "top_down_mean_context_drive": top_down_drive_sum / episode_denominator,
            "top_down_gamma_dynamic_episode_fraction": (
                top_down_dynamic_episodes / episode_denominator
            ),
            "top_down_weights_normalized": cfg.normalize_top_down_weights,
            "top_down_weight_sum_min": weight_sum_min,
            "top_down_weight_sum_max": weight_sum_max,
            "mean_state_norm_by_level": [
                value / episode_denominator for value in eval_norm_sums
            ],
            "max_state_norm_by_level": eval_max_state_norms,
            "max_stability_ratio_by_level": max_stability,
            "stability_margin": 0.9,
            "stability_pass_by_level": [value <= 0.9 for value in max_stability],
            "stability_evidence_recorded": all(
                math.isfinite(value) for value in max_stability
            ),
            "finite_state_pass": all_finite,
        },
    }


def _verify_pairing(
    baseline_rows: list[dict[str, Any]],
    g4_rows: list[dict[str, Any]],
    eval_stories,
) -> bool:
    base = {int(row["seed"]): row for row in baseline_rows}
    vector = {int(row["seed"]): row for row in g4_rows}
    if set(base) != set(range(_EXPECTED_SEEDS)) or set(vector) != set(base):
        return False
    identity_fields = ("story_index", "task", "raw_passage", "question", "target")
    for seed in range(_EXPECTED_SEEDS):
        left, right = base[seed], vector[seed]
        if left.get("initial_readout_sha256") != right.get("initial_readout_sha256"):
            return False
        if left.get("training_order_sha256") != right.get("training_order_sha256"):
            return False
        lp, rp = left.get("predictions", []), right.get("predictions", [])
        if len(lp) != len(eval_stories) or len(rp) != len(eval_stories):
            return False
        for a, b in zip(lp, rp):
            if not all(a.get(field) == b.get(field) for field in identity_fields):
                return False
    return True


def run_g4_validation(
    *,
    config: G4Config | None = None,
    data_dir: Path = _DATA_DIR,
) -> dict[str, Any]:
    cfg = config or G4Config()
    data_dir = Path(data_dir)
    train_path = data_dir / "babi_train.jsonl"
    valid_path = data_dir / "babi_valid.jsonl"
    if not train_path.is_file() or not valid_path.is_file():
        raise FileNotFoundError("G4 requires train and valid bAbI files")
    split_fingerprints = {
        "train": {"sha256": _sha256(train_path), "bytes": train_path.stat().st_size},
        "valid": {"sha256": _sha256(valid_path), "bytes": valid_path.stat().st_size},
    }

    training_dataset = BabiDataset.from_jsonl(train_path)
    validation_dataset = BabiDataset.from_jsonl(valid_path)
    memory = memory_config_for_variant("full")
    g3_cfg = G3Config(
        tasks=(3, 5), dim=cfg.dim, seeds=cfg.seeds, epochs=cfg.epochs,
        learning_rate=cfg.learning_rate, feature_scale=cfg.feature_scale,
        bootstrap_samples=cfg.bootstrap_samples, bottom_up_gain=cfg.bottom_up_gain,
        top_down_gain=cfg.top_down_gain, context_gamma=cfg.context_gamma,
        context_w=cfg.context_w, context_leak=cfg.context_leak,
        normalize_top_down_weights=cfg.normalize_top_down_weights,
        f0_spine_audit=False,
    )
    # Matched G3 reference is regenerated on validation only; neither this
    # module nor the workflow accepts the locked test split.
    g3_reference = run_g3_experiment(
        config=g3_cfg, data_dir=data_dir, eval_split="valid"
    )

    all_tasks: dict[str, Any] = {}
    paired_effects: dict[str, Any] = {}
    for task in (3, 5):
        train_stories = training_dataset.filter_task(task).stories
        eval_stories = validation_dataset.filter_task(task).stories
        if not train_stories or not eval_stories:
            raise ValueError(f"G4 QA{task} requires non-empty train and validation sets")
        vocab = build_babi_vocabulary(train_stories + eval_stories)
        answer_set = build_babi_answer_set(train_stories)
        bridge = build_babi_bridge(vocab, dim=cfg.dim)
        orders = _orders(len(train_stories), cfg.seeds, cfg.epochs)
        baseline_task = g3_reference["tasks"][str(task)]
        baseline_rows = baseline_task["arms"][_BASELINE_ARM]["seed_results"]
        g4_rows: list[dict[str, Any]] = []
        for seed in range(cfg.seeds):
            order_hash = _fingerprint(orders[seed])
            expected_g3_hash = baseline_task["training_order_hashes_by_seed"][str(seed)]
            if order_hash != expected_g3_hash:
                raise AssertionError(
                    f"G3/G4 training-order identity mismatch on QA{task}, seed {seed}"
                )
            g4_rows.append(_run_vectorized_seed(
                task=task, seed=seed, train_stories=train_stories,
                eval_stories=eval_stories, answer_set=answer_set, bridge=bridge,
                memory=memory, cfg=cfg, epoch_orders=orders[seed],
                data_order_hash=order_hash,
            ))

        paired = _verify_pairing(baseline_rows, g4_rows, eval_stories)
        baseline_by_seed = {int(row["seed"]): row for row in baseline_rows}
        g4_by_seed = {int(row["seed"]): row for row in g4_rows}
        task_effects: dict[str, Any] = {}
        for metric in ("accuracy", "nll", "brier", "fixed_ece", "adaptive_ece", "aurc"):
            differences = []
            for seed in range(cfg.seeds):
                left = float(baseline_by_seed[seed]["accuracy"]) if metric == "accuracy" else _confidence_value(baseline_by_seed[seed], metric)
                right = float(g4_by_seed[seed]["accuracy"]) if metric == "accuracy" else _confidence_value(g4_by_seed[seed], metric)
                differences.append(right - left)
            task_effects[metric] = _paired_effect(
                differences, f"G4 vectorized hierarchy minus matched G3 scalar hierarchy: {metric}"
            )
        mechanism = all(
            row["runtime"]["backend"] == "cpp_vectorized_at_both_hierarchy_levels"
            and row["hierarchy_diagnostics"]["finite_state_pass"]
            and row["hierarchy_diagnostics"]["bottom_up_coupling_configured"]
            and row["hierarchy_diagnostics"]["top_down_coupling_configured"]
            and row["hierarchy_diagnostics"]["bottom_up_context_activity_mean_state_norm"] > 0.0
            and row["hierarchy_diagnostics"]["top_down_gamma_dynamic_episode_fraction"] > 0.0
            and (
                not cfg.normalize_top_down_weights
                or (
                    abs(row["hierarchy_diagnostics"]["top_down_weight_sum_min"] - 1.0) <= 1e-9
                    and abs(row["hierarchy_diagnostics"]["top_down_weight_sum_max"] - 1.0) <= 1e-9
                )
            )
            for row in g4_rows
        )
        stability_recorded = all(
            row["hierarchy_diagnostics"]["stability_evidence_recorded"]
            for row in g4_rows
        )
        stability_qualified = all(
            all(row["hierarchy_diagnostics"]["stability_pass_by_level"])
            for row in g4_rows
        )
        confidence_complete = all(
            row["confidence"].get("negative_log_likelihood") is not None
            and row["confidence"].get("multiclass_brier_score") is not None
            and row["confidence"].get("calibration", {}).get("fixed_bin_top_label_ece") is not None
            and row["confidence"].get("calibration", {}).get("adaptive_equal_mass_top_label_ece") is not None
            and row["confidence"].get("selective_prediction", {}).get("area_under_risk_coverage_curve") is not None
            for row in g4_rows
        )
        base_summary = _summary_for_arm([
            {
                "seed": row["seed"], "accuracy": row["accuracy"],
                "confidence": row["confidence"],
                "initial_readout_sha256": row["initial_readout_sha256"],
                "training_order_sha256": row["training_order_sha256"],
            }
            for row in baseline_rows
        ])
        g4_summary = _summary_for_arm(g4_rows)
        all_tasks[str(task)] = {
            "task": task,
            "task_label": f"QA{task}",
            "training_story_count": len(train_stories),
            "validation_story_count": len(eval_stories),
            "answer_classes": answer_set,
            "baseline_g3_scalar_hierarchy": base_summary,
            "g4_vectorized_hierarchy": g4_summary,
            "paired_seed_and_example_identity_verified": paired,
            "paired_effects_g4_minus_g3": task_effects,
            "mechanism_audit": {
                "passed": mechanism,
                "per_seed": [
                    {"seed": row["seed"], **row["hierarchy_diagnostics"]}
                    for row in g4_rows
                ],
            },
            "stability_qualification": {
                "evidence_recorded": stability_recorded,
                "passed": stability_qualified,
                "margin": 0.9,
                "interpretation": (
                    "All observed vectorized kernels remain within the frozen margin."
                    if stability_qualified else
                    "Executed and measured, but at least one observed kernel exceeds the frozen 0.9 margin; no automatic rescaling was applied."
                ),
            },
            "confidence_evidence": {"passed": confidence_complete},
        }

    coverage = set(all_tasks) == {"3", "5"}
    pairing = all(row["paired_seed_and_example_identity_verified"] for row in all_tasks.values())
    mechanism = all(row["mechanism_audit"]["passed"] for row in all_tasks.values())
    stability_recorded = all(row["stability_qualification"]["evidence_recorded"] for row in all_tasks.values())
    confidence_complete = all(row["confidence_evidence"]["passed"] for row in all_tasks.values())
    return {
        "schema_version": 1,
        "experiment": "g4_vectorized_hierarchical_relational",
        "experiment_id": "G4-VECTORIZED-HIERARCHY",
        "status": "completed",
        "started_at_utc": g3_reference.get("started_at_utc"),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_revision": g3_reference.get("code_revision", "unknown"),
        "protocol": {
            **asdict(cfg),
            "tasks": [3, 5],
            "evaluation_split": "valid",
            "test_split_opened": False,
            "seed_ids": list(range(cfg.seeds)),
            "baseline": "G3 scalar-ABI two-level hierarchy, rerun on the same validation split",
            "g4_backend": "new vector-valued C++ digital-memory ABI at both hierarchy levels",
            "hierarchy": "two levels; full bottom-up plus top-down; G3 gains retained",
            "normalization": "G3 top-down positive-kernel weight normalization retained; unit-sum behavior recorded",
            "memory_form": "negative_feedback at each level; chi is channel-by-state-dimension vector",
            "numerical_difference_from_g3": (
                "G3 scalar hierarchy uses the legacy ABERSOE scalar chi and pre-step feedback; "
                "G4 vector digital ABI updates vector chi and computes feedback from post-step chi. "
                "This is an explicit numerical substrate change, not claimed to be bitwise parity."
            ),
            "stability_margin": 0.9,
            "locked_test_policy": "validation-only; this workflow does not download or open the test split",
        },
        "dataset": {"profile_id": "babi_canonical", "split_fingerprints": split_fingerprints},
        "tasks": all_tasks,
        "acceptance": {
            "qa3_qa5_validation_coverage": {"passed": coverage, "tasks": sorted(all_tasks)},
            "five_seed_pairing_and_example_identity": {"passed": pairing, "seeds": cfg.seeds},
            "vectorized_two_level_bidirectional_mechanism": {"passed": mechanism},
            "stability_evidence_recorded": {"passed": stability_recorded},
            "stability_within_frozen_margin": {
                "passed": all(row["stability_qualification"]["passed"] for row in all_tasks.values()),
                "qualification_only": True,
            },
            "confidence_evidence": {"passed": confidence_complete},
        },
        "claim_boundary": (
            "G4 measures the combined effect of vector-valued memory at both levels and the "
            "post-step vector feedback contract, while retaining G3 hierarchy topology, gains, "
            "top-down normalization, readout, seed identities and validation examples. It does "
            "not isolate vector dimensionality from the changed discrete feedback timing."
        ),
    }


def save_g4_bundle(result: dict[str, Any], output_dir: Path = _OUTPUT_DIR) -> dict[str, str]:
    run_id = f"g4_vectorized_hierarchy_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    bundle = output_dir / run_id
    bundle.mkdir(parents=True, exist_ok=True)
    summary = bundle / "summary.json"
    summary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return {"bundle_dir": str(bundle), "summary_json": str(summary)}


def main(argv: list[str] | None = None) -> dict[str, str]:
    parser = argparse.ArgumentParser(description="Run G4 vectorized hierarchy on QA3/QA5 validation only.")
    parser.add_argument("--data-dir", type=Path, default=_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=_OUTPUT_DIR)
    parser.add_argument("--seeds", type=int, default=_EXPECTED_SEEDS, choices=[_EXPECTED_SEEDS])
    parser.add_argument("--bootstrap-samples", type=int, default=400)
    args = parser.parse_args(argv)
    result = run_g4_validation(
        config=G4Config(seeds=args.seeds, bootstrap_samples=args.bootstrap_samples),
        data_dir=args.data_dir,
    )
    paths = save_g4_bundle(result, output_dir=args.output_dir)
    for task, row in result["tasks"].items():
        baseline = row["baseline_g3_scalar_hierarchy"]["mean_accuracy_over_seeds"]
        current = row["g4_vectorized_hierarchy"]["mean_accuracy_over_seeds"]
        effect = row["paired_effects_g4_minus_g3"]["accuracy"]["mean"]
        print(
            f"QA{task}: G3={baseline:.4f} G4={current:.4f} "
            f"paired_delta={effect:+.4f} "
            f"stability={row['stability_qualification']['passed']}"
        )
    print("acceptance=" + json.dumps(result["acceptance"], sort_keys=True))
    print(f"bundle={paths['bundle_dir']}")
    return paths


if __name__ == "__main__":
    main()
