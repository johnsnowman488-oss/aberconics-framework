"""G2: relational composition, matched QA3/QA5 controls and confidence audit.

G2 keeps the learned token-readout comparison (E0 vs isolated E-F0) distinct
from the structured relational fixtures. The latter are a semantic control,
not a natural-language parser, not bAbI accuracy, and not a learned result.
G1-L remains a QA2-only learned binder and is not silently extrapolated to
QA3/QA5.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .babi_qa import BabiQAConfig, run_babi_qa_experiment
from .babi_f0 import BabiF0Config, run_babi_f0_experiment
from .confidence_metrics import summarize_confidence
from ..digital.babi import BabiDataset

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DATA_DIR = _REPO_ROOT / "code" / "python" / "d2c" / "progress" / "babi_data"
_FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "g2_relational_cases.json"
_TASK_LABELS = {3: "QA3 · Three supporting facts", 5: "QA5 · Three-argument relations"}
_T_CRITICAL_975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
    7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
    13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
    19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064,
    25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


@dataclass(frozen=True, slots=True)
class G2Config:
    tasks: tuple[int, ...] = (3, 5)
    dim: int = 64
    seeds: int = 5
    epochs: int = 5
    learning_rate: float = 0.05
    feature_scale: float = 100.0
    readout: str = "linear"
    bootstrap_samples: int = 400

    def __post_init__(self) -> None:
        if not self.tasks or any(task not in {3, 5} for task in self.tasks):
            raise ValueError("G2 supports QA3 (task 3) and QA5 (task 5) only")
        if len(set(self.tasks)) != len(self.tasks):
            raise ValueError("G2 task list must not contain duplicates")
        if self.dim < 4 or self.seeds < 3 or self.seeds > 20 or self.epochs < 1:
            raise ValueError("dimension, seed count (3–20), or epochs are invalid")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.feature_scale) or self.feature_scale <= 0:
            raise ValueError("feature_scale must be finite and positive")
        if self.readout not in {"linear", "mlp"}:
            raise ValueError("readout must be linear or mlp")
        if self.bootstrap_samples < 200 or self.bootstrap_samples > 5000:
            raise ValueError("bootstrap_samples must be between 200 and 5000 for G2 confidence gates")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _revision() -> str:
    if os.environ.get("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT,
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _resolve_person(name: str, aliases: dict[str, str]) -> str:
    key = name.strip().lower()
    seen: set[str] = set()
    while key in aliases:
        if key in seen:
            raise ValueError(f"cyclic coreference alias at {key!r}")
        seen.add(key)
        key = aliases[key].strip().lower()
    return key


def replay_relational_fixture(case: dict[str, Any]) -> dict[str, Any]:
    """Replay hand-authored typed events; aliases are explicit oracle inputs."""
    aliases = {str(k).lower(): str(v).lower() for k, v in case.get("aliases", {}).items()}
    person_locations: dict[str, str] = {}
    object_holders: dict[str, str] = {}
    object_locations: dict[str, str | None] = {}
    ledger: list[dict[str, Any]] = []
    for index, event in enumerate(case.get("events", [])):
        kind = event["type"]
        if kind == "move":
            actor = _resolve_person(event["actor"], aliases)
            before = person_locations.get(actor)
            person_locations[actor] = str(event["location"]).lower()
            ledger.append({"index": index, "type": kind, "actor": actor, "before": before, "after": person_locations[actor]})
        elif kind == "pickup":
            actor, obj = _resolve_person(event["actor"], aliases), str(event["object"]).lower()
            before = object_locations.get(obj)
            object_holders[obj] = actor
            object_locations.pop(obj, None)
            ledger.append({"index": index, "type": kind, "actor": actor, "object": obj, "before": before, "after_holder": actor})
        elif kind == "transfer":
            obj = str(event["object"]).lower()
            previous = object_holders.get(obj)
            expected_from = _resolve_person(event["from"], aliases)
            recipient = _resolve_person(event["to"], aliases)
            if previous != expected_from:
                raise ValueError(f"transfer at event {index} expected holder {expected_from!r}, found {previous!r}")
            object_holders[obj] = recipient
            object_locations.pop(obj, None)
            ledger.append({"index": index, "type": kind, "object": obj, "from": previous, "to": recipient})
        elif kind == "release":
            actor, obj = _resolve_person(event["actor"], aliases), str(event["object"]).lower()
            previous = object_holders.get(obj)
            if previous != actor:
                raise ValueError(f"release at event {index} actor {actor!r} is not current holder {previous!r}")
            location = person_locations.get(actor)
            object_holders.pop(obj, None)
            object_locations[obj] = location
            ledger.append({"index": index, "type": kind, "actor": actor, "object": obj, "fixed_location": location})
        else:
            raise ValueError(f"unsupported fixture event type: {kind!r}")
    query_object = str(case["query_object"]).lower()
    holder = object_holders.get(query_object)
    location = person_locations.get(holder) if holder is not None else object_locations.get(query_object)
    return {
        "query_object": query_object,
        "actual_holder": holder,
        "actual_location": location,
        "person_locations": dict(sorted(person_locations.items())),
        "object_holders": dict(sorted(object_holders.items())),
        "fixed_object_locations": dict(sorted(object_locations.items())),
        "event_ledger": ledger,
    }


def run_fixture_suite(fixture_path: Path = _FIXTURE_PATH) -> dict[str, Any]:
    payload = json.loads(Path(fixture_path).read_text(encoding="utf-8"))
    results = []
    for case in payload["cases"]:
        try:
            actual = replay_relational_fixture(case)
            errors = []
        except (KeyError, TypeError, ValueError) as exc:
            actual = {"actual_holder": None, "actual_location": None, "event_ledger": []}
            errors = [str(exc)]
        expected_holder = case.get("expected_holder")
        expected_location = case.get("expected_location")
        passed = (
            actual.get("actual_holder") == expected_holder
            and actual.get("actual_location") == expected_location
            and not errors
        )
        results.append({
            "fixture_id": case["id"],
            "family": case["family"],
            "category": case["category"],
            "description": case["description"],
            "expected_holder": expected_holder,
            "actual_holder": actual.get("actual_holder"),
            "expected_location": expected_location,
            "actual_location": actual.get("actual_location"),
            "passed": passed,
            "errors": errors,
            "state": actual,
        })
    categories: dict[str, dict[str, int]] = {}
    for row in results:
        item = categories.setdefault(row["category"], {"passed": 0, "total": 0})
        item["total"] += 1
        item["passed"] += int(row["passed"])
    return {
        "schema_version": payload.get("schema_version", 1),
        "claim_boundary": payload.get("claim_boundary"),
        "total": len(results),
        "passed": sum(row["passed"] for row in results),
        "failed": sum(not row["passed"] for row in results),
        "pass_rate": sum(row["passed"] for row in results) / len(results) if results else 0.0,
        "by_category": categories,
        "results": results,
    }


def _paired_interval(differences: list[float]) -> dict[str, Any]:
    n = len(differences)
    if not n:
        return {"n": 0, "mean": None, "standard_deviation": None, "standard_error": None, "ci95": None}
    mean = statistics.mean(differences)
    sd = statistics.stdev(differences) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else None
    if n < 2:
        ci = None
    else:
        critical = _T_CRITICAL_975.get(n - 1, 1.96)
        half = critical * se
        ci = [mean - half, mean + half]
    positive = sum(value > 0 for value in differences)
    negative = sum(value < 0 for value in differences)
    nonzero = positive + negative
    minority = min(positive, negative)
    sign_p = (
        min(1.0, 2.0 * sum(math.comb(nonzero, k) for k in range(minority + 1)) / (2 ** nonzero))
        if nonzero else 1.0
    )
    return {
        "n": n, "mean": mean, "standard_deviation": sd,
        "standard_error": se, "ci95_two_sided_t": ci,
        "exact_two_sided_sign_test_p": sign_p,
        "sign_test_nonzero_pairs": nonzero,
        "sign_test_interpretation": "Exact paired sign test; robust to difference magnitudes but low-power with few seeds. Report alongside—not instead of—the paired t interval.",
        "positive_seed_differences": sum(value > 0 for value in differences),
        "negative_seed_differences": sum(value < 0 for value in differences),
        "zero_seed_differences": sum(value == 0 for value in differences),
        "differences_ef0_minus_e0": differences,
    }


def _prediction_identity(rows: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    return [
        (
            int(row.get("story_index", -1)),
            str(row.get("raw_passage", "")),
            str(row.get("question")),
            str(row.get("target")),
        )
        for row in rows
    ]


def _confidence_for_seed(seed_row: dict[str, Any], *, labels: list[str], bootstrap_samples: int, bootstrap_seed: int) -> dict[str, Any]:
    predictions = seed_row["predictions"]
    diag = summarize_confidence(
        predictions, labels, calibration_bins=10,
        bootstrap_samples=bootstrap_samples, bootstrap_seed=bootstrap_seed,
    )
    # Ensure the score derived from the probability vector agrees with the runner.
    runner_correct = sum(bool(row["correct"]) for row in predictions)
    if diag.get("correct") != runner_correct:
        raise AssertionError("confidence diagnostics disagree with runner correctness")
    return diag


def _arm_summary(result: dict[str, Any], *, arm: str, bootstrap_samples: int, task: int) -> dict[str, Any]:
    seed_rows = result["seed_results"]
    if not seed_rows:
        raise ValueError(f"{arm} returned no seed results for task {task}")
    predictions = seed_rows[0]["predictions"]
    labels = list(predictions[0]["probabilities"]) if predictions else []
    confidence = []
    enriched_seed_rows = []
    for seed_row in seed_rows:
        per_seed_diag = _confidence_for_seed(
            seed_row, labels=labels, bootstrap_samples=bootstrap_samples,
            bootstrap_seed=(task * 10_000 + int(seed_row["seed"]) * 101 + (0 if arm == "e0" else 1)),
        )
        flags = []
        fixed_ece = per_seed_diag["calibration"]["fixed_bin_top_label_ece"]
        adaptive_ece = per_seed_diag["calibration"]["adaptive_equal_mass_top_label_ece"]
        wrong = per_seed_diag["incorrect_count"]
        wrong_high = per_seed_diag["overconfident_error_fraction"]["0.9"]
        high_confidence = per_seed_diag["selective_prediction"]["confidence_thresholds"]["0.9"]
        if max(fixed_ece, adaptive_ece) > 0.10:
            flags.append("calibration_error_above_0.10")
        if wrong >= 20 and wrong_high is not None and wrong_high > 0.25:
            flags.append("more_than_25_percent_of_errors_have_confidence_at_least_0.90")
        if high_confidence["count"] >= 30 and high_confidence["risk"] is not None and high_confidence["risk"] > 0.10:
            flags.append("high_confidence_selective_risk_above_0.10")
        enriched_seed_rows.append({
            "seed": int(seed_row["seed"]),
            "correct": int(seed_row["correct"]),
            "total": int(seed_row["total"]),
            "accuracy": float(seed_row["accuracy"]),
            "confidence": per_seed_diag,
            "confidence_quality_flags": flags,
            "predictions": seed_row["predictions"],
        })
        confidence.append(per_seed_diag)
    return {
        "arm": arm,
        "experiment": result["experiment"],
        "mean_accuracy_over_seeds": result["mean_accuracy"],
        "standard_error_over_seeds": result.get("standard_error", result.get("se_accuracy")),
        "evaluation_split": result["evaluation_split"],
        "evaluation_stories": result["evaluation_stories"],
        "seed_results": enriched_seed_rows,
        "confidence_summary": {
            "mean_fixed_bin_top_label_ece_over_seeds": statistics.mean(
                row["calibration"]["fixed_bin_top_label_ece"] for row in confidence
            ),
            "mean_adaptive_top_label_ece_over_seeds": statistics.mean(
                row["calibration"]["adaptive_equal_mass_top_label_ece"] for row in confidence
            ),
            "mean_nll_over_seeds": statistics.mean(row["negative_log_likelihood"] for row in confidence),
            "mean_brier_over_seeds": statistics.mean(row["multiclass_brier_score"] for row in confidence),
            "mean_aurc_over_seeds": statistics.mean(
                row["selective_prediction"]["area_under_risk_coverage_curve"] for row in confidence
            ),
            "overconfident_error_fraction_at_0_90_by_seed": [
                row["overconfident_error_fraction"]["0.9"] for row in confidence
            ],
            "quality_flags_by_seed": [
                {"seed": row["seed"], "flags": row["confidence_quality_flags"]}
                for row in enriched_seed_rows
            ],
            "interpretation": "Arithmetic mean of per-seed diagnostics; do not pool seed × story predictions as independent observations.",
        },
    }


def run_g2_experiment(
    *,
    config: G2Config | None = None,
    data_dir: Path = _DATA_DIR,
    fixture_path: Path = _FIXTURE_PATH,
    eval_split: str = "test",
) -> dict[str, Any]:
    cfg = config or G2Config()
    if eval_split not in {"valid", "test"}:
        raise ValueError("eval_split must be valid or test")
    started = datetime.now(timezone.utc).isoformat()
    data_dir = Path(data_dir)
    required = [data_dir / "babi_train.jsonl", data_dir / f"babi_{eval_split}.jsonl"]
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("G2 requires dataset files: " + ", ".join(missing))
    dataset = {
        split: {
            "sha256": _sha256(data_dir / f"babi_{split}.jsonl"),
            "bytes": (data_dir / f"babi_{split}.jsonl").stat().st_size,
        }
        for split in ("train", "valid", "test")
        if (data_dir / f"babi_{split}.jsonl").is_file()
    }
    counts = {}
    for split in ("train", eval_split):
        ds = BabiDataset.from_jsonl(data_dir / f"babi_{split}.jsonl")
        counts[split] = {str(task): sum(story.task == task for story in ds.stories) for task in cfg.tasks}
        if any(counts[split][str(task)] == 0 for task in cfg.tasks):
            raise ValueError(f"G2 split {split} is missing QA3 or QA5 stories")

    fixtures = run_fixture_suite(fixture_path)
    task_results: dict[str, Any] = {}
    for task in cfg.tasks:
        qa_config = BabiQAConfig(
            task=task, variant="full", dim=cfg.dim, seeds=cfg.seeds,
            hidden_dim=32, learning_rate=cfg.learning_rate,
            readout_type=cfg.readout, feature_scale=cfg.feature_scale,
            epochs=cfg.epochs,
        )
        f0_config = BabiF0Config(
            task=task, variant="full", dim=cfg.dim, seeds=cfg.seeds,
            epochs=cfg.epochs, hidden_dim=32,
            learning_rate=cfg.learning_rate, readout_type=cfg.readout,
            feature_scale=cfg.feature_scale, mode="isolated",
            exercise_spine=False,
        )
        # These are paired, pre-registered settings. The locked test split is
        # evaluated once and is not used to select hyperparameters.
        e0_result = run_babi_qa_experiment(qa_config, data_dir=data_dir, eval_split=eval_split)
        ef0_result = run_babi_f0_experiment(f0_config, data_dir=data_dir, eval_split=eval_split)
        e0 = _arm_summary(e0_result, arm="e0", bootstrap_samples=cfg.bootstrap_samples, task=task)
        ef0 = _arm_summary(ef0_result, arm="ef0", bootstrap_samples=cfg.bootstrap_samples, task=task)

        e0_seeds = [row["seed"] for row in e0["seed_results"]]
        ef0_seeds = [row["seed"] for row in ef0["seed_results"]]
        e0_labels = list(e0_result["seed_results"][0]["predictions"][0]["probabilities"])
        ef0_labels = list(ef0_result["seed_results"][0]["predictions"][0]["probabilities"])
        paired_ok = (
            e0_seeds == ef0_seeds == list(range(cfg.seeds))
            and e0_labels == ef0_labels
            and e0_result["evaluation_stories"] == ef0_result["evaluation_stories"]
        )
        for e0_row, ef0_row in zip(e0_result["seed_results"], ef0_result["seed_results"]):
            if _prediction_identity(e0_row["predictions"]) != _prediction_identity(ef0_row["predictions"]):
                paired_ok = False
                break
        if not paired_ok:
            raise AssertionError(f"E0/E-F0 seed or example pairing failed for QA{task}")
        differences = [
            float(ef0_row["accuracy"]) - float(e0_row["accuracy"])
            for e0_row, ef0_row in zip(e0_result["seed_results"], ef0_result["seed_results"])
        ]
        task_results[str(task)] = {
            "task": task,
            "task_label": _TASK_LABELS[task],
            "evaluation_split": eval_split,
            "train_stories": counts["train"][str(task)],
            "evaluation_stories": counts[eval_split][str(task)],
            "paired_controls_verified": True,
            "paired_accuracy_difference_ef0_minus_e0": _paired_interval(differences),
            "arms": {"e0": e0, "ef0": ef0},
        }

    return {
        "schema_version": 1,
        "experiment": "g2_relational_composition",
        "status": "completed",
        "started_at_utc": started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            **asdict(cfg),
            "tasks": list(cfg.tasks),
            "variant": "full",
            "mode": "isolated",
            "same_seed_ids_across_arms": True,
            "evaluation_split": eval_split,
            "test_split_used_for_selection": False,
            "soe_kernel_modified": False,
            "runtime_ablation": "deferred",
            "hierarchy": "held_fixed_for_G2",
        },
        "dataset": {"fingerprints": dataset, "task_story_counts": counts},
        "fixture_control": fixtures,
        "tasks": task_results,
        "acceptance": {
            "fixture_suite": {"passed": fixtures["failed"] == 0 and fixtures["total"] >= 12, "passed": fixtures["passed"], "total": fixtures["total"], "minimum_required": 12},
            "qa3_qa5_benchmark_coverage": {
                "passed": set(cfg.tasks) == {3, 5},
                "tasks_requested": list(cfg.tasks),
                "note": "A task-subset run is useful for diagnosis but does not close the full G2 task-coverage gate.",
            },
            "matched_seed_and_example_controls": {
                "passed": all(row["paired_controls_verified"] for row in task_results.values()),
                "seed_count": cfg.seeds,
            },
            "confidence_evidence": {
                "passed": all(
                    all(
                        seed["confidence"].get("status") == "ok"
                        and seed["confidence"].get("bootstrap", {}).get("replicates_completed", 0) >= 200
                        and seed["confidence"].get("accuracy_wilson_95_ci") is not None
                        for arm in task_row["arms"].values() for seed in arm["seed_results"]
                    )
                    for task_row in task_results.values()
                ),
                "required_metrics": [
                    "Wilson 95% accuracy interval",
                    "story-bootstrap 95% intervals for accuracy, NLL, Brier, fixed/adaptive ECE and AURC",
                    "multiclass NLL and Brier proper scoring rules",
                    "fixed-width, equal-mass and classwise calibration error",
                    "confidence-on-error and high-confidence risk",
                    "risk-coverage curve and AURC",
                    "per-seed results and matched-seed paired confidence interval",
                ],
            },
            "claim_boundary": "Passing G2 mechanics and benchmark coverage qualifies the G2 evidence bundle; it does not establish that E-F0 is superior unless the paired seed interval supports that conclusion.",
        },
        "claim_boundary": (
            "G2 benchmark scores are E0 and E-F0 token-readout results on bAbI QA3/QA5. "
            "Structured relational fixtures validate explicit event semantics only. "
            "G1-L remains QA2-only and is not represented as a QA3/QA5 learned result; "
            "F0 is isolated and used as a runtime-integrated arm, not as proof that its "
            "hidden state is learned by the separate G1-L binder."
        ),
    }


def save_g2_bundle(result: dict[str, Any], *, output_dir: Path) -> dict[str, str]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    bundle = Path(output_dir) / f"g2_{stamp}"
    bundle.mkdir(parents=True, exist_ok=True)
    (bundle / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    rows = []
    for task_key, task in result["tasks"].items():
        for arm_name, arm in task["arms"].items():
            for seed in arm["seed_results"]:
                rows.append({
                    "task": task_key, "arm": arm_name, "seed": seed["seed"],
                    "correct": seed["correct"], "total": seed["total"],
                    "accuracy": seed["accuracy"],
                    "nll": seed["confidence"]["negative_log_likelihood"],
                    "brier": seed["confidence"]["multiclass_brier_score"],
                    "fixed_ece": seed["confidence"]["calibration"]["fixed_bin_top_label_ece"],
                    "adaptive_ece": seed["confidence"]["calibration"]["adaptive_equal_mass_top_label_ece"],
                    "aurc": seed["confidence"]["selective_prediction"]["area_under_risk_coverage_curve"],
                })
    fields = list(rows[0]) if rows else ["task", "arm", "seed", "correct", "total", "accuracy", "nll", "brier", "fixed_ece", "adaptive_ece", "aurc"]
    with (bundle / "per_seed_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    per_example_rows = []
    for task_key, task in result["tasks"].items():
        for arm_name, arm in task["arms"].items():
            for seed in arm["seed_results"]:
                for prediction in seed["predictions"]:
                    per_example_rows.append({
                        "task": int(task_key), "arm": arm_name, "seed": seed["seed"], **prediction
                    })
    with (bundle / "per_example.jsonl").open("w", encoding="utf-8") as handle:
        for row in per_example_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    example_fields = [
        "task", "arm", "seed", "story_index", "question", "target", "prediction",
        "correct", "confidence", "top_two_margin", "failure_category", "probability_vector",
    ]
    with (bundle / "per_example.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=example_fields)
        writer.writeheader()
        for row in per_example_rows:
            writer.writerow({
                key: json.dumps(row.get(key), sort_keys=True) if key == "probability_vector" else row.get(key)
                for key in example_fields
            })
    with (bundle / "fixtures.jsonl").open("w", encoding="utf-8") as handle:
        for row in result["fixture_control"]["results"]:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    report = [
        "# G2 relational composition report",
        "",
        f"- Fixture control: {result['fixture_control']['passed']}/{result['fixture_control']['total']} passed",
        f"- Full QA3 + QA5 coverage: {result['acceptance']['qa3_qa5_benchmark_coverage']['passed']}",
        f"- Matched controls: {result['acceptance']['matched_seed_and_example_controls']['passed']}",
        f"- Confidence evidence complete: {result['acceptance']['confidence_evidence']['passed']}",
        "",
        "## Paired benchmark comparisons",
        "",
    ]
    for key, task in result["tasks"].items():
        delta = task["paired_accuracy_difference_ef0_minus_e0"]
        report += [
            f"### QA{key}",
            f"- Stories: train={task['train_stories']}, {task['evaluation_split']}={task['evaluation_stories']}",
            f"- E0 mean accuracy across seeds: {task['arms']['e0']['mean_accuracy_over_seeds']:.4f}",
            f"- E-F0 mean accuracy across seeds: {task['arms']['ef0']['mean_accuracy_over_seeds']:.4f}",
            f"- Paired E-F0 − E0 accuracy: {delta['mean']:.4f}; 95% t interval: {delta['ci95_two_sided_t']}",
            f"- Exact paired sign-test p: {delta['exact_two_sided_sign_test_p']:.4f}; positive/negative/zero seeds: {delta['positive_seed_differences']}/{delta['negative_seed_differences']}/{delta['zero_seed_differences']}",
            f"- E0 mean fixed/adaptive ECE: {task['arms']['e0']['confidence_summary']['mean_fixed_bin_top_label_ece_over_seeds']:.4f} / {task['arms']['e0']['confidence_summary']['mean_adaptive_top_label_ece_over_seeds']:.4f}",
            f"- E-F0 mean fixed/adaptive ECE: {task['arms']['ef0']['confidence_summary']['mean_fixed_bin_top_label_ece_over_seeds']:.4f} / {task['arms']['ef0']['confidence_summary']['mean_adaptive_top_label_ece_over_seeds']:.4f}",
            "",
        ]
    report += [
        "## Confidence interpretation",
        "",
        "All confidence metrics are descriptive on the designated evaluation split. No post-hoc temperature scaling or threshold tuning is performed on test. Fixed-bin ECE, adaptive equal-mass ECE and classwise ECE are all reported because calibration error is bin-dependent and sample-sensitive. NLL and multiclass Brier are proper scoring rules. Story-bootstrap intervals quantify within-run sampling uncertainty; paired t intervals across seed-level differences quantify training-seed variability.",
        "",
        f"**Claim boundary:** {result['claim_boundary']}",
    ]
    (bundle / "summary.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    now = datetime.now(timezone.utc).isoformat()
    metadata = {
        "schema_version": 1,
        "run_id": bundle.name,
        "experiment": result["experiment"],
        "status": "completed",
        "created_at": result["started_at_utc"],
        "completed_at": result["completed_at_utc"],
        "code_revision": _revision(),
        "dataset_fingerprints": result["dataset"]["fingerprints"],
        "configuration": result["configuration"],
        "result_files": ["summary.json", "summary.md", "per_seed_metrics.csv", "per_example.csv", "per_example.jsonl", "fixtures.jsonl"],
    }
    (bundle / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (bundle / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (bundle / "config.json").write_text(json.dumps(result["configuration"], indent=2), encoding="utf-8")
    register = {
        "schema_version": 1,
        "experiment": "G2",
        "run_id": bundle.name,
        "code_revision": metadata["code_revision"],
        "dataset_fingerprints": result["dataset"]["fingerprints"],
        "configuration": result["configuration"],
        "acceptance": result["acceptance"],
        "fixture_control": {
            "passed": result["fixture_control"]["passed"],
            "total": result["fixture_control"]["total"],
            "by_category": result["fixture_control"]["by_category"],
        },
        "tasks": {
            key: {
                "paired_accuracy_difference": task["paired_accuracy_difference_ef0_minus_e0"],
                "e0": task["arms"]["e0"]["confidence_summary"],
                "ef0": task["arms"]["ef0"]["confidence_summary"],
            }
            for key, task in result["tasks"].items()
        },
        "claim_boundary": result["claim_boundary"],
    }
    (bundle / "experiment_lab_register.json").write_text(json.dumps(register, indent=2), encoding="utf-8")
    return {
        "bundle_dir": str(bundle),
        "summary_json": str(bundle / "summary.json"),
        "summary_md": str(bundle / "summary.md"),
        "lab_register": str(bundle / "experiment_lab_register.json"),
    }


def _parse_tasks(value: str) -> tuple[int, ...]:
    try:
        tasks = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("tasks must be a comma-separated subset of 3,5") from exc
    if not tasks or any(task not in {3, 5} for task in tasks) or len(set(tasks)) != len(tasks):
        raise argparse.ArgumentTypeError("tasks must be 3, 5, or 3,5")
    return tasks


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run G2 relational composition QA3/QA5 benchmark and controls.")
    parser.add_argument("--tasks", type=_parse_tasks, default=(3, 5))
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--feature-scale", type=float, default=100.0)
    parser.add_argument("--readout", choices=("linear", "mlp"), default="linear")
    parser.add_argument("--bootstrap-samples", type=int, default=400)
    parser.add_argument("--eval-split", choices=("valid", "test"), default="test")
    parser.add_argument("--data-dir", type=Path, default=_DATA_DIR)
    parser.add_argument("--fixture-path", type=Path, default=_FIXTURE_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> dict[str, str]:
    args = build_arg_parser().parse_args(argv)
    config = G2Config(
        tasks=args.tasks, dim=args.dim, seeds=args.seeds, epochs=args.epochs,
        learning_rate=args.learning_rate, feature_scale=args.feature_scale,
        readout=args.readout, bootstrap_samples=args.bootstrap_samples,
    )
    result = run_g2_experiment(config=config, data_dir=args.data_dir, fixture_path=args.fixture_path, eval_split=args.eval_split)
    paths = save_g2_bundle(result, output_dir=args.output_dir)
    for task, row in result["tasks"].items():
        delta = row["paired_accuracy_difference_ef0_minus_e0"]
        print(
            f"QA{task}: E0={row['arms']['e0']['mean_accuracy_over_seeds']:.4%}, "
            f"E-F0={row['arms']['ef0']['mean_accuracy_over_seeds']:.4%}, "
            f"paired delta={delta['mean']:.4%} CI={delta['ci95_two_sided_t']}"
        )
    print(f"fixtures={result['fixture_control']['passed']}/{result['fixture_control']['total']}")
    print(f"confidence_evidence={result['acceptance']['confidence_evidence']['passed']}")
    print(f"bundle={paths['bundle_dir']}")
    return paths


if __name__ == "__main__":
    main()
