"""Local-first Experiment Lab API. Run with: python -m d2c.experiment_lab.server"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[4]
RUNS = ROOT / "runs" / "experiment-lab"
PROFILE_REGISTRY = Path(__file__).with_name("dataset_profiles.json")
LOCK = threading.Lock()
ACTIVE = None
JOBS = {}

EXPERIMENTS = [
    {
        "id": "e0",
        "name": "E0 bAbI baseline",
        "description": "Supervised readout over frozen SOE state.",
        "modes": ["isolated"],
        "tasks": "babi",
        "defaults": {
            "task": 2, "dim": 64, "seeds": 3, "epochs": 5,
            "learning_rate": 0.05, "feature_scale": 100.0,
            "readout": "linear", "mode": "isolated",
            "eval_split": "test", "quick": False,
        },
    },
    {
        "id": "ef0",
        "name": "E-F0 runtime-integrated bAbI",
        "description": "E0 readout with F0 runtime scheduling and diagnostics.",
        "modes": ["isolated", "continuous"],
        "tasks": "babi",
        "defaults": {
            "task": 2, "dim": 64, "seeds": 3, "epochs": 5,
            "learning_rate": 0.05, "feature_scale": 100.0,
            "readout": "linear", "mode": "isolated",
            "eval_split": "test", "exercise_spine": True, "quick": False,
        },
    },
    {
        "id": "g1",
        "name": "G1 explicit relational-state QA2 diagnostic",
        "description": "Deterministic symbolic control; not a learned QA score.",
        "modes": ["isolated"],
        "tasks": [2],
        "defaults": {"task": 2, "dim": 64, "mode": "isolated", "quick": False},
    },
    {
        "id": "g1l",
        "name": "G1-L learned relational binding",
        "description": "QA2 typed-event binder with a validation pilot and gated test confirmation.",
        "modes": ["isolated"],
        "tasks": [2],
        "stages": ["pilot", "all"],
        "defaults": {
            "task": 2, "dim": 64, "stage": "pilot", "epochs": 3,
            "learning_rate": 0.02, "train_limit": 200, "eval_limit": 100,
            "confirmation_seeds": 3, "mode": "isolated",
        },
    },
    {
        "id": "g2",
        "name": "G2 relational composition QA3/QA5",
        "description": "Matched E0/E-F0 benchmark plus structured semantic fixtures.",
        "modes": ["isolated"],
        "tasks": [3, 5],
        "defaults": {
            "tasks": [3, 5], "dim": 64, "seeds": 5, "epochs": 5,
            "learning_rate": 0.05, "feature_scale": 100.0,
            "readout": "linear", "mode": "isolated",
            "eval_split": "test", "bootstrap_samples": 400,
        },
    },
    {
        "id": "g3",
        "name": "G3 flat ABI vs full hierarchy",
        "description": "Matched QA3/QA5 benchmark: flat ABI control versus full bottom-up + top-down hierarchy, with per-seed F0 spine audit.",
        "modes": ["isolated"],
        "tasks": [3, 5],
        "defaults": {
            "tasks": [3, 5], "dim": 64, "seeds": 20, "epochs": 3,
            "learning_rate": 0.05, "feature_scale": 10.0,
            "readout": "linear", "mode": "isolated",
            "eval_split": "test", "bootstrap_samples": 400,
        },
    },
]
EXPERIMENT_BY_ID = {item["id"]: item for item in EXPERIMENTS}


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def code_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def dataset_profiles() -> dict[str, dict]:
    """Load trusted, repository-relative dataset entries from the registry."""
    registry = json.loads(PROFILE_REGISTRY.read_text(encoding="utf-8"))
    profiles = {}
    for entry in registry.get("profiles", []):
        profile_id = entry.get("id")
        name = entry.get("name")
        relative_directory = entry.get("directory")
        splits = entry.get("splits")
        if (
            not isinstance(profile_id, str)
            or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", profile_id)
            or not isinstance(name, str)
            or not isinstance(relative_directory, str)
            or Path(relative_directory).is_absolute()
            or not isinstance(splits, dict)
            or not {"train", "valid", "test"}.issubset(splits)
        ):
            raise ValueError("invalid dataset profile registry entry")
        directory = (ROOT / relative_directory).resolve()
        try:
            relative_path = directory.relative_to(ROOT)
        except ValueError as exc:
            raise ValueError(f"dataset profile {profile_id!r} must stay inside the repository") from exc
        files = {}
        for split in ("train", "valid", "test"):
            filename = splits[split]
            expected_filename = f"babi_{split}.jsonl"
            if filename != expected_filename:
                raise ValueError(
                    f"dataset profile {profile_id!r} must use {expected_filename!r} "
                    "because the bAbI runners expect canonical split filenames"
                )
            path = directory / filename
            try:
                path.resolve().relative_to(directory)
            except ValueError as exc:
                raise ValueError(f"{split} file in dataset profile {profile_id!r} must stay in its directory") from exc
            files[split] = path
        if profile_id in profiles:
            raise ValueError(f"duplicate dataset profile id: {profile_id}")
        profiles[profile_id] = {
            "id": profile_id,
            "name": name,
            "directory": directory,
            "relative_directory": relative_path.as_posix(),
            "files": files,
        }
    return profiles


def profile_status(profile: dict) -> dict:
    files = {split: path.is_file() for split, path in profile["files"].items()}
    return {
        "id": profile["id"],
        "name": profile["name"],
        "directory": profile["relative_directory"],
        "files": files,
        "ready_for_benchmark": files["train"] and files["test"],
        "ready_for_g1l": all(files.values()),
    }


def dataset_fingerprints(profile: dict) -> dict:
    return {
        split: {"sha256": file_sha256(path), "bytes": path.stat().st_size}
        for split, path in profile["files"].items()
        if path.is_file()
    }


def _positive_number(value, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def build_launch(data: dict, directory: Path) -> tuple[list[str], dict, dict]:
    experiment = data.get("experiment")
    if experiment not in EXPERIMENT_BY_ID:
        raise ValueError("unknown experiment")
    defaults = EXPERIMENT_BY_ID[experiment]["defaults"]
    settings = defaults | data
    profile_id = settings.get("dataset_profile", "babi_canonical")
    profiles = dataset_profiles()
    profile = profiles.get(profile_id)
    if profile is None:
        raise ValueError("unknown dataset profile")

    mode = settings.get("mode", "isolated")
    dim = int(settings.get("dim", defaults["dim"]))
    required_splits = (
        ("train", "valid", "test") if experiment == "g1l"
        else ("train", settings.get("eval_split", "test"))
    )
    if experiment == "g1":
        required_splits = ("train", "test")
    missing = [split for split in required_splits if not profile["files"][split].is_file()]
    if missing:
        raise ValueError(f"dataset profile {profile_id!r} is missing required splits: {', '.join(missing)}")

    if mode not in EXPERIMENT_BY_ID[experiment]["modes"]:
        raise ValueError("execution mode is not supported by this experiment")
    if not 4 <= dim <= 512 or (experiment == "g1l" and dim < 8):
        raise ValueError("memory dimension is outside this experiment's supported range")
    task = int(settings.get("task", 2))
    if experiment in {"g1", "g1l"} and task != 2:
        raise ValueError(f"{experiment.upper()} supports QA2 only")

    fingerprint_splits = (
        ("train", settings.get("eval_split", "test"))
        if experiment == "g3" else ("train", "valid", "test")
    )
    fingerprints = {
        split: {"sha256": file_sha256(profile["files"][split]), "bytes": profile["files"][split].stat().st_size}
        for split in fingerprint_splits
        if profile["files"][split].is_file()
    }
    config = {
        "experiment": experiment,
        "dataset": {
            "id": profile["id"],
            "name": profile["name"],
            "directory": profile["relative_directory"],
        },
        "dataset_fingerprints": fingerprints,
        "task": task if experiment not in {"g2", "g3"} else None,
        "tasks": None,
        "dim": dim,
        "mode": mode,
        "variant": "full",
    }
    module = {
        "e0": "d2c.experiments.babi_qa",
        "ef0": "d2c.experiments.babi_f0",
        "g1": "d2c.experiments.babi_g1",
        "g1l": "d2c.experiments.babi_g1l",
        "g2": "d2c.experiments.babi_g2",
        "g3": "d2c.experiments.babi_g3",
    }[experiment]
    args = [
        sys.executable, "-m", module, "--dim", str(dim),
        "--data-dir", str(profile["directory"]), "--output-dir", str(directory),
    ]

    if experiment in {"e0", "ef0"}:
        seeds = int(settings["seeds"])
        requested_epochs = int(settings["epochs"])
        learning_rate = _positive_number(settings["learning_rate"], "learning rate")
        feature_scale = _positive_number(settings["feature_scale"], "feature scale")
        readout = settings.get("readout", "linear")
        eval_split = settings.get("eval_split", "test")
        quick = settings.get("quick", False)
        if not 1 <= task <= 20 or not 1 <= seeds <= 100 or not 1 <= requested_epochs <= 100:
            raise ValueError("task, seed count or epoch count is outside its supported range")
        if readout not in {"linear", "mlp"} or eval_split not in {"valid", "test"}:
            raise ValueError("readout or evaluation split is invalid")
        if not isinstance(quick, bool):
            raise ValueError("quick must be boolean")
        if quick and experiment not in {"e0", "ef0"}:
            raise ValueError("quick mode is not supported by this experiment")
        epochs = 2 if quick else requested_epochs
        config.update({
            "seeds": seeds,
            "epochs": epochs,
            "requested_epochs": requested_epochs,
            "learning_rate": learning_rate,
            "feature_scale": feature_scale,
            "readout": readout,
            "evaluation_split": eval_split,
            "quick": quick,
        })
        args += [
            "--task", str(task), "--variant", "full", "--seeds", str(seeds),
            "--epochs", str(epochs), "--lr", str(learning_rate),
            "--feature-scale", str(feature_scale), "--readout", readout,
            "--eval-split", eval_split,
        ]
        if quick:
            args.append("--quick")
        if experiment == "ef0":
            exercise_spine = settings.get("exercise_spine", True)
            if not isinstance(exercise_spine, bool):
                raise ValueError("F0 component audit must be boolean")
            config["exercise_spine"] = exercise_spine
            args += ["--mode", mode]
            if exercise_spine:
                args.append("--exercise-spine")
    elif experiment == "g1":
        quick = settings.get("quick", False)
        if not isinstance(quick, bool):
            raise ValueError("quick must be boolean")
        config.update({
            "evaluation_split": "test",
            "learning_performed": False,
            "readout": "relational_state_resolution",
            "quick": quick,
        })
        args += ["--task", "2", "--variant", "full", "--trace-capacity", "20000"]
        if quick:
            args.append("--quick")
    elif experiment == "g1l":
        stage = settings.get("stage", "pilot")
        epochs = int(settings["epochs"])
        learning_rate = _positive_number(settings["learning_rate"], "learning rate")
        train_limit = int(settings["train_limit"])
        eval_limit = int(settings["eval_limit"])
        confirmation_seeds = int(settings["confirmation_seeds"])
        if stage not in {"pilot", "all"}:
            raise ValueError("G1-L stage must be pilot or all")
        if not 1 <= epochs <= 100 or not 1 <= train_limit <= 900 or not 1 <= eval_limit <= 1000:
            raise ValueError("G1-L epoch or story limits are outside their supported ranges")
        if stage == "all" and not 1 <= confirmation_seeds <= 10:
            raise ValueError("G1-L confirmation seeds must be between 1 and 10")
        config.update({
            "stage": stage,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "train_limit": train_limit,
            "eval_limit": eval_limit,
            "confirmation_seeds": confirmation_seeds if stage == "all" else None,
            "evaluation_split": "validation_then_test",
            "learning_performed": True,
            "readout": "learned_relational_binding",
            "f0_companion": True,
            "learning_path_uses_f0_features": False,
        })
        args += [
            "--stage", stage, "--task", "2", "--learning-rate", str(learning_rate),
            "--epochs", str(epochs), "--train-limit", str(train_limit),
            "--eval-limit", str(eval_limit),
        ]
        if stage == "all":
            args += ["--confirm-seeds", str(confirmation_seeds)]
    elif experiment == "g2":
        tasks = settings.get("tasks", [3, 5])
        if isinstance(tasks, str):
            tasks = [int(value.strip()) for value in tasks.split(",") if value.strip()]
        else:
            tasks = [int(value) for value in tasks]
        seeds = int(settings["seeds"])
        epochs = int(settings["epochs"])
        learning_rate = _positive_number(settings["learning_rate"], "learning rate")
        feature_scale = _positive_number(settings["feature_scale"], "feature scale")
        readout = settings.get("readout", "linear")
        eval_split = settings.get("eval_split", "test")
        bootstrap_samples = int(settings.get("bootstrap_samples", 400))
        if (
            not tasks or any(task not in {3, 5} for task in tasks)
            or len(set(tasks)) != len(tasks)
            or not 3 <= seeds <= 20
            or not 1 <= epochs <= 100
            or readout not in {"linear", "mlp"}
            or eval_split not in {"valid", "test"}
            or not 200 <= bootstrap_samples <= 5000
        ):
            raise ValueError("G2 task set or benchmark configuration is invalid")
        config.update({
            "task": None,
            "tasks": tasks,
            "seeds": seeds,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "feature_scale": feature_scale,
            "readout": readout,
            "evaluation_split": eval_split,
            "bootstrap_samples": bootstrap_samples,
            "learning_performed": True,
            "confidence_metrics": "full probability, calibration, selective risk and story bootstrap",
        })
        args += [
            "--tasks", ",".join(str(task) for task in tasks),
            "--seeds", str(seeds), "--epochs", str(epochs),
            "--learning-rate", str(learning_rate),
            "--feature-scale", str(feature_scale), "--readout", readout,
            "--bootstrap-samples", str(bootstrap_samples),
            "--eval-split", eval_split,
        ]
    elif experiment == "g3":
        tasks = settings.get("tasks", [3, 5])
        if isinstance(tasks, str):
            tasks = [int(value.strip()) for value in tasks.split(",") if value.strip()]
        else:
            tasks = [int(value) for value in tasks]
        seeds = int(settings.get("seeds", 20))
        epochs = int(settings.get("epochs", 3))
        learning_rate = _positive_number(settings.get("learning_rate", 0.05), "learning rate")
        feature_scale = _positive_number(settings.get("feature_scale", 10.0), "feature scale")
        eval_split = settings.get("eval_split", "test")
        bootstrap_samples = int(settings.get("bootstrap_samples", 400))
        if (
            tasks != [3, 5] or not 3 <= seeds <= 20 or epochs != 3
            or learning_rate != 0.05 or feature_scale != 10.0
            or eval_split not in {"valid", "test"}
            or not 200 <= bootstrap_samples <= 5000
        ):
            raise ValueError("G3 requires QA3+QA5, 3 epochs, LR 0.05, feature scale 10, 3–20 seeds, and 200–5000 bootstrap samples")
        config.update({
            "task": None, "tasks": tasks, "seeds": seeds, "epochs": epochs,
            "learning_rate": learning_rate, "feature_scale": feature_scale,
            "readout": "linear", "evaluation_split": eval_split,
            "bootstrap_samples": bootstrap_samples, "mode": "ABI scalar-chi",
            "arms": ["flat_abi", "hier_bu_td"],
            "hierarchy": "full bottom-up + top-down",
            "bottom_up_gain": 0.4, "top_down_gain": 0.5,
            "context_gamma": [0.35, 0.08], "context_w": [0.05, 0.02],
            "context_leak": 0.5, "hierarchy_kernel_form": "negative_feedback_form_b",
            "top_down_weight_normalization": False,
            "active_kernel_stability_margin": 0.9,
            "f0_spine_audit": True, "static_weight_control": "deferred",
            "learning_performed": True,
            "confidence_metrics": "NLL, Brier, fixed/adaptive ECE, AURC, Wilson and story bootstrap",
        })
        args += [
            "--tasks", "3,5", "--seeds", str(seeds), "--epochs", "3",
            "--lr", "0.05", "--feature-scale", "10.0",
            "--bootstrap-samples", str(bootstrap_samples),
            "--eval-split", eval_split,
        ]

    return args, config, config["dataset_fingerprints"]


def write_manifest(job: dict) -> None:
    directory = Path(job["directory"])
    payload = public_job(job)
    payload.update({
        "manifest_version": 1,
        "command": job.get("command", []),
        "code_revision": job.get("code_revision", "unknown"),
        "python_version": platform.python_version(),
        "dataset_fingerprints": job.get("dataset_fingerprints", {}),
        "result_files": sorted(
            path.relative_to(directory).as_posix()
            for path in directory.rglob("*")
            if path.is_file() and path.name not in {"metadata.json", "manifest.json"}
        ),
    })
    (directory / "metadata.json").write_text(json.dumps(payload, indent=2))
    (directory / "manifest.json").write_text(json.dumps(payload, indent=2))


def load_history() -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    for path in RUNS.glob("*/metadata.json"):
        job = json.loads(path.read_text(encoding="utf-8"))
        job.setdefault("config", {}).setdefault("dataset", {
            "id": "legacy", "name": "Legacy run", "directory": None,
        })
        JOBS[path.parent.name] = job


def public_job(job: dict) -> dict:
    return {key: value for key, value in job.items() if key not in {"process", "command"}}


def _run_summary(job: dict) -> dict:
    directory = Path(job["directory"])
    summaries = list(directory.glob("**/summary.json"))
    result = json.loads(summaries[0].read_text(encoding="utf-8")) if summaries else {}
    experiment = job.get("experiment")
    config = job.get("config", {})
    stages = result.get("stages", {})
    l1 = stages.get("L1", {})
    l2 = stages.get("L2", {})
    g2_summary = {}
    g3_summary = {}
    if experiment == "g1l":
        confirmed = l2.get("status") == "completed"
        accuracy = l2.get("mean_accuracy") if confirmed else l1.get("validation_after_training", {}).get("accuracy")
        split = "test" if confirmed else "validation"
        error = l2.get("standard_error") if confirmed else None
        stage = "L2 confirmation" if confirmed else "L1 pilot"
    elif experiment == "g3":
        task_rows = result.get("tasks", {})
        hierarchy_accuracies = []
        for task_key, task in task_rows.items():
            arms = task.get("arms", {})
            flat = arms.get("flat_abi", {})
            hierarchy = arms.get("hier_bu_td", {})
            effect = task.get("paired_effects", {}).get("accuracy", {})
            diagnostics = task.get("hierarchy_mechanism_audit", {}).get("per_seed", [])
            hierarchy_accuracies.append(hierarchy.get("mean_accuracy_over_seeds", 0.0))
            stability_values = [
                row.get("max_hierarchy_active_stability_ratio")
                for row in diagnostics
                if row.get("max_hierarchy_active_stability_ratio") is not None
            ]
            g3_summary[task_key] = {
                "flat_mean_accuracy": flat.get("mean_accuracy_over_seeds"),
                "hierarchy_mean_accuracy": hierarchy.get("mean_accuracy_over_seeds"),
                "paired_delta": effect.get("mean"),
                "paired_ci95": effect.get("ci95_two_sided_t"),
                "sign_test_p": effect.get("exact_two_sided_sign_test_p"),
                "holm_adjusted_p": effect.get("holm_adjusted_exact_sign_test_p_across_tasks"),
                "mean_context_activity": (
                    sum(row.get("bottom_up_context_activity", 0.0) for row in diagnostics) / len(diagnostics)
                    if diagnostics else None
                ),
                "mean_top_down_gamma_fraction": (
                    sum(row.get("top_down_dynamic_gamma_fraction", 0.0) for row in diagnostics) / len(diagnostics)
                    if diagnostics else None
                ),
                "max_active_stability_ratio": max(stability_values) if stability_values else None,
                "active_stability_pass": all(row.get("all_active_stability_pass", False) for row in diagnostics),
            }
        accuracy = (
            sum(hierarchy_accuracies) / len(hierarchy_accuracies)
            if hierarchy_accuracies else None
        )
        split = config.get("evaluation_split", result.get("protocol", {}).get("evaluation_split", "test"))
        error = None
        stage = "full hierarchy benchmark"
    else:
        accuracy = result.get("mean_accuracy")
        split = config.get("evaluation_split", result.get("evaluation_split", "test"))
        error = result.get("standard_error", result.get("se_accuracy"))
        stage = None
    if experiment == "g2":
        for task_key, task in result.get("tasks", {}).items():
            difference = task.get("paired_accuracy_difference_ef0_minus_e0", {})
            g2_summary[task_key] = {
                "e0_mean_accuracy": task.get("arms", {}).get("e0", {}).get("mean_accuracy_over_seeds"),
                "ef0_mean_accuracy": task.get("arms", {}).get("ef0", {}).get("mean_accuracy_over_seeds"),
                "paired_delta": difference.get("mean"),
                "paired_ci95": difference.get("ci95_two_sided_t"),
                "sign_test_p": difference.get("exact_two_sided_sign_test_p"),
                "holm_adjusted_p": difference.get("holm_adjusted_exact_sign_test_p"),
            }
    return {
        "accuracy": accuracy,
        "uncertainty": error,
        "evaluation_split": split,
        "stage": stage,
        "g2_paired_results": g2_summary,
        "g3_paired_results": g3_summary,
        "dataset": config.get("dataset"),
        "task": config.get("task"),
        "tasks": config.get("tasks"),
        "seed_count": config.get("seeds"),
        "claim_boundary": result.get("claim_boundary"),
    }

def _matched_ef0_pair(left: dict, right: dict) -> tuple[bool, str]:
    experiments = {left.get("experiment"), right.get("experiment")}
    if experiments != {"e0", "ef0"}:
        return False, "Only E0 versus E-F0 is a paired-arm comparison."
    left_config = left.get("config", {})
    right_config = right.get("config", {})
    keys = (
        "task", "dim", "seeds", "epochs", "learning_rate", "feature_scale",
        "readout", "evaluation_split", "mode", "variant", "quick",
    )
    mismatches = [key for key in keys if left_config.get(key) != right_config.get(key)]
    if left_config.get("dataset", {}).get("id") != right_config.get("dataset", {}).get("id"):
        mismatches.append("dataset_profile")
    if left_config.get("dataset_fingerprints") != right_config.get("dataset_fingerprints"):
        mismatches.append("dataset_fingerprints")
    if mismatches:
        return False, "Protocol mismatch: " + ", ".join(mismatches)
    return True, "Matched E0/E-F0 configuration and dataset fingerprints."


def _paired_seed_comparison(left: dict, right: dict) -> dict:
    matched, reason = _matched_ef0_pair(left, right)
    if not matched:
        return {"comparable": False, "reason": reason, "seed_differences": []}
    if left.get("status") != "completed" or right.get("status") != "completed":
        return {
            "comparable": False,
            "reason": "Both runs must complete before comparing seed outcomes.",
            "seed_differences": [],
        }

    e0_job = left if left.get("experiment") == "e0" else right
    ef0_job = left if left.get("experiment") == "ef0" else right
    summaries = []
    for job in (e0_job, ef0_job):
        files = list(Path(job["directory"]).glob("**/summary.json"))
        if not files:
            return {
                "comparable": False,
                "reason": "A completed run is missing summary.json.",
                "seed_differences": [],
            }
        summaries.append(json.loads(files[0].read_text(encoding="utf-8")))
    by_seed = [
        {int(row["seed"]): row for row in summary.get("seed_results", [])}
        for summary in summaries
    ]
    if not by_seed[0] or set(by_seed[0]) != set(by_seed[1]):
        return {
            "comparable": False,
            "reason": "Completed runs do not have matching seed IDs.",
            "seed_differences": [],
        }

    differences = []
    for seed in sorted(by_seed[0]):
        e0 = by_seed[0][seed]
        ef0 = by_seed[1][seed]
        e0_predictions = e0.get("predictions", [])
        ef0_predictions = ef0.get("predictions", [])
        identity_fields = ("story_index", "task", "question", "raw_passage", "target")
        identities_match = (
            bool(e0_predictions)
            and len(e0_predictions) == e0.get("total")
            and len(ef0_predictions) == ef0.get("total")
            and len(e0_predictions) == len(ef0_predictions)
            and all(
                all(a.get(field) == b.get(field) for field in identity_fields)
                for a, b in zip(e0_predictions, ef0_predictions)
            )
        )
        if not identities_match:
            return {
                "comparable": False,
                "reason": f"Evaluation examples differ for matched seed {seed}.",
                "seed_differences": [],
            }
        differences.append({
            "seed": seed,
            "e0_accuracy": e0["accuracy"],
            "ef0_accuracy": ef0["accuracy"],
            "delta": ef0["accuracy"] - e0["accuracy"],
        })
    return {
        "comparable": True,
        "reason": "Matched seeds and ordered evaluation-example identities.",
        "seed_differences": differences,
        "mean_delta": sum(item["delta"] for item in differences) / len(differences),
        "interpretation": "Descriptive paired seed outcomes; no cross-run significance test is computed.",
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def send(self, status: int, obj, content_type: str = "application/json"):
        body = (
            obj if isinstance(obj, bytes)
            else json.dumps(obj).encode() if content_type == "application/json"
            else str(obj).encode()
        )
        self.send_response(status)
        self.send_header("Content-Type", content_type + "; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def body(self) -> dict:
        size = int(self.headers.get("Content-Length", "0"))
        if size > 16384:
            raise ValueError("request too large")
        value = json.loads(self.rfile.read(size) or b"{}")
        if not isinstance(value, dict):
            raise ValueError("request body must be a JSON object")
        return value

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            page = Path(__file__).with_name("index.html")
            return self.send(200, page.read_bytes(), "text/html")
        if path == "/api/experiments":
            return self.send(200, {"experiments": EXPERIMENTS})
        if path == "/api/datasets":
            return self.send(200, {"profiles": [profile_status(item) for item in dataset_profiles().values()]})
        if path == "/api/runs":
            runs = sorted(JOBS.values(), key=lambda job: job["created_at"], reverse=True)
            return self.send(200, {"runs": [public_job(job) for job in runs]})
        if path == "/api/compare":
            ids = [item for item in parse_qs(urlparse(self.path).query).get("ids", [""])[0].split(",") if item]
            if len(ids) < 2 or len(ids) > 5:
                return self.send(400, {"error": "select between two and five runs to compare"})
            if len(set(ids)) != len(ids):
                return self.send(400, {"error": "select each run only once"})
            selected = []
            for run_id in ids:
                job = JOBS.get(run_id)
                if not job:
                    return self.send(404, {"error": f"run not found: {run_id}"})
                selected.append(job)
            pairwise = []
            for index, left in enumerate(selected):
                for right in selected[index + 1:]:
                    matched, reason = _matched_ef0_pair(left, right)
                    pairwise.append({
                        "left_run_id": left["run_id"],
                        "right_run_id": right["run_id"],
                        "matched_ef0_pair": matched,
                        "interpretation": reason,
                        "paired_seed_comparison": _paired_seed_comparison(left, right),
                    })
            rows = []
            for job in selected:
                row = _run_summary(job)
                config = job.get("config", {})
                row.update({
                    "run_id": job["run_id"],
                    "experiment": job.get("experiment"),
                    "status": job.get("status"),
                    "created_at": job.get("created_at"),
                    "mode": config.get("mode"),
                    "dimension": config.get("dim"),
                    "seed_count": config.get("seeds"),
                    "claim_type": (
                        "deterministic diagnostic" if job.get("experiment") == "g1"
                        else "validation pilot / gated confirmation" if job.get("experiment") == "g1l"
                        else "paired benchmark with internal E0/E-F0 comparison" if job.get("experiment") == "g2"
                        else "paired flat ABI versus full hierarchy benchmark" if job.get("experiment") == "g3"
                        else "learned benchmark arm"
                    ),
                })
                rows.append(row)
            return self.send(200, {"runs": rows, "pairwise": pairwise})
        file_match = re.fullmatch(
            r"/api/runs/([a-f0-9-]+)/files/(summary\.json|summary\.md|per_example\.csv|"
            r"per_example\.jsonl|per_seed_metrics\.csv|fixtures\.jsonl|manifest\.json|"
            r"config\.json|experiment_lab_register\.json)",
            path,
        )
        if file_match:
            run_id, filename = file_match.groups()
            job = JOBS.get(run_id)
            if not job:
                return self.send(404, {"error": "run not found"})
            matches = [item for item in Path(job["directory"]).glob(f"**/{filename}") if item.is_file()]
            if not matches:
                return self.send(404, {"error": "result file not found"})
            types = {
                "summary.json": "application/json", "summary.md": "text/markdown",
                "per_example.csv": "text/csv", "per_example.jsonl": "application/x-ndjson",
                "per_seed_metrics.csv": "text/csv", "fixtures.jsonl": "application/x-ndjson",
                "manifest.json": "application/json", "config.json": "application/json",
                "experiment_lab_register.json": "application/json",
            }
            return self.send(200, matches[0].read_bytes(), types[filename])
        match = re.fullmatch(r"/api/runs/([a-f0-9-]+)(?:/(results|logs))?", path)
        if match:
            job = JOBS.get(match.group(1))
            if not job:
                return self.send(404, {"error": "run not found"})
            if match.group(2) == "logs":
                directory = Path(job["directory"])
                stdout = directory / "stdout.log"
                stderr = directory / "stderr.log"
                return self.send(200, {
                    "stdout": stdout.read_text(errors="replace")[-12000:] if stdout.exists() else "",
                    "stderr": stderr.read_text(errors="replace")[-12000:] if stderr.exists() else "",
                })
            if match.group(2) == "results":
                summaries = list(Path(job["directory"]).glob("**/summary.json"))
                return self.send(
                    200,
                    json.loads(summaries[0].read_text(encoding="utf-8")) if summaries else {"pending": True},
                )
            return self.send(200, public_job(job))
        return self.send(404, {"error": "not found"})

    def do_DELETE(self):
        run_id = urlparse(self.path).path.removeprefix("/api/runs/")
        if not re.fullmatch(r"[a-f0-9-]{36}", run_id):
            return self.send(400, {"error": "invalid run id"})
        with LOCK:
            job = JOBS.get(run_id)
            if not job:
                return self.send(404, {"error": "run not found"})
            process = job.get("process")
            if job.get("status") == "running" or (process is not None and process.poll() is None):
                return self.send(409, {"error": "a running experiment cannot be deleted"})
            run_directory = Path(job["directory"]).resolve()
            runs_directory = RUNS.resolve()
            if run_directory.parent != runs_directory or run_directory.name != run_id:
                return self.send(400, {"error": "run directory is outside the Lab run store"})
            if not run_directory.is_dir():
                return self.send(404, {"error": "run directory not found"})
            try:
                shutil.rmtree(run_directory)
            except OSError as exc:
                return self.send(500, {"error": f"could not delete run directory: {exc}"})
            del JOBS[run_id]
        return self.send(200, {"deleted": run_id})

    def do_POST(self):
        global ACTIVE
        path = urlparse(self.path).path
        try:
            data = self.body()
        except (ValueError, json.JSONDecodeError) as exc:
            return self.send(400, {"error": str(exc)})
        if path == "/api/runs":
            with LOCK:
                if ACTIVE and ACTIVE.poll() is None:
                    return self.send(409, {"error": "another experiment is running"})
                run_id = str(uuid.uuid4())
                directory = RUNS / run_id
                try:
                    args, config, fingerprints = build_launch(data, directory)
                except (KeyError, TypeError, ValueError, OSError) as exc:
                    return self.send(400, {"error": str(exc)})
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
                job = {
                    "run_id": run_id,
                    "experiment": config["experiment"],
                    "config": config,
                    "status": "running",
                    "created_at": utc(),
                    "started_at": utc(),
                    "directory": str(directory),
                    "command": args,
                    "code_revision": code_revision(),
                    "dataset_fingerprints": fingerprints,
                }
                write_manifest(job)
                stdout = (directory / "stdout.log").open("w", encoding="utf-8")
                stderr = (directory / "stderr.log").open("w", encoding="utf-8")
                env = os.environ.copy()
                env["PYTHONPATH"] = str(ROOT / "code/python")
                library_name = "libgfe_core.dll" if os.name == "nt" else "libgfe_core.so"
                env["GFE_CORE_LIB"] = str(ROOT / "code/c++ core/build-shared" / library_name)
                try:
                    process = subprocess.Popen(
                        args, cwd=ROOT, env=env, stdout=stdout, stderr=stderr
                    )
                except OSError as exc:
                    stdout.close()
                    stderr.close()
                    directory_error = directory / "stderr.log"
                    directory_error.write_text(str(exc), encoding="utf-8")
                    return self.send(500, {"error": f"could not start experiment: {exc}"})
                ACTIVE = process
                job["process"] = process
                job["pid"] = process.pid
                JOBS[run_id] = job

                def watch():
                    global ACTIVE
                    code = process.wait()
                    stdout.close()
                    stderr.close()
                    job["status"] = "completed" if code == 0 else "failed"
                    job["exit_code"] = code
                    job["completed_at"] = utc()
                    job.pop("process", None)
                    write_manifest(job)
                    with LOCK:
                        if ACTIVE is process:
                            ACTIVE = None

                threading.Thread(target=watch, daemon=True).start()
                return self.send(202, public_job(job))
        if path.startswith("/api/runs/") and path.endswith("/cancel"):
            run_id = path.split("/")[-2]
            job = JOBS.get(run_id)
            process = job.get("process") if job else None
            if not process or process.poll() is not None:
                return self.send(409, {"error": "run is not active"})
            process.terminate()
            return self.send(202, {"status": "cancellation_requested"})
        return self.send(404, {"error": "not found"})


def main():
    load_history()
    port = int(os.environ.get("D2C_LAB_PORT", "8765"))
    print(f"Experiment Lab: http://127.0.0.1:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
