"""Controlled, validation-only G2 reproducibility and confidence diagnostic.

Loads only bAbI train and validation splits. The original held-out test split is
never opened. Repeats feature construction and readout fitting to locate any
run-to-run divergence at the memory, feature, optimization, or prediction stage.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import platform
import random
import sys
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..digital.babi import (
    BabiDataset,
    babi_story_to_episode,
    build_babi_answer_set,
    build_babi_bridge,
    build_babi_memory_config,
    build_babi_vocabulary,
)
from ..digital.bridge import seed_memory_state, step_memory
from ..digital.readout import LinearReadout, query_conditioned_features

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DATA_DIR = _REPO_ROOT / "code" / "python" / "d2c" / "progress" / "babi_data"
SCALES = (10.0, 100.0)  # leading candidate + high-scale sensitivity reference


def _hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _hash_array(value: np.ndarray) -> str:
    arr = np.ascontiguousarray(value)
    header = json.dumps({"dtype": str(arr.dtype), "shape": list(arr.shape)}, sort_keys=True).encode()
    return _hash_bytes(header + b"\0" + arr.tobytes(order="C"))


def _hash_model(model: LinearReadout) -> str:
    return _hash_bytes(bytes.fromhex(_hash_array(model.W)) + bytes.fromhex(_hash_array(model.b)))


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _ece(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for index in range(bins):
        mask = (conf >= edges[index]) & (
            (conf <= edges[index + 1]) if index == bins - 1 else (conf < edges[index + 1])
        )
        if np.any(mask):
            total += float(np.mean(mask)) * abs(float(np.mean(conf[mask])) - float(np.mean(correct[mask])))
    return total


def _evaluate(model: LinearReadout, x: np.ndarray, y: np.ndarray, answers: list[str],
              stories: list, *, detailed: bool) -> tuple[dict, np.ndarray, np.ndarray]:
    logits = x @ model.W.T + model.b
    probabilities = _softmax(logits)
    predictions = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)
    correct = (predictions == y).astype(np.float64)
    true_probability = probabilities[np.arange(len(y)), y]
    high = confidence >= 0.90
    metrics = {
        "accuracy": float(np.mean(correct)),
        "nll": float(-np.mean(np.log(np.maximum(true_probability, 1e-12)))),
        "brier": float(np.mean(np.sum((probabilities - np.eye(probabilities.shape[1])[y]) ** 2, axis=1))),
        "ece_15": _ece(confidence, correct),
        "mean_confidence": float(np.mean(confidence)),
        "mean_confidence_on_errors": float(np.mean(confidence[correct == 0])) if np.any(correct == 0) else None,
        "error_confidence_p90": float(np.quantile(confidence[correct == 0], 0.90)) if np.any(correct == 0) else None,
        "high_confidence_coverage_90": float(np.mean(high)),
        "high_confidence_error_fraction_90": float(np.mean(correct[high] == 0)) if np.any(high) else None,
        "mean_logit_abs_max": float(np.mean(np.max(np.abs(logits), axis=1))),
        "p95_logit_abs_max": float(np.quantile(np.max(np.abs(logits), axis=1), 0.95)),
    }
    if detailed:
        confidence_bins = []
        cuts = (0.0, 0.50, 0.70, 0.80, 0.90, 0.95, 0.99, 1.0000001)
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            mask = (confidence >= lo) & (confidence < hi)
            if np.any(mask):
                confidence_bins.append({
                    "range": [lo, min(hi, 1.0)],
                    "count": int(np.sum(mask)),
                    "mean_confidence": float(np.mean(confidence[mask])),
                    "accuracy": float(np.mean(correct[mask])),
                    "error_fraction": float(np.mean(correct[mask] == 0)),
                })
        per_class = []
        for class_index, answer in enumerate(answers):
            mask = y == class_index
            if np.any(mask):
                per_class.append({
                    "answer": answer,
                    "support": int(np.sum(mask)),
                    "accuracy": float(np.mean(correct[mask])),
                    "mean_confidence": float(np.mean(confidence[mask])),
                    "errors": int(np.sum(correct[mask] == 0)),
                    "high_confidence_errors": int(np.sum((correct[mask] == 0) & (confidence[mask] >= 0.90))),
                })
        error_indices = np.flatnonzero(correct == 0)
        ranked = sorted(error_indices.tolist(), key=lambda i: float(confidence[i]), reverse=True)[:20]
        metrics["confidence_bins"] = confidence_bins
        metrics["per_answer_class"] = per_class
        metrics["top_confident_errors"] = [{
            "validation_index": int(i),
            "question": stories[i].question,
            "true_answer": answers[int(y[i])],
            "predicted_answer": answers[int(predictions[i])],
            "confidence": float(confidence[i]),
        } for i in ranked]
    return metrics, logits, predictions


def _prepare(task: int, data_dir: Path, dim: int) -> dict:
    train = BabiDataset.from_jsonl(data_dir / "babi_train.jsonl").filter_task(task)
    valid = BabiDataset.from_jsonl(data_dir / "babi_valid.jsonl").filter_task(task)
    answers = build_babi_answer_set(train.stories)
    vocab = build_babi_vocabulary(train.stories + valid.stories)
    bridge = build_babi_bridge(vocab, dim=dim)
    memory = build_babi_memory_config(variant="full")

    def encode(stories):
        stored_states = []
        targets = []
        for story in stories:
            episode = babi_story_to_episode(story, bridge, answers)
            state = seed_memory_state(state_dim=dim, channel_count=memory.channel_count)
            for forcing in episode.forcings:
                state = step_memory(state, forcing, memory)
            stored_states.append((state, episode.query))
            targets.append(answers.index(episode.target))
        state_rows = []
        query_rows = []
        for state, query in stored_states:
            state_rows.append(np.asarray(state.u + [v for channel in state.chi for v in channel], dtype=np.float64))
            query_rows.append(np.asarray(query, dtype=np.float64))
        state_matrix = np.vstack(state_rows)
        query_matrix = np.vstack(query_rows)
        features = {
            scale: np.asarray([
                query_conditioned_features(state, memory, query, feature_scale=scale)
                for state, query in stored_states
            ], dtype=np.float64)
            for scale in SCALES
        }
        fingerprints = {
            "memory_state_sha256": _hash_array(state_matrix),
            "query_sha256": _hash_array(query_matrix),
            "features_sha256": {str(scale): _hash_array(matrix) for scale, matrix in features.items()},
        }
        return features, np.asarray(targets, dtype=np.int64), fingerprints

    train_x, train_y, train_fp = encode(train.stories)
    valid_x, valid_y, valid_fp = encode(valid.stories)
    return {
        "train_x": train_x, "train_y": train_y, "valid_x": valid_x, "valid_y": valid_y,
        "answers": answers, "valid_stories": valid.stories,
        "train_count": len(train.stories), "valid_count": len(valid.stories),
        "train_fp": train_fp, "valid_fp": valid_fp,
    }


def _fit_once(x_train: np.ndarray, y_train: np.ndarray, x_valid: np.ndarray,
              y_valid: np.ndarray, answers: list[str], stories: list, seed: int,
              epochs: int, learning_rate: float) -> dict:
    model = LinearReadout(input_dim=x_train.shape[1], output_dim=len(answers), seed=seed)
    initial_hash = _hash_model(model)
    rng = random.Random(seed)
    history = []
    for epoch in range(1, epochs + 1):
        order = list(range(len(y_train)))
        rng.shuffle(order)
        training_order_sha256 = _hash_array(np.asarray(order, dtype=np.int64))
        losses = []
        for index in order:
            losses.append(float(model.train_step(x_train[index], int(y_train[index]), learning_rate=learning_rate)))
        metrics, logits, predictions = _evaluate(
            model, x_valid, y_valid, answers, stories, detailed=(epoch == epochs)
        )
        history.append({
            "epoch": epoch,
            "mean_train_loss": float(np.mean(losses)),
            "training_order_sha256": training_order_sha256,
            "weight_sha256": _hash_model(model),
            "logits_sha256": _hash_array(logits),
            "predictions_sha256": _hash_array(predictions.astype(np.int64)),
            "metrics": metrics,
        })
    return {
        "initial_weight_sha256": initial_hash,
        "history": history,
        "final_weights": model.W.copy(),
        "final_bias": model.b.copy(),
        "final_logits": logits.copy(),
        "final_predictions": predictions.copy(),
    }


def run(data_dir: Path, output_dir: Path, tasks=(3, 5), seeds=5, repeats=2,
        epochs=3, dim=64, learning_rate=0.05) -> dict:
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    train_path = data_dir / "babi_train.jsonl"
    valid_path = data_dir / "babi_valid.jsonl"
    if not train_path.is_file() or not valid_path.is_file():
        raise FileNotFoundError("train and validation splits are required")
    if repeats < 2 or seeds < 3 or epochs != 3:
        raise ValueError("diagnostic requires >=2 repeats, >=3 seeds, and exactly 3 epochs")

    prepared_first = {task: _prepare(task, data_dir, dim) for task in tasks}
    prepared_second = {task: _prepare(task, data_dir, dim) for task in tasks}
    tasks_out = {}
    seed_comparisons = []
    csv_rows = []

    for task in tasks:
        first = prepared_first[task]
        second = prepared_second[task]
        feature_comparison = {
            "train_memory_state_exact_match": first["train_fp"]["memory_state_sha256"] == second["train_fp"]["memory_state_sha256"],
            "valid_memory_state_exact_match": first["valid_fp"]["memory_state_sha256"] == second["valid_fp"]["memory_state_sha256"],
            "train_query_exact_match": first["train_fp"]["query_sha256"] == second["train_fp"]["query_sha256"],
            "valid_query_exact_match": first["valid_fp"]["query_sha256"] == second["valid_fp"]["query_sha256"],
            "train_features_exact_match_by_scale": {
                str(scale): first["train_fp"]["features_sha256"][str(scale)] == second["train_fp"]["features_sha256"][str(scale)]
                for scale in SCALES
            },
            "valid_features_exact_match_by_scale": {
                str(scale): first["valid_fp"]["features_sha256"][str(scale)] == second["valid_fp"]["features_sha256"][str(scale)]
                for scale in SCALES
            },
        }
        task_conditions = []
        for scale in SCALES:
            x_train = first["train_x"][scale]
            x_valid = first["valid_x"][scale]
            y_train, y_valid = first["train_y"], first["valid_y"]
            for seed in range(seeds):
                runs = []
                for repeat in range(repeats):
                    fit = _fit_once(
                        x_train, y_train, x_valid, y_valid, first["answers"],
                        first["valid_stories"], seed, epochs, learning_rate,
                    )
                    public_history = fit["history"]
                    runs.append({
                        "repeat": repeat + 1,
                        "initial_weight_sha256": fit["initial_weight_sha256"],
                        "epochs": public_history,
                        "_weights": fit["final_weights"],
                        "_bias": fit["final_bias"],
                        "_logits": fit["final_logits"],
                        "_predictions": fit["final_predictions"],
                    })
                    for row in public_history:
                        metrics = row["metrics"]
                        csv_rows.append({
                            "task": task, "feature_scale": scale, "seed": seed,
                            "repeat": repeat + 1, "epoch": row["epoch"],
                            "train_loss": row["mean_train_loss"],
                            "training_order_sha256": row["training_order_sha256"],
                            "validation_accuracy": metrics["accuracy"],
                            "validation_nll": metrics["nll"],
                            "validation_brier": metrics["brier"],
                            "validation_ece_15": metrics["ece_15"],
                            "mean_confidence": metrics["mean_confidence"],
                            "mean_confidence_on_errors": metrics["mean_confidence_on_errors"],
                            "high_confidence_coverage_90": metrics["high_confidence_coverage_90"],
                            "high_confidence_error_fraction_90": metrics["high_confidence_error_fraction_90"],
                            "p95_logit_abs_max": metrics["p95_logit_abs_max"],
                            "weight_sha256": row["weight_sha256"],
                            "logits_sha256": row["logits_sha256"],
                            "predictions_sha256": row["predictions_sha256"],
                        })
                a, b = runs[0], runs[1]
                epoch_exact = [
                    {
                        "epoch": ea["epoch"],
                        "training_order_exact_match": ea["training_order_sha256"] == eb["training_order_sha256"],
                        "weights_exact_match": ea["weight_sha256"] == eb["weight_sha256"],
                        "logits_exact_match": ea["logits_sha256"] == eb["logits_sha256"],
                        "predictions_exact_match": ea["predictions_sha256"] == eb["predictions_sha256"],
                    }
                    for ea, eb in zip(a["epochs"], b["epochs"])
                ]
                comparison = {
                    "task": task, "feature_scale": scale, "seed": seed,
                    "initial_weights_exact_match": a["initial_weight_sha256"] == b["initial_weight_sha256"],
                    "epoch_comparisons": epoch_exact,
                    "final_weight_max_abs_delta": float(np.max(np.abs(a["_weights"] - b["_weights"]))),
                    "final_bias_max_abs_delta": float(np.max(np.abs(a["_bias"] - b["_bias"]))),
                    "final_logit_max_abs_delta": float(np.max(np.abs(a["_logits"] - b["_logits"]))),
                    "final_prediction_disagreement_fraction": float(np.mean(a["_predictions"] != b["_predictions"])),
                }
                seed_comparisons.append(comparison)
                task_conditions.append({
                    "feature_scale": scale, "seed": seed,
                    "repeat_1": {"initial_weight_sha256": a["initial_weight_sha256"], "epochs": a["epochs"]},
                    "repeat_2": {"initial_weight_sha256": b["initial_weight_sha256"], "epochs": b["epochs"]},
                    "repeat_comparison": comparison,
                })
        tasks_out[str(task)] = {
            "task": task,
            "train_stories": first["train_count"],
            "validation_stories": first["valid_count"],
            "answer_count": len(first["answers"]),
            "answers": first["answers"],
            "feature_construction_fingerprints_repeat_1": {
                "train": first["train_fp"], "validation": first["valid_fp"]
            },
            "feature_construction_fingerprints_repeat_2": {
                "train": second["train_fp"], "validation": second["valid_fp"]
            },
            "feature_construction_comparison": feature_comparison,
            "conditions": task_conditions,
        }

    feature_checks = []
    for task_key, task_data in tasks_out.items():
        comp = task_data["feature_construction_comparison"]
        feature_checks.extend(v for k, v in comp.items() if k.endswith("exact_match") and isinstance(v, bool))
        for key in ("train_features_exact_match_by_scale", "valid_features_exact_match_by_scale"):
            feature_checks.extend(comp[key].values())
    all_feature_exact = all(feature_checks)
    all_initial_exact = all(row["initial_weights_exact_match"] for row in seed_comparisons)
    all_orders_exact = all(
        epoch["training_order_exact_match"]
        for row in seed_comparisons for epoch in row["epoch_comparisons"]
    )
    all_weights_exact = all(
        epoch["weights_exact_match"]
        for row in seed_comparisons for epoch in row["epoch_comparisons"]
    )
    all_logits_exact = all(
        epoch["logits_exact_match"]
        for row in seed_comparisons for epoch in row["epoch_comparisons"]
    )
    all_predictions_exact = all(
        epoch["predictions_exact_match"]
        for row in seed_comparisons for epoch in row["epoch_comparisons"]
    )
    env_capture = io.StringIO()
    with redirect_stdout(env_capture):
        try:
            np.show_config()
        except Exception as exc:
            print(f"numpy_config_capture_error={exc!r}")

    data_fingerprints = {
        "train_sha256": _hash_bytes(train_path.read_bytes()),
        "validation_sha256": _hash_bytes(valid_path.read_bytes()),
    }
    result = {
        "experiment": "g2_reproducibility_and_confidence_diagnostic",
        "experiment_id": "G2-REPRO-DIAGNOSTIC",
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_split": "validation",
        "held_out_test_split_opened": False,
        "claim_boundary": (
            "Controlled repeated feature construction and readout fitting using train/validation only. "
            "No held-out test claims; no production setting is confirmed."
        ),
        "config": {
            "tasks": list(tasks), "seeds": seeds, "repeats_per_seed": repeats,
            "epochs": epochs, "dim": dim, "learning_rate": learning_rate,
            "feature_scales": list(SCALES),
            "role_of_scales": {"10": "leading candidate", "100": "high-scale sensitivity reference"},
        },
        "dataset_fingerprints": data_fingerprints,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "thread_environment": {
                key: os.environ.get(key) for key in (
                    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
                    "OPENBLAS_CORETYPE", "NPY_DISABLE_CPU_FEATURES"
                )
            },
            "github_run_id": os.environ.get("GITHUB_RUN_ID"),
            "github_sha": os.environ.get("GITHUB_SHA"),
            "numpy_build_configuration": env_capture.getvalue(),
            "readout_path": "Python LinearReadout + NumPy; fixed Python-side bAbI memory bridge",
            "cxx_runtime_note": "CI builds the unchanged C++ runtime for regression context; this diagnostic's feature/readout path uses the Python bridge and does not invoke the C++ runtime.",
        },
        "tasks": tasks_out,
        "reproducibility_summary": {
            "feature_construction_all_exact": all_feature_exact,
            "initial_weights_all_exact": all_initial_exact,
            "training_orders_all_exact": all_orders_exact,
            "trained_weights_all_exact": all_weights_exact,
            "validation_logits_all_exact": all_logits_exact,
            "predictions_all_exact": all_predictions_exact,
            "classification": (
                "bitwise_repeatable_in_this_runner"
                if all([all_feature_exact, all_initial_exact, all_orders_exact, all_weights_exact, all_logits_exact, all_predictions_exact])
                else "divergence_detected"
            ),
            "seed_comparisons": seed_comparisons,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    fields = list(csv_rows[0].keys()) if csv_rows else []
    with (output_dir / "per_seed_repeat_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)
    lab = {
        "experiment_id": result["experiment_id"],
        "status": result["status"],
        "run_timestamp": result["created_at"],
        "evaluation_split": "validation",
        "held_out_test_split_opened": False,
        "config": result["config"],
        "dataset_fingerprints": result["dataset_fingerprints"],
        "reproducibility_summary": {
            key: value for key, value in result["reproducibility_summary"].items()
            if key != "seed_comparisons"
        },
        "confidence_diagnostics": {
            task: {
                condition["feature_scale"]: {
                    "seed": condition["seed"],
                    "repeat_1_epoch3": condition["repeat_1"]["epochs"][-1]["metrics"],
                    "repeat_2_epoch3": condition["repeat_2"]["epochs"][-1]["metrics"],
                }
                for condition in task_data["conditions"]
            }
            for task, task_data in tasks_out.items()
        },
        "artifacts": ["summary.json", "per_seed_repeat_metrics.csv", "experiment_lab_register.json"],
    }
    (output_dir / "experiment_lab_register.json").write_text(json.dumps(lab, indent=2), encoding="utf-8")
    return result


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", default="3,5")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    args = parser.parse_args()
    tasks = tuple(int(item) for item in args.tasks.split(","))
    if not tasks or any(task not in (3, 5) for task in tasks) or len(set(tasks)) != len(tasks):
        parser.error("--tasks must be 3, 5, or 3,5")
    result = run(args.data_dir, args.output_dir, tasks, args.seeds, args.repeats,
                 args.epochs, args.dim, args.learning_rate)
    print(json.dumps({
        "experiment_id": result["experiment_id"],
        "reproducibility_summary": {
            key: value for key, value in result["reproducibility_summary"].items()
            if key != "seed_comparisons"
        },
        "held_out_test_split_opened": result["held_out_test_split_opened"],
        "output": str(args.output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
