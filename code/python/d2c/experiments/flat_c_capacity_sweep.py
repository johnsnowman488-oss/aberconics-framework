"""Validation-only flat E-F0 capacity sweep over memory channels C and width D.

The existing G2/F0 flat runtime is retained. C is the primary factor; D is a
small secondary factor. Train and validation only are opened; test stays sealed.
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
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..digital.babi import (
    BabiDataset,
    babi_story_to_episode,
    build_babi_answer_set,
    build_babi_bridge,
    build_babi_vocabulary,
)
from ..digital.bridge import DigitalMemoryConfig
from ..digital.readout import LinearReadout, query_conditioned_features
from ..runtime.digital_runtime import DigitalRuntimeConfig
from .babi_f0 import BabiF0Config, _make_runtime
from .g0_diagnostics import (
    build_prediction_record,
    summarize_predictions,
    summarize_training_losses,
)

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DATA_DIR = _REPO_ROOT / "code" / "python" / "d2c" / "progress" / "babi_data"
DEFAULT_DIMS = (64, 96, 128)
DEFAULT_CHANNELS = (3, 4, 6)
BASE_KERNEL_MASS = 0.24
BASE_STABILITY_RATIO = 1.38


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_channel_config(channel_count: int) -> DigitalMemoryConfig:
    """Expand timescale coverage while preserving total |w| mass and stability ratio.

    C=3 is the exact established G2 flat kernel. For C=4/6, rates extend the
    spectrum down to 0.05 and fill intermediate log-timescales. The minimum-
    norm non-negative weights satisfy sum(w)=0.24 and sum(w/gamma)=1.38
    (leak_rate=1), so C is not confounded with aggregate kernel mass/gain.
    """
    rates_by_count = {
        3: [2.0, 0.5, 0.1],
        4: [2.0, 0.5, 0.1, 0.05],
        6: [2.0, 1.0, 0.5, 0.2, 0.1, 0.05],
    }
    if channel_count not in rates_by_count:
        raise ValueError("channel_count must be one of 3, 4, or 6")
    gamma = np.asarray(rates_by_count[channel_count], dtype=np.float64)
    if channel_count == 3:
        weights = np.asarray([0.04, 0.08, 0.12], dtype=np.float64)
    else:
        constraints = np.vstack([np.ones(channel_count), 1.0 / gamma])
        targets = np.asarray([BASE_KERNEL_MASS, BASE_STABILITY_RATIO])
        weights = constraints.T @ np.linalg.solve(constraints @ constraints.T, targets)
    config = DigitalMemoryConfig(
        gamma=gamma.tolist(), w=weights.tolist(), dt=0.05, leak_rate=1.0,
        form="input_driven",
    )
    if not math.isclose(sum(config.w), BASE_KERNEL_MASS, rel_tol=0.0, abs_tol=1e-10):
        raise AssertionError("channel expansion changed total kernel mass")
    if not math.isclose(config.stability_ratio(), BASE_STABILITY_RATIO, rel_tol=0.0, abs_tol=1e-10):
        raise AssertionError("channel expansion changed the aggregate stability ratio")
    if any(weight <= 0.0 for weight in config.w):
        raise AssertionError("channel expansion produced a non-positive channel weight")
    return config


def _run_cell(*, task: int, dim: int, channels: int, seeds: int, epochs: int,
              learning_rate: float, feature_scale: float, train_stories,
              valid_stories, answer_set: list[str], vocabulary, memory_config,
) -> dict:
    bridge = build_babi_bridge(vocabulary, dim=dim)
    feature_dim = dim * 2 + dim * dim
    seed_rows = []
    order_hashes = []
    for seed in range(seeds):
        rng = random.Random(seed)
        readout = LinearReadout(input_dim=feature_dim, output_dim=len(answer_set), seed=seed)
        cfg = BabiF0Config(
            task=task, variant="full", dim=dim, seeds=seeds, epochs=epochs,
            learning_rate=learning_rate, readout_type="linear",
            feature_scale=feature_scale, mode="isolated", memory_backend="python_ref",
            runtime=DigitalRuntimeConfig(), exercise_spine=False,
        )
        train_runtime = _make_runtime(
            cfg, memory_config, session_id=f"flat-c-sweep-train-t{task}-d{dim}-c{channels}-s{seed}",
        )
        epoch_losses = []
        training_orders = []
        for epoch in range(epochs):
            order = list(enumerate(train_stories))
            rng.shuffle(order)
            training_orders.append([int(index) for index, _ in order])
            losses = []
            for story_index, story in order:
                episode = babi_story_to_episode(story, bridge, answer_set)
                result = train_runtime.run_episode(
                    episode, episode_id=f"train-e{epoch}-s{story_index}", reset_memory=True,
                )
                features = query_conditioned_features(
                    result.final_state, memory_config, result.query,
                    feature_scale=feature_scale,
                )
                loss = readout.train_step(
                    features, answer_set.index(result.target), learning_rate=learning_rate,
                )
                losses.append(float(loss))
            epoch_losses.append(float(statistics.mean(losses)) if losses else 0.0)
        order_hash = hashlib.sha256(
            json.dumps(training_orders, separators=(",", ":")).encode()
        ).hexdigest()
        order_hashes.append(order_hash)

        eval_runtime = _make_runtime(
            cfg, memory_config, session_id=f"flat-c-sweep-valid-t{task}-d{dim}-c{channels}-s{seed}",
        )
        predictions = []
        for story_index, story in enumerate(valid_stories):
            episode = babi_story_to_episode(story, bridge, answer_set)
            result = eval_runtime.run_episode(
                episode, episode_id=f"valid-s{story_index}", reset_memory=True,
            )
            features = query_conditioned_features(
                result.final_state, memory_config, result.query,
                feature_scale=feature_scale,
            )
            probabilities = readout.probabilities(features)
            record = build_prediction_record(story, story_index, answer_set, probabilities)
            record["episode_id"] = result.episode_id
            record["runtime_steps"] = result.last_step - result.first_step + 1
            predictions.append(record)
        accuracy = sum(bool(row["correct"]) for row in predictions) / len(predictions)
        seed_rows.append({
            "seed": seed, "accuracy": accuracy, "correct": sum(bool(r["correct"]) for r in predictions),
            "total": len(predictions), "training_order_sha256": order_hash,
            "epoch_mean_train_losses": epoch_losses,
            "stability_ratio": memory_config.stability_ratio(),
            "effective_channel_count": memory_config.deff(),
            "train_runtime": train_runtime.runtime_summary(),
            "valid_runtime": eval_runtime.runtime_summary(),
            "predictions": predictions,
            "diagnostics": summarize_predictions(predictions, answer_set),
        })
    if len(set(order_hashes)) != 1:
        raise AssertionError("training order is not matched across capacity cells")
    accuracies = [row["accuracy"] for row in seed_rows]
    return {
        "task": task, "dim": dim, "channels": channels,
        "memory_config": memory_config.to_mapping(),
        "mean_validation_accuracy": statistics.mean(accuracies),
        "seed_accuracy_sd": statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0,
        "seed_rows": seed_rows,
        "training_order_sha256": order_hashes[0],
    }


def run_sweep(*, data_dir: Path, output_dir: Path, tasks=(3, 5), dims=DEFAULT_DIMS,
              channels=DEFAULT_CHANNELS, seeds=3, epochs=3, learning_rate=0.05,
              feature_scale=10.0) -> dict:
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    train_path, valid_path = data_dir / "babi_train.jsonl", data_dir / "babi_valid.jsonl"
    if not train_path.is_file() or not valid_path.is_file():
        raise FileNotFoundError("flat C sweep requires babi_train.jsonl and babi_valid.jsonl")
    if seeds < 3 or epochs != 3:
        raise ValueError("validation protocol requires at least 3 seeds and exactly 3 epochs")
    if not dims or any(d < 16 for d in dims) or len(set(dims)) != len(dims):
        raise ValueError("dimensions must be unique values >=16")
    if tuple(channels) != (3, 4, 6):
        raise ValueError("registered channel sweep is C=3,4,6")
    train_ds = BabiDataset.from_jsonl(train_path)
    valid_ds = BabiDataset.from_jsonl(valid_path)
    results = []
    for task in tasks:
        train_stories = train_ds.filter_task(task).stories
        valid_stories = valid_ds.filter_task(task).stories
        if not train_stories or not valid_stories:
            raise ValueError(f"task {task} has an empty train or validation split")
        answer_set = build_babi_answer_set(train_stories)
        for dim in dims:
            vocabulary = build_babi_vocabulary(train_stories + valid_stories)
            for count in channels:
                memory_config = build_channel_config(count)
                results.append(_run_cell(
                    task=task, dim=dim, channels=count, seeds=seeds, epochs=epochs,
                    learning_rate=learning_rate, feature_scale=feature_scale,
                    train_stories=train_stories, valid_stories=valid_stories,
                    answer_set=answer_set, vocabulary=vocabulary,
                    memory_config=memory_config,
                ))
    result = {
        "experiment": "flat_ef0_memory_channel_capacity_sweep",
        "experiment_id": "FLAT-C-CAPACITY-VALIDATION",
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "revision": os.environ.get("GITHUB_SHA", "unknown"),
        "protocol": {
            "tasks": list(tasks), "dims": list(dims), "channels": list(channels),
            "seeds": list(range(seeds)), "epochs": epochs, "learning_rate": learning_rate,
            "feature_scale": feature_scale, "readout": "linear",
            "memory_runtime": "existing G2 flat E-F0 DigitalRuntime, isolated episodes",
            "evaluation_split": "valid", "test_split_opened": False,
            "train_stories_per_task": len(train_ds.filter_task(tasks[0]).stories),
            "validation_stories_per_task": len(valid_ds.filter_task(tasks[0]).stories),
            "training_order_matched_across_capacity_cells": True,
            "kernel_invariants": {
                "total_positive_weight_mass": BASE_KERNEL_MASS,
                "stability_ratio": BASE_STABILITY_RATIO,
                "dt": 0.05, "leak_rate": 1.0, "form": "input_driven",
            },
            "claim_boundary": "Validation-only capacity screen. No test split opened; no production setting selected.",
        },
        "dataset_fingerprints": {
            "train_sha256": sha256(train_path), "valid_sha256": sha256(valid_path),
        },
        "cells": results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (output_dir / "per_seed_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["task", "dim", "channels", "seed", "accuracy", "correct", "total",
                  "stability_ratio", "effective_channel_count", "final_train_loss",
                  "training_order_sha256"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for cell in results:
            for row in cell["seed_rows"]:
                writer.writerow({
                    "task": cell["task"], "dim": cell["dim"], "channels": cell["channels"],
                    "seed": row["seed"], "accuracy": row["accuracy"], "correct": row["correct"],
                    "total": row["total"], "stability_ratio": row["stability_ratio"],
                    "effective_channel_count": row["effective_channel_count"],
                    "final_train_loss": row["epoch_mean_train_losses"][-1],
                    "training_order_sha256": row["training_order_sha256"],
                })
    with (output_dir / "per_example.jsonl").open("w", encoding="utf-8") as handle:
        for cell in results:
            for row in cell["seed_rows"]:
                for prediction in row["predictions"]:
                    record = dict(prediction)
                    record.update({"task": cell["task"], "dim": cell["dim"],
                                   "channels": cell["channels"], "seed": row["seed"]})
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    lines = [
        "# Flat E-F0 memory-channel capacity sweep", "",
        "Validation-only; train and validation splits only. The held-out test split was not opened.", "",
        f"Dims: {', '.join(map(str, dims))}; C: {', '.join(map(str, channels))}; "
        f"seeds: {seeds}; epochs: {epochs}; LR: {learning_rate:g}; feature scale: {feature_scale:g}.", "",
        "| Task | D | C | Mean validation accuracy | Seed SD | Stability ratio | Effective C |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in results:
        lines.append(
            f"| QA{cell['task']} | {cell['dim']} | {cell['channels']} | "
            f"{cell['mean_validation_accuracy']:.4f} | {cell['seed_accuracy_sd']:.4f} | "
            f"{cell['memory_config']['w'] and sum(w/(g*cell['memory_config']['leak_rate']) for w,g in zip(cell['memory_config']['w'],cell['memory_config']['gamma'])):.4f} | "
            f"{statistics.mean(r['effective_channel_count'] for r in cell['seed_rows']):.3f} |"
        )
    lines += ["", "## Interpretation boundary", "",
        "C is the primary factor; D is a secondary robustness factor. The C=3 kernel is the exact "
        "G2 flat baseline. C=4/6 extend timescale coverage while holding total positive kernel mass "
        "and the aggregate stability ratio fixed. The sweep is exploratory and validation-only; "
        "do not treat the best cell as a production choice or open the test split for tuning.", ""]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    register = {
        "experiment_id": result["experiment_id"], "status": result["status"],
        "revision": result["revision"], "created_at": result["created_at"],
        "protocol": result["protocol"], "dataset_fingerprints": result["dataset_fingerprints"],
        "cells": [{
            "task": c["task"], "dim": c["dim"], "channels": c["channels"],
            "mean_validation_accuracy": c["mean_validation_accuracy"],
            "seed_accuracy_sd": c["seed_accuracy_sd"],
            "memory_config": c["memory_config"],
        } for c in results],
        "artifacts": ["summary.json", "summary.md", "per_seed_metrics.csv", "per_example.jsonl",
                      "experiment_lab_register.json"],
    }
    (output_dir / "experiment_lab_register.json").write_text(
        json.dumps(register, indent=2), encoding="utf-8"
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", default="3,5")
    parser.add_argument("--dims", default="64,96,128")
    parser.add_argument("--channels", default="3,4,6")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--feature-scale", type=float, default=10.0)
    args = parser.parse_args(argv)
    result = run_sweep(
        data_dir=args.data_dir, output_dir=args.output_dir,
        tasks=tuple(int(x) for x in args.tasks.split(",")),
        dims=tuple(int(x) for x in args.dims.split(",")),
        channels=tuple(int(x) for x in args.channels.split(",")),
        seeds=args.seeds, epochs=args.epochs, learning_rate=args.learning_rate,
        feature_scale=args.feature_scale,
    )
    for cell in result["cells"]:
        print(f"QA{cell['task']} D={cell['dim']} C={cell['channels']} "
              f"valid_acc={cell['mean_validation_accuracy']:.4f}")
    print(f"validation-only evidence: {args.output_dir}")


if __name__ == "__main__":
    main()
