"""Reproducible staged G0 readout sweep with validation-only selection.

Stage 1 compares scales. Stage 2 compares learning rates for the selected scale.
Stage 3 checks an MLP. The selected configuration is then confirmed once on the
held-out test split for E0 and E-F0; test results never select hyperparameters.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from .babi_qa import BabiQAConfig, run_babi_qa_experiment
from .babi_f0 import BabiF0Config, run_babi_f0_experiment


def _compact(result: dict) -> dict:
    rows = result.get("seed_results", [])
    diagnostics = [row.get("diagnostics", {}) for row in rows]
    eces = [item.get("calibration", {}).get("expected_calibration_error") for item in diagnostics]
    nlls = [item.get("calibration", {}).get("negative_log_likelihood") for item in diagnostics]
    return {
        "mean_accuracy": result.get("mean_accuracy"),
        "standard_error": result.get("standard_error", result.get("se_accuracy")),
        "per_seed_accuracy": [row.get("accuracy") for row in rows],
        "mean_ece": sum(x for x in eces if x is not None) / max(1, sum(x is not None for x in eces)),
        "mean_nll": sum(x for x in nlls if x is not None) / max(1, sum(x is not None for x in nlls)),
        "training": [row.get("training") for row in rows],
    }


def run_sweep(*, task: int = 2, dim: int = 64, seeds: int = 3, epochs: int = 5,
              scales=(1.0, 10.0, 100.0), learning_rates=(0.001, 0.01, 0.05),
              data_dir: Path | None = None, output_dir: Path | None = None) -> dict:
    if data_dir is None:
        data_dir = Path(__file__).resolve().parents[1] / "progress" / "babi_data"
    cells = []
    cache = {}

    def execute(stage: str, readout: str, scale: float, learning_rate: float) -> None:
        key = (readout, float(scale), float(learning_rate))
        if key in cache:
            return
        config = BabiQAConfig(
            task=task, dim=dim, seeds=seeds, epochs=epochs,
            learning_rate=learning_rate, readout_type=readout,
            feature_scale=scale,
        )
        result = run_babi_qa_experiment(config, data_dir=data_dir, eval_split="valid")
        row = {
            "stage": stage, "readout": readout, "feature_scale": scale,
            "learning_rate": learning_rate, "epochs": epochs,
            "validation": _compact(result),
        }
        cells.append(row)
        cache[key] = row

    for scale in scales:
        execute("stage_1_scale", "linear", float(scale), float(learning_rates[1]))
    best_scale_row = max(
        (row for row in cells if row["stage"] == "stage_1_scale"),
        key=lambda row: (
            row["validation"]["mean_accuracy"],
            -row["validation"]["mean_nll"],
            -row["validation"]["standard_error"],
        ),
    )
    selected_scale = float(best_scale_row["feature_scale"])
    for rate in learning_rates:
        execute("stage_2_learning_rate", "linear", selected_scale, float(rate))
    best_linear = max(
        (row for row in cells if row["readout"] == "linear"),
        key=lambda row: (
            row["validation"]["mean_accuracy"],
            -row["validation"]["mean_nll"],
            -row["validation"]["standard_error"],
        ),
    )
    selected_rate = float(best_linear["learning_rate"])
    execute("stage_3_nonlinear_readout", "mlp", selected_scale, selected_rate)
    best = max(
        cells,
        key=lambda row: (
            row["validation"]["mean_accuracy"],
            -row["validation"]["mean_nll"],
            -row["validation"]["standard_error"],
        ),
    )
    chosen = {
        "readout_type": best["readout"],
        "feature_scale": best["feature_scale"],
        "learning_rate": best["learning_rate"],
        "epochs": epochs,
        "selection_split": "valid",
        "selection_metric": "mean validation accuracy; ties broken by lower NLL then lower seed standard error",
    }
    e0_test = run_babi_qa_experiment(
        BabiQAConfig(
            task=task, dim=dim, seeds=seeds, epochs=epochs,
            learning_rate=chosen["learning_rate"],
            readout_type=chosen["readout_type"],
            feature_scale=chosen["feature_scale"],
        ),
        data_dir=data_dir, eval_split="test",
    )
    f0_test = run_babi_f0_experiment(
        BabiF0Config(
            task=task, dim=dim, seeds=seeds, epochs=epochs,
            learning_rate=chosen["learning_rate"],
            readout_type=chosen["readout_type"],
            feature_scale=chosen["feature_scale"],
            mode="isolated", exercise_spine=True,
        ),
        data_dir=data_dir, eval_split="test",
    )
    result = {
        "experiment": "g0_staged_readout_sweep",
        "protocol": {
            "task": task, "dim": dim, "seeds": seeds, "epochs": epochs,
            "validation_split": "valid", "test_split": "test",
            "sweep_stages": [
                "linear readout over feature scales",
                "linear readout over learning rates at selected scale",
                "nonlinear MLP at selected scale and learning rate",
            ],
        },
        "validation_cells": cells,
        "selected_configuration": chosen,
        "test_confirmation": {
            "e0": e0_test,
            "ef0": f0_test,
            "paired_mean_accuracy_difference_ef0_minus_e0": (
                f0_test["mean_accuracy"] - e0_test["mean_accuracy"]
            ),
        },
    }
    if output_dir is not None:
        run_id = "g0_readout_sweep_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        bundle_dir = output_dir / run_id
        bundle_dir.mkdir(parents=True, exist_ok=True)
        (bundle_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        result["bundle_dir"] = str(bundle_dir)
    return result


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Run G0 staged bAbI readout investigation.")
    parser.add_argument("--task", type=int, default=2)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--scales", default="1,10,100")
    parser.add_argument("--learning-rates", default="0.001,0.01,0.05")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "progress" / "babi_data")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/experiment-lab/g0-sweep"))
    args = parser.parse_args(argv)
    result = run_sweep(
        task=args.task, dim=args.dim, seeds=args.seeds, epochs=args.epochs,
        scales=tuple(float(x) for x in args.scales.split(",")),
        learning_rates=tuple(float(x) for x in args.learning_rates.split(",")),
        data_dir=args.data_dir, output_dir=args.output_dir,
    )
    print(json.dumps({
        "selected_configuration": result["selected_configuration"],
        "validation_cells": result["validation_cells"],
        "test_confirmation": {
            "e0": _compact(result["test_confirmation"]["e0"]),
            "ef0": _compact(result["test_confirmation"]["ef0"]),
            "paired_mean_accuracy_difference_ef0_minus_e0":
                result["test_confirmation"]["paired_mean_accuracy_difference_ef0_minus_e0"],
        },
        "bundle_dir": result.get("bundle_dir"),
    }, indent=2))
    return result


if __name__ == "__main__":
    main()
