"""Milestone E0: bAbI supervised QA experiment.

Usage (quick smoke):
    PYTHONPATH=code/python python3 -m d2c.experiments.babi_qa \
        --task 1 --variant full --dim 64 --seeds 3 --quick

Usage (full E0 claim run):
    PYTHONPATH=code/python python3 -m d2c.experiments.babi_qa \
        --task 1 --variant full --dim 64 --seeds 20
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..digital.babi import (
    BabiDataset,
    babi_story_to_episode,
    build_babi_answer_set,
    build_babi_bridge,
    build_babi_memory_config,
    build_babi_vocabulary,
)
from ..digital.bridge import seed_memory_state, step_memory
from ..digital.readout import (
    LinearReadout,
    QueryConditionedMLPReadout,
    query_conditioned_features,
)
from .g0_diagnostics import (
    build_prediction_record,
    summarize_predictions,
    summarize_training_losses,
)

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DATA_DIR = _REPO_ROOT / "code" / "python" / "d2c" / "progress" / "babi_data"
_PROGRESS_DIR = _REPO_ROOT / "code" / "python" / "d2c" / "progress" / "babi_qa"


@dataclass(slots=True)
class BabiQAConfig:
    task: int = 1
    variant: str = "full"
    dim: int = 64
    seeds: int = 3
    hidden_dim: int = 32
    learning_rate: float = 0.05
    readout_type: str = "linear"
    feature_scale: float = 100.0
    epochs: int = 5


def run_babi_qa_experiment(
    config: BabiQAConfig,
    *,
    data_dir: Path = _DATA_DIR,
    quick: bool = False,
    max_stories: int = 0,
    eval_split: str = "test",
) -> dict:
    """Run one bAbI QA experiment cell.  Returns summary dict."""
    train_path = data_dir / "babi_train.jsonl"
    if eval_split not in {"valid", "test"}:
        raise ValueError("eval_split must be 'valid' or 'test'")
    eval_path = data_dir / f"babi_{eval_split}.jsonl"
    train_ds = BabiDataset.from_jsonl(train_path).filter_task(config.task)
    eval_ds = BabiDataset.from_jsonl(eval_path).filter_task(config.task)

    if quick:
        train_ds.stories = train_ds.stories[:20]
        eval_ds.stories = eval_ds.stories[:20]
    elif max_stories > 0:
        train_ds.stories = train_ds.stories[:max_stories]

    vocab = build_babi_vocabulary(train_ds.stories + eval_ds.stories)
    answer_set = build_babi_answer_set(train_ds.stories)
    bridge = build_babi_bridge(vocab, dim=config.dim)
    memory_cfg = build_babi_memory_config(variant=config.variant)
    epochs = 2 if quick else config.epochs

    seed_results: list[dict] = []

    for seed in range(config.seeds):
        rng = random.Random(seed)
        feat_dim = config.dim + config.dim + config.dim * config.dim
        if config.readout_type == "linear":
            readout = LinearReadout(
                input_dim=feat_dim,
                output_dim=len(answer_set),
                seed=seed,
            )
        else:
            readout = QueryConditionedMLPReadout(
                input_dim=feat_dim,
                output_dim=len(answer_set),
                hidden_dim=config.hidden_dim,
                seed=seed,
            )

        # Training; retain epoch means so divergence is visible in the bundle.
        training_epoch_losses: list[float] = []
        for _ in range(epochs):
            stories = list(train_ds.stories)
            rng.shuffle(stories)
            epoch_losses: list[float] = []
            for story in stories:
                ep = babi_story_to_episode(story, bridge, answer_set)
                state = seed_memory_state(
                    state_dim=config.dim,
                    channel_count=memory_cfg.channel_count,
                )
                for forcing in ep.forcings:
                    state = step_memory(state, forcing, memory_cfg)
                features = query_conditioned_features(
                    state, memory_cfg, ep.query, feature_scale=config.feature_scale
                )
                target_idx = answer_set.index(ep.target)
                loss = readout.train_step(
                    features, target_idx, learning_rate=config.learning_rate
                )
                epoch_losses.append(float(loss))
            training_epoch_losses.append(
                sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
            )

        # Evaluation: preserve the complete evidence and answer distribution.
        correct = 0
        total = len(eval_ds.stories)
        predictions: list[dict] = []
        for story_index, story in enumerate(eval_ds.stories):
            ep = babi_story_to_episode(story, bridge, answer_set)
            state = seed_memory_state(
                state_dim=config.dim,
                channel_count=memory_cfg.channel_count,
            )
            for forcing in ep.forcings:
                state = step_memory(state, forcing, memory_cfg)
            features = query_conditioned_features(
                state, memory_cfg, ep.query, feature_scale=config.feature_scale
            )
            probabilities = readout.probabilities(features)
            record = build_prediction_record(
                story, story_index, answer_set, probabilities
            )
            predictions.append(record)
            correct += int(record["correct"])

        accuracy = correct / total if total > 0 else 0.0
        seed_results.append({
            "seed": seed, "correct": correct, "total": total,
            "accuracy": accuracy,
            "stability_ratio": memory_cfg.stability_ratio(),
            "deff": memory_cfg.deff(),
            "predictions": predictions,
            "diagnostics": summarize_predictions(predictions, answer_set),
            "training": summarize_training_losses(training_epoch_losses),
        })

    accuracies = [r["accuracy"] for r in seed_results]
    mean_acc = sum(accuracies) / len(accuracies) if accuracies else 0.0
    se = (
        math.sqrt(sum((a - mean_acc) ** 2 for a in accuracies) / len(accuracies))
        / math.sqrt(len(accuracies))
        if len(accuracies) > 1 else 0.0
    )

    all_predictions = [
        prediction for seed_result in seed_results
        for prediction in seed_result["predictions"]
    ]
    return {
        "experiment": "babi_e0_baseline",
        "task": config.task,
        "variant": config.variant,
        "dim": config.dim,
        "readout_type": config.readout_type,
        "feature_scale": config.feature_scale,
        "learning_rate": config.learning_rate,
        "epochs": epochs,
        "evaluation_split": eval_split,
        "seeds": config.seeds,
        "train_stories": len(train_ds.stories),
        "evaluation_stories": len(eval_ds.stories),
        "test_stories": len(eval_ds.stories) if eval_split == "test" else None,
        "vocab_size": vocab.size,
        "answer_count": len(answer_set),
        "mean_accuracy": mean_acc,
        "se_accuracy": se,
        "seed_results": seed_results,
        "diagnostics": summarize_predictions(all_predictions, answer_set),
        "memory_config": memory_cfg.to_mapping(),
    }


def save_babi_qa_bundle(result: dict, *, output_dir: Path = _PROGRESS_DIR) -> dict:
    run_id = f"babi_qa_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    bundle_dir = output_dir / run_id
    bundle_dir.mkdir(parents=True, exist_ok=True)
    summary_path = bundle_dir / "summary.json"
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return {"bundle_dir": str(bundle_dir), "summary_json": str(summary_path)}


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run E0 bAbI QA experiment.")
    p.add_argument("--task", type=int, default=1)
    p.add_argument("--variant", choices=["full", "no_slow", "collapsed_gamma"],
                    default="full")
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--feature-scale", type=float, default=100.0)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--eval-split", choices=["valid", "test"], default="test")
    p.add_argument("--data-dir", type=Path, default=_DATA_DIR)
    p.add_argument("--max-stories", type=int, default=0,
                    help="Cap training stories (0 = all)")
    p.add_argument("--readout", choices=["linear", "mlp"], default="linear",
                    help="Readout type (default: linear)")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--output-dir", type=Path, default=_PROGRESS_DIR)
    return p


def main(argv: list[str] | None = None) -> dict:
    args = build_arg_parser().parse_args(argv)
    config = BabiQAConfig(
        task=args.task, variant=args.variant, dim=args.dim,
        seeds=args.seeds, hidden_dim=args.hidden_dim,
        learning_rate=args.lr, readout_type=args.readout,
        feature_scale=args.feature_scale, epochs=args.epochs,
    )
    result = run_babi_qa_experiment(
        config, data_dir=args.data_dir, quick=args.quick, max_stories=args.max_stories,
        eval_split=args.eval_split,
    )
    paths = save_babi_qa_bundle(result, output_dir=args.output_dir)
    print(f"task={result['task']} variant={result['variant']} dim={result['dim']}")
    print(f"mean accuracy: {result['mean_accuracy']:.3f} +/- {result['se_accuracy']:.3f}")
    for sr in result["seed_results"]:
        print(f"  seed {sr['seed']}: {sr['correct']}/{sr['total']} = {sr['accuracy']:.3f}")
    print(f"bundle: {paths['bundle_dir']}")
    return paths


if __name__ == "__main__":
    main()