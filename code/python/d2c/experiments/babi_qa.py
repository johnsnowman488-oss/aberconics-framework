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
    QueryConditionedMLPReadout,
    query_conditioned_features,
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


def run_babi_qa_experiment(
    config: BabiQAConfig,
    *,
    data_dir: Path = _DATA_DIR,
    quick: bool = False,
    max_stories: int = 0,
) -> dict:
    """Run one bAbI QA experiment cell.  Returns summary dict."""
    train_path = data_dir / "babi_train.jsonl"
    test_path = data_dir / "babi_test.jsonl"
    train_ds = BabiDataset.from_jsonl(train_path).filter_task(config.task)
    test_ds = BabiDataset.from_jsonl(test_path).filter_task(config.task)

    if quick:
        train_ds.stories = train_ds.stories[:20]
        test_ds.stories = test_ds.stories[:20]
    elif max_stories > 0:
        train_ds.stories = train_ds.stories[:max_stories]

    vocab = build_babi_vocabulary(train_ds.stories + test_ds.stories)
    answer_set = build_babi_answer_set(train_ds.stories)
    bridge = build_babi_bridge(vocab, dim=config.dim)
    memory_cfg = build_babi_memory_config(variant=config.variant)
    epochs = 2 if quick else 5

    seed_results: list[dict] = []

    for seed in range(config.seeds):
        rng = random.Random(seed)
        feat_dim = config.dim + config.dim + config.dim * config.dim
        readout = QueryConditionedMLPReadout(
            input_dim=feat_dim,
            output_dim=len(answer_set),
            hidden_dim=config.hidden_dim,
            seed=seed,
        )

        # Training
        for _ in range(epochs):
            stories = list(train_ds.stories)
            rng.shuffle(stories)
            for story in stories:
                ep = babi_story_to_episode(story, bridge, answer_set)
                state = seed_memory_state(
                    state_dim=config.dim,
                    channel_count=memory_cfg.channel_count,
                )
                for forcing in ep.forcings:
                    state = step_memory(state, forcing, memory_cfg)
                features = query_conditioned_features(state, memory_cfg, ep.query)
                target_idx = answer_set.index(ep.target)
                readout.train_step(features, target_idx,
                                   learning_rate=config.learning_rate)

        # Evaluation
        correct = 0
        total = len(test_ds.stories)
        for story in test_ds.stories:
            ep = babi_story_to_episode(story, bridge, answer_set)
            state = seed_memory_state(
                state_dim=config.dim,
                channel_count=memory_cfg.channel_count,
            )
            for forcing in ep.forcings:
                state = step_memory(state, forcing, memory_cfg)
            features = query_conditioned_features(state, memory_cfg, ep.query)
            pred_idx, _ = readout.predict(features)
            if answer_set[pred_idx] == ep.target:
                correct += 1

        accuracy = correct / total if total > 0 else 0.0
        seed_results.append({
            "seed": seed, "correct": correct, "total": total,
            "accuracy": accuracy,
            "stability_ratio": memory_cfg.stability_ratio(),
            "deff": memory_cfg.deff(),
        })

    accuracies = [r["accuracy"] for r in seed_results]
    mean_acc = sum(accuracies) / len(accuracies) if accuracies else 0.0
    se = (
        math.sqrt(sum((a - mean_acc) ** 2 for a in accuracies) / len(accuracies))
        / math.sqrt(len(accuracies))
        if len(accuracies) > 1 else 0.0
    )

    return {
        "task": config.task,
        "variant": config.variant,
        "dim": config.dim,
        "seeds": config.seeds,
        "train_stories": len(train_ds.stories),
        "test_stories": len(test_ds.stories),
        "vocab_size": vocab.size,
        "answer_count": len(answer_set),
        "mean_accuracy": mean_acc,
        "se_accuracy": se,
        "seed_results": seed_results,
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
    p.add_argument("--max-stories", type=int, default=0,
                    help="Cap training stories (0 = all)")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--output-dir", type=Path, default=_PROGRESS_DIR)
    return p


def main(argv: list[str] | None = None) -> dict:
    args = build_arg_parser().parse_args(argv)
    config = BabiQAConfig(
        task=args.task, variant=args.variant, dim=args.dim,
        seeds=args.seeds, hidden_dim=args.hidden_dim,
        learning_rate=args.lr,
    )
    result = run_babi_qa_experiment(config, quick=args.quick,
                                     max_stories=args.max_stories)
    paths = save_babi_qa_bundle(result, output_dir=args.output_dir)
    print(f"task={result['task']} variant={result['variant']} dim={result['dim']}")
    print(f"mean accuracy: {result['mean_accuracy']:.3f} +/- {result['se_accuracy']:.3f}")
    for sr in result["seed_results"]:
        print(f"  seed {sr['seed']}: {sr['correct']}/{sr['total']} = {sr['accuracy']:.3f}")
    print(f"bundle: {paths['bundle_dir']}")
    return paths


if __name__ == "__main__":
    main()