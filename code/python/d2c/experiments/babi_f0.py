"""bAbI question answering through the F0 stateful digital runtime.

The original E0 runner remains available in experiments.babi_qa. Isolated
mode resets episodic memory per story while preserving runtime phase/counters.
Continuous mode carries episodic state between stories and is a stress test,
not the official independent-story benchmark.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from ..digital.babi import (
    BabiDataset,
    babi_story_to_episode,
    build_babi_answer_set,
    build_babi_bridge,
    build_babi_memory_config,
    build_babi_vocabulary,
)
from ..digital.readout import (
    LinearReadout,
    QueryConditionedMLPReadout,
    query_conditioned_features,
)
from ..runtime.digital_runtime import DigitalRuntime, DigitalRuntimeConfig
from ..runtime.phase_controller import DigitalPhaseConfig


_REPO_ROOT = Path(__file__).resolve().parents[4]
_DATA_DIR = _REPO_ROOT / "code" / "python" / "d2c" / "progress" / "babi_data"
_PROGRESS_DIR = _REPO_ROOT / "code" / "python" / "d2c" / "progress" / "babi_f0"


@dataclass(slots=True)
class BabiF0Config:
    task: int = 1
    variant: str = "full"
    dim: int = 64
    seeds: int = 3
    epochs: int = 5
    hidden_dim: int = 32
    learning_rate: float = 0.005
    readout_type: Literal["linear", "mlp"] = "linear"
    mode: Literal["isolated", "continuous"] = "isolated"
    silence_steps_between_statements: int = 0
    runtime: DigitalRuntimeConfig = field(default_factory=DigitalRuntimeConfig)

    def __post_init__(self) -> None:
        if self.task <= 0 or self.dim <= 0 or self.seeds <= 0 or self.epochs <= 0:
            raise ValueError("task, dim, seeds, and epochs must be positive")
        if self.readout_type not in {"linear", "mlp"}:
            raise ValueError("readout_type must be linear or mlp")
        if self.mode not in {"isolated", "continuous"}:
            raise ValueError("mode must be isolated or continuous")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.silence_steps_between_statements < 0:
            raise ValueError("silence_steps_between_statements must be non-negative")


def _make_runtime(config: BabiF0Config, memory_config, *, session_id: str) -> DigitalRuntime:
    return DigitalRuntime(
        memory_config=memory_config,
        state_dim=config.dim,
        config=config.runtime,
        session_id=session_id,
    )


def run_babi_f0_experiment(
    config: BabiF0Config,
    *,
    data_dir: Path = _DATA_DIR,
    quick: bool = False,
    max_train_stories: int = 0,
) -> dict[str, object]:
    """Run one bAbI task through F0 and return reproducible metrics."""
    train_path = data_dir / "babi_train.jsonl"
    test_path = data_dir / "babi_test.jsonl"
    train_dataset = BabiDataset.from_jsonl(train_path).filter_task(config.task)
    test_dataset = BabiDataset.from_jsonl(test_path).filter_task(config.task)
    if not train_dataset.stories or not test_dataset.stories:
        raise ValueError(f"bAbI task {config.task} requires non-empty train and test splits")

    if quick:
        train_dataset.stories = train_dataset.stories[:20]
        test_dataset.stories = test_dataset.stories[:20]
    elif max_train_stories > 0:
        train_dataset.stories = train_dataset.stories[:max_train_stories]

    vocabulary = build_babi_vocabulary(train_dataset.stories + test_dataset.stories)
    answer_set = build_babi_answer_set(train_dataset.stories)
    if len(answer_set) < 2:
        raise ValueError("the selected task must have at least two training answer classes")
    bridge = build_babi_bridge(vocabulary, dim=config.dim)
    memory_config = build_babi_memory_config(variant=config.variant)
    epochs = 1 if quick else config.epochs

    seed_results: list[dict[str, object]] = []
    for seed in range(config.seeds):
        rng = random.Random(seed)
        feature_dim = config.dim * 2 + config.dim * config.dim
        if config.readout_type == "linear":
            readout = LinearReadout(
                input_dim=feature_dim,
                output_dim=len(answer_set),
                seed=seed,
            )
        else:
            readout = QueryConditionedMLPReadout(
                input_dim=feature_dim,
                output_dim=len(answer_set),
                hidden_dim=config.hidden_dim,
                seed=seed,
            )

        train_runtime = _make_runtime(
            config,
            memory_config,
            session_id=f"babi-train-task{config.task}-seed{seed}",
        )
        for epoch in range(epochs):
            order = list(enumerate(train_dataset.stories))
            rng.shuffle(order)
            for story_index, story in order:
                episode = babi_story_to_episode(
                    story,
                    bridge,
                    answer_set,
                    silence_steps=config.silence_steps_between_statements,
                )
                result = train_runtime.run_episode(
                    episode,
                    episode_id=f"train-e{epoch}-s{story_index}",
                    reset_memory=(config.mode == "isolated"),
                )
                features = query_conditioned_features(
                    result.final_state,
                    train_runtime.memory_config,
                    result.query,
                )
                if result.target is None:
                    raise ValueError("bAbI training story is missing an answer target")
                target_index = answer_set.index(result.target)
                readout.train_step(
                    features,
                    target_index,
                    learning_rate=config.learning_rate,
                )

        # Held-out evaluation starts with a fresh runtime. In continuous mode,
        # state carries across test stories, but never from train into test.
        test_runtime = _make_runtime(
            config,
            memory_config,
            session_id=f"babi-test-task{config.task}-seed{seed}",
        )
        correct = 0
        predictions: list[dict[str, object]] = []
        for story_index, story in enumerate(test_dataset.stories):
            episode = babi_story_to_episode(
                story,
                bridge,
                answer_set,
                silence_steps=config.silence_steps_between_statements,
            )
            result = test_runtime.run_episode(
                episode,
                episode_id=f"test-s{story_index}",
                reset_memory=(config.mode == "isolated"),
            )
            features = query_conditioned_features(
                result.final_state,
                test_runtime.memory_config,
                result.query,
            )
            prediction_index, confidence = readout.predict(features)
            predicted_answer = answer_set[prediction_index]
            is_correct = predicted_answer == result.target
            correct += int(is_correct)
            predictions.append({
                "episode_id": result.episode_id,
                "target": result.target,
                "prediction": predicted_answer,
                "confidence": confidence,
                "correct": is_correct,
                "runtime_steps": result.last_step - result.first_step + 1,
            })

        total = len(test_dataset.stories)
        seed_results.append({
            "seed": seed,
            "correct": correct,
            "total": total,
            "accuracy": correct / total if total else 0.0,
            "train_runtime": train_runtime.runtime_summary(),
            "test_runtime": test_runtime.runtime_summary(),
            "predictions": predictions,
        })

    accuracies = [float(row["accuracy"]) for row in seed_results]
    mean_accuracy = sum(accuracies) / len(accuracies)
    standard_error = (
        math.sqrt(sum((value - mean_accuracy) ** 2 for value in accuracies) / len(accuracies))
        / math.sqrt(len(accuracies))
        if len(accuracies) > 1 else 0.0
    )
    return {
        "experiment": "babi_f0_runtime_integrated",
        "task": config.task,
        "variant": config.variant,
        "dim": config.dim,
        "readout_type": config.readout_type,
        "mode": config.mode,
        "seeds": config.seeds,
        "epochs": epochs,
        "train_stories": len(train_dataset.stories),
        "test_stories": len(test_dataset.stories),
        "vocabulary_size": vocabulary.size,
        "answer_count": len(answer_set),
        "mean_accuracy": mean_accuracy,
        "standard_error": standard_error,
        "runtime_config": config.runtime.to_mapping(),
        "memory_config": memory_config.to_mapping(),
        "seed_results": seed_results,
        "notes": [
            "The SOE kernel is frozen in F0; F0 coordinates phases and records prediction/stability diagnostics.",
            "Readout learning remains supervised to preserve the current E0 comparison.",
            "Use isolated mode for independent bAbI accuracy; continuous mode is a separate cross-story interference stress test.",
        ],
    }


def save_babi_f0_bundle(
    result: dict[str, object],
    *,
    output_dir: Path = _PROGRESS_DIR,
) -> dict[str, str]:
    run_id = f"babi_f0_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    bundle_dir = output_dir / run_id
    bundle_dir.mkdir(parents=True, exist_ok=True)
    summary_path = bundle_dir / "summary.json"
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return {"bundle_dir": str(bundle_dir), "summary_json": str(summary_path)}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run bAbI question answering through the F0 digital runtime."
    )
    parser.add_argument("--task", type=int, default=1)
    parser.add_argument("--variant", choices=["full", "no_slow", "collapsed_gamma"], default="full")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--readout", choices=["linear", "mlp"], default="linear")
    parser.add_argument("--mode", choices=["isolated", "continuous"], default="isolated")
    parser.add_argument("--silence-steps", type=int, default=0)
    parser.add_argument("--max-train-stories", type=int, default=0)
    parser.add_argument("--data-dir", type=Path, default=_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=_PROGRESS_DIR)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--warmup-steps", type=int, default=8)
    parser.add_argument("--explore-steps", type=int, default=64)
    parser.add_argument("--consolidate-steps", type=int, default=8)
    parser.add_argument("--evaluate-every", type=int, default=128)
    parser.add_argument("--trace-capacity", type=int, default=20_000)
    return parser


def main(argv: list[str] | None = None) -> dict[str, str]:
    args = build_arg_parser().parse_args(argv)
    runtime_config = DigitalRuntimeConfig(
        trace_capacity=args.trace_capacity,
        phase=DigitalPhaseConfig(
            warmup_steps=args.warmup_steps,
            explore_steps=args.explore_steps,
            consolidate_steps=args.consolidate_steps,
            evaluation_interval=args.evaluate_every,
        ),
    )
    config = BabiF0Config(
        task=args.task,
        variant=args.variant,
        dim=args.dim,
        seeds=args.seeds,
        epochs=args.epochs,
        learning_rate=args.lr,
        readout_type=args.readout,
        mode=args.mode,
        silence_steps_between_statements=args.silence_steps,
        runtime=runtime_config,
    )
    result = run_babi_f0_experiment(
        config,
        data_dir=args.data_dir,
        quick=args.quick,
        max_train_stories=args.max_train_stories,
    )
    paths = save_babi_f0_bundle(result, output_dir=args.output_dir)
    print(
        f"task={result['task']} variant={result['variant']} dim={result['dim']} "
        f"mode={result['mode']}"
    )
    print(f"accuracy={result['mean_accuracy']:.3f} +/- {result['standard_error']:.3f}")
    for row in result["seed_results"]:
        print(
            f"seed={row['seed']} accuracy={row['accuracy']:.3f} "
            f"runtime_steps={row['test_runtime']['step_count']} "
            f"stability_violations={row['test_runtime']['stability_violations']}"
        )
    print(f"bundle={paths['bundle_dir']}")
    return paths


if __name__ == "__main__":
    main()
