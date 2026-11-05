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
from .g0_diagnostics import (
    build_prediction_record,
    summarize_predictions,
    summarize_training_losses,
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
    learning_rate: float = 0.05
    readout_type: Literal["linear", "mlp"] = "linear"
    feature_scale: float = 100.0
    mode: Literal["isolated", "continuous"] = "isolated"
    silence_steps_between_statements: int = 0
    runtime: DigitalRuntimeConfig = field(default_factory=DigitalRuntimeConfig)
    exercise_spine: bool = False

    def __post_init__(self) -> None:
        if self.task <= 0 or self.dim <= 0 or self.seeds <= 0 or self.epochs <= 0:
            raise ValueError("task, dim, seeds, and epochs must be positive")
        if self.readout_type not in {"linear", "mlp"}:
            raise ValueError("readout_type must be linear or mlp")
        if self.mode not in {"isolated", "continuous"}:
            raise ValueError("mode must be isolated or continuous")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if not math.isfinite(self.feature_scale) or self.feature_scale <= 0.0:
            raise ValueError("feature_scale must be finite and positive")
        if self.silence_steps_between_statements < 0:
            raise ValueError("silence_steps_between_statements must be non-negative")


def _make_runtime(config: BabiF0Config, memory_config, *, session_id: str) -> DigitalRuntime:
    return DigitalRuntime(
        memory_config=memory_config,
        state_dim=config.dim,
        config=config.runtime,
        session_id=session_id,
    )


def exercise_f0_spine(
    config: BabiF0Config,
    memory_config,
    *,
    seed: int = 0,
) -> dict[str, object]:
    """Exercise every F0 runtime component without touching benchmark readouts."""
    runtime = _make_runtime(
        config, memory_config, session_id=f"f0-spine-audit-seed{seed}"
    )
    kernel_before = runtime.memory_config.to_mapping()
    audit_step = 0

    def advance() -> str:
        nonlocal audit_step
        forcing = [0.0] * config.dim
        forcing[(audit_step * 7 + seed) % config.dim] = 0.25
        record = runtime.step(
            forcing,
            token=f"spine-audit-{audit_step}",
            episode_id=f"spine-audit-seed{seed}",
            metadata={"audit": "f0_spine"},
        )
        audit_step += 1
        return record.phase

    # Exercise the configured WARMUP path before scheduling maintenance.
    while runtime.phase.value != "EXPLORE":
        advance()

    runtime.request_prune(reason="F0 audit: non-mutating prune proposal")
    advance()
    if runtime.phase.value != "PRUNE":
        raise AssertionError("PRUNE proposal was not scheduled from EXPLORE")
    if advance() != "PRUNE" or runtime.phase.value != "EXPLORE":
        raise AssertionError("PRUNE proposal did not execute and resume EXPLORE")

    runtime.request_grow(reason="F0 audit: non-mutating grow proposal")
    advance()
    if runtime.phase.value != "GROW":
        raise AssertionError("GROW proposal was not scheduled from EXPLORE")
    if advance() != "GROW" or runtime.phase.value != "EXPLORE":
        raise AssertionError("GROW proposal did not execute and resume EXPLORE")

    phase_config = config.runtime.phase
    required_counts = {
        "WARMUP": max(1, phase_config.warmup_steps),
        "EXPLORE": phase_config.explore_steps,
        "CONSOLIDATE": phase_config.consolidate_steps,
    }
    while any(
        runtime.phase_counts.get(name, 0) < required
        for name, required in required_counts.items()
    ):
        advance()

    evaluated_phase = runtime.evaluate(
        lambda: runtime.phase.value,
        reason="F0 audit: evaluator callback",
    )
    if evaluated_phase != "EVALUATE":
        raise AssertionError("evaluation callback did not execute in EVALUATE")
    while runtime.phase.value in {"EVALUATE", "PRUNE", "GROW"}:
        advance()

    # The audit makes an explicit reset and samples only from its new
    # continuity segment; the circular store retains earlier segments too.
    runtime.reset_memory()
    for _ in range(3):
        advance()
    windows = runtime.trace_store.build_windows(
        3,
        session_id=runtime.session_id,
        continuity_id=runtime.continuity_id,
    )
    if not windows:
        raise AssertionError("no continuity-valid trace window was produced")
    sampled = runtime.trace_store.sample_contiguous(
        3,
        seed=seed,
        session_id=runtime.session_id,
        continuity_id=runtime.continuity_id,
    )
    sampled_steps = [row.step_index for row in sampled.records]
    if len({row.continuity_id for row in sampled.records}) != 1:
        raise AssertionError("sample crossed a memory-reset continuity boundary")
    if sampled_steps != list(range(sampled_steps[0], sampled_steps[0] + 3)):
        raise AssertionError("sampled trace window is not step-contiguous")

    while runtime.phase.value in {"EVALUATE", "PRUNE", "GROW"}:
        advance()
    checkpoint = runtime.snapshot()
    restored = _make_runtime(
        config, memory_config, session_id=f"f0-spine-restore-seed{seed}"
    )
    restored.restore(checkpoint)
    if restored.snapshot() != checkpoint:
        raise AssertionError("checkpoint restore did not reproduce runtime state")

    continuation_forcing = [0.0] * config.dim
    continuation_forcing[(audit_step * 7 + seed) % config.dim] = 0.25
    for candidate in (runtime, restored):
        candidate.step(
            continuation_forcing,
            token="checkpoint-continuation",
            episode_id=f"spine-audit-seed{seed}",
        )
    if runtime.memory_state.to_mapping() != restored.memory_state.to_mapping():
        raise AssertionError("restored runtime diverged on deterministic continuation")
    if runtime.phase_counts != restored.phase_counts or runtime.step_count != restored.step_count:
        raise AssertionError("restored runtime counters diverged on continuation")

    kernel_unchanged = runtime.memory_config.to_mapping() == kernel_before
    if not kernel_unchanged:
        raise AssertionError("F0 maintenance audit mutated the SOE kernel")
    observed_phases = set(runtime.phase_counts)
    for event in runtime.phase_controller.events:
        observed_phases.update((event.previous, event.current))
    required_phases = {"WARMUP", "EXPLORE", "CONSOLIDATE", "EVALUATE", "PRUNE", "GROW"}
    missing = sorted(required_phases - observed_phases)
    if missing:
        raise AssertionError(f"F0 spine audit missed phases: {missing}")

    return {
        "status": "passed",
        "seed": seed,
        "phase_coverage": {name: name in observed_phases for name in sorted(required_phases)},
        "phase_step_counts": dict(runtime.phase_counts),
        "phase_events": [event.to_mapping() for event in runtime.phase_controller.events],
        "maintenance": {
            "prune_proposal_steps": runtime.phase_counts.get("PRUNE", 0),
            "grow_proposal_steps": runtime.phase_counts.get("GROW", 0),
            "kernel_unchanged": kernel_unchanged,
            "kernel_mutation_enabled": False,
        },
        "evaluation": {"callback_phase": evaluated_phase},
        "trace_store": {
            "capacity": runtime.trace_store.capacity,
            "records_retained": len(runtime.trace_store),
            "continuity_id_sampled": sampled.records[0].continuity_id,
            "window_count_in_current_continuity": len(windows),
            "sampled_start_step": sampled.start_step,
            "sampled_end_step": sampled.end_step,
            "sample_is_contiguous": True,
        },
        "checkpoint_restore": {
            "schema_version": checkpoint["schema_version"],
            "snapshot_round_trip_equal": True,
            "deterministic_continuation_equal": True,
            "trace_history_is_separate": True,
        },
        "diagnostics": {
            "prediction_error_samples": runtime.step_count,
            "last_prediction_error": runtime.last_prediction_error,
            "stability_checks": runtime.stability_checks,
            "stability_violations": runtime.stability_violations,
        },
    }


def run_babi_f0_experiment(
    config: BabiF0Config,
    *,
    data_dir: Path = _DATA_DIR,
    quick: bool = False,
    max_train_stories: int = 0,
    eval_split: str = "test",
) -> dict[str, object]:
    """Run one bAbI task through F0 and return reproducible metrics."""
    train_path = data_dir / "babi_train.jsonl"
    if eval_split not in {"valid", "test"}:
        raise ValueError("eval_split must be 'valid' or 'test'")
    eval_path = data_dir / f"babi_{eval_split}.jsonl"
    train_dataset = BabiDataset.from_jsonl(train_path).filter_task(config.task)
    eval_dataset = BabiDataset.from_jsonl(eval_path).filter_task(config.task)
    if not train_dataset.stories or not eval_dataset.stories:
        raise ValueError(f"bAbI task {config.task} requires non-empty train and {eval_split} splits")

    if quick:
        train_dataset.stories = train_dataset.stories[:20]
        eval_dataset.stories = eval_dataset.stories[:20]
    elif max_train_stories > 0:
        train_dataset.stories = train_dataset.stories[:max_train_stories]

    vocabulary = build_babi_vocabulary(train_dataset.stories + eval_dataset.stories)
    answer_set = build_babi_answer_set(train_dataset.stories)
    if len(answer_set) < 2:
        raise ValueError("the selected task must have at least two training answer classes")
    bridge = build_babi_bridge(vocabulary, dim=config.dim)
    memory_config = build_babi_memory_config(variant=config.variant)
    epochs = 2 if quick else config.epochs

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
        training_epoch_losses: list[float] = []
        for epoch in range(epochs):
            order = list(enumerate(train_dataset.stories))
            rng.shuffle(order)
            epoch_losses: list[float] = []
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
                    feature_scale=config.feature_scale,
                )
                if result.target is None:
                    raise ValueError("bAbI training story is missing an answer target")
                target_index = answer_set.index(result.target)
                loss = readout.train_step(
                    features,
                    target_index,
                    learning_rate=config.learning_rate,
                )
                epoch_losses.append(float(loss))
            training_epoch_losses.append(
                sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
            )

        # Held-out evaluation starts with a fresh runtime. In continuous mode,
        # state carries across test stories, but never from train into test.
        test_runtime = _make_runtime(
            config,
            memory_config,
            session_id=f"babi-{eval_split}-task{config.task}-seed{seed}",
        )
        correct = 0
        predictions: list[dict[str, object]] = []
        for story_index, story in enumerate(eval_dataset.stories):
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
                feature_scale=config.feature_scale,
            )
            prediction_index, confidence = test_runtime.evaluate(
                lambda: readout.predict(features),
                reason=f"held-out prediction for {result.episode_id}",
            )
            probabilities = readout.probabilities(features)
            record = build_prediction_record(
                story, story_index, answer_set, probabilities
            )
            if answer_set[prediction_index] != record["prediction"]:
                raise AssertionError("runtime evaluate and probability diagnostics disagree")
            record["episode_id"] = result.episode_id
            record["runtime_steps"] = result.last_step - result.first_step + 1
            record["runtime_confidence"] = float(confidence)
            predictions.append(record)
            correct += int(record["correct"])

        total = len(eval_dataset.stories)
        spine_exercise = (
            exercise_f0_spine(config, memory_config, seed=seed)
            if config.exercise_spine else None
        )
        seed_results.append({
            "seed": seed,
            "spine_exercise": spine_exercise,
            "correct": correct,
            "total": total,
            "accuracy": correct / total if total else 0.0,
            "train_runtime": train_runtime.runtime_summary(),
            "test_runtime": test_runtime.runtime_summary(),
            "predictions": predictions,
            "diagnostics": summarize_predictions(predictions, answer_set),
            "training": summarize_training_losses(training_epoch_losses),
        })

    accuracies = [float(row["accuracy"]) for row in seed_results]
    mean_accuracy = sum(accuracies) / len(accuracies)
    standard_error = (
        math.sqrt(sum((value - mean_accuracy) ** 2 for value in accuracies) / len(accuracies))
        / math.sqrt(len(accuracies))
        if len(accuracies) > 1 else 0.0
    )
    all_predictions = [
        prediction for seed_result in seed_results
        for prediction in seed_result["predictions"]
    ]
    return {
        "experiment": "babi_f0_runtime_integrated",
        "task": config.task,
        "variant": config.variant,
        "dim": config.dim,
        "readout_type": config.readout_type,
        "feature_scale": config.feature_scale,
        "learning_rate": config.learning_rate,
        "mode": config.mode,
        "seeds": config.seeds,
        "epochs": epochs,
        "evaluation_split": eval_split,
        "train_stories": len(train_dataset.stories),
        "evaluation_stories": len(eval_dataset.stories),
        "test_stories": len(eval_dataset.stories) if eval_split == "test" else None,
        "vocabulary_size": vocabulary.size,
        "answer_count": len(answer_set),
        "mean_accuracy": mean_accuracy,
        "standard_error": standard_error,
        "runtime_config": config.runtime.to_mapping(),
        "memory_config": memory_config.to_mapping(),
        "seed_results": seed_results,
        "diagnostics": summarize_predictions(all_predictions, answer_set),
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
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--feature-scale", type=float, default=100.0)
    parser.add_argument("--eval-split", choices=["valid", "test"], default="test")
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
    parser.add_argument("--exercise-spine", action="store_true", help="run the deterministic all-component F0 spine audit for each seed")
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
        feature_scale=args.feature_scale,
        mode=args.mode,
        silence_steps_between_statements=args.silence_steps,
        runtime=runtime_config,
        exercise_spine=args.exercise_spine,
    )
    result = run_babi_f0_experiment(
        config,
        data_dir=args.data_dir,
        quick=args.quick,
        max_train_stories=args.max_train_stories,
        eval_split=args.eval_split,
    )
    paths = save_babi_f0_bundle(result, output_dir=args.output_dir)
    print(
        f"task={result['task']} variant={result['variant']} dim={result['dim']} "
        f"mode={result['mode']}"
    )
    print(f"accuracy={result['mean_accuracy']:.3f} +/- {result['standard_error']:.3f}")
    if args.exercise_spine:
        passed = all(row["spine_exercise"]["status"] == "passed" for row in result["seed_results"])
        print(f"f0_spine_exercise={'passed' if passed else 'failed'} seeds={len(result['seed_results'])}")
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
