"""G4.2 corrected hierarchy: matched H x C x D static sweep and adaptive follow-up."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from ..digital.babi import (
    BabiDataset, BabiStory, build_babi_answer_set, build_babi_bridge,
    babi_story_to_episode,
)
from ..digital.bridge import DigitalMemoryConfig
from ..digital.tokens import Vocabulary
from ..digital.readout import LinearReadout, query_conditioned_features
from ..digital.hierarchy_g42 import NLevelVectorizedHierarchyDirector, _fit_positive_spectrum
from .confidence_metrics import summarize_confidence
from .g0_diagnostics import build_prediction_record, summarize_training_losses
from .symbolic_induction import memory_config_for_variant

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATA = ROOT / "code/python/d2c/progress/babi_data"
CHANNEL_TEMPLATE = (3, 2, 2, 2)
BASELINE_CHANNEL_BUDGET = sum(CHANNEL_TEMPLATE[:2])


@dataclass(frozen=True, slots=True)
class G42Config:
    depth: int = 2
    capacity: int = 3
    dim: int = 64
    seeds: int = 3
    epochs: int = 3
    checkpoint_epochs: tuple[int, ...] = (3,)
    learning_rate: float = 0.05
    feature_scale: float = 10.0
    bootstrap_samples: int = 400
    bottom_up_gain: float = 0.4
    top_down_gain: float = 0.5
    context_gamma: tuple[float, ...] = (0.35, 0.08)
    context_w: tuple[float, ...] = (0.05, 0.02)
    context_leak: float = 0.5
    growth_factor: float = 0.5

    def __post_init__(self):
        if self.depth not in (2, 3, 4):
            raise ValueError("G4.2 depth must be H2, H3, or H4")
        if self.capacity not in (3, 4, 6):
            raise ValueError("G4.2 capacity C must be 3, 4, or 6")
        if self.dim not in (64, 128):
            raise ValueError("G4.2 representation dimension D must be 64 or 128")
        if self.seeds != 3 or not 3 <= self.epochs <= 12:
            raise ValueError("G4.2 duration checks fix three seeds and allow 3–12 epochs")
        if (
            not self.checkpoint_epochs
            or tuple(sorted(set(self.checkpoint_epochs))) != self.checkpoint_epochs
            or self.checkpoint_epochs[-1] != self.epochs
            or any(epoch < 1 or epoch > self.epochs for epoch in self.checkpoint_epochs)
        ):
            raise ValueError("checkpoint_epochs must be sorted, unique, within training, and include the final epoch")
        if self.learning_rate != 0.05 or self.feature_scale != 10.0:
            raise ValueError("G4.2 retains the G4.1 learning rate and feature scale")
        if self.bootstrap_samples < 200:
            raise ValueError("at least 200 bootstrap samples are required")


def _fingerprint(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _readout_hash(readout: LinearReadout) -> str:
    return hashlib.sha256(readout.W.tobytes() + readout.b.tobytes()).hexdigest()


def _training_vocab(stories: list[BabiStory]) -> Vocabulary:
    tokens: set[str] = {"<UNK>"}
    for story in stories:
        for statement in story.statements:
            tokens.update(statement.split())
        tokens.update(story.question.split())
        tokens.add(story.answer)
    return Vocabulary(sorted(tokens))


def _map_validation_story(story: BabiStory, vocab: Vocabulary, answers: list[str]) -> BabiStory:
    known = set(vocab.tokens)

    def map_text(value: str) -> str:
        return " ".join(token if token in known else "<UNK>" for token in value.split())

    return BabiStory(
        statements=[map_text(value) for value in story.statements],
        question=map_text(story.question),
        answer=story.answer if story.answer in answers else "<UNK>",
        task=story.task, raw_passage=story.raw_passage,
    )


def _orders(count: int, seeds: int, epochs: int) -> list[list[list[int]]]:
    all_orders = []
    for seed in range(seeds):
        rng = random.Random(4100 + seed)
        seed_orders = []
        for _ in range(epochs):
            order = list(range(count))
            rng.shuffle(order)
            seed_orders.append(order)
        all_orders.append(seed_orders)
    return all_orders


def _channel_counts(depth: int, capacity: int) -> list[int]:
    """Fixed total budget 5*C, distributed by the [3,2,2,2] level template."""
    if depth not in (2, 3, 4) or capacity not in (3, 4, 6):
        raise ValueError("unsupported G4.2 depth/capacity")
    weights = list(CHANNEL_TEMPLATE[:depth])
    total_budget = capacity * BASELINE_CHANNEL_BUDGET
    raw = [total_budget * value / sum(weights) for value in weights]
    counts = [max(1, math.floor(value)) for value in raw]
    remainder = total_budget - sum(counts)
    order = sorted(
        range(depth),
        key=lambda index: (-(raw[index] - math.floor(raw[index])), index),
    )
    for index in order[:remainder]:
        counts[index] += 1
    if sum(counts) != total_budget or min(counts) < 2:
        raise RuntimeError("channel budget allocation failed")
    return counts


def _resample_kernel(config: DigitalMemoryConfig, channel_count: int) -> DigitalMemoryConfig:
    if channel_count < 2:
        raise ValueError("each hierarchy level needs at least two channels")
    if channel_count == config.channel_count:
        return DigitalMemoryConfig(
            gamma=list(config.gamma), w=list(config.w), dt=config.dt,
            leak_rate=config.leak_rate, form="negative_feedback",
        )
    maximum, minimum = max(config.gamma), min(config.gamma)
    if channel_count == 1:
        return _fit_positive_spectrum(config, [1.0 / (
            config.stability_ratio() * config.leak_rate / sum(config.w)
        )])
    gammas = [
        maximum * (minimum / maximum) ** (index / (channel_count - 1))
        for index in range(channel_count)
    ]
    return _fit_positive_spectrum(config, gammas)


def _level_configs(config: G42Config) -> list[DigitalMemoryConfig]:
    fast = memory_config_for_variant("full")
    context = DigitalMemoryConfig(
        gamma=list(config.context_gamma), w=list(config.context_w),
        dt=fast.dt, leak_rate=config.context_leak, form="negative_feedback",
    )
    base = [fast] + [
        DigitalMemoryConfig(
            gamma=list(context.gamma), w=list(context.w),
            dt=context.dt, leak_rate=context.leak_rate, form="negative_feedback",
        )
        for _ in range(config.depth - 1)
    ]
    counts = _channel_counts(config.depth, config.capacity)
    return [_resample_kernel(level, count) for level, count in zip(base, counts)]


def _multi_level_features(
    result, level_configs: Sequence[DigitalMemoryConfig], query: Sequence[float],
    *, feature_scale: float,
) -> list[float]:
    """Concatenate normalized per-level memory/query interactions.

    The query vector is included once at level zero. Upper-level blocks retain
    their state and query-gated state terms but zero the duplicate bare query,
    so an upper level cannot pass the contribution gate using query-only input.
    """
    depth = len(level_configs)
    state_dim = len(query)
    block_scale = 1.0 / math.sqrt(depth)
    features: list[float] = []
    for level, (state, memory_config) in enumerate(zip(result.final_states, level_configs)):
        block = query_conditioned_features(
            state, memory_config, query, feature_scale=feature_scale,
        )
        if level > 0:
            block[state_dim:2 * state_dim] = [0.0] * state_dim
        features.extend(value * block_scale for value in block)
    return features


def _level_contribution_l1(readout: LinearReadout, features: list[float], block_dim: int, depth: int) -> list[float]:
    baseline = readout.probabilities(features)
    contributions = []
    for level in range(depth):
        ablated = list(features)
        start = level * block_dim
        ablated[start:start + block_dim] = [0.0] * block_dim
        changed = readout.probabilities(ablated)
        contributions.append(sum(abs(a - b) for a, b in zip(baseline, changed)))
    return contributions


def _run_seed(
    *,
    task: int, seed: int, train_stories: list[BabiStory],
    valid_stories: list[BabiStory], answer_set: list[str], bridge,
    cfg: G42Config, orders: list[list[int]], adaptive: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    level_configs = _level_configs(cfg)
    level_feature_dim = cfg.dim * 2 + cfg.dim * cfg.dim
    readout = LinearReadout(
        input_dim=cfg.depth * level_feature_dim, output_dim=len(answer_set), seed=seed,
    )
    initial_hash = _readout_hash(readout)
    director = NLevelVectorizedHierarchyDirector(
        level_configs=level_configs, state_dim=cfg.dim,
        bottom_up_gain=cfg.bottom_up_gain, top_down_gain=cfg.top_down_gain,
        preserve_top_down_weight_mass=True,
    )
    losses, mutation_events, learning_curve = [], [], []
    final_predictions = None
    final_confidence = None
    final_diagnostics = None

    def evaluate_validation(*, final_checkpoint: bool):
        predictions = []
        max_stability = [0.0] * cfg.depth
        max_norm = [0.0] * cfg.depth
        readout_contribution = [0.0] * cfg.depth
        top_down_observed = [0.0] * cfg.depth
        bottom_up_observed = [0.0] * cfg.depth
        all_finite = True

        for story_index, story in enumerate(valid_stories):
            episode = babi_story_to_episode(story, bridge, answer_set)
            result = director.run_episode(
                episode,
                experiment_name=f"g42-h{cfg.depth}-c{cfg.capacity}-d{cfg.dim}-qa{task}-s{seed}-valid",
            )
            features = _multi_level_features(
                result, director.level_configs, episode.query, feature_scale=cfg.feature_scale,
            )
            probabilities = readout.probabilities(features)
            predictions.append(build_prediction_record(
                story, story_index, answer_set, probabilities,
            ))
            if final_checkpoint:
                contributions = _level_contribution_l1(
                    readout, features, level_feature_dim, cfg.depth,
                )
                for level, value in enumerate(contributions):
                    readout_contribution[level] += value
                for level in range(cfg.depth):
                    max_stability[level] = max(
                        max_stability[level], max(result.stability_ratios[level], default=0.0),
                    )
                    max_norm[level] = max(max_norm[level], result.max_state_norms[level])
                    top_down_observed[level] += result.top_down_input_norms[level]
                    bottom_up_observed[level] += result.bottom_up_input_norms[level]
                all_finite = all_finite and all(
                    math.isfinite(value)
                    for state in result.final_states
                    for value in state.u
                ) and all(
                    math.isfinite(value)
                    for trace in result.level_traces
                    for step in trace.steps
                    for value in (*step.u, *(item for channel in step.chi for item in channel))
                )

        count = max(1, len(valid_stories))
        confidence = summarize_confidence(
            predictions, answer_set, calibration_bins=10,
            bootstrap_samples=cfg.bootstrap_samples if final_checkpoint else 0,
            bootstrap_seed=task * 10000 + seed * 101 + cfg.depth * 13 + cfg.capacity * 7 + cfg.dim,
        )
        diagnostics = None
        if final_checkpoint:
            readout_contribution = [value / count for value in readout_contribution]
            top_down_observed = [value / count for value in top_down_observed]
            bottom_up_observed = [value / count for value in bottom_up_observed]
            diagnostics = {
                "max_observed_stability_ratio_by_level": max_stability,
                "max_state_norm_by_level": max_norm,
                "stability_pass": all(value <= 0.9 for value in max_stability),
                "finite_state_pass": all_finite,
                "mean_top_down_input_l2_by_level": top_down_observed,
                "mean_bottom_up_input_l2_by_level": bottom_up_observed,
                "level_readout_contribution_l1": readout_contribution,
                "cross_level_top_down_pass": all(
                    top_down_observed[level] > 1e-12 for level in range(cfg.depth - 1)
                ),
                "cross_level_bottom_up_pass": all(
                    bottom_up_observed[level] > 1e-12 for level in range(1, cfg.depth)
                ),
                "all_levels_readout_pass": all(value > 1e-8 for value in readout_contribution),
            }
        return predictions, confidence, diagnostics

    for epoch_index, order in enumerate(orders):
        epoch_losses = []
        for index in order:
            story = train_stories[index]
            episode = babi_story_to_episode(story, bridge, answer_set)
            result = director.run_episode(
                episode, experiment_name=f"g42-h{cfg.depth}-c{cfg.capacity}-d{cfg.dim}-qa{task}-s{seed}-train",
            )
            features = _multi_level_features(
                result, director.level_configs, episode.query, feature_scale=cfg.feature_scale,
            )
            epoch_losses.append(readout.train_step(
                features, answer_set.index(story.answer), learning_rate=cfg.learning_rate,
            ))
        losses.append(statistics.mean(epoch_losses) if epoch_losses else 0.0)

        epoch_number = epoch_index + 1
        if adaptive and epoch_index == 0:
            mutation_events = director.prune_grow(growth_factor=cfg.growth_factor)

        if epoch_number in cfg.checkpoint_epochs:
            predictions, confidence, diagnostics = evaluate_validation(
                final_checkpoint=(epoch_number == cfg.epochs),
            )
            learning_curve.append({
                "epoch": epoch_number,
                "correct": sum(int(row["correct"]) for row in predictions),
                "total": len(predictions),
                "accuracy": sum(int(row["correct"]) for row in predictions) / max(1, len(predictions)),
                "fixed_bin_top_label_ece": confidence["calibration"]["fixed_bin_top_label_ece"],
                "negative_log_likelihood": confidence["negative_log_likelihood"],
            })
            if epoch_number == cfg.epochs:
                final_predictions = predictions
                final_confidence = confidence
                final_diagnostics = diagnostics

    if final_predictions is None or final_confidence is None or final_diagnostics is None:
        raise RuntimeError("final validation checkpoint was not evaluated")

    channels = [config.channel_count for config in director.level_configs]
    return {
        "seed": seed,
        "correct": sum(int(row["correct"]) for row in final_predictions),
        "total": len(final_predictions),
        "accuracy": sum(int(row["correct"]) for row in final_predictions) / max(1, len(final_predictions)),
        "initial_readout_sha256": initial_hash,
        "training_order_sha256": _fingerprint(orders),
        "training": summarize_training_losses(losses),
        "epoch_checkpoints": learning_curve,
        "predictions": final_predictions,
        "confidence": final_confidence,
        "compute_seconds": time.perf_counter() - started,
        "hierarchy": {
            "depth": cfg.depth, "capacity": cfg.capacity, "dim": cfg.dim,
            "channels_per_level_after_run": channels,
            "total_channels_after_run": sum(channels),
            "kernel_mass_by_level": [sum(config.w) for config in director.level_configs],
            "stability_ratio_by_level": [config.stability_ratio() for config in director.level_configs],
            **final_diagnostics,
            "stability_margin": 0.9,
            "top_down_weight_mass_preserved": True,
        },
        "structural_adaptation": {
            "enabled": adaptive,
            "events": mutation_events,
            "channel_births": sum(event.get("channel_births", 0) for event in mutation_events),
            "channel_deaths": sum(event.get("channel_deaths", 0) for event in mutation_events),
            "capacity_delta_by_level": [event.get("capacity_delta", 0) for event in mutation_events],
            "channels_before_by_level": [event["channel_count_before"] for event in mutation_events],
            "channels_after_by_level": [event["channel_count_after"] for event in mutation_events],
        },
    }

def _task_arm_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    accuracies = [row["accuracy"] for row in rows]
    depth = len(rows[0]["hierarchy"]["channels_per_level_after_run"])
    checkpoint_epochs = rows[0]["epoch_checkpoints"]
    learning_curve = []
    for checkpoint in checkpoint_epochs:
        epoch = checkpoint["epoch"]
        matching = [
            next(item for item in row["epoch_checkpoints"] if item["epoch"] == epoch)
            for row in rows
        ]
        curve_accuracy = [item["accuracy"] for item in matching]
        learning_curve.append({
            "epoch": epoch,
            "mean_accuracy_over_seeds": statistics.mean(curve_accuracy),
            "standard_deviation_over_seeds": statistics.stdev(curve_accuracy),
            "mean_fixed_bin_top_label_ece": statistics.mean(
                item["fixed_bin_top_label_ece"] for item in matching
            ),
            "mean_negative_log_likelihood": statistics.mean(
                item["negative_log_likelihood"] for item in matching
            ),
        })
    return {
        "mean_accuracy_over_seeds": statistics.mean(accuracies),
        "standard_deviation_over_seeds": statistics.stdev(accuracies),
        "mean_compute_seconds": statistics.mean(row["compute_seconds"] for row in rows),
        "learning_curve": learning_curve,
        "mean_level_readout_contribution_l1": [
            statistics.mean(row["hierarchy"]["level_readout_contribution_l1"][level] for row in rows)
            for level in range(depth)
        ],
        "stability_pass": all(row["hierarchy"]["stability_pass"] for row in rows),
        "finite_state_pass": all(row["hierarchy"]["finite_state_pass"] for row in rows),
        "all_levels_readout_pass": all(row["hierarchy"]["all_levels_readout_pass"] for row in rows),
        "cross_level_top_down_pass": all(row["hierarchy"]["cross_level_top_down_pass"] for row in rows),
        "cross_level_bottom_up_pass": all(row["hierarchy"]["cross_level_bottom_up_pass"] for row in rows),
        "training_order_hashes": [row["training_order_sha256"] for row in rows],
        "seed_results": rows,
    }


def run_g42(*, config: G42Config, data_dir: Path, mode: str = "static") -> dict[str, Any]:
    if mode not in {"static", "prune_grow"}:
        raise ValueError("mode must be static or prune_grow")
    started = datetime.now(timezone.utc).isoformat()
    train_path, valid_path = data_dir / "babi_train.jsonl", data_dir / "babi_valid.jsonl"
    if not train_path.is_file() or not valid_path.is_file():
        raise FileNotFoundError("G4.2 requires train and validation splits only")
    training = BabiDataset.from_jsonl(train_path)
    validation = BabiDataset.from_jsonl(valid_path)
    tasks: dict[str, Any] = {}
    initial_channels = _channel_counts(config.depth, config.capacity)

    for task in (3, 5):
        train = training.filter_task(task).stories
        raw_valid = validation.filter_task(task).stories
        if not train or not raw_valid:
            raise ValueError(f"QA{task} requires non-empty train and validation stories")
        vocab = _training_vocab(train)
        answers = build_babi_answer_set(train)
        if "<UNK>" not in answers:
            answers = sorted([*answers, "<UNK>"])
        valid = [_map_validation_story(story, vocab, answers) for story in raw_valid]
        bridge = build_babi_bridge(vocab, dim=config.dim)
        seed_orders = _orders(len(train), config.seeds, config.epochs)
        arm_names = ["static"] if mode == "static" else ["static", "prune_grow"]
        arms: dict[str, Any] = {}
        for arm in arm_names:
            rows = [
                _run_seed(
                    task=task, seed=seed, train_stories=train, valid_stories=valid,
                    answer_set=answers, bridge=bridge, cfg=config,
                    orders=seed_orders[seed], adaptive=(arm == "prune_grow"),
                )
                for seed in range(config.seeds)
            ]
            arms[arm] = _task_arm_summary(rows)

        task_row = {
            "task": task, "task_label": f"QA{task}",
            "training_story_count": len(train), "validation_story_count": len(valid),
            "answer_classes": answers, "training_only_vocabulary_size": vocab.size,
            "validation_unknown_answer_count": sum(story.answer == "<UNK>" for story in valid),
            "arms": arms,
        }
        if mode == "prune_grow":
            static_rows = arms["static"]["seed_results"]
            adaptive_rows = arms["prune_grow"]["seed_results"]
            task_row["paired_accuracy_delta_by_seed"] = [
                adaptive_rows[index]["accuracy"] - static_rows[index]["accuracy"]
                for index in range(config.seeds)
            ]
            task_row["mean_paired_accuracy_delta"] = statistics.mean(
                task_row["paired_accuracy_delta_by_seed"],
            )
            task_row["matched_readout_initialization"] = all(
                static_rows[index]["initial_readout_sha256"] == adaptive_rows[index]["initial_readout_sha256"]
                for index in range(config.seeds)
            )
            task_row["matched_training_order"] = all(
                static_rows[index]["training_order_sha256"] == adaptive_rows[index]["training_order_sha256"]
                for index in range(config.seeds)
            )
            task_row["capacity_delta_by_seed_and_level"] = [
                adaptive_rows[index]["structural_adaptation"]["capacity_delta_by_level"]
                for index in range(config.seeds)
            ]
        tasks[str(task)] = task_row

    all_arms = [
        arm for task in tasks.values() for arm in task["arms"].values()
    ]
    stability = all(arm["stability_pass"] for arm in all_arms)
    finite = all(arm["finite_state_pass"] for arm in all_arms)
    readout = all(arm["all_levels_readout_pass"] for arm in all_arms)
    top_down = all(arm["cross_level_top_down_pass"] for arm in all_arms)
    bottom_up = all(arm["cross_level_bottom_up_pass"] for arm in all_arms)
    adaptation_rows = [
        seed for task in tasks.values()
        for seed in task["arms"].get("prune_grow", {}).get("seed_results", [])
    ]
    decision_count = sum(len(seed["structural_adaptation"]["events"]) for seed in adaptation_rows)
    births = sum(seed["structural_adaptation"]["channel_births"] for seed in adaptation_rows)
    deaths = sum(seed["structural_adaptation"]["channel_deaths"] for seed in adaptation_rows)

    result = {
        "schema_version": 2,
        "experiment": "G4.2 corrected vector-coupled multi-level hierarchy",
        "experiment_id": f"G4.2-H{config.depth}-C{config.capacity}-D{config.dim}-E{config.epochs}-{mode.upper()}",
        "mode": mode, "status": "completed",
        "started_at_utc": started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_revision": os.environ.get("GITHUB_SHA", "unknown"),
        "protocol": {
            **asdict(config), "tasks": [3, 5], "evaluation_split": "valid",
            "test_split_opened": False, "seed_ids": [0, 1, 2],
            "training_epochs": config.epochs, "checkpoint_epochs": list(config.checkpoint_epochs), "initial_channels_per_level": initial_channels,
            "initial_total_channels": sum(initial_channels),
            "capacity_interpretation": (
                "C is a global channel-budget multiplier. Total channels are 5*C "
                "at every H; the budget is allocated proportionally to [3,2,2,2] "
                "over the active levels using deterministic largest-remainder rounding."
            ),
            "dimension_interpretation": "D is the vector state and query dimension.",
            "baseline": "G2 E-F0 spine with G4.1 mass-preserving hierarchy kernel definitions",
            "coupling": (
                "simultaneous adjacent-level structured vector messages; each message "
                "combines 0.25*u with weighted chi memory, applies tanh per coordinate, "
                "then clips its L2 norm to the configured gain"
            ),
            "readout": (
                "normalized concatenation of every level's memory and query-gated memory; "
                "the bare query vector is supplied only once at level zero"
            ),
            "kernel_invariants": "positive weights; each level's original kernel mass and stability ratio preserved during channel resampling and adaptive edits",
            "adaptation_policy": (
                "after epoch one, each level prunes a near-dormant channel when relative "
                "activity is <= 1%; otherwise grows a slower channel when observed activity "
                "share is >= 5%; exact mass and stability are preserved"
                if mode == "prune_grow" else "static topology and channel allocation"
            ),
            "selection_policy": (
                "only after all 18 static configurations succeed; require finite states, "
                "stability <= 0.9, measurable top-down and bottom-up influence, and "
                "measurable contribution from every level; rank QA3 accuracy, QA5 accuracy, "
                "QA3 fixed-bin ECE, compute time, then resource cost"
            ),
            "locked_test_policy": "validation-only; test split is not downloaded or opened",
        },
        "dataset": {
            "train": {"sha256": _sha256(train_path), "bytes": train_path.stat().st_size},
            "valid": {"sha256": _sha256(valid_path), "bytes": valid_path.stat().st_size},
        },
        "tasks": tasks,
        "acceptance": {
            "qa3_qa5_validation_coverage": set(tasks) == {"3", "5"},
            "training_checkpoint_coverage": all(
                all(
                    [item["epoch"] for item in seed["epoch_checkpoints"]] == list(config.checkpoint_epochs)
                    for seed in arm["seed_results"]
                )
                for task in tasks.values() for arm in task["arms"].values()
            ),
            "stability_within_0_9_margin": stability,
            "all_states_finite": finite,
            "all_levels_contribute_to_readout": readout,
            "bottom_up_and_top_down_coupling_observed": top_down and bottom_up,
            "three_matched_seed_orders": all(
                len(set(arm["training_order_hashes"])) == config.seeds
                for arm in all_arms
            ),
            "matched_static_adaptive_initialization": (
                mode == "static" or all(
                    task.get("matched_readout_initialization", False)
                    and task.get("matched_training_order", False)
                    for task in tasks.values()
                )
            ),
            "adaptive_decisions_recorded": (
                mode == "static" or decision_count == config.seeds * config.depth * 2
            ),
            "adaptive_channel_birth_or_death_observed": (
                mode == "static" or births + deaths > 0
            ),
            "adaptive_channel_births": births,
            "adaptive_channel_deaths": deaths,
        },
    }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="G4.2 corrected H x C x D hierarchy experiment.")
    parser.add_argument("--depth", type=int, choices=(2, 3, 4), required=True)
    parser.add_argument("--capacity", type=int, choices=(3, 4, 6), required=True)
    parser.add_argument("--dim", type=int, choices=(64, 128), required=True)
    parser.add_argument("--mode", choices=("static", "prune_grow"), default="static")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--checkpoint-epochs", type=str, default=None,
        help="Comma-separated validation checkpoints; must include the final epoch.",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/experiment-lab/g42")
    parser.add_argument("--bootstrap-samples", type=int, default=400)
    args = parser.parse_args(argv)
    checkpoint_epochs = (
        tuple(int(value.strip()) for value in args.checkpoint_epochs.split(",") if value.strip())
        if args.checkpoint_epochs else (args.epochs,)
    )
    config = G42Config(
        depth=args.depth, capacity=args.capacity, dim=args.dim,
        epochs=args.epochs, checkpoint_epochs=checkpoint_epochs,
        bootstrap_samples=args.bootstrap_samples,
    )
    result = run_g42(config=config, data_dir=args.data_dir, mode=args.mode)
    run_id = (
        f"g42_h{args.depth}_c{args.capacity}_d{args.dim}_{args.mode}_"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    bundle = args.output_dir / run_id
    bundle.mkdir(parents=True, exist_ok=True)
    (bundle / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "experiment_id": result["experiment_id"],
        "acceptance": result["acceptance"],
        "tasks": {
            task: {
                arm: {
                    "mean_accuracy": summary["mean_accuracy_over_seeds"],
                    "all_levels_contribute": summary["all_levels_readout_pass"],
                    "stable": summary["stability_pass"],
                }
                for arm, summary in row["arms"].items()
            }
            for task, row in result["tasks"].items()
        },
        "bundle": str(bundle),
    }, sort_keys=True))
    gates = [
        "qa3_qa5_validation_coverage", "training_checkpoint_coverage", "all_states_finite",
        "all_levels_contribute_to_readout",
        "bottom_up_and_top_down_coupling_observed",
        "three_matched_seed_orders",
    ]
    if args.mode == "prune_grow":
        gates.extend([
            "matched_static_adaptive_initialization",
            "adaptive_decisions_recorded",
            "adaptive_channel_birth_or_death_observed",
        ])
    failed = [key for key in gates if not result["acceptance"][key]]
    if failed:
        raise SystemExit("G4.2 execution/evidence gates failed: " + ", ".join(failed))
    # Stability is a scientific qualification and selector filter. Failed
    # configurations retain their full evidence bundle rather than disappearing.
    return result


if __name__ == "__main__":
    main()
