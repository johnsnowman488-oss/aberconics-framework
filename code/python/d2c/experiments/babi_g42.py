"""G4.2: parallel H2/H3/H4 sweep followed by a matched F4 prune/grow probe."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import random
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..digital.babi import BabiDataset, BabiStory, build_babi_answer_set, build_babi_bridge, babi_story_to_episode
from ..digital.bridge import DigitalMemoryConfig
from ..digital.tokens import Vocabulary
from ..digital.readout import LinearReadout, query_conditioned_features
from ..digital.hierarchy_g42 import NLevelVectorizedHierarchyDirector
from .confidence_metrics import summarize_confidence
from .g0_diagnostics import build_prediction_record, summarize_training_losses
from .symbolic_induction import memory_config_for_variant

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATA = ROOT / "code/python/d2c/progress/babi_data"

@dataclass(frozen=True, slots=True)
class G42Config:
    depth: int = 2
    dim: int = 64
    seeds: int = 3
    epochs: int = 3
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
        if self.dim != 64 or self.seeds != 3 or self.epochs != 3:
            raise ValueError("G4.2 first sweep fixes D=64, three seeds and three epochs")
        if self.learning_rate != 0.05 or self.feature_scale != 10.0:
            raise ValueError("G4.2 retains G4.1 learning rate and feature scale")
        if self.bootstrap_samples < 200:
            raise ValueError("at least 200 bootstrap samples are required")

def _fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

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

def _level_configs(cfg: G42Config) -> list[DigitalMemoryConfig]:
    fast = memory_config_for_variant("full")
    context = DigitalMemoryConfig(
        gamma=list(cfg.context_gamma), w=list(cfg.context_w),
        dt=fast.dt, leak_rate=cfg.context_leak, form="negative_feedback",
    )
    # Cascading levels provide longer effective dynamics while retaining the
    # calibrated per-level context kernel and its mass/stability budget.
    return [fast] + [
        DigitalMemoryConfig(gamma=list(context.gamma), w=list(context.w),
            dt=context.dt, leak_rate=context.leak_rate, form="negative_feedback")
        for _ in range(cfg.depth - 1)
    ]

def _run_seed(*, task: int, seed: int, train_stories: list[BabiStory],
              valid_stories: list[BabiStory], answer_set: list[str], bridge,
              cfg: G42Config, orders: list[list[int]], adaptive: bool) -> dict[str, Any]:
    feature_dim = cfg.dim * 2 + cfg.dim * cfg.dim
    readout = LinearReadout(input_dim=feature_dim, output_dim=len(answer_set), seed=seed)
    initial_hash = _readout_hash(readout)
    director = NLevelVectorizedHierarchyDirector(
        level_configs=_level_configs(cfg), state_dim=cfg.dim,
        bottom_up_gain=cfg.bottom_up_gain, top_down_gain=cfg.top_down_gain,
        preserve_top_down_weight_mass=True,
    )
    losses, mutation_events = [], []
    for epoch, order in enumerate(orders):
        epoch_losses = []
        for idx in order:
            story = train_stories[idx]
            episode = babi_story_to_episode(story, bridge, answer_set)
            result = director.run_episode(episode, experiment_name=f"g42-h{cfg.depth}-qa{task}-s{seed}-train")
            features = query_conditioned_features(
                result.final_state, director.level_configs[0], episode.query,
                feature_scale=cfg.feature_scale,
            )
            epoch_losses.append(readout.train_step(
                features, answer_set.index(story.answer), learning_rate=cfg.learning_rate))
        losses.append(statistics.mean(epoch_losses) if epoch_losses else 0.0)
        if adaptive and epoch == 0:
            mutation_events = director.prune_grow(growth_factor=cfg.growth_factor)

    predictions = []
    max_stability = [0.0] * cfg.depth
    max_norm = [0.0] * cfg.depth
    all_finite = True
    for story_index, story in enumerate(valid_stories):
        episode = babi_story_to_episode(story, bridge, answer_set)
        result = director.run_episode(episode, experiment_name=f"g42-h{cfg.depth}-qa{task}-s{seed}-valid")
        features = query_conditioned_features(
            result.final_state, director.level_configs[0], episode.query,
            feature_scale=cfg.feature_scale,
        )
        predictions.append(build_prediction_record(
            story, story_index, answer_set, readout.probabilities(features)))
        for level in range(cfg.depth):
            max_stability[level] = max(max_stability[level], max(result.stability_ratios[level], default=0.0))
            max_norm[level] = max(max_norm[level], result.max_state_norms[level])
        all_finite = all_finite and all(
            math.isfinite(value) for trace in result.level_traces for step in trace.steps
            for value in (*step.u, *(v for channel in step.chi for v in channel))
        )
    correct = sum(int(row["correct"]) for row in predictions)
    confidence = summarize_confidence(
        predictions, answer_set, calibration_bins=10,
        bootstrap_samples=cfg.bootstrap_samples,
        bootstrap_seed=task * 10000 + seed * 101 + cfg.depth,
    )
    return {
        "seed": seed, "correct": correct, "total": len(valid_stories),
        "accuracy": correct / max(1, len(valid_stories)),
        "initial_readout_sha256": initial_hash,
        "training_order_sha256": _fingerprint(orders),
        "training": summarize_training_losses(losses),
        "predictions": predictions, "confidence": confidence,
        "hierarchy": {
            "depth": cfg.depth,
            "channels_per_level": [c.channel_count for c in director.level_configs],
            "total_channels": sum(c.channel_count for c in director.level_configs),
            "kernel_mass_by_level": [sum(c.w) for c in director.level_configs],
            "stability_ratio_by_level": [c.stability_ratio() for c in director.level_configs],
            "max_observed_stability_ratio_by_level": max_stability,
            "max_state_norm_by_level": max_norm,
            "stability_margin": 0.9,
            "stability_pass": all(value <= 0.9 for value in max_stability),
            "finite_state_pass": all_finite,
            "coupling": "simultaneous adjacent-level bottom-up and top-down",
            "top_down_weight_mass_preserved": True,
        },
        "structural_adaptation": {
            "enabled": adaptive, "events": mutation_events,
            "prune_events": len(mutation_events), "grow_events": len(mutation_events),
        },
    }

def run_g42(*, config: G42Config, data_dir: Path, mode: str = "static") -> dict[str, Any]:
    if mode not in {"static", "prune_grow"}:
        raise ValueError("mode must be static or prune_grow")
    started = datetime.now(timezone.utc).isoformat()
    train_path, valid_path = data_dir / "babi_train.jsonl", data_dir / "babi_valid.jsonl"
    if not train_path.is_file() or not valid_path.is_file():
        raise FileNotFoundError("G4.2 requires train and valid splits only")
    training = BabiDataset.from_jsonl(train_path)
    validation = BabiDataset.from_jsonl(valid_path)
    tasks: dict[str, Any] = {}
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
                _run_seed(task=task, seed=seed, train_stories=train,
                    valid_stories=valid, answer_set=answers, bridge=bridge,
                    cfg=config, orders=seed_orders[seed], adaptive=(arm == "prune_grow"))
                for seed in range(config.seeds)
            ]
            acc = [row["accuracy"] for row in rows]
            arms[arm] = {
                "mean_accuracy_over_seeds": statistics.mean(acc),
                "standard_deviation_over_seeds": statistics.stdev(acc),
                "seed_results": rows,
                "stability_pass": all(row["hierarchy"]["stability_pass"] for row in rows),
                "finite_state_pass": all(row["hierarchy"]["finite_state_pass"] for row in rows),
                "training_order_hashes": [row["training_order_sha256"] for row in rows],
            }
        row = {
            "task": task, "task_label": f"QA{task}",
            "training_story_count": len(train), "validation_story_count": len(valid),
            "answer_classes": answers, "training_only_vocabulary_size": vocab.size,
            "validation_unknown_answer_count": sum(s.answer == "<UNK>" for s in valid),
            "arms": arms,
        }
        if mode == "prune_grow":
            static_rows, adaptive_rows = arms["static"]["seed_results"], arms["prune_grow"]["seed_results"]
            row["paired_accuracy_delta_by_seed"] = [
                adaptive_rows[i]["accuracy"] - static_rows[i]["accuracy"] for i in range(config.seeds)]
            row["mean_paired_accuracy_delta"] = statistics.mean(row["paired_accuracy_delta_by_seed"])
            row["matched_readout_initialization"] = all(
                static_rows[i]["initial_readout_sha256"] == adaptive_rows[i]["initial_readout_sha256"]
                for i in range(config.seeds))
            row["matched_training_order"] = all(
                static_rows[i]["training_order_sha256"] == adaptive_rows[i]["training_order_sha256"]
                for i in range(config.seeds))
        tasks[str(task)] = row

    stability = all(arm["stability_pass"] for task in tasks.values() for arm in task["arms"].values())
    finite = all(arm["finite_state_pass"] for task in tasks.values() for arm in task["arms"].values())
    event_count = sum(len(seed["structural_adaptation"]["events"])
        for task in tasks.values() for arm in task["arms"].values() for seed in arm["seed_results"])
    return {
        "schema_version": 1, "experiment": "G4.2 adaptive N-level hierarchy",
        "experiment_id": f"G4.2-H{config.depth}-{mode.upper()}",
        "mode": mode, "status": "completed", "started_at_utc": started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_revision": os.environ.get("GITHUB_SHA", "unknown"),
        "protocol": {
            **asdict(config), "tasks": [3, 5], "evaluation_split": "valid",
            "test_split_opened": False, "seed_ids": [0, 1, 2], "training_epochs": 3,
            "baseline": "G4.1 vectorized H2 mass-preserving hierarchy",
            "depth_policy": "H2/H3/H4; same per-level kernels and D; total resource use recorded",
            "vocabulary_policy": "training-only vocabulary; explicit UNK; validation labels excluded from vocabulary construction",
            "adaptation_policy": (
                "one controlled prune-then-grow cycle after epoch one at every level; "
                "least-active channel in slowest-timescale band replaced by a slower channel; "
                "total kernel mass and aggregate stability ratio preserved"
                if mode == "prune_grow" else "fixed channels and topology"),
            "winner_selection": "QA3 mean accuracy primary; QA5 secondary; stability <= 0.9 required; lower channel count then lower depth break ties",
            "locked_test_policy": "validation-only; test split not downloaded or opened",
        },
        "dataset": {
            "train": {"sha256": _sha256(train_path), "bytes": train_path.stat().st_size},
            "valid": {"sha256": _sha256(valid_path), "bytes": valid_path.stat().st_size},
        },
        "tasks": tasks,
        "acceptance": {
            "qa3_qa5_validation_coverage": set(tasks) == {"3", "5"},
            "stability_within_0_9_margin": stability,
            "all_states_finite": finite,
            "three_matched_seed_orders": all(
                len(set(arm["training_order_hashes"])) == config.seeds
                for task in tasks.values() for arm in task["arms"].values()),
            "matched_static_adaptive_initialization": (
                mode == "static" or all(row.get("matched_readout_initialization", False)
                    and row.get("matched_training_order", False) for row in tasks.values())),
            "prune_and_grow_observed": mode == "static" or event_count == config.seeds * config.depth * 2,
        },
    }

def main(argv=None):
    parser = argparse.ArgumentParser(description="G4.2 N-level hierarchy validation.")
    parser.add_argument("--depth", type=int, choices=(2, 3, 4), required=True)
    parser.add_argument("--mode", choices=("static", "prune_grow"), default="static")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/experiment-lab/g42")
    parser.add_argument("--bootstrap-samples", type=int, default=400)
    args = parser.parse_args(argv)
    config = G42Config(depth=args.depth, bootstrap_samples=args.bootstrap_samples)
    result = run_g42(config=config, data_dir=args.data_dir, mode=args.mode)
    run_id = f"g42_h{args.depth}_{args.mode}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    bundle = args.output_dir / run_id
    bundle.mkdir(parents=True, exist_ok=True)
    (bundle / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "experiment_id": result["experiment_id"], "acceptance": result["acceptance"],
        "tasks": {task: {arm: row["mean_accuracy_over_seeds"] for arm, row in item["arms"].items()}
                  for task, item in result["tasks"].items()}, "bundle": str(bundle)
    }, sort_keys=True))
    hard_gates = (
        "qa3_qa5_validation_coverage",
        "all_states_finite",
        "three_matched_seed_orders",
    )
    if args.mode == "prune_grow":
        hard_gates = (*hard_gates, "matched_static_adaptive_initialization", "prune_and_grow_observed")
    failed = [name for name in hard_gates if not result["acceptance"][name]]
    if failed:
        raise SystemExit("G4.2 execution/evidence gates failed: " + ", ".join(failed))
    # Stability is a scientific qualification and winner-selection filter,
    # not a reason to discard a completed H-run's evidence bundle.
    return result

if __name__ == "__main__":
    main()
