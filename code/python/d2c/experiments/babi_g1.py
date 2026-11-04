"""G1: explicit relational-state QA2 baseline beside the F0/SOE stream.

The parser and transitions are deterministic and inspectable. This is a
symbolic relational-state control, not evidence of learned entity binding.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..digital.babi import BabiDataset, build_babi_answer_set, build_babi_bridge, build_babi_memory_config, build_babi_vocabulary
from ..digital.relational import RelationalWorldState, parse_babi_question, parse_babi_statement
from ..runtime.digital_runtime import DigitalRuntime, DigitalRuntimeConfig

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DATA_DIR = _REPO_ROOT / "code" / "python" / "d2c" / "progress" / "babi_data"


@dataclass(frozen=True, slots=True)
class BabiG1Config:
    task: int = 2
    dim: int = 64
    variant: str = "full"
    trace_capacity: int = 20_000

    def __post_init__(self) -> None:
        if self.task != 2:
            raise ValueError("G1 currently supports bAbI QA2 (task 2) only")
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if self.variant not in {"full", "no_slow", "collapsed_gamma"}:
            raise ValueError(f"unknown memory variant: {self.variant!r}")
        if self.trace_capacity <= 0:
            raise ValueError("trace_capacity must be positive")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _wilson_interval(correct: int, total: int) -> list[float]:
    if total <= 0:
        return [0.0, 0.0]
    z = 1.959963984540054
    p = correct / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / total + z2 / (4.0 * total * total)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def run_babi_g1_experiment(*, config: BabiG1Config | None = None, data_dir: Path = _DATA_DIR, quick: bool = False, max_test_stories: int = 0) -> dict[str, Any]:
    cfg = config or BabiG1Config()
    data_dir = Path(data_dir)
    train_path, test_path = data_dir / "babi_train.jsonl", data_dir / "babi_test.jsonl"
    if not train_path.is_file() or not test_path.is_file():
        raise FileNotFoundError("G1 requires babi_train.jsonl and babi_test.jsonl")
    train_ds = BabiDataset.from_jsonl(train_path).filter_task(cfg.task)
    test_ds = BabiDataset.from_jsonl(test_path).filter_task(cfg.task)
    if not train_ds.stories or not test_ds.stories:
        raise ValueError("G1 requires non-empty QA2 train and test splits")
    if quick:
        train_ds.stories, test_ds.stories = train_ds.stories[:20], test_ds.stories[:20]
    elif max_test_stories > 0:
        test_ds.stories = test_ds.stories[:max_test_stories]

    vocabulary = build_babi_vocabulary(train_ds.stories + test_ds.stories)
    answer_set = build_babi_answer_set(train_ds.stories)
    bridge = build_babi_bridge(vocabulary, dim=cfg.dim)
    memory_config = build_babi_memory_config(variant=cfg.variant)
    runtime = DigitalRuntime(
        memory_config=memory_config,
        state_dim=cfg.dim,
        config=DigitalRuntimeConfig(trace_capacity=cfg.trace_capacity),
        session_id="babi-g1-qa2-relational-state",
    )

    predictions: list[dict[str, Any]] = []
    statement_count = parsed_count = move_count = pickup_count = release_count = 0
    resolved_count = correct_count = 0
    actor_ids: set[str] = set()
    object_ids: set[str] = set()
    failure_counts: dict[str, int] = {}

    for index, story in enumerate(test_ds.stories):
        episode_id = f"g1-test-{index}"
        runtime.reset_memory()
        runtime.episode_count += 1
        first_step = runtime.step_count
        checks_before = runtime.stability_checks
        violations_before = runtime.stability_violations
        phase_counts: dict[str, int] = {}
        world = RelationalWorldState()
        parse_errors: list[dict[str, Any]] = []

        # The F0/SOE stream consumes every token in order. After each sentence,
        # the explicit relation state applies the corresponding event.
        for statement_index, statement in enumerate(story.statements):
            for token in statement.split():
                record = runtime.step(
                    bridge.forcing_for_token(token),
                    token=token,
                    episode_id=episode_id,
                    metadata={"statement_index": statement_index},
                )
                phase_counts[record.phase] = phase_counts.get(record.phase, 0) + 1
            statement_count += 1
            try:
                event = parse_babi_statement(statement)
                world.apply(event)
                parsed_count += 1
                actor_ids.add(event.actor)
                if event.object is not None:
                    object_ids.add(event.object)
                if event.kind == "move":
                    move_count += 1
                elif event.kind == "pickup":
                    pickup_count += 1
                elif event.kind == "release":
                    release_count += 1
            except ValueError as exc:
                parse_errors.append({"statement_index": statement_index, "statement": statement, "error": str(exc)})

        for token in story.question.split():
            record = runtime.step(
                bridge.forcing_for_token(token), token=token, episode_id=episode_id,
                metadata={"query": True},
            )
            phase_counts[record.phase] = phase_counts.get(record.phase, 0) + 1
        record = runtime.step(
            bridge.zero_forcing(), token=None, episode_id=episode_id,
            metadata={"readout_step": True},
        )
        phase_counts[record.phase] = phase_counts.get(record.phase, 0) + 1
        last_step = runtime.step_count - 1

        try:
            query_object = parse_babi_question(story.question)
            object_ids.add(query_object)
            prediction = runtime.evaluate(
                lambda: world.resolve_object(query_object),
                reason=f"G1 relational lookup for {query_object}",
            )
        except ValueError as exc:
            query_object, prediction = None, None
            parse_errors.append({"statement_index": -1, "statement": story.question, "error": str(exc)})

        resolved = prediction is not None
        correct = prediction == story.answer
        resolved_count += int(resolved)
        correct_count += int(correct)
        if parse_errors:
            key = "unsupported_input_pattern"
        elif not resolved:
            key = "unresolved_object_location"
        elif not correct:
            key = "wrong_resolved_location"
        else:
            key = None
        if key:
            failure_counts[key] = failure_counts.get(key, 0) + 1

        predictions.append({
            "example_index": index,
            "episode_id": episode_id,
            "question": story.question,
            "query_object": query_object,
            "target": story.answer,
            "prediction": prediction if prediction is not None else "<insufficient_evidence>",
            "resolved": resolved,
            "correct": correct,
            "statement_count": len(story.statements),
            "parsed_statement_count": world.event_count,
            "parse_coverage": world.event_count / len(story.statements) if story.statements else 1.0,
            "parse_errors": parse_errors,
            "world_state": world.to_mapping(),
            "event_log": world.event_log,
            "f0_runtime": {
                "steps": last_step - first_step + 1,
                "stability_checks": runtime.stability_checks - checks_before,
                "stability_violations": runtime.stability_violations - violations_before,
                "phase_counts": phase_counts,
            },
        })

    total = len(test_ds.stories)
    accuracy = correct_count / total if total else 0.0
    return {
        "schema_version": 1,
        "experiment": "babi_g1_explicit_relational_state",
        "task": cfg.task,
        "method": "deterministic event parser + explicit relational state transitions + F0/SOE companion stream",
        "learning_performed": False,
        "claim_boundary": "Inspectable symbolic relational-state diagnostic; this is not a learned-binding result.",
        "configuration": {
            **asdict(cfg), "mode": "isolated",
            "readout": "relational_state_resolution", "training_epochs": 0, "seed": None,
        },
        "dataset": {
            "train_stories": len(train_ds.stories),
            "test_stories": total,
            "train_sha256": _sha256(train_path),
            "test_sha256": _sha256(test_path),
            "task": cfg.task,
        },
        "train_stories": len(train_ds.stories),
        "test_stories": total,
        "vocabulary_size": vocabulary.size,
        "answer_count": len(answer_set),
        "correct": correct_count,
        "mean_accuracy": accuracy,
        "accuracy": accuracy,
        "standard_error": math.sqrt(accuracy * (1.0 - accuracy) / total) if total else None,
        "wilson_95_ci": _wilson_interval(correct_count, total),
        "parse_coverage": parsed_count / statement_count if statement_count else 1.0,
        "parsed_statements": parsed_count,
        "total_statements": statement_count,
        "query_resolution_coverage": resolved_count / total if total else 1.0,
        "resolved_queries": resolved_count,
        "event_counts": {
            "moves": move_count, "pickups": pickup_count, "releases": release_count,
            "actors_seen": len(actor_ids), "objects_seen": len(object_ids),
        },
        "failure_counts": failure_counts,
        "runtime": runtime.runtime_summary(),
        "per_example": predictions,
        "failure_examples": [row for row in predictions if not row["correct"]][:100],
        "notes": [
            "The existing full SOE kernel and F0 runtime process every QA2 story in isolated mode.",
            "Relational updates are explicit and inspectable: move, pickup, release, then query-time holder-to-location resolution.",
            "No supervised fitting or seed averaging is used by G1; its accuracy is a deterministic held-out diagnostic.",
            "The experiment covers the current QA2 sentence templates and does not establish general natural-language entity resolution.",
        ],
    }


def save_babi_g1_bundle(result: dict[str, Any], *, output_dir: Path) -> dict[str, str]:
    run_id = f"babi_g1_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    bundle_dir = Path(output_dir) / run_id
    bundle_dir.mkdir(parents=True, exist_ok=True)
    summary_path = bundle_dir / "summary.json"
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    csv_path = bundle_dir / "per_example.csv"
    fields = ["example_index", "question", "query_object", "target", "prediction", "resolved", "correct", "statement_count", "parsed_statement_count", "parse_coverage"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in result["per_example"]:
            writer.writerow({key: row.get(key) for key in fields})

    jsonl_path = bundle_dir / "per_example.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in result["per_example"]:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    low, high = result["wilson_95_ci"]
    report_path = bundle_dir / "summary.md"
    report_path.write_text(
        "# G1 QA2 explicit relational-state report\n\n"
        f"- Accuracy: {result['correct']}/{result['test_stories']} ({result['mean_accuracy']:.4%})\n"
        f"- 95% Wilson interval: {low:.4%}–{high:.4%}\n"
        f"- Parse coverage: {result['parse_coverage']:.4%} ({result['parsed_statements']}/{result['total_statements']})\n"
        f"- Query resolution: {result['query_resolution_coverage']:.4%}\n"
        f"- Event counts: {json.dumps(result['event_counts'], sort_keys=True)}\n"
        f"- F0 stability violations: {result['runtime']['stability_violations']}\n\n"
        f"**Interpretation boundary:** {result['claim_boundary']}\n\n"
        "See per_example.csv for answer-level results and per_example.jsonl for full world states and event ledgers.\n",
        encoding="utf-8",
    )
    now = datetime.now(timezone.utc).isoformat()
    metadata = {
        "schema_version": 1, "run_id": run_id,
        "experiment": result["experiment"],
        "created_at": now, "completed_at": now,
        "status": "completed", "exit_code": 0,
        "code_revision": os.environ.get("GITHUB_SHA"),
        "configuration": result["configuration"],
        "dataset": result["dataset"],
        "result_files": ["summary.json", "summary.md", "per_example.csv", "per_example.jsonl"],
    }
    (bundle_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (bundle_dir / "config.json").write_text(json.dumps(result["configuration"], indent=2), encoding="utf-8")
    return {
        "bundle_dir": str(bundle_dir), "summary_json": str(summary_path),
        "per_example_csv": str(csv_path), "summary_md": str(report_path),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run G1 explicit relational-state bAbI QA2 diagnostic.")
    parser.add_argument("--task", type=int, default=2)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--variant", choices=["full", "no_slow", "collapsed_gamma"], default="full")
    parser.add_argument("--trace-capacity", type=int, default=20_000)
    parser.add_argument("--data-dir", type=Path, default=_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-test-stories", type=int, default=0)
    parser.add_argument("--quick", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> dict[str, str]:
    args = build_arg_parser().parse_args(argv)
    config = BabiG1Config(task=args.task, dim=args.dim, variant=args.variant, trace_capacity=args.trace_capacity)
    result = run_babi_g1_experiment(
        config=config, data_dir=args.data_dir,
        quick=args.quick, max_test_stories=args.max_test_stories,
    )
    paths = save_babi_g1_bundle(result, output_dir=args.output_dir)
    print(f"G1 QA2 accuracy: {result['correct']}/{result['test_stories']} = {result['mean_accuracy']:.4%}")
    print(f"statement parse coverage: {result['parsed_statements']}/{result['total_statements']} = {result['parse_coverage']:.4%}")
    print(f"query resolution coverage: {result['resolved_queries']}/{result['test_stories']} = {result['query_resolution_coverage']:.4%}")
    print(f"F0 runtime steps={result['runtime']['step_count']} stability_violations={result['runtime']['stability_violations']}")
    print(f"bundle={paths['bundle_dir']}")
    return paths


if __name__ == "__main__":
    main()
