"""G0 QA2 oracle ladder and controlled entity-binding counterfactuals."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from ..digital.babi import BabiDataset

_MOVEMENT = re.compile(
    r"^(?P<person>[A-Za-z]+)\s+(?:went|moved|travelled|journeyed)(?:\s+back)?"
    r"\s+to\s+the\s+(?P<place>[A-Za-z]+)\.?$", re.IGNORECASE
)
_PICKUP = re.compile(
    r"^(?P<person>[A-Za-z]+)\s+(?:got|grabbed|took|picked\s+up)"
    r"\s+the\s+(?P<object>[A-Za-z]+)\s+there\.?$", re.IGNORECASE
)
_DROPOFF = re.compile(
    r"^(?P<person>[A-Za-z]+)\s+(?:left|discarded|dropped|put\s+down)"
    r"\s+the\s+(?P<object>[A-Za-z]+)(?:\s+there)?\.?$", re.IGNORECASE
)
_QUERY = re.compile(r"^where\s+is\s+the\s+(?P<object>[A-Za-z]+)\??$", re.IGNORECASE)


def parse_event(statement: str) -> dict | None:
    for kind, pattern in (("move", _MOVEMENT), ("pickup", _PICKUP), ("drop", _DROPOFF)):
        match = pattern.match(statement.strip())
        if match:
            values = {key: value.lower() for key, value in match.groupdict().items()}
            return {"kind": kind, **values, "source": statement}
    return None


def replay_world(statements: list[str]) -> dict:
    locations: dict[str, str] = {}
    holders: dict[str, str] = {}
    object_locations: dict[str, str] = {}
    events, unparsed, binding_errors, binding_warnings = [], [], [], []
    for index, statement in enumerate(statements):
        event = parse_event(statement)
        if event is None:
            unparsed.append({"index": index, "statement": statement})
            continue
        event["index"] = index
        person = event["person"]
        if event["kind"] == "move":
            locations[person] = event["place"]
        elif event["kind"] == "pickup":
            # "there" can refer to an initial location not repeated in this
            # story. Bind the object to its carrier now; a later move resolves it.
            holders[event["object"]] = person
            object_locations.pop(event["object"], None)
            if person not in locations:
                binding_warnings.append({
                    "index": index, "reason": "carrier_location_not_yet_observed",
                    "person": person, "object": event["object"],
                })
        else:
            if person not in locations:
                binding_errors.append({
                    "index": index, "reason": "drop_actor_location_unknown",
                    "person": person, "object": event["object"],
                })
                holders.pop(event["object"], None)
            else:
                object_locations[event["object"]] = locations[person]
                holders.pop(event["object"], None)
        events.append(event)
    return {
        "events": events, "unparsed_statements": unparsed,
        "binding_errors": binding_errors, "binding_warnings": binding_warnings,
        "person_locations": locations, "object_holders": holders,
        "object_locations": object_locations,
    }


def answer_from_world(question: str, world: dict) -> tuple[str | None, str | None]:
    match = _QUERY.match(question.strip())
    if not match:
        return None, None
    obj = match.group("object").lower()
    if obj in world["object_holders"]:
        holder = world["object_holders"][obj]
        return world["person_locations"].get(holder), obj
    return world["object_locations"].get(obj), obj


def evaluate_story(story) -> dict:
    world = replay_world(story.statements)
    location, queried_object = answer_from_world(story.question, world)
    parsed = len(world["events"])
    total_statements = len(story.statements)
    binding_ok = (
        not world["binding_errors"] and queried_object is not None
        and (queried_object in world["object_holders"]
             or queried_object in world["object_locations"])
    )
    return {
        "question": story.question, "target": story.answer,
        "oracle_prediction": location, "correct": location == story.answer,
        "queried_object": queried_object, "event_count": parsed,
        "statement_count": total_statements,
        "event_coverage": parsed / total_statements if total_statements else 1.0,
        "entity_binding_ok": binding_ok, "state_resolved": location is not None,
        "world": world,
    }


def controlled_entity_binding_tests() -> dict:
    base = [
        "Mary moved to the kitchen.", "John travelled to the bedroom.",
        "Mary got the football there.", "Mary went to the hallway.",
    ]
    tests = [
        ("base", base, "Where is the football?", "hallway"),
        ("pickup_before_observed_location", [
            "Mary got the milk there.", "John moved to the bedroom.",
            "Mary travelled to the hallway.",
        ], "Where is the milk?", "hallway"),
        ("consistent_rename", [
            "Alice moved to the kitchen.", "John travelled to the bedroom.",
            "Alice got the ball there.", "Alice went to the hallway.",
        ], "Where is the ball?", "hallway"),
        ("irrelevant_entity_move", base + ["Sandra journeyed to the office."],
         "Where is the football?", "hallway"),
        ("irrelevant_fact_reordered", [
            "John travelled to the bedroom.", "Mary moved to the kitchen.",
            "Mary got the football there.", "Mary went to the hallway.",
        ], "Where is the football?", "hallway"),
        ("supporting_move_changed", [
            "Mary moved to the kitchen.", "John travelled to the bedroom.",
            "Mary got the football there.", "Mary went to the garden.",
        ], "Where is the football?", "garden"),
        ("last_support_removed", base[:3], "Where is the football?", "kitchen"),
        ("drop_releases_object", base + [
            "Mary put down the football.", "Mary went to the garden.",
        ], "Where is the football?", "hallway"),
    ]
    results = []
    for name, statements, question, expected in tests:
        actual, _ = answer_from_world(question, replay_world(statements))
        results.append({
            "name": name, "expected": expected, "actual": actual,
            "passed": actual == expected,
        })
    return {"passed": all(row["passed"] for row in results), "tests": results}


def run_probe(data_dir: Path, *, task: int = 2, split: str = "test",
              max_stories: int = 0) -> dict:
    dataset = BabiDataset.from_jsonl(data_dir / f"babi_{split}.jsonl").filter_task(task)
    stories = dataset.stories[:max_stories] if max_stories > 0 else dataset.stories
    details = [evaluate_story(story) for story in stories]
    total = len(details)
    total_statements = sum(row["statement_count"] for row in details)
    parsed_events = sum(row["event_count"] for row in details)
    resolved = sum(row["state_resolved"] for row in details)
    bindings = sum(row["entity_binding_ok"] for row in details)
    correct = sum(row["correct"] for row in details)
    return {
        "experiment": "g0_qa2_oracle_ladder", "task": task,
        "split": split, "stories": total,
        "levels": {
            "L0_raw_baseline": {
                "description": "Reference accuracy from the E0/E-F0 token-level readout; recorded in the paired Lab comparison.",
                "accuracy": None,
            },
            "L1_event_segmentation": {
                "description": "Recognize sentence-level move, pickup and drop events.",
                "recognized_event_statements": parsed_events,
                "total_statements": total_statements,
                "event_coverage": parsed_events / total_statements if total_statements else 0.0,
                "unparsed_statements": sum(
                    len(row["world"]["unparsed_statements"]) for row in details
                ),
            },
            "L2_entity_binding": {
                "description": "Bind person and object identities, including an object acquired before the carrier's location is observed.",
                "stories_with_valid_bindings": bindings,
                "binding_rate": bindings / total if total else 0.0,
                "binding_errors": sum(
                    len(row["world"]["binding_errors"]) for row in details
                ),
                "deferred_location_bindings": sum(
                    len(row["world"]["binding_warnings"]) for row in details
                ),
            },
            "L3_relational_temporal_state": {
                "description": "Apply object pickup/drop transitions and propagate an object's location while held by a moving person.",
                "resolved_world_states": resolved,
                "state_resolution_rate": resolved / total if total else 0.0,
                "accuracy": correct / total if total else 0.0,
            },
            "L4_oracle_answer_readout": {
                "description": "Resolve Where-is(object) by looking up the final holder location or stored object location.",
                "accuracy": correct / total if total else 0.0,
                "correct": correct, "total": total,
            },
        },
        "controlled_entity_binding": controlled_entity_binding_tests(),
        "failure_examples": [row for row in details if not row["correct"]][:100],
        "per_example": details,
    }


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Run G0 QA2 oracle and entity-binding probes.")
    parser.add_argument("--task", type=int, default=2)
    parser.add_argument("--split", choices=("train", "valid", "test"), default="test")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "progress" / "babi_data")
    parser.add_argument("--max-stories", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    result = run_probe(args.data_dir, task=args.task, split=args.split, max_stories=args.max_stories)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["levels"], indent=2))
    print("controlled_entity_binding:", result["controlled_entity_binding"]["passed"])
    return result


if __name__ == "__main__":
    main()
