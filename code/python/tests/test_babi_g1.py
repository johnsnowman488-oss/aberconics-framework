"""G1 relational-state parser, transition, and QA2 runner tests."""
from __future__ import annotations
import json
from pathlib import Path
import pytest

from d2c.digital.relational import RelationalWorldState, parse_babi_question, parse_babi_statement
from d2c.experiments.babi_g1 import BabiG1Config, run_babi_g1_experiment, save_babi_g1_bundle


@pytest.mark.parametrize(("statement", "kind"), [
    ("Mary moved to the kitchen.", "move"),
    ("Mary went to the garden.", "move"),
    ("John went back to the office.", "move"),
    ("Sandra travelled to the hallway.", "move"),
    ("Daniel journeyed to the bedroom.", "move"),
    ("Mary got the football there.", "pickup"),
    ("Mary grabbed the football there.", "pickup"),
    ("Mary took the football there.", "pickup"),
    ("Mary picked up the football there.", "pickup"),
    ("Mary left the football.", "release"),
    ("Mary left the football there.", "release"),
    ("Mary dropped the football.", "release"),
    ("Mary dropped the football there.", "release"),
    ("Mary discarded the football.", "release"),
    ("Mary discarded the football there.", "release"),
    ("Mary put down the football.", "release"),
    ("Mary put down the football there.", "release"),
])
def test_qa2_statement_templates_parse(statement: str, kind: str) -> None:
    assert parse_babi_statement(statement).kind == kind


def test_object_location_follows_holder_across_movements() -> None:
    state = RelationalWorldState()
    for statement in ("Mary moved to the kitchen.", "Mary got the football there.", "Mary moved to the hallway."):
        state.apply(parse_babi_statement(statement))
    assert state.object_holders["football"] == "Mary"
    assert state.resolve_object("football") == "hallway"


def test_release_and_handoff_update_object_binding() -> None:
    state = RelationalWorldState()
    statements = (
        "John moved to the office.", "John got the milk there.",
        "John moved to the garden.", "John dropped the milk.",
        "Mary moved to the bathroom.", "Mary picked up the milk there.",
    )
    for statement in statements:
        state.apply(parse_babi_statement(statement))
    assert state.object_holders["milk"] == "Mary"
    assert state.resolve_object("milk") == "bathroom"


def test_question_parser_rejects_other_task_shapes() -> None:
    assert parse_babi_question("Where is the football?") == "football"
    with pytest.raises(ValueError, match="unrecognized"):
        parse_babi_question("Where is Mary?")


def _write_split(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def test_g1_runner_records_relational_results_and_f0_runtime(tmp_path: Path) -> None:
    train = [
        {"passage": "Mary moved to the kitchen.\nMary got the football there.\n", "question": "Where is the football?", "answer": "kitchen", "task": 2},
        {"passage": "John moved to the office.\nJohn got the milk there.\n", "question": "Where is the milk?", "answer": "office", "task": 2},
    ]
    test = [
        {"passage": "Mary moved to the kitchen.\nMary got the football there.\nMary moved to the hallway.\n", "question": "Where is the football?", "answer": "hallway", "task": 2},
        {"passage": "John moved to the office.\nJohn got the milk there.\nJohn moved to the garden.\n", "question": "Where is the milk?", "answer": "garden", "task": 2},
    ]
    _write_split(tmp_path / "babi_train.jsonl", train)
    _write_split(tmp_path / "babi_test.jsonl", test)
    result = run_babi_g1_experiment(config=BabiG1Config(dim=8, trace_capacity=256), data_dir=tmp_path)

    assert result["experiment"] == "babi_g1_explicit_relational_state"
    assert result["learning_performed"] is False
    assert result["correct"] == 2 and result["test_stories"] == 2
    assert result["mean_accuracy"] == 1.0
    assert result["parse_coverage"] == 1.0
    assert result["query_resolution_coverage"] == 1.0
    assert result["runtime"]["episode_count"] == 2
    assert result["runtime"]["step_count"] > 0
    assert result["runtime"]["stability_violations"] == 0
    assert result["per_example"][0]["world_state"]["objects"]["football"] == {"held_by": "Mary", "location": "hallway"}

    paths = save_babi_g1_bundle(result, output_dir=tmp_path / "results")
    bundle = Path(paths["bundle_dir"])
    for name in ("summary.json", "metadata.json", "summary.md", "per_example.csv", "per_example.jsonl"):
        assert (bundle / name).is_file()
    assert len((bundle / "per_example.jsonl").read_text().splitlines()) == 2


def test_g1_is_scoped_to_qa2() -> None:
    with pytest.raises(ValueError, match="QA2"):
        BabiG1Config(task=1)


_DATA_DIR = Path(__file__).resolve().parents[1] / "d2c" / "progress" / "babi_data"
@pytest.mark.skipif(
    not (_DATA_DIR / "babi_train.jsonl").exists() or not (_DATA_DIR / "babi_test.jsonl").exists(),
    reason="downloaded bAbI dataset is not present",
)
def test_full_qa2_splits_are_covered_by_the_event_parser() -> None:
    from d2c.digital.babi import BabiDataset
    for split in ("train", "valid", "test"):
        path = _DATA_DIR / f"babi_{split}.jsonl"
        if path.exists():
            dataset = BabiDataset.from_jsonl(path).filter_task(2)
            for story in dataset.stories:
                for statement in story.statements:
                    parse_babi_statement(statement)
                parse_babi_question(story.question)
