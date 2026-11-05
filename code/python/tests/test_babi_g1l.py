"""Fast L0 sanity checks and G1-L result-contract tests."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from d2c.experiments.babi_g1l import (
    G1LConfig,
    LearnedRelationalBinder,
    ParsedStory,
    _run_l0,
    _evaluate,
    run_g1l_experiment,
    save_g1l_bundle,
)
from d2c.digital.relational import RelationalEvent
from d2c.experiment_lab.server import write_manifest


def _vocabulary():
    return {
        "people": ["John", "Mary"],
        "objects": ["football", "milk"],
        "locations": ["garden", "hallway", "kitchen", "office"],
        "answers": ["garden", "hallway", "kitchen", "office"],
    }


def test_l0_binding_and_gradient_sanity() -> None:
    result = _run_l0(16, 0.02)
    assert result["passed"], result
    assert all(result["checks"].values())
    assert result["final_loss"] < result["initial_loss"]


def test_holder_attention_is_normalized_and_names_target_holder() -> None:
    model = LearnedRelationalBinder(**_vocabulary(), dim=16, seed=3)
    story = ParsedStory((
        RelationalEvent("move", "John", location="garden"),
        RelationalEvent("move", "Mary", location="kitchen"),
        RelationalEvent("pickup", "Mary", object="football"),
        RelationalEvent("move", "Mary", location="hallway"),
    ), "Where is the football?", "football", "hallway")
    result = model._forward(story)
    assert result["is_held"] is True
    assert result["target_holder"] == "Mary"
    assert set(result["attention"]) == {"John", "Mary"}
    assert abs(sum(result["attention"].values()) - 1.0) < 1e-10


def test_evaluation_reports_held_and_fixed_relation_metrics_separately() -> None:
    model = LearnedRelationalBinder(**_vocabulary(), dim=16, seed=5)
    held = ParsedStory((
        RelationalEvent("move", "John", location="garden"),
        RelationalEvent("move", "Mary", location="kitchen"),
        RelationalEvent("pickup", "Mary", object="football"),
        RelationalEvent("move", "Mary", location="hallway"),
    ), "Where is the football?", "football", "hallway")
    fixed = ParsedStory((
        RelationalEvent("move", "John", location="office"),
        RelationalEvent("pickup", "John", object="milk"),
        RelationalEvent("move", "John", location="garden"),
        RelationalEvent("release", "John", object="milk"),
    ), "Where is the milk?", "milk", "garden")
    result = _evaluate(model, [held, fixed])
    assert result["relation_metrics"]["object_holder_to_person_location"]["total"] == 1
    assert result["relation_metrics"]["fixed_object_location"]["total"] == 1
    assert result["held_query_total"] == 1
    assert result["fixed_location_total"] == 1
    assert result["holder_retrieval_total"] == 1


def test_lab_manifest_writer_records_result_inventory(tmp_path: Path) -> None:
    run_dir = tmp_path / "lab-run"
    run_dir.mkdir()
    (run_dir / "config.json").write_text("{}", encoding="utf-8")
    write_manifest({
        "run_id": "manifest-test", "directory": str(run_dir),
        "experiment": "g1l", "status": "running", "created_at": "now",
        "config": {},
    })
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "config.json" in manifest["result_files"]
    assert manifest["run_id"] == "manifest-test"


def test_release_without_known_actor_location_uses_unknown_state() -> None:
    model = LearnedRelationalBinder(**_vocabulary(), dim=16, seed=13)
    story = ParsedStory((
        RelationalEvent("move", "John", location="office"),
        RelationalEvent("pickup", "John", object="football"),
        RelationalEvent("release", "Mary", object="football"),
    ), "Where is the football?", "football", "office")
    result = model._forward(story)
    assert result["is_held"] is False
    assert result["attention"] == {}
    assert np.isfinite(result["probabilities"]).all()


def test_release_changes_query_to_fixed_location_memory() -> None:
    model = LearnedRelationalBinder(**_vocabulary(), dim=16, seed=7)
    story = ParsedStory((
        RelationalEvent("move", "Mary", location="kitchen"),
        RelationalEvent("pickup", "Mary", object="football"),
        RelationalEvent("move", "Mary", location="hallway"),
        RelationalEvent("release", "Mary", object="football"),
        RelationalEvent("move", "Mary", location="garden"),
    ), "Where is the football?", "football", "hallway")
    result = model._forward(story)
    assert result["is_held"] is False
    assert result["target_holder"] is None
    assert result["attention"] == {}


def test_supervised_update_is_finite_and_changes_parameters() -> None:
    model = LearnedRelationalBinder(**_vocabulary(), dim=16, seed=11)
    story = ParsedStory((
        RelationalEvent("move", "Mary", location="kitchen"),
        RelationalEvent("pickup", "Mary", object="football"),
        RelationalEvent("move", "Mary", location="hallway"),
    ), "Where is the football?", "football", "hallway")
    before = model.p["W_output"].copy()
    loss = model._train_one(story, 0.02)
    assert np.isfinite(loss)
    assert not np.array_equal(before, model.p["W_output"])
    assert all(np.isfinite(value).all() for value in model.p.values())


def _write_split(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _rows():
    return [
        {
            "passage": (
                "Mary moved to the kitchen.\n"
                "Mary got the football there.\n"
                "Mary moved to the hallway.\n"
            ),
            "question": "Where is the football?",
            "answer": "hallway",
            "task": 2,
        },
        {
            "passage": (
                "John moved to the office.\n"
                "John got the milk there.\n"
                "John moved to the garden.\n"
            ),
            "question": "Where is the milk?",
            "answer": "garden",
            "task": 2,
        },
        {
            "passage": (
                "Mary moved to the kitchen.\n"
                "Mary got the milk there.\n"
                "Mary moved to the garden.\n"
            ),
            "question": "Where is the milk?",
            "answer": "garden",
            "task": 2,
        },
        {
            "passage": (
                "John moved to the office.\n"
                "John got the football there.\n"
                "John moved to the hallway.\n"
            ),
            "question": "Where is the football?",
            "answer": "hallway",
            "task": 2,
        },
    ]


def test_pilot_bundle_has_manifest_and_explicit_stage_gate(tmp_path: Path) -> None:
    rows = _rows()
    _write_split(tmp_path / "babi_train.jsonl", rows)
    _write_split(tmp_path / "babi_valid.jsonl", rows)
    _write_split(tmp_path / "babi_test.jsonl", rows)
    result = run_g1l_experiment(
        config=G1LConfig(dim=16, epochs=1, train_limit=4, eval_limit=4, confirm_seeds=1),
        data_dir=tmp_path,
        stage="pilot",
        exercise_f0=False,
    )
    assert result["stages"]["L0"]["passed"]
    assert "L1" in result["stages"] and "L2" in result["stages"]
    assert result["stages"]["L2"]["status"] == "skipped"
    assert result["preprocessing"]["split_loads"] == 3
    paths = save_g1l_bundle(result, output_dir=tmp_path / "out")
    bundle = Path(paths["bundle_dir"])
    for name in ("summary.json", "summary.md", "metadata.json", "manifest.json",
                 "config.json", "per_example.csv", "per_example.jsonl"):
        assert (bundle / name).is_file()
    saved = json.loads((bundle / "summary.json").read_text())
    assert saved["experiment"] == "babi_g1l_learned_relational_binding"
    metadata = json.loads((bundle / "manifest.json").read_text())
    assert metadata["created_at"] <= metadata["completed_at"]
    assert metadata["dataset_fingerprints"]["train_sha256"]
