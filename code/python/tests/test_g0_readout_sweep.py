import json

from d2c.experiments.g0_readout_sweep import run_sweep


def _records():
    return [
        {"passage": "Mary moved to the kitchen.\nMary got the football there.\nMary went to the hallway.\n",
         "question": "Where is the football?", "answer": "hallway", "task": 2},
        {"passage": "John travelled to the bedroom.\nJohn got the milk there.\nJohn went to the office.\n",
         "question": "Where is the milk?", "answer": "office", "task": 2},
        {"passage": "Mary moved to the garden.\nMary got the apple there.\nMary went to the kitchen.\n",
         "question": "Where is the apple?", "answer": "kitchen", "task": 2},
        {"passage": "John moved to the office.\nJohn got the book there.\nJohn went to the hallway.\n",
         "question": "Where is the book?", "answer": "hallway", "task": 2},
    ]


def _write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_staged_readout_sweep_selects_on_validation_and_confirms_on_test(tmp_path):
    rows = _records()
    _write(tmp_path / "babi_train.jsonl", rows)
    _write(tmp_path / "babi_valid.jsonl", rows[:2])
    _write(tmp_path / "babi_test.jsonl", rows[2:])

    result = run_sweep(
        task=2, dim=8, seeds=1, epochs=1,
        scales=(1.0, 10.0), learning_rates=(0.001, 0.01, 0.05),
        data_dir=tmp_path, output_dir=tmp_path / "results",
    )

    assert result["selected_configuration"]["selection_split"] == "valid"
    assert result["test_confirmation"]["e0"]["evaluation_split"] == "test"
    assert result["test_confirmation"]["ef0"]["evaluation_split"] == "test"
    assert len(result["validation_cells"]) >= 4
    assert result["test_confirmation"]["ef0"]["seed_results"][0]["spine_exercise"]["status"] == "passed"
    assert (tmp_path / "results" / result["bundle_dir"].split("/")[-1] / "summary.json").exists()
