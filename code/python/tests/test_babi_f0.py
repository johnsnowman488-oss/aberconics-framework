"""Dataset-independent integration tests for the bAbI F0 experiment runner."""

from __future__ import annotations

import json

from d2c.experiments.babi_f0 import BabiF0Config, run_babi_f0_experiment
from d2c.runtime.digital_runtime import DigitalRuntimeConfig
from d2c.runtime.phase_controller import DigitalPhaseConfig


def write_split(path, records) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in records),
        encoding="utf-8",
    )


def test_babi_f0_runner_executes_train_and_test_without_external_dataset(tmp_path) -> None:
    train = [
        {"passage": "Mary moved to the kitchen.\n", "question": "Where is Mary?", "answer": "kitchen", "task": 1},
        {"passage": "John moved to the hallway.\n", "question": "Where is John?", "answer": "hallway", "task": 1},
        {"passage": "Mary moved to the garden.\n", "question": "Where is Mary?", "answer": "garden", "task": 1},
        {"passage": "John moved to the office.\n", "question": "Where is John?", "answer": "office", "task": 1},
    ]
    test = [
        {"passage": "Mary moved to the garden.\n", "question": "Where is Mary?", "answer": "garden", "task": 1},
        {"passage": "John moved to the hallway.\n", "question": "Where is John?", "answer": "hallway", "task": 1},
    ]
    write_split(tmp_path / "babi_train.jsonl", train)
    write_split(tmp_path / "babi_test.jsonl", test)

    result = run_babi_f0_experiment(
        BabiF0Config(
            task=1,
            dim=8,
            seeds=1,
            epochs=1,
            learning_rate=0.001,
            runtime=DigitalRuntimeConfig(
                trace_capacity=64,
                phase=DigitalPhaseConfig(
                    warmup_steps=2,
                    explore_steps=4,
                    consolidate_steps=2,
                    evaluation_interval=8,
                ),
            ),
        ),
        data_dir=tmp_path,
    )

    assert result["experiment"] == "babi_f0_runtime_integrated"
    assert result["train_stories"] == 4
    assert result["test_stories"] == 2
    assert 0.0 <= result["mean_accuracy"] <= 1.0
    seed = result["seed_results"][0]
    assert seed["train_runtime"]["step_count"] > 0
    assert seed["test_runtime"]["step_count"] > 0
    assert seed["test_runtime"]["stability_violations"] == 0
    assert seed["test_runtime"]["phase_counts"]


def test_continuous_mode_keeps_one_test_memory_continuity(tmp_path) -> None:
    train = [
        {"passage": "Mary moved to the kitchen.\n", "question": "Where is Mary?", "answer": "kitchen", "task": 1},
        {"passage": "John moved to the hallway.\n", "question": "Where is John?", "answer": "hallway", "task": 1},
    ]
    test = [
        {"passage": "Mary moved to the kitchen.\n", "question": "Where is Mary?", "answer": "kitchen", "task": 1},
        {"passage": "John moved to the hallway.\n", "question": "Where is John?", "answer": "hallway", "task": 1},
    ]
    write_split(tmp_path / "babi_train.jsonl", train)
    write_split(tmp_path / "babi_test.jsonl", test)

    result = run_babi_f0_experiment(
        BabiF0Config(
            dim=8,
            seeds=1,
            epochs=1,
            mode="continuous",
            runtime=DigitalRuntimeConfig(
                trace_capacity=64,
                phase=DigitalPhaseConfig(
                    warmup_steps=0,
                    explore_steps=20,
                    consolidate_steps=2,
                    evaluation_interval=0,
                ),
            ),
        ),
        data_dir=tmp_path,
    )
    test_runtime = result["seed_results"][0]["test_runtime"]
    assert test_runtime["episode_count"] == 2
    assert test_runtime["continuity_id"] == 0


def test_isolated_f0_runner_reproduces_legacy_e0_accuracy_on_same_fixture(tmp_path) -> None:
    """F0 orchestration must not change the frozen E0 numerical baseline."""
    from d2c.experiments.babi_qa import BabiQAConfig, run_babi_qa_experiment

    train = [
        {"passage": "Mary moved to the kitchen.\\n", "question": "Where is Mary?", "answer": "kitchen", "task": 1},
        {"passage": "John moved to the hallway.\\n", "question": "Where is John?", "answer": "hallway", "task": 1},
        {"passage": "Mary moved to the garden.\\n", "question": "Where is Mary?", "answer": "garden", "task": 1},
        {"passage": "John moved to the office.\\n", "question": "Where is John?", "answer": "office", "task": 1},
    ]
    test = [
        {"passage": "Mary moved to the garden.\\n", "question": "Where is Mary?", "answer": "garden", "task": 1},
        {"passage": "John moved to the hallway.\\n", "question": "Where is John?", "answer": "hallway", "task": 1},
    ]
    write_split(tmp_path / "babi_train.jsonl", train)
    write_split(tmp_path / "babi_test.jsonl", test)

    baseline = run_babi_qa_experiment(
        BabiQAConfig(
            task=1,
            variant="full",
            dim=8,
            seeds=1,
            learning_rate=0.001,
            readout_type="linear",
        ),
        data_dir=tmp_path,
        quick=True,
    )
    f0 = run_babi_f0_experiment(
        BabiF0Config(
            task=1,
            variant="full",
            dim=8,
            seeds=1,
            epochs=5,
            learning_rate=0.001,
            readout_type="linear",
            mode="isolated",
            runtime=DigitalRuntimeConfig(
                trace_capacity=64,
                phase=DigitalPhaseConfig(
                    warmup_steps=2,
                    explore_steps=4,
                    consolidate_steps=2,
                    evaluation_interval=8,
                ),
            ),
        ),
        data_dir=tmp_path,
        quick=True,
    )
    assert f0["mean_accuracy"] == baseline["mean_accuracy"]
    assert f0["seed_results"][0]["correct"] == baseline["seed_results"][0]["correct"]
