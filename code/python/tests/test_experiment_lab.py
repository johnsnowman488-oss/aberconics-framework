from __future__ import annotations

import json
import threading
import uuid
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from d2c.experiment_lab import server as lab


def _write_profile(root, *, directory="data"):
    data_dir = root / directory
    data_dir.mkdir(parents=True)
    splits = {}
    for split in ("train", "valid", "test"):
        filename = f"babi_{split}.jsonl"
        (data_dir / filename).write_text('{"task": 2}\n', encoding="utf-8")
        splits[split] = filename
    (root / "profiles.json").write_text(json.dumps({
        "profiles": [{
            "id": "fixture",
            "name": "Fixture bAbI",
            "directory": directory,
            "splits": splits,
        }]
    }), encoding="utf-8")
    return data_dir


def test_experiment_catalogue_api_returns_protocol_defaults():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), lab.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        with urlopen(f"http://127.0.0.1:{httpd.server_port}/api/experiments") as response:
            catalogue = json.load(response)["experiments"]
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join()

    by_id = {entry["id"]: entry for entry in catalogue}
    assert by_id["g2"]["defaults"]["seeds"] == 5
    assert by_id["g2"]["defaults"]["bootstrap_samples"] == 400
    assert by_id["g1l"]["defaults"]["epochs"] == 3
    assert by_id["g1l"]["defaults"]["learning_rate"] == 0.02


def test_dataset_profiles_require_repository_relative_paths(tmp_path, monkeypatch):
    _write_profile(tmp_path)
    monkeypatch.setattr(lab, "ROOT", tmp_path)
    monkeypatch.setattr(lab, "PROFILE_REGISTRY", tmp_path / "profiles.json")

    profile = lab.dataset_profiles()["fixture"]
    assert lab.profile_status(profile)["ready_for_g1l"]
    assert set(lab.dataset_fingerprints(profile)) == {"train", "valid", "test"}

    registry = json.loads(lab.PROFILE_REGISTRY.read_text(encoding="utf-8"))
    registry["profiles"][0]["directory"] = "../outside"
    lab.PROFILE_REGISTRY.write_text(json.dumps(registry), encoding="utf-8")
    with pytest.raises(ValueError, match="stay inside the repository"):
        lab.dataset_profiles()

    registry["profiles"][0]["directory"] = "data"
    registry["profiles"][0]["splits"]["test"] = "custom_test.jsonl"
    lab.PROFILE_REGISTRY.write_text(json.dumps(registry), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical split filenames"):
        lab.dataset_profiles()


def test_launch_records_dataset_fingerprints_and_e0_data_directory(tmp_path, monkeypatch):
    data_dir = _write_profile(tmp_path)
    monkeypatch.setattr(lab, "ROOT", tmp_path)
    monkeypatch.setattr(lab, "PROFILE_REGISTRY", tmp_path / "profiles.json")
    monkeypatch.setattr(lab, "sys", type("Sys", (), {"executable": "python"})())

    command, config, fingerprints = lab.build_launch(
        {"experiment": "e0", "dataset_profile": "fixture"},
        tmp_path / "runs" / "e0",
    )

    assert command[command.index("--data-dir") + 1] == str(data_dir)
    assert command[command.index("--eval-split") + 1] == "test"
    assert config["seeds"] == 3
    assert config["dataset"]["id"] == "fixture"
    assert fingerprints["train"]["sha256"]


def test_quick_launch_records_effective_epoch_count(tmp_path, monkeypatch):
    _write_profile(tmp_path)
    monkeypatch.setattr(lab, "ROOT", tmp_path)
    monkeypatch.setattr(lab, "PROFILE_REGISTRY", tmp_path / "profiles.json")
    monkeypatch.setattr(lab, "sys", type("Sys", (), {"executable": "python"})())

    command, config, _ = lab.build_launch(
        {"experiment": "e0", "dataset_profile": "fixture", "quick": True},
        tmp_path / "runs" / "quick",
    )

    assert config["epochs"] == 2
    assert config["requested_epochs"] == 5
    assert command[command.index("--epochs") + 1] == "2"
    assert "--quick" in command


def test_g1l_pilot_defaults_do_not_claim_confirmation_seeds(tmp_path, monkeypatch):
    _write_profile(tmp_path)
    monkeypatch.setattr(lab, "ROOT", tmp_path)
    monkeypatch.setattr(lab, "PROFILE_REGISTRY", tmp_path / "profiles.json")
    monkeypatch.setattr(lab, "sys", type("Sys", (), {"executable": "python"})())

    command, config, _ = lab.build_launch(
        {"experiment": "g1l", "dataset_profile": "fixture"},
        tmp_path / "runs" / "g1l",
    )

    assert config["stage"] == "pilot"
    assert config["epochs"] == 3
    assert config["learning_rate"] == 0.02
    assert config["confirmation_seeds"] is None
    assert "--confirm-seeds" not in command
    assert command[command.index("--data-dir") + 1] == str(tmp_path / "data")


def test_unknown_dataset_profile_is_rejected_before_launch(tmp_path, monkeypatch):
    _write_profile(tmp_path)
    monkeypatch.setattr(lab, "ROOT", tmp_path)
    monkeypatch.setattr(lab, "PROFILE_REGISTRY", tmp_path / "profiles.json")

    with pytest.raises(ValueError, match="unknown dataset profile"):
        lab.build_launch(
            {"experiment": "e0", "dataset_profile": "unregistered"},
            tmp_path / "runs" / "invalid",
        )


def test_comparison_only_marks_exact_e0_ef0_protocols_as_matched():
    config = {
        "task": 2,
        "dim": 64,
        "seeds": 3,
        "epochs": 5,
        "learning_rate": 0.05,
        "feature_scale": 100.0,
        "readout": "linear",
        "evaluation_split": "test",
        "mode": "isolated",
        "variant": "full",
        "quick": False,
        "dataset": {"id": "babi_canonical"},
        "dataset_fingerprints": {"train": {"sha256": "abc"}},
    }
    e0 = {"experiment": "e0", "config": config}
    ef0 = {"experiment": "ef0", "config": config.copy()}
    assert lab._matched_ef0_pair(e0, ef0)[0]

    ef0["config"] = config | {"evaluation_split": "valid"}
    matched, reason = lab._matched_ef0_pair(e0, ef0)
    assert not matched
    assert "evaluation_split" in reason
    assert not lab._matched_ef0_pair(e0, {"experiment": "g1", "config": config})[0]


def test_paired_comparison_requires_matching_example_identity(tmp_path):
    config = {
        "task": 2,
        "dim": 64,
        "seeds": 1,
        "epochs": 5,
        "learning_rate": 0.05,
        "feature_scale": 100.0,
        "readout": "linear",
        "evaluation_split": "test",
        "mode": "isolated",
        "variant": "full",
        "quick": False,
        "dataset": {"id": "babi_canonical"},
        "dataset_fingerprints": {"train": {"sha256": "abc"}},
    }

    def job(experiment, story_index):
        run_dir = tmp_path / experiment
        bundle = run_dir / "bundle"
        bundle.mkdir(parents=True, exist_ok=True)
        prediction = {
            "story_index": story_index,
            "task": 2,
            "question": "Where is John?",
            "raw_passage": "John moved to the hallway.",
            "target": "hallway",
        }
        summary = {
            "seed_results": [{
                "seed": 0,
                "accuracy": 0.75 if experiment == "e0" else 1.0,
                "total": 1,
                "predictions": [prediction],
            }]
        }
        (bundle / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return {
            "experiment": experiment,
            "status": "completed",
            "config": config,
            "directory": str(run_dir),
        }

    e0 = job("e0", 0)
    ef0 = job("ef0", 0)
    paired = lab._paired_seed_comparison(ef0, e0)
    assert paired["comparable"]
    assert paired["mean_delta"] == 0.25

    mismatched = job("ef0", 1)
    paired = lab._paired_seed_comparison(e0, mismatched)
    assert not paired["comparable"]
    assert "evaluation examples differ" in paired["reason"].lower()


def test_api_rejects_unregistered_profile(tmp_path, monkeypatch):
    _write_profile(tmp_path)
    monkeypatch.setattr(lab, "ROOT", tmp_path)
    monkeypatch.setattr(lab, "PROFILE_REGISTRY", tmp_path / "profiles.json")
    monkeypatch.setattr(lab, "RUNS", tmp_path / "runs")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), lab.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    request = Request(
        f"http://127.0.0.1:{httpd.server_port}/api/runs",
        data=json.dumps({
            "experiment": "e0",
            "dataset_profile": "not-registered",
        }).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with pytest.raises(HTTPError) as error:
            urlopen(request)
        assert error.value.code == 400
        assert json.load(error.value)["error"] == "unknown dataset profile"
        assert not lab.RUNS.exists()
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join()


@pytest.mark.parametrize("status", ["completed", "failed"])
def test_delete_saved_run_removes_only_its_run_directory(tmp_path, monkeypatch, status):
    runs = tmp_path / "runs"
    run_id = str(uuid.uuid4())
    run_dir = runs / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text("{}", encoding="utf-8")
    (tmp_path / "keep.txt").write_text("keep", encoding="utf-8")
    monkeypatch.setattr(lab, "RUNS", runs)
    monkeypatch.setattr(lab, "JOBS", {
        run_id: {"run_id": run_id, "status": status, "directory": str(run_dir)},
    })
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), lab.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    request = Request(f"http://127.0.0.1:{httpd.server_port}/api/runs/{run_id}", method="DELETE")
    try:
        with urlopen(request) as response:
            assert response.status == 200
            assert json.load(response)["deleted"] == run_id
        assert not run_dir.exists()
        assert (tmp_path / "keep.txt").read_text(encoding="utf-8") == "keep"
        assert run_id not in lab.JOBS
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join()


def test_delete_running_run_is_rejected_without_removing_files(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    run_id = str(uuid.uuid4())
    run_dir = runs / run_id
    run_dir.mkdir(parents=True)
    marker = run_dir / "stdout.log"
    marker.write_text("still running", encoding="utf-8")
    monkeypatch.setattr(lab, "RUNS", runs)
    monkeypatch.setattr(lab, "JOBS", {
        run_id: {"run_id": run_id, "status": "running", "directory": str(run_dir)},
    })
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), lab.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    request = Request(f"http://127.0.0.1:{httpd.server_port}/api/runs/{run_id}", method="DELETE")
    try:
        with pytest.raises(HTTPError) as error:
            urlopen(request)
        assert error.value.code == 409
        assert json.load(error.value)["error"] == "a running experiment cannot be deleted"
        assert marker.read_text(encoding="utf-8") == "still running"
        assert run_id in lab.JOBS
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join()


def test_delete_refuses_run_directory_outside_lab_store(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep", encoding="utf-8")
    run_id = str(uuid.uuid4())
    monkeypatch.setattr(lab, "RUNS", runs)
    monkeypatch.setattr(lab, "JOBS", {
        run_id: {"run_id": run_id, "status": "completed", "directory": str(outside)},
    })
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), lab.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    request = Request(f"http://127.0.0.1:{httpd.server_port}/api/runs/{run_id}", method="DELETE")
    try:
        with pytest.raises(HTTPError) as error:
            urlopen(request)
        assert error.value.code == 400
        assert json.load(error.value)["error"] == "run directory is outside the Lab run store"
        assert (outside / "keep.txt").read_text(encoding="utf-8") == "keep"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join()


def test_g3_lab_launch_registers_two_arm_frozen_protocol(tmp_path, monkeypatch):
    _write_profile(tmp_path)
    monkeypatch.setattr(lab, "ROOT", tmp_path)
    monkeypatch.setattr(lab, "PROFILE_REGISTRY", tmp_path / "profiles.json")
    monkeypatch.setattr(lab, "sys", type("Sys", (), {"executable": "python"})())

    command, config, fingerprints = lab.build_launch(
        {"experiment": "g3", "dataset_profile": "fixture"},
        tmp_path / "runs" / "g3",
    )

    assert command[command.index("-m") + 1] == "d2c.experiments.babi_g3"
    assert command[command.index("--tasks") + 1] == "3,5"
    assert command[command.index("--epochs") + 1] == "3"
    assert command[command.index("--feature-scale") + 1] == "10.0"
    assert config["tasks"] == [3, 5]
    assert config["seeds"] == 20
    assert config["epochs"] == 3
    assert config["feature_scale"] == 10.0
    assert config["arms"] == ["flat_abi", "hier_bu_td"]
    assert config["f0_spine_audit"] is True
    assert fingerprints["train"]["sha256"]


def test_g3_lab_rejects_unregistered_or_unmatched_protocol_values(tmp_path, monkeypatch):
    _write_profile(tmp_path)
    monkeypatch.setattr(lab, "ROOT", tmp_path)
    monkeypatch.setattr(lab, "PROFILE_REGISTRY", tmp_path / "profiles.json")
    with pytest.raises(ValueError, match="G3 requires QA3"):
        lab.build_launch(
            {"experiment": "g3", "dataset_profile": "fixture", "feature_scale": 100},
            tmp_path / "runs" / "g3",
        )
    with pytest.raises(ValueError, match="G3 requires QA3"):
        lab.build_launch(
            {"experiment": "g3", "dataset_profile": "fixture", "tasks": [3]},
            tmp_path / "runs" / "g3",
        )
