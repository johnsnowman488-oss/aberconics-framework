"""G2 fixtures, pairing and confidence-evidence tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from d2c.experiments.babi_g2 import (
    G2Config,
    _paired_interval,
    replay_relational_fixture,
    run_fixture_suite,
)
from d2c.experiments.confidence_metrics import summarize_confidence

_FIXTURE = Path(__file__).resolve().parents[1] / "d2c" / "experiments" / "fixtures" / "g2_relational_cases.json"


def test_g2_fixture_suite_covers_all_expected_relation_families() -> None:
    result = run_fixture_suite(_FIXTURE)
    assert result["total"] == 12
    assert result["failed"] == 0
    assert result["passed"] == 12
    assert result["by_category"]["three_relation_chain_transfer"]["passed"] == 1
    assert result["by_category"]["explicit_coreference_alias"]["passed"] == 2


def test_g2_transfer_rejects_wrong_source_holder() -> None:
    bad = {
        "aliases": {},
        "events": [
            {"type": "move", "actor": "mary", "location": "kitchen"},
            {"type": "pickup", "actor": "mary", "object": "book"},
            {"type": "transfer", "object": "book", "from": "john", "to": "mary"},
        ],
        "query_object": "book",
    }
    with pytest.raises(ValueError, match="expected holder"):
        replay_relational_fixture(bad)


def test_g2_config_restricts_tasks_and_requires_multiple_seeds() -> None:
    with pytest.raises(ValueError, match="QA3"):
        G2Config(tasks=(2,))
    with pytest.raises(ValueError, match="seed count"):
        G2Config(seeds=1)


def test_paired_interval_reports_between_seed_uncertainty() -> None:
    report = _paired_interval([0.02, 0.01, 0.00, 0.03, -0.01])
    assert report["n"] == 5
    assert report["mean"] == pytest.approx(0.01)
    assert report["ci95_two_sided_t"][0] < report["mean"] < report["ci95_two_sided_t"][1]
    assert report["positive_seed_differences"] == 3
    assert report["negative_seed_differences"] == 1


def test_confidence_metrics_report_proper_scores_calibration_and_selective_risk() -> None:
    labels = ["kitchen", "garden"]
    rows = [
        {"target": "kitchen", "prediction": "kitchen", "probability_vector": [0.90, 0.10]},
        {"target": "garden", "prediction": "kitchen", "probability_vector": [0.95, 0.05]},
        {"target": "garden", "prediction": "garden", "probability_vector": [0.15, 0.85]},
        {"target": "kitchen", "prediction": "kitchen", "probability_vector": [0.80, 0.20]},
    ]
    report = summarize_confidence(rows, labels, bootstrap_samples=80, bootstrap_seed=3)
    assert report["accuracy"] == pytest.approx(0.75)
    assert report["accuracy_wilson_95_ci"][0] < 0.75
    assert report["negative_log_likelihood"] > 0
    assert report["multiclass_brier_score"] > 0
    assert report["calibration"]["fixed_bin_top_label_ece"] >= 0
    assert report["calibration"]["adaptive_equal_mass_top_label_ece"] >= 0
    assert set(report["calibration"]["classwise_ece_by_label"]) == set(labels)
    assert report["selective_prediction"]["area_under_risk_coverage_curve"] >= 0
    assert report["overconfident_error_fraction"]["0.8"] == pytest.approx(1.0)
    assert report["bootstrap"]["replicates_completed"] == 80
    assert report["bootstrap"]["intervals_95"]["negative_log_likelihood"] is not None


def test_confidence_bootstrap_is_reproducible_and_flags_small_samples() -> None:
    rows = [
        {"target": "a", "probability_vector": [0.8, 0.2]},
        {"target": "b", "probability_vector": [0.3, 0.7]},
    ]
    first = summarize_confidence(rows, ["a", "b"], bootstrap_samples=40, bootstrap_seed=11)
    second = summarize_confidence(rows, ["a", "b"], bootstrap_samples=40, bootstrap_seed=11)
    assert first["bootstrap"] == second["bootstrap"]
    assert first["calibration_reliability"] == "low_sample_caution"


def test_confidence_rejects_invalid_probability_vectors() -> None:
    with pytest.raises(ValueError, match="invalid probabilities"):
        summarize_confidence(
            [{"target": "a", "probability_vector": [float("nan"), 1.0]}],
            ["a", "b"],
            bootstrap_samples=0,
        )
    with pytest.raises(ValueError, match="zero probability mass"):
        summarize_confidence(
            [{"target": "a", "probability_vector": [0.0, 0.0]}],
            ["a", "b"],
            bootstrap_samples=0,
        )


def test_g2_fixture_json_is_valid_and_stable() -> None:
    payload = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    ids = [row["id"] for row in payload["cases"]]
    assert len(ids) == len(set(ids))
    assert all(row["query_object"] for row in payload["cases"])
