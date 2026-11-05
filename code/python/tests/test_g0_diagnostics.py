from d2c.digital.babi import BabiStory
from d2c.experiments.g0_diagnostics import build_prediction_record, summarize_predictions


def test_prediction_bundle_preserves_evidence_and_probabilities() -> None:
    story = BabiStory(
        statements=["Mary moved to the kitchen.", "Mary got the football there."],
        question="Where is the football?", answer="kitchen", task=2,
        raw_passage="Mary moved to the kitchen.\nMary got the football there.\n",
    )
    record = build_prediction_record(story, 4, ["garden", "kitchen"], [0.03, 0.97])
    assert record["story_index"] == 4
    assert record["statements"] == story.statements
    assert record["probabilities"]["kitchen"] == 0.97
    assert record["correct"] is True
    assert record["failure_category"] == "correct"


def test_calibration_and_failure_categories_capture_overconfident_errors() -> None:
    story = BabiStory(
        statements=["Mary moved to the kitchen."],
        question="Where is Mary?", answer="garden", task=1,
    )
    record = build_prediction_record(story, 0, ["garden", "kitchen"], [0.01, 0.99])
    report = summarize_predictions([record], ["garden", "kitchen"])
    assert report["accuracy"] == 0.0
    assert report["confidence"]["overconfident_error_count"] == 1
    assert report["failure_categories"] == {"overconfident_wrong": 1}
    assert report["calibration"]["expected_calibration_error"] > 0.9
    assert report["confusion_matrix"]["garden"]["kitchen"] == 1
