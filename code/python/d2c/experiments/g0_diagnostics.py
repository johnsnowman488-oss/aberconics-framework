"""Shared per-example, calibration, answer-distribution and training diagnostics for G0."""
from __future__ import annotations

import math
from collections import Counter
from typing import Sequence


def build_prediction_record(story, story_index: int, answer_set: Sequence[str],
                            probabilities: Sequence[float]) -> dict:
    if len(probabilities) != len(answer_set):
        raise ValueError("probability vector must match answer_set")
    probs = [float(value) for value in probabilities]
    if not all(math.isfinite(value) and value >= 0.0 for value in probs):
        raise ValueError("probabilities must be finite and non-negative")
    total = sum(probs)
    if total <= 0.0:
        raise ValueError("probability vector must have positive mass")
    probs = [value / total for value in probs]
    index = max(range(len(probs)), key=probs.__getitem__)
    target = story.answer
    prediction = answer_set[index]
    confidence = probs[index]
    ordered = sorted(probs, reverse=True)
    correct = prediction == target
    failure_category = (
        "correct" if correct else
        "overconfident_wrong" if confidence >= 0.90 else
        "confident_wrong" if confidence >= 0.70 else
        "low_confidence_wrong"
    )
    return {
        "story_index": int(story_index),
        "task": int(story.task),
        "question": story.question,
        "statements": list(story.statements),
        "raw_passage": story.raw_passage,
        "target": target,
        "prediction": prediction,
        "correct": correct,
        "confidence": confidence,
        "top_two_margin": confidence - (ordered[1] if len(ordered) > 1 else 0.0),
        "probability_vector": probs,
        "probabilities": {label: probs[i] for i, label in enumerate(answer_set)},
        "failure_category": failure_category,
    }


def summarize_predictions(rows: Sequence[dict], answer_set: Sequence[str],
                          *, calibration_bins: int = 10) -> dict:
    labels = list(answer_set)
    total = len(rows)
    if not total:
        return {
            "total": 0, "correct": 0, "accuracy": 0.0,
            "target_distribution": {}, "predicted_distribution": {},
            "per_class": {}, "confusion_matrix": {}, "calibration": {},
            "failure_categories": {},
        }
    target_counts = Counter(str(row["target"]) for row in rows)
    predicted_counts = Counter(str(row["prediction"]) for row in rows)
    confusion = {label: {candidate: 0 for candidate in labels} for label in labels}
    per_class = {}
    correct = sum(bool(row["correct"]) for row in rows)
    confidences = [float(row["confidence"]) for row in rows]
    margins = [float(row.get("top_two_margin", 0.0)) for row in rows]
    wrong = [row for row in rows if not row["correct"]]
    nll_values = []
    brier_values = []
    calibration = [{"count": 0, "confidence_sum": 0.0, "accuracy_sum": 0}
                   for _ in range(calibration_bins)]
    for row in rows:
        target = str(row["target"])
        prediction = str(row["prediction"])
        if target in confusion and prediction in confusion[target]:
            confusion[target][prediction] += 1
        p = float(row["confidence"])
        bin_index = min(calibration_bins - 1, int(p * calibration_bins))
        calibration[bin_index]["count"] += 1
        calibration[bin_index]["confidence_sum"] += p
        calibration[bin_index]["accuracy_sum"] += int(bool(row["correct"]))
        distribution = row.get("probability_vector")
        if distribution is not None:
            by_label = {label: float(distribution[i]) for i, label in enumerate(labels)}
        else:
            by_label = {label: float(row.get("probabilities", {}).get(label, 0.0))
                        for label in labels}
        target_probability = max(by_label.get(target, 0.0), 1e-12)
        nll_values.append(-math.log(target_probability))
        brier_values.append(sum(
            (by_label.get(label, 0.0) - (1.0 if label == target else 0.0)) ** 2
            for label in labels
        ))
    for label in labels:
        tp = confusion[label][label]
        support = target_counts[label]
        predicted = predicted_counts[label]
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "support": support, "predicted": predicted, "correct": tp,
            "precision": precision, "recall": recall, "f1": f1,
        }
    calibration_rows = []
    ece = 0.0
    for index, bucket in enumerate(calibration):
        count = bucket["count"]
        mean_confidence = bucket["confidence_sum"] / count if count else 0.0
        mean_accuracy = bucket["accuracy_sum"] / count if count else 0.0
        gap = abs(mean_confidence - mean_accuracy) if count else 0.0
        ece += count / total * gap
        calibration_rows.append({
            "lower": index / calibration_bins,
            "upper": (index + 1) / calibration_bins,
            "count": count,
            "mean_confidence": mean_confidence,
            "accuracy": mean_accuracy,
            "absolute_gap": gap,
        })
    failure_counts = Counter(str(row.get("failure_category", "unspecified"))
                             for row in rows if not row["correct"])
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total,
        "target_distribution": dict(sorted(target_counts.items())),
        "predicted_distribution": dict(sorted(predicted_counts.items())),
        "per_class": per_class,
        "confusion_matrix": confusion,
        "confidence": {
            "mean": sum(confidences) / total,
            "mean_on_errors": sum(float(row["confidence"]) for row in wrong) / len(wrong) if wrong else 0.0,
            "overconfident_error_count": sum(float(row["confidence"]) >= 0.90 for row in wrong),
            "overconfident_error_fraction": (
                sum(float(row["confidence"]) >= 0.90 for row in wrong) / len(wrong)
                if wrong else 0.0
            ),
            "mean_top_two_margin": sum(margins) / total,
        },
        "calibration": {
            "expected_calibration_error": ece,
            "negative_log_likelihood": sum(nll_values) / total,
            "multiclass_brier_score": sum(brier_values) / total,
            "bins": calibration_rows,
        },
        "failure_categories": dict(sorted(failure_counts.items())),
    }


def summarize_training_losses(epoch_losses: Sequence[float]) -> dict:
    values = [float(value) for value in epoch_losses]
    finite = [value for value in values if math.isfinite(value)]
    return {
        "epoch_mean_loss": values,
        "all_finite": len(finite) == len(values),
        "initial_loss": values[0] if values else None,
        "final_loss": values[-1] if values else None,
        "final_to_initial_ratio": (
            values[-1] / values[0] if values and values[0] > 0.0 else None
        ),
        "max_loss": max(finite) if finite else None,
        "non_finite_epochs": len(values) - len(finite),
    }
