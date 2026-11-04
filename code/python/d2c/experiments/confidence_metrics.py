"""High-assurance confidence, calibration and selective-prediction diagnostics.

All metrics are computed from the complete class-probability vector. Confidence
scores are descriptive diagnostics; they are never used to select a model on
the held-out test split. Bootstrap intervals resample stories within a run and
must not be mistaken for between-seed uncertainty.
"""
from __future__ import annotations

import math
import random
from collections import Counter
from typing import Sequence


def _wilson(correct: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total <= 0:
        return None
    p = correct / total
    z2 = z * z
    den = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / den
    radius = z * math.sqrt(p * (1.0 - p) / total + z2 / (4.0 * total * total)) / den
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _percentile_interval(values: list[float]) -> list[float] | None:
    if not values:
        return None
    values.sort()
    lo = int(0.025 * (len(values) - 1))
    hi = int(0.975 * (len(values) - 1))
    return [values[lo], values[hi]]


def _extract_rows(
    predictions: Sequence[dict],
    class_labels: Sequence[str] | None,
) -> tuple[list[str], list[list[float]], list[int]]:
    if not predictions:
        return list(class_labels or ()), [], []
    labels = list(class_labels or ())
    if not labels:
        first = predictions[0]
        if isinstance(first.get("probabilities"), dict):
            labels = list(first["probabilities"])
        else:
            raise ValueError("class_labels are required when predictions have no probability mapping")
    if len(set(labels)) != len(labels) or not labels:
        raise ValueError("class_labels must be non-empty and unique")
    vectors: list[list[float]] = []
    targets: list[int] = []
    for row_index, row in enumerate(predictions):
        raw = row.get("probability_vector")
        if raw is None:
            mapping = row.get("probabilities")
            if not isinstance(mapping, dict):
                raise ValueError(f"prediction {row_index} has no probability vector")
            raw = [mapping.get(label, 0.0) for label in labels]
        if len(raw) != len(labels):
            raise ValueError(f"prediction {row_index} probability width does not match labels")
        vector = [float(value) for value in raw]
        if any(not math.isfinite(value) or value < 0.0 for value in vector):
            raise ValueError(f"prediction {row_index} has invalid probabilities")
        mass = sum(vector)
        if not math.isfinite(mass) or mass <= 0.0:
            raise ValueError(f"prediction {row_index} has zero probability mass")
        vectors.append([value / mass for value in vector])
        target = str(row.get("target"))
        if target not in labels:
            raise ValueError(f"prediction {row_index} target {target!r} is not in class_labels")
        targets.append(labels.index(target))
    return labels, vectors, targets


def _bin_ece(probabilities: list[float], outcomes: list[int], bins: int, *, equal_mass: bool) -> tuple[float, float, list[dict]]:
    n = len(probabilities)
    if n == 0:
        return 0.0, 0.0, []
    indexed = list(range(n))
    if equal_mass:
        indexed.sort(key=lambda i: probabilities[i])
        groups = [indexed[i * n // bins:(i + 1) * n // bins] for i in range(bins)]
        groups = [group for group in groups if group]
    else:
        groups = [[] for _ in range(bins)]
        for i, p in enumerate(probabilities):
            groups[min(bins - 1, int(p * bins))].append(i)
        groups = [group for group in groups if group]
    ece = mce = squared = 0.0
    details = []
    for group in groups:
        conf = sum(probabilities[i] for i in group) / len(group)
        acc = sum(outcomes[i] for i in group) / len(group)
        gap = abs(conf - acc)
        weight = len(group) / n
        ece += weight * gap
        squared += weight * gap * gap
        mce = max(mce, gap)
        details.append({
            "count": len(group),
            "mean_confidence": conf,
            "accuracy": acc,
            "absolute_gap": gap,
            "weight": weight,
        })
    return ece, mce, details


def _core(labels: list[str], vectors: list[list[float]], targets: list[int], bins: int) -> dict:
    n = len(vectors)
    if not n:
        return {}
    corrects: list[int] = []
    confidences: list[float] = []
    margins: list[float] = []
    entropies: list[float] = []
    nll: list[float] = []
    brier: list[float] = []
    predictions: list[int] = []
    for probs, target in zip(vectors, targets):
        order = sorted(range(len(probs)), key=lambda i: (-probs[i], i))
        pred = order[0]
        conf = probs[pred]
        correct = int(pred == target)
        predictions.append(pred)
        corrects.append(correct)
        confidences.append(conf)
        margins.append(conf - (probs[order[1]] if len(order) > 1 else 0.0))
        entropies.append(-sum(p * math.log(max(p, 1e-15)) for p in probs))
        nll.append(-math.log(max(probs[target], 1e-15)))
        brier.append(sum((p - (1.0 if i == target else 0.0)) ** 2 for i, p in enumerate(probs)))

    ece, mce, fixed_bins = _bin_ece(confidences, corrects, bins, equal_mass=False)
    adaptive_bins = min(bins, max(1, n // 10))
    adaptive_ece, _, adaptive_rows = _bin_ece(confidences, corrects, adaptive_bins, equal_mass=True)
    class_ece = {}
    for class_index, label in enumerate(labels):
        class_ece[label] = _bin_ece(
            [probs[class_index] for probs in vectors],
            [int(target == class_index) for target in targets],
            bins,
            equal_mass=False,
        )[0]
    sorted_indices = sorted(range(n), key=lambda i: (-confidences[i], i))
    curve = []
    area = 0.0
    previous_coverage, previous_risk = 0.0, 0.0
    cumulative_correct = 0
    curve_ranks = {max(1, round(n * pct)) for pct in (0.25, 0.50, 0.75, 0.90)}
    for rank, index in enumerate(sorted_indices, start=1):
        cumulative_correct += corrects[index]
        coverage = rank / n
        risk = 1.0 - cumulative_correct / rank
        area += (coverage - previous_coverage) * (risk + previous_risk) / 2.0
        previous_coverage, previous_risk = coverage, risk
        if rank in curve_ranks or rank == n:
            curve.append({"coverage": coverage, "risk": risk, "retained": rank})
    wrong_conf = [confidences[i] for i, c in enumerate(corrects) if not c]
    threshold_stats = {}
    for threshold in (0.70, 0.80, 0.90, 0.95):
        selected = [i for i, p in enumerate(confidences) if p >= threshold]
        threshold_stats[str(threshold)] = {
            "count": len(selected),
            "coverage": len(selected) / n,
            "accuracy": sum(corrects[i] for i in selected) / len(selected) if selected else None,
            "risk": 1.0 - sum(corrects[i] for i in selected) / len(selected) if selected else None,
        }
    return {
        "n": n,
        "correct": sum(corrects),
        "accuracy": sum(corrects) / n,
        "accuracy_wilson_95_ci": _wilson(sum(corrects), n),
        "mean_top_label_confidence": sum(confidences) / n,
        "confidence_accuracy_gap": sum(confidences) / n - sum(corrects) / n,
        "mean_confidence_on_errors": sum(wrong_conf) / len(wrong_conf) if wrong_conf else None,
        "overconfident_error_fraction": {
            str(threshold): sum(1 for p in wrong_conf if p >= threshold) / len(wrong_conf) if wrong_conf else None
            for threshold in (0.80, 0.90)
        },
        "mean_top_two_margin": sum(margins) / n,
        "mean_predictive_entropy_nats": sum(entropies) / n,
        "negative_log_likelihood": sum(nll) / n,
        "multiclass_brier_score": sum(brier) / n,
        "calibration": {
            "fixed_bin_top_label_ece": ece,
            "adaptive_equal_mass_top_label_ece": adaptive_ece,
            "max_calibration_error": mce,
            "rms_calibration_error": math.sqrt(sum(row["weight"] * row["absolute_gap"] ** 2 for row in fixed_bins)),
            "classwise_ece_macro": sum(class_ece.values()) / len(class_ece) if class_ece else 0.0,
            "classwise_ece_by_label": class_ece,
            "fixed_bins": fixed_bins,
            "adaptive_bins": adaptive_rows,
            "bin_count": bins,
            "adaptive_bin_count": adaptive_bins,
        },
        "selective_prediction": {
            "risk_coverage_curve": curve,
            "area_under_risk_coverage_curve": area,
            "confidence_thresholds": threshold_stats,
        },
    }


def summarize_confidence(
    predictions: Sequence[dict],
    class_labels: Sequence[str] | None = None,
    *,
    calibration_bins: int = 10,
    bootstrap_samples: int = 400,
    bootstrap_seed: int = 0,
) -> dict:
    """Summarize predictive confidence and story-bootstrap intervals.

    ECE is reported in both fixed-width and equal-mass forms because either can
    be unstable on its own. Classwise ECE exposes class-specific calibration
    failures. NLL and multiclass Brier are proper scoring rules. Selective
    risk/coverage shows whether confidence meaningfully ranks correctness.
    """
    if calibration_bins < 2:
        raise ValueError("calibration_bins must be at least 2")
    if bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be non-negative")
    labels, vectors, targets = _extract_rows(predictions, class_labels)
    if not vectors:
        return {
            "n": 0,
            "status": "no_predictions",
            "calibration_reliability": "not_available",
            "bootstrap": {"replicates_requested": bootstrap_samples, "intervals": {}},
        }
    result = _core(labels, vectors, targets, calibration_bins)
    n = len(vectors)
    result["status"] = "ok"
    result["calibration_reliability"] = (
        "low_sample_caution" if n < 100 else "descriptive_only_test_split"
    )
    intervals: dict[str, list[float] | None] = {}
    rng = random.Random(bootstrap_seed)
    names = (
        ("accuracy", "accuracy"),
        ("negative_log_likelihood", "negative_log_likelihood"),
        ("multiclass_brier_score", "multiclass_brier_score"),
        ("fixed_bin_top_label_ece", "calibration.fixed_bin_top_label_ece"),
        ("adaptive_equal_mass_top_label_ece", "calibration.adaptive_equal_mass_top_label_ece"),
        ("area_under_risk_coverage_curve", "selective_prediction.area_under_risk_coverage_curve"),
    )
    samples = {name: [] for name, _ in names}
    for _ in range(bootstrap_samples):
        indices = [rng.randrange(n) for _ in range(n)]
        boot = _core(
            labels,
            [vectors[i] for i in indices],
            [targets[i] for i in indices],
            calibration_bins,
        )
        for name, path in names:
            value = boot
            for part in path.split("."):
                value = value[part]
            samples[name].append(float(value))
    for name, _ in names:
        intervals[name] = _percentile_interval(samples[name])
    result["bootstrap"] = {
        "method": "nonparametric story-level percentile bootstrap",
        "replicates_requested": bootstrap_samples,
        "replicates_completed": bootstrap_samples,
        "seed": bootstrap_seed,
        "confidence_level": 0.95,
        "intervals_95": intervals,
        "interpretation": "Within-run story-sampling uncertainty only; inspect per-seed results for training-seed variability.",
    }
    return result
