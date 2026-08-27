"""Human-readable reports for digital D2C experiments."""

from __future__ import annotations

from typing import Mapping


def format_symbolic_induction_report(result: Mapping[str, object]) -> str:
    summary = result["summary"]
    lines = [
        "Symbolic Induction Report",
        "=========================",
        f"Experiment: {result['experiment_name']}",
        f"Variant:    {result['variant']}",
        f"Pairs:      {result['pair_count']}",
        f"Distractors: {result.get('distractor_pair_count', 0)}",
        f"Binding:    {result.get('binding_mode', 'token_code')}",
        f"Gap:        {result['gap']}",
        f"Seeds:      {result['seed_count']}",
        "",
        "Metrics",
        "-------",
        f"Accuracy:   {summary['accuracy']:.3f} ({summary['correct']}/{summary['count']})",
        f"Mean Score: {summary['mean_score']:.6f}",
        "",
        "Memory",
        "------",
        f"Gamma:           {result['memory_config']['gamma']}",
        f"Weights:         {result['memory_config']['w']}",
        f"D_eff:           {result['memory_diagnostics']['deff']:.3f}",
        f"Stability Ratio: {result['memory_diagnostics']['stability_ratio']:.3f}",
    ]
    if "minimum_binding_score" in result:
        lines.append(f"Min. Evidence: {result['minimum_binding_score']:.6f}")
    if result.get("binding_noise_std", 0.0):
        lines.append(f"Binding Noise: {result['binding_noise_std']:.4f}")
    if result.get("baseline"):
        baseline = result["baseline"]
        lines.extend([
            "",
            "Baseline",
            "--------",
            f"Window:   {baseline['window']}",
            f"Accuracy: {baseline['accuracy']:.3f} ({baseline['correct']}/{baseline['count']})",
        ])
    return "\n".join(lines)
