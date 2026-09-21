from __future__ import annotations

import json
import math
from pathlib import Path

root = Path("/home/ubuntu/aberconics-framework/code/python/d2c/progress/babi_qa/sweep16")
files = {
    "full": "babi_qa_20260920T221538Z/summary.json",
    "no_slow": "babi_qa_20260920T221955Z/summary.json",
    "collapsed_gamma": "babi_qa_20260920T222415Z/summary.json",
}
results = {}
for variant, rel in files.items():
    data = json.loads((root / rel).read_text())
    results[variant] = {row["seed"]: row["accuracy"] for row in data["seed_results"]}

for variant in results:
    values = list(results[variant].values())
    mean = sum(values) / len(values)
    sd = math.sqrt(sum((x - mean) ** 2 for x in values) / (len(values) - 1))
    se = sd / math.sqrt(len(values))
    print(f"{variant}: mean={mean:.6f} sd={sd:.6f} se={se:.6f} 95ci=+/-{2.093024 * se:.6f}")

for a, b in [("no_slow", "full"), ("collapsed_gamma", "full"), ("full", "no_slow")]:
    diffs = [results[a][seed] - results[b][seed] for seed in sorted(results[a])]
    mean = sum(diffs) / len(diffs)
    sd = math.sqrt(sum((x - mean) ** 2 for x in diffs) / (len(diffs) - 1))
    se = sd / math.sqrt(len(diffs))
    t = mean / se if se else float("inf")
    ci = 2.093024 * se
    positive = sum(x > 0 for x in diffs)
    print(f"paired {a} - {b}: mean={mean:.6f} ci95=({mean-ci:.6f},{mean+ci:.6f}) t={t:.3f} positive={positive}/{len(diffs)}")
