from __future__ import annotations

import json
import math
from pathlib import Path

path = Path('/home/ubuntu/aberconics-framework/code/python/d2c/progress/babi_qa/readout_diagnostic_confirm/diagnostic_20260920T225447Z.json')
data = json.loads(path.read_text())
by = {}
for row in data['results']:
    by.setdefault(row['variant'], {})[row['seed']] = row['test_accuracy']
for variant, vals in by.items():
    xs = list(vals.values())
    mean = sum(xs) / len(xs)
    sd = math.sqrt(sum((x-mean)**2 for x in xs)/(len(xs)-1))
    se = sd / math.sqrt(len(xs))
    print(f'{variant}: mean={mean:.4f} sd={sd:.4f} se={se:.4f} ci95=+/-{2.776445*se:.4f}')
diffs = [by['no_slow'][s] - by['full'][s] for s in sorted(by['full'])]
mean = sum(diffs)/len(diffs)
sd = math.sqrt(sum((x-mean)**2 for x in diffs)/(len(diffs)-1))
se = sd/math.sqrt(len(diffs))
print(f'paired no_slow-full: mean={mean:.4f} ci95=({mean-2.776445*se:.4f},{mean+2.776445*se:.4f}) t={mean/se:.3f} positive={sum(x>0 for x in diffs)}/{len(diffs)}')
