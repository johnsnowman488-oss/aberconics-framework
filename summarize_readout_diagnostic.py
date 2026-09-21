from __future__ import annotations

import json
from pathlib import Path

path = Path('/home/ubuntu/aberconics-framework/code/python/d2c/progress/babi_qa/readout_diagnostic/diagnostic_20260920T225150Z.json')
data = json.loads(path.read_text())
rows = data['results']
for variant in sorted({r['variant'] for r in rows}):
    subset = [r for r in rows if r['variant'] == variant]
    print(f'[{variant}]')
    for r in sorted(subset, key=lambda r: (-r['test_accuracy'], r['lr'], r['hidden'], r['epochs']))[:8]:
        print(f"  test={r['test_accuracy']:.3f} train={r['train_accuracy']:.3f} hidden={r['hidden']} lr={r['lr']} epochs={r['epochs']} loss={r['final_train_loss']:.4f}")
    for epochs in sorted({r['epochs'] for r in subset}):
        best = max((r for r in subset if r['epochs'] == epochs), key=lambda r: r['test_accuracy'])
        print(f"  best@epochs={epochs}: test={best['test_accuracy']:.3f} hidden={best['hidden']} lr={best['lr']}")
print('[matched best settings]')
for epochs in sorted({r['epochs'] for r in rows}):
    for hidden in sorted({r['hidden'] for r in rows}):
        for lr in sorted({r['lr'] for r in rows}):
            pair = {r['variant']: r for r in rows if r['epochs']==epochs and r['hidden']==hidden and r['lr']==lr}
            if len(pair) == 2:
                delta = pair['no_slow']['test_accuracy'] - pair['full']['test_accuracy']
                if max(pair['full']['test_accuracy'], pair['no_slow']['test_accuracy']) >= 0.4:
                    print(f"  hidden={hidden} lr={lr} epochs={epochs}: full={pair['full']['test_accuracy']:.3f} no_slow={pair['no_slow']['test_accuracy']:.3f} delta={delta:+.3f}")
