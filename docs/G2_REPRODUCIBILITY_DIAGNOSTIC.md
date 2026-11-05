# G2 Reproducibility and Confidence Diagnostic

**Status:** completed, validation-only diagnostic  
**Experiment ID:** `G2-REPRO-DIAGNOSTIC`  
**Repository branch:** `g2/reproducibility-diagnostic` (diagnostic implementation; isolated)  
**Record branch:** `g2/diagnostic-record`  
**Final CI run:** [42041847620](https://github.com/johnsnowman488-oss/aberconics-framework/actions/runs/42041847620)  
**Artifact ID:** `13421922107`  
**Artifact digest:** `sha256:99114dd5b79608ca99c0d85d775641f4e07ce820b08690295808fdb8f383d73f`

## Purpose and claim boundary

This diagnostic investigates the QA3 run-to-run variation observed in the earlier G2 validation readout and characterizes remaining confidence errors. It is not a replacement for the registered G2 E0/E-F0 benchmark, does not confirm a production readout setting, and does not close G2.

Only bAbI training and validation splits were opened. The original G2 held-out test split was not opened. All metrics below are validation diagnostics.

## Protocol

- Tasks: QA3 (task 3) and QA5 (task 5).
- Matched seeds: 0–4.
- Independent repeats per task/scale/seed: 2.
- Epochs: exactly 3; this is the provisional validation checkpoint.
- Dimension: 64.
- Learning rate: 0.05.
- Feature scales: 10 (leading candidate) and 100 (high-scale sensitivity reference).
- Memory path: Python bAbI bridge and NumPy readout.
- C++ core: built unchanged as CI regression context; this diagnostic's actual feature/readout computation does not invoke the C++ hierarchy ABI or C++ readout.
- Numerical controls in pinned jobs: Python 3.11.18, single-thread numerical libraries, OpenBLAS Haswell target, and NumPy CPU dispatch restrictions.

Train and validation split SHA-256 fingerprints:

- Train: `faa5d639178ff7a6a082e6766918949b3a154cc567c5b7c903389ff6e11232a6`
- Validation: `259236ebfb13ce337bde221e90777e98ea7258e5e08a80d130ccd3ab00159952`

## Reproducibility findings

The initial repeated-run investigation found that data, memory-state, query, feature, and initial-weight fingerprints matched, but QA3 training outputs diverged between independent CI jobs. The jobs used different Python patch versions and different detected CPU/SIMD feature sets.

After pinning the numerical environment, two independent CI jobs matched exactly across all 120 epoch-level records (20 task × scale × seed conditions, two repeats, three epochs): trained weights, validation logits, and predictions were identical, with zero maximum deltas and zero prediction disagreement. Feature construction and initial weights also matched. The final parity run added training-order hashes and confirmed repeat-level training-order identity inside the pinned run.

**Interpretation:** numerical-kernel/CPU-dispatch variation is the leading explanation for the earlier cross-job drift. The combined environment controls removed the observed drift; the individual causal contribution of Python pinning, OpenBLAS target pinning, and NumPy dispatch restrictions has not been isolated.

## Epoch-3 validation metrics

Values below are means across the five matched seeds, using repeat 1. The second repeat reproduced exactly within each pinned job.

| Task | Feature scale | Accuracy | NLL | Brier | ECE-15 | Mean confidence | High-confidence error fraction (confidence ≥0.90) | p95 max |logit| |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| QA3 | 10 | 0.280 | 6.016 | 1.246 | 0.594 | 0.870 | 0.704 | 21.27 |
| QA3 | 100 | 0.228 | 21.272 | 1.543 | 0.771 | 0.999 | 0.771 | 2349.63 |
| QA5 | 10 | 0.638 | 0.917 | 0.484 | 0.172 | 0.795 | 0.103 | 11.28 |
| QA5 | 100 | 0.612 | 10.192 | 0.772 | 0.386 | 0.998 | 0.385 | 1033.00 |

Feature scale 10 substantially improves NLL, Brier, ECE, and logit magnitude relative to scale 100 on both tasks. Accuracy changes are modest and task-dependent. QA3 remains poorly calibrated and weak in accuracy; QA5 is materially stronger but retains non-zero high-confidence risk.

The aggregate QA3 high-confidence error fraction is 0.704, not a statement that every answer class or every seed has that error rate. Per-seed and per-answer-class details are retained in the raw summary.

## Decisions

1. Retain feature scale 10 as the leading candidate, not a confirmed production setting.
2. Use three epochs as the provisional validation checkpoint for controlled follow-up experiments.
3. Preserve the pinned numerical environment for controlled CI comparisons.
4. Investigate QA3 feature/readout behavior and remaining confidence errors before any locked G2 evaluation.
5. Keep the original G2 held-out test split sealed. Do not use it for tuning, calibration, or checkpoint selection.

## G3 implications

G2 establishes that the current token-readout pathway is highly sensitive to feature scaling and that QA3 location retrieval remains weak. G3 should therefore introduce hierarchy as a separately controlled architectural factor, not combine it with a new readout, new training schedule, or runtime ablation.

The G3 proposal should compare a flat control against a hierarchy-enabled arm with identical story order, seeds, training data, epoch-3 checkpoint, and confidence metrics. It should separately report bottom-up and top-down coupling diagnostics, per-level state/spectral evidence, and task-wise QA3/QA5 outcomes. A hierarchy improvement should be attributed only if the hierarchy arm improves relational accuracy or confidence quality without violating stability and coupling invariants.

## Artifacts

The final CI artifact contains:

- `summary.json`
- `per_seed_repeat_metrics.csv`
- `experiment_lab_register.json`
- `diagnostic_report.md`

GitHub Actions artifact: [g2-reproducibility-diagnostic](https://github.com/johnsnowman488-oss/aberconics-framework/actions/runs/42041847620)

## Limitations

- Validation-only evidence; no held-out test results.
- Five seeds provide limited sampling precision.
- The Python bridge/readout diagnostic is not a direct test of the C++ hierarchy ABI.
- Environment pinning resolved observed cross-job drift, but the precise low-level source of the original numerical sensitivity remains unisolated.
- QA3 and QA5 are not closed by this diagnostic.
