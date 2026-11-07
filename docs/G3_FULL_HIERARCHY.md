# G3: Flat ABI versus Full Bidirectional Hierarchy

**Status:** registered experiment; implementation on branch g3/full-hierarchy-relational  
**Experiment ID:** G3-FULL-HIERARCHY  
**Scope:** QA3 + QA5, matched flat ABI versus full bottom-up + top-down hierarchy.

## 1. Question and claim boundary

G3 tests whether a slower relational-context level, connected through bottom-up event coupling and top-down kernel modulation, improves learned relational retrieval over the established flat ABI path.

The benchmark is deliberately restricted to two arms:

1. **flat_abi** — one-level C++ ABI scalar-chi reference, with no hierarchy edges.
2. **hier_bu_td** — the same lower-level C++ ABI substrate plus a slower context level and both bottom-up and top-down couplings.

There is no G3 static weight-strengthening arm. The active lower-level kernel weights and decay rates are nevertheless recorded at every episode so that the normalization side effect observed in D4A is visible. Results must not attribute an improvement to context adaptation alone if the active weight sum indicates that the hierarchy also changed aggregate feedback strength. A dedicated static-strengthening control may be added later if this first two-arm comparison warrants it.

The F0 six-phase runtime spine is exercised once per seed as a companion qualification audit. It does not replace the hierarchy state engine or change the benchmark's ABI dynamics. Both learned benchmark arms use the same C++ ABI scalar-chi substrate, avoiding a flat-Python-vector versus hierarchy-ABI mismatch.

G3 is not a natural-language parser test, not a general language-reasoning claim, and not an implementation of the later F-series learning mechanisms.

## 2. Frozen protocol

| Setting | G3 value |
|---|---|
| Tasks | QA3 and QA5; both required |
| Seeds | 20 matched seeds, numbered 0–19 |
| Epochs | 3 |
| State dimension | 64 |
| Readout | Linear |
| Learning rate | 0.05 |
| Feature scale | 10 |
| Evaluation | Locked test split, once |
| Confidence bootstrap | 400 story-bootstrap replicates per seed |
| Mode | Isolated episode per story |
| Memory variant | Full |
| F0 spine | All six phases, trace continuity and checkpoint/restore, every seed |
| Runtime ablation | Deferred |
| Static weight-strengthening arm | Deferred |

The three-epoch and feature-scale-10 settings follow the provisional G2 validation readout. G3 does not re-select these settings using its held-out test results. The numerical environment is pinned in CI: Python 3.11.18, NumPy 2.4.6, single-thread numerical libraries, and the same OpenBLAS/NumPy dispatch constraints used for the G2 repeatability diagnostic.

Train and evaluation split fingerprints, exact protocol, training-order hashes, initial readout fingerprints, code revision and Python version are saved in the run bundle.

## 3. Hierarchy configuration

The lower level uses the existing full ABI memory kernel, negative-feedback Form-B stepping, external token forcing at level 0, and coupling index 0.

The hierarchy arm adds one slower context level and two edges:

- Bottom-up: level 0 → level 1; gain 0.4; normalized positive kernel materialization.
- Top-down: level 1 → level 0; gain 0.5; positive kernel materialization with weight normalization disabled so the base aggregate weight scale is preserved at zero context drive.

The context level uses gamma [0.35, 0.08], weights [0.05, 0.02], and leak 0.5. Both levels use the same state dimension and timestep. The level-0 active kernel is captured throughout every episode.

The top-down edge uses normalize_weights=False. This corrects the D4A confound: normalization had increased aggregate fast-level feedback to a constant sum independent of context. The G3 top-down gain is conservatively set to 0.5 so the unnormalized co-modulation has room to vary while retaining the 0.9 active-kernel stability margin. G3 instead starts from the same base weights as flat ABI and permits context-dependent co-modulation of weights and decay rates. Both active weight sums and per-channel gamma ranges are recorded. The C++ Form-B stability ratio sum(w / (gamma * leak)) is computed on every active lower-level kernel; every observed ratio must remain at or below the 0.9 safety margin. A performance gain is not sufficient evidence for context-dependent hierarchy if the modulation is inactive or destabilizes the lower level.

## 4. Matching and training controls

For each task and seed:

- Both arms receive the same train and evaluation stories in the same order.
- The training shuffle order is generated once and reused across arms; its hash is recorded.
- Both arms start from the same seeded readout initialization; the weight/bias fingerprint is checked.
- The tokenizer bridge, answer classes, readout, training epochs, learning rate and feature scale are identical.
- Every story starts from a fresh episode state in both arms.
- The held-out split is not used for training, model selection or confidence calibration.
- Prediction identity is checked using story index, task, passage, question and target before paired metrics are computed.

A failed identity check invalidates the paired comparison and is recorded as an acceptance failure.

## 5. Metrics

### Task outcome

Per-seed correct/total and accuracy; task-wise mean accuracy; hierarchy-minus-flat paired seed differences; 95% paired-seed t intervals; exact two-sided sign tests and Holm correction across QA3 and QA5.

### Confidence quality

Per-seed NLL, multiclass Brier, fixed-bin and adaptive equal-mass ECE, classwise diagnostics, confidence-on-errors, high-confidence error fraction, risk/coverage and AURC, Wilson accuracy intervals and story-bootstrap intervals.

Story-bootstrap uncertainty and between-seed paired uncertainty are kept separate.

### Hierarchy mechanism

Per-level state norm, D_eff, M_cap, M_scale, and H_mem where emitted; lower-level active kernel weight and gamma ranges; weight-sum normalization; top-down gamma variation; bottom-up context activity; positive-kernel and finite-state checks.

### F0 spine companion

WARMUP, EXPLORE, CONSOLIDATE, EVALUATE, PRUNE and GROW coverage; kernel immutability; contiguous trace sampling; checkpoint round-trip; deterministic continuation; and zero runtime stability violations.

### Semantic fixtures

The existing 12 structured relational fixtures are replayed and reported separately. They check object-holder-location semantics and are not learned QA3/QA5 predictions.

## 6. Acceptance criteria

A technically valid G3 evidence bundle requires:

1. Both QA3 and QA5 complete.
2. Matched seed IDs, initial readout fingerprints, training-order hashes and evaluation example identities.
3. Both bottom-up and top-down hierarchy edges are active; context activity is non-zero; state/spectral outputs remain finite; active kernel values remain finite and positive; top-down gamma variation is non-zero; the per-step lower-level stability ratio remains at or below 0.9.
4. All F0 six-phase audits pass for every seed, with no stability violations and successful checkpoint/trace checks.
5. Complete confidence evidence for both arms and both tasks.
6. All 12 structured relational fixtures pass.

The experiment does not hard-code a required accuracy uplift. Whether hierarchy improves accuracy, calibration or selective risk is an empirical result, not a CI pass/fail condition. The paired effects, confidence intervals and seed directions determine the scientific interpretation.

A hierarchy advantage is strongest when it is repeatable across seeds, appears on QA3 without materially regressing QA5, improves confidence quality or accuracy, and is accompanied by active context-dependent kernel changes rather than only a normalized weight-sum shift.

## 7. Result bundle and Lab integration

The runner is python -m d2c.experiments.babi_g3. It writes:

- summary.json and summary.md
- per_seed_metrics.csv
- per_example.csv and per_example.jsonl
- config.json
- manifest.json and metadata.json
- experiment_lab_register.json

The Experiment Lab registers G3 with fixed tasks, three epochs, linear readout, learning rate 0.05 and feature scale 10. Seeds, dimension, evaluation split and bootstrap sample count remain visible controls. The Results page displays paired task effects, confidence comparisons, per-seed outcomes, hierarchy mechanism diagnostics, F0 companion audits and per-example evidence.

## 8. Execution

Validation-only preflight:

    PYTHONPATH=code/python python -m d2c.experiments.babi_g3 \
      --tasks 3,5 --dim 64 --seeds 5 --epochs 3 \
      --lr 0.05 --feature-scale 10 --bootstrap-samples 400 \
      --eval-split valid --output-dir runs/experiment-lab/g3-validation

Locked test evaluation:

    PYTHONPATH=code/python python -m d2c.experiments.babi_g3 \
      --tasks 3,5 --dim 64 --seeds 20 --epochs 3 \
      --lr 0.05 --feature-scale 10 --bootstrap-samples 400 \
      --eval-split test --output-dir runs/experiment-lab/g3

CI builds the shared C++ core, runs G3 and hierarchy regression tests, performs the validation preflight, executes the locked 20-seed test evaluation and uploads the complete Lab bundle. Julia remains skipped.
