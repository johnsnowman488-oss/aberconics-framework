# G3: G2 E-F0 Vectorized Runtime versus Full Scalar-ABI Hierarchy

**Status:** G3 baseline completed; G3-N top-down normalization validation on branch `g3/normalized-top-down-kernel`  
**Experiment ID:** G3-FULL-HIERARCHY  
**Scope:** QA3 + QA5; two matched arms only.

## 1. Research question and claim boundary

G3 asks whether the full bottom-up/top-down scalar-ABI hierarchy package improves relational QA over the established G2 E-F0 vectorized runtime package.

The two arms are:

1. **`ef0_vectorized` — flat control.** Reuse the existing G2 E-F0 runner directly: `d2c.experiments.babi_f0.run_babi_f0_experiment`. This is the existing Python-vectorized `DigitalRuntime.step → step_memory` execution path with the six-phase F0 runtime spine active, in isolated mode. Its token bridge, memory kernel, readout extraction, episode reset semantics, phase coordination and runtime configuration are not reimplemented for G3.
2. **`hier_bu_td` — hierarchy.** The existing two-level C++ hierarchy chain-spec ABI with scalar chi per channel, a fast event-memory level, a slower relational-context level, and both bottom-up and top-down coupling.

There is no E0-only arm, no scalar-ABI flat third arm, and no static weight-strengthening arm. Runtime ablation and static strengthening are deferred.

**Interpretation boundary:** the arms are two complete execution packages. They differ in backend and memory feedback formulation as well as hierarchy. A positive result supports the scalar-ABI bidirectional hierarchy package over the G2 E-F0 vectorized package under this protocol; it does not isolate hierarchy as the only causal difference. Do not make a hierarchy-only causal claim from G3.

The existing per-task/per-seed F0 spine exercise is also retained as a companion qualification audit. It does not replace the F0 runtime that is actively used to compute the flat arm's predictions.

## 2. Frozen protocol

| Setting | G3 value |
|---|---|
| Tasks | QA3 and QA5; both required |
| Arms | G2 E-F0 vectorized flat; full scalar-ABI bottom-up/top-down hierarchy |
| Seeds | 3 matched seeds, numbered 0–2 for G3-N validation (reduced CI evaluation) |
| Epochs | 3 |
| State dimension | 64 |
| Readout | Linear |
| Learning rate | 0.05 |
| Feature scale | 10 |
| Evaluation | 3-seed QA3/QA5 validation only; previously consumed test split remains closed |
| Confidence bootstrap | 200 replicates in preflight; 400 in held-out test |
| Mode | Isolated episode per story |
| Runtime ablation | Deferred |
| Static weight-strengthening arm | Deferred |

The 3-epoch and feature-scale-10 settings are the frozen G3 checkpoint. Both arms use the same number of epochs, learning rate, feature scale, readout architecture, readout seed, bridge vocabulary, task split and story order.

## 3. Arm execution details

### Flat: existing G2 E-F0

The G3 flat path calls the existing `run_babi_f0_experiment` implementation rather than creating a second vectorized runner. It preserves:

- `BabiF0Config(mode="isolated", exercise_spine=True)`;
- existing `DigitalRuntime` defaults and the full six-phase controller;
- existing `build_babi_memory_config("full")` memory kernel;
- existing vectorized Python `step_memory` updates;
- the existing bAbI bridge, feature extraction and linear readout;
- separate training and evaluation runtimes, with episodic memory reset before each story;
- per-task and per-seed F0 maintenance/trace/checkpoint audit.

The only G3-specific settings passed into this existing runner are the frozen seed count, dimension, 3 epochs, learning rate 0.05, feature scale 10, linear readout, QA3/QA5 task and selected evaluation split.

### Hierarchy: scalar ABI

The existing C++ hierarchy arm is retained:

- level 0: fast event memory, using the established G3 stable full kernel;
- level 1: slow relational context;
- bottom-up level 0 → level 1, gain 0.4;
- top-down level 1 → level 0, gain 0.5;
- context gamma `[0.35, 0.08]`;
- context weights `[0.05, 0.02]`;
- context leak 0.5;
- Form-B negative-feedback ABI stepping;
- top-down positive-kernel weight normalization enabled (G3-N); active fast-level kernel is normalized to unit total weight;
- the existing 0.9 stability margin is retained. For the configured fast kernel, unit-sum normalization raises the zero-context stability ratio from 0.8 to approximately 2.56; G3-N is therefore a diagnostic of the normalization/stability interaction, not presumed to be a stable performance candidate.
- active per-level kernels, state norms, spectral diagnostics and stability ratios captured.

The hierarchy's scalar-ABI stability contract remains unchanged. The flat E-F0 memory kernel is not silently substituted into the hierarchy, since it belongs to a different vectorized/input-driven execution formulation.

## 4. Pairing and training controls

For each task and seed:

- Both arms receive the same ordered training and evaluation stories.
- The E-F0 runner's actual epoch permutations are hashed and checked against the G3 precomputed order schedule.
- Both arms start from the same seeded linear-readout initialization; initial weight/bias fingerprints must match.
- Both use the same token vocabulary, bridge dimension, answer classes, epoch count, learning rate and feature scale.
- Each story begins with fresh episodic memory in both arms.
- The held-out test split is not used for training, model selection or confidence calibration.
- Prediction identity is checked by seed, story index, task, passage, question and target before paired statistics are computed.

A failed pairing, training-order, readout-fingerprint or example-identity check invalidates the paired comparison.

## 5. Metrics

### Task outcome
Per-seed correct/total and accuracy; mean task accuracy; hierarchy-minus-E-F0 paired seed differences; 95% paired-seed t intervals; exact two-sided sign tests and Holm correction across QA3 and QA5.

### Confidence quality
Per-seed NLL, multiclass Brier, fixed-bin and adaptive equal-mass ECE, classwise diagnostics, confidence-on-errors, high-confidence error fraction, risk/coverage and AURC, Wilson accuracy intervals and story-bootstrap intervals.

Story-bootstrap uncertainty and between-seed paired uncertainty are kept separate.

### Hierarchy mechanism
Per-level state norm, D_eff, M_cap, M_scale and H_mem where emitted; lower-level active kernel weight and gamma ranges; bottom-up context activity; top-down gamma variation; positive-kernel and finite-state checks at both levels; and per-level stability ratios.

### F0 runtime and spine
The flat arm reports its actual train/evaluation runtime counters and stability violations. The companion audit checks WARMUP, EXPLORE, CONSOLIDATE, EVALUATE, PRUNE and GROW coverage; kernel immutability; contiguous trace sampling; checkpoint round-trip; deterministic continuation; and zero runtime stability violations.

### Semantic fixtures
The existing 12 structured relational fixtures are replayed and reported separately. They check explicit object-holder-location event semantics; they are not learned QA3/QA5 predictions.

## 6. Acceptance criteria

A technically valid G3 evidence bundle requires:

1. Both QA3 and QA5 complete.
2. Matched seed IDs, initial readout fingerprints, training-order hashes and evaluation example identities.
3. Both bottom-up and top-down hierarchy edges are active; context activity is non-zero; states and spectral outputs remain finite; active kernels at both levels remain finite and positive; top-down gamma variation is non-zero; and the per-step stability ratio at each hierarchy level remains at or below 0.9.
4. All per-task/per-seed F0 spine audits pass, with no stability violations and successful checkpoint/trace checks.
5. Complete confidence evidence for both arms and both tasks.
6. All 12 structured relational fixtures pass.

CI enforces technical validity, not a required accuracy uplift. Whether hierarchy improves accuracy, calibration or selective risk is an empirical result.

## 7. Result bundle and Lab integration

The runner is `python -m d2c.experiments.babi_g3`. It writes:

- `summary.json` and `summary.md`;
- `per_seed_metrics.csv`;
- `per_example.csv` and `per_example.jsonl`;
- `config.json`;
- `manifest.json` and `metadata.json`;
- `experiment_lab_register.json`.

The Lab catalogue describes the G2 E-F0 vectorized flat arm against the scalar-ABI bidirectional hierarchy. The result bundle records the backend for every seed, flat-arm runtime counters, F0 spine audits, hierarchy mechanism diagnostics, paired outcomes, confidence metrics and per-example evidence.

## 8. Unified execution

One GitHub Actions workflow owns the full sequence. There is no separate manual boolean input or second locked-test workflow path.

1. Build the shared C++ runtime.
2. Prepare train and validation only; do not download or open test.
3. Run F0, hierarchy, G2 fixture, G3 and Lab tests; compile Python modules and validate Lab JavaScript.
4. Run the 3-seed QA3/QA5 validation preflight.
5. Keep the previously consumed test split closed; G3-N is a validation-stage normalizer check only.
6. Upload one compact validation evidence bundle. A fresh locked holdout is reserved for a later confirmatory stage.

Julia remains out of scope and is skipped. The workflow does not gate on an observed accuracy improvement.
