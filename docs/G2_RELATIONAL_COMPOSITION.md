# G2 — Relational Composition and Confidence Qualification

Status: implemented on the G1-L work branch; full benchmark workflow is defined in `.github/workflows/g2-relational.yml`.

## Purpose and boundary

G2 tests whether the current learned token-readout pathway can compose more demanding relational evidence on bAbI QA3 and QA5, and whether its reported confidence is trustworthy. The measured learned arms are:

- **E0:** original, supervised token-readout baseline.
- **E-F0:** the same supervised readout path through the F0 runtime, in isolated mode.

Both arms use the same full memory variant, task data, feature dimension, readout, learning rate, feature scale, epochs, and seed IDs. The SOE kernel is not changed. Runtime ablation and hierarchy comparisons remain deferred.

The relational fixtures are a separate semantic control. They replay structured move, pickup, transfer, release and explicitly supplied coreference-alias events. They verify state transitions and answer derivation; they are **not** natural-language parsing, not official bAbI examples, and not model accuracy. In particular, a supplied pronoun alias validates identity continuity after resolution, not the model's ability to infer that pronoun.

G1 remains the deterministic QA2 control. G1-L remains a learned QA2 binder with a QA2-only typed-event parser. Neither is reported as a QA3/QA5 learned result. G1-L is retained as context for holder-retrieval and overconfidence behaviour, not silently extended beyond its supported task.

## Fixture suite

The curated fixture set is versioned at `code/python/d2c/experiments/fixtures/g2_relational_cases.json`. It contains 12 independently specified cases across:

- three-fact composition and holder-dependent location;
- object transfer / hand-off and post-transfer movement;
- release, fixed-location persistence, and reacquisition;
- temporal updates where the latest event must dominate stale state;
- multiple actors and objects without cross-binding;
- distractor interference and role assignment;
- explicit coreference aliases;
- insufficient evidence, where the answer must remain unresolved rather than be fabricated.

Each fixture stores an expected holder and expected location. The deterministic replay emits an event ledger and final world state. A bad transfer source or invalid release actor is rejected instead of being repaired silently. The suite is run by tests and included in every G2 result bundle.

## Matched benchmark protocol

Default run: tasks 3 and 5, 5 matched seeds (0–4), dimension 64, 5 epochs, linear readout, learning rate 0.05, feature scale 100, full memory variant, isolated execution. The CLI allows 3–20 seeds, dimension/readout/epoch adjustments, and a task subset for diagnostic runs. The full G2 closure gate requires both QA3 and QA5.

The fixed comparison is E0 versus E-F0 for each task. Seed IDs and the ordered question/target pairs must match exactly; the runner fails if they do not. Train/validation/test split files are SHA-256 fingerprinted where present, and train/evaluation task counts are recorded. The held-out test split is evaluated once under this pre-registered configuration; no test-based tuning, threshold selection, or model selection is performed. Validation-only exploratory work must be a separate run and must not replace this locked confirmation.

G1-L is not a third arm on QA3/QA5 because its current parser only supports QA2. Comparing it numerically with the E0/E-F0 QA3/QA5 scores would be an architectural and task mismatch.

## Metrics

### Predictive performance and sampling uncertainty

Each arm records overall and per-seed accuracy, correct/total counts, and the per-example prediction record. Per-seed accuracy has a Wilson 95% interval. E-F0 minus E0 is computed as a paired difference for the same seed and task; the report includes the number of paired seeds, mean difference, sample standard deviation, standard error, sign counts, and a two-sided 95% Student-t interval. It also reports an exact two-sided paired sign-test p-value, which does not depend on the size of the seed-level differences and is therefore a useful low-power robustness check with few seeds. The seed-level interval is the principal uncertainty summary for comparing training initializations; with five seeds, treat all inferential results as preliminary and inspect the per-seed differences rather than relying on a single p-value.

### Stronger confidence and calibration evidence

The complete class-probability vector is retained and validated (finite, non-negative, positive mass, normalized before scoring). We report:

- negative log-likelihood (NLL) and multiclass Brier score as proper scoring rules;
- top-label fixed-width ECE and equal-mass/adaptive ECE;
- classwise ECE by label and macro classwise ECE;
- maximum calibration error and RMS calibration error;
- mean top-label confidence, confidence–accuracy gap, mean confidence on errors, and top-two probability margin;
- incorrect-prediction overconfidence fractions at confidence thresholds 0.80 and 0.90;
- high-confidence coverage, accuracy and risk at thresholds 0.70, 0.80, 0.90 and 0.95;
- risk–coverage curve and area under the risk–coverage curve (AURC);
- mean predictive entropy.

In addition to per-seed Wilson intervals, the runner generates deterministic non-parametric **story-level bootstrap** 95% percentile intervals for accuracy, NLL, Brier, fixed ECE, adaptive ECE and AURC. These intervals describe finite-example sampling within a run. They do not replace the between-seed paired interval, and repeated seed-by-story rows are not pooled as though independent.

Calibration is sensitive to binning and sample size. Therefore fixed, adaptive and classwise ECE are all reported together, with bin counts and per-bin sample counts. Runs with fewer than 100 evaluation examples carry a low-sample caution. Confidence metrics are descriptive on the locked evaluation split; no post-hoc calibrator or confidence threshold is fitted to test.

### Interpretation flags

A confidence warning is raised when high-confidence errors are frequent (especially among predictions at or above 0.90), or when fixed/adaptive ECE is high. These are diagnostic warnings, not automatic exclusion of a benchmark run. The previous G1-L issue—high confidence on incorrect holder-dependent answers—must remain visible, not be hidden by overall accuracy.

## Acceptance criteria

G2 has two distinct acceptance layers.

**Evidence-bundle / execution gates (required for G2 completion):**

1. All 12 structured semantic fixtures pass, with category counts and event ledgers present.
2. Both QA3 and QA5 are evaluated for full closure.
3. E0 and E-F0 complete with at least 3 matched seeds (5 by default), identical seed IDs and identical ordered evaluation question/target pairs.
4. Dataset fingerprints, configuration, code revision, split sizes, per-seed outcomes and per-example evidence are recorded.
5. Probability vectors validate, and each seed contains Wilson, bootstrap, NLL, Brier, calibration and selective-risk diagnostics.
6. The held-out split is used only for fixed, pre-registered evaluation—not selection or calibration fitting.

A diagnostic single-task run can pass its local mechanics but cannot close the two-task G2 coverage gate.

**Scientific interpretation (not a CI pass/fail gate):**

- Claim an E-F0 accuracy improvement on a task only when the paired 95% seed-level interval is wholly above zero; report the effect size and seed signs. A small or crossing interval is *inconclusive*, not proof of equivalence.
- A decline is reported symmetrically when the interval is wholly below zero.
- Calibration is assessed separately from accuracy. Stronger accuracy does not compensate for worse NLL/Brier or high-confidence error risk. Conversely, improved ECE alone does not establish better reasoning.
- Report QA3 and QA5 separately. Do not let a good task average conceal a weak relation category or seed.

## Result contract and Lab integration

The G2 runner is `python -m d2c.experiments.babi_g2`. It writes a versioned `summary.json`, `summary.md`, `metadata.json`, `manifest.json`, `config.json`, `experiment_lab_register.json`, `per_seed_metrics.csv`, and `fixtures.jsonl`; per-example JSONL/CSV preserves the full prediction evidence. The manifest includes code revision and dataset fingerprints.

The local Experiment Lab exposes G2 as a registered, allow-listed experiment. It launches the paired runner with validated QA3/QA5 selection, seed count, dimension, readout, learning rate and epochs; records the command/configuration/revision/dataset hashes; and provides links to result files. It does not accept arbitrary shell commands or paths. The Lab displays paired seed scores, confidence summaries and fixture status while preserving the deterministic-fixture/learned-model claim boundary.

The dedicated `.github/workflows/g2-relational.yml` workflow builds the C++ core, downloads the bAbI splits, runs the fixture and Python tests, executes the paired QA3/QA5 benchmark, checks the evidence gates, and uploads the Lab bundle. Julia remains out of scope. No hierarchy ablation, F1 mechanism, or runtime ablation is introduced by G2.
