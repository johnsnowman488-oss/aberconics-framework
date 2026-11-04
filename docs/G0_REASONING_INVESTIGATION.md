# G0 — QA2 Reasoning Capability Investigation

**Status:** execution protocol; implementation and measurements are gated on CI validation.  
**Scope:** diagnostic instrumentation, readout sweep, QA2 oracle/entity-binding probes.  
**Exclusions:** F0 runtime ablation, SOE-kernel changes, F1–F5 learning mechanisms, Julia CI.

## Scientific question

Does the current SOE memory representation contain enough information to solve QA2's two-supporting-fact questions? If not, is the limiting factor feature scale, supervised readout capacity, entity binding, or relational state composition?

The previous matched QA2 control run used task 2, dimension 64, full kernel, linear readout, five epochs, isolated mode and three seeds. E0 and E-F0 both achieved 24.4% mean accuracy (seed accuracies 22.3%, 28.0%, 22.9%). F0 coordinates execution and records diagnostics; the SOE kernel and supervised readout remain unchanged.

## G0-A — Diagnostic parity

For every test example and seed, both E0 and E-F0 must emit:
- stable story/example identifier and task;
- question, gold answer, predicted answer;
- ordered answer-class list and full probability vector;
- top confidence, prediction margin, correctness and negative log-likelihood;
- story statement count, token count and runtime step count where available;
- evidence-backed failure labels.

E0 and E-F0 must use the same field names and schema version. Existing E-F0 prediction records are the starting point; E0 must be brought to parity.

Aggregate metrics: answer-class target/prediction distributions, confusion matrix, per-class precision/recall/F1, accuracy, macro-F1, NLL, multiclass Brier score, expected calibration error with documented bins, confidence on correct/incorrect predictions, and per-seed uncertainty.

Failure labels may be multi-label: entity_binding, two_fact_chain, temporal_update, distractor_interference, answer_class_bias, readout_or_calibration, unknown. Do not infer a cognitive failure category from a wrong answer alone. Preserve raw story, question and prediction for human review.

## G0-B — Readout and feature-scale sweep

Factors:
- readout: linear, MLP;
- memory feature multiplier: 1, 10, 100 (100 is the current default);
- learning rate: 0.001, 0.005, 0.01, 0.05;
- epochs: 5 initially, then 20 only for promising validation configurations;
- seeds: matched seeds 0–4 initially; expand finalists to 10 or more.

Measure memory/query/gated feature blocks separately. Scaling the memory block must not silently change the query block. Record feature norms, gradient/update norms, training loss, non-finite values and validation metrics.

Use train for fitting, validation for configuration selection, and test only for locked final confirmation. The existing runners currently train on train and report test; the sweep must explicitly load babi_valid.jsonl. Never choose hyperparameters from test accuracy.

Stage the work: deterministic fixture smoke tests; coarse validation screen; select at most three configurations using validation metrics and stability; confirm finalists on untouched test; preserve raw cell results and selection rationale.

## G0-C — QA2 oracle ladder and controlled entity binding

| Level | Supplied representation | Diagnostic interpretation |
|---|---|---|
| L0 | Existing token forcing and SOE state | Current end-to-end baseline |
| L1 | Parsed statement/fact boundaries | Gain isolates input segmentation/encoding |
| L2 | Correct entity-role bindings | Additional gain isolates entity binding |
| L3 | Correct relational world state with temporal updates | Additional gain isolates composition/update |
| L4 | Answer from gold final world state | Parser/dataset/oracle ceiling |

L4 is a data/oracle validity check, not a model score. Test the parser and state updater against hand-checked QA2 fixtures before full-dataset execution.

Counterfactual controls: rename/swap entity names while preserving roles; insert irrelevant statements; reorder irrelevant statements; reorder facts only where semantics permit; remove one essential supporting fact; change an entity's location after a relevant object transfer. Expected answers must be computed from explicit world-state rules and asserted in tests. The system should be invariant to irrelevant insertions and name renaming, sensitive to removal of an essential fact, and respect temporal updates.

## Result bundle contract

Each immutable run includes:
- metadata.json: schema version, run ID, code revision, UTC time, command/config, dataset source and SHA-256 split fingerprints, Python/NumPy versions;
- summary.json with aggregate metrics and protocol;
- per_seed.csv, per_example.jsonl, confusion_matrix.csv;
- sweep_results.csv and selection.json for sweep runs;
- oracle_ladder.json and counterfactuals.json for reasoning probes;
- summary.md, stdout.log and stderr.log.

Unavailable metrics are omitted or explicitly marked unavailable. Failed/incomplete runs retain metadata and logs and are marked as such.

## CI gates

Keep Python + C++ CI; Julia remains excluded. Add deterministic tests for metric calculations, probability validation, calibration bins, QA2 parser/state updates, counterfactual invariants and result schema. Add a bounded quick G0 integration profile to PR CI using synthetic fixtures (no full dataset download). Add a manually dispatched full G0 workflow that downloads/validates bAbI, runs matched E0/E-F0 diagnostics, the staged validation sweep and oracle/counterfactual suite, then uploads the result bundle.

A report is accepted only when CI is green and the artifact includes the exact config, revision and split fingerprints.

## Execution order

1. Bring E0 diagnostics to parity with E-F0; add metric/schema tests.
2. Validate result bundles and CI quick profile.
3. Run matched QA2 diagnostic baseline and inspect per-example failures.
4. Implement and test oracle ladder/counterfactuals.
5. Run validation-only readout sweep.
6. Lock finalists and perform held-out test confirmation.
7. Publish reproducible Experiment Lab report; only then choose the next architecture experiment.

## Interpretation of the existing baseline

The previous QA2 artifact exposed per-example predictions for E-F0 but not E0. It supports the observation that both achieved 24.4% mean accuracy, but it cannot support a matched per-example error-category or calibration comparison. New G0 measurements must supersede it for those analyses.
