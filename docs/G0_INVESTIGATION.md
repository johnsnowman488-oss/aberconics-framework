# G0 — QA2 diagnostics, readout selection and reasoning probes

G0 investigates the current QA2 bottleneck without changing the SOE kernel or attributing execution coordination to reasoning capability.

## Reproducible sequence

1. Run the E0 and E-F0 baselines on the same task, dimension, seed IDs, training split, readout, feature scale and epochs. E-F0 uses isolated mode. Its runtime audit is recorded separately from prediction quality.
2. Preserve every evaluated story's question, statement sequence, target, prediction, full class-probability vector, confidence, top-two margin and failure category. Aggregate answer distributions, per-class precision / recall / F1, confusion matrix, calibration bins, ECE, NLL, Brier score and training loss summaries.
3. Select readout settings only on valid: compare linear feature scales, compare learning rates at the selected scale, then compare a nonlinear MLP. Re-run the selected setting on test once for E0 and E-F0. The default candidate set is scales 1, 10, 100 and learning rates 0.001, 0.01, 0.05.
4. Run the QA2 event oracle over the requested split. It reports event parsing, actor/object binding, final state resolution and answer correctness. The controlled suite tests renaming, irrelevant entity movement, fact order, altered/removal of supporting facts, pickup before location observation and drop semantics.

## Interpretation boundaries

- The symbolic oracle is an explicit rule-based diagnostic for the generated bAbI QA2 event grammar. It is not a learned model and its accuracy must not be reported as Aberconics accuracy.
- An oracle jump from the token baseline to event parsing indicates an input segmentation/representation opportunity; improvement after entity binding indicates a binding issue; improvement after chronological object-state replay indicates a temporal relation/update issue. Remaining errors may be parser coverage or question/readout mapping errors and require inspection.
- The sweep selects on validation. The test split is reserved for the baseline and one selected-configuration confirmation.
- Runtime phase/trace/checkpoint/stability results are not reasoning scores. F0 remains isolated for the primary comparison.
- The SOE memory kernel remains fixed. Runtime ablation is deferred.

## Outputs

The CI workflow emits G0_REPORT.md, comparison.json, comparison.csv, and complete raw result bundles for E0, E-F0, the sweep and oracle. The same runner settings can be used in the local Experiment Lab. Configurations and source commit are retained with the workflow artifact.
