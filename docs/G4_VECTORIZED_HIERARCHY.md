# G4 — vectorized full hierarchy validation

## Goal

G4 carries the G3 two-level full bottom-up + top-down hierarchy forward, but
uses the new vector-valued C++ digital-memory ABI at both levels. The run is
validation-only on QA3 and QA5, with five matched seeds and the established
three-epoch, learning-rate 0.05, feature-scale 10 checkpoint.

## What stays fixed

- Two hierarchy levels, bottom-up gain 0.4, top-down gain 0.5.
- G3 top-down positive-kernel modulation, including unit-sum weight
  normalization.
- Full memory kernel, 64-dimensional state, isolated stories, linear readout,
  seed identities, readout initialization and per-epoch training order.
- The 0.9 stability margin; no automatic rescaling, clipping, or stability
  repair is introduced.
- Only train and validation data are downloaded or opened.

The existing G3 scalar hierarchy is re-run on the same validation split to
provide a matched reference. The G3 flat E-F0 arm remains available in that
reference bundle but is not the G4 comparison target.

## G4 numerical update

Each level stores u[D] and chi[C,D] in contiguous vector memory. Both levels
use the vectorized C++ negative-feedback stepper. On every event:

1. Top-down modulation of the fast-level kernel is computed from the previous
   context-level state.
2. Bottom-up forcing for the context level is computed from the previous
   fast-level state.
3. Each level advances once through the native vector ABI.

This preserves the hierarchy's simultaneous two-way coupling schedule.
Top-down normalization is kept identical to G3 and its active weight sum is
recorded on each event. Stability ratios are observed; no kernel weights are
silently adjusted to force a pass.

## Important comparison boundary

This is a meaningful implementation-package comparison, not bitwise
numerical parity with G3. G3's scalar ABERSOE hierarchy uses scalar chi and
pre-step memory feedback. G4's vector digital ABI uses channel-by-dimension
chi and post-step memory feedback. G4 also exposes vector-valued memory to
the readout instead of broadcasting one scalar per channel across all
coordinates. These changes are explicit in the result protocol.

## Evaluation and evidence

The workflow runs five seeds (0–4) for QA3 and QA5, three epochs, on
babi_valid.jsonl. It re-runs the G3 scalar hierarchy with the same seed,
readout and example identities, then computes paired G4-minus-G3 effects for
accuracy, NLL, Brier, calibration and selective-prediction risk. The evidence
bundle records dataset fingerprints, per-seed predictions, confidence
diagnostics, training fingerprints, active kernel sums, state norms, and
stability ratios.

The locked test split is not downloaded or opened. A separate evaluation
split may be considered only after reviewing the validation results.
