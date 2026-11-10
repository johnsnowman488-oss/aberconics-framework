# Flat E-F0 C-capacity validation sweep

## Purpose

Test whether adding memory components/timescales improves the existing G2 flat
E-F0 pathway, while using only a small number of state-width variants to check
whether the result depends on D. This is a validation screen, not a production
selection and not a locked test evaluation.

## Registered design

- Tasks: bAbI QA3 and QA5.
- Primary factor: memory component count C = 3, 4, 6.
- Secondary factor: state width D = 64, 96, 128.
- Matched seeds: 0, 1, 2.
- Training checkpoint: exactly 3 epochs.
- Learning rate: 0.05.
- Feature scale: 10.
- Readout: linear.
- Runtime: existing G2 flat E-F0 DigitalRuntime, isolated episodes.
- Split: train for fitting, validation for comparison. Test is not opened.

## Kernel controls

C=3 is the exact G2 flat kernel: gamma = [2.0, 0.5, 0.1], w =
[0.04, 0.08, 0.12], dt = 0.05, leak = 1.0. Its positive weight mass is 0.24
and stability ratio is 1.38.

C=4 and C=6 extend timescale coverage down to gamma=0.05 and fill intermediate
rates. Their positive weights are computed by a deterministic minimum-norm
solution subject to the same two aggregate constraints:

- sum(w) = 0.24 (preserved kernel mass);
- sum(w / (gamma * leak_rate)) = 1.38 (preserved stability ratio).

This deliberately holds aggregate dynamical strength/stability fixed while
testing a richer timescale basis. The individual channel weights necessarily
change; therefore the result tests component-count plus spectral allocation,
not an isolated abstract count independent of timescale placement.

## Evidence

The runner records per-seed and per-example predictions, epoch losses, runtime
summaries, effective channel count, dataset fingerprints, training-order hashes,
and a Lab register. All capacity cells use identical task story identities,
seed identities, and per-seed training-order hashes.

The sweep has 18 cells (2 tasks x 3 D values x 3 C values), 3 seeds per cell.
Interpretation should emphasize paired comparisons across C at fixed D and
across D at fixed C. Do not select a production configuration from this
validation screen alone; any later confirmation must be separately specified.
