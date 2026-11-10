# G4.1 — mass-preserving fast-kernel correction

## Objective

G4.1 corrects the fast-level stability failure observed in G4 without changing
the two-level hierarchy, vector C++ ABI, bottom-up/top-down coupling schedule,
readout, feature scale, learning rate, or three-epoch checkpoint.

G4 used a calibrated fast kernel with gamma=[2.0, 0.45, 0.035],
w=[0.16, 0.108, 0.0448], and leak rate 2.0. The original kernel mass is
sum(w)=0.3128. The calibrated stability ratio before top-down modulation is 0.8.

## Diagnosis

The active fast kernel applies top-down decay modulation:

    gamma_i' = gamma_i exp(-g d)

G4 then divided every positive weight by the original kernel mass to force
sum(w')=1. Because the decay rates were simultaneously reduced, the stability
ratio R = sum_i w_i / (gamma_i leak_rate) was multiplied by approximately
1 / 0.3128 before accounting for the small context-drive adjustment. This
explains the observed fast-level ratio near 2.567. The upper-level ratio was
already below the 0.9 margin.

## Correction

G4.1 retains the top-down decay modulation and preserves the calibrated
positive-kernel mass:

    w_i' = w_i

rather than applying unit-sum rescaling. This is a targeted mass-preserving
mode, not a new hierarchy gain or an arbitrary hard-coded attenuation factor.

The expected fast stability ratio is therefore the calibrated base ratio
multiplied by the top-down decay factor exp(g d). The run records the actual
maximum per-level ratio for every seed and requires both levels to remain at
or below 0.9.

The legacy G4 unit-sum mode remains available for regression testing. The
matched G3 scalar reference is rerun with its original unit-sum setting so
that the historical comparison is not silently changed.

## Frozen controls

- Existing G2 E-F0 spine and G3 two-level hierarchy.
- 64-dimensional vector memory at both levels.
- Bottom-up gain 0.4 and top-down gain 0.5.
- Full memory kernel and context kernel.
- Feature scale 10, learning rate 0.05, three epochs.
- QA3 and QA5, same validation examples and matched seed identities 0, 1, 2.
- Three seeds for this short corrective validation.
- Validation split only; locked test split is not downloaded or opened.
- No static weight strengthening, ABI redesign, new hierarchy arms, or C++ vectorization changes.

## Acceptance

1. Native vector stepper and hierarchy tests pass.
2. Original calibrated kernel mass is preserved at every fast-path event.
3. Fast and upper stability ratios are at most 0.9 for every seed.
4. States remain finite and both coupling directions remain active.
5. QA3/QA5 pairing, example identity, and confidence evidence are complete.
6. QA3 and QA5 performance are compared with the matched G3 reference; no
   predictive improvement is assumed in advance.

This is a three-seed screening validation, not a final statistical claim.
The locked test split remains unopened. A separate evaluation split is only
considered after reviewing these results.
