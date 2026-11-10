# G4.2 — N-Level Adaptive Hierarchy

## Scope

G4.2 extends the G4.1 mass-preserving vectorized hierarchy to H2/H3/H4. The first sweep fixes D=64, three epochs, learning rate 0.05, feature scale 10 and three matched seeds. QA3 is the primary selection task; QA5 is the secondary guardrail.

## Architecture

- Level 0 uses the G4.1 full negative-feedback kernel.
- Every upper level uses the calibrated context kernel.
- Adjacent levels update simultaneously from the previous event's states.
- Bottom-up drive propagates from level l-1 to l.
- Top-down modulation propagates from level l+1 to l and preserves positive kernel weight mass.
- Cascading levels provide longer effective dynamics while retaining the calibrated per-level kernel.
- Per-level channel count, kernel mass, state norms and active stability ratios are recorded.
- The locked test split is never downloaded or opened.

The initial depth sweep holds per-level kernel definitions constant, not total channel count. Total channels and runtime therefore rise with depth. Treat this as a depth-and-resource screening experiment, not a definitive cost-matched causal claim.

## Training-only vocabulary

The vocabulary is built from training stories only. Validation-only tokens map to an explicit <UNK> token. Validation-only answers map to the <UNK> answer class; validation labels are never added to vocabulary construction.

## H-depth sweep

Three GitHub Actions matrix jobs execute concurrently: H2, H3 and H4. Each uses seeds 0, 1 and 2 and the three-epoch checkpoint. Each job uploads its own evidence bundle.

The selector requires stability <= 0.9 for both tasks, ranks by QA3 mean validation accuracy, uses QA5 mean validation accuracy as the secondary criterion, then uses lower total channel count and lower depth as tie-breakers. If no depth passes stability, the adaptive follow-up is withheld.

## F4 prune/grow follow-up

The selected depth receives a matched static versus prune/grow run with identical seed identities, training order, vocabulary, readout initialization and validation examples.

After epoch one, each level performs one controlled channel replacement:
- Select the least-active channel within the slowest-timescale band.
- Prune that channel.
- Grow a new channel at half its previous decay rate.
- Reallocate positive weights so total kernel mass and aggregate stability ratio are preserved.
- Record activity, old/new gamma and weights, channel count, mass and stability before/after.

This is a controlled F4 mechanism activation probe. It verifies real channel death/birth and its downstream effect, but does not yet claim autonomous prediction-error spectral growth or a learned online prune/grow policy.

## Acceptance

- QA3 and QA5 validation coverage complete.
- Three distinct matched seed training orders.
- All states finite.
- Every observed active kernel stability ratio <= 0.9.
- Prune/grow events recorded at every level, for both tasks and all three seeds.
- Static/adaptive readout initialization and training order match.
- Dataset fingerprints and per-example predictions/confidence retained.
- No test data access.
