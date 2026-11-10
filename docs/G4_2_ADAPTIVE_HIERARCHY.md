# G4.2 — Corrected vector-coupled hierarchy campaign

## Status and historical treatment

The earlier G4.2 run is retained as a **superseded coupling/depth pilot**, not accepted as a definitive H2/H3/H4 comparison. Its fixed-D, scalar-context / bottom-level-readout implementation and selection by depth alone did not test the agreed C and D factors or prove meaningful contribution from every level. Its GitHub Actions evidence remains historical and must not be overwritten.

The corrected campaign uses the existing G2 E-F0 spine and G4.1 mass-preserving kernel definitions. It does not change the C++ ABI or native memory runtime.

## R1 — Coupling correction

At level l, define the structured vector state

`s_l = 0.25 u_l + \sum_i w_{l,i}\chi_{l,i}`.

For each adjacent edge, construct a bounded message

`m(v,g) = clip_{L2 <= g}(tanh(v))`.

All messages are computed from the previous event's states. The level input is

`x_l = 1[l=0] e + 1[l>0] m(s_{l-1}, g_BU) + 1[l<H-1] m(s_{l+1}, g_TD)`.

This makes bottom-up and top-down context vector-valued, includes both state and memory channels, and bounds the L2 magnitude of each message. The memory kernel itself is not rescaled during coupling; its positive weights, mass, and stability ratio remain unchanged. This removes the previous scalar mean-absolute-activity bottleneck without changing the C++ ABI.

The readout is a normalized concatenation of per-level memory and query-gated-memory features. The bare query vector is provided only once at level zero. Every upper-level readout block therefore has to contribute state-dependent information rather than duplicate query-only features.

## R2 — Coupling acceptance gates

Before the static campaign, native integration tests require:

1. Perturbing an upper-level vector changes a lower-level computation.
2. Disabling top-down coupling changes the lower-level output.
3. Disabling bottom-up propagation changes the upper-level representation.
4. Every level has a non-empty state-dependent readout block; every validation run measures the output-distribution change when each level block is ablated.
5. All states remain finite and every active kernel stability ratio remains at or below 0.9. Kernel mass and stability are preserved during channel resampling and adaptation.

The first three are intervention tests, not passive diagnostics. Static configuration selection also requires measurable per-level readout contribution, measurable bottom-up and top-down messages, finite states, and the stability margin.

## R3 — Full concurrent H x C x D sweep

The static matrix contains 18 independent configurations and all launch concurrently:

- H = 2, 3, 4 hierarchy levels.
- C = 3, 4, 6 global capacity multipliers.
- D = 64, 128 vector-state dimensions.
- Seeds = 0, 1, 2; epochs = 3 initially.
- QA3 is the primary reasoning metric; QA5 is the secondary control.
- Only train and validation splits are downloaded. The test split is not opened.

**Capacity meaning:** C is a global channel-budget multiplier. Every depth receives the same total budget of 5*C channels. That budget is distributed over active levels proportionally to [3,2,2,2] using deterministic largest-remainder rounding. Thus H4 does not get more total channels than H2 for the same C. Per-level channel kernels are resampled on a log-spaced spectrum, with positive weights fitted to preserve the original per-level kernel mass and stability ratio.

Each matrix job stores configuration, seed-level metrics, per-example predictions, confidence/calibration, level norms, stability, coupling influence, channel allocation, wall-clock cost, dataset fingerprints, and code revision. Each job uploads an independent artifact.

## R4 — Selection barrier

The selector runs only after **all 18 static jobs succeed**. Missing, failed, or duplicate configurations stop selection; no partial matrix can produce a winner.

Candidates must pass:
- all QA3/QA5 validation and three-seed evidence checks;
- finite state and stability <= 0.9 at every level;
- measurable bottom-up and top-down influence;
- measurable state-dependent readout contribution from every level.

Eligible configurations rank by QA3 mean validation accuracy, then QA5 accuracy, lower QA3 fixed-bin ECE, lower measured compute time, and lower resource cost. The complete candidate table and selected configuration are uploaded with the aggregate campaign index. A configuration with inactive or non-contributing upper levels cannot win.

## R5 — Adaptive capacity follow-up

Only after R4 selects H, C, and D does the workflow launch the matched static-versus-adaptive comparison. Each seed uses identical training order, task examples, vocabulary, bridge, seed identity, and initial readout parameters.

After epoch one, each hierarchy level evaluates its own observed channel activity. A near-dormant channel may be pruned when its activity is at most 1% of the most active channel, provided the remaining spectrum can preserve positive mass and the existing stability ratio. Otherwise, if observed activity share reaches the 5% growth criterion, a new slower channel is added. Capacity can therefore genuinely increase or decrease; the mechanism is no longer limited to one-for-one channel replacement. Every birth, death, decision, activity value, channel count, mass, and stability value is recorded.

The adaptive stage is one controlled first activation, not a claim that a learned prediction-error policy has already been established. The selected configuration and the complete static matrix remain unchanged.

## R6 — Experiment Lab evidence

Each static job uploads its complete result bundle. The selector collects the 18 bundles, writes a campaign manifest with configuration keys and summary hashes, preserves the full comparison table, and uploads an aggregate index plus winner record. The adaptive matched comparison is uploaded separately. The earlier pilot is preserved as historical/superseded evidence.

## CI topology

One GitHub Actions workflow performs coupling validation, fans out the 18 static matrix jobs concurrently, waits for all of them, selects one eligible configuration, and then starts the adaptive comparison. There are no separate lock-test controls or hidden workflows. A failed or incomplete static matrix cannot trigger adaptive execution.

## Historical pilot reference

Superseded pilot commit: 7568425260efece2c371e26d6eebecdcf93f41dc.
Historical run: https://github.com/johnsnowman488-oss/aberconics-framework/actions/runs/42836766289.
