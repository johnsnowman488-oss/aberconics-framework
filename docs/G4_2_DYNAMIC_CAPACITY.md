# G4.2 Corrected Dynamic Channel-Capacity Experiment

## Purpose

The static training-duration campaign established fixed-topology learning curves for H2-C4-D64, H2-C6-D64, and H4-C6-D64. This experiment preserves those three architectures and measures whether bounded, evidence-driven channel pruning and growth improve or harm their validation performance.

This is a channel-capacity experiment only. Hierarchy depth and per-input level activation remain fixed and are not part of this stage.

## Candidate matrix and execution

Three candidate jobs run concurrently:

| Candidate | Depth | Initial capacity | Dimension |
|---|---:|---:|---:|
| H4-C6-D64 | 4 | 6 | 64 |
| H2-C4-D64 | 2 | 4 | 64 |
| H2-C6-D64 | 2 | 6 | 64 |

Each candidate runs matched **static** and **prune_grow** arms, with seeds 0, 1, and 2, 12 epochs, and validation checkpoints at epochs 3, 5, 8, and 12. The same training orders and readout initialization are used between the two arms. QA3 and QA5 are evaluated on validation only; the held-out test split is not downloaded or opened.

The static arm is deliberately rerun within each candidate job as a matched control. The existing G4.2 duration campaign remains the historical fixed-topology baseline and can also be compared after this run completes.

## Corrected activity measurement

The previous implementation sampled activity only from the final memory state of each episode. The corrected director accumulates each channel's RMS memory activation after every native transition across all training episodes in an epoch.

Activity is averaged over the whole training window. Validation episodes call the hierarchy with activity recording disabled, so validation cannot influence prune/grow decisions.

## Pruning policy

- A channel is considered dormant when its activity is no more than 3% of the most active channel in that level.
- A channel must meet that criterion for two consecutive epoch windows before it becomes prune-eligible.
- A level retains at least two channels.
- A prune is accepted only if the remaining positive spectrum can preserve the original kernel mass and stability ratio.
- If a candidate spectrum is infeasible, the event records the rejection and the level retains its current topology unless a separately valid growth decision applies.

The two-window rule avoids pruning on one noisy or transiently inactive window.

## Growth policy

Growth is based on the effective active-channel count:

N_eff = 1 / sum(channel_share_i ** 2)

The level may grow only when N_eff / channel_count >= 0.80 and total observed activity exceeds the numerical floor. This detects broad participation across channels. It replaces the old reversed maximum-share condition, which incorrectly interpreted a concentrated maximum share as evidence for growth.

Growth is bounded to at most two additional channels per level relative to its initial count. A two-window cooldown follows each birth. The added channel uses a slower positive decay rate, and its positive spectrum is fitted to preserve the original kernel mass and stability ratio.

## Mutation and runtime integrity

Every epoch boundary produces one explicit decision per level: prune, grow, or retain. Each record includes:

- time-integrated activity and normalized channel shares;
- relative activity, effective active-channel count, and eligible dormant indices;
- operation, reason, and any rejected prune candidates;
- channel counts and capacity delta before/after;
- kernel mass and stability ratio before/after;
- epoch, activity-window sample count, dormancy streaks, cooldown, and per-level cap.

Every accepted edit rebuilds the affected native C++ vector stepper. Integration tests exercise both birth and death through the actual hierarchy director and then run another episode using the changed state shape. Pure policy tests also verify that concentrated activity does not trigger growth and that pruning waits for director eligibility.

A no-mutation result in the benchmark is allowed: retain may be the correct policy decision. CI validates that all decisions are accounted for and justified; the tests independently prove that both structural-edit paths are executable. Mutation counts are reported as experimental outcomes, not treated as an optimization objective.

## Acceptance and interpretation

The workflow requires:

- all three candidates and both arms to complete;
- QA3/QA5 checkpoint coverage for all matched seeds;
- matched initial readout weights and deterministic training orders;
- one decision per level at every training epoch for the adaptive arm;
- activity and reason evidence for every decision;
- finite states, stability within the 0.9 margin, and observed cross-level communication;
- exact kernel-mass and stability preservation for every accepted edit;
- integration coverage for actual pruning, growth, stepper rebuild, and post-edit execution.

The comparison manifest reports per-checkpoint accuracy, ECE, NLL, compute, paired static/adaptive deltas, mutation counts, reasons, and final channels per level and seed. Results are descriptive with three seeds and validation examples; they do not establish statistical significance or justify opening the held-out test split.

## Separation from full adaptive hierarchy

This stage changes channels inside each fixed hierarchy level. The number of levels and their participation remain unchanged. Structural level addition/removal and per-input level activation are reserved for later experiments, so this result will isolate channel-capacity adaptation.
