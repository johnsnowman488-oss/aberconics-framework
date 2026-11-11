# G4.2 Training-Duration Check on the Three Retained Candidates

## Purpose

The initial G4.2 H×C×D campaign fixed training at three epochs. That was appropriate for a concurrent architecture screen, but it did not establish whether the selected candidates had converged or whether their ranking depended on training duration.

This follow-up holds architecture and learning settings constant while extending each retained candidate to twelve epochs. It records validation checkpoints at epochs 3, 5, 8, and 12 from the same uninterrupted training trajectory.

## Candidates

Three independent jobs execute concurrently:

| Candidate | Depth | Capacity | Dimension | Rationale |
|---|---:|---:|---:|---|
| H4-C6-D64 | 4 | 6 | 64 | Current QA3-first selector winner |
| H2-C4-D64 | 2 | 4 | 64 | Efficient alternative with stronger calibration |
| H2-C6-D64 | 2 | 6 | 64 | Strong QA5 alternative |

## Frozen controls

- G2 E-F0 spine and corrected G4.2 vector-coupled hierarchy.
- Learning rate 0.05 and feature scale 10.
- Three matched training seeds: 0, 1, and 2.
- Same train/validation files, answer vocabulary construction, bridge, seed identities, and deterministic example-order generator.
- The first three epoch orders for each seed are identical to the original G4.2 three-epoch protocol.
- QA3 and QA5 are evaluated on validation only. The held-out test split is not downloaded or opened.
- Static topology only. Prune/grow remains disabled for this comparison.

## Checkpoint measurements

At the end of epochs 3, 5, 8, and 12, each seed is evaluated against the same validation examples. The run records:

- accuracy and correct/total counts;
- fixed-bin top-label ECE;
- negative log-likelihood;
- per-seed training loss history;
- final confidence, stability, finite-state, cross-level-coupling, and readout evidence.

The campaign aggregates checkpoint accuracy across the three seeds and retains each seed's checkpoint record. Candidate ranking is not automatically changed by this run: the result is a training-duration diagnosis, not a new architecture sweep or a test-set evaluation.

## Interpretation

- Continued improvement from epoch 8 to 12 indicates the candidate may still be learning.
- A plateau suggests diminishing benefit from additional training.
- A drop in validation accuracy or worsening calibration after a peak is evidence consistent with overfitting or instability; it must be interpreted alongside per-seed variation and loss curves.
- A changed ranking across checkpoints means the three-epoch screen selected partly for early learning speed rather than only eventual performance.

This is a screening follow-up with three seeds. It should guide the next confirmatory design, not be represented as a production-training prescription.

## Separation from adaptive capacity work

The G4.2 adaptive follow-up remains a separate, unresolved engineering task. The earlier run recorded channel births but no deaths. Its permissive growth threshold and unexercised pruning decision are not changed by this duration experiment. The next prune/grow correction must provide activity-share diagnostics, explicit retain/prune/grow reasons, observed topology deltas, stepper-rebuild verification, and integration tests that prove both birth and death paths can mutate the active hierarchy.
