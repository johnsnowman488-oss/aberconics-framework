# Python D2C Workspace

This folder is the dedicated home for Python-side HierAberConic-D2C work.

It exists to keep three concerns separate:
- low-level ABI bindings in `code/python/gfe_ctypes.py`
- higher-level D2C runtime and experiment logic
- progress notes and experiment-specific internal files

## Current Contents

- ABI-facing convenience schemas and scenario/report helpers
- Lorenz runtime orchestration, traces, scheduling, stability checks, and
  learning scaffolding
- `digital/`: deterministic event streams, forcing bridges, a Python SOE
  stepper, generic traces, `DigitalDirector`, retrieval baselines, and readouts
- `experiments/symbolic_induction.py`: explicit-binding long-gap retrieval
- `experiments/thinking_between_tokens.py`: evolving versus frozen silence and
  elapsed-interval probes
- `experiments/learned_symbolic_retrieval.py`: learned query-conditioned
  readout over frozen D2C state
- `experiments/d3_randomized.py`: independently sampled indexed rule retrieval
  with learned terminal-query-conditioned soft eligibility and causal
  value-intervention diagnostics

## Working Rule

Use:
- `Context.md` for repo-wide milestones
- this folder for Python D2C implementation work and experiment-specific notes

## Digital D2C Status

D0, D1, and D2 are implemented at single-level scope. The supported narrow
claims are that separated slow SOE channels retain explicit bindings longer
than the tested ablations, and that evolving state carries elapsed-time
information. The learned readout generalizes query-conditioned selection over
held-out assignments, but the outer-product binding writer is still supplied
explicitly; this is not learned binding or general symbolic reasoning.

D3-R now provides the non-canonical delayed-feedback surface: it trains only
the eligibility/action readout over fixed SOE memory and uses supplied
slot/value coordinates as a bounded binding control. Its initial calibration
does not yet establish long-delay distractor robustness or a full/no-slow
advantage. Adaptive kernels and stateful hierarchy stepping remain later work.

## Reference Plan

The next planned documents and modules are described in:
- [`../PYTHON_D2C_PLAN.md`](../PYTHON_D2C_PLAN.md)
