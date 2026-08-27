# Python D2C Plan

This document records the Python-facing implementation track for
HierAberConic-D2C on top of the current C++ core, C ABI, and `ctypes` wrapper.
The original Phases 1–5 are substantially implemented; the active next step is
to integrate delayed-feedback learning into the digital runtime.

The goal is not to rebuild the C++ engine in Python.
The goal is to build the Python-side application layer that can:
- orchestrate experiments
- manage stateful runs and traces
- expose cleaner D2C-facing APIs than raw `ctypes`
- host learning/runtime logic that does not yet belong in the C++ substrate

## Why This Plan Exists

The repository already has a strong lower-level foundation:
- C++ runtime for single-level and hierarchical runs
- typed C ABI for runtime/report surfaces
- Python `ctypes` wrapper with tested access to:
  - SOE fitting
  - `abersoe` scenarios
  - hierarchy scenarios
  - cross-level reports
  - renorm reports
  - constrained custom chain specs

The Python-side D2C layer now includes Director-style orchestration, trace
storage, training/learning scaffolding, reproducible experiment modules, and
a single-level digital token/event substrate. What remains missing is online
digital delayed-feedback learning, learned binding writing, and a stateful
hierarchy-step surface for Python.

## Proposed Python Workspace

Recommended home:
- `code/python/d2c/`

Recommended responsibilities:
- `code/python/gfe_ctypes.py`
  - stay thin and ABI-focused
  - do not turn into the D2C application layer
- `code/python/d2c/`
  - D2C-facing Python modules
  - experiment definitions
  - progress notes
  - runtime orchestration

This separation keeps:
- ABI binding concerns isolated
- higher-level research/runtime code easier to evolve
- future migration from `ctypes` to another binding strategy possible without rewriting the D2C layer

## Proposed Module Layout

Initial target layout:

```text
code/python/d2c/
├── README.md
├── __init__.py
├── ffi.py
├── schemas.py
├── reports.py
├── scenarios.py
├── runtime/
│   ├── __init__.py
│   ├── director.py
│   ├── scheduler.py
│   ├── trace_store.py
│   └── stability.py
├── learning/
│   ├── __init__.py
│   ├── predictive.py
│   ├── critics.py
│   ├── updates.py
│   └── consolidation.py
├── experiments/
│   ├── __init__.py
│   ├── ou_noise.py
│   ├── lorenz63.py
│   ├── hierarchy_ablation.py
│   ├── multi_sine_prediction.py
│   ├── delayed_reward.py
│   └── continual_learning.py
├── configs/
│   ├── __init__.py
│   ├── kernels.py
│   ├── hierarchy.py
│   └── training.py
└── progress/
    └── README.md
```

This should be treated as an implementation target, not a rigid final shape.

## Implementation Status and Development Phases

### Phase 1: Python D2C shell over the existing wrapper — implemented

Deliver:
- `d2c/ffi.py`
  - thin convenience layer over `gfe_ctypes`
  - centralizes library loading and common conversions
- `d2c/schemas.py`
  - typed Python dataclasses or `TypedDict` structures for:
    - kernels
    - level specs
    - edge specs
    - run results
    - cross-level report rows
    - renorm analyses
- `d2c/reports.py`
  - report normalization helpers
  - CSV-to-structured helpers only if needed
- `d2c/scenarios.py`
  - canonical Python entrypoints for:
    - `abersoe` scenarios
    - hierarchy scenarios
    - constrained custom chains

Definition of done:
- Python callers no longer need to work directly against raw `gfe_ctypes` output dictionaries for common workflows
- the new layer remains a wrapper over the current tested ABI, not a duplicate engine

### Phase 2: Experiment organization and reproducibility — implemented

Deliver:
- experiment modules for:
  - OU noise
  - Lorenz63
  - hierarchy ablations
- repeatable run config objects
- standard output locations for:
  - summaries
  - cross-level reports
  - renorm reports
  - plots and metrics
- progress logging conventions under `d2c/progress/`

Definition of done:
- every core experiment can be launched from Python in a repeatable way
- results are emitted in predictable locations
- ablations do not require direct CLI usage

### Phase 3: Runtime orchestration layer — implemented for Lorenz; digital core implemented

Deliver:
- `runtime/director.py`
  - early `AberconicsDirector` shell
- `runtime/trace_store.py`
  - contiguous window trace storage
- `runtime/scheduler.py`
  - state/phase scheduling logic
- `runtime/stability.py`
  - Python-side stability checks and thresholds

Important scope rule:
- this phase is about orchestration first
- do not block on having full TD learning or predictive coding before the runtime shell exists

Definition of done:
- Python can manage stateful experiment loops without re-encoding the logic in notebooks or ad hoc scripts
- trace collection and stability inspection become reusable components

### Phase 4: Learning scaffolding — implemented, digital integration pending

Deliver:
- predictive-coding placeholders and interfaces
- critic/value-head interfaces
- three-factor update scaffolding
- consolidation policy scaffolding

Important note:
- the first pass should focus on interface design and integration points
- we do not need to solve the entire learning problem before defining the runtime contract

Definition of done:
- the D2C learning pieces have explicit Python homes
- experiments can begin to attach learned heads without destabilizing the core runtime surface

### Phase 5: Language/discrete bridge — implemented at synthetic-task scope

Deliver:
- token-to-forcing adapters
- continuous-to-discrete readout shells
- synthetic long-context experiments

Definition of done:
- the repo has a real place for Section 7 of `D2C.md`
- token-style experiments can be developed without contaminating the low-level wrapper layer

Implemented additions:

- `d2c/digital/` supplies vocabulary and event schemas, token/binding forcing,
  SOE stepping, traces, a generic Director, deterministic decoding, and a
  dependency-free trained query-conditioned MLP readout.
- D1/D2 entrypoints cover noisy long-gap retrieval, matched silence controls,
  and elapsed-interval classification.

Active next step:

- connect `d2c/learning` TD, three-factor, and consolidation primitives to
  `DigitalDirector` phases for D3 delayed-feedback tasks. Keep kernel weights
  and explicit bindings frozen in the first experiment.

## Recommended ABI Additions

These are the highest-value next ABI additions for Python D2C work.

### 1. Stateful step-level execution

Need:
- single-step execution for `abersoe`
- single-step execution for constrained hierarchy runs

Why:
- the current ABI is run-oriented
- Director/TraceStore logic will be much cleaner if Python can step live state instead of re-running short episodes repeatedly

Recommended shapes:
- `gfe_c_abersoe_step(...)`
- `gfe_c_hierarchical_step_chain_spec(...)`

### 2. Mutable-state roundtrip helpers

Need:
- explicit helpers for:
  - state validation
  - active kernel retrieval after top-down modulation
  - sampled state export for online loops

Why:
- Python-side training/runtime logic needs to inspect evolving state without relying only on final summaries

### 3. Stability-monitor support

Need:
- lightweight ABI surface for stability ratios / checks tied to the theorem-enforced constraints

Why:
- the D2C runtime wants a `StabilityMonitor`
- Python should not have to reconstruct all enforcement semantics from logs or CSVs

### 4. Optional future training-facing ABI

Do not rush this, but likely future candidates:
- per-level prediction error buffers
- writable kernel update entrypoints
- channel growth/pruning hooks

These should come only after the step-level runtime surface proves stable.

## Experiment Milestones

### Milestone A: Foundation parity

Experiments:
- OU colored-noise reproduction
- Lorenz63 chaos-suppression reproduction
- hierarchy scenario smoke parity from Python

Success criteria:
- Python-run outputs match current C++ expectations
- no hidden wrapper drift

### Milestone B: Hierarchy diagnostics baseline

Experiments:
- `two_level_bidirectional` ablations
- `three_level_homeostatic` ablations
- custom 2-level chain experiments through Python spec assembly

Success criteria:
- bottom-up only, top-down only, and bidirectional runs show expected directional changes
- cross-level and renorm reports are stable enough to compare automatically

### Milestone C: D2C-planned synthetic tasks

Experiments:
- multi-sine multi-timescale prediction
- delayed reward control
- continual learning without forgetting

Mapped to `D2C.md` planned experiments:
- multi-timescale prediction
- delayed-reward control
- continual learning without forgetting

Success criteria:
- Python experiment harness exists for each task
- reports include `D_eff`, `M_cap`, `M_scale`, and outcome metrics

### Milestone D: Long-context and token bridge

Experiments:
- synthetic long-context sequence modeling
- token-to-forcing adapter tests
- continuous-to-discrete readout tests

Success criteria:
- the Python D2C layer can host token-like experiments without requiring a full production language model stack

## Progress Logging

Recommended rule:
- record major and minor milestones in `Context.md`
- keep Python-subsystem-specific working notes under `code/python/d2c/progress/`

Suggested use:
- `Context.md`
  - repo-wide milestone notebook
- `code/python/d2c/progress/`
  - experiment notes
  - internal TODO snapshots
  - short status memos

## Practical Recommendation

Yes, it is a good idea to create a dedicated folder under `code/python` for:
- D2C experiments
- progress files
- internal supporting folders

Recommended folder:
- `code/python/d2c/`

This is the right time to do it because:
- the wrapper layer is already mature enough to support higher-level work
- keeping D2C work separate from `gfe_ctypes.py` will reduce confusion
- the repo needs a clean home for Python-side orchestration before learning code starts to spread into ad hoc scripts
