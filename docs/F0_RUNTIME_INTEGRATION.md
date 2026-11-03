# E-F0: Runtime-Integrated Digital Reasoning

Status: implementation milestone for the digital F0 runtime spine.

## Purpose

The original bAbI E0 experiment processes every story through a fresh memory
state and a fixed SOE kernel. Its readout is trained with task labels, but the
experiment does not coordinate persistent runtime phases, trace management,
stability checks, or consolidation scheduling.

E-F0 adds a digital runtime around the existing memory stepper. It is an
orchestration layer, not a replacement for the C++ numerical core and not yet
the complete autonomous learning stack.

## Components

- runtime/digital_runtime.py: persistent runtime, state, episode execution,
  checkpoint/restore, prediction-error diagnostics, stability monitoring.
- runtime/phase_controller.py: WARMUP, EXPLORE, CONSOLIDATE, EVALUATE,
  PRUNE, and GROW scheduling.
- runtime/digital_trace_store.py: bounded circular storage and contiguous
  windows that cannot cross a memory reset or session boundary.
- experiments/babi_f0.py: bAbI runner using the new runtime.
- experiments/babi_qa.py: preserved original E0 baseline.

## Important experimental controls

1. Isolated mode is the default. Each story starts with a zero episodic memory
   state, while runtime phase/counter history persists across the session.
2. Continuous mode carries memory state between stories. It is a cross-story
   interference stress test, not the default independent-story bAbI score.
3. The memory kernel is frozen in E-F0. The readout remains supervised so the
   new runtime can be compared with the existing E0 results. Three-factor
   proposals, learned critics, predictive-coding heads, composite objectives,
   and channel birth/death remain later stages.
4. The bAbI answer readout is called through `DigitalRuntime.evaluate`, so prediction is explicitly phase-accounted. EVALUATE can be entered by the scheduler or temporarily around a task callback. PRUNE and GROW are observable, non-mutating maintenance phases. PRUNE/GROW do not
   change the channel count; those mutations belong to F4.
5. Prediction error in F0 is a transparent last-state predictor diagnostic.
   It is not presented as a trained predictive-coding head.

## Numerical and stability contract

The runtime reuses digital.bridge.step_memory, so the F0 numerical stepping
path is equivalent to the earlier Python digital substrate for an identical
forcing schedule and initial state.

The stability monitor distinguishes formulations. Input-driven memory has a
triangular linear system and is checked for finite states, non-negative kernel
weights, and explicit-Euler decay-step bounds. The negative-feedback form is
additionally checked against the configured analytical margin. The Form-B
margin is not incorrectly used as the stability gate for input-driven runs.

## Run

With the bAbI JSONL splits already available:

    PYTHONPATH=code/python python -m d2c.experiments.babi_f0 --quick
    PYTHONPATH=code/python python -m d2c.experiments.babi_f0 --task 2 --dim 64 --seeds 20
    PYTHONPATH=code/python python -m d2c.experiments.babi_f0 --task 3 --mode continuous

The previous baseline remains available:

    PYTHONPATH=code/python python -m d2c.experiments.babi_qa --task 1 --dim 64 --seeds 20

## Tests and execution policy

The F0 unit/integration tests do not require the external bAbI dataset. They
create tiny JSONL fixtures and test the runtime, phase transitions, checkpoint
restore, ring-buffer windows, formulation-aware stability checks, and bAbI
training/evaluation execution.

The repository GitHub Actions workflow builds the C++ shared library, runs
the full Python test suite, and runs Julia tests on pull requests to main.
Those CI checks are the authoritative integration check where the source
cannot be cloned into a local sandbox.
