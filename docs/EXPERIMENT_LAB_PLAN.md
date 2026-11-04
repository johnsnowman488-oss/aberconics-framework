# Experiment Lab: Browser UI and Results Explorer

Status: local browser UI and result API support E0, E-F0, and G1.

## Goal

Give the researcher a local browser-based control panel to select, configure, launch, monitor, compare, and inspect experiments without using a terminal. The UI operates the repository's Python runners and C++ core; it does not reimplement scientific logic in JavaScript.

## Design principles

1. Reproducibility: record experiment, exact parameters, seeds, dataset/split identity, code revision, timestamps, exit status, and output locations for every run.
2. One source of truth: Python runners write structured result files; the UI visualizes those files. Never infer or fabricate metrics.
3. Human- and LLM-readable: JSON metadata/metrics, CSV tables, Markdown summary, and raw logs with stable field names and units.
4. Safe execution: expose only allow-listed experiments and validated parameters. Never accept arbitrary shell commands, Python expressions, or filesystem paths from the browser.
5. Local-first: run a small service on the researcher's machine and open it at localhost.
6. Progressive: add experiments through a registry and common result contract.

## Current experiment catalogue

- E0 bAbI baseline: supervised linear or MLP readout over frozen SOE state.
- E-F0 runtime-integrated bAbI: supervised readout with F0 runtime orchestration and diagnostics.
- G1 explicit relational-state QA2 diagnostic: deterministic event parser and explicit person/object/location state, while the full F0/SOE runtime consumes the same story tokens.

## G1 interpretation boundary

G1 currently supports bAbI Task 2 (QA2) only. The state keeps person-location and object-holder/object-location relationships separately. A held object's current location is derived from its holder's latest location; release converts the carried relation to a fixed location. The event ledger records every state transition.

G1 performs no supervised fitting and has no random seeds. It is a deterministic symbolic control and parser-coverage diagnostic, not a learned entity-binding result and not directly comparable as a trained model against E0/E-F0. Its result bundle records:

- SHA-256 fingerprints for the source train and test splits;
- exact run configuration and claim boundary;
- held-out accuracy, binomial standard error and Wilson 95% interval;
- statement parsing and query-resolution coverage;
- move/pickup/release counts, actors and objects;
- F0 phase, step, trace and stability diagnostics;
- per-example predictions, final world states, event ledgers and parse errors;
- summary JSON/Markdown, metadata, configuration, per-example CSV and JSONL.

The local Lab catalogue registers G1. It restricts the task to QA2, hides irrelevant seed/epoch controls, displays G1 as a deterministic diagnostic, and exposes result files through a filename allow-list.

## Hierarchy and next learned experiment

G1 holds the current full SOE/F0 configuration fixed. Hierarchical-memory conditions are not changed in this first relational-state baseline. A separate G1-H matched experiment will test full, no-slow, and collapsed-timescale conditions after the explicit relation mechanism is established.

The subsequent learned milestone should use the G1 event/state ledger as a trusted control for learning identity resolution, semantic role assignment, relation updates and query-conditioned retrieval. Symbolic G1 accuracy must not be described as evidence that the learned pathway has acquired these capabilities.

## Local startup

Build the C++ shared library and install the Python runtime dependencies first:

    PYTHONPATH=code/python python -m d2c.digital.download_babi
    cmake -S "code/c++ core" -B "code/c++ core/build-shared" -DBUILD_SHARED_LIBS=ON
    cmake --build "code/c++ core/build-shared" -j
    PYTHONPATH=code/python GFE_CORE_LIB="code/c++ core/build-shared/libgfe_core.so" python -m d2c.experiment_lab.server

Then open http://127.0.0.1:8765.

## API and result files

The local API provides the experiment catalogue, dataset status, run launch/status/history/logs, structured results and allow-listed result downloads. Runs are stored under runs/experiment-lab/. G1 is exposed in run history like E0/E-F0, but its report labels the deterministic symbolic method and its distinct uncertainty interpretation.

A GitHub Actions workflow runs G1 parser/transition tests, the E0/E-F0 comparison and the separate G1 QA2 diagnostic, then uploads a Lab register and full result bundle for remote inspection.
