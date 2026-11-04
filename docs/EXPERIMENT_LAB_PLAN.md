# Experiment Lab: Browser UI and Results Explorer

Status: local browser UI and result API support E0, E-F0, G1, and the G1-L learned-binding pilot; L0/L1/L2 also run as one reproducible GitHub Actions workflow.

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
- G1-L learned relational binding: learned entity embeddings, role-specific relation encoders, soft object-holder-to-person retrieval, and supervised answer readout over typed QA2 events.

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

## G1-L: learned relational binding

G1-L follows G1 as a learned counterpart to the deterministic relational-state reference. The supported QA2 sentence templates are parsed into typed move, pickup and release events. This parser is a fixed input adapter; the learned part is the role-specific representation, object-holder representation, soft retrieval of a person from the object-held-by vector, and location answer readout. The model is trained from final answer supervision with a NumPy/Adam implementation so the research path does not require a heavyweight ML framework.

The staged workflow executes all three levels in one process:

- **L0 — binding sanity:** controlled multi-person/multi-object cases; verify identity vector separation, holder attention normalization, finite updates and a falling supervised loss on a controlled example.
- **L1 — pilot:** one seed, 200 training stories, 100 validation stories and three epochs by default. It reports validation accuracy before/after training, holder-retrieval accuracy, training loss and per-example attention evidence. The QA2 test split is not read for model selection.
- **L2 — conditional confirmation:** only runs if L0 passes, L1 loss decreases, validation answer accuracy improves over initialization, and holder retrieval exceeds the uniform-person chance baseline by five percentage points. It retrains from scratch on the full training split and evaluates the held-out test split with three matched seeds by default. A failed gate is recorded as a skipped confirmation, not hidden or treated as a failed CI run.

The workflow caches typed event representations in memory and loads each data split once. L1 and L2 reuse the parsed stories. The full F0/SOE runtime is exercised once over the same train/validation/test stories as a fixed companion audit (full variant, isolated episodes, unchanged kernel); its stability and trace statistics are registered. **The G1-L learner currently does not consume F0 hidden-state vectors.** This separation prevents us from implying that F0 state integration has been learned when the current experiment tests a distinct typed-event relational binder. A future fused model can be tested as a separate controlled factor.

Hierarchy is held fixed at the full configuration. No no-slow or collapsed-timescale comparison is included in G1-L; those belong to the later G1-H experiment. Runtime ablation remains deferred.

## Reproducibility, comparison and evidence

Every local run receives a manifest with a unique run ID, experiment, exact command and configuration, code revision, Python version, dataset-split SHA-256 fingerprints and file sizes, timestamps, exit code, and result-file inventory. The Lab exposes allow-listed downloads for summary JSON/Markdown, configuration, manifests, and per-example CSV/JSONL.

The comparison endpoint supports two to five saved runs and displays experiment, accuracy and staged L1/L2 status side by side. The results explorer surfaces G1-L validation and confirmation separately, holder-retrieval accuracy and attention maps, F0 companion audit, stage gates, and per-example evidence.

G1-L results are not directly comparable with E0/E-F0 as though they shared an architecture: the learned relation module consumes typed events, while the baseline systems use token forcing and query-conditioned SOE features. The report states this claim boundary explicitly.

## Local startup

Build the C++ shared library and install the Python runtime dependencies first:

    PYTHONPATH=code/python python -m d2c.digital.download_babi
    cmake -S "code/c++ core" -B "code/c++ core/build-shared" -DBUILD_SHARED_LIBS=ON
    cmake --build "code/c++ core/build-shared" -j
    PYTHONPATH=code/python GFE_CORE_LIB="code/c++ core/build-shared/libgfe_core.so" python -m d2c.experiment_lab.server

Then open http://127.0.0.1:8765.

## API and result files

The local API provides the experiment catalogue, dataset status, run launch/status/history/logs, structured results, two-to-five-run comparison and allow-listed result downloads. Runs are stored under runs/experiment-lab/. G1 is labelled as a deterministic symbolic control; G1-L separates pilot validation from held-out test confirmation and exposes holder-retrieval evidence.

A GitHub Actions workflow runs G1 parser/transition tests, the E0/E-F0 comparison and the separate G1 QA2 diagnostic, then uploads a Lab register and full result bundle for remote inspection.
