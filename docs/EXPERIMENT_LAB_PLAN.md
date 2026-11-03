# Experiment Lab: Browser UI and Results Explorer

Status: proposed implementation plan. The interface and API described here are not yet implemented.

## Goal

Give the researcher a local browser-based control panel to select, configure, launch, monitor, compare, and inspect experiments without using a terminal. The UI must operate the repository's existing Python runners and C++ core; it must not reimplement scientific logic in JavaScript.

## Design principles

1. Reproducibility: record experiment, exact parameters, seeds, dataset/split identity, code revision, timestamps, exit status, and output locations for every run.
2. One source of truth: Python runners write structured result files; the UI visualizes those files. Never infer or fabricate metrics.
3. Human- and LLM-readable: JSON metadata/metrics, CSV tables, Markdown summary, and raw logs with stable field names and units.
4. Safe execution: expose only allow-listed experiments and validated parameters. Never accept arbitrary shell commands, Python expressions, or filesystem paths from the browser.
5. Local-first: run a small service on the researcher's machine and open it at localhost. No cloud account is needed for the first release.
6. Progressive: begin with E0 and E-F0 bAbI, then add experiments through a registry and common result contract.

## Proposed architecture

```text
Browser UI (HTML + CSS + JavaScript)
  | HTTP + JSON
  v
Local Python Experiment Lab service
  |-- experiment registry and parameter validation
  |-- run manager: start / status / logs / cancel
  |-- results index and comparison checks
  |
  +--> existing E0 and E-F0 Python runners --> C++ shared core
  |
  +--> runs/<run-id>/
       config.json, metadata.json, metrics.json
       per_seed.csv, summary.md, stdout.log, stderr.log
       optional plots and detailed traces
```

The first service can use Python's standard-library HTTP server to avoid a new framework dependency. A small maintained web framework can be considered later if asynchronous job management warrants it. Keep the browser layer plain HTML, CSS, and JavaScript.

## User experience

### Experiment catalogue

List available experiments, descriptions, implementation status, inputs, and whether each is a smoke test, benchmark, or stress test. Initial entries:
- E0 bAbI baseline (babi_qa.py)
- E-F0 runtime-integrated bAbI (babi_f0.py)

### Configuration panel

Expose only parameters supported by the selected runner. For the first bAbI workflow: task number, model dimension, seed count or explicit seed list, execution mode (isolated or continuous where supported), quick versus full run, and dataset/split availability. Show defaults, valid ranges, and clear descriptions. Never silently alter runner defaults.

### Run control

Allow launch, status tracking, elapsed time, recent log viewing, and graceful cancellation. A run must continue if the browser tab closes. Retain run history across service restarts. Initially allow one active job at a time to avoid CPU contention and confusing comparisons.

### Results dashboard

- Accuracy: E0 versus E-F0, with per-seed points and mean plus uncertainty.
- Per-seed table: seed, evaluation count, accuracy, duration, and status.
- Task-level comparisons only when the runner actually emits comparable task metrics.
- Runtime diagnostics when present: phase counts, stability checks and violations, prediction-change diagnostics, trace records retained, resets, and session counts.
- Run metadata, raw logs, and downloads.

Use a chart when it clarifies the data and a table when exact values matter more. Show 'not recorded' for absent metrics, never zero. Label continuous-mode interference results separately from standard isolated-mode bAbI scores.

### Compare and export

Allow selection of completed runs. Before comparing scores, check task, dataset/split, evaluation protocol, dimension, and other relevant parameters. Highlight mismatches rather than presenting incomparable scores as a fair head-to-head result. Each run should be a portable directory with JSON, CSV, Markdown, and logs.

## Common result contract

Every runner should eventually emit a versioned result bundle.

metadata.json: schema_version, run_id, experiment_id/name, code_revision, UTC timestamps, status, exit_code, parameters, dataset/source/version, split fingerprints, and environment details.

metrics.json: named metrics with definitions and units, plus optional per-seed or per-episode series. Omit unavailable metrics or mark them unavailable; never fabricate them.

per_seed.csv: one row per seed with stable column names.

summary.md: experiment purpose, protocol, results, caveats, and paths to detailed data.

Include stdout and stderr logs. Version the schema so future experiments can extend it without breaking older readers.

## Local API outline

- GET /api/experiments — catalogue and supported parameter schemas.
- GET /api/datasets — locally available datasets and validation state.
- POST /api/runs — launch an allow-listed experiment with validated config.
- GET /api/runs — list prior runs and statuses.
- GET /api/runs/{id} — metadata and current status.
- GET /api/runs/{id}/logs — bounded recent log output.
- POST /api/runs/{id}/cancel — request graceful cancellation.
- GET /api/runs/{id}/results — structured result bundle.
- GET /api/runs/{id}/files/{name} — download allow-listed result files.

Bind to localhost by default. Validate run IDs and file names, prevent path traversal, enforce one active job in the MVP, and do not expose a generic command-execution endpoint.

## Delivery plan

### Phase A — Result contract and runner integration

1. Inspect the current E0 and E-F0 CLI arguments and output formats.
2. Add a shared result writer without changing numerical algorithms.
3. Make both runners emit metadata, metrics, per-seed CSV, and a readable summary in a unique run directory.
4. Test schema validity, deterministic metadata, and incomplete/failed runs.

Acceptance: a terminal-launched E0 or E-F0 run produces a complete, self-describing bundle that can be analysed without parsing console output.

### Phase B — Local service and run manager

1. Add a local Python service and experiment registry.
2. Validate parameters against each runner's declared schema.
3. Launch only known runner modules using argument arrays, never shell strings.
4. Capture logs, persist job status, support graceful cancellation, and retain history.
5. Add API and security tests.

Acceptance: a user can start, monitor, and retrieve a run through the local API, with no arbitrary command execution path.

### Phase C — Browser UI

1. Build the experiment catalogue and configuration forms.
2. Add run status, elapsed time, and live log tail.
3. Build result views with charts, tables, metadata, and downloads.
4. Add run comparison with comparability checks.
5. Make the layout responsive for laptop use.

Acceptance: the standard E0/E-F0 workflow can be completed in the browser without opening a terminal.

### Phase D — First scientific workflow

1. Validate the UI with quick task-1 runs for E0 and E-F0.
2. Run a matched isolated-mode multi-seed comparison using identical splits, dimension, evaluation protocol, and seed list.
3. Inspect per-seed results and uncertainty; preserve raw bundles.
4. Run continuous mode separately as a cross-story interference stress test.
5. Record limitations and choose the next experiment from the evidence.

Acceptance: the comparison is independently reproducible from saved configuration and result bundles.

### Phase E — Extend to later experiments

Add later experiments through the same registry and result contract. Add charts only for meaningful emitted metrics, and always preserve access to raw tables and exports.

## Out of scope for the first release

- Public internet-facing hosting or multi-user authentication.
- Arbitrary notebook execution or a browser terminal.
- Exposing model parameters unsupported by the underlying runner.
- Replacing Python/C++ execution with JavaScript.
- Automated scientific conclusions presented as ground truth.
- Parallel job scheduling before resource controls exist.

## Immediate next actions

1. Inspect the E0 and E-F0 runner CLIs and agree on the first parameter contract.
2. Define and test the versioned result bundle.
3. Implement Phase A before charts, so the UI and LLM analysis consume the same trustworthy data.