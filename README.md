 we can delete  for selected runs   and other details that we have seen matter in analysing the results in the graphics




Physics-grounded memory-kernel tooling and runtime infrastructure for Aberconics / HierAberConic-D2C experiments.

This repository currently contains:
- a reusable C++ `gfe_core` library with tests, CLI tools, and an installable package
- `abersoe` single-level augmented-memory runtime components
- `hierarchical_min` multi-level hierarchy, diagnostics, and renormalization utilities
- a C ABI over the core runtime surface
- thin Python `ctypes` and Julia `ccall` wrappers
- reference experiments and generated result artifacts

## What Is Here

### Core areas
- `code/c++ core`
  - primary implementation surface
  - C++ library, apps, tests, and C ABI
- `code/python`
  - typed `ctypes` wrapper over `gfe_c_api.h`
  - wrapper tests and smoke script
- `code/julia`
  - Julia reference module plus C ABI smoke integration
- `results`
  - generated experiment outputs for OU noise and Lorenz runs
- `docs`
  - theory notes and figures

### Implemented runtime surfaces
- SOE kernel fitting with backend selection
- ABERSOE single-level scenario execution
- hierarchy scenario execution
- hierarchy cross-level diagnostics
- hierarchy renormalization reports
- constrained custom chain-style hierarchy assembly through the C ABI and Python wrapper

### Important references
- Top-level technical reference extract: [D2C.md](D2C.md)
- Source PDF: `HierAberConic_D2C_Technical_Reference_v1.0.pdf`
- C++ details: [`code/c++ core/README.md`](code/c++%20core/README.md)
- Python wrapper details: [`code/python/README.md`](code/python/README.md)
- Julia details: [`code/julia/README.md`](code/julia/README.md)

## Current Status

The repo is no longer just a kernel-fitting prototype.

The current foundation includes:
- compiled `gfe_core` with `abersoe`, `hierarchical_min`, diagnostics, renorm, and C API
- shared-library builds for Python/Julia wrapper use
- canonical runtime scenarios for single-level and hierarchical runs
- typed ABI access to hierarchy reports and constrained custom chain specs
- a Python digital token/event substrate with SOE stepping, generic traces,
  explicit-binding retrieval, silence controls, and a learned
  query-conditioned readout
- passing C++ and Python smoke/regression-style tests in the local tree

What is not yet fully implemented from the D2C technical reference:
- predictive-coding training heads
- digital delayed-feedback integration for the existing TD, three-factor, and
  consolidation helpers
- learned binding writing and a full online digital training runtime
- stateful hierarchy stepping for event-by-event Python digital experiments

So the repo is ready for:
- runtime experimentation
- diagnostics/reporting workflows
- Python module development on top of the current ABI

It is not yet a complete end-to-end D2C training system.

## Quick Start

### 1. Build the C++ core

```bash
cmake -S "code/c++ core" -B "code/c++ core/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "code/c++ core/build" -j
```

### 2. Build the shared library for wrappers

```bash
cmake -S "code/c++ core" -B "code/c++ core/build-shared" -DBUILD_SHARED_LIBS=ON
cmake --build "code/c++ core/build-shared" -j
```

### 3. Run core tests

```bash
ctest --test-dir "code/c++ core/build" --output-on-failure
```

### 4. Run the Python wrapper smoke/test path

```bash
export GFE_CORE_LIB="$(pwd)/code/c++ core/build-shared/libgfe_core.so"
python3 code/python/ctypes_smoke.py
pytest -q code/python/tests
```

### 5. Run the Julia C ABI smoke path

```bash
export GFE_CORE_LIB="$(pwd)/code/c++ core/build-shared/libgfe_core.so"
julia code/julia/examples/05_capi_ccall_smoke.jl
```

## Repo Map

```text
.
├── code/
│   ├── c++ core/      # Main library, CLI tools, tests, C ABI
│   ├── python/        # ctypes wrapper and tests
│   └── julia/         # Julia module and ccall wrapper
├── docs/              # Theory notes and figures
├── results/           # Generated experiment outputs
├── D2C.md             # Markdown extract of the technical reference PDF
├── Context.md         # Project context and implementation notes (through 2026-09-15)
├── Step_Context.md    # Active progress ledger for Phase E+F
└── d2c_milestone_f.md # Unified E+F roadmap
```

## Recommended Entry Points

If you are new to the repo:
- read this file first
- then read [`code/python/README.md`](code/python/README.md) for the cleanest wrapper overview
- then read [`code/c++ core/README.md`](code/c++%20core/README.md) for detailed runtime and CLI surfaces

If you want to work from Python:
- start with `code/python/gfe_ctypes.py`
- run `code/python/tests/test_gfe_ctypes.py`

If you want to work from the core:
- start with `code/c++ core/include/gfe/gfe_c_api.h`
- then `code/c++ core/include/abersoe/` and `code/c++ core/src/`

## Visuals

![Lorenz Chaos Suppression](docs/figures/lorenz_before_after.png)

![OU Colored Noise Memory Extension](docs/figures/ou_trajectories_comparison.png)

## E-F0: Runtime-Integrated Digital Reasoning

The original bAbI E0 experiment remains available as the frozen baseline:
code/python/d2c/experiments/babi_qa.py.

The digital F0 runtime adds a persistent digital session, six-phase scheduling,
bounded circular trace storage with continuity-aware windows, stability
monitoring, prediction-change diagnostics, and versioned state checkpoints.
A new bAbI runner is available at code/python/d2c/experiments/babi_f0.py.

The default isolated mode resets episodic contents for each story while
preserving runtime phase history. Continuous mode carries story memory across
boundaries and is a separate interference stress test, not an independent-story
benchmark score. F0 freezes the SOE kernel; PRUNE/GROW are proposal phases only.
See docs/F0_RUNTIME_INTEGRATION.md for controls and run commands.

For a fresh checkout, retrieve the bAbI dataset before running either bAbI
experiment:

```bash
PYTHONPATH=code/python python -m d2c.digital.download_babi
PYTHONPATH=code/python python -m d2c.experiments.babi_f0 --quick
```

The downloader validates all JSONL splits and reuses valid files unless
`--force` is supplied.

## Experiment Lab

The local-first browser Lab runs the registered Python experiments without
requiring a C++ build. From the repository root, install the Python
dependencies and download the bAbI splits if they are not already present.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:PYTHONPATH = "code\python"
.\.venv\Scripts\python.exe -m d2c.digital.download_babi
.\.venv\Scripts\python.exe -m d2c.experiment_lab.server
```

### Linux

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
export PYTHONPATH=code/python
python -m d2c.digital.download_babi
python -m d2c.experiment_lab.server
```

Open <http://127.0.0.1:8765> while the server is running. Stop it with
Ctrl+C. To use another local port, set `D2C_LAB_PORT` before starting the
server (`$env:D2C_LAB_PORT = "8766"` in PowerShell or
`D2C_LAB_PORT=8766 python -m d2c.experiment_lab.server` on Linux).

The Lab has separate **Configure experiment**, **Results & compare**, and
**Datasets & protocols** pages. Choose an experiment and a registered dataset
profile, adjust the settings shown for that protocol, and launch. Use the
**Smoke test** option for a fast 20-story check (E0/E-F0 use two epochs); it is
diagnostic only, not benchmark evidence. Test evaluation is the default locked evaluation;
validation is explicitly exploratory. Saved runs, manifests, logs, hashes,
and result bundles are written to `runs/experiment-lab/`.

The Results page presents experiment-specific findings, readable metric and
stage summaries, and searchable/filterable per-example evidence with paging.
Expand the raw result and log sections when needed. Use **Delete** beside a
saved run to permanently remove its run record and files; running experiments
cannot be deleted.

Dataset profiles are maintained in
[`code/python/d2c/experiment_lab/dataset_profiles.json`](code/python/d2c/experiment_lab/dataset_profiles.json).
Add each local profile with a unique ID, a repository-relative directory, and
the expected filenames `babi_train.jsonl`, `babi_valid.jsonl`, and
`babi_test.jsonl`. The Lab rejects unregistered profiles and directories
outside the repository and records hashes for available splits. See
[`docs/EXPERIMENT_LAB_PLAN.md`](docs/EXPERIMENT_LAB_PLAN.md) for profile,
protocol-default, and comparison details.
