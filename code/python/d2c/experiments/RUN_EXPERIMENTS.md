# Running the Experiments

These commands are written for Windows PowerShell and should be run from the
repository root (`D:\Projectfile\aberconics-framework`). The experiment modules
are under `code\python\d2c\experiments`; commands use `python -m` so package
imports work consistently.

## 1. Fresh setup, build, and tests

### Prerequisites

- Python 3.10 or newer (`python --version`)
- MSYS2 MinGW-w64 tools installed under `C:\msys64\mingw64\bin`
- Git, available from PowerShell

The repository build script uses the MinGW compiler at that exact path. MSVC
alone is not sufficient for this script.

### Create the virtual environment and install Python dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Build the C++ core and shared library, then run all tests

```powershell
.\scripts\build_and_test.ps1 -SkipVenv -Clean
```

The script builds the static C++ core and tests, runs CTest, builds
`code\c++ core\build-shared\libgfe_core.dll`, and runs the Python suite. Omit
`-SkipVenv` if the virtual environment still needs to be created and the
dependencies installed. Omit `-Clean` to reuse existing CMake build folders.

To run tests only after a successful build:

```powershell
$env:PATH = "C:\msys64\mingw64\bin;$env:PATH"
ctest --test-dir "code\c++ core\build" --output-on-failure
$env:PYTHONPATH = "code\python"
.\.venv\Scripts\python.exe -m pytest code\python\tests -q
```

To build only, without running tests:

```powershell
$env:PATH = "C:\msys64\mingw64\bin;$env:PATH"
cmake --build "code\c++ core\build" --parallel
cmake --build "code\c++ core\build-shared" --parallel
```

### Environment for running experiments

Run once in each new PowerShell window:

```powershell
$env:PATH = "C:\msys64\mingw64\bin;$env:PATH"
$env:PYTHONPATH = "code\python"
$Py = ".\.venv\Scripts\python.exe"
```

The `PATH` entry lets Windows find MinGW runtime DLLs used by the shared core.
Experiments that call the C++ ABI need the shared-library build above.

## 2. bAbI data

The bAbI experiments require the train and test JSONL files. They are stored
under the ignored progress directory and may not be present in a fresh clone.
Download them if these files are missing:

```powershell
$Data = "code\python\d2c\progress\babi_data"
New-Item -ItemType Directory -Force $Data | Out-Null
curl.exe -L --fail "https://huggingface.co/datasets/Muennighoff/babi/resolve/main/babi_train.jsonl" -o "$Data\babi_train.jsonl"
curl.exe -L --fail "https://huggingface.co/datasets/Muennighoff/babi/resolve/main/babi_test.jsonl" -o "$Data\babi_test.jsonl"
curl.exe -L --fail "https://huggingface.co/datasets/Muennighoff/babi/resolve/main/babi_valid.jsonl" -o "$Data\babi_valid.jsonl"
```

## 3. CLI experiments

Use `--help` on these modules for the full option list. The commands below run
the default experiment unless options are shown.

### bAbI QA (`babi_qa`)

Quick smoke run:

```powershell
& $Py -m d2c.experiments.babi_qa --task 1 --variant full --dim 16 --seeds 1 --quick
```

Standard QA1 run (the script defaults to 3 seeds):

```powershell
& $Py -m d2c.experiments.babi_qa --task 1 --variant full --dim 64 --seeds 3
```

Full 20-seed run:

```powershell
& $Py -m d2c.experiments.babi_qa --task 1 --variant full --dim 64 --seeds 20
```

Variants are `full`, `no_slow`, and `collapsed_gamma`; tasks are selected with
`--task`. Results default to `code\python\d2c\progress\babi_qa`.

### Symbolic induction (`symbolic_induction`)

```powershell
& $Py -m d2c.experiments.symbolic_induction --variant full --seeds 8
```

Results default to `code\python\d2c\progress\symbolic_induction`.

### Thinking between tokens (`thinking_between_tokens`)

```powershell
& $Py -m d2c.experiments.thinking_between_tokens --variant full --gap 16 --silence-steps 4 --seeds 16
```

### Temporal logic (`temporal_logic`)

```powershell
& $Py -m d2c.experiments.temporal_logic --variant full --gap 32 --epochs 30
```

Optional switches select credit/eligibility behavior. For example:

```powershell
& $Py -m d2c.experiments.temporal_logic --forced-distractors --learned-eligibility --adaptive-kernel
```

### Learned symbolic retrieval (`learned_symbolic_retrieval`)

```powershell
& $Py -m d2c.experiments.learned_symbolic_retrieval --variant full --gap 32 --train-count 128 --test-count 64 --epochs 30
```

### Lorenz63 (`lorenz63`)

Default paired Python-baseline/C++-memory comparison:

```powershell
& $Py -m d2c.experiments.lorenz63 --library-path "code\c++ core\build-shared\libgfe_core.dll"
```

Multi-initial-condition batch:

```powershell
& $Py -m d2c.experiments.lorenz63 --batch --library-path "code\c++ core\build-shared\libgfe_core.dll"
```

Online learning episode (default 2000 steps):

```powershell
& $Py -m d2c.experiments.lorenz63 --learning --learning-steps 2000 --library-path "code\c++ core\build-shared\libgfe_core.dll"
```

Add `--integration-method euler` to select Euler instead of the default RK4
Python baseline. Outputs default to the Lorenz63 progress directory.

## 4. bAbI research scripts

These scripts run fixed sweeps and do not accept CLI flags. They require the
bAbI files from section 2 and can take considerably longer than the smoke
tests.

### QA1 full/no-slow/collapsed-gamma ablation

```powershell
& $Py -m d2c.experiments.e0_ablation_run
```

### QA1 convergence and hyperparameter sweep

```powershell
& $Py -m d2c.experiments.e0_convergence_test
```

### QA1 dimension-32 ablation

```powershell
& $Py -m d2c.experiments.e0_dim32_ablation
```

### Adaptive-kernel pilot

This writes per-seed JSON files. Create the output folder first; the runner
does not create it automatically.

```powershell
$Out = "code\python\d2c\progress\e0_adaptive_pilot"
New-Item -ItemType Directory -Force $Out | Out-Null
& $Py -c "from d2c.experiments.e0_adaptive_pilot import main; main(output_dir=r'$Out')"
```

### Signal-repair comparison

This also writes per-seed JSON files and expects the output folder to exist.

```powershell
$Out = "code\python\d2c\progress\e1_signal_repair"
New-Item -ItemType Directory -Force $Out | Out-Null
& $Py -c "from d2c.experiments.e1_signal_repair import main; main(output_dir=r'$Out')"
```

For a shorter check, run just the frozen arm for one seed:

```powershell
& $Py -c "from d2c.experiments.e1_signal_repair import main; main(output_dir=r'$Out', seeds=range(101, 102), arms=('frozen',))"
```

## 5. Programmatic experiment runners

The following modules provide runner functions, not standalone CLI commands.
These examples execute one compact run and print its result.

### Randomized indexed-rule retrieval (`d3_randomized`)

```powershell
& $Py -c "from d2c.experiments.d3_randomized import D3RandomizedConfig, run_d3_randomized_experiment; print(run_d3_randomized_experiment(D3RandomizedConfig(train_count=32, test_count=8, epochs=3)))"
```

### Learned binding (`d3_learned_binding`)

```powershell
& $Py -c "from d2c.experiments.d3_learned_binding import run_default_experiment; print(run_default_experiment())"
```

To configure a learned-binding run:

```powershell
& $Py -c "from d2c.experiments.d3_learned_binding import LearnedBindingConfig, run_learned_binding_experiment; print(run_learned_binding_experiment(LearnedBindingConfig(train_count=32, test_count=8, epochs=3)))"
```

### Learned binding through the ABI (`d3_lb_abi`)

Without a supplied hierarchy director, this runner uses the Python substrate:

```powershell
& $Py -c "from d2c.experiments.d3_learned_binding import LearnedBindingConfig; from d2c.experiments.d3_lb_abi import run_d3_lb_abi; print(run_d3_lb_abi(LearnedBindingConfig(train_count=32, test_count=8, epochs=3)))"
```

To exercise the C++ ABI, supply a `HierarchicalDigitalDirector`; it requires
the shared library. `d4a_probe` below is the ready-made ABI-backed run.

### D4A ABI hierarchy probe (`d4a_probe`)

Supported arms are `flat_abi`, `w_boost_flat`, `hier_bu`, and `hier_bu_td`.
This compact command runs the bottom-up/top-down hierarchy arm:

```powershell
& $Py -c "from d2c.experiments.d3_randomized import D3RandomizedConfig; from d2c.experiments.d4a_probe import run_d4a_probe; print(run_d4a_probe(D3RandomizedConfig(train_count=32, test_count=8, epochs=3), arm='hier_bu_td'))"
```

To compare arms, change `arm` to each supported value. This experiment loads
`code\c++ core\build-shared\libgfe_core.dll` by default.
