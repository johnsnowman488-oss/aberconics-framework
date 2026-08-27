# Python Wrapper (ctypes)

This folder provides a Python `ctypes` wrapper over the C ABI in:
- `code/c++ core/include/gfe/gfe_c_api.h`

## Files

- `gfe_ctypes.py`: typed `ctypes` structs/signatures + helpers for:
  - `fit_soe_kernel(...)`
  - `list_abersoe_scenarios(...)`
  - `run_abersoe_scenario(...)`
  - `run_abersoe_scenario_with_overrides(...)`
  - `list_hierarchical_scenarios(...)`
  - `run_hierarchical_scenario(...)`
  - `validate_hierarchical_chain_spec(...)`
  - `run_hierarchical_chain_spec(...)`
  - `get_hierarchical_cross_level_report_for_chain_spec(...)`
  - `get_hierarchical_renorm_report_for_chain_spec(...)`
  - `get_hierarchical_cross_level_report(...)`
  - `get_hierarchical_renorm_report(...)`
  - `write_hierarchical_summary_csv(...)`
  - `write_hierarchical_cross_level_report_csv(...)`
  - `write_hierarchical_renorm_report_csv(...)`
  - `write_hierarchical_export_bundle(...)`
  - `write_hierarchical_chain_export_bundle(...)`
  - `hierarchy_*_csv_header(...)`
- `ctypes_smoke.py`: proof-of-call script
- `tests/test_gfe_ctypes.py`: tiny pytest smoke test

## Build Shared Library

```bash
cmake -S "code/c++ core" -B "code/c++ core/build-shared" -DBUILD_SHARED_LIBS=ON
cmake --build "code/c++ core/build-shared" -j
```

## Run Smoke Script

```bash
export GFE_CORE_LIB="$(pwd)/code/c++ core/build-shared/libgfe_core.so"
python3 code/python/ctypes_smoke.py
```

## Run Pytest

```bash
python3 -m pip install pytest
export GFE_CORE_LIB="$(pwd)/code/c++ core/build-shared/libgfe_core.so"
pytest -q code/python/tests
```

## Run the Lorenz63 experiment (practical notes)

If you want to run the full `lorenz63` experiment script (the one under `d2c/experiments`), follow these extra steps I used to run it successfully:

- Build the C++ core as a shared library (so the Python `ctypes` loader can find it):

```bash
cd "code/c++ core"
cmake -S . -B build-shared -DBUILD_SHARED_LIBS=ON
cmake --build build-shared -- -j
cd -
```

- Make sure Python can find the local Python package sources (run from the repo root):

```bash
cd /path/to/aberconics-framework
export PYTHONPATH=code/python:$PYTHONPATH
```

- If the shared library is in the default build location, point the loader to it (optional; `gfe_ctypes` also tries common build paths):

```bash
export GFE_CORE_LIB="$PWD/code/c++ core/build-shared/libgfe_core.so"
```

- The experiment expects to be run as a package so relative imports resolve. Run it like this from the repository root:

```bash
/usr/bin/python3 -m d2c.experiments.lorenz63
```

- Install plotting dependencies if you want the script to emit PNG plots (the script uses `matplotlib`):

```bash
/usr/bin/python3 -m pip install --user matplotlib
```

Notes:
- Run the script from the repository root so the relative imports in `d2c` work correctly.
- If you see "No module named 'gfe_ctypes'", confirm `PYTHONPATH` includes `code/python` (or set up an editable install).
- If the script errors loading the shared library, verify `GFE_CORE_LIB` points to `code/c++ core/build-shared/libgfe_core.so` or rebuild with `BUILD_SHARED_LIBS=ON`.


## Minimal Example

```python
from gfe_ctypes import (
    GFE_C_ABERSOE_SCENARIO_LINEAR,
    GFE_C_ABERSOE_SCENARIO_LORENZ63,
    GFE_C_HIERARCHICAL_SCENARIO_THREE_LEVEL_HOMEOSTATIC,
    GFE_C_HIERARCHICAL_RELATION_BOTTOM_UP,
    GFE_C_HIERARCHICAL_RELATION_TOP_DOWN,
    GFE_C_HIERARCHICAL_RENORM_DIAGNOSTIC_FIT_INSTABILITY,
    GFE_C_HIERARCHICAL_WARNING_CODE_LARGE_KERNEL_SHIFT,
    GFE_C_COUPLING_FORM_B,
    get_hierarchical_cross_level_report_for_chain_spec,
    get_hierarchical_renorm_report_for_chain_spec,
    get_hierarchical_cross_level_report,
    get_hierarchical_renorm_report,
    fit_soe_kernel,
    hierarchy_cross_level_csv_header,
    hierarchy_renorm_csv_header,
    list_abersoe_scenarios,
    list_hierarchical_scenarios,
    load_gfe_library,
    run_abersoe_scenario,
    run_abersoe_scenario_with_overrides,
    run_hierarchical_chain_spec,
    run_hierarchical_scenario,
    validate_hierarchical_chain_spec,
    write_hierarchical_cross_level_report_csv,
    write_hierarchical_chain_export_bundle,
    write_hierarchical_export_bundle,
    write_hierarchical_renorm_report_csv,
    write_hierarchical_summary_csv,
)

lib = load_gfe_library()  # uses GFE_CORE_LIB or default build-shared path
t = [0.1 * i for i in range(81)]
y = [0.65 * (2.718281828459045 ** (-0.8 * x)) + 0.35 * (2.718281828459045 ** (-0.12 * x)) for x in t]
out = fit_soe_kernel(lib, t, y)
print(len(out["gamma_fit"]), len(out["fit"]))

print(list_abersoe_scenarios(lib))

run = run_abersoe_scenario(
    lib,
    GFE_C_ABERSOE_SCENARIO_LINEAR,
    dt=0.01,
    steps=20,
    sample_stride=5,
)
print(run["final_state"]["t"], run["diagnostics"]["steps_executed"])

lorenz = run_abersoe_scenario_with_overrides(
    lib,
    GFE_C_ABERSOE_SCENARIO_LORENZ63,
    dt=0.01,
    steps=20,
    sample_stride=1,
    initial_u=[2.0, -1.0, 8.0],
    initial_chi=[0.1, -0.2, 0.05],
    initial_t=0.25,
    gamma=[1.4, 0.6, 0.2],
    w=[0.45, 0.35, 0.2],
)
print(lorenz["active_kernel"]["gamma"], lorenz["trajectory"]["u"][0])

print(list_hierarchical_scenarios(lib))

hier = run_hierarchical_scenario(
    lib,
    GFE_C_HIERARCHICAL_SCENARIO_THREE_LEVEL_HOMEOSTATIC,
    steps=24,
    sample_stride=6,
)
print(hier["diagnostics"]["level_count"], hier["levels"][0]["deff"])

cross = get_hierarchical_cross_level_report(
    lib,
    GFE_C_HIERARCHICAL_SCENARIO_THREE_LEVEL_HOMEOSTATIC,
    steps=24,
    sample_stride=6,
)
print(cross["contract"]["all_finite_pass"])
print(cross["relations"][0]["relation_type"] == GFE_C_HIERARCHICAL_RELATION_BOTTOM_UP)

renorm = get_hierarchical_renorm_report(
    lib,
    GFE_C_HIERARCHICAL_SCENARIO_THREE_LEVEL_HOMEOSTATIC,
)
print(renorm[0]["primary_diagnostic"] == GFE_C_HIERARCHICAL_RENORM_DIAGNOSTIC_FIT_INSTABILITY)

print(hierarchy_cross_level_csv_header(lib))
print(hierarchy_renorm_csv_header(lib))

levels = [
    {
        "name": "custom_fast",
        "gamma": [1.0, 0.4],
        "w": [0.65, 0.35],
        "u": [1.0],
        "chi": [0.0, 0.0],
        "dt": 0.01,
        "linear_decay": [0.25],
        "forcing_bias": [0.0],
        "form": GFE_C_COUPLING_FORM_B,
        "coupling_index": 0,
    },
    {
        "name": "custom_slow",
        "gamma": [0.7, 0.15],
        "w": [0.6, 0.4],
        "u": [0.1],
        "chi": [0.0, 0.0],
        "dt": 0.01,
        "linear_decay": [0.08],
        "forcing_bias": [0.01],
        "form": GFE_C_COUPLING_FORM_B,
        "coupling_index": 0,
    },
]
edges = [
    {
        "source_level": 0,
        "target_level": 1,
        "relation": GFE_C_HIERARCHICAL_RELATION_BOTTOM_UP,
        "gain": 0.35,
        "normalize_weights": True,
    },
    {
        "source_level": 1,
        "target_level": 0,
        "relation": GFE_C_HIERARCHICAL_RELATION_TOP_DOWN,
        "gain": 0.18,
        "normalize_weights": True,
    },
]
validate_hierarchical_chain_spec(lib, levels, edges)
custom = run_hierarchical_chain_spec(lib, levels, edges, steps=18, sample_stride=6)
print(custom["diagnostics"]["level_count"], custom["levels"][0]["deff"])

custom_cross = get_hierarchical_cross_level_report_for_chain_spec(
    lib, levels, edges, steps=18, sample_stride=6
)
print(custom_cross["contract"]["all_finite_pass"], len(custom_cross["relations"]))

custom_renorm = get_hierarchical_renorm_report_for_chain_spec(lib, levels, edges)
print(len(custom_renorm), custom_renorm[0]["retained_weight_fraction"])

write_hierarchical_summary_csv(lib, "/tmp/custom_summary.csv", custom, level_names=[level["name"] for level in levels])
write_hierarchical_cross_level_report_csv(lib, "/tmp/custom_cross.csv", custom_cross)
write_hierarchical_renorm_report_csv(lib, "/tmp/custom_renorm.csv", custom_renorm)

bundle = write_hierarchical_export_bundle(
    lib,
    "/tmp",
    "custom_case",
    custom,
    custom_cross,
    custom_renorm,
    level_names=[level["name"] for level in levels],
)
print(bundle)

direct_bundle = write_hierarchical_chain_export_bundle(
    lib,
    "/tmp",
    "custom_case_direct",
    levels,
    edges,
    steps=18,
    sample_stride=6,
)
print(direct_bundle)
```

## Single-Level Override Path

For single-level ABERSOE experiments, `run_abersoe_scenario_with_overrides(...)` is the wrapper entrypoint when you need:
- custom Lorenz63 initial conditions
- custom single-level `gamma` / `w`
- direct access to the active runtime kernel
- sampled full-state trajectories for both `u` and `chi`

Notes:
- the returned `trajectory` payload is sampled using the normal runtime `sample_stride`
- for dense memory-side analysis, set `sample_stride=1`
- `gamma` and `w` must be provided together when overriding the kernel

## Constrained Custom Chains

The Python wrapper now exposes the first Phase E custom-chain slice through plain Python dictionaries and lists.

Helpers:
- `validate_hierarchical_chain_spec(lib, levels, edges)`
- `run_hierarchical_chain_spec(lib, levels, edges, *, steps=..., sample_stride=..., strict_finite=...)`
- `get_hierarchical_cross_level_report_for_chain_spec(lib, levels, edges, *, steps=..., sample_stride=..., strict_finite=...)`
- `get_hierarchical_renorm_report_for_chain_spec(lib, levels, edges)`
- `write_hierarchical_summary_csv(lib, path, run_result, *, level_names=None)`
- `write_hierarchical_cross_level_report_csv(lib, path, report)`
- `write_hierarchical_renorm_report_csv(lib, path, analyses)`
- `write_hierarchical_export_bundle(lib, output_dir, bundle_name, run_result, cross_report, renorm_report, *, level_names=None)`
- `write_hierarchical_chain_export_bundle(lib, output_dir, bundle_name, levels, edges, *, steps=..., sample_stride=..., strict_finite=...)`

Expected level fields:
- `name`
- `gamma`
- `w`
- `u`
- optional `chi`
- optional `t`
- `dt`
- optional `linear_decay`
- optional `forcing_bias`
- optional `form`
- optional `coupling_index`

Expected edge fields:
- `source_level`
- `target_level`
- `relation`
- optional `gain`
- optional `normalize_weights`

This surface is intentionally constrained:
- chain-style hierarchies only
- adjacent bottom-up/top-down edges only
- no arbitrary callback injection
- custom chains now support the same typed cross-level and renorm report access pattern as canonical hierarchy scenarios
- lightweight Python-side CSV export helpers sit above the typed report/run data when an application wants canonical column order without rebuilding it
- named export bundles add a small convenience layer for writing the standard summary/cross-level/renorm trio with predictable filenames

## Hierarchy Report Enums

The hierarchy report helpers return integer-coded enums so Python callers can make stable assertions without parsing text.

### Relation type

- `GFE_C_HIERARCHICAL_RELATION_BOTTOM_UP = 0`
- `GFE_C_HIERARCHICAL_RELATION_TOP_DOWN = 1`

### Warning severity

- `GFE_C_HIERARCHICAL_WARNING_INFO = 0`
- `GFE_C_HIERARCHICAL_WARNING_WARNING = 1`
- `GFE_C_HIERARCHICAL_WARNING_ERROR = 2`

### Warning code

- `GFE_C_HIERARCHICAL_WARNING_CODE_UNKNOWN = 0`
- `GFE_C_HIERARCHICAL_WARNING_CODE_NON_FINITE_RUN = 1`
- `GFE_C_HIERARCHICAL_WARNING_CODE_KERNEL_SHIFT_DETECTED = 2`
- `GFE_C_HIERARCHICAL_WARNING_CODE_LARGE_KERNEL_SHIFT = 3`
- `GFE_C_HIERARCHICAL_WARNING_CODE_BOTTOM_UP_MCAP_INVERSION = 4`
- `GFE_C_HIERARCHICAL_WARNING_CODE_BOTTOM_UP_MSCALE_INVERSION = 5`
- `GFE_C_HIERARCHICAL_WARNING_CODE_TOP_DOWN_TIMESCALE_SHIFT = 6`
- `GFE_C_HIERARCHICAL_WARNING_CODE_TOP_DOWN_DEFF_SHIFT = 7`

### Warning metric

- `GFE_C_HIERARCHICAL_WARNING_METRIC_UNKNOWN = 0`
- `GFE_C_HIERARCHICAL_WARNING_METRIC_ALL_FINITE = 1`
- `GFE_C_HIERARCHICAL_WARNING_METRIC_KERNEL_SHIFT = 2`
- `GFE_C_HIERARCHICAL_WARNING_METRIC_DELTA_MCAP = 3`
- `GFE_C_HIERARCHICAL_WARNING_METRIC_DELTA_MSCALE = 4`
- `GFE_C_HIERARCHICAL_WARNING_METRIC_DELTA_DEFF = 5`

### Renorm diagnostic

- `GFE_C_HIERARCHICAL_RENORM_DIAGNOSTIC_NONE = 0`
- `GFE_C_HIERARCHICAL_RENORM_DIAGNOSTIC_FIT_INSTABILITY = 1`
- `GFE_C_HIERARCHICAL_RENORM_DIAGNOSTIC_OVER_PRUNING = 2`
- `GFE_C_HIERARCHICAL_RENORM_DIAGNOSTIC_PLAUSIBLE_CHANNEL_MERGING = 3`

## Schema/Header Helpers

The Python wrapper now exposes the hierarchy CSV schema/header helpers directly:

- `hierarchy_run_csv_header(lib)`
- `hierarchy_summary_csv_header(lib)`
- `hierarchy_diagnostics_csv_header(lib)`
- `hierarchy_cross_level_csv_header(lib)`
- `hierarchy_renorm_csv_header(lib)`

These return the exact CSV header strings exported by the C ABI, which is useful when:
- writing report data to your own files
- validating schema expectations in tests
- aligning application-side parsers with the core report contract

Typical patterns:
- use `get_hierarchical_cross_level_report(...)` and `get_hierarchical_renorm_report(...)` for typed programmatic access
- use `hierarchy_*_csv_header(...)` when you need the canonical column order for file or table output

## Report Field Glossary

Short meanings for a few fields that are easy to misread at first glance.

- `target_mcap_not_lower`:
  for a reported cross-level relation, indicates whether the target level's `Mcap` is not lower than the source level's `Mcap`. This is useful for checking whether a hierarchy respects expected memory-capacity ordering across levels.

- `target_mscale_not_lower`:
  similar to `target_mcap_not_lower`, but for `Mscale`. It helps flag cases where the target level is not preserving the expected scale ordering relative to the source level.

- `actionable_warning_count`:
  the number of warnings in the cross-level contract report that are marked actionable. These are the warnings most suitable for application logic, alerts, or regression assertions.

- `primary_diagnostic`:
  the main renorm interpretation code for one adjacent renorm analysis. It maps to one of:
  - `GFE_C_HIERARCHICAL_RENORM_DIAGNOSTIC_NONE`
  - `GFE_C_HIERARCHICAL_RENORM_DIAGNOSTIC_FIT_INSTABILITY`
  - `GFE_C_HIERARCHICAL_RENORM_DIAGNOSTIC_OVER_PRUNING`
  - `GFE_C_HIERARCHICAL_RENORM_DIAGNOSTIC_PLAUSIBLE_CHANNEL_MERGING`

- `retained_weight_fraction`:
  fraction of kernel weight retained after pruning in a renorm analysis. Values closer to `1.0` mean little weight was discarded.

- `raw_output_channels`:
  number of output channels before pruning or post-fit simplification in the renorm path.

- `output_channels`:
  number of channels after the main renorm fit step, before any pruning-specific count is reported separately.

- `pruned_channels`:
  number of channels remaining after pruning.

- `consistency_pass`:
  overall boolean for whether the renorm consistency checks passed for that adjacent-level analysis.
