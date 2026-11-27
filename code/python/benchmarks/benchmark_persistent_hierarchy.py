#!/usr/bin/env python3
"""Reproducible end-to-end benchmark: persistent hierarchy runtime vs D4B.

The D4B path reconstructs the chain model on every event. The persistent path
constructs one native runtime per episode. Timings include the Python ctypes
wrapper and state/diagnostic transfer on both paths. Python tracemalloc figures
exclude native C++ allocations and are reported only as a Python-side proxy.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "code" / "python"))

import gfe_ctypes as gfe  # noqa: E402


def workload():
    levels = [
        {"name": "sensory", "gamma": [1.4, 0.5], "w": [0.7, 0.3],
         "u": [0.15], "chi": [0.0, 0.0], "dt": 0.01,
         "linear_decay": [0.2], "forcing_bias": [0.0],
         "form": gfe.GFE_C_COUPLING_FORM_B, "coupling_index": 0},
        {"name": "context", "gamma": [0.8, 0.2], "w": [0.55, 0.45],
         "u": [-0.05], "chi": [0.0, 0.0], "dt": 0.01,
         "linear_decay": [0.12], "forcing_bias": [0.005],
         "form": gfe.GFE_C_COUPLING_FORM_B, "coupling_index": 0},
        {"name": "abstract", "gamma": [0.35, 0.08], "w": [0.6, 0.4],
         "u": [0.025], "chi": [0.0, 0.0], "dt": 0.01,
         "linear_decay": [0.06], "forcing_bias": [0.0],
         "form": gfe.GFE_C_COUPLING_FORM_B, "coupling_index": 0},
    ]
    edges = [
        {"source_level": 0, "target_level": 1,
         "relation": gfe.GFE_C_HIERARCHICAL_RELATION_BOTTOM_UP, "gain": 0.3},
        {"source_level": 1, "target_level": 2,
         "relation": gfe.GFE_C_HIERARCHICAL_RELATION_BOTTOM_UP, "gain": 0.2},
        {"source_level": 2, "target_level": 1,
         "relation": gfe.GFE_C_HIERARCHICAL_RELATION_TOP_DOWN, "gain": 0.1},
        {"source_level": 1, "target_level": 0,
         "relation": gfe.GFE_C_HIERARCHICAL_RELATION_TOP_DOWN, "gain": 0.15},
    ]
    return levels, edges


def sequence_for(events: int):
    return [0.12 * math.sin(i * 0.071) + (0.04 if i % 37 == 0 else 0.0)
            for i in range(events)]


def run_d4b(lib, levels, edges, sequence):
    states = [{"u": list(level["u"]), "chi": list(level["chi"]),
               "t": float(level.get("t", 0.0))} for level in levels]
    trace = []
    started = time.perf_counter()
    for forcing in sequence:
        result = gfe.step_hierarchical_chain_spec(
            lib, levels, edges, level_states=states,
            external_forcing=[forcing], forcing_level=0,
        )
        states = result["level_states"]
        trace.append(result)
    return (time.perf_counter() - started), trace


def run_persistent(lib, levels, edges, sequence):
    create_started = time.perf_counter()
    runtime = gfe.create_persistent_hierarchy_runtime(lib, levels, edges)
    create_seconds = time.perf_counter() - create_started
    trace = []
    try:
        started = time.perf_counter()
        for forcing in sequence:
            trace.append(runtime.step({0: [forcing]}))
        step_seconds = time.perf_counter() - started
    finally:
        runtime.close()
    return create_seconds, step_seconds, trace


def max_parity_error(reference, candidate):
    error = 0.0
    for left, right in zip(reference, candidate):
        for ls, rs in zip(left["level_states"], right["level_states"]):
            for key in ("u", "chi"):
                error = max(error, *(abs(a - b) for a, b in zip(ls[key], rs[key])))
            error = max(error, abs(ls["t"] - rs["t"]))
        for lk, rk in zip(left["active_kernels"], right["active_kernels"]):
            for key in ("gamma", "w"):
                error = max(error, *(abs(a - b) for a, b in zip(lk[key], rk[key])))
        for lsp, rsp in zip(left["spectral_units"], right["spectral_units"]):
            for key in ("mcap", "mscale", "mres", "hmem", "hnorm", "deff"):
                error = max(error, abs(lsp[key] - rsp[key]))
    return error


def python_peak_bytes(callable_):
    tracemalloc.start()
    try:
        callable_()
        _, peak = tracemalloc.get_traced_memory()
        return peak
    finally:
        tracemalloc.stop()


def summarize(values):
    ordered = sorted(values)
    return {
        "samples_ms": [round(v * 1000.0, 6) for v in values],
        "median_ms": statistics.median(values) * 1000.0,
        "min_ms": min(values) * 1000.0,
        "max_ms": max(values) * 1000.0,
        "p25_ms": ordered[max(0, int(0.25 * (len(ordered) - 1)))] * 1000.0,
        "p75_ms": ordered[min(len(ordered) - 1, int(0.75 * (len(ordered) - 1)))] * 1000.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=int, default=400)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.events < 1 or args.repeats < 3 or args.warmup < 0:
        parser.error("events >= 1, repeats >= 3, and warmup >= 0 are required")

    lib = gfe.load_gfe_library()
    levels, edges = workload()
    sequence = sequence_for(args.events)

    for _ in range(args.warmup):
        _, d4b_trace = run_d4b(lib, levels, edges, sequence)
        _, _, persistent_trace = run_persistent(lib, levels, edges, sequence)
        if max_parity_error(d4b_trace, persistent_trace) > 1e-10:
            raise SystemExit("warm-up parity failure")

    d4b_times, create_times, persistent_times, errors = [], [], [], []
    for repeat_index in range(args.repeats):
        if repeat_index % 2 == 0:
            create_elapsed, persistent_elapsed, persistent_trace = run_persistent(
                lib, levels, edges, sequence
            )
            d4b_elapsed, d4b_trace = run_d4b(lib, levels, edges, sequence)
        else:
            d4b_elapsed, d4b_trace = run_d4b(lib, levels, edges, sequence)
            create_elapsed, persistent_elapsed, persistent_trace = run_persistent(
                lib, levels, edges, sequence
            )
        error = max_parity_error(d4b_trace, persistent_trace)
        if error > 1e-10:
            raise SystemExit(f"numerical parity failure: max_abs_error={error:.3e}")
        d4b_times.append(d4b_elapsed)
        create_times.append(create_elapsed)
        persistent_times.append(persistent_elapsed)
        errors.append(error)

    def measure_d4b():
        run_d4b(lib, levels, edges, sequence)

    def measure_persistent():
        run_persistent(lib, levels, edges, sequence)

    d4b_peak = python_peak_bytes(measure_d4b)
    persistent_peak = python_peak_bytes(measure_persistent)
    d4b_median = statistics.median(d4b_times)
    persistent_step_median = statistics.median(persistent_times)
    create_median = statistics.median(create_times)
    result = {
        "benchmark": "persistent_hierarchy_vs_d4b",
        "version": 1,
        "events_per_episode": args.events,
        "repeats": args.repeats,
        "warmup_episodes": args.warmup,
        "environment": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "implementation": platform.python_implementation(),
        },
        "hierarchy_levels": len(levels),
        "coupling_edges": len(edges),
        "timing_scope": "end-to-end Python ctypes calls including per-event state and diagnostics transfer",
        "construction_counts_per_episode": {
            "d4b_chain_model_constructions": args.events,
            "persistent_chain_model_constructions": 1,
        },
        "d4b_episode": summarize(d4b_times),
        "persistent_creation": summarize(create_times),
        "persistent_step_loop": summarize(persistent_times),
        "persistent_total_episode_ms_median": (create_median + persistent_step_median) * 1000.0,
        "persistent_amortized_total_per_event_ms": (
            (create_median + persistent_step_median) * 1000.0 / args.events
        ),
        "d4b_per_event_ms_median": d4b_median * 1000.0 / args.events,
        "persistent_step_per_event_ms_median": persistent_step_median * 1000.0 / args.events,
        "persistent_total_episode_speedup_vs_d4b": (
            d4b_median / (create_median + persistent_step_median)
        ),
        "max_abs_numerical_parity_error": max(errors),
        "python_tracemalloc_peak_bytes": {
            "d4b": d4b_peak,
            "persistent": persistent_peak,
            "scope_note": "Python allocations only; native C++ allocations are not included",
        },
        "native_allocation_count": {
            "status": "not_instrumented",
            "note": "No native allocator hook is enabled; do not infer C++ allocation reductions from tracemalloc",
        },
        "raw_samples_seconds": {
            "d4b_episode": d4b_times,
            "persistent_creation": create_times,
            "persistent_step_loop": persistent_times,
        },
    }
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
