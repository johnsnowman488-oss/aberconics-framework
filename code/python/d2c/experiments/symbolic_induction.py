"""Milestone D symbolic induction experiments."""

from __future__ import annotations

import argparse
from dataclasses import replace
import math
import random
import sys
from pathlib import Path


if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[4]
    python_root = repo_root / "code" / "python"
    if str(python_root) not in sys.path:
        sys.path.insert(0, str(python_root))

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from typing import Mapping

try:
    from ..digital import (
        DigitalMemoryConfig,
        DigitalStabilityContract,
        DigitalPrediction,
        DigitalMemoryState,
        DigitalTrace,
        DigitalTraceStep,
        KeyValueBindingBridge,
        SymbolicInductionConfig,
        build_symbolic_induction_vocabulary,
        decode_key_value_binding,
        format_symbolic_induction_report,
        generate_symbolic_induction_example,
        seed_memory_state,
        step_memory,
        summarize_predictions,
        window_limited_lookup_prediction,
    )
except ImportError:  # pragma: no cover - direct script execution fallback
    from d2c.digital import (
        DigitalMemoryConfig,
        DigitalStabilityContract,
        DigitalPrediction,
        DigitalMemoryState,
        DigitalTrace,
        DigitalTraceStep,
        KeyValueBindingBridge,
        SymbolicInductionConfig,
        build_symbolic_induction_vocabulary,
        decode_key_value_binding,
        format_symbolic_induction_report,
        generate_symbolic_induction_example,
        seed_memory_state,
        step_memory,
        summarize_predictions,
        window_limited_lookup_prediction,
    )


@dataclass(slots=True)
class SymbolicInductionRunConfig:
    """Run settings for the multi-pair symbolic retrieval probe."""

    task: SymbolicInductionConfig
    variant: str = "full"
    seed_count: int = 8
    pulse_steps: int = 2
    silence_steps: int = 1
    bridge_dim: int = 8
    bridge_seed: int = 17
    code_mode: str = "one_hot"
    baseline_window: int = 4
    binding_mode: str = "key_value_outer_product"
    minimum_binding_score: float | None = None
    binding_noise_std: float = 0.0
    calibration_seed_count: int = 16
    calibration_quantile: float = 0.99
    silence_mode: str = "evolve"

    def __post_init__(self) -> None:
        if self.minimum_binding_score is not None and self.minimum_binding_score < 0.0:
            raise ValueError("minimum_binding_score must be non-negative")
        if self.binding_noise_std < 0.0:
            raise ValueError("binding_noise_std must be non-negative")
        if self.calibration_seed_count <= 0:
            raise ValueError("calibration_seed_count must be positive")
        if not 0.0 < self.calibration_quantile <= 1.0:
            raise ValueError("calibration_quantile must be in (0, 1]")
        if self.silence_mode not in {"evolve", "freeze"}:
            raise ValueError("silence_mode must be evolve or freeze")

    def to_mapping(self) -> dict[str, object]:
        data = asdict(self)
        data["task"] = self.task.to_mapping()
        return data

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "SymbolicInductionRunConfig":
        return cls(
            task=SymbolicInductionConfig.from_mapping(data.get("task", {})),
            variant=str(data.get("variant", "full")),
            seed_count=int(data.get("seed_count", 8)),
            pulse_steps=int(data.get("pulse_steps", 2)),
            silence_steps=int(data.get("silence_steps", 1)),
            bridge_dim=int(data.get("bridge_dim", 8)),
            bridge_seed=int(data.get("bridge_seed", 17)),
            code_mode=str(data.get("code_mode", "one_hot")),
            baseline_window=int(data.get("baseline_window", 4)),
            binding_mode=str(data.get("binding_mode", "key_value_outer_product")),
            minimum_binding_score=(
                None if data.get("minimum_binding_score") is None else float(data["minimum_binding_score"])
            ),
            binding_noise_std=float(data.get("binding_noise_std", 0.0)),
            calibration_seed_count=int(data.get("calibration_seed_count", 16)),
            calibration_quantile=float(data.get("calibration_quantile", 0.99)),
            silence_mode=str(data.get("silence_mode", "evolve")),
        )


_DIGITAL_STABILITY_CONTRACT = DigitalStabilityContract()


def memory_config_for_variant(variant: str) -> DigitalMemoryConfig:
    """Return matched stable kernels under the shared digital contract."""

    if variant == "full":
        return _DIGITAL_STABILITY_CONTRACT.config(
            gamma=[2.0, 0.45, 0.035], ratio_allocation=[0.05, 0.15, 0.80],
        )
    if variant == "no_slow":
        return _DIGITAL_STABILITY_CONTRACT.config(
            gamma=[2.0, 0.9], ratio_allocation=[0.25, 0.75],
        )
    if variant == "collapsed_gamma":
        return _DIGITAL_STABILITY_CONTRACT.config(
            gamma=[0.6, 0.6, 0.6], ratio_allocation=[0.2, 0.3, 0.5],
        )
    raise ValueError(f"unknown symbolic induction variant: {variant}")


def run_symbolic_induction_trial(
    config: SymbolicInductionRunConfig,
    *,
    example_index: int,
    binding_amplitude: float = 1.0,
) -> tuple[DigitalPrediction, DigitalTrace, dict[str, object]]:
    vocab = build_symbolic_induction_vocabulary(config.task)
    memory_cfg = memory_config_for_variant(config.variant)
    example = generate_symbolic_induction_example(config.task, example_index=example_index)
    binding = KeyValueBindingBridge(
        key_tokens=[token for token in vocab.tokens if token.startswith("K")],
        value_tokens=[token for token in vocab.tokens if token.startswith("V")],
        amplitude=binding_amplitude,
    )
    state = seed_memory_state(state_dim=binding.state_dim, channel_count=memory_cfg.channel_count)
    trace = DigitalTrace(
        experiment_name="symbolic_induction",
        metadata={
            "variant": config.variant,
            "example": example.to_mapping(),
            "memory_config": memory_cfg.to_mapping(),
        },
    )

    noise_rng = random.Random(config.bridge_seed + example_index)

    def binding_forcing(key_token: str, value_token: str) -> list[float]:
        forcing = binding.binding_for_pair(key_token, value_token)
        if config.binding_noise_std:
            return [value + noise_rng.gauss(0.0, config.binding_noise_std) for value in forcing]
        return forcing

    schedule: list[tuple[list[float], str | None]] = []
    for key_token, value_token in [*example.pairs, *example.distractor_pairs]:
        schedule.extend([(binding_forcing(key_token, value_token), value_token)] * config.pulse_steps)
        schedule.extend([(binding.zero_forcing(), None)] * config.silence_steps)
    for filler_token in example.filler_tokens:
        schedule.extend([(binding.zero_forcing(), filler_token)] * config.pulse_steps)
        schedule.extend([(binding.zero_forcing(), None)] * config.silence_steps)
    schedule.extend([(binding.zero_forcing(), example.key_token)] * config.pulse_steps)
    schedule.extend([(binding.zero_forcing(), None)] * config.silence_steps)

    for step_idx, (forcing, token) in enumerate(schedule):
        if token is None and config.silence_mode == "freeze":
            state = DigitalMemoryState(
                u=list(state.u),
                chi=[list(channel) for channel in state.chi],
                t=state.t + memory_cfg.dt,
            )
        else:
            state = step_memory(state, forcing, memory_cfg)
        trace.add_step(
            DigitalTraceStep(
                step=step_idx,
                t=state.t,
                token=token,
                forcing=list(forcing),
                u=list(state.u),
                chi=[list(channel) for channel in state.chi],
            )
        )

    predicted, score = decode_key_value_binding(
        state,
        memory_cfg,
        binding,
        example.key_token,
        minimum_score=config.minimum_binding_score or 0.0,
    )
    prediction = DigitalPrediction(
        target=example.value_token,
        predicted=predicted,
        score=score,
    )
    diagnostics = {
        "deff": memory_cfg.deff(),
        "stability_ratio": memory_cfg.stability_ratio(),
        "final_t": state.t,
        "step_count": len(schedule),
        "binding_mode": config.binding_mode,
        "minimum_binding_score": config.minimum_binding_score,
        "binding_noise_std": config.binding_noise_std,
        "distractor_pair_count": config.task.distractor_pair_count,
        "silence_mode": config.silence_mode,
    }
    return prediction, trace, diagnostics


def calibrate_minimum_binding_score(config: SymbolicInductionRunConfig) -> dict[str, object]:
    """Calibrate evidence rejection from held-out noise-only binding trials."""

    null_config = replace(config, minimum_binding_score=0.0)
    null_scores = [
        run_symbolic_induction_trial(
            null_config,
            example_index=config.seed_count + idx,
            binding_amplitude=0.0,
        )[0].score
        for idx in range(config.calibration_seed_count)
    ]
    sorted_scores = sorted(null_scores)
    index = max(0, math.ceil(config.calibration_quantile * len(sorted_scores)) - 1)
    null_quantile = sorted_scores[index]
    # The floor preserves the pre-existing numerical-evidence rule when the
    # null distribution is exactly zero (for example, in noiseless runs).
    threshold = max(0.001, null_quantile)
    return {
        "method": "noise_only_null_quantile",
        "seed_count": config.calibration_seed_count,
        "quantile": config.calibration_quantile,
        "null_quantile_score": null_quantile,
        "numerical_floor": 0.001,
        "minimum_binding_score": threshold,
    }


def run_symbolic_induction_experiment(config: SymbolicInductionRunConfig) -> dict[str, object]:
    calibration = None
    if config.minimum_binding_score is None:
        calibration = calibrate_minimum_binding_score(config)
        config = replace(config, minimum_binding_score=float(calibration["minimum_binding_score"]))
    predictions: list[DigitalPrediction] = []
    traces: list[DigitalTrace] = []
    diagnostics: list[dict[str, object]] = []
    baseline_correct = 0

    for idx in range(config.seed_count):
        prediction, trace, diag = run_symbolic_induction_trial(config, example_index=idx)
        predictions.append(prediction)
        traces.append(trace)
        diagnostics.append(diag)
        example = generate_symbolic_induction_example(config.task, example_index=idx)
        baseline = window_limited_lookup_prediction(example.stream, window=config.baseline_window)
        baseline_correct += int(baseline == example.value_token)

    summary = summarize_predictions(predictions)
    memory_cfg = memory_config_for_variant(config.variant)
    return {
        "experiment_name": "symbolic_induction",
        "variant": config.variant,
        "gap": config.task.gap,
        "pair_count": config.task.pair_count,
        "distractor_pair_count": config.task.distractor_pair_count,
        "binding_mode": config.binding_mode,
        "minimum_binding_score": config.minimum_binding_score,
        "binding_noise_std": config.binding_noise_std,
        "seed_count": config.seed_count,
        "config": config.to_mapping(),
        "summary": summary.to_mapping(),
        "predictions": [prediction.to_mapping() for prediction in predictions],
        "memory_config": memory_cfg.to_mapping(),
        "memory_diagnostics": {
            "deff": memory_cfg.deff(),
            "stability_ratio": memory_cfg.stability_ratio(),
            "trial_diagnostics": diagnostics,
        },
        "baseline": {
            "name": "window_limited_lookup",
            "window": config.baseline_window,
            "count": config.seed_count,
            "correct": baseline_correct,
            "accuracy": baseline_correct / config.seed_count if config.seed_count else 0.0,
        },
        "trace_count": len(traces),
        "traces": [trace.to_mapping() for trace in traces],
        "threshold_calibration": calibration,
    }


def save_symbolic_induction_bundle(
    result: Mapping[str, object],
    *,
    output_dir: str | Path,
    bundle_name: str | None = None,
) -> dict[str, str]:
    root = Path(output_dir)
    run_id = bundle_name or f"symbolic_induction_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    bundle_dir = root / run_id
    bundle_dir.mkdir(parents=True, exist_ok=True)

    summary_path = bundle_dir / "summary.json"
    report_path = bundle_dir / "report.txt"
    trace_path = bundle_dir / "traces.json"

    summary = dict(result)
    traces = summary.pop("traces", [])
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    trace_path.write_text(json.dumps(traces, indent=2), encoding="utf-8")
    report_path.write_text(format_symbolic_induction_report(result), encoding="utf-8")

    return {
        "bundle_dir": str(bundle_dir),
        "summary_json": str(summary_path),
        "traces_json": str(trace_path),
        "report_txt": str(report_path),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Milestone D multi-pair symbolic retrieval probe.")
    parser.add_argument("--variant", choices=["full", "no_slow", "collapsed_gamma"], default="full")
    parser.add_argument("--gap", type=int, default=8)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--key-count", type=int, default=4)
    parser.add_argument("--value-count", type=int, default=4)
    parser.add_argument("--filler-count", type=int, default=3)
    parser.add_argument("--distractor-pairs", type=int, default=0)
    parser.add_argument("--binding-noise-std", type=float, default=0.0)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=Path("code/python/d2c/progress/symbolic_induction"))
    parser.add_argument("--bundle-name", default=None)
    return parser


def main(argv: list[str] | None = None) -> dict[str, str]:
    args = build_arg_parser().parse_args(argv)
    config = SymbolicInductionRunConfig(
        task=SymbolicInductionConfig(
            key_count=args.key_count,
            value_count=args.value_count,
            filler_count=args.filler_count,
            pair_count=args.pairs,
            distractor_pair_count=args.distractor_pairs,
            gap=args.gap,
        ),
        variant=args.variant,
        seed_count=args.seeds,
        binding_noise_std=args.binding_noise_std,
    )
    result = run_symbolic_induction_experiment(config)
    paths = save_symbolic_induction_bundle(result, output_dir=args.output_dir, bundle_name=args.bundle_name)
    print(format_symbolic_induction_report(result))
    return paths


if __name__ == "__main__":
    main()
