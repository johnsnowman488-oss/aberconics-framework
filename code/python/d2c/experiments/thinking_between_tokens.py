"""Milestone D2 probe: continuous free evolution between token events."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import random
import sys
from pathlib import Path
from typing import Mapping


if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[4]
    python_root = repo_root / "code" / "python"
    if str(python_root) not in sys.path:
        sys.path.insert(0, str(python_root))

try:
    from ..digital import (
        DigitalPrediction,
        DigitalMemoryState,
        SymbolicInductionConfig,
        build_symbolic_induction_vocabulary,
        decode_key_value_binding,
        generate_symbolic_induction_example,
        seed_memory_state,
        step_memory,
    )
    from .symbolic_induction import (
        SymbolicInductionRunConfig,
        memory_config_for_variant,
        run_symbolic_induction_experiment,
        run_symbolic_induction_trial,
    )
except ImportError:  # pragma: no cover - direct script execution fallback
    from d2c.digital import (
        DigitalPrediction,
        DigitalMemoryState,
        SymbolicInductionConfig,
        build_symbolic_induction_vocabulary,
        decode_key_value_binding,
        generate_symbolic_induction_example,
        seed_memory_state,
        step_memory,
    )
    from d2c.experiments.symbolic_induction import (
        SymbolicInductionRunConfig,
        memory_config_for_variant,
        run_symbolic_induction_experiment,
        run_symbolic_induction_trial,
    )


@dataclass(slots=True)
class ThinkingBetweenTokensConfig:
    """Settings for a matched zero-interval/free-evolution comparison."""

    task: SymbolicInductionConfig
    variant: str = "full"
    seed_count: int = 16
    pulse_steps: int = 2
    silence_steps: int = 4
    binding_noise_std: float = 0.0
    baseline_window: int = 4
    timing_short_steps: int = 4
    timing_long_steps: int = 24
    timing_cue_noise_std: float = 0.02

    def __post_init__(self) -> None:
        if self.seed_count <= 0:
            raise ValueError("seed_count must be positive")
        if self.pulse_steps <= 0:
            raise ValueError("pulse_steps must be positive")
        if self.silence_steps <= 0:
            raise ValueError("silence_steps must be positive for the free-evolution condition")
        if self.binding_noise_std < 0.0:
            raise ValueError("binding_noise_std must be non-negative")
        if self.timing_short_steps <= 0 or self.timing_long_steps <= self.timing_short_steps:
            raise ValueError("timing intervals must satisfy 0 < short < long")
        if self.timing_cue_noise_std < 0.0:
            raise ValueError("timing_cue_noise_std must be non-negative")

    def to_mapping(self) -> dict[str, object]:
        data = asdict(self)
        data["task"] = self.task.to_mapping()
        return data


def _state_energy(step: Mapping[str, object]) -> float:
    chi = step["chi"]
    return sum(value * value for channel in chi for value in channel)


def _silence_diagnostics(
    run_config: SymbolicInductionRunConfig,
    *,
    example_index: int,
) -> dict[str, float]:
    """Summarize state evolution during silent trace steps for one episode."""

    _, trace, _ = run_symbolic_induction_trial(run_config, example_index=example_index)
    silent = [step.to_mapping() for step in trace.steps if step.token is None]
    if not silent:
        return {"silent_steps": 0.0, "silent_start_energy": 0.0, "silent_end_energy": 0.0}
    return {
        "silent_steps": float(len(silent)),
        "silent_start_energy": _state_energy(silent[0]),
        "silent_end_energy": _state_energy(silent[-1]),
    }


def _run_condition(run_config: SymbolicInductionRunConfig) -> dict[str, object]:
    result = run_symbolic_induction_experiment(run_config)
    silence = [_silence_diagnostics(run_config, example_index=index) for index in range(run_config.seed_count)]
    count = len(silence)
    return {
        "summary": result["summary"],
        "baseline": result["baseline"],
        "minimum_binding_score": result["minimum_binding_score"],
        "threshold_calibration": result["threshold_calibration"],
        "mean_silent_steps": sum(item["silent_steps"] for item in silence) / count,
        "mean_silent_start_energy": sum(item["silent_start_energy"] for item in silence) / count,
        "mean_silent_end_energy": sum(item["silent_end_energy"] for item in silence) / count,
    }


def _timing_score(
    *,
    variant: str,
    silent_steps: int,
    mode: str,
    cue_amplitude: float,
) -> float:
    """Return the pre-query scalar state after a cue and a logical interval."""

    memory = memory_config_for_variant(variant)
    state = seed_memory_state(state_dim=1, channel_count=memory.channel_count)
    for _ in range(2):
        state = step_memory(state, [cue_amplitude], memory)
    for _ in range(silent_steps):
        if mode == "freeze":
            state = DigitalMemoryState(u=list(state.u), chi=[list(channel) for channel in state.chi], t=state.t + memory.dt)
        else:
            state = step_memory(state, [0.0], memory)
    return state.u[0]


def _interval_timing_probe(config: ThinkingBetweenTokensConfig) -> dict[str, object]:
    """Test whether silent evolution creates a usable elapsed-time state."""

    development_scores: dict[str, list[float]] = {"short": [], "long": []}
    for index in range(config.seed_count, config.seed_count * 2):
        rng = random.Random(10_000 + index)
        amplitude = 1.0 + rng.gauss(0.0, config.timing_cue_noise_std)
        development_scores["short"].append(_timing_score(
            variant=config.variant,
            silent_steps=config.timing_short_steps,
            mode="evolve",
            cue_amplitude=amplitude,
        ))
        development_scores["long"].append(_timing_score(
            variant=config.variant,
            silent_steps=config.timing_long_steps,
            mode="evolve",
            cue_amplitude=amplitude,
        ))
    threshold = (sum(development_scores["short"]) / len(development_scores["short"]) +
                 sum(development_scores["long"]) / len(development_scores["long"])) / 2.0

    outcomes: dict[str, dict[str, object]] = {}
    for mode in ("freeze", "evolve"):
        predictions: list[DigitalPrediction] = []
        scores: dict[str, list[float]] = {"short": [], "long": []}
        for index in range(config.seed_count):
            label = "SHORT" if index % 2 == 0 else "LONG"
            steps = config.timing_short_steps if label == "SHORT" else config.timing_long_steps
            rng = random.Random(index)
            score = _timing_score(
                variant=config.variant,
                silent_steps=steps,
                mode=mode,
                cue_amplitude=1.0 + rng.gauss(0.0, config.timing_cue_noise_std),
            )
            predicted = "SHORT" if score >= threshold else "LONG"
            predictions.append(DigitalPrediction(target=label, predicted=predicted, score=score))
            scores[label.lower()].append(score)
        outcomes[mode] = {
            "count": len(predictions),
            "correct": sum(prediction.correct for prediction in predictions),
            "accuracy": sum(prediction.correct for prediction in predictions) / len(predictions),
            "mean_short_score": sum(scores["short"]) / len(scores["short"]),
            "mean_long_score": sum(scores["long"]) / len(scores["long"]),
        }
    return {
        "task": "elapsed_interval_classification",
        "short_steps": config.timing_short_steps,
        "long_steps": config.timing_long_steps,
        "cue_noise_std": config.timing_cue_noise_std,
        "development_threshold": threshold,
        "frozen_state": outcomes["freeze"],
        "free_evolution": outcomes["evolve"],
    }


def run_thinking_between_tokens_experiment(config: ThinkingBetweenTokensConfig) -> dict[str, object]:
    """Run the zero-interval control and free-evolution condition.

    Both conditions use identical generated episodes and the same pulse count.
    The free-evolution arm inserts zero-forcing SOE steps; the zero-interval
    arm performs no update between token pulses, as specified by D2.
    """

    base = SymbolicInductionRunConfig(
        task=config.task,
        variant=config.variant,
        seed_count=config.seed_count,
        pulse_steps=config.pulse_steps,
        silence_steps=0,
        baseline_window=config.baseline_window,
        binding_noise_std=config.binding_noise_std,
    )
    frozen = replace(base, silence_steps=config.silence_steps, silence_mode="freeze")
    free = replace(base, silence_steps=config.silence_steps, silence_mode="evolve")
    zero_result = _run_condition(base)
    frozen_result = _run_condition(frozen)
    free_result = _run_condition(free)
    memory = memory_config_for_variant(config.variant)
    return {
        "experiment_name": "thinking_between_tokens",
        "variant": config.variant,
        "task": config.task.to_mapping(),
        "config": config.to_mapping(),
        "memory_config": memory.to_mapping(),
        "memory_diagnostics": {
            "deff": memory.deff(),
            "stability_ratio": memory.stability_ratio(),
            "note": "D_eff is kernel-defined and therefore constant during fixed-kernel silent evolution.",
        },
        "zero_interval": zero_result,
        "frozen_state": frozen_result,
        "free_evolution": free_result,
        "interval_timing": _interval_timing_probe(config),
        "accuracy_delta": free_result["summary"]["accuracy"] - zero_result["summary"]["accuracy"],
        "confidence_delta": free_result["summary"]["mean_score"] - zero_result["summary"]["mean_score"],
    }


def format_thinking_between_tokens_report(result: Mapping[str, object]) -> str:
    zero = result["zero_interval"]
    frozen = result["frozen_state"]
    free = result["free_evolution"]
    timing = result["interval_timing"]
    task = result["task"]
    return "\n".join([
        "Thinking Between Tokens Report",
        "==============================",
        f"Variant: {result['variant']}",
        f"Gap: {task['gap']}; pairs: {task['pair_count']}",
        "",
        "Zero-interval control",
        f"  Accuracy: {zero['summary']['accuracy']:.3f}",
        f"  Confidence: {zero['summary']['mean_score']:.6f}",
        "",
        "Duration-matched frozen state",
        f"  Accuracy: {frozen['summary']['accuracy']:.3f}",
        f"  Confidence: {frozen['summary']['mean_score']:.6f}",
        "",
        "Free evolution",
        f"  Accuracy: {free['summary']['accuracy']:.3f}",
        f"  Confidence: {free['summary']['mean_score']:.6f}",
        f"  Silent steps: {free['mean_silent_steps']:.1f}",
        f"  Silent memory energy: {free['mean_silent_start_energy']:.6f} -> {free['mean_silent_end_energy']:.6f}",
        "",
        f"Accuracy delta: {result['accuracy_delta']:.3f}",
        f"Confidence delta: {result['confidence_delta']:.6f}",
        "",
        "Interval timing task",
        f"  Frozen accuracy: {timing['frozen_state']['accuracy']:.3f}",
        f"  Free-evolution accuracy: {timing['free_evolution']['accuracy']:.3f}",
    ])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Milestone D2 free-evolution probe.")
    parser.add_argument("--variant", choices=["full", "no_slow", "collapsed_gamma"], default="full")
    parser.add_argument("--gap", type=int, default=16)
    parser.add_argument("--silence-steps", type=int, default=4)
    parser.add_argument("--seeds", type=int, default=16)
    return parser


def main(argv: list[str] | None = None) -> dict[str, object]:
    args = build_arg_parser().parse_args(argv)
    result = run_thinking_between_tokens_experiment(
        ThinkingBetweenTokensConfig(
            task=SymbolicInductionConfig(gap=args.gap),
            variant=args.variant,
            seed_count=args.seeds,
            silence_steps=args.silence_steps,
        )
    )
    print(format_thinking_between_tokens_report(result))
    return result


if __name__ == "__main__":
    main()
