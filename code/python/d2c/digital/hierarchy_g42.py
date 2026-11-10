"""G4.2 multi-level vector coupling and capacity-adaptive channel maintenance."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .bridge import DigitalMemoryConfig, DigitalMemoryState, seed_memory_state
from .director import DigitalEpisode
from .native_stepper import CppVectorizedMemoryStepper
from .traces import DigitalTrace, DigitalTraceStep


def _copy_config(config: DigitalMemoryConfig) -> DigitalMemoryConfig:
    return DigitalMemoryConfig(
        gamma=list(config.gamma), w=list(config.w), dt=config.dt,
        leak_rate=config.leak_rate, form="negative_feedback",
    )


def _l2(values: Sequence[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in values))


def _structured_state_vector(state: DigitalMemoryState, config: DigitalMemoryConfig) -> list[float]:
    """Combine the vector state and weighted channel memory without scalar pooling."""
    if not state.chi:
        return [float(value) for value in state.u]
    memory = [
        sum(float(weight) * float(channel[index])
            for weight, channel in zip(config.w, state.chi))
        for index in range(len(state.u))
    ]
    return [0.25 * float(value) + memory[index]
            for index, value in enumerate(state.u)]


def _bounded_context(values: Sequence[float], gain: float) -> list[float]:
    """Tanh each coordinate, then cap the entire vector's L2 norm at gain."""
    if not math.isfinite(gain) or gain < 0.0:
        raise ValueError("context gain must be finite and non-negative")
    bounded = [math.tanh(float(value)) for value in values]
    norm = _l2(bounded)
    if norm > gain and norm > 0.0:
        scale = gain / norm
        bounded = [value * scale for value in bounded]
    return bounded


def prune_grow_kernel(
    config: DigitalMemoryConfig,
    channel_activity: Sequence[float],
    *,
    growth_factor: float = 0.5,
):
    """Legacy one-for-one replacement primitive, retained for historical tests."""
    if len(config.gamma) < 2 or len(channel_activity) != len(config.gamma):
        raise ValueError("prune/grow requires at least two channels and one activity per channel")
    if not 0.0 < growth_factor < 1.0:
        raise ValueError("growth_factor must be between zero and one")
    if any(not math.isfinite(float(v)) or float(v) < 0.0 for v in channel_activity):
        raise ValueError("channel activity must be finite and non-negative")
    slowest = min(config.gamma)
    candidates = [i for i, value in enumerate(config.gamma)
                  if math.isclose(value, slowest, rel_tol=1e-12, abs_tol=1e-15)]
    index = min(candidates, key=lambda i: (float(channel_activity[i]), config.w[i]))
    mass, ratio = sum(config.w), config.stability_ratio()
    keep = [i for i in range(len(config.gamma)) if i != index]
    kept_mass = sum(config.w[i] for i in keep)
    kept_ratio = sum(config.w[i] / (config.gamma[i] * config.leak_rate) for i in keep)
    new_gamma = slowest * growth_factor
    mean_ratio = kept_ratio / kept_mass
    denominator = 1.0 / (new_gamma * config.leak_rate) - mean_ratio
    numerator = ratio - mass * mean_ratio
    if denominator <= 0.0 or numerator <= 0.0:
        raise RuntimeError("positive mass/stability-preserving growth solution does not exist")
    new_weight = numerator / denominator
    scale = (mass - new_weight) / kept_mass
    if new_weight <= 0.0 or scale <= 0.0:
        raise RuntimeError("prune/grow produced a non-positive kernel weight")
    pairs = [(config.gamma[i], config.w[i] * scale) for i in keep] + [(new_gamma, new_weight)]
    pairs.sort(key=lambda pair: pair[0], reverse=True)
    updated = DigitalMemoryConfig(
        gamma=[p[0] for p in pairs], w=[p[1] for p in pairs],
        dt=config.dt, leak_rate=config.leak_rate, form="negative_feedback",
    )
    if not math.isclose(sum(updated.w), mass, abs_tol=1e-10):
        raise RuntimeError("kernel mass changed during prune/grow")
    if not math.isclose(updated.stability_ratio(), ratio, abs_tol=1e-10):
        raise RuntimeError("stability ratio changed during prune/grow")
    return updated, {
        "operation": "prune_then_grow", "pruned_index": index,
        "pruned_gamma": slowest, "pruned_weight": config.w[index],
        "pruned_activity": float(channel_activity[index]),
        "grown_gamma": new_gamma, "grown_weight": new_weight,
        "growth_factor": growth_factor, "channel_count_before": config.channel_count,
        "channel_count_after": updated.channel_count,
        "kernel_mass_before": mass, "kernel_mass_after": sum(updated.w),
        "stability_ratio_before": ratio, "stability_ratio_after": updated.stability_ratio(),
        "selection_rule": "least-active channel within the slowest-timescale band",
        "controlled_probe": True,
    }


def _fit_positive_spectrum(
    config: DigitalMemoryConfig, gamma_values: Sequence[float]
) -> DigitalMemoryConfig:
    """Fit positive weights on a new spectrum while exactly preserving mass and stability."""
    gammas = sorted((float(value) for value in gamma_values), reverse=True)
    if len(gammas) < 2 or any(not math.isfinite(v) or v <= 0.0 for v in gammas):
        raise ValueError("a fitted spectrum needs at least two finite positive decay rates")
    mass = sum(config.w)
    target_inverse_gamma = config.stability_ratio() * config.leak_rate / mass
    inverse = [1.0 / value for value in gammas]
    low, high = min(inverse), max(inverse)
    if not low < target_inverse_gamma < high:
        raise RuntimeError("new spectrum cannot preserve the existing mass/stability pair")
    span = high - low
    coordinates = [(value - low) / span for value in inverse]
    target = (target_inverse_gamma - low) / span

    def distribution(alpha: float) -> tuple[list[float], float]:
        logits = [alpha * value for value in coordinates]
        peak = max(logits)
        weights = [math.exp(value - peak) for value in logits]
        total = sum(weights)
        probs = [value / total for value in weights]
        mean = sum(p * x for p, x in zip(probs, coordinates))
        return probs, mean

    left, right = -80.0, 80.0
    for _ in range(90):
        middle = 0.5 * (left + right)
        _, mean = distribution(middle)
        if mean < target:
            left = middle
        else:
            right = middle
    probs, _ = distribution(0.5 * (left + right))
    weights = [mass * value for value in probs]
    fitted = DigitalMemoryConfig(
        gamma=gammas, w=weights, dt=config.dt,
        leak_rate=config.leak_rate, form="negative_feedback",
    )
    if not math.isclose(sum(fitted.w), mass, rel_tol=0.0, abs_tol=1e-10):
        raise RuntimeError("spectrum fitting failed to preserve kernel mass")
    if not math.isclose(fitted.stability_ratio(), config.stability_ratio(),
                        rel_tol=0.0, abs_tol=1e-9):
        raise RuntimeError("spectrum fitting failed to preserve stability")
    return fitted


def adaptive_capacity_update(
    config: DigitalMemoryConfig,
    channel_activity: Sequence[float],
    *,
    growth_factor: float = 0.5,
    prune_relative_threshold: float = 0.01,
    growth_share_threshold: float = 0.05,
) -> tuple[DigitalMemoryConfig, dict]:
    """Prune a nearly inactive channel or grow a slower channel when activity is distributed.

    A prune is allowed only when the remaining spectrum can still represent the
    existing positive kernel mass and stability ratio. Otherwise the policy
    grows a slower channel when observed activity is non-zero and no channel
    is clearly dormant. Channel count can therefore genuinely increase or
    decrease; this is not the old fixed-count replacement operation.
    """
    if len(config.gamma) != len(channel_activity) or len(config.gamma) < 2:
        raise ValueError("activity must contain one value per configured channel")
    if not 0.0 < growth_factor < 1.0:
        raise ValueError("growth_factor must be between zero and one")
    if not 0.0 <= prune_relative_threshold < 1.0:
        raise ValueError("prune threshold must be in [0, 1)")
    if not 0.0 < growth_share_threshold <= 1.0:
        raise ValueError("growth share threshold must be in (0, 1]")
    activity = [float(value) for value in channel_activity]
    if any(not math.isfinite(value) or value < 0.0 for value in activity):
        raise ValueError("channel activity must be finite and non-negative")

    mass, stability = sum(config.w), config.stability_ratio()
    maximum = max(activity, default=0.0)
    minimum = min(activity, default=0.0)
    before = config.channel_count
    if before > 2 and maximum > 0.0 and minimum / maximum <= prune_relative_threshold:
        for index in sorted(range(before), key=lambda i: (activity[i], config.w[i])):
            spectrum = [value for i, value in enumerate(config.gamma) if i != index]
            try:
                updated = _fit_positive_spectrum(config, spectrum)
            except (ValueError, RuntimeError):
                continue
            return updated, {
                "operation": "prune", "pruned_index": index,
                "pruned_gamma": config.gamma[index], "pruned_weight": config.w[index],
                "pruned_activity": activity[index], "grown_gamma": None,
                "channel_count_before": before, "channel_count_after": updated.channel_count,
                "capacity_delta": -1, "channel_deaths": 1, "channel_births": 0,
                "kernel_mass_before": mass, "kernel_mass_after": sum(updated.w),
                "stability_ratio_before": stability,
                "stability_ratio_after": updated.stability_ratio(),
                "selection_rule": "lowest-activity channel, subject to exact mass/stability feasibility",
            }

    total_activity = sum(activity)
    maximum_share = maximum / total_activity if total_activity > 0.0 else 0.0
    if maximum > 0.0 and maximum_share >= growth_share_threshold:
        new_gamma = min(config.gamma) * growth_factor
        updated = _fit_positive_spectrum(config, [*config.gamma, new_gamma])
        return updated, {
            "operation": "grow", "pruned_index": None, "pruned_gamma": None,
            "pruned_activity": None, "grown_gamma": new_gamma,
            "channel_count_before": before, "channel_count_after": updated.channel_count,
            "capacity_delta": 1, "channel_deaths": 0, "channel_births": 1,
            "kernel_mass_before": mass, "kernel_mass_after": sum(updated.w),
            "stability_ratio_before": stability,
            "stability_ratio_after": updated.stability_ratio(),
            "maximum_activity_share": maximum_share,
            "growth_factor": growth_factor,
            "selection_rule": "add a slower positive channel when observed activity has no dormant channel",
        }

    unchanged = _copy_config(config)
    return unchanged, {
        "operation": "retain", "channel_count_before": before,
        "channel_count_after": before, "capacity_delta": 0,
        "channel_deaths": 0, "channel_births": 0,
        "kernel_mass_before": mass, "kernel_mass_after": mass,
        "stability_ratio_before": stability, "stability_ratio_after": stability,
        "maximum_activity_share": maximum_share,
        "selection_rule": "retain when activity is absent or below the growth criterion",
    }


@dataclass(slots=True)
class NLevelHierarchyResult:
    final_state: DigitalMemoryState
    final_states: list[DigitalMemoryState]
    level_traces: list[DigitalTrace]
    stability_ratios: list[list[float]]
    max_state_norms: list[float]
    mean_state_norms: list[float]
    bottom_up_input_norms: list[float]
    top_down_input_norms: list[float]

    @property
    def trace(self):
        return self.level_traces[0]


class NLevelVectorizedHierarchyDirector:
    """Simultaneous N-level vector hierarchy with bounded structured messages.

    Every level sends its combined state/memory vector up and down one edge.
    Message vectors are tanh-bounded and L2-clipped before entering the next
    level's forcing. The native runtime remains unchanged; only orchestration
    and coupling live in Python.
    """
    def __init__(
        self, *, level_configs: Sequence[DigitalMemoryConfig], state_dim: int,
        bottom_up_gain: float = 0.4, top_down_gain: float = 0.5,
        preserve_top_down_weight_mass: bool = True, library_path: str | None = None,
    ):
        if len(level_configs) < 2:
            raise ValueError("G4.2 hierarchy requires at least two levels")
        if state_dim <= 0 or not math.isfinite(bottom_up_gain) or bottom_up_gain <= 0.0:
            raise ValueError("state dimension and bottom-up gain must be positive and finite")
        if not math.isfinite(top_down_gain) or top_down_gain <= 0.0:
            raise ValueError("top-down gain must be positive and finite")
        self.state_dim = int(state_dim)
        self.bottom_up_gain = float(bottom_up_gain)
        self.top_down_gain = float(top_down_gain)
        # Kept as a compatibility attribute. Coupling no longer changes gamma
        # or weights: kernel mass and stability are invariant by construction.
        self.preserve_top_down_weight_mass = bool(preserve_top_down_weight_mass)
        self.library_path = library_path
        self.level_configs = [_copy_config(config) for config in level_configs]
        self.steppers = [
            CppVectorizedMemoryStepper(config, library_path=library_path)
            for config in self.level_configs
        ]
        self.activity_sums = [[0.0] * config.channel_count for config in self.level_configs]
        self.activity_samples = 0
        self.step_count = 0

    def reset_activity(self):
        self.activity_sums = [[0.0] * config.channel_count for config in self.level_configs]
        self.activity_samples = 0

    def prune_grow(self, *, growth_factor: float = 0.5):
        if self.activity_samples <= 0:
            raise RuntimeError("prune/grow requires observed training activity")
        events = []
        for level, config in enumerate(self.level_configs):
            activity = [value / self.activity_samples for value in self.activity_sums[level]]
            updated, event = adaptive_capacity_update(
                config, activity, growth_factor=growth_factor,
            )
            event["level"] = level
            if event["operation"] != "retain":
                self.level_configs[level] = updated
                self.steppers[level] = CppVectorizedMemoryStepper(
                    updated, library_path=self.library_path,
                )
            events.append(event)
        self.reset_activity()
        return events

    def run_episode(
        self, episode: DigitalEpisode, *, experiment_name: str = "g4_2_nlevel",
        enable_bottom_up: bool = True, enable_top_down: bool = True,
        initial_states: Sequence[DigitalMemoryState] | None = None,
    ):
        if len(episode.forcings) != len(episode.tokens) or not episode.forcings:
            raise ValueError("episode must have matching non-empty forcings and tokens")
        if initial_states is None:
            states = [
                seed_memory_state(state_dim=self.state_dim, channel_count=config.channel_count)
                for config in self.level_configs
            ]
        else:
            if len(initial_states) != len(self.level_configs):
                raise ValueError("initial_states must contain one state per hierarchy level")
            states = list(initial_states)
            for level, (state, config) in enumerate(zip(states, self.level_configs)):
                if len(state.u) != self.state_dim or len(state.chi) != config.channel_count:
                    raise ValueError(f"initial state shape does not match hierarchy level {level}")
                if any(len(channel) != self.state_dim for channel in state.chi):
                    raise ValueError(f"initial memory shape does not match hierarchy level {level}")

        traces = [
            DigitalTrace(
                experiment_name=f"{experiment_name}::level{level}",
                metadata={"level": level, "depth": len(states)},
            )
            for level in range(len(states))
        ]
        ratios = [[] for _ in states]
        norms = [[] for _ in states]
        bottom_inputs = [[] for _ in states]
        top_inputs = [[] for _ in states]

        for step_index, (forcing, token) in enumerate(zip(episode.forcings, episode.tokens)):
            external = [float(value) for value in forcing]
            if len(external) != self.state_dim:
                raise ValueError("forcing dimension must match state_dim")
            next_states = []
            active_configs = []
            messages = []
            # Messages are computed exclusively from the previous event's
            # states; all levels then advance simultaneously.
            for level, base in enumerate(self.level_configs):
                source = external.copy() if level == 0 else [0.0] * self.state_dim
                bottom = [0.0] * self.state_dim
                top = [0.0] * self.state_dim
                if enable_bottom_up and level > 0:
                    vector = _structured_state_vector(states[level - 1], self.level_configs[level - 1])
                    bottom = _bounded_context(vector, self.bottom_up_gain)
                if enable_top_down and level < len(states) - 1:
                    vector = _structured_state_vector(states[level + 1], self.level_configs[level + 1])
                    top = _bounded_context(vector, self.top_down_gain)
                source = [source[i] + bottom[i] + top[i] for i in range(self.state_dim)]
                next_states.append(self.steppers[level](states[level], source, base))
                active_configs.append(base)
                bottom_inputs[level].append(_l2(bottom))
                top_inputs[level].append(_l2(top))
                messages.append((bottom, top))

            for level, (state, trace, active) in enumerate(zip(next_states, traces, active_configs)):
                if any(not math.isfinite(value) for value in state.u):
                    raise RuntimeError(f"non-finite state at hierarchy level {level}")
                if any(not math.isfinite(value) for channel in state.chi for value in channel):
                    raise RuntimeError(f"non-finite memory at hierarchy level {level}")
                ratio = active.stability_ratio()
                ratios[level].append(ratio)
                norm = _l2(state.u)
                norms[level].append(norm)
                bottom, top = messages[level]
                source = external if level == 0 else bottom
                if level == 0:
                    source = [source[i] + top[i] for i in range(self.state_dim)]
                elif level < len(states) - 1:
                    source = [source[i] + top[i] for i in range(self.state_dim)]
                trace.add_step(DigitalTraceStep(
                    step=step_index, t=state.t,
                    token=token if level == 0 else None,
                    forcing=list(source), u=list(state.u),
                    chi=[list(row) for row in state.chi],
                    metadata={
                        "backend": "cpp_vectorized",
                        "stability_ratio": ratio,
                        "bottom_up_message_l2": _l2(bottom),
                        "top_down_message_l2": _l2(top),
                        "kernel_mass": sum(active.w),
                    },
                ))

            states = next_states
            self.step_count += 1

        for level, (state, config) in enumerate(zip(states, self.level_configs)):
            for channel, values in enumerate(state.chi):
                self.activity_sums[level][channel] += _l2(values) / max(1, len(values))
        self.activity_samples += 1

        return NLevelHierarchyResult(
            final_state=states[0], final_states=list(states), level_traces=traces,
            stability_ratios=ratios,
            max_state_norms=[max(row, default=0.0) for row in norms],
            mean_state_norms=[sum(row) / len(row) if row else 0.0 for row in norms],
            bottom_up_input_norms=[
                sum(row) / len(row) if row else 0.0 for row in bottom_inputs
            ],
            top_down_input_norms=[
                sum(row) / len(row) if row else 0.0 for row in top_inputs
            ],
        )
