"""G4.2 N-level vectorized hierarchy and F4 channel mutation."""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Sequence
from .bridge import DigitalMemoryConfig, DigitalMemoryState, seed_memory_state
from .director import DigitalEpisode
from .native_stepper import CppVectorizedMemoryStepper
from .traces import DigitalTrace, DigitalTraceStep
from .vectorized_hierarchy_director import vectorized_top_down_kernel

def prune_grow_kernel(config: DigitalMemoryConfig, channel_activity: Sequence[float], *, growth_factor: float = 0.5):
    if len(config.gamma) < 2 or len(channel_activity) != len(config.gamma):
        raise ValueError("prune/grow requires at least two channels and one activity per channel")
    if not 0.0 < growth_factor < 1.0:
        raise ValueError("growth_factor must be between zero and one")
    if any(not math.isfinite(float(v)) or float(v) < 0.0 for v in channel_activity):
        raise ValueError("channel activity must be finite and non-negative")
    slowest = min(config.gamma)
    candidates = [i for i, value in enumerate(config.gamma) if math.isclose(value, slowest, rel_tol=1e-12, abs_tol=1e-15)]
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
    updated = DigitalMemoryConfig(gamma=[p[0] for p in pairs], w=[p[1] for p in pairs],
                                  dt=config.dt, leak_rate=config.leak_rate, form="negative_feedback")
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

@dataclass(slots=True)
class NLevelHierarchyResult:
    final_state: DigitalMemoryState
    level_traces: list[DigitalTrace]
    stability_ratios: list[list[float]]
    max_state_norms: list[float]
    mean_state_norms: list[float]

    @property
    def trace(self):
        return self.level_traces[0]

class NLevelVectorizedHierarchyDirector:
    def __init__(self, *, level_configs: Sequence[DigitalMemoryConfig], state_dim: int,
                 bottom_up_gain: float = 0.4, top_down_gain: float = 0.5,
                 preserve_top_down_weight_mass: bool = True, library_path: str | None = None):
        if len(level_configs) < 2:
            raise ValueError("G4.2 hierarchy requires at least two levels")
        if state_dim <= 0 or bottom_up_gain <= 0.0 or top_down_gain <= 0.0:
            raise ValueError("state dimension and coupling gains must be positive")
        self.state_dim = int(state_dim)
        self.bottom_up_gain = float(bottom_up_gain)
        self.top_down_gain = float(top_down_gain)
        self.preserve_top_down_weight_mass = bool(preserve_top_down_weight_mass)
        self.library_path = library_path
        self.level_configs = [DigitalMemoryConfig(gamma=list(c.gamma), w=list(c.w), dt=c.dt,
            leak_rate=c.leak_rate, form="negative_feedback") for c in level_configs]
        self.steppers = [CppVectorizedMemoryStepper(c, library_path=library_path) for c in self.level_configs]
        self.activity_sums = [[0.0] * c.channel_count for c in self.level_configs]
        self.activity_samples = 0
        self.step_count = 0

    def reset_activity(self):
        self.activity_sums = [[0.0] * c.channel_count for c in self.level_configs]
        self.activity_samples = 0

    def prune_grow(self, *, growth_factor: float = 0.5):
        if self.activity_samples <= 0:
            raise RuntimeError("prune/grow requires observed training activity")
        events = []
        for level, config in enumerate(self.level_configs):
            activity = [v / self.activity_samples for v in self.activity_sums[level]]
            updated, event = prune_grow_kernel(config, activity, growth_factor=growth_factor)
            event["level"] = level
            self.level_configs[level] = updated
            self.steppers[level] = CppVectorizedMemoryStepper(updated, library_path=self.library_path)
            events.append(event)
        self.reset_activity()
        return events

    def run_episode(self, episode: DigitalEpisode, *, experiment_name: str = "g4_2_nlevel"):
        if len(episode.forcings) != len(episode.tokens) or not episode.forcings:
            raise ValueError("episode must have matching non-empty forcings and tokens")
        states = [seed_memory_state(state_dim=self.state_dim, channel_count=c.channel_count)
                  for c in self.level_configs]
        traces = [DigitalTrace(experiment_name=f"{experiment_name}::level{level}",
                  metadata={"level": level, "depth": len(states)}) for level in range(len(states))]
        ratios, norms = [[] for _ in states], [[] for _ in states]
        for step_index, (forcing, token) in enumerate(zip(episode.forcings, episode.tokens)):
            external = [float(v) for v in forcing]
            if len(external) != self.state_dim:
                raise ValueError("forcing dimension must match state_dim")
            next_states, active_configs = [], []
            for level, base in enumerate(self.level_configs):
                active = base
                if level < len(states) - 1:
                    active = vectorized_top_down_kernel(base, states[level + 1].u,
                        gain=self.top_down_gain, normalize_weights=True,
                        preserve_weight_mass=self.preserve_top_down_weight_mass)
                source = external if level == 0 else [
                    self.bottom_up_gain * v for v in states[level - 1].u]
                next_states.append(self.steppers[level](states[level], source, active))
                active_configs.append(active)
            for level, (state, trace, active) in enumerate(zip(next_states, traces, active_configs)):
                if any(not math.isfinite(v) for v in state.u):
                    raise RuntimeError(f"non-finite state at hierarchy level {level}")
                if any(not math.isfinite(v) for channel in state.chi for v in channel):
                    raise RuntimeError(f"non-finite memory at hierarchy level {level}")
                ratio = active.stability_ratio()
                ratios[level].append(ratio)
                norm = math.sqrt(sum(v * v for v in state.u))
                norms[level].append(norm)
                source = external if level == 0 else [
                    self.bottom_up_gain * v for v in states[level - 1].u]
                trace.add_step(DigitalTraceStep(step=step_index, t=state.t,
                    token=token if level == 0 else None, forcing=list(source),
                    u=list(state.u), chi=[list(row) for row in state.chi],
                    metadata={"backend": "cpp_vectorized",
                              "active_kernel": {"gamma": list(active.gamma), "w": list(active.w)},
                              "stability_ratio": ratio}))
            states = next_states
            self.step_count += 1
        for level, state in enumerate(states):
            for channel, values in enumerate(state.chi):
                self.activity_sums[level][channel] += math.sqrt(sum(v*v for v in values)) / max(1, len(values))
        self.activity_samples += 1
        return NLevelHierarchyResult(
            final_state=states[0], level_traces=traces, stability_ratios=ratios,
            max_state_norms=[max(row, default=0.0) for row in norms],
            mean_state_norms=[sum(row)/len(row) if row else 0.0 for row in norms])
