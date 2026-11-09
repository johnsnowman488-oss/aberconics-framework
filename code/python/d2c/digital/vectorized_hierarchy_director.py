"""Two-level hierarchy whose memory state at every level uses the vector C++ ABI.

The hierarchy coupling schedule remains simultaneous and bottom-up/top-down:
both couplings read the previous event's level states; each level then takes
one vector-valued negative-feedback memory step. Top-down kernel modulation
preserves G3's positive-kernel rule and optional unit-sum weight normalization.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .bridge import DigitalMemoryConfig, DigitalMemoryState, seed_memory_state
from .director import DigitalEpisode
from .native_stepper import CppVectorizedMemoryStepper
from .traces import DigitalTrace, DigitalTraceStep


def vectorized_top_down_kernel(
    base: DigitalMemoryConfig,
    context_state: Sequence[float],
    *,
    gain: float,
    normalize_weights: bool = True,
) -> DigitalMemoryConfig:
    """Materialize G3-style positive top-down kernel modulation.

    The observable drive is mean absolute context activity. Gamma is scaled
    by exp(-gain*drive); positive weights are scaled by exp(gain*drive), then
    optionally normalized to unit sum. A stable normalization avoids
    overflow while preserving the G3 materialization rule.
    """
    if not math.isfinite(gain) or gain <= 0.0:
        raise ValueError("top-down gain must be finite and positive")
    if not context_state:
        raise ValueError("context_state must not be empty")
    if any(not math.isfinite(float(value)) for value in context_state):
        raise ValueError("context_state must be finite")
    drive = sum(abs(float(value)) for value in context_state) / len(context_state)
    gamma_scale = math.exp(-gain * drive)
    gamma = [float(value) * gamma_scale for value in base.gamma]
    if any(not math.isfinite(value) or value <= 0.0 for value in gamma):
        raise RuntimeError("top-down modulation underflowed a positive decay rate")
    if normalize_weights:
        # The exp(gain*drive) term is common to every positive channel weight,
        # so it cancels exactly under unit-sum normalization.
        base_total = sum(base.w)
        weights = [float(value) / base_total for value in base.w]
    else:
        weight_scale = math.exp(gain * drive)
        weights = [float(value) * weight_scale for value in base.w]
    paired = sorted(zip(gamma, weights), key=lambda item: item[0], reverse=True)
    return DigitalMemoryConfig(
        gamma=[item[0] for item in paired],
        w=[item[1] for item in paired],
        dt=base.dt,
        leak_rate=base.leak_rate,
        form="negative_feedback",
    )


@dataclass(slots=True)
class VectorizedHierarchyEpisodeResult:
    final_state: DigitalMemoryState
    level_traces: list[DigitalTrace]
    active_kernels: list[list[dict[str, list[float]]]]
    stability_ratios: list[list[float]]
    max_state_norms: list[float]
    mean_state_norms: list[float]
    top_down_drive_mean: float
    top_down_gamma_changed: bool
    top_down_weight_sum_min: float
    top_down_weight_sum_max: float

    @property
    def trace(self) -> DigitalTrace:
        return self.level_traces[0]


class VectorizedHierarchicalDigitalDirector:
    """Run a two-level bottom-up/top-down hierarchy with vector memory at both levels.

    The level updates are simultaneous: top-down modulation and bottom-up
    forcing are computed from the pre-event state, then both levels advance.
    The C++ stepper performs only the numerical memory update; orchestration,
    coupling, traces, and diagnostics remain in Python.
    """

    def __init__(
        self,
        *,
        memory_config: DigitalMemoryConfig,
        state_dim: int,
        bottom_up_gain: float = 0.4,
        top_down_gain: float = 0.5,
        context_gamma: Sequence[float] = (0.35, 0.08),
        context_w: Sequence[float] = (0.05, 0.02),
        context_leak: float = 0.5,
        normalize_top_down_weights: bool = True,
        library_path: str | None = None,
    ) -> None:
        if state_dim <= 0:
            raise ValueError("state_dim must be positive")
        if not math.isfinite(bottom_up_gain) or bottom_up_gain <= 0.0:
            raise ValueError("bottom_up_gain must be finite and positive")
        if not math.isfinite(top_down_gain) or top_down_gain <= 0.0:
            raise ValueError("top_down_gain must be finite and positive")
        if len(context_gamma) != len(context_w) or not context_gamma:
            raise ValueError("context gamma and weights must have matching non-empty lengths")
        self.state_dim = int(state_dim)
        self.bottom_up_gain = float(bottom_up_gain)
        self.top_down_gain = float(top_down_gain)
        self.normalize_top_down_weights = bool(normalize_top_down_weights)
        self.base_memory = DigitalMemoryConfig(
            gamma=list(memory_config.gamma), w=list(memory_config.w),
            dt=memory_config.dt, leak_rate=memory_config.leak_rate,
            form="negative_feedback",
        )
        self.context_memory = DigitalMemoryConfig(
            gamma=[float(value) for value in context_gamma],
            w=[float(value) for value in context_w],
            dt=memory_config.dt, leak_rate=float(context_leak),
            form="negative_feedback",
        )
        self.fast_stepper = CppVectorizedMemoryStepper(
            self.base_memory, library_path=library_path
        )
        self.context_stepper = CppVectorizedMemoryStepper(
            self.context_memory, library_path=library_path
        )
        self.step_count = 0

    def run_episode(
        self,
        episode: DigitalEpisode,
        *,
        experiment_name: str = "g4_vectorized_hierarchy",
    ) -> VectorizedHierarchyEpisodeResult:
        if len(episode.forcings) != len(episode.tokens):
            raise ValueError("episode forcings and tokens must have matching lengths")
        if not episode.forcings:
            raise ValueError("episode must contain at least one forcing step")

        states = [
            seed_memory_state(state_dim=self.state_dim, channel_count=self.base_memory.channel_count),
            seed_memory_state(state_dim=self.state_dim, channel_count=self.context_memory.channel_count),
        ]
        traces = [
            DigitalTrace(experiment_name=f"{experiment_name}::level{level}", metadata={"level": level})
            for level in range(2)
        ]
        kernels: list[list[dict[str, list[float]]]] = []
        ratios: list[list[float]] = [[], []]
        norms: list[list[float]] = [[], []]
        drives: list[float] = []
        fast_gamma_start = list(self.base_memory.gamma)
        weight_sums: list[float] = []

        for step_index, (forcing, token) in enumerate(zip(episode.forcings, episode.tokens)):
            external = [float(value) for value in forcing]
            if len(external) != self.state_dim:
                raise ValueError("episode forcing dimension must match state_dim")

            # Both coupling paths use the previous event's state, matching
            # the C++ hierarchy's simultaneous level update semantics.
            context_u = states[1].u
            drive = sum(abs(value) for value in context_u) / self.state_dim
            drives.append(drive)
            active_fast = vectorized_top_down_kernel(
                self.base_memory, context_u, gain=self.top_down_gain,
                normalize_weights=self.normalize_top_down_weights,
            )
            bottom_up = [self.bottom_up_gain * value for value in states[0].u]

            next_fast = self.fast_stepper(states[0], external, active_fast)
            next_context = self.context_stepper(states[1], bottom_up, self.context_memory)
            next_states = [next_fast, next_context]

            event_kernels = [
                {"gamma": list(active_fast.gamma), "w": list(active_fast.w)},
                {"gamma": list(self.context_memory.gamma), "w": list(self.context_memory.w)},
            ]
            kernels.append(event_kernels)
            weight_sums.append(sum(active_fast.w))
            ratios[0].append(active_fast.stability_ratio())
            ratios[1].append(self.context_memory.stability_ratio())

            for level, (state, trace) in enumerate(zip(next_states, traces)):
                if any(not math.isfinite(value) for value in state.u):
                    raise RuntimeError(f"G4 hierarchy level {level} became non-finite")
                if any(not math.isfinite(value) for channel in state.chi for value in channel):
                    raise RuntimeError(f"G4 hierarchy memory level {level} became non-finite")
                norm = math.sqrt(sum(value * value for value in state.u))
                norms[level].append(norm)
                trace.add_step(DigitalTraceStep(
                    step=step_index,
                    t=state.t,
                    token=token if level == 0 else None,
                    forcing=external if level == 0 else bottom_up,
                    u=list(state.u),
                    chi=[list(channel) for channel in state.chi],
                    metadata={
                        "backend": "cpp_vectorized",
                        "active_kernel": event_kernels[level],
                        "stability_ratio": ratios[level][-1],
                        "top_down_drive": drive,
                    },
                ))

            states = next_states
            self.step_count += 1

        final = states[0]
        return VectorizedHierarchyEpisodeResult(
            final_state=final,
            level_traces=traces,
            active_kernels=kernels,
            stability_ratios=ratios,
            max_state_norms=[max(row, default=0.0) for row in norms],
            mean_state_norms=[
                sum(row) / len(row) if row else 0.0 for row in norms
            ],
            top_down_drive_mean=sum(drives) / len(drives) if drives else 0.0,
            top_down_gamma_changed=any(
                abs(a - b) > 1e-12
                for row in kernels for a, b in zip(row[0]["gamma"], fast_gamma_start)
            ),
            top_down_weight_sum_min=min(weight_sums, default=0.0),
            top_down_weight_sum_max=max(weight_sums, default=0.0),
        )
