"""Generic event-episode runner for the Python digital D2C substrate."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from .bridge import DigitalMemoryConfig, DigitalMemoryState, seed_memory_state, step_memory
from .traces import DigitalTrace, DigitalTraceStep
from ..learning import (
    ConsolidationConfig,
    ConsolidationState,
    KernelUpdateProposal,
    TemporalDifferenceError,
    ThreeFactorUpdateConfig,
    consolidate_weights,
    per_channel_td_errors,
    propose_three_factor_update,
)


@dataclass(slots=True)
class DigitalEpisode:
    """Task-provided forcing schedule and supervised decision metadata."""

    forcings: list[list[float]]
    tokens: list[str | None]
    target: str | None = None
    query: list[float] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.forcings) != len(self.tokens):
            raise ValueError("forcings and tokens must have matching lengths")
        if not self.forcings:
            raise ValueError("episode must contain at least one forcing step")


@dataclass(slots=True)
class DigitalEpisodeResult:
    final_state: DigitalMemoryState
    trace: DigitalTrace
    target: str | None
    query: list[float]
    metadata: dict[str, object]


@dataclass(slots=True)
class DigitalFeedbackResult:
    """Delayed-feedback diagnostics for one completed digital episode.

    Kernel updates remain proposals: early D3 experiments keep the SOE kernel
    fixed while learning only a task readout. This makes credit assignment
    inspectable without silently changing the memory substrate.
    """

    phase: str
    reward: float
    td_error: TemporalDifferenceError
    kernel_proposal: KernelUpdateProposal
    consolidation: dict[str, object]

    def to_mapping(self) -> dict[str, object]:
        return {
            "phase": self.phase,
            "reward": self.reward,
            "td_error": self.td_error.to_mapping(),
            "kernel_proposal": self.kernel_proposal.to_mapping(),
            "consolidation": dict(self.consolidation),
        }


@dataclass(slots=True)
class DigitalDirector:
    """Run arbitrary digital forcing episodes through a fixed SOE memory model.

    Tasks own token construction, forcing schedules, targets, and rewards. This
    class owns only state initialization, stepping, and generic trace capture.
    """

    memory_config: DigitalMemoryConfig
    state_dim: int

    def __post_init__(self) -> None:
        if self.state_dim <= 0:
            raise ValueError("state_dim must be positive")

    def run_episode(self, episode: DigitalEpisode, *, experiment_name: str = "digital_episode") -> DigitalEpisodeResult:
        state = seed_memory_state(state_dim=self.state_dim, channel_count=self.memory_config.channel_count)
        trace = DigitalTrace(experiment_name=experiment_name, metadata=dict(episode.metadata))
        for step_index, (forcing, token) in enumerate(zip(episode.forcings, episode.tokens)):
            state = step_memory(state, forcing, self.memory_config)
            trace.add_step(DigitalTraceStep(
                step=step_index,
                t=state.t,
                token=token,
                forcing=list(forcing),
                u=list(state.u),
                chi=[list(channel) for channel in state.chi],
            ))
        return DigitalEpisodeResult(
            final_state=state,
            trace=trace,
            target=episode.target,
            query=list(episode.query),
            metadata=dict(episode.metadata),
        )

    def apply_delayed_feedback(
        self,
        result: DigitalEpisodeResult,
        *,
        reward: float,
        prediction_error: float,
        current_values: Sequence[float] | None = None,
        phase: str = "LEARN",
    ) -> DigitalFeedbackResult:
        """Turn terminal feedback into per-channel credit diagnostics.

        The returned update and consolidation values are intentionally not
        applied to ``memory_config``. A task may use them to update a readout,
        while a later kernel-learning experiment can explicitly opt into
        applying stable proposals.
        """

        activity = [
            sum(abs(value) for value in channel) / len(channel)
            for channel in result.final_state.chi
        ]
        values = activity if current_values is None else [float(value) for value in current_values]
        if len(values) != self.memory_config.channel_count:
            raise ValueError("current_values must provide one value per memory channel")
        td_error = per_channel_td_errors(
            reward=reward,
            current_values=values,
            next_values=[0.0 for _ in activity],
            gamma=self.memory_config.gamma,
            dt=self.memory_config.dt,
        )
        proposal = propose_three_factor_update(
            weights=self.memory_config.w,
            gamma=self.memory_config.gamma,
            channel_activity=activity,
            prediction_error=prediction_error,
            td_error=td_error.errors,
            config=ThreeFactorUpdateConfig(
                dt=self.memory_config.dt,
                leak_rate=self.memory_config.leak_rate,
            ),
        )
        consolidation = consolidate_weights(
            ConsolidationState(
                fast_weights=proposal.clipped_weights,
                slow_weights=list(self.memory_config.w),
            ),
            ConsolidationConfig(),
        )
        return DigitalFeedbackResult(
            phase=phase,
            reward=float(reward),
            td_error=td_error,
            kernel_proposal=proposal,
            consolidation=consolidation.to_mapping(),
        )

    def commit_kernel_proposal(self, proposal: KernelUpdateProposal) -> None:
        """Apply a previously checked stable proposal to the live SOE kernel."""

        if len(proposal.clipped_weights) != self.memory_config.channel_count:
            raise ValueError("proposal channel count must match the live memory config")
        self.memory_config.w = list(proposal.clipped_weights)
