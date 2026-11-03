"""Stateful digital runtime spine for E-F0 reasoning experiments.

F0 coordinates memory stepping, phase transitions, continuous trace storage,
checkpointing, prediction-error diagnostics, and stability checks. It keeps
kernel adaptation disabled so that orchestration can be validated separately
from the F1-F5 learning mechanisms.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Mapping, Sequence

import numpy as np

from ..digital.bridge import (
    DigitalMemoryConfig,
    DigitalMemoryState,
    seed_memory_state,
    step_memory,
)
from ..digital.director import DigitalEpisode
from ..digital.traces import DigitalTrace, DigitalTraceStep
from .digital_trace_store import DigitalRuntimeRecord, DigitalTraceStore
from .phase_controller import (
    DigitalPhaseConfig,
    DigitalPhaseController,
    DigitalRuntimePhase,
)


@dataclass(slots=True)
class DigitalRuntimeConfig:
    trace_capacity: int = 20_000
    prediction_error_alpha: float = 0.1
    phase: DigitalPhaseConfig = field(default_factory=DigitalPhaseConfig)
    fail_on_stability_violation: bool = True

    def __post_init__(self) -> None:
        if self.trace_capacity <= 0:
            raise ValueError("trace_capacity must be positive")
        if not 0.0 < self.prediction_error_alpha <= 1.0:
            raise ValueError("prediction_error_alpha must be in (0, 1]")

    def to_mapping(self) -> dict[str, object]:
        return {
            "trace_capacity": self.trace_capacity,
            "prediction_error_alpha": self.prediction_error_alpha,
            "phase": self.phase.to_mapping(),
            "fail_on_stability_violation": self.fail_on_stability_violation,
        }


@dataclass(slots=True)
class DigitalStabilityAssessment:
    passed: bool
    form: str
    all_finite: bool
    euler_step_stable: bool
    analytical_margin_applicable: bool
    stability_ratio: float
    notes: list[str] = field(default_factory=list)

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class DigitalStabilityMonitor:
    """Check finite states and the correct stability contract for each form.

    Input-driven Formulation A has a triangular linear system: its numerical
    condition for explicit Euler is dt*rate < 2 for each decay rate. The
    negative-feedback condition sum(w/(gamma*leak)) < margin applies only to
    Formulation B. The ratio is reported for Formulation A, but is not used as
    its stability gate.
    """

    stability_margin: float = 0.9
    tolerance: float = 1e-12

    def __post_init__(self) -> None:
        if not 0.0 < self.stability_margin < 1.0:
            raise ValueError("stability_margin must be in (0, 1)")
        if self.tolerance < 0.0:
            raise ValueError("tolerance must be non-negative")

    def assess(
        self,
        state: DigitalMemoryState,
        config: DigitalMemoryConfig,
    ) -> DigitalStabilityAssessment:
        numeric = [state.t, *state.u]
        numeric.extend(value for channel in state.chi for value in channel)
        finite = all(math.isfinite(float(value)) for value in numeric)
        config_finite = all(
            math.isfinite(float(value))
            for value in [config.dt, config.leak_rate, *config.gamma, *config.w]
        )
        positive_kernel = config_finite and all(rate > 0.0 for rate in config.gamma) and all(
            weight >= 0.0 for weight in config.w
        )

        # Form the exact discrete amplification matrix for step_memory().
        # That step first updates chi, then computes memory feedback from
        # chi_next, so a continuous-time Jacobian is not the exact map.
        euler_stable = False
        if positive_kernel:
            channel_count = config.channel_count
            dt = config.dt
            amplification = np.zeros((channel_count + 1, channel_count + 1), dtype=np.float64)
            amplification[0, 0] = 1.0 - dt * config.leak_rate
            for index, (weight, rate) in enumerate(zip(config.w, config.gamma), start=1):
                chi_decay = 1.0 - dt * rate
                amplification[index, index] = chi_decay
                if config.form == "input_driven":
                    amplification[0, index] = dt * weight * chi_decay
                else:
                    amplification[index, 0] = dt
                    amplification[0, index] = -dt * weight * chi_decay
                    amplification[0, 0] -= (dt ** 2) * weight
            try:
                eigenvalues = np.linalg.eigvals(amplification)
                euler_stable = bool(
                    np.all(np.isfinite(eigenvalues))
                    and np.max(np.abs(eigenvalues)) < 1.0
                )
            except np.linalg.LinAlgError:
                euler_stable = False
        ratio = config.stability_ratio()
        margin_applicable = config.form == "negative_feedback"
        notes: list[str] = []
        passed = finite and positive_kernel and euler_stable
        if not finite:
            notes.append("non-finite memory state or time")
        if not positive_kernel:
            notes.append("kernel weights must be non-negative and decay rates positive")
        if not euler_stable:
            notes.append("explicit Euler amplification matrix has spectral radius >= 1")
        if margin_applicable and ratio > self.stability_margin + self.tolerance:
            passed = False
            notes.append("negative-feedback kernel exceeds the configured analytical stability margin")
        elif not margin_applicable:
            notes.append("input-driven triangular dynamics checked with the Euler amplification matrix")
        return DigitalStabilityAssessment(
            passed=passed,
            form=config.form,
            all_finite=finite,
            euler_step_stable=euler_stable,
            analytical_margin_applicable=margin_applicable,
            stability_ratio=ratio,
            notes=notes,
        )


@dataclass(slots=True)
class DigitalRuntimeEpisodeResult:
    final_state: DigitalMemoryState
    trace: DigitalTrace
    target: str | None
    query: list[float]
    episode_id: str
    reset_before_episode: bool
    first_step: int
    last_step: int
    phase_counts: dict[str, int]
    mean_prediction_error: float
    stability_checks: int
    stability_violations: int

    def to_mapping(self) -> dict[str, object]:
        return {
            "final_state": self.final_state.to_mapping(),
            "trace": self.trace.to_mapping(),
            "target": self.target,
            "query": list(self.query),
            "episode_id": self.episode_id,
            "reset_before_episode": self.reset_before_episode,
            "first_step": self.first_step,
            "last_step": self.last_step,
            "phase_counts": dict(self.phase_counts),
            "mean_prediction_error": self.mean_prediction_error,
            "stability_checks": self.stability_checks,
            "stability_violations": self.stability_violations,
        }


@dataclass(slots=True)
class DigitalRuntime:
    """Persistent coordinator around the existing digital SOE stepper."""

    memory_config: DigitalMemoryConfig
    state_dim: int
    config: DigitalRuntimeConfig = field(default_factory=DigitalRuntimeConfig)
    session_id: str = "digital-session"
    monitor: DigitalStabilityMonitor = field(default_factory=DigitalStabilityMonitor)
    memory_state: DigitalMemoryState = field(init=False)
    trace_store: DigitalTraceStore = field(init=False)
    phase_controller: DigitalPhaseController = field(init=False)
    continuity_id: int = field(default=0, init=False)
    episode_count: int = field(default=0, init=False)
    step_count: int = field(default=0, init=False)
    smoothed_prediction_error: float = field(default=0.0, init=False)
    last_prediction_error: float | None = field(default=None, init=False)
    stability_checks: int = field(default=0, init=False)
    stability_violations: int = field(default=0, init=False)
    phase_counts: dict[str, int] = field(default_factory=dict, init=False)
    _previous_u: list[float] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.state_dim <= 0:
            raise ValueError("state_dim must be positive")
        if not self.session_id:
            raise ValueError("session_id must be non-empty")
        self.memory_config = DigitalMemoryConfig.from_mapping(self.memory_config.to_mapping())
        self.memory_state = seed_memory_state(
            state_dim=self.state_dim,
            channel_count=self.memory_config.channel_count,
        )
        self.trace_store = DigitalTraceStore(capacity=self.config.trace_capacity)
        self.phase_controller = DigitalPhaseController(config=self.config.phase)
        self._previous_u = list(self.memory_state.u)

    @property
    def phase(self) -> DigitalRuntimePhase:
        return self.phase_controller.phase

    def reset_memory(self) -> None:
        """Reset episodic contents but preserve runtime phase and counters."""
        self.memory_state = seed_memory_state(
            state_dim=self.state_dim,
            channel_count=self.memory_config.channel_count,
        )
        self._previous_u = list(self.memory_state.u)
        self.continuity_id += 1
        self.smoothed_prediction_error = 0.0
        self.last_prediction_error = None

    def evaluate(self, callback, *, reason: str = "task evaluation"):
        """Run an evaluator inside an explicit EVALUATE phase.

        A scheduled EVALUATE phase may already be active at an event boundary.
        In that case this callback consumes the scheduled phase rather than
        trying to nest another transient phase.
        """
        already_evaluating = self.phase == DigitalRuntimePhase.EVALUATE
        if self.phase in {DigitalRuntimePhase.PRUNE, DigitalRuntimePhase.GROW}:
            raise RuntimeError("cannot run task evaluation during a maintenance phase")
        if not already_evaluating:
            self.phase_controller.begin_transient(DigitalRuntimePhase.EVALUATE, reason=reason)
        try:
            return callback()
        finally:
            if not already_evaluating:
                self.phase_controller.finish_transient("evaluation callback completed")

    def request_prune(self, *, reason: str = "F0 maintenance review") -> None:
        """Schedule a non-mutating prune proposal phase; F4 owns actual pruning."""
        self.phase_controller.request_maintenance("PRUNE", reason=reason)

    def request_grow(self, *, reason: str = "F0 capacity review") -> None:
        """Schedule a non-mutating grow proposal phase; F4 owns channel creation."""
        self.phase_controller.request_maintenance("GROW", reason=reason)

    @staticmethod
    def _copy_state(state: DigitalMemoryState) -> DigitalMemoryState:
        return DigitalMemoryState(
            u=list(state.u),
            chi=[list(channel) for channel in state.chi],
            t=float(state.t),
        )

    def step(
        self,
        forcing: Sequence[float],
        *,
        token: str | None = None,
        episode_id: str = "stream",
        reward: float | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> DigitalRuntimeRecord:
        if len(forcing) != self.state_dim:
            raise ValueError("forcing dimension must match runtime state_dim")
        active_phase = self.phase_controller.phase
        previous_u = list(self.memory_state.u)
        next_state = step_memory(self.memory_state, forcing, self.memory_config)
        assessment = self.monitor.assess(next_state, self.memory_config)
        self.stability_checks += 1
        if not assessment.passed:
            self.stability_violations += 1
            if self.config.fail_on_stability_violation:
                raise RuntimeError(
                    "digital runtime stability violation: " + "; ".join(assessment.notes)
                )

        squared = sum((current - previous) ** 2 for current, previous in zip(next_state.u, previous_u))
        prediction_error = math.sqrt(squared / self.state_dim)
        alpha = self.config.prediction_error_alpha
        self.smoothed_prediction_error = (
            prediction_error
            if self.last_prediction_error is None
            else (1.0 - alpha) * self.smoothed_prediction_error + alpha * prediction_error
        )
        self.last_prediction_error = prediction_error
        self.memory_state = next_state
        record = DigitalRuntimeRecord(
            step_index=self.step_count,
            t=next_state.t,
            session_id=self.session_id,
            continuity_id=self.continuity_id,
            episode_id=episode_id,
            phase=active_phase.value,
            u=list(next_state.u),
            chi=[list(channel) for channel in next_state.chi],
            forcing=[float(value) for value in forcing],
            token=token,
            prediction_error=prediction_error,
            reward=None if reward is None else float(reward),
            stability_ratio=assessment.stability_ratio,
            metadata=dict(metadata or {}),
        )
        self.trace_store.append(record)
        self.phase_counts[active_phase.value] = self.phase_counts.get(active_phase.value, 0) + 1
        self.step_count += 1
        self._previous_u = list(next_state.u)
        self.phase_controller.complete_step()
        return record

    def run_episode(
        self,
        episode: DigitalEpisode,
        *,
        episode_id: str | None = None,
        reset_memory: bool = True,
    ) -> DigitalRuntimeEpisodeResult:
        """Run one task episode, optionally carrying memory from its predecessor."""
        if reset_memory:
            self.reset_memory()
        checks_before = self.stability_checks
        violations_before = self.stability_violations
        self.episode_count += 1
        active_episode_id = episode_id or f"episode-{self.episode_count}"
        first_step = self.step_count
        local_phase_counts: dict[str, int] = {}
        local_errors: list[float] = []
        local_trace = DigitalTrace(
            experiment_name="digital_f0_episode",
            metadata={
                **dict(episode.metadata),
                "session_id": self.session_id,
                "episode_id": active_episode_id,
                "continuity_id": self.continuity_id,
                "reset_before_episode": reset_memory,
            },
        )
        for step_index, (forcing, token) in enumerate(zip(episode.forcings, episode.tokens)):
            record = self.step(
                forcing,
                token=token,
                episode_id=active_episode_id,
                metadata={"episode_step": step_index},
            )
            local_phase_counts[record.phase] = local_phase_counts.get(record.phase, 0) + 1
            if record.prediction_error is not None:
                local_errors.append(record.prediction_error)
            local_trace.add_step(
                DigitalTraceStep(
                    step=step_index,
                    t=record.t,
                    token=record.token,
                    forcing=list(record.forcing),
                    u=list(record.u),
                    chi=[list(channel) for channel in record.chi],
                    metadata={
                        "runtime_step": record.step_index,
                        "phase": record.phase,
                        "prediction_error": record.prediction_error,
                        "continuity_id": record.continuity_id,
                    },
                )
            )
        last_step = self.step_count - 1
        return DigitalRuntimeEpisodeResult(
            final_state=self._copy_state(self.memory_state),
            trace=local_trace,
            target=episode.target,
            query=list(episode.query),
            episode_id=active_episode_id,
            reset_before_episode=reset_memory,
            first_step=first_step,
            last_step=last_step,
            phase_counts=local_phase_counts,
            mean_prediction_error=(sum(local_errors) / len(local_errors) if local_errors else 0.0),
            stability_checks=self.stability_checks - checks_before,
            stability_violations=self.stability_violations - violations_before,
        )

    def runtime_summary(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "step_count": self.step_count,
            "episode_count": self.episode_count,
            "continuity_id": self.continuity_id,
            "phase": self.phase.value,
            "phase_counts": dict(self.phase_counts),
            "phase_events": [event.to_mapping() for event in self.phase_controller.events],
            "stability_checks": self.stability_checks,
            "stability_violations": self.stability_violations,
            "trace_records_retained": len(self.trace_store),
            "trace_capacity": self.trace_store.capacity,
            "smoothed_prediction_error": self.smoothed_prediction_error,
            "memory_config": self.memory_config.to_mapping(),
        }

    def snapshot(self) -> dict[str, object]:
        """Return a versioned JSON-compatible checkpoint of runtime state."""
        return {
            "schema_version": 1,
            "state_dim": self.state_dim,
            "session_id": self.session_id,
            "memory_config": self.memory_config.to_mapping(),
            "runtime_config": self.config.to_mapping(),
            "memory_state": self.memory_state.to_mapping(),
            "continuity_id": self.continuity_id,
            "episode_count": self.episode_count,
            "step_count": self.step_count,
            "smoothed_prediction_error": self.smoothed_prediction_error,
            "last_prediction_error": self.last_prediction_error,
            "stability_checks": self.stability_checks,
            "stability_violations": self.stability_violations,
            "phase_counts": dict(self.phase_counts),
            "phase_controller": self.phase_controller.to_mapping(),
            "monitor": {
                "stability_margin": self.monitor.stability_margin,
                "tolerance": self.monitor.tolerance,
            },
        }

    def restore(self, snapshot: Mapping[str, object]) -> None:
        """Restore dynamics and phase state; trace history remains a separate artifact."""
        if int(snapshot.get("schema_version", 0)) != 1:
            raise ValueError("unsupported digital runtime checkpoint schema")
        if int(snapshot["state_dim"]) != self.state_dim:
            raise ValueError("checkpoint state_dim does not match runtime")
        memory_config = DigitalMemoryConfig.from_mapping(dict(snapshot["memory_config"]))
        runtime_data = dict(snapshot["runtime_config"])
        phase_data = dict(runtime_data["phase"])
        restored_config = DigitalRuntimeConfig(
            trace_capacity=int(runtime_data["trace_capacity"]),
            prediction_error_alpha=float(runtime_data["prediction_error_alpha"]),
            phase=DigitalPhaseConfig(**phase_data),
            fail_on_stability_violation=bool(runtime_data["fail_on_stability_violation"]),
        )
        monitor_data = dict(snapshot.get("monitor", {}))
        restored_monitor = DigitalStabilityMonitor(
            stability_margin=float(monitor_data.get("stability_margin", 0.9)),
            tolerance=float(monitor_data.get("tolerance", 1e-12)),
        )
        state_data = dict(snapshot["memory_state"])
        state = DigitalMemoryState(
            u=[float(value) for value in state_data["u"]],
            chi=[[float(value) for value in channel] for channel in state_data["chi"]],
            t=float(state_data["t"]),
        )
        if len(state.u) != self.state_dim or len(state.chi) != memory_config.channel_count:
            raise ValueError("checkpoint memory-state dimensions are invalid")
        if any(len(channel) != self.state_dim for channel in state.chi):
            raise ValueError("checkpoint channel dimensions are invalid")
        assessment = restored_monitor.assess(state, memory_config)
        if not assessment.passed:
            raise ValueError("checkpoint state/configuration fails stability checks")
        restored_controller = DigitalPhaseController.from_mapping(
            dict(snapshot["phase_controller"])
        )
        if restored_controller.total_steps != int(snapshot["step_count"]):
            raise ValueError("checkpoint step counters are inconsistent")
        self.config = restored_config
        self.monitor = restored_monitor
        self.trace_store = DigitalTraceStore(capacity=restored_config.trace_capacity)
        self.phase_controller = restored_controller
        self.memory_config = memory_config
        self.memory_state = state
        self.session_id = str(snapshot["session_id"])
        self.continuity_id = int(snapshot["continuity_id"])
        self.episode_count = int(snapshot["episode_count"])
        self.step_count = int(snapshot["step_count"])
        self.smoothed_prediction_error = float(snapshot["smoothed_prediction_error"])
        last_error = snapshot.get("last_prediction_error")
        self.last_prediction_error = None if last_error is None else float(last_error)
        self.stability_checks = int(snapshot["stability_checks"])
        self.stability_violations = int(snapshot["stability_violations"])
        self.phase_counts = {
            str(key): int(value) for key, value in dict(snapshot["phase_counts"]).items()
        }
        self._previous_u = list(state.u)
