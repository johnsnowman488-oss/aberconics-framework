"""Lorenz63 memory experiment scaffold mirroring the Julia reference workflow."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from statistics import mean, pstdev
import sys


if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[4]
    python_root = repo_root / "code" / "python"
    if str(python_root) not in sys.path:
        sys.path.insert(0, str(python_root))
import threading
import time
from typing import TYPE_CHECKING, Any, Callable, Literal, Mapping, Sequence, TextIO

import gfe_ctypes as gfe

try:
    from ..configs.training import ExperimentRecord, ExperimentStatus, TrainingPhaseConfig
    from ..ffi import load_library
    from ..schemas import AbersoeRunResult
except ImportError:  # pragma: no cover - direct script execution fallback
    from d2c.configs.training import ExperimentRecord, ExperimentStatus, TrainingPhaseConfig
    from d2c.ffi import load_library
    from d2c.schemas import AbersoeRunResult

if TYPE_CHECKING:
    try:
        from ..runtime.director import AberconicsDirector, DirectorState
    except ImportError:  # pragma: no cover - direct script execution fallback
        from d2c.runtime.director import AberconicsDirector, DirectorState

SolverHint = Literal["auto", "stiff", "explicit"]


def _ensure_descending(values: Sequence[float], *, name: str) -> None:
    if any(values[i] <= values[i + 1] for i in range(len(values) - 1)):
        raise ValueError(f"{name} must be strictly descending")


def _coerce_float_list(values: Sequence[float], *, name: str) -> list[float]:
    out = [float(value) for value in values]
    if not out:
        raise ValueError(f"{name} must not be empty")
    return out


@dataclass(slots=True)
class LorenzMemoryParameters:
    """Lorenz constants plus memory-channel decay and feedback parameters."""

    sigma: float = 10.0
    beta: float = 8.0 / 3.0
    rho0: float = 28.0
    gamma: list[float] = field(default_factory=lambda: [0.2, 0.05, 0.01])
    weights: list[float] = field(default_factory=lambda: [0.5, 0.2, 0.1])

    def __post_init__(self) -> None:
        self.gamma = _coerce_float_list(self.gamma, name="gamma")
        self.weights = _coerce_float_list(self.weights, name="weights")
        if len(self.gamma) != len(self.weights):
            raise ValueError("gamma and weights must have the same length")
        if any(value <= 0.0 for value in self.gamma):
            raise ValueError("gamma values must be positive")
        if any(value < 0.0 for value in self.weights):
            raise ValueError("weights must be non-negative")
        _ensure_descending(self.gamma, name="gamma")

    @property
    def memory_channels(self) -> int:
        return len(self.gamma)

    @property
    def state_dim(self) -> int:
        return 3 + self.memory_channels

    def zero_memory_copy(self) -> "LorenzMemoryParameters":
        return LorenzMemoryParameters(
            sigma=self.sigma,
            beta=self.beta,
            rho0=self.rho0,
            gamma=list(self.gamma),
            weights=[0.0 for _ in self.weights],
        )

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "LorenzMemoryParameters":
        return cls(
            sigma=float(data.get("sigma", 10.0)),
            beta=float(data.get("beta", 8.0 / 3.0)),
            rho0=float(data.get("rho0", 28.0)),
            gamma=[float(value) for value in data.get("gamma", [0.2, 0.05, 0.01])],
            weights=[float(value) for value in data.get("weights", [0.5, 0.2, 0.1])],
        )


@dataclass(slots=True)
class LorenzRuntimeConfig:
    """Continuous-time solve settings corresponding to the Julia helper defaults."""

    tspan: tuple[float, float] = (0.0, 200.0)
    saveat: float = 0.1
    solver_hint: SolverHint = "auto"
    reltol: float = 1e-6
    abstol: float = 1e-8
    maxiters: int = 800_000

    def __post_init__(self) -> None:
        if len(self.tspan) != 2 or self.tspan[1] <= self.tspan[0]:
            raise ValueError("tspan must contain (start, end) with end > start")
        if self.saveat <= 0.0:
            raise ValueError("saveat must be positive")

    def to_mapping(self) -> dict[str, object]:
        return {
            "tspan": [float(self.tspan[0]), float(self.tspan[1])],
            "saveat": self.saveat,
            "solver_hint": self.solver_hint,
            "reltol": self.reltol,
            "abstol": self.abstol,
            "maxiters": self.maxiters,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "LorenzRuntimeConfig":
        raw_tspan = data.get("tspan", (0.0, 200.0))
        return cls(
            tspan=(float(raw_tspan[0]), float(raw_tspan[1])),
            saveat=float(data.get("saveat", 0.1)),
            solver_hint=str(data.get("solver_hint", "auto")),
            reltol=float(data.get("reltol", 1e-6)),
            abstol=float(data.get("abstol", 1e-8)),
            maxiters=int(data.get("maxiters", 800_000)),
        )


@dataclass(slots=True)
class LorenzLyapunovConfig:
    """Lyapunov-estimation settings following the Julia reference experiment."""

    transient: float = 50.0
    dt_renorm: float = 1.0
    seed: int = 1234
    solver_hint: SolverHint = "stiff"

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "LorenzLyapunovConfig":
        return cls(
            transient=float(data.get("transient", 50.0)),
            dt_renorm=float(data.get("dt_renorm", 1.0)),
            seed=int(data.get("seed", 1234)),
            solver_hint=str(data.get("solver_hint", "stiff")),
        )


@dataclass(slots=True)
class LorenzOptimizationStage:
    """One optimization stage in the two-pass Julia Lorenz workflow."""

    name: str
    tspan: tuple[float, float]
    transient: float
    maxiters: int
    training_initial_condition_indices: list[int]
    status: ExperimentStatus = "draft"
    solver_hint: SolverHint = "stiff"
    notes: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    def to_mapping(self) -> dict[str, object]:
        return {
            "name": self.name,
            "tspan": [float(self.tspan[0]), float(self.tspan[1])],
            "transient": self.transient,
            "maxiters": self.maxiters,
            "training_initial_condition_indices": list(self.training_initial_condition_indices),
            "status": self.status,
            "solver_hint": self.solver_hint,
            "notes": list(self.notes),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "LorenzOptimizationStage":
        raw_tspan = data["tspan"]
        return cls(
            name=str(data["name"]),
            tspan=(float(raw_tspan[0]), float(raw_tspan[1])),
            transient=float(data["transient"]),
            maxiters=int(data["maxiters"]),
            training_initial_condition_indices=[int(value) for value in data["training_initial_condition_indices"]],
            status=str(data.get("status", "draft")),
            solver_hint=str(data.get("solver_hint", "stiff")),
            notes=[str(value) for value in data.get("notes", [])],
            metadata=dict(data.get("metadata", {})),
        )

    def to_training_phase(self) -> TrainingPhaseConfig:
        return TrainingPhaseConfig(
            name=self.name,
            status=self.status,
            max_epochs=self.maxiters,
            eval_stride=1,
            early_stop_on_instability=True,
            notes=list(self.notes),
            metadata={
                "tspan": [self.tspan[0], self.tspan[1]],
                "transient": self.transient,
                "solver_hint": self.solver_hint,
                "training_initial_condition_indices": list(self.training_initial_condition_indices),
                **self.metadata,
            },
        )


@dataclass(slots=True)
class LorenzExperimentScaffold:
    """Reusable Python-side scaffold for the Julia Lorenz63 memory experiment."""

    name: str
    status: ExperimentStatus
    memory: LorenzMemoryParameters
    runtime: LorenzRuntimeConfig
    lyapunov: LorenzLyapunovConfig
    training_initial_conditions: list[list[float]]
    validation_initial_conditions: list[list[float]]
    stages: list[LorenzOptimizationStage]
    evaluation_limit: int = 200
    quick_mode: bool = True
    validated_train_lyapunov: float | None = None
    validated_periodic_initial_conditions: int | None = None
    notes: list[str] = field(default_factory=list)
    next_actions: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        expected_dim = self.memory.state_dim
        for collection_name, states in (
            ("training_initial_conditions", self.training_initial_conditions),
            ("validation_initial_conditions", self.validation_initial_conditions),
        ):
            if not states:
                raise ValueError(f"{collection_name} must not be empty")
            for state in states:
                if len(state) != expected_dim:
                    raise ValueError(f"{collection_name} states must have length {expected_dim}")

    def stage_initial_conditions(self, stage: LorenzOptimizationStage) -> list[list[float]]:
        return [self.training_initial_conditions[index] for index in stage.training_initial_condition_indices]

    def to_mapping(self) -> dict[str, object]:
        return {
            "name": self.name,
            "status": self.status,
            "memory": self.memory.to_mapping(),
            "runtime": self.runtime.to_mapping(),
            "lyapunov": self.lyapunov.to_mapping(),
            "training_initial_conditions": [list(state) for state in self.training_initial_conditions],
            "validation_initial_conditions": [list(state) for state in self.validation_initial_conditions],
            "stages": [stage.to_mapping() for stage in self.stages],
            "evaluation_limit": self.evaluation_limit,
            "quick_mode": self.quick_mode,
            "validated_train_lyapunov": self.validated_train_lyapunov,
            "validated_periodic_initial_conditions": self.validated_periodic_initial_conditions,
            "notes": list(self.notes),
            "next_actions": list(self.next_actions),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "LorenzExperimentScaffold":
        return cls(
            name=str(data["name"]),
            status=str(data.get("status", "draft")),
            memory=LorenzMemoryParameters.from_mapping(data["memory"]),
            runtime=LorenzRuntimeConfig.from_mapping(data.get("runtime", {})),
            lyapunov=LorenzLyapunovConfig.from_mapping(data.get("lyapunov", {})),
            training_initial_conditions=[
                [float(value) for value in state] for state in data["training_initial_conditions"]
            ],
            validation_initial_conditions=[
                [float(value) for value in state] for state in data["validation_initial_conditions"]
            ],
            stages=[LorenzOptimizationStage.from_mapping(stage) for stage in data["stages"]],
            evaluation_limit=int(data.get("evaluation_limit", 200)),
            quick_mode=bool(data.get("quick_mode", True)),
            validated_train_lyapunov=(
                None if data.get("validated_train_lyapunov") is None else float(data["validated_train_lyapunov"])
            ),
            validated_periodic_initial_conditions=(
                None
                if data.get("validated_periodic_initial_conditions") is None
                else int(data["validated_periodic_initial_conditions"])
            ),
            notes=[str(value) for value in data.get("notes", [])],
            next_actions=[str(value) for value in data.get("next_actions", [])],
            metadata=dict(data.get("metadata", {})),
        )

    def to_experiment_record(self) -> ExperimentRecord:
        return ExperimentRecord(
            name=self.name,
            status=self.status,
            hypothesis=(
                "Memory-only optimization can reduce the largest Lyapunov exponent "
                "and stabilize some Lorenz63 initial conditions without changing the base physics."
            ),
            phases=[stage.to_training_phase() for stage in self.stages],
            notes=list(self.notes),
            next_actions=list(self.next_actions),
            metadata={
                "memory": self.memory.to_mapping(),
                "runtime": self.runtime.to_mapping(),
                "lyapunov": self.lyapunov.to_mapping(),
                "training_initial_conditions": [list(state) for state in self.training_initial_conditions],
                "validation_initial_conditions": [list(state) for state in self.validation_initial_conditions],
                "evaluation_limit": self.evaluation_limit,
                "quick_mode": self.quick_mode,
                "validated_train_lyapunov": self.validated_train_lyapunov,
                "validated_periodic_initial_conditions": self.validated_periodic_initial_conditions,
                **self.metadata,
            },
        )


@dataclass(slots=True)
class LorenzCppBaselineConfig:
    """Mapping from the Julia-style Lorenz runtime settings to the current C++ ABI."""

    dt: float = 0.01
    form: int = gfe.GFE_C_COUPLING_FORM_B
    steps: int = 20_000
    sample_stride: int = 10
    strict_finite: bool = True
    notes: list[str] = field(default_factory=list)

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_runtime(
        cls,
        runtime: LorenzRuntimeConfig,
        *,
        dt: float = 0.01,
        form: int = gfe.GFE_C_COUPLING_FORM_B,
        strict_finite: bool = True,
    ) -> "LorenzCppBaselineConfig":
        horizon = runtime.tspan[1] - runtime.tspan[0]
        steps = int(round(horizon / dt))
        sample_stride = max(1, int(round(runtime.saveat / dt)))
        return cls(
            dt=dt,
            form=form,
            steps=steps,
            sample_stride=sample_stride,
            strict_finite=strict_finite,
            notes=[
                "Uses the built-in C++ abersoe Lorenz63 scenario through the C ABI.",
                "Uses the override-capable ABI path so Python can set initial state and active kernel per run.",
            ],
        )


@dataclass(slots=True)
class PythonLorenzBaselineConfig:
    """Pure Python Lorenz63 baseline configuration without explicit memory."""

    sigma: float = 10.0
    beta: float = 8.0 / 3.0
    rho: float = 28.0
    dt: float = 0.01
    tspan: tuple[float, float] = (0.0, 200.0)
    sample_stride: int = 10
    integration_method: str = "rk4"
    strict_finite: bool = True
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if len(self.tspan) != 2 or self.tspan[1] <= self.tspan[0]:
            raise ValueError("tspan must contain (start, end) with end > start")
        if self.sample_stride <= 0:
            raise ValueError("sample_stride must be positive")
        if self.integration_method not in {"euler", "rk4"}:
            raise ValueError("integration_method must be 'euler' or 'rk4'")

    @property
    def steps(self) -> int:
        return int(round((self.tspan[1] - self.tspan[0]) / self.dt))

    def to_mapping(self) -> dict[str, object]:
        return {
            "sigma": self.sigma,
            "beta": self.beta,
            "rho": self.rho,
            "dt": self.dt,
            "tspan": [self.tspan[0], self.tspan[1]],
            "sample_stride": self.sample_stride,
            "integration_method": self.integration_method,
            "strict_finite": self.strict_finite,
            "notes": list(self.notes),
        }

    @classmethod
    def from_runtime(
        cls,
        runtime: LorenzRuntimeConfig,
        *,
        sigma: float = 10.0,
        beta: float = 8.0 / 3.0,
        rho: float = 28.0,
        dt: float = 0.01,
        integration_method: str = "rk4",
        strict_finite: bool = True,
    ) -> "PythonLorenzBaselineConfig":
        return cls(
            sigma=sigma,
            beta=beta,
            rho=rho,
            dt=dt,
            tspan=runtime.tspan,
            sample_stride=max(1, int(round(runtime.saveat / dt))),
            integration_method=integration_method,
            strict_finite=strict_finite,
            notes=[
                "Pure Python Lorenz63 reference baseline matching the original non-memory system.",
                "Uses a local deterministic integrator so the control condition stays separate from the C++ memory runtime.",
            ],
        )


@dataclass(slots=True)
class PythonLorenzBaselineDiagnostics:
    """Summary diagnostics for the pure Python Lorenz63 baseline run."""

    steps_executed: int
    all_finite: bool
    final_u_l2: float
    mean_abs_u: float
    max_abs_u: float

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class PythonLorenzBaselineRun:
    """Stored result for a pure Python Lorenz63 baseline trajectory."""

    config: PythonLorenzBaselineConfig
    initial_state: list[float]
    final_state: list[float]
    diagnostics: PythonLorenzBaselineDiagnostics
    sample_t: list[float]
    sample_u: list[list[float]]

    def to_mapping(self) -> dict[str, object]:
        return {
            "config": self.config.to_mapping(),
            "initial_state": list(self.initial_state),
            "final_state": list(self.final_state),
            "diagnostics": self.diagnostics.to_mapping(),
            "sample_t": list(self.sample_t),
            "sample_u": [list(state) for state in self.sample_u],
        }


@dataclass(slots=True)
class PythonLorenzBaselineScaffold:
    """Reference baseline runner scaffold for Lorenz63 without memory."""

    config: PythonLorenzBaselineConfig
    initial_conditions: list[list[float]]
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.initial_conditions:
            raise ValueError("initial_conditions must not be empty")
        for state in self.initial_conditions:
            if len(state) != 3:
                raise ValueError("baseline initial conditions must have length 3")

    def run_default(self) -> PythonLorenzBaselineRun:
        return run_python_lorenz63_baseline(self.config, initial_state=self.initial_conditions[0])

    def to_mapping(self) -> dict[str, object]:
        return {
            "config": self.config.to_mapping(),
            "initial_conditions": [list(state) for state in self.initial_conditions],
            "notes": list(self.notes),
        }


@dataclass(slots=True)
class LorenzBaselineMemoryComparison:
    """Pair a pure Python baseline run with a C++ memory-backed Lorenz63 run."""

    baseline: PythonLorenzBaselineRun
    memory: AbersoeRunResult
    notes: list[str] = field(default_factory=list)
    memory_trace_t: list[float] = field(default_factory=list, repr=False)
    memory_trace_u: list[list[float]] = field(default_factory=list, repr=False)
    memory_trace_chi: list[list[float]] = field(default_factory=list, repr=False)
    memory_kernel_gamma: list[float] = field(default_factory=list, repr=False)
    memory_kernel_weights: list[float] = field(default_factory=list, repr=False)

    def summary(self) -> dict[str, object]:
        baseline_final_x = self.baseline.final_state[0]
        memory_final_x = self.memory.final_state.u[0] if self.memory.final_state.u else float("nan")
        contribution_stats = _channel_contribution_stats(
            self.memory_trace_chi,
            self.memory_kernel_weights,
        )
        return {
            "baseline": {
                "steps_executed": self.baseline.diagnostics.steps_executed,
                "all_finite": self.baseline.diagnostics.all_finite,
                "final_time": self.baseline.config.tspan[1],
                "final_x": baseline_final_x,
                "final_u_l2": self.baseline.diagnostics.final_u_l2,
                "sample_count": len(self.baseline.sample_t),
                "xyz_ranges": _component_range_mapping(self.baseline.sample_u),
            },
            "memory": {
                "steps_executed": self.memory.diagnostics.steps_executed,
                "all_finite": self.memory.diagnostics.all_finite,
                "final_time": self.memory.final_state.t,
                "final_x": memory_final_x,
                "final_u_l2": self.memory.diagnostics.final_u_l2,
                "sample_count": len(self.memory.samples.t),
                "memory_channels": self.memory.config.memory_channels,
                "xyz_ranges": _component_range_mapping(self.memory_trace_u),
                "kernel_gamma": list(self.memory_kernel_gamma),
                "kernel_weights": list(self.memory_kernel_weights),
                "channel_peak_abs_contrib": contribution_stats["peak_abs_per_channel"],
                "total_feedback_range": contribution_stats["total_feedback_range"],
            },
            "paired_metrics": {
                "same_step_count": self.baseline.diagnostics.steps_executed == self.memory.diagnostics.steps_executed,
                "same_sample_count": len(self.baseline.sample_t) == len(self.memory.samples.t),
                "baseline_has_no_memory": True,
                "memory_run_has_explicit_memory": self.memory.config.memory_channels > 0,
            },
            "notes": list(self.notes),
        }


@dataclass(slots=True)
class LorenzLyapunovEstimate:
    """Estimated or placeholder Lyapunov result for saved reports."""

    status: str
    value: float | None = None
    method: str = ""
    note: str = ""

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class SpectralDiagnosis:
    """Memory-kernel spectral diagnosis used by the framework."""

    status: str
    source: str
    gamma: list[float]
    weights: list[float]
    mcap: float | None
    mscale: float | None
    mres: float | None
    hmem: float | None
    hnorm: float | None
    deff: float | None
    note: str = ""

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class LorenzPerICSummary:
    """Per-initial-condition batch summary for later vetting."""

    ic_index: int
    initial_state: list[float]
    baseline_final_state: list[float]
    baseline_final_u_l2: float
    baseline_all_finite: bool
    baseline_lyapunov: LorenzLyapunovEstimate
    memory_status: str
    memory_final_state: list[float] | None = None
    memory_final_u_l2: float | None = None
    memory_all_finite: bool | None = None
    memory_lyapunov: LorenzLyapunovEstimate | None = None
    spectral_diagnosis: SpectralDiagnosis | None = None
    note: str = ""
    baseline_trace_t: list[float] = field(default_factory=list, repr=False)
    baseline_trace_u: list[list[float]] = field(default_factory=list, repr=False)
    memory_trace_t: list[float] = field(default_factory=list, repr=False)
    memory_trace_u: list[list[float]] = field(default_factory=list, repr=False)
    memory_trace_chi: list[list[float]] = field(default_factory=list, repr=False)
    memory_kernel_gamma: list[float] = field(default_factory=list, repr=False)
    memory_kernel_weights: list[float] = field(default_factory=list, repr=False)

    def to_mapping(self) -> dict[str, object]:
        contribution_stats = _channel_contribution_stats(
            self.memory_trace_chi,
            self.memory_kernel_weights,
        )
        return {
            "ic_index": self.ic_index,
            "initial_state": list(self.initial_state),
            "baseline_final_state": list(self.baseline_final_state),
            "baseline_final_u_l2": self.baseline_final_u_l2,
            "baseline_all_finite": self.baseline_all_finite,
            "baseline_lyapunov": self.baseline_lyapunov.to_mapping(),
            "baseline_xyz_ranges": _component_range_mapping(self.baseline_trace_u),
            "memory_status": self.memory_status,
            "memory_final_state": None if self.memory_final_state is None else list(self.memory_final_state),
            "memory_final_u_l2": self.memory_final_u_l2,
            "memory_all_finite": self.memory_all_finite,
            "memory_lyapunov": None if self.memory_lyapunov is None else self.memory_lyapunov.to_mapping(),
            "spectral_diagnosis": None if self.spectral_diagnosis is None else self.spectral_diagnosis.to_mapping(),
            "memory_xyz_ranges": _component_range_mapping(self.memory_trace_u),
            "memory_kernel_gamma": list(self.memory_kernel_gamma),
            "memory_kernel_weights": list(self.memory_kernel_weights),
            "memory_channel_peak_abs_contrib": contribution_stats["peak_abs_per_channel"],
            "memory_total_feedback_range": contribution_stats["total_feedback_range"],
            "note": self.note,
        }


@dataclass(slots=True)
class LorenzBatchComparison:
    """Batch comparison across multiple initial conditions."""

    per_ic: list[LorenzPerICSummary]
    notes: list[str] = field(default_factory=list)

    def to_mapping(self) -> dict[str, object]:
        completed_memory = sum(1 for item in self.per_ic if item.memory_status == "completed")
        return {
            "per_ic": [item.to_mapping() for item in self.per_ic],
            "completed_memory_runs": completed_memory,
            "total_initial_conditions": len(self.per_ic),
            "notes": list(self.notes),
        }


def _progress_dir_root() -> Path:
    return Path(__file__).resolve().parents[1] / "progress" / "lorenz63"


def _timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _format_elapsed(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _emit_progress(stream: TextIO | None, message: str) -> None:
    if stream is None:
        return
    stream.write(message + "\n")
    stream.flush()


def ascii_sparkline(values: Sequence[float], *, width: int = 64, charset: str = " .:-=+*#%@") -> str:
    """Render a tiny ASCII graph for terminal-side inspection."""

    if width <= 0:
        raise ValueError("width must be positive")
    if len(charset) < 2:
        raise ValueError("charset must contain at least two characters")
    if not values:
        return ""

    clean_values = [float(value) for value in values]
    if len(clean_values) > width:
        stride = len(clean_values) / width
        reduced = []
        for idx in range(width):
            start = int(idx * stride)
            end = max(start + 1, int((idx + 1) * stride))
            bucket = clean_values[start:end]
            reduced.append(sum(bucket) / len(bucket))
        clean_values = reduced

    minimum = min(clean_values)
    maximum = max(clean_values)
    if math.isclose(minimum, maximum):
        return charset[len(charset) // 2] * len(clean_values)

    out = []
    scale = len(charset) - 1
    for value in clean_values:
        normalized = (value - minimum) / (maximum - minimum)
        out.append(charset[min(scale, max(0, int(round(normalized * scale))))])
    return "".join(out)


def _ensure_matplotlib():
    mpl_dir = Path("/tmp/matplotlib-codex")
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _series_range(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    numeric = [float(value) for value in values]
    return {
        "min": min(numeric),
        "max": max(numeric),
    }


def _component_range_mapping(states: Sequence[Sequence[float]]) -> dict[str, dict[str, float]] | None:
    if not states:
        return None
    dim = len(states[0])
    labels = ["x", "y", "z"] + [f"c{i}" for i in range(max(0, dim - 3))]
    out: dict[str, dict[str, float]] = {}
    for axis in range(dim):
        series = [float(state[axis]) for state in states if len(state) > axis]
        rng = _series_range(series)
        if rng is not None:
            out[labels[axis] if axis < len(labels) else f"a{axis}"] = rng
    return out


def _channel_contributions(
    chi_states: Sequence[Sequence[float]],
    weights: Sequence[float],
) -> list[list[float]]:
    return [
        [float(weights[channel]) * float(chi[channel]) for channel in range(min(len(weights), len(chi)))]
        for chi in chi_states
    ]


def _channel_contribution_stats(
    chi_states: Sequence[Sequence[float]],
    weights: Sequence[float],
) -> dict[str, object]:
    if not chi_states or not weights:
        return {
            "peak_abs_per_channel": [],
            "total_feedback_range": None,
        }
    contributions = _channel_contributions(chi_states, weights)
    if not contributions:
        return {
            "peak_abs_per_channel": [],
            "total_feedback_range": None,
        }
    peak_abs = [
        max(abs(sample[channel]) for sample in contributions if len(sample) > channel)
        for channel in range(len(weights))
    ]
    total_feedback = [sum(sample) for sample in contributions]
    return {
        "peak_abs_per_channel": peak_abs,
        "total_feedback_range": _series_range(total_feedback),
    }


def _format_range_text(mapping: dict[str, dict[str, float]] | None) -> str:
    if not mapping:
        return "n/a"
    parts = []
    for key in ("x", "y", "z"):
        if key in mapping:
            parts.append(f"{key}=[{mapping[key]['min']:.3f},{mapping[key]['max']:.3f}]")
    return " ".join(parts) if parts else "n/a"


def _save_comparison_plots(comparison: LorenzBaselineMemoryComparison, bundle_dir: Path) -> dict[str, str]:
    plt = _ensure_matplotlib()

    baseline_t = comparison.baseline.sample_t
    baseline_u = comparison.baseline.sample_u
    memory_t = comparison.memory_trace_t or list(comparison.memory.samples.t)
    memory_u = comparison.memory_trace_u or [[value] for value in comparison.memory.samples.u0]
    memory_chi = comparison.memory_trace_chi
    channel_contrib = _channel_contributions(memory_chi, comparison.memory_kernel_weights)
    total_feedback = [sum(sample) for sample in channel_contrib] if channel_contrib else []
    memory_dim = len(memory_u[0]) if memory_u else 0

    overlay_path = bundle_dir / "xyz_comparison.png"
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for axis, label in enumerate(("x", "y", "z")):
        axes[axis].plot(
            baseline_t,
            [state[axis] for state in baseline_u],
            label=f"baseline {label}(t)",
            linewidth=1.1,
        )
        if memory_u and memory_dim > axis:
            axes[axis].plot(
                memory_t,
                [state[axis] for state in memory_u],
                label=f"memory {label}(t)",
                linewidth=1.1,
            )
        elif axis == 0:
            axes[axis].plot(
                memory_t,
                list(comparison.memory.samples.u0),
                label="memory x(t)",
                linewidth=1.1,
            )
        axes[axis].set_ylabel(label)
        axes[axis].grid(True, alpha=0.25)
        axes[axis].legend(loc="upper right")
    axes[-1].set_xlabel("time")
    fig.suptitle("Lorenz63 Baseline vs Memory Traces", y=0.995)
    fig.tight_layout()
    fig.savefig(overlay_path, dpi=160)
    plt.close(fig)

    phase_path = bundle_dir / "phase_xz.png"
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot([state[0] for state in baseline_u], [state[2] for state in baseline_u], label="baseline", linewidth=1.0)
    if memory_u and memory_dim >= 3:
        ax.plot([state[0] for state in memory_u], [state[2] for state in memory_u], label="memory", linewidth=1.0)
    else:
        ax.text(
            0.5,
            0.5,
            "memory x-z trajectory unavailable\n(full override trace not attached)",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    ax.set_title("Lorenz63 Phase Portrait (x-z)")
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(phase_path, dpi=160)
    plt.close(fig)

    aux_path = bundle_dir / "memory_contributions.png"
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for channel in range(len(comparison.memory_kernel_weights)):
        if memory_chi and len(memory_chi[0]) > channel:
            axes[0].plot(memory_t, [state[channel] for state in memory_chi], linewidth=1.0, label=f"chi{channel}")
    axes[0].set_title("Memory State Channels")
    axes[0].set_ylabel("chi")
    axes[0].grid(True, alpha=0.25)
    if len(comparison.memory_kernel_weights) > 0:
        axes[0].legend(loc="upper right", ncol=max(1, min(3, len(comparison.memory_kernel_weights))))

    for channel in range(len(comparison.memory_kernel_weights)):
        if channel_contrib and len(channel_contrib[0]) > channel:
            axes[1].plot(memory_t, [sample[channel] for sample in channel_contrib], linewidth=1.0, label=f"w{channel}*chi{channel}")
    axes[1].set_title("Per-Channel Feedback Contributions")
    axes[1].set_ylabel("w_i chi_i")
    axes[1].grid(True, alpha=0.25)
    if len(comparison.memory_kernel_weights) > 0:
        axes[1].legend(loc="upper right", ncol=max(1, min(3, len(comparison.memory_kernel_weights))))

    if total_feedback:
        axes[2].plot(memory_t, total_feedback, color="tab:red", linewidth=1.1, label="sum(w_i chi_i)")
    if memory_u and memory_dim > 0:
        axes[2].plot(memory_t, [state[0] for state in memory_u], color="tab:blue", linewidth=0.9, alpha=0.8, label="memory x")
    else:
        axes[2].plot(memory_t, list(comparison.memory.samples.u0), color="tab:blue", linewidth=0.9, alpha=0.8, label="memory x")
    axes[2].set_title("Total Feedback vs Memory x")
    axes[2].set_xlabel("time")
    axes[2].set_ylabel("feedback / x")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(aux_path, dpi=160)
    plt.close(fig)

    return {
        "xyz_comparison_png": str(overlay_path),
        "phase_xz_png": str(phase_path),
        "memory_contributions_png": str(aux_path),
    }


def _save_batch_plots(batch: LorenzBatchComparison, bundle_dir: Path) -> dict[str, str]:
    plt = _ensure_matplotlib()
    plot_paths: dict[str, str] = {}

    if batch.per_ic:
        per_ic_dir = bundle_dir / "per_ic_plots"
        per_ic_dir.mkdir(parents=True, exist_ok=True)
        for item in batch.per_ic:
            if item.baseline_trace_u and item.memory_trace_u:
                traces_path = per_ic_dir / f"ic_{item.ic_index:02d}_traces_xyz.png"
                fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
                for axis, label in enumerate(("x", "y", "z")):
                    axes[axis].plot(
                        item.baseline_trace_t,
                        [state[axis] for state in item.baseline_trace_u],
                        linewidth=1.0,
                        label=f"baseline {label}",
                    )
                    axes[axis].plot(
                        item.memory_trace_t,
                        [state[axis] for state in item.memory_trace_u],
                        linewidth=1.0,
                        label=f"memory {label}",
                    )
                    axes[axis].set_ylabel(label)
                    axes[axis].grid(True, alpha=0.25)
                    axes[axis].legend(loc="upper right")
                axes[-1].set_xlabel("time")
                fig.suptitle(f"IC {item.ic_index} Baseline vs Memory Traces", y=0.995)
                fig.tight_layout()
                fig.savefig(traces_path, dpi=160)
                plt.close(fig)
                plot_paths[f"ic_{item.ic_index:02d}_traces_xyz_png"] = str(traces_path)

                phase_path = per_ic_dir / f"ic_{item.ic_index:02d}_phase_xz.png"
                fig, ax = plt.subplots(figsize=(6.5, 5.5))
                ax.plot(
                    [state[0] for state in item.baseline_trace_u],
                    [state[2] for state in item.baseline_trace_u],
                    linewidth=1.0,
                    label="baseline",
                )
                ax.plot(
                    [state[0] for state in item.memory_trace_u],
                    [state[2] for state in item.memory_trace_u],
                    linewidth=1.0,
                    label="memory",
                )
                ax.set_title(f"IC {item.ic_index} Phase Portrait (x-z)")
                ax.set_xlabel("x")
                ax.set_ylabel("z")
                ax.grid(True, alpha=0.25)
                ax.legend()
                fig.tight_layout()
                fig.savefig(phase_path, dpi=160)
                plt.close(fig)
                plot_paths[f"ic_{item.ic_index:02d}_phase_xz_png"] = str(phase_path)

            if item.memory_trace_t and item.memory_trace_chi and item.memory_kernel_weights:
                contrib = _channel_contributions(item.memory_trace_chi, item.memory_kernel_weights)
                total_feedback = [sum(sample) for sample in contrib] if contrib else []
                contrib_path = per_ic_dir / f"ic_{item.ic_index:02d}_memory_contributions.png"
                fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
                for channel in range(len(item.memory_kernel_weights)):
                    axes[0].plot(
                        item.memory_trace_t,
                        [state[channel] for state in item.memory_trace_chi],
                        linewidth=1.0,
                        label=f"chi{channel}",
                    )
                axes[0].set_title(f"IC {item.ic_index} Memory States")
                axes[0].set_ylabel("chi")
                axes[0].grid(True, alpha=0.25)
                axes[0].legend(loc="upper right", ncol=max(1, min(3, len(item.memory_kernel_weights))))

                for channel in range(len(item.memory_kernel_weights)):
                    axes[1].plot(
                        item.memory_trace_t,
                        [sample[channel] for sample in contrib],
                        linewidth=1.0,
                        label=f"w{channel}*chi{channel}",
                    )
                axes[1].set_title(f"IC {item.ic_index} Per-Channel Feedback")
                axes[1].set_ylabel("w_i chi_i")
                axes[1].grid(True, alpha=0.25)
                axes[1].legend(loc="upper right", ncol=max(1, min(3, len(item.memory_kernel_weights))))

                axes[2].plot(item.memory_trace_t, total_feedback, color="tab:red", linewidth=1.1, label="sum(w_i chi_i)")
                axes[2].plot(
                    item.memory_trace_t,
                    [state[0] for state in item.memory_trace_u],
                    color="tab:blue",
                    linewidth=0.9,
                    alpha=0.8,
                    label="memory x",
                )
                axes[2].set_title(f"IC {item.ic_index} Total Feedback vs x")
                axes[2].set_xlabel("time")
                axes[2].set_ylabel("feedback / x")
                axes[2].grid(True, alpha=0.25)
                axes[2].legend(loc="upper right")
                fig.tight_layout()
                fig.savefig(contrib_path, dpi=160)
                plt.close(fig)
                plot_paths[f"ic_{item.ic_index:02d}_memory_contributions_png"] = str(contrib_path)

            if item.spectral_diagnosis is not None and item.spectral_diagnosis.status == "computed":
                diag = item.spectral_diagnosis
                spectral_path = per_ic_dir / f"ic_{item.ic_index:02d}_spectral.png"
                fig, axes = plt.subplots(2, 1, figsize=(8, 6))
                axes[0].bar(range(len(diag.gamma)), diag.weights, color="tab:blue")
                axes[0].set_title(f"IC {item.ic_index} Memory Weights")
                axes[0].set_ylabel("weight")
                axes[1].bar(range(len(diag.gamma)), diag.gamma, color="tab:green")
                axes[1].set_title(f"IC {item.ic_index} Memory Gamma")
                axes[1].set_ylabel("gamma")
                axes[1].set_xlabel("channel")
                for ax in axes:
                    ax.grid(True, alpha=0.2)
                fig.tight_layout()
                fig.savefig(spectral_path, dpi=160)
                plt.close(fig)
                plot_paths[f"ic_{item.ic_index:02d}_spectral_png"] = str(spectral_path)

        summary_path = bundle_dir / "baseline_final_x_by_ic.png"
        fig, ax = plt.subplots(figsize=(8, 4.5))
        xs = [item.ic_index for item in batch.per_ic]
        ys = [item.baseline_final_state[0] for item in batch.per_ic]
        ax.bar(xs, ys, color="tab:purple")
        ax.set_title("Baseline Final x by Initial Condition")
        ax.set_xlabel("IC index")
        ax.set_ylabel("final x")
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(summary_path, dpi=160)
        plt.close(fig)
        plot_paths["baseline_final_x_by_ic_png"] = str(summary_path)

        memory_summary_path = bundle_dir / "memory_final_x_by_ic.png"
        fig, ax = plt.subplots(figsize=(8, 4.5))
        memory_ys = [
            float(item.memory_final_state[0]) if item.memory_final_state is not None else float("nan")
            for item in batch.per_ic
        ]
        ax.bar(xs, memory_ys, color="tab:orange")
        ax.set_title("Memory Final x by Initial Condition")
        ax.set_xlabel("IC index")
        ax.set_ylabel("final x")
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(memory_summary_path, dpi=160)
        plt.close(fig)
        plot_paths["memory_final_x_by_ic_png"] = str(memory_summary_path)

    return plot_paths


def _baseline_x_series(run: PythonLorenzBaselineRun) -> list[float]:
    return [state[0] for state in run.sample_u]


def _memory_x_series(run: AbersoeRunResult) -> list[float]:
    return list(run.samples.u0)


def default_cpp_lorenz63_kernel_metadata() -> tuple[list[float], list[float]]:
    """Current built-in C++ Lorenz63 kernel defaults mirrored from the core scenario source."""

    return [1.2, 0.4, 0.1], [0.5, 0.3, 0.2]


def spectral_diagnosis_from_kernel(
    gamma: Sequence[float],
    weights: Sequence[float],
    *,
    source: str,
    note: str = "",
) -> SpectralDiagnosis:
    """Compute framework-style spectral memory diagnostics for a kernel."""

    gamma_values = [float(value) for value in gamma]
    weight_values = [float(value) for value in weights]
    if len(gamma_values) != len(weight_values) or not gamma_values:
        raise ValueError("gamma and weights must be non-empty and the same length")
    tau = [1.0 / value for value in gamma_values]
    wpos = [max(value, 0.0) for value in weight_values]
    wsum = sum(wpos)
    if wsum <= 0.0:
        return SpectralDiagnosis(
            status="invalid",
            source=source,
            gamma=gamma_values,
            weights=weight_values,
            mcap=None,
            mscale=None,
            mres=None,
            hmem=None,
            hnorm=None,
            deff=None,
            note="Kernel weights were non-positive after clipping; unable to compute spectral units.",
        )
    wtil = [value / wsum for value in wpos]
    mcap_num = sum(w * (t ** 2) for w, t in zip(wpos, tau, strict=True))
    mcap_den = sum(w * t for w, t in zip(wpos, tau, strict=True))
    mcap = mcap_num / mcap_den if mcap_den > 0.0 else None
    tau_min = min(tau)
    tau_max = max(tau)
    mscale = math.log10((tau_max / tau_min) + 1e-30) if tau_min > 0.0 else None
    hmem = -sum(value * math.log(value + 1e-20) for value in wtil)
    hnorm = hmem / math.log(len(wtil)) if len(wtil) > 1 else 1.0
    deff = len(wtil) * math.exp(hmem)
    mres = (len(wtil) / mscale) if mscale and not math.isclose(mscale, 0.0) else None
    return SpectralDiagnosis(
        status="computed",
        source=source,
        gamma=gamma_values,
        weights=weight_values,
        mcap=mcap,
        mscale=mscale,
        mres=mres,
        hmem=hmem,
        hnorm=hnorm,
        deff=deff,
        note=note,
    )


def format_lorenz_comparison_report(
    comparison: LorenzBaselineMemoryComparison,
    *,
    plot_paths: Mapping[str, str] | None = None,
) -> str:
    """Create structured terminal-friendly output for the paired Lorenz comparison."""

    summary = comparison.summary()
    lines = [
        "Lorenz63 Baseline vs Memory",
        "===========================",
        "",
        "Runs",
        f"  baseline steps={summary['baseline']['steps_executed']} finite={summary['baseline']['all_finite']} final_x={summary['baseline']['final_x']:.6f} final_u_l2={summary['baseline']['final_u_l2']:.6f}",
        f"  memory   steps={summary['memory']['steps_executed']} finite={summary['memory']['all_finite']} final_x={summary['memory']['final_x']:.6f} final_u_l2={summary['memory']['final_u_l2']:.6f} channels={summary['memory']['memory_channels']}",
        "",
        "Trace Ranges",
        f"  baseline {_format_range_text(summary['baseline']['xyz_ranges'])}",
        f"  memory   {_format_range_text(summary['memory']['xyz_ranges'])}",
        "",
        "Kernel",
        f"  gamma={summary['memory']['kernel_gamma']}",
        f"  weights={summary['memory']['kernel_weights']}",
        f"  peak_abs_channel_contrib={summary['memory']['channel_peak_abs_contrib']}",
        f"  total_feedback_range={summary['memory']['total_feedback_range']}",
        "",
        "Pairing",
        f"  same_step_count={summary['paired_metrics']['same_step_count']}",
        f"  same_sample_count={summary['paired_metrics']['same_sample_count']}",
        f"  baseline_has_no_memory={summary['paired_metrics']['baseline_has_no_memory']}",
        f"  memory_run_has_explicit_memory={summary['paired_metrics']['memory_run_has_explicit_memory']}",
    ]
    if plot_paths:
        lines.extend(["", "Plots"])
        for key, value in plot_paths.items():
            lines.append(f"  {key}={value}")
    if comparison.notes:
        lines.extend(["", "Notes"])
        lines.extend([f"  - {note}" for note in comparison.notes])
    return "\n".join(lines)


def format_lorenz_batch_report(
    batch: LorenzBatchComparison,
    *,
    plot_paths: Mapping[str, str] | None = None,
) -> str:
    """Create a terminal-friendly batch report over all initial conditions."""

    lines = [
        "Lorenz63 Batch Summary",
        "======================",
        "",
    ]
    for item in batch.per_ic:
        mapping = item.to_mapping()
        lines.append(
            f"IC {item.ic_index}: baseline finite={item.baseline_all_finite} "
            f"baseline_final_x={item.baseline_final_state[0]:.6f} "
            f"baseline_lyap={item.baseline_lyapunov.value if item.baseline_lyapunov.value is not None else 'n/a'} "
            f"memory_status={item.memory_status}"
        )
        lines.append(f"  baseline ranges {_format_range_text(mapping['baseline_xyz_ranges'])}")
        if item.memory_status == "completed":
            lines.append(
                f"  memory finite={item.memory_all_finite} memory_final_x={item.memory_final_state[0]:.6f} "
                f"memory_lyap={item.memory_lyapunov.value if item.memory_lyapunov and item.memory_lyapunov.value is not None else 'n/a'}"
            )
            lines.append(f"  memory ranges {_format_range_text(mapping['memory_xyz_ranges'])}")
            lines.append(
                f"  kernel gamma={mapping['memory_kernel_gamma']} weights={mapping['memory_kernel_weights']}"
            )
            lines.append(
                f"  channel_peak_abs_contrib={mapping['memory_channel_peak_abs_contrib']} "
                f"total_feedback_range={mapping['memory_total_feedback_range']}"
            )
        if item.spectral_diagnosis is not None and item.spectral_diagnosis.status == "computed":
            lines.append(
                f"  spectral Mcap={item.spectral_diagnosis.mcap:.6f} "
                f"Mscale={item.spectral_diagnosis.mscale:.6f} "
                f"Deff={item.spectral_diagnosis.deff:.6f}"
            )
        if item.note:
            lines.append(f"  note: {item.note}")
    if batch.notes:
        lines.extend(["", "Notes"])
        lines.extend([f"  - {note}" for note in batch.notes])
    if plot_paths:
        lines.extend(["", "Plots"])
        for key, value in plot_paths.items():
            lines.append(f"  {key}={value}")
    return "\n".join(lines)


def _write_series_csv(path: Path, header: str, rows: Sequence[Sequence[Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(header + "\n")
        for row in rows:
            handle.write(",".join(str(value) for value in row) + "\n")


def save_lorenz_comparison_bundle(
    comparison: LorenzBaselineMemoryComparison,
    *,
    output_dir: str | Path | None = None,
    bundle_name: str | None = None,
    report_text: str | None = None,
) -> dict[str, str]:
    """Persist terminal report plus series snapshots under the progress folder."""

    bundle_root = Path(output_dir) if output_dir is not None else _progress_dir_root()
    bundle_root.mkdir(parents=True, exist_ok=True)
    bundle_dir = bundle_root / (bundle_name or f"run_{_timestamp_slug()}")
    bundle_dir.mkdir(parents=True, exist_ok=True)

    summary = comparison.summary()
    plot_paths = _save_comparison_plots(comparison, bundle_dir)
    report = report_text or format_lorenz_comparison_report(comparison, plot_paths=plot_paths)

    report_path = bundle_dir / "terminal_report.txt"
    report_path.write_text(report + "\n", encoding="utf-8")

    summary_path = bundle_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    baseline_csv = bundle_dir / "baseline_x.csv"
    _write_series_csv(
        baseline_csv,
        "t,x,y,z",
        (
            (comparison.baseline.sample_t[idx], *comparison.baseline.sample_u[idx])
            for idx in range(len(comparison.baseline.sample_t))
        ),
    )

    memory_csv = bundle_dir / "memory_samples.csv"
    _write_series_csv(
        memory_csv,
        "t,u0,chi0,u_l2,chi_l2",
        (
            (
                comparison.memory.samples.t[idx],
                comparison.memory.samples.u0[idx],
                comparison.memory.samples.chi0[idx],
                comparison.memory.samples.u_l2[idx],
                comparison.memory.samples.chi_l2[idx],
            )
            for idx in range(len(comparison.memory.samples.t))
        ),
    )

    if comparison.memory_trace_t and comparison.memory_trace_u:
        memory_trace_csv = bundle_dir / "memory_trace_xyz.csv"
        _write_series_csv(
            memory_trace_csv,
            "t,x,y,z",
            (
                (comparison.memory_trace_t[idx], *comparison.memory_trace_u[idx][:3])
                for idx in range(len(comparison.memory_trace_t))
            ),
        )
    else:
        memory_trace_csv = None

    if comparison.memory_trace_t and comparison.memory_trace_chi and comparison.memory_kernel_weights:
        contribution_rows = _channel_contributions(comparison.memory_trace_chi, comparison.memory_kernel_weights)
        contribution_csv = bundle_dir / "memory_contributions.csv"
        header = ["t"]
        header.extend(f"chi{i}" for i in range(len(comparison.memory_kernel_weights)))
        header.extend(f"w{i}_chi{i}" for i in range(len(comparison.memory_kernel_weights)))
        header.append("total_feedback")
        _write_series_csv(
            contribution_csv,
            ",".join(header),
            (
                (
                    comparison.memory_trace_t[idx],
                    *comparison.memory_trace_chi[idx][: len(comparison.memory_kernel_weights)],
                    *contribution_rows[idx][: len(comparison.memory_kernel_weights)],
                    sum(contribution_rows[idx]),
                )
                for idx in range(len(comparison.memory_trace_t))
            ),
        )
    else:
        contribution_csv = None

    return {
        "bundle_dir": str(bundle_dir),
        "report_txt": str(report_path),
        "summary_json": str(summary_path),
        "baseline_csv": str(baseline_csv),
        "memory_csv": str(memory_csv),
        **({"memory_trace_csv": str(memory_trace_csv)} if memory_trace_csv is not None else {}),
        **({"memory_contributions_csv": str(contribution_csv)} if contribution_csv is not None else {}),
        **plot_paths,
    }


def save_lorenz_batch_bundle(
    batch: LorenzBatchComparison,
    *,
    output_dir: str | Path | None = None,
    bundle_name: str | None = None,
    report_text: str | None = None,
) -> dict[str, str]:
    """Persist multi-IC batch summaries and reports under the progress folder."""

    bundle_root = Path(output_dir) if output_dir is not None else _progress_dir_root()
    bundle_root.mkdir(parents=True, exist_ok=True)
    bundle_dir = bundle_root / (bundle_name or f"batch_{_timestamp_slug()}")
    bundle_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = _save_batch_plots(batch, bundle_dir)
    report = report_text or format_lorenz_batch_report(batch, plot_paths=plot_paths)

    report_path = bundle_dir / "batch_report.txt"
    report_path.write_text(report + "\n", encoding="utf-8")
    summary_path = bundle_dir / "batch_summary.json"
    summary_path.write_text(json.dumps(batch.to_mapping(), indent=2, sort_keys=True), encoding="utf-8")

    per_ic_dir = bundle_dir / "per_ic"
    per_ic_dir.mkdir(parents=True, exist_ok=True)
    for item in batch.per_ic:
        (per_ic_dir / f"ic_{item.ic_index:02d}.json").write_text(
            json.dumps(item.to_mapping(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if item.baseline_trace_t and item.baseline_trace_u:
            _write_series_csv(
                per_ic_dir / f"ic_{item.ic_index:02d}_baseline_trace.csv",
                "t,x,y,z",
                (
                    (item.baseline_trace_t[idx], *item.baseline_trace_u[idx][:3])
                    for idx in range(len(item.baseline_trace_t))
                ),
            )
        if item.memory_trace_t and item.memory_trace_u:
            _write_series_csv(
                per_ic_dir / f"ic_{item.ic_index:02d}_memory_trace.csv",
                "t,x,y,z",
                (
                    (item.memory_trace_t[idx], *item.memory_trace_u[idx][:3])
                    for idx in range(len(item.memory_trace_t))
                ),
            )
        if item.memory_trace_t and item.memory_trace_chi and item.memory_kernel_weights:
            contribution_rows = _channel_contributions(item.memory_trace_chi, item.memory_kernel_weights)
            header = ["t"]
            header.extend(f"chi{i}" for i in range(len(item.memory_kernel_weights)))
            header.extend(f"w{i}_chi{i}" for i in range(len(item.memory_kernel_weights)))
            header.append("total_feedback")
            _write_series_csv(
                per_ic_dir / f"ic_{item.ic_index:02d}_memory_contributions.csv",
                ",".join(header),
                (
                    (
                        item.memory_trace_t[idx],
                        *item.memory_trace_chi[idx][: len(item.memory_kernel_weights)],
                        *contribution_rows[idx][: len(item.memory_kernel_weights)],
                        sum(contribution_rows[idx]),
                    )
                    for idx in range(len(item.memory_trace_t))
                ),
            )

    return {
        "bundle_dir": str(bundle_dir),
        "report_txt": str(report_path),
        "summary_json": str(summary_path),
        "per_ic_dir": str(per_ic_dir),
        **plot_paths,
    }


def _run_with_ticking_elapsed(
    label: str,
    fn: Callable[[], Any],
    *,
    stream: TextIO | None = None,
    tick_seconds: float = 1.0,
) -> Any:
    result_box: dict[str, Any] = {}
    error_box: dict[str, BaseException] = {}

    def _worker() -> None:
        try:
            result_box["value"] = fn()
        except BaseException as exc:  # pragma: no cover - exercised through caller behavior
            error_box["error"] = exc

    thread = threading.Thread(target=_worker, daemon=True)
    start = time.perf_counter()
    _emit_progress(stream, f"[run] {label} started")
    thread.start()
    while thread.is_alive():
        elapsed = time.perf_counter() - start
        _emit_progress(stream, f"[tick] {label} elapsed={_format_elapsed(elapsed)}")
        thread.join(timeout=tick_seconds)
    elapsed = time.perf_counter() - start
    if "error" in error_box:
        raise error_box["error"]
    _emit_progress(stream, f"[done] {label} elapsed={_format_elapsed(elapsed)}")
    return result_box["value"]


def default_lorenz_memory_parameters() -> LorenzMemoryParameters:
    """Return the Julia notebook's default 3-channel memory initialization."""

    return LorenzMemoryParameters(
        sigma=10.0,
        beta=8.0 / 3.0,
        rho0=28.0,
        gamma=[0.2, 0.05, 0.01],
        weights=[0.5, 0.2, 0.1],
    )


def default_lorenz_zero_memory_parameters() -> LorenzMemoryParameters:
    """Return the fixed-physics zero-memory calibration reference used in Julia."""

    return default_lorenz_memory_parameters().zero_memory_copy()


def default_lorenz_initial_conditions(memory_channels: int) -> list[list[float]]:
    """Return the three initial conditions used in the Julia experiment."""

    zeros = [0.0] * memory_channels
    return [
        [1.0, 1.0, 1.0, *zeros],
        [2.0, 3.0, 4.0, *zeros],
        [-1.5, 1.0, 2.0, *zeros],
    ]


def default_python_lorenz_initial_conditions() -> list[list[float]]:
    """Return the same Lorenz63 initial conditions without appended memory channels."""

    return [state[:3] for state in default_lorenz_initial_conditions(0)]


def python_lorenz63_rhs(state: Sequence[float], config: PythonLorenzBaselineConfig) -> list[float]:
    """Plain Lorenz63 right-hand side used by the pure Python baseline."""

    if len(state) != 3:
        raise ValueError("baseline state must have length 3")
    x, y, z = (float(state[0]), float(state[1]), float(state[2]))
    return [
        config.sigma * (y - x),
        x * (config.rho - z) - y,
        x * y - config.beta * z,
    ]


def _vector_add_scaled(state: Sequence[float], deriv: Sequence[float], scale: float) -> list[float]:
    return [float(s) + scale * float(d) for s, d in zip(state, deriv, strict=True)]


def _rk4_step(state: Sequence[float], dt: float, rhs) -> list[float]:
    k1 = rhs(state)
    k2 = rhs(_vector_add_scaled(state, k1, 0.5 * dt))
    k3 = rhs(_vector_add_scaled(state, k2, 0.5 * dt))
    k4 = rhs(_vector_add_scaled(state, k3, dt))
    return [
        float(s) + (dt / 6.0) * (float(a) + 2.0 * float(b) + 2.0 * float(c) + float(d))
        for s, a, b, c, d in zip(state, k1, k2, k3, k4, strict=True)
    ]


def run_python_lorenz63_baseline(
    config: PythonLorenzBaselineConfig,
    *,
    initial_state: Sequence[float] | None = None,
    progress_label: str = "python_baseline",
    progress_stream: TextIO | None = None,
    progress_every_steps: int | None = None,
) -> PythonLorenzBaselineRun:
    """Run a deterministic pure Python Lorenz63 baseline trajectory."""

    state = [1.0, 1.0, 1.0] if initial_state is None else [float(value) for value in initial_state]
    if len(state) != 3:
        raise ValueError("initial_state must have length 3")

    t = config.tspan[0]
    start = time.perf_counter()
    sample_t = [t]
    sample_u = [list(state)]
    sum_abs_u = sum(abs(value) for value in state)
    max_abs_u = max(abs(value) for value in state)
    all_finite = all(math.isfinite(value) for value in state)
    step_interval = progress_every_steps if progress_every_steps is not None else max(1, config.steps // 10)
    _emit_progress(
        progress_stream,
        f"[run] {progress_label} started steps={config.steps} dt={config.dt} method={config.integration_method}",
    )

    def rhs(current: Sequence[float]) -> list[float]:
        return python_lorenz63_rhs(current, config)

    for step_idx in range(config.steps):
        if config.integration_method == "rk4":
            state = _rk4_step(state, config.dt, rhs)
        else:
            derivative = rhs(state)
            state = _vector_add_scaled(state, derivative, config.dt)
        t = config.tspan[0] + (step_idx + 1) * config.dt

        is_finite = all(math.isfinite(value) for value in state)
        all_finite = all_finite and is_finite
        if config.strict_finite and not is_finite:
            raise RuntimeError("Python Lorenz63 baseline became non-finite")

        sum_abs_u += sum(abs(value) for value in state)
        max_abs_u = max(max_abs_u, max(abs(value) for value in state))
        if (step_idx + 1) % config.sample_stride == 0:
            sample_t.append(t)
            sample_u.append(list(state))
        if progress_stream is not None and ((step_idx + 1) % step_interval == 0 or step_idx + 1 == config.steps):
            elapsed = time.perf_counter() - start
            _emit_progress(
                progress_stream,
                f"[tick] {progress_label} step={step_idx + 1}/{config.steps} t={t:.2f} elapsed={_format_elapsed(elapsed)}",
            )

    if sample_t[-1] != t:
        sample_t.append(t)
        sample_u.append(list(state))

    diagnostics = PythonLorenzBaselineDiagnostics(
        steps_executed=config.steps,
        all_finite=all_finite,
        final_u_l2=math.sqrt(sum(value * value for value in state)),
        mean_abs_u=sum_abs_u / ((config.steps + 1) * 3.0),
        max_abs_u=max_abs_u,
    )
    return PythonLorenzBaselineRun(
        config=config,
        initial_state=[float(value) for value in (initial_state if initial_state is not None else [1.0, 1.0, 1.0])],
        final_state=list(state),
        diagnostics=diagnostics,
        sample_t=sample_t,
        sample_u=sample_u,
    )


def estimate_python_lorenz63_baseline_lyapunov(
    config: PythonLorenzBaselineConfig,
    *,
    initial_state: Sequence[float],
    transient_fraction: float = 0.25,
    renorm_every_steps: int = 10,
    epsilon: float = 1e-8,
) -> LorenzLyapunovEstimate:
    """Approximate the largest Lyapunov exponent from the pure Python baseline."""

    base = [float(value) for value in initial_state]
    if len(base) != 3:
        raise ValueError("initial_state must have length 3")
    perturbed = [base[0] + epsilon, base[1], base[2]]
    def rhs(current: Sequence[float]) -> list[float]:
        return python_lorenz63_rhs(current, config)

    transient_steps = int(config.steps * transient_fraction)
    accum = 0.0
    count = 0

    for step_idx in range(config.steps):
        if config.integration_method == "rk4":
            base = _rk4_step(base, config.dt, rhs)
            perturbed = _rk4_step(perturbed, config.dt, rhs)
        else:
            base = _vector_add_scaled(base, rhs(base), config.dt)
            perturbed = _vector_add_scaled(perturbed, rhs(perturbed), config.dt)
        if (step_idx + 1) % renorm_every_steps == 0:
            delta = [p - b for p, b in zip(perturbed, base, strict=True)]
            norm = math.sqrt(sum(value * value for value in delta))
            if norm > 0.0:
                if step_idx + 1 > transient_steps:
                    accum += math.log(norm / epsilon)
                    count += 1
                scale = epsilon / norm
                perturbed = [b + scale * d for b, d in zip(base, delta, strict=True)]

    if count == 0:
        return LorenzLyapunovEstimate(
            status="placeholder",
            value=None,
            method="finite_difference_twin_trajectory",
            note="Insufficient post-transient renormalization windows for estimate.",
        )
    value = accum / (count * renorm_every_steps * config.dt)
    return LorenzLyapunovEstimate(
        status="computed",
        value=value,
        method="finite_difference_twin_trajectory",
        note="Approximate largest Lyapunov estimate from paired deterministic trajectories.",
    )


def _trajectory_distance_series(
    base_u: Sequence[Sequence[float]],
    perturbed_u: Sequence[Sequence[float]],
) -> list[float]:
    length = min(len(base_u), len(perturbed_u))
    return [
        math.sqrt(
            sum(
                (float(perturbed_u[idx][axis]) - float(base_u[idx][axis])) ** 2
                for axis in range(min(len(base_u[idx]), len(perturbed_u[idx])))
            )
        )
        for idx in range(length)
    ]


def _estimate_lyapunov_from_full_trajectories(
    sample_t: Sequence[float],
    base_u: Sequence[Sequence[float]],
    perturbed_u: Sequence[Sequence[float]],
    *,
    epsilon: float = 1e-8,
    transient_fraction: float = 0.25,
) -> LorenzLyapunovEstimate:
    if len(sample_t) < 4:
        return LorenzLyapunovEstimate(
            status="placeholder",
            value=None,
            method="paired_override_trajectory_slope",
            note="Too few saved trajectory samples for a memory-side Lyapunov estimate.",
        )

    distances = _trajectory_distance_series(base_u, perturbed_u)
    start_idx = max(1, int(len(sample_t) * transient_fraction))
    start_t = float(sample_t[start_idx])
    usable_t: list[float] = []
    usable_y: list[float] = []

    for idx in range(start_idx, min(len(sample_t), len(distances))):
        distance = distances[idx]
        if not math.isfinite(distance) or distance <= 0.0:
            continue
        shifted_t = float(sample_t[idx]) - start_t
        if shifted_t <= 0.0:
            continue
        usable_t.append(shifted_t)
        usable_y.append(math.log(distance / epsilon))

    if len(usable_t) < 3:
        return LorenzLyapunovEstimate(
            status="placeholder",
            value=None,
            method="paired_override_trajectory_slope",
            note="Insufficient post-transient separation samples for a memory-side Lyapunov estimate.",
        )

    mean_t = mean(usable_t)
    mean_y = mean(usable_y)
    variance_t = sum((value - mean_t) ** 2 for value in usable_t)
    if variance_t <= 0.0:
        return LorenzLyapunovEstimate(
            status="placeholder",
            value=None,
            method="paired_override_trajectory_slope",
            note="Degenerate time support prevented a memory-side Lyapunov estimate.",
        )

    covariance = sum((t - mean_t) * (y - mean_y) for t, y in zip(usable_t, usable_y, strict=True))
    slope = covariance / variance_t
    return LorenzLyapunovEstimate(
        status="computed",
        value=slope,
        method="paired_override_trajectory_slope",
        note=(
            "Approximate largest Lyapunov estimate from two override-backed C++ trajectories. "
            "This uses a log-separation slope over saved post-transient samples, so treat it as a screening metric."
        ),
    )


def estimate_cpp_lorenz63_memory_lyapunov(
    *,
    lib=None,
    library_path: str | None = None,
    config: LorenzCppBaselineConfig | None = None,
    scaffold: LorenzExperimentScaffold | None = None,
    initial_state: Sequence[float],
    epsilon: float = 1e-8,
    transient_fraction: float = 0.25,
) -> LorenzLyapunovEstimate:
    """Approximate a memory-side Lyapunov value using two full override-backed C++ runs."""

    memory_source = scaffold or build_reference_lorenz63_scaffold()
    base_state = [float(value) for value in initial_state]
    if len(base_state) not in {3, 3 + memory_source.memory.memory_channels}:
        raise ValueError("initial_state must match the Lorenz state or augmented state size")
    perturbed_state = list(base_state)
    perturbed_state[0] += epsilon

    base_run = run_cpp_lorenz63_with_overrides(
        lib=lib,
        library_path=library_path,
        config=config,
        scaffold=memory_source,
        initial_state=base_state,
    )
    perturbed_run = run_cpp_lorenz63_with_overrides(
        lib=lib,
        library_path=library_path,
        config=config,
        scaffold=memory_source,
        initial_state=perturbed_state,
    )
    return _estimate_lyapunov_from_full_trajectories(
        base_run["trajectory"]["t"],
        base_run["trajectory"]["u"],
        perturbed_run["trajectory"]["u"],
        epsilon=epsilon,
        transient_fraction=transient_fraction,
    )


def build_python_lorenz63_baseline_scaffold(
    runtime: LorenzRuntimeConfig | None = None,
    *,
    dt: float = 0.01,
    integration_method: str = "rk4",
) -> PythonLorenzBaselineScaffold:
    """Build the pure Python Lorenz63 baseline control scaffold."""

    runtime_cfg = runtime or LorenzRuntimeConfig(tspan=(0.0, 200.0), saveat=0.1)
    config = PythonLorenzBaselineConfig.from_runtime(
        runtime_cfg,
        dt=dt,
        integration_method=integration_method,
    )
    return PythonLorenzBaselineScaffold(
        config=config,
        initial_conditions=default_python_lorenz_initial_conditions(),
        notes=[
            "Reference Lorenz63 baseline without explicit memory channels.",
            "Pairs with the C++ abersoe Lorenz63 memory run for baseline-vs-memory comparison.",
        ],
    )


def pack_memory_params(gamma: Sequence[float], weights: Sequence[float]) -> list[float]:
    """Pack descending gamma and non-negative weights in the Julia log-space style."""

    gamma_values = _coerce_float_list(gamma, name="gamma")
    weight_values = _coerce_float_list(weights, name="weights")
    if len(gamma_values) != len(weight_values):
        raise ValueError("gamma and weights must have the same length")
    if any(value < 0.0 for value in weight_values):
        raise ValueError("weights must be non-negative")
    if any(value <= 0.0 for value in weight_values):
        raise ValueError("weights must be strictly positive for log packing")
    _ensure_descending(gamma_values, name="gamma")

    deltas = [0.0] * len(gamma_values)
    deltas[-1] = max(gamma_values[-1], 1e-12)
    for index in range(len(gamma_values) - 2, -1, -1):
        deltas[index] = max(gamma_values[index] - gamma_values[index + 1], 1e-12)
    return [math.log(value) for value in deltas] + [math.log(value) for value in weight_values]


def unpack_memory_params(theta: Sequence[float]) -> tuple[list[float], list[float]]:
    """Inverse of `pack_memory_params`, matching the Julia helper logic."""

    values = [float(value) for value in theta]
    if len(values) == 0 or len(values) % 2 != 0:
        raise ValueError("theta must contain an even number of entries")

    count = len(values) // 2
    deltas = [math.exp(value) for value in values[:count]]
    gamma = [0.0] * count
    gamma[-1] = deltas[-1]
    for index in range(count - 2, -1, -1):
        gamma[index] = gamma[index + 1] + deltas[index]
    weights = [math.exp(value) for value in values[count:]]
    return gamma, weights


def rho_value(state: Sequence[float], params: LorenzMemoryParameters) -> float:
    """Compute the time-varying Lorenz rho parameter induced by memory feedback."""

    if len(state) != params.state_dim:
        raise ValueError(f"state must have length {params.state_dim}")
    chis = state[3:]
    return params.rho0 + sum(weight * chi for weight, chi in zip(params.weights, chis, strict=True))


def rho_stats(states: Sequence[Sequence[float]], params: LorenzMemoryParameters) -> dict[str, float]:
    """Return mean/std/min/max of rho(t) over a saved trajectory."""

    rho_values = [rho_value(state, params) for state in states]
    return {
        "mean": mean(rho_values),
        "std": pstdev(rho_values) if len(rho_values) > 1 else 0.0,
        "min": min(rho_values),
        "max": max(rho_values),
    }


def lorenz_memory_rhs(state: Sequence[float], params: LorenzMemoryParameters) -> list[float]:
    """Deterministic Lorenz63 + memory dynamics from the Julia reference experiment."""

    if len(state) != params.state_dim:
        raise ValueError(f"state must have length {params.state_dim}")

    x, y, z = (float(state[0]), float(state[1]), float(state[2]))
    chis = [float(value) for value in state[3:]]
    rho_t = rho_value(state, params)

    derivatives = [
        params.sigma * (y - x),
        x * (rho_t - z) - y,
        x * y - params.beta * z,
    ]
    derivatives.extend((-gamma_i * chi_i) + x for gamma_i, chi_i in zip(params.gamma, chis, strict=True))
    return derivatives


def lorenz_memory_variational_rhs(
    augmented_state: Sequence[float],
    params: LorenzMemoryParameters,
) -> list[float]:
    """Variational Lorenz63 memory system used for Lyapunov estimation in Julia."""

    n = params.state_dim
    if len(augmented_state) != 2 * n:
        raise ValueError(f"augmented_state must have length {2 * n}")

    base = [float(value) for value in augmented_state[:n]]
    tangent = [float(value) for value in augmented_state[n:]]
    x, y, z = base[0], base[1], base[2]
    delta_x, delta_y, delta_z = tangent[0], tangent[1], tangent[2]
    delta_chis = tangent[3:]

    base_derivative = lorenz_memory_rhs(base, params)
    rho_t = rho_value(base, params)
    memory_tangent = sum(weight * delta_chi for weight, delta_chi in zip(params.weights, delta_chis, strict=True))

    tangent_derivative = [
        (-params.sigma * delta_x) + (params.sigma * delta_y),
        ((rho_t - z) * delta_x) - delta_y - (x * delta_z) + (x * memory_tangent),
        (y * delta_x) + (x * delta_y) - (params.beta * delta_z),
    ]
    tangent_derivative.extend((-gamma_i * delta_chi_i) + delta_x for gamma_i, delta_chi_i in zip(params.gamma, delta_chis, strict=True))
    return base_derivative + tangent_derivative


def _quick_stages() -> list[LorenzOptimizationStage]:
    return [
        LorenzOptimizationStage(
            name="stage1_coarse",
            tspan=(0.0, 80.0),
            transient=20.0,
            maxiters=4,
            training_initial_condition_indices=[0],
            status="baseline",
            notes=["Shorter horizon coarse pass from the Julia quick-mode workflow."],
        ),
        LorenzOptimizationStage(
            name="stage2_refine",
            tspan=(0.0, 140.0),
            transient=35.0,
            maxiters=3,
            training_initial_condition_indices=[0],
            status="baseline",
            notes=["Longer horizon refinement stage from the Julia quick-mode workflow."],
        ),
    ]


def _full_stages() -> list[LorenzOptimizationStage]:
    return [
        LorenzOptimizationStage(
            name="stage1_coarse",
            tspan=(0.0, 120.0),
            transient=30.0,
            maxiters=12,
            training_initial_condition_indices=[0, 1, 2],
            status="baseline",
            notes=["Coarse search over all three initial conditions."],
        ),
        LorenzOptimizationStage(
            name="stage2_refine",
            tspan=(0.0, 200.0),
            transient=50.0,
            maxiters=8,
            training_initial_condition_indices=[0, 1, 2],
            status="baseline",
            notes=["Longer refinement stage over the full initial-condition set."],
        ),
    ]


def build_reference_lorenz63_scaffold(*, quick_mode: bool = True) -> LorenzExperimentScaffold:
    """Build the Python scaffold that mirrors the validated Julia Lorenz63 workflow."""

    memory = default_lorenz_memory_parameters()
    initial_conditions = default_lorenz_initial_conditions(memory.memory_channels)
    runtime = LorenzRuntimeConfig(tspan=(0.0, 200.0), saveat=0.1, solver_hint="auto")
    lyapunov = LorenzLyapunovConfig(transient=50.0, dt_renorm=1.0, seed=1234, solver_hint="stiff")
    stages = _quick_stages() if quick_mode else _full_stages()

    return LorenzExperimentScaffold(
        name="lorenz63_memory_only_reference",
        status="baseline",
        memory=memory,
        runtime=runtime,
        lyapunov=lyapunov,
        training_initial_conditions=initial_conditions,
        validation_initial_conditions=initial_conditions,
        stages=stages,
        evaluation_limit=200,
        quick_mode=quick_mode,
        validated_train_lyapunov=-0.02449,
        validated_periodic_initial_conditions=2,
        notes=[
            "Mirrors the Julia Lorenz63 memory-only optimization structure.",
            "Keeps base Lorenz physics fixed while optimizing only memory decay and feedback parameters.",
            "Use as the Python-side reference scaffold before adding solver-backed optimization loops.",
        ],
        next_actions=[
            "Wire this scaffold into a Python solver loop for trajectory evaluation.",
            "Add Lyapunov estimation and baseline-vs-optimized reporting.",
            "Track unsuccessful runs and basin-dependent failures rather than overwriting them.",
        ],
        metadata={
            "julia_reference": "code/julia/examples/02_lorenz_chaos_suppression.jl",
            "validated_report": "abercsonics_experiment_report.txt",
            "recommended_mode": "memory_only",
            "cpp_baseline": LorenzCppBaselineConfig.from_runtime(runtime).to_mapping(),
            },
    )


def cpp_baseline_config_from_scaffold(
    scaffold: LorenzExperimentScaffold,
    *,
    dt: float = 0.01,
    form: int = gfe.GFE_C_COUPLING_FORM_B,
    strict_finite: bool = True,
) -> LorenzCppBaselineConfig:
    """Build the current C++/ABI runtime mapping for the Lorenz63 scenario."""

    return LorenzCppBaselineConfig.from_runtime(
        scaffold.runtime,
        dt=dt,
        form=form,
        strict_finite=strict_finite,
    )


def _split_lorenz_override_state(
    initial_state: Sequence[float] | None,
    *,
    memory_channels: int,
) -> tuple[list[float], list[float]]:
    if initial_state is None:
        return [1.0, 1.0, 1.0], [0.0] * memory_channels

    values = [float(value) for value in initial_state]
    if len(values) == 3:
        return values, [0.0] * memory_channels
    if len(values) == 3 + memory_channels:
        return values[:3], values[3:]
    raise ValueError(f"initial_state must have length 3 or {3 + memory_channels}")


def run_cpp_lorenz63_with_overrides(
    *,
    lib=None,
    library_path: str | None = None,
    config: LorenzCppBaselineConfig | None = None,
    scaffold: LorenzExperimentScaffold | None = None,
    initial_state: Sequence[float] | None = None,
    gamma: Sequence[float] | None = None,
    weights: Sequence[float] | None = None,
) -> dict[str, object]:
    """Run the C++ Lorenz63 scenario with explicit Python-provided state/kernel overrides."""

    memory_source = scaffold or build_reference_lorenz63_scaffold()
    runtime_cfg = config or cpp_baseline_config_from_scaffold(memory_source)
    initial_u, initial_chi = _split_lorenz_override_state(
        initial_state,
        memory_channels=memory_source.memory.memory_channels,
    )
    active_gamma = [float(value) for value in (gamma if gamma is not None else memory_source.memory.gamma)]
    active_weights = [float(value) for value in (weights if weights is not None else memory_source.memory.weights)]
    if len(active_gamma) != len(active_weights):
        raise ValueError("gamma and weights must have the same length")
    if len(active_gamma) != memory_source.memory.memory_channels:
        raise ValueError("override kernel size must match the scaffold memory-channel count")

    library = lib if lib is not None else load_library(library_path)
    return gfe.run_abersoe_scenario_with_overrides(
        library,
        gfe.GFE_C_ABERSOE_SCENARIO_LORENZ63,
        dt=runtime_cfg.dt,
        form=runtime_cfg.form,
        initial_u=initial_u,
        initial_chi=initial_chi,
        initial_t=memory_source.runtime.tspan[0],
        gamma=active_gamma,
        w=active_weights,
        steps=runtime_cfg.steps,
        sample_stride=runtime_cfg.sample_stride,
        strict_finite=runtime_cfg.strict_finite,
    )


def run_cpp_lorenz63_baseline(
    *,
    lib=None,
    library_path: str | None = None,
    config: LorenzCppBaselineConfig | None = None,
    scaffold: LorenzExperimentScaffold | None = None,
) -> AbersoeRunResult:
    """Run the current C++ Lorenz63 scenario through the override-capable ABI layer."""

    raw = run_cpp_lorenz63_with_overrides(
        lib=lib,
        library_path=library_path,
        config=config,
        scaffold=scaffold,
    )
    return AbersoeRunResult.from_mapping(raw)


def compare_python_baseline_to_cpp_memory(
    *,
    lib=None,
    library_path: str | None = None,
    baseline: PythonLorenzBaselineRun | None = None,
    baseline_scaffold: PythonLorenzBaselineScaffold | None = None,
    memory_scaffold: LorenzExperimentScaffold | None = None,
    cpp_config: LorenzCppBaselineConfig | None = None,
) -> LorenzBaselineMemoryComparison:
    """Build a paired comparison between pure Python baseline and C++ memory run."""

    if baseline is None:
        baseline_source = baseline_scaffold or build_python_lorenz63_baseline_scaffold(
            (memory_scaffold.runtime if memory_scaffold is not None else None)
        )
        baseline = baseline_source.run_default()
    raw_memory = run_cpp_lorenz63_with_overrides(
        lib=lib,
        library_path=library_path,
        config=cpp_config,
        scaffold=memory_scaffold,
    )
    memory_run = AbersoeRunResult.from_mapping(raw_memory)
    return LorenzBaselineMemoryComparison(
        baseline=baseline,
        memory=memory_run,
        notes=[
            "Baseline is plain Python Lorenz63 with no explicit memory.",
            "Memory run uses the current C++ abersoe Lorenz63 scenario through the override-capable ABI.",
        ],
        memory_trace_t=[float(value) for value in raw_memory["trajectory"]["t"]],
        memory_trace_u=[[float(value) for value in state] for state in raw_memory["trajectory"]["u"]],
        memory_trace_chi=[[float(value) for value in state] for state in raw_memory["trajectory"]["chi"]],
        memory_kernel_gamma=[float(value) for value in raw_memory["active_kernel"]["gamma"]],
        memory_kernel_weights=[float(value) for value in raw_memory["active_kernel"]["w"]],
    )


def run_multi_ic_lorenz_batch(
    *,
    lib=None,
    library_path: str | None = None,
    baseline_scaffold: PythonLorenzBaselineScaffold | None = None,
    memory_scaffold: LorenzExperimentScaffold | None = None,
    cpp_config: LorenzCppBaselineConfig | None = None,
    stream: TextIO | None = None,
    tick_seconds: float = 1.0,
) -> LorenzBatchComparison:
    """Run all baseline ICs and pair each one with a real override-backed C++ memory run."""

    baseline_source = baseline_scaffold or build_python_lorenz63_baseline_scaffold(
        (memory_scaffold.runtime if memory_scaffold is not None else None)
    )
    memory_source = memory_scaffold or build_reference_lorenz63_scaffold(quick_mode=True)
    runtime_cfg = cpp_config or cpp_baseline_config_from_scaffold(memory_source)
    per_ic: list[LorenzPerICSummary] = []

    for ic_index, initial_state in enumerate(baseline_source.initial_conditions):
        label = f"python_baseline_ic{ic_index}"
        baseline_run = run_python_lorenz63_baseline(
            baseline_source.config,
            initial_state=initial_state,
            progress_label=label,
            progress_stream=stream,
        )
        baseline_lyap = estimate_python_lorenz63_baseline_lyapunov(
            baseline_source.config,
            initial_state=initial_state,
        )
        memory_raw = _run_with_ticking_elapsed(
            f"cpp_memory_ic{ic_index}",
            lambda current_state=initial_state: run_cpp_lorenz63_with_overrides(
                lib=lib,
                library_path=library_path,
                config=runtime_cfg,
                scaffold=memory_source,
                initial_state=current_state,
            ),
            stream=stream,
            tick_seconds=tick_seconds,
        )
        spectral = spectral_diagnosis_from_kernel(
            memory_raw["active_kernel"]["gamma"],
            memory_raw["active_kernel"]["w"],
            source="override_backed_active_kernel",
            note="Computed from the active kernel returned by the override-capable C++ Lorenz63 ABI run.",
        )
        memory_lyap = estimate_cpp_lorenz63_memory_lyapunov(
            lib=lib,
            library_path=library_path,
            config=runtime_cfg,
            scaffold=memory_source,
            initial_state=initial_state,
        )
        per_ic.append(
            LorenzPerICSummary(
                ic_index=ic_index,
                initial_state=list(initial_state),
                baseline_final_state=list(baseline_run.final_state),
                baseline_final_u_l2=baseline_run.diagnostics.final_u_l2,
                baseline_all_finite=baseline_run.diagnostics.all_finite,
                baseline_lyapunov=baseline_lyap,
                memory_status="completed",
                memory_final_state=[float(value) for value in memory_raw["final_state"]["u"]],
                memory_final_u_l2=float(memory_raw["diagnostics"]["final_u_l2"]),
                memory_all_finite=bool(memory_raw["diagnostics"]["all_finite"]),
                memory_lyapunov=memory_lyap,
                spectral_diagnosis=spectral,
                note="Memory-side run used the override-capable C++ Lorenz63 ABI with matched initial state and active kernel export.",
                baseline_trace_t=[float(value) for value in baseline_run.sample_t],
                baseline_trace_u=[[float(value) for value in state] for state in baseline_run.sample_u],
                memory_trace_t=[float(value) for value in memory_raw["trajectory"]["t"]],
                memory_trace_u=[[float(value) for value in state] for state in memory_raw["trajectory"]["u"]],
                memory_trace_chi=[[float(value) for value in state] for state in memory_raw["trajectory"]["chi"]],
                memory_kernel_gamma=[float(value) for value in memory_raw["active_kernel"]["gamma"]],
                memory_kernel_weights=[float(value) for value in memory_raw["active_kernel"]["w"]],
            )
        )

    return LorenzBatchComparison(
        per_ic=per_ic,
        notes=[
            "All three Julia-style initial conditions were run for the pure Python baseline.",
            "All three memory-side runs now use the override-capable C++ Lorenz63 ABI, so the paired comparison covers the full Julia initial-condition set.",
            "Spectral diagnosis is computed from the active kernel returned by the core rather than mirrored source metadata.",
            "Memory-side Lyapunov values are approximate screening metrics derived from paired full-trajectory override runs.",
        ],
    )


def run_terminal_lorenz_comparison(
    *,
    lib=None,
    library_path: str | None = None,
    baseline_scaffold: PythonLorenzBaselineScaffold | None = None,
    memory_scaffold: LorenzExperimentScaffold | None = None,
    cpp_config: LorenzCppBaselineConfig | None = None,
    output_dir: str | Path | None = None,
    bundle_name: str | None = None,
    stream: TextIO | None = None,
    tick_seconds: float = 1.0,
    use_director: bool = True,
    director: "AberconicsDirector" | None = None,
) -> dict[str, str]:
    """Run the baseline-vs-memory comparison with terminal progress and saved outputs."""

    if use_director:
        try:
            from ..runtime.director import AberconicsDirector
        except ImportError:  # pragma: no cover - direct script execution fallback
            from d2c.runtime.director import AberconicsDirector

        active_director = director or AberconicsDirector(
            lib=lib,
            library_path=library_path,
            baseline_scaffold=baseline_scaffold,
            memory_scaffold=memory_scaffold,
            cpp_config=cpp_config,
            stream=stream,
        )
        _emit_progress(stream, "[run] python_baseline started (director)")
        _emit_progress(stream, "[run] cpp_memory started (director)")
        comparison_result = active_director.run_lorenz63_comparison(ic_index=0)
        summary = comparison_result.summary
        trace_record = comparison_result.trace_record
        baseline_mean_abs_u = (
            sum(abs(value) for state in trace_record.baseline_trace_u for value in state)
            / max(1.0, float(len(trace_record.baseline_trace_u) * 3))
        )
        memory_mean_abs_u = (
            sum(abs(value) for state in trace_record.memory_trace_u for value in state)
            / max(1.0, float(len(trace_record.memory_trace_u) * 3))
        )
        memory_mean_abs_chi = (
            sum(abs(value) for state in trace_record.memory_trace_chi for value in state)
            / max(1.0, float(len(trace_record.memory_trace_chi) * max(1, len(trace_record.memory_kernel_gamma))))
            if trace_record.memory_trace_chi
            else 0.0
        )
        baseline_run = PythonLorenzBaselineRun(
            config=active_director.baseline_scaffold.config,
            initial_state=list(summary.initial_state),
            final_state=list(summary.baseline_final_state),
            diagnostics=PythonLorenzBaselineDiagnostics(
                steps_executed=max(0, len(trace_record.baseline_trace_t) - 1),
                all_finite=bool(summary.baseline_all_finite),
                final_u_l2=float(summary.baseline_final_u_l2),
                mean_abs_u=baseline_mean_abs_u,
                max_abs_u=float(trace_record.baseline_stability["max_abs_u"]),
            ),
            sample_t=list(trace_record.baseline_trace_t),
            sample_u=[list(state) for state in trace_record.baseline_trace_u],
        )
        memory_run = AbersoeRunResult.from_mapping(
            {
                "final_state": {
                    "u": summary.memory_final_state,
                    "chi": trace_record.memory_trace_chi[-1] if trace_record.memory_trace_chi else [],
                    "t": trace_record.memory_trace_t[-1],
                },
                "diagnostics": {
                    "steps_executed": max(0, len(trace_record.memory_trace_t) - 1),
                    "all_finite": summary.memory_all_finite,
                    "max_abs_u": trace_record.memory_stability["max_abs_u"],
                    "max_abs_chi": trace_record.memory_stability["max_abs_chi"],
                    "final_u_l2": summary.memory_final_u_l2,
                    "final_chi_l2": trace_record.memory_stability["final_chi_l2"],
                    "mean_abs_u": memory_mean_abs_u,
                    "mean_abs_chi": memory_mean_abs_chi,
                },
                "config": {
                    "memory_channels": len(trace_record.memory_kernel_gamma),
                    "dt": active_director.cpp_config.dt if active_director.cpp_config is not None else 0.01,
                    "coupling_index": 0,
                    "steps": max(0, len(trace_record.memory_trace_t) - 1),
                    "sample_stride": 1,
                    "strict_finite": True,
                    "stochastic_forcing": False,
                    "forcing_noise_std": 0.0,
                    "stochastic_channel": 0,
                    "seed": 0,
                    "fit_backend": 0,
                    "hebbian_enabled": False,
                    "hebbian_rule": 0,
                    "hebbian_learning_rate": 0.0,
                    "hebbian_decay": 0.0,
                },
                "samples": {
                    "t": trace_record.memory_trace_t,
                    "u0": [state[0] for state in trace_record.memory_trace_u],
                    "chi0": [state[0] for state in trace_record.memory_trace_chi] if trace_record.memory_trace_chi else [],
                    "u_l2": [math.sqrt(sum(value * value for value in state)) for state in trace_record.memory_trace_u],
                    "chi_l2": [
                        math.sqrt(sum(value * value for value in state)) for state in trace_record.memory_trace_chi
                    ]
                    if trace_record.memory_trace_chi
                    else [],
                },
            }
        )
        comparison = LorenzBaselineMemoryComparison(
            baseline=baseline_run,
            memory=memory_run,
            notes=[
                "Baseline and memory comparison executed through the AberconicsDirector path.",
                "TraceStore retained the stored trajectories used for this bundle.",
            ],
            memory_trace_t=list(trace_record.memory_trace_t),
            memory_trace_u=[list(state) for state in trace_record.memory_trace_u],
            memory_trace_chi=[list(state) for state in trace_record.memory_trace_chi],
            memory_kernel_gamma=list(trace_record.memory_kernel_gamma),
            memory_kernel_weights=list(trace_record.memory_kernel_weights),
        )
        paths = save_lorenz_comparison_bundle(
            comparison,
            output_dir=output_dir,
            bundle_name=bundle_name,
        )
        trace_store_path = Path(paths["bundle_dir"]) / "trace_store.json"
        trace_store_path.write_text(
            json.dumps(active_director.trace_store.to_mapping(), indent=2, sort_keys=True), encoding="utf-8"
        )
        try:
            from ..learning import analyze_trace_store_windows, format_learning_analysis_report
        except ImportError:  # pragma: no cover - direct script execution fallback
            from d2c.learning import analyze_trace_store_windows, format_learning_analysis_report

        learning_analysis = analyze_trace_store_windows(
            active_director.trace_store,
            source="memory",
            window_size=8,
            stride=4,
        )
        learning_analysis_path = Path(paths["bundle_dir"]) / "learning_analysis.json"
        learning_analysis_path.write_text(
            json.dumps(learning_analysis.to_mapping(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        learning_report = format_learning_analysis_report(learning_analysis)
        learning_report_path = Path(paths["bundle_dir"]) / "learning_report.txt"
        learning_report_path.write_text(learning_report + "\n", encoding="utf-8")

        paths["trace_store_json"] = str(trace_store_path)
        paths["learning_analysis_json"] = str(learning_analysis_path)
        paths["learning_report_txt"] = str(learning_report_path)

        report = Path(paths["report_txt"]).read_text(encoding="utf-8").rstrip()
        _emit_progress(stream, report)
        _emit_progress(stream, f"[save] bundle_dir={paths['bundle_dir']}")
        _emit_progress(stream, f"[save] report_txt={paths['report_txt']}")
        _emit_progress(stream, f"[save] summary_json={paths['summary_json']}")
        _emit_progress(stream, f"[save] trace_store_json={paths['trace_store_json']}")
        return paths

    baseline_source = baseline_scaffold or build_python_lorenz63_baseline_scaffold(
        (memory_scaffold.runtime if memory_scaffold is not None else None)
    )
    baseline = run_python_lorenz63_baseline(
        baseline_source.config,
        initial_state=baseline_source.initial_conditions[0],
        progress_label="python_baseline",
        progress_stream=stream,
    )
    memory_raw = _run_with_ticking_elapsed(
        "cpp_memory",
        lambda: run_cpp_lorenz63_with_overrides(
            lib=lib,
            library_path=library_path,
            config=cpp_config,
            scaffold=memory_scaffold,
        ),
        stream=stream,
        tick_seconds=tick_seconds,
    )
    memory_run = AbersoeRunResult.from_mapping(memory_raw)
    comparison = LorenzBaselineMemoryComparison(
        baseline=baseline,
        memory=memory_run,
        notes=[
            "Baseline is a pure Python Lorenz63 control condition.",
            "Memory run is the current C++ abersoe Lorenz63 scenario invoked through the override-capable ABI.",
        ],
        memory_trace_t=[float(value) for value in memory_raw["trajectory"]["t"]],
        memory_trace_u=[[float(value) for value in state] for state in memory_raw["trajectory"]["u"]],
        memory_trace_chi=[[float(value) for value in state] for state in memory_raw["trajectory"]["chi"]],
        memory_kernel_gamma=[float(value) for value in memory_raw["active_kernel"]["gamma"]],
        memory_kernel_weights=[float(value) for value in memory_raw["active_kernel"]["w"]],
    )
    paths = save_lorenz_comparison_bundle(
        comparison,
        output_dir=output_dir,
        bundle_name=bundle_name,
    )
    report = Path(paths["report_txt"]).read_text(encoding="utf-8").rstrip()
    _emit_progress(stream, report)
    _emit_progress(stream, f"[save] bundle_dir={paths['bundle_dir']}")
    _emit_progress(stream, f"[save] report_txt={paths['report_txt']}")
    _emit_progress(stream, f"[save] summary_json={paths['summary_json']}")
    return paths


def run_terminal_lorenz_batch(
    *,
    lib=None,
    library_path: str | None = None,
    baseline_scaffold: PythonLorenzBaselineScaffold | None = None,
    memory_scaffold: LorenzExperimentScaffold | None = None,
    cpp_config: LorenzCppBaselineConfig | None = None,
    output_dir: str | Path | None = None,
    bundle_name: str | None = None,
    stream: TextIO | None = None,
    tick_seconds: float = 1.0,
    use_director: bool = True,
    director: "AberconicsDirector" | None = None,
) -> dict[str, str]:
    """Run multi-IC batch analysis with saved per-IC reports."""

    if use_director:
        try:
            from ..runtime.director import AberconicsDirector
        except ImportError:  # pragma: no cover - direct script execution fallback
            from d2c.runtime.director import AberconicsDirector

        active_director = director or AberconicsDirector(
            lib=lib,
            library_path=library_path,
            baseline_scaffold=baseline_scaffold,
            memory_scaffold=memory_scaffold,
            cpp_config=cpp_config,
            stream=stream,
        )
        result = active_director.run_lorenz63_batch()
        paths = save_lorenz_batch_bundle(
            result.batch,
            output_dir=output_dir,
            bundle_name=bundle_name,
        )
        trace_store_path = Path(paths["bundle_dir"]) / "trace_store.json"
        trace_store_path.write_text(json.dumps(result.trace_store.to_mapping(), indent=2, sort_keys=True), encoding="utf-8")
        window_manifest = {
            "baseline_window_count_size_8_stride_4": len(
                result.trace_store.build_all_windows(source="baseline", window_size=8, stride=4)
            ),
            "memory_window_count_size_8_stride_4": len(
                result.trace_store.build_all_windows(source="memory", window_size=8, stride=4)
            ),
        }
        window_manifest_path = Path(paths["bundle_dir"]) / "trace_windows.json"
        window_manifest_path.write_text(json.dumps(window_manifest, indent=2, sort_keys=True), encoding="utf-8")
        try:
            from ..learning import analyze_trace_store_windows, format_learning_analysis_report
        except ImportError:  # pragma: no cover - direct script execution fallback
            from d2c.learning import analyze_trace_store_windows, format_learning_analysis_report

        learning_analysis = analyze_trace_store_windows(
            result.trace_store,
            source="memory",
            window_size=8,
            stride=4,
        )
        learning_analysis_path = Path(paths["bundle_dir"]) / "learning_analysis.json"
        learning_analysis_path.write_text(
            json.dumps(learning_analysis.to_mapping(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        learning_report = format_learning_analysis_report(learning_analysis)
        learning_report_path = Path(paths["bundle_dir"]) / "learning_report.txt"
        learning_report_path.write_text(learning_report + "\n", encoding="utf-8")

        paths["trace_store_json"] = str(trace_store_path)
        paths["trace_windows_json"] = str(window_manifest_path)
        paths["learning_analysis_json"] = str(learning_analysis_path)
        paths["learning_report_txt"] = str(learning_report_path)
        report = Path(paths["report_txt"]).read_text(encoding="utf-8").rstrip()
        _emit_progress(stream, report)
        _emit_progress(stream, f"[save] batch_bundle_dir={paths['bundle_dir']}")
        _emit_progress(stream, f"[save] batch_summary_json={paths['summary_json']}")
        _emit_progress(stream, f"[save] trace_store_json={paths['trace_store_json']}")
        _emit_progress(stream, f"[save] trace_windows_json={paths['trace_windows_json']}")
        _emit_progress(stream, f"[save] learning_analysis_json={paths['learning_analysis_json']}")
        return paths

    batch = run_multi_ic_lorenz_batch(
        lib=lib,
        library_path=library_path,
        baseline_scaffold=baseline_scaffold,
        memory_scaffold=memory_scaffold,
        cpp_config=cpp_config,
        stream=stream,
        tick_seconds=tick_seconds,
    )
    paths = save_lorenz_batch_bundle(
        batch,
        output_dir=output_dir,
        bundle_name=bundle_name,
    )
    report = Path(paths["report_txt"]).read_text(encoding="utf-8").rstrip()
    _emit_progress(stream, report)
    _emit_progress(stream, f"[save] batch_bundle_dir={paths['bundle_dir']}")
    _emit_progress(stream, f"[save] batch_summary_json={paths['summary_json']}")
    return paths


def summarize_cpp_lorenz63_baseline(run: AbersoeRunResult) -> dict[str, object]:
    """Return a lightweight summary of the currently exposed C++ Lorenz63 run surface."""

    return {
        "final_state": {
            "u": list(run.final_state.u),
            "chi": list(run.final_state.chi),
            "t": run.final_state.t,
        },
        "diagnostics": {
            "steps_executed": run.diagnostics.steps_executed,
            "all_finite": run.diagnostics.all_finite,
            "final_u_l2": run.diagnostics.final_u_l2,
            "final_chi_l2": run.diagnostics.final_chi_l2,
            "mean_abs_u": run.diagnostics.mean_abs_u,
            "mean_abs_chi": run.diagnostics.mean_abs_chi,
        },
        "config": {
            "memory_channels": run.config.memory_channels,
            "dt": run.config.dt,
            "steps": run.config.steps,
            "sample_stride": run.config.sample_stride,
            "hebbian_enabled": run.config.hebbian_enabled,
        },
        "available_series": {
            "t_samples": len(run.samples.t),
            "u0_samples": len(run.samples.u0),
            "chi0_samples": len(run.samples.chi0),
            "note": "The lightweight baseline summary keeps the standard sample traces; the override-backed path also exposes full trajectory and active-kernel data.",
        },
    }


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run terminal Lorenz63 baseline-vs-memory comparison.")
    parser.add_argument("--library-path", default=None, help="Optional path to libgfe_core shared library.")
    parser.add_argument("--bundle-name", default=None, help="Optional bundle directory name under progress/lorenz63.")
    parser.add_argument("--output-dir", default=None, help="Optional output directory override.")
    parser.add_argument("--tick-seconds", type=float, default=1.0, help="Seconds between C++ memory tick messages.")
    parser.add_argument("--batch", action="store_true", help="Run multi-IC batch mode instead of the single paired comparison.")
    parser.add_argument(
        "--integration-method",
        choices=("rk4", "euler"),
        default="rk4",
        help="Integration method for the pure Python Lorenz baseline.",
    )
    parser.add_argument(
        "--legacy-entrypoint",
        action="store_true",
        help="Use the pre-Director terminal orchestration path instead of the Director-managed default.",
    )
    parser.add_argument(
        "--learning",
        action="store_true",
        help="Run an online learning episode through the Director for the first IC.",
    )
    parser.add_argument(
        "--learning-steps",
        type=int,
        default=2000,
        help="Total steps for the online learning episode.",
    )
    return parser


def _save_learning_plots(history: list[DirectorState], bundle_dir: Path) -> dict[str, str]:
    """Generate plots for a single-IC online learning episode history."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return {}

    # Redirect matplotlib config to a writable directory if needed
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    plot_paths = {}
    ts = [s.t for s in history]
    
    # 1. Trajectory (x, y, z)
    traj_path = bundle_dir / "learning_trajectory.png"
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    xs = [s.u[0] for s in history]
    ys = [s.u[1] for s in history]
    zs = [s.u[2] for s in history]
    
    axes[0].plot(ts, xs, color="tab:blue", label="x")
    axes[1].plot(ts, ys, color="tab:orange", label="y")
    axes[2].plot(ts, zs, color="tab:green", label="z")
    
    axes[0].set_title("Lorenz63 Learning Trajectory")
    axes[2].set_xlabel("time (s)")
    for ax, label in zip(axes, ["x", "y", "z"]):
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.2)
        ax.legend(loc="upper right")
        
    fig.tight_layout()
    fig.savefig(traj_path, dpi=160)
    plt.close(fig)
    plot_paths["learning_trajectory_png"] = str(traj_path)

    # 2. Weights Evolution
    weights_path = bundle_dir / "learning_weights.png"
    fig, ax = plt.subplots(figsize=(10, 5))
    num_channels = len(history[0].w)
    for i in range(num_channels):
        wi = [s.w[i] for s in history]
        ax.plot(ts, wi, label=f"w{i}")
        
    ax.set_title("Memory Weights Evolution")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("weight")
    ax.grid(True, alpha=0.2)
    ax.legend()
    
    fig.tight_layout()
    fig.savefig(weights_path, dpi=160)
    plt.close(fig)
    plot_paths["learning_weights_png"] = str(weights_path)

    # 3. Smoothed Prediction Error
    error_path = bundle_dir / "learning_error.png"
    fig, ax = plt.subplots(figsize=(10, 5))
    errors = [s.smoothed_error for s in history]
    ax.plot(ts, errors, color="tab:red", label="smoothed error")
    ax.set_title("Smoothed Prediction Error")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("error (L2)")
    ax.grid(True, alpha=0.2)
    ax.set_yscale("log")
    
    fig.tight_layout()
    fig.savefig(error_path, dpi=160)
    plt.close(fig)
    plot_paths["learning_error_png"] = str(error_path)

    return plot_paths


def run_terminal_learning_episode(
    *,
    lib=None,
    library_path: str | None = None,
    baseline_scaffold: PythonLorenzBaselineScaffold | None = None,
    memory_scaffold: LorenzExperimentScaffold | None = None,
    cpp_config: LorenzCppBaselineConfig | None = None,
    output_dir: str | Path | None = None,
    bundle_name: str | None = None,
    stream: TextIO | None = None,
    total_steps: int = 2000,
) -> dict[str, str]:
    """Run a single-IC online learning episode with saved reports and traces."""
    try:
        from ..runtime.director import AberconicsDirector
    except ImportError:  # pragma: no cover - direct script execution fallback
        from d2c.runtime.director import AberconicsDirector

    active_director = AberconicsDirector(
        lib=lib,
        library_path=library_path,
        baseline_scaffold=baseline_scaffold,
        memory_scaffold=memory_scaffold,
        cpp_config=cpp_config,
        stream=stream,
    )

    initial_u = baseline_scaffold.initial_conditions[0] if baseline_scaffold else [1.0, 1.0, 1.0]
    _emit_progress(stream, f"[director] starting online learning episode for initial_u={initial_u}")

    history = active_director.run_learning_episode(initial_u, total_steps=total_steps)
    
    # Save results
    bundle_root = Path(output_dir) if output_dir is not None else _progress_dir_root()
    bundle_root.mkdir(parents=True, exist_ok=True)
    bundle_dir = bundle_root / (bundle_name or f"learning_{_timestamp_slug()}")
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # Save history as JSON
    history_data = []
    for s in history:
        history_data.append({
            "t": s.t,
            "u": s.u,
            "chi": s.chi,
            "w": s.w,
            "phase": s.phase.name,
            "step": s.step_count,
        })
    
    history_path = bundle_dir / "learning_history.json"
    history_path.write_text(json.dumps(history_data, indent=2), encoding="utf-8")

    # Generate plots
    plot_paths = _save_learning_plots(history, bundle_dir)

    # Generate a summary report
    final_state = history[-1]
    phases_reached = sorted(list(set(s.phase.name for s in history)))
    
    report_lines = [
        "Online Learning Episode Summary",
        "===============================",
        f"Initial u: {initial_u}",
        f"Final t:   {final_state.t:.3f}",
        f"Steps:     {len(history)}",
        f"Phases:    {', '.join(phases_reached)}",
        "",
        "Final Weights (w):",
        f"  {final_state.w}",
        "",
        "Final u:",
        f"  {final_state.u}",
    ]
    if plot_paths:
        report_lines.append("")
        report_lines.append("Plots:")
        for name, path in plot_paths.items():
            report_lines.append(f"  {name}={path}")
    
    report = "\n".join(report_lines)
    report_path = bundle_dir / "learning_report.txt"
    report_path.write_text(report + "\n", encoding="utf-8")

    _emit_progress(stream, report)
    _emit_progress(stream, f"[save] learning_bundle_dir={bundle_dir}")
    
    return {
        "bundle_dir": str(bundle_dir),
        "history_json": str(history_path),
        "report_txt": str(report_path),
        **plot_paths,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_cli_parser()
    args = parser.parse_args(argv)
    memory_scaffold = build_reference_lorenz63_scaffold(quick_mode=True)
    baseline_scaffold = build_python_lorenz63_baseline_scaffold(
        memory_scaffold.runtime,
        integration_method=args.integration_method,
    )
    if args.learning:
        run_terminal_learning_episode(
            library_path=args.library_path,
            baseline_scaffold=baseline_scaffold,
            memory_scaffold=memory_scaffold,
            output_dir=args.output_dir,
            bundle_name=args.bundle_name,
            stream=sys.stdout,
            total_steps=args.learning_steps,
        )
    elif args.batch:
        run_terminal_lorenz_batch(
            library_path=args.library_path,
            baseline_scaffold=baseline_scaffold,
            memory_scaffold=memory_scaffold,
            output_dir=args.output_dir,
            bundle_name=args.bundle_name,
            stream=sys.stdout,
            tick_seconds=args.tick_seconds,
            use_director=not args.legacy_entrypoint,
        )
    else:
        run_terminal_lorenz_comparison(
            library_path=args.library_path,
            baseline_scaffold=baseline_scaffold,
            memory_scaffold=memory_scaffold,
            output_dir=args.output_dir,
            bundle_name=args.bundle_name,
            stream=sys.stdout,
            tick_seconds=args.tick_seconds,
            use_director=not args.legacy_entrypoint,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
