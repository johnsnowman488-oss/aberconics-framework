"""Deterministic six-phase controller for the digital F0 runtime."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum


class DigitalRuntimePhase(str, Enum):
    WARMUP = "WARMUP"
    EXPLORE = "EXPLORE"
    CONSOLIDATE = "CONSOLIDATE"
    EVALUATE = "EVALUATE"
    PRUNE = "PRUNE"
    GROW = "GROW"


_TRANSIENT_PHASES = {
    DigitalRuntimePhase.EVALUATE,
    DigitalRuntimePhase.PRUNE,
    DigitalRuntimePhase.GROW,
}


@dataclass(slots=True)
class DigitalPhaseConfig:
    warmup_steps: int = 8
    explore_steps: int = 64
    consolidate_steps: int = 8
    evaluation_interval: int = 128

    def __post_init__(self) -> None:
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.explore_steps <= 0 or self.consolidate_steps <= 0:
            raise ValueError("explore_steps and consolidate_steps must be positive")
        if self.evaluation_interval < 0:
            raise ValueError("evaluation_interval must be non-negative")

    def to_mapping(self) -> dict[str, int]:
        return asdict(self)


@dataclass(slots=True)
class DigitalPhaseEvent:
    step_index: int
    previous: str
    current: str
    reason: str

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class DigitalPhaseController:
    """Own phase transitions; maintenance phases are observational in F0.

    PRUNE and GROW are one-step proposal/report phases. They deliberately do
    not change channel count; actual channel lifecycle operations belong to F4.
    EVALUATE is also a one-step phase pulse that returns to the previous phase.
    """

    config: DigitalPhaseConfig = field(default_factory=DigitalPhaseConfig)
    phase: DigitalRuntimePhase = DigitalRuntimePhase.WARMUP
    total_steps: int = 0
    phase_steps: int = 0
    _resume_phase: DigitalRuntimePhase | None = field(default=None, repr=False)
    _resume_phase_steps: int = field(default=0, repr=False)
    _pending_maintenance: tuple[DigitalRuntimePhase, str] | None = field(default=None, repr=False)
    events: list[DigitalPhaseEvent] = field(default_factory=list)

    def _transition(self, phase: DigitalRuntimePhase, reason: str) -> None:
        previous = self.phase
        if previous == phase:
            return
        self.phase = phase
        self.phase_steps = 0
        self.events.append(
            DigitalPhaseEvent(
                step_index=self.total_steps,
                previous=previous.value,
                current=phase.value,
                reason=reason,
            )
        )

    def request_maintenance(self, phase: str, *, reason: str) -> None:
        normalized = phase.upper()
        if normalized not in {"PRUNE", "GROW"}:
            raise ValueError("maintenance phase must be PRUNE or GROW")
        self._pending_maintenance = (DigitalRuntimePhase(normalized), reason)

    def complete_step(self) -> None:
        """Advance counters and select the phase for the next integration step."""
        self.total_steps += 1
        self.phase_steps += 1

        if self.phase in _TRANSIENT_PHASES:
            resume = self._resume_phase or DigitalRuntimePhase.EXPLORE
            resume_steps = self._resume_phase_steps
            self._resume_phase = None
            self._resume_phase_steps = 0
            self._transition(resume, "transient phase complete")
            self.phase_steps = resume_steps
            return

        if self.phase == DigitalRuntimePhase.WARMUP:
            if self.phase_steps >= self.config.warmup_steps:
                self._transition(DigitalRuntimePhase.EXPLORE, "warmup budget reached")
            return

        if self.phase == DigitalRuntimePhase.CONSOLIDATE:
            if self.phase_steps >= self.config.consolidate_steps:
                self._transition(DigitalRuntimePhase.EXPLORE, "consolidation budget reached")
            return

        if self.phase == DigitalRuntimePhase.EXPLORE:
            if self._pending_maintenance is not None:
                target, reason = self._pending_maintenance
                self._pending_maintenance = None
                self._resume_phase = DigitalRuntimePhase.EXPLORE
                self._resume_phase_steps = self.phase_steps
                self._transition(target, reason)
                return
            if (
                self.config.evaluation_interval > 0
                and self.total_steps % self.config.evaluation_interval == 0
            ):
                self._resume_phase = DigitalRuntimePhase.EXPLORE
                self._resume_phase_steps = self.phase_steps
                self._transition(DigitalRuntimePhase.EVALUATE, "evaluation interval reached")
                return
            if self.phase_steps >= self.config.explore_steps:
                self._transition(DigitalRuntimePhase.CONSOLIDATE, "exploration budget reached")

    def to_mapping(self) -> dict[str, object]:
        return {
            "config": self.config.to_mapping(),
            "phase": self.phase.value,
            "total_steps": self.total_steps,
            "phase_steps": self.phase_steps,
            "resume_phase": None if self._resume_phase is None else self._resume_phase.value,
            "resume_phase_steps": self._resume_phase_steps,
            "pending_maintenance": None if self._pending_maintenance is None else {
                "phase": self._pending_maintenance[0].value,
                "reason": self._pending_maintenance[1],
            },
            "events": [event.to_mapping() for event in self.events],
        }

    @classmethod
    def from_mapping(cls, data: dict[str, object]) -> "DigitalPhaseController":
        controller = cls(
            config=DigitalPhaseConfig(**dict(data["config"])),
            phase=DigitalRuntimePhase(str(data["phase"])),
            total_steps=int(data["total_steps"]),
            phase_steps=int(data["phase_steps"]),
        )
        resume = data.get("resume_phase")
        controller._resume_phase = None if resume is None else DigitalRuntimePhase(str(resume))
        controller._resume_phase_steps = int(data.get("resume_phase_steps", 0))
        pending = data.get("pending_maintenance")
        if pending is not None:
            pending_data = dict(pending)
            controller._pending_maintenance = (
                DigitalRuntimePhase(str(pending_data["phase"])),
                str(pending_data["reason"]),
            )
        controller.events = [
            DigitalPhaseEvent(
                step_index=int(row["step_index"]),
                previous=str(row["previous"]),
                current=str(row["current"]),
                reason=str(row["reason"]),
            )
            for row in data.get("events", [])
        ]
        return controller
