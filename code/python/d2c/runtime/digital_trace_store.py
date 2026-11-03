"""Bounded, continuity-aware trace storage for digital F0 runtimes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import random
from typing import Sequence


@dataclass(slots=True)
class DigitalRuntimeRecord:
    """One post-step sample from a continuously managed digital runtime."""

    step_index: int
    t: float
    session_id: str
    continuity_id: int
    episode_id: str
    phase: str
    u: list[float]
    chi: list[list[float]]
    forcing: list[float]
    token: str | None = None
    prediction_error: float | None = None
    reward: float | None = None
    stability_ratio: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.step_index < 0 or self.continuity_id < 0:
            raise ValueError("step_index and continuity_id must be non-negative")
        if not self.session_id or not self.episode_id:
            raise ValueError("session_id and episode_id must be non-empty")
        if not self.u or not self.forcing:
            raise ValueError("u and forcing must be non-empty")
        if any(len(channel) != len(self.u) for channel in self.chi):
            raise ValueError("each chi channel must match the u dimension")
        values = list(self.u) + list(self.forcing)
        values.extend(value for channel in self.chi for value in channel)
        values.extend(
            value for value in (self.t, self.prediction_error, self.reward, self.stability_ratio)
            if value is not None
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("trace records may contain only finite numeric values")

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class DigitalTraceWindow:
    """A contiguous window. The optional next_u is the following state."""

    records: list[DigitalRuntimeRecord]
    next_u: list[float] | None = None

    @property
    def start_step(self) -> int:
        return self.records[0].step_index

    @property
    def end_step(self) -> int:
        return self.records[-1].step_index


@dataclass(slots=True)
class DigitalTraceStore:
    """Fixed-capacity circular trace buffer that never samples across resets.

    Windows must be consecutive in global step index and share both a session
    and continuity id. A memory reset increments continuity_id, preventing
    replay windows from joining unrelated episodic states.
    """

    capacity: int = 20_000
    _buffer: list[DigitalRuntimeRecord | None] = field(init=False, repr=False)
    _next_index: int = field(default=0, init=False, repr=False)
    _size: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        self._buffer = [None] * self.capacity

    def __len__(self) -> int:
        return self._size

    @property
    def records(self) -> list[DigitalRuntimeRecord]:
        """Return records in chronological order, oldest first."""
        if self._size == 0:
            return []
        start = (self._next_index - self._size) % self.capacity
        result: list[DigitalRuntimeRecord] = []
        for offset in range(self._size):
            record = self._buffer[(start + offset) % self.capacity]
            if record is not None:
                result.append(record)
        return result

    def append(self, record: DigitalRuntimeRecord) -> None:
        stored = DigitalRuntimeRecord(
            step_index=record.step_index,
            t=float(record.t),
            session_id=record.session_id,
            continuity_id=record.continuity_id,
            episode_id=record.episode_id,
            phase=record.phase,
            u=list(record.u),
            chi=[list(channel) for channel in record.chi],
            forcing=list(record.forcing),
            token=record.token,
            prediction_error=record.prediction_error,
            reward=record.reward,
            stability_ratio=record.stability_ratio,
            metadata=dict(record.metadata),
        )
        self._buffer[self._next_index] = stored
        self._next_index = (self._next_index + 1) % self.capacity
        self._size = min(self.capacity, self._size + 1)

    @staticmethod
    def _is_contiguous(records: Sequence[DigitalRuntimeRecord]) -> bool:
        if not records:
            return False
        first = records[0]
        for previous, current in zip(records, records[1:]):
            if current.session_id != first.session_id:
                return False
            if current.continuity_id != first.continuity_id:
                return False
            if current.step_index != previous.step_index + 1:
                return False
        return True

    def build_windows(
        self,
        window_size: int,
        *,
        stride: int = 1,
        session_id: str | None = None,
        continuity_id: int | None = None,
        include_next_u: bool = True,
    ) -> list[DigitalTraceWindow]:
        if window_size <= 0 or stride <= 0:
            raise ValueError("window_size and stride must be positive")
        rows = self.records
        windows: list[DigitalTraceWindow] = []
        for start in range(0, max(0, len(rows) - window_size + 1), stride):
            candidate = rows[start : start + window_size]
            if len(candidate) != window_size or not self._is_contiguous(candidate):
                continue
            if session_id is not None and candidate[0].session_id != session_id:
                continue
            if continuity_id is not None and candidate[0].continuity_id != continuity_id:
                continue
            next_index = start + window_size
            next_u = None
            if include_next_u and next_index < len(rows):
                next_record = rows[next_index]
                if self._is_contiguous(candidate + [next_record]):
                    next_u = list(next_record.u)
            windows.append(DigitalTraceWindow(records=candidate, next_u=next_u))
        return windows

    def sample_contiguous(
        self,
        window_size: int,
        *,
        stride: int = 1,
        seed: int = 0,
        session_id: str | None = None,
        continuity_id: int | None = None,
    ) -> DigitalTraceWindow:
        windows = self.build_windows(
            window_size,
            stride=stride,
            session_id=session_id,
            continuity_id=continuity_id,
        )
        if not windows:
            raise LookupError("no contiguous trace window satisfies the requested constraints")
        return random.Random(seed).choice(windows)

    def to_mapping(self) -> dict[str, object]:
        return {
            "capacity": self.capacity,
            "size": self._size,
            "records": [record.to_mapping() for record in self.records],
        }
