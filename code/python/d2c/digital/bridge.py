"""Token-to-forcing bridge and lightweight digital memory stepping."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Mapping, Sequence

from .streams import DigitalStream
from .tokens import Vocabulary


def _coerce(values: Sequence[float], *, name: str) -> list[float]:
    out = [float(value) for value in values]
    if not out:
        raise ValueError(f"{name} must not be empty")
    return out


@dataclass(slots=True)
class TokenForcingBridge:
    """Deterministic token-to-vector forcing bridge."""

    vocabulary: Vocabulary
    dim: int | None = None
    seed: int = 0
    amplitude: float = 1.0
    code_mode: str = "dense"

    def __post_init__(self) -> None:
        if self.code_mode not in {"dense", "one_hot"}:
            raise ValueError("code_mode must be dense or one_hot")
        if self.code_mode == "one_hot" and self.dim is not None and self.dim != self.vocabulary.size:
            raise ValueError("one_hot bridge dim must match vocabulary size")

    @property
    def state_dim(self) -> int:
        return self.vocabulary.size if self.code_mode == "one_hot" else int(self.dim or self.vocabulary.size)

    def forcing_for_token(self, token: str) -> list[float]:
        if self.code_mode == "one_hot":
            return [self.amplitude * value for value in self.vocabulary.one_hot(token)]
        return [
            self.amplitude * value
            for value in self.vocabulary.deterministic_code(token, dim=self.state_dim, seed=self.seed)
        ]

    def zero_forcing(self) -> list[float]:
        return [0.0 for _ in range(self.state_dim)]

    def to_mapping(self) -> dict[str, object]:
        data = asdict(self)
        data["vocabulary"] = self.vocabulary.to_mapping()
        return data


@dataclass(slots=True)
class KeyValueBindingBridge:
    """Deterministic outer-product slots for key-conditioned retrieval.

    A binding occupies one coordinate for each ``(key, value)`` pair.  The
    resulting vector can be stored by the same SOE memory stepper as ordinary
    token forcing, then queried by contracting the stored slots with a key.
    """

    key_tokens: list[str]
    value_tokens: list[str]
    amplitude: float = 1.0

    def __post_init__(self) -> None:
        if not self.key_tokens or not self.value_tokens:
            raise ValueError("key_tokens and value_tokens must not be empty")
        if len(set(self.key_tokens)) != len(self.key_tokens):
            raise ValueError("key_tokens must be unique")
        if len(set(self.value_tokens)) != len(self.value_tokens):
            raise ValueError("value_tokens must be unique")

    @property
    def state_dim(self) -> int:
        return len(self.key_tokens) * len(self.value_tokens)

    def binding_for_pair(self, key_token: str, value_token: str) -> list[float]:
        try:
            key_index = self.key_tokens.index(key_token)
            value_index = self.value_tokens.index(value_token)
        except ValueError as exc:
            raise ValueError("binding pair tokens must be in the configured vocabularies") from exc
        forcing = self.zero_forcing()
        forcing[key_index * len(self.value_tokens) + value_index] = self.amplitude
        return forcing

    def zero_forcing(self) -> list[float]:
        return [0.0 for _ in range(self.state_dim)]

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class DigitalMemoryConfig:
    gamma: list[float]
    w: list[float]
    dt: float = 0.05
    leak_rate: float = 1.0
    form: str = "input_driven"

    def __post_init__(self) -> None:
        self.gamma = _coerce(self.gamma, name="gamma")
        self.w = _coerce(self.w, name="w")
        if len(self.gamma) != len(self.w):
            raise ValueError("gamma and w must have matching lengths")
        if any(rate <= 0.0 for rate in self.gamma):
            raise ValueError("gamma values must be positive")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.leak_rate <= 0.0:
            raise ValueError("leak_rate must be positive")
        if self.form not in {"input_driven", "negative_feedback"}:
            raise ValueError("form must be input_driven or negative_feedback")

    @property
    def channel_count(self) -> int:
        return len(self.gamma)

    def stability_ratio(self) -> float:
        return sum(weight / (rate * self.leak_rate) for weight, rate in zip(self.w, self.gamma))

    def deff(self) -> float:
        total = sum(max(0.0, weight) for weight in self.w)
        if total <= 0.0:
            return 0.0
        probs = [max(0.0, weight) / total for weight in self.w]
        return 1.0 / sum(prob * prob for prob in probs if prob > 0.0)

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "DigitalMemoryConfig":
        return cls(
            gamma=[float(value) for value in data["gamma"]],
            w=[float(value) for value in data["w"]],
            dt=float(data.get("dt", 0.05)),
            leak_rate=float(data.get("leak_rate", 1.0)),
            form=str(data.get("form", "input_driven")),
        )


@dataclass(slots=True)
class DigitalStabilityContract:
    """Shared construction/validation contract for adaptive digital kernels."""

    leak_rate: float = 2.0
    dt: float = 0.05
    stability_margin: float = 0.9
    target_ratio: float = 0.8

    def __post_init__(self) -> None:
        if self.leak_rate <= 0.0 or self.dt <= 0.0:
            raise ValueError("leak_rate and dt must be positive")
        if not 0.0 < self.target_ratio <= self.stability_margin <= 1.0:
            raise ValueError("require 0 < target_ratio <= stability_margin <= 1")

    def config(self, *, gamma: Sequence[float], ratio_allocation: Sequence[float]) -> DigitalMemoryConfig:
        rates = _coerce(gamma, name="gamma")
        allocation = _coerce(ratio_allocation, name="ratio_allocation")
        if len(rates) != len(allocation) or any(value < 0.0 for value in allocation):
            raise ValueError("ratio_allocation must be non-negative and match gamma")
        total = sum(allocation)
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("ratio_allocation must sum to one")
        weights = [self.target_ratio * part * rate * self.leak_rate for rate, part in zip(rates, allocation)]
        config = DigitalMemoryConfig(gamma=rates, w=weights, dt=self.dt, leak_rate=self.leak_rate)
        if config.stability_ratio() > self.stability_margin:
            raise RuntimeError("constructed kernel violates the stability contract")
        return config


@dataclass(slots=True)
class DigitalMemoryState:
    u: list[float]
    chi: list[list[float]]
    t: float = 0.0

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)


def seed_memory_state(*, state_dim: int, channel_count: int) -> DigitalMemoryState:
    if state_dim <= 0:
        raise ValueError("state_dim must be positive")
    if channel_count <= 0:
        raise ValueError("channel_count must be positive")
    return DigitalMemoryState(
        u=[0.0 for _ in range(state_dim)],
        chi=[[0.0 for _ in range(state_dim)] for _ in range(channel_count)],
        t=0.0,
    )


def step_memory(
    state: DigitalMemoryState,
    forcing: Sequence[float],
    config: DigitalMemoryConfig,
) -> DigitalMemoryState:
    """Euler step for the digital forcing memory substrate.

    This is a Python-side bridge while the C ABI lacks arbitrary forcing in
    stateful ABERSOE step calls. The forms mirror the stable ABERSOE variants
    used elsewhere in the repo.
    """

    x = _coerce(forcing, name="forcing")
    if len(state.u) != len(x):
        raise ValueError("forcing dimension must match state.u")
    if len(state.chi) != config.channel_count:
        raise ValueError("state channel count must match config")

    next_chi: list[list[float]] = []
    source = x if config.form == "input_driven" else state.u
    for channel, rate in zip(state.chi, config.gamma):
        if len(channel) != len(x):
            raise ValueError("chi channel dimension must match state.u")
        next_chi.append([
            value + config.dt * (-rate * value + source_i)
            for value, source_i in zip(channel, source)
        ])

    memory_feedback = [
        sum(weight * channel[i] for weight, channel in zip(config.w, next_chi))
        for i in range(len(x))
    ]
    if config.form == "input_driven":
        next_u = [
            value + config.dt * (-config.leak_rate * value + x_i + feedback)
            for value, x_i, feedback in zip(state.u, x, memory_feedback)
        ]
    else:
        next_u = [
            value + config.dt * (-config.leak_rate * value - feedback + x_i)
            for value, x_i, feedback in zip(state.u, x, memory_feedback)
        ]

    if any(not math.isfinite(value) for value in next_u):
        raise RuntimeError("digital memory state became non-finite")

    return DigitalMemoryState(u=next_u, chi=next_chi, t=state.t + config.dt)


@dataclass(slots=True)
class ForcingSchedule:
    forcings: list[list[float]]
    tokens: list[str | None]
    metadata: dict[str, object] = field(default_factory=dict)


def build_forcing_schedule(
    stream: DigitalStream,
    bridge: TokenForcingBridge,
    *,
    pulse_steps: int = 1,
    silence_steps: int = 0,
) -> ForcingSchedule:
    if pulse_steps <= 0:
        raise ValueError("pulse_steps must be positive")
    if silence_steps < 0:
        raise ValueError("silence_steps must be non-negative")

    forcings: list[list[float]] = []
    tokens: list[str | None] = []
    for event in stream.events:
        forcing = bridge.forcing_for_token(event.token)
        for _ in range(pulse_steps):
            forcings.append(list(forcing))
            tokens.append(event.token)
        for _ in range(silence_steps):
            forcings.append(bridge.zero_forcing())
            tokens.append(None)
    return ForcingSchedule(
        forcings=forcings,
        tokens=tokens,
        metadata={"pulse_steps": pulse_steps, "silence_steps": silence_steps},
    )
