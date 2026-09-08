"""Three-factor kernel update proposals for offline D2C learning passes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Sequence


def _coerce_float_list(values: Sequence[float], *, name: str) -> list[float]:
    out = [float(value) for value in values]
    if not out:
        raise ValueError(f"{name} must not be empty")
    return out


def _mean_abs(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(abs(float(value)) for value in values) / len(values)


@dataclass(slots=True)
class ThreeFactorUpdateConfig:
    """Hyperparameters for a stable offline three-factor weight proposal."""

    eta_pred: float = 0.01
    eta_val: float = 0.01
    decay: float = 0.001
    dt: float = 1.0
    min_weight: float = 0.0
    max_weight: float | None = None
    leak_rate: float = 1.0
    stability_margin: float = 0.9
    signed_hebbian: bool = False
    error_modulated_decay: bool = False

    def __post_init__(self) -> None:
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.leak_rate <= 0.0:
            raise ValueError("leak_rate must be positive")
        if not 0.0 < self.stability_margin <= 1.0:
            raise ValueError("stability_margin must be in (0, 1]")
        if self.min_weight < 0.0:
            raise ValueError("min_weight must be non-negative")
        if self.max_weight is not None and self.max_weight < self.min_weight:
            raise ValueError("max_weight must be >= min_weight")

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class KernelUpdateProposal:
    """A proposed kernel-weight update plus stability diagnostics."""

    old_weights: list[float]
    proposed_weights: list[float]
    clipped_weights: list[float]
    delta: list[float]
    stability_ratio_before: float
    stability_ratio_after: float
    stability_rescaled: bool
    notes: list[str] = field(default_factory=list)

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)


def stability_ratio(weights: Sequence[float], gamma: Sequence[float], leak_rate: float) -> float:
    """Compute the Formulation-B stability ratio sum(w / (gamma * leak))."""

    rates = _coerce_float_list(gamma, name="gamma")
    values = _coerce_float_list(weights, name="weights")
    if len(rates) != len(values):
        raise ValueError("weights and gamma must have matching lengths")
    if leak_rate <= 0.0:
        raise ValueError("leak_rate must be positive")
    if any(rate <= 0.0 for rate in rates):
        raise ValueError("gamma values must be positive")
    return sum(weight / (rate * leak_rate) for weight, rate in zip(values, rates))


def propose_three_factor_update(
    *,
    weights: Sequence[float],
    gamma: Sequence[float],
    channel_activity: Sequence[float],
    prediction_error: Sequence[float] | float,
    td_error: Sequence[float] | float | None = None,
    hebbian_signal: Sequence[float] | None = None,
    config: ThreeFactorUpdateConfig | None = None,
) -> KernelUpdateProposal:
    """Propose non-negative kernel weights from prediction and value signals.

    This is intentionally side-effect free. Later runtime code can decide whether
    to apply the proposal through a step-level or writable-kernel ABI surface.
    """

    cfg = config or ThreeFactorUpdateConfig()
    old = _coerce_float_list(weights, name="weights")
    rates = _coerce_float_list(gamma, name="gamma")
    activity = _coerce_float_list(channel_activity, name="channel_activity")
    if len(old) != len(rates) or len(old) != len(activity):
        raise ValueError("weights, gamma, and channel_activity must have matching lengths")

    if isinstance(prediction_error, (float, int)):
        eps = float(prediction_error)
        pred_signal = abs(eps)
    else:
        errors = _coerce_float_list(prediction_error, name="prediction_error")
        eps = sum(errors) / len(errors)
        pred_signal = _mean_abs(prediction_error)

    if hebbian_signal is not None:
        hebb_signals = _coerce_float_list(hebbian_signal, name="hebbian_signal")
        if len(hebb_signals) != len(old):
            raise ValueError("hebbian_signal must match weights length")
    elif cfg.signed_hebbian:
        # Hebbian term chi_l * eps: signed per-channel correlation with the
        # prediction error, so anticorrelated channels are weakened.
        hebb_signals = [value * eps for value in activity]
    else:
        hebb_signals = [abs(value) * pred_signal for value in activity]

    if td_error is None:
        td_signals = [0.0 for _ in old]
    elif isinstance(td_error, (float, int)):
        td_signals = [float(td_error) for _ in old]
    else:
        td_signals = _coerce_float_list(td_error, name="td_error")
        if len(td_signals) != len(old):
            raise ValueError("td_error must match weights length")

    # Error-modulated decay (D2C reference Eq. 33): forget faster when
    # predictions are accurate and slower when errors are large.
    decay = cfg.decay / (1.0 + pred_signal) if cfg.error_modulated_decay else cfg.decay

    proposed: list[float] = []
    deltas: list[float] = []
    for weight, h_signal, td_value in zip(old, hebb_signals, td_signals):
        delta = cfg.dt * (
            cfg.eta_pred * h_signal
            + cfg.eta_val * td_value
            - decay * weight
        )
        deltas.append(delta)
        proposed.append(weight + delta)

    clipped = [max(cfg.min_weight, value) for value in proposed]
    if cfg.max_weight is not None:
        clipped = [min(cfg.max_weight, value) for value in clipped]

    before = stability_ratio(old, rates, cfg.leak_rate)
    after = stability_ratio(clipped, rates, cfg.leak_rate)
    rescaled = False
    notes: list[str] = []
    if cfg.signed_hebbian and hebbian_signal is None:
        notes.append("signed hebbian signal (chi * eps)")
    if cfg.error_modulated_decay:
        notes.append(f"error-modulated decay beta={decay:.6g}")
    if after > cfg.stability_margin:
        scale = cfg.stability_margin / after
        clipped = [value * scale for value in clipped]
        after = stability_ratio(clipped, rates, cfg.leak_rate)
        rescaled = True
        notes.append("weights rescaled to satisfy stability margin")

    return KernelUpdateProposal(
        old_weights=old,
        proposed_weights=proposed,
        clipped_weights=clipped,
        delta=deltas,
        stability_ratio_before=before,
        stability_ratio_after=after,
        stability_rescaled=rescaled,
        notes=notes,
    )
