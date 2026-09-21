"""Readout helpers for digital D2C states."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Sequence

import numpy as np

from .bridge import DigitalMemoryConfig, DigitalMemoryState, KeyValueBindingBridge
from .tokens import Vocabulary, nearest_token


def memory_readout_vector(state: DigitalMemoryState, config: DigitalMemoryConfig) -> list[float]:
    """Weighted memory-channel readout vector."""

    if not state.chi:
        return list(state.u)
    return [
        sum(weight * channel[i] for weight, channel in zip(config.w, state.chi))
        for i in range(len(state.u))
    ]


def combined_readout_vector(
    state: DigitalMemoryState,
    config: DigitalMemoryConfig,
    *,
    state_weight: float = 0.25,
    memory_weight: float = 1.0,
) -> list[float]:
    memory = memory_readout_vector(state, config)
    return [
        state_weight * u_value + memory_weight * memory_value
        for u_value, memory_value in zip(state.u, memory)
    ]


def decode_state(
    state: DigitalMemoryState,
    config: DigitalMemoryConfig,
    vocabulary: Vocabulary,
    *,
    candidates: Sequence[str] | None = None,
    seed: int = 0,
    code_mode: str = "dense",
) -> tuple[str, float]:
    vector = combined_readout_vector(state, config)
    return nearest_token(vector, vocabulary, candidates=candidates, dim=len(vector), seed=seed, code_mode=code_mode)


def decode_key_value_binding(
    state: DigitalMemoryState,
    config: DigitalMemoryConfig,
    binding: KeyValueBindingBridge,
    query_key: str,
    minimum_score: float = 0.0,
) -> tuple[str, float]:
    """Decode the strongest value slot associated with ``query_key``."""

    try:
        key_index = binding.key_tokens.index(query_key)
    except ValueError as exc:
        raise ValueError("query_key must be in the binding bridge") from exc
    vector = combined_readout_vector(state, config)
    if len(vector) != binding.state_dim:
        raise ValueError("binding state dimension must match the memory state")
    start = key_index * len(binding.value_tokens)
    scores = vector[start : start + len(binding.value_tokens)]
    best_index = max(range(len(scores)), key=scores.__getitem__)
    score = scores[best_index]
    if score < minimum_score:
        return "<insufficient_evidence>", score
    return binding.value_tokens[best_index], score


# ---------------------------------------------------------------------------
# NumPy-backed softmax helper
# ---------------------------------------------------------------------------


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


# ---------------------------------------------------------------------------
# Numpy-ized MLP readout
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class QueryConditionedMLPReadout:
    """Small trainable nonlinear readout over frozen D2C state and a query.

    Weights are stored as numpy arrays for performance.  The public interface
    accepts ``list[float]`` features and returns Python scalars, so downstream
    code (D3, D4A, temporal_logic experiments) is unaffected.
    """

    input_dim: int
    output_dim: int
    hidden_dim: int = 32
    seed: int = 0
    w1: np.ndarray = field(init=False)
    b1: np.ndarray = field(init=False)
    w2: np.ndarray = field(init=False)
    b2: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if self.input_dim <= 0 or self.output_dim <= 1 or self.hidden_dim <= 0:
            raise ValueError("input_dim/hidden_dim must be positive and output_dim must exceed one")
        rng = np.random.RandomState(self.seed)
        scale1 = 1.0 / math.sqrt(self.input_dim)
        scale2 = 1.0 / math.sqrt(self.hidden_dim)
        self.w1 = rng.uniform(-scale1, scale1, (self.hidden_dim, self.input_dim))
        self.b1 = np.zeros(self.hidden_dim)
        self.w2 = rng.uniform(-scale2, scale2, (self.output_dim, self.hidden_dim))
        self.b2 = np.zeros(self.output_dim)

    def _forward(self, features: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.asarray(features, dtype=np.float64)
        if len(x) != self.input_dim:
            raise ValueError("feature dimension does not match readout input_dim")
        pre_hidden = self.w1 @ x + self.b1
        hidden = np.maximum(0.0, pre_hidden)
        logits = self.w2 @ hidden + self.b2
        return pre_hidden, hidden, logits

    def probabilities(self, features: Sequence[float]) -> list[float]:
        _, _, logits = self._forward(features)
        return _softmax(logits).tolist()

    def predict(self, features: Sequence[float]) -> tuple[int, float]:
        _, _, logits = self._forward(features)
        probs = _softmax(logits)
        index = int(np.argmax(probs))
        return index, float(probs[index])

    def train_step(self, features: Sequence[float], target_index: int, *, learning_rate: float = 0.05) -> float:
        loss, _ = self.train_step_with_input_gradient(
            features, target_index, learning_rate=learning_rate,
        )
        return loss

    def train_step_with_input_gradient(
        self,
        features: Sequence[float],
        target_index: int,
        *,
        learning_rate: float = 0.05,
    ) -> tuple[float, list[float]]:
        """Train once and return the pre-update loss gradient at the input."""
        if not 0 <= target_index < self.output_dim:
            raise ValueError("target_index out of range")
        x = np.asarray(features, dtype=np.float64)
        pre_hidden, hidden, _ = self._forward(x)
        probs = _softmax(self.w2 @ hidden + self.b2)
        loss = -math.log(max(float(probs[target_index]), 1e-12))

        # Backward pass
        output_grad = probs.copy()
        output_grad[target_index] -= 1.0
        hidden_grad = self.w2.T @ output_grad
        pre_hidden_grad = hidden_grad * (pre_hidden > 0.0).astype(np.float64)
        input_gradient = self.w1.T @ pre_hidden_grad

        # Weight updates
        lr = learning_rate
        self.w2 -= lr * np.outer(output_grad, hidden)
        self.b2 -= lr * output_grad
        self.w1 -= lr * np.outer(pre_hidden_grad, x)
        self.b1 -= lr * pre_hidden_grad

        return loss, input_gradient.tolist()


def query_conditioned_features(
    state: DigitalMemoryState,
    config: DigitalMemoryConfig,
    query: Sequence[float],
) -> list[float]:
    """Build normalized state, query, and query-gated state features.

    The gate supplies the multiplicative interaction required for selecting a
    query-specific slice of a distributed memory state. Its downstream mapping
    remains entirely trainable by ``QueryConditionedMLPReadout``.
    """

    memory = np.array(combined_readout_vector(state, config), dtype=np.float64)
    query_arr = np.asarray(query, dtype=np.float64)
    normalized_memory = memory * 100.0
    gated = np.outer(query_arr, normalized_memory).ravel()
    return np.concatenate([normalized_memory, query_arr, gated]).tolist()


# ---------------------------------------------------------------------------
# Linear readout (Path B — fastest, most interpretable)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class LinearReadout:
    """Single-layer linear readout: features @ W -> softmax -> argmax.

    Faster than the MLP, directly interpretable, and sufficient for E0.
    Accepts ``list[float]`` features; returns Python scalars.
    """

    input_dim: int
    output_dim: int
    seed: int = 0
    W: np.ndarray = field(init=False)
    b: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if self.input_dim <= 0 or self.output_dim <= 1:
            raise ValueError("input_dim must be positive and output_dim must exceed one")
        rng = np.random.RandomState(self.seed)
        scale = 1.0 / math.sqrt(self.input_dim)
        self.W = rng.uniform(-scale, scale, (self.output_dim, self.input_dim))
        self.b = np.zeros(self.output_dim)

    def predict(self, features: Sequence[float]) -> tuple[int, float]:
        x = np.asarray(features, dtype=np.float64)
        logits = self.W @ x + self.b
        probs = _softmax(logits)
        idx = int(np.argmax(probs))
        return idx, float(probs[idx])

    def train_step(self, features: Sequence[float], target_index: int, *, learning_rate: float = 0.05) -> float:
        if not 0 <= target_index < self.output_dim:
            raise ValueError("target_index out of range")
        x = np.asarray(features, dtype=np.float64)
        logits = self.W @ x + self.b
        probs = _softmax(logits)
        loss = -math.log(max(float(probs[target_index]), 1e-12))

        grad = probs.copy()
        grad[target_index] -= 1.0
        self.W -= learning_rate * np.outer(grad, x)
        self.b -= learning_rate * grad
        return loss

    def train_step_with_input_gradient(
        self,
        features: Sequence[float],
        target_index: int,
        *,
        learning_rate: float = 0.05,
    ) -> tuple[float, list[float]]:
        """Train once and return loss + gradient at input (for eligibility)."""
        if not 0 <= target_index < self.output_dim:
            raise ValueError("target_index out of range")
        x = np.asarray(features, dtype=np.float64)
        logits = self.W @ x + self.b
        probs = _softmax(logits)
        loss = -math.log(max(float(probs[target_index]), 1e-12))

        grad = probs.copy()
        grad[target_index] -= 1.0
        input_gradient = self.W.T @ grad

        self.W -= learning_rate * np.outer(grad, x)
        self.b -= learning_rate * grad
        return loss, input_gradient.tolist()
