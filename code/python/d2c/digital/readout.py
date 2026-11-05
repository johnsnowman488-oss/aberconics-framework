"""Readout helpers for digital D2C states."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Sequence

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


@dataclass(slots=True)
class QueryConditionedMLPReadout:
    """Small trainable nonlinear readout over frozen D2C state and a query.

    The model is intentionally dependency-free and uses one ReLU hidden layer,
    making its learning surface explicit for early mechanism experiments.
    """

    input_dim: int
    output_dim: int
    hidden_dim: int = 32
    seed: int = 0
    w1: list[list[float]] = field(init=False)
    b1: list[float] = field(init=False)
    w2: list[list[float]] = field(init=False)
    b2: list[float] = field(init=False)

    def __post_init__(self) -> None:
        if self.input_dim <= 0 or self.output_dim <= 1 or self.hidden_dim <= 0:
            raise ValueError("input_dim/hidden_dim must be positive and output_dim must exceed one")
        rng = random.Random(self.seed)
        scale1 = 1.0 / math.sqrt(self.input_dim)
        scale2 = 1.0 / math.sqrt(self.hidden_dim)
        self.w1 = [[rng.uniform(-scale1, scale1) for _ in range(self.input_dim)] for _ in range(self.hidden_dim)]
        self.b1 = [0.0 for _ in range(self.hidden_dim)]
        self.w2 = [[rng.uniform(-scale2, scale2) for _ in range(self.hidden_dim)] for _ in range(self.output_dim)]
        self.b2 = [0.0 for _ in range(self.output_dim)]

    def _forward(self, features: Sequence[float]) -> tuple[list[float], list[float], list[float]]:
        values = [float(value) for value in features]
        if len(values) != self.input_dim:
            raise ValueError("feature dimension does not match readout input_dim")
        pre_hidden = [bias + sum(weight * value for weight, value in zip(row, values)) for row, bias in zip(self.w1, self.b1)]
        hidden = [max(0.0, value) for value in pre_hidden]
        logits = [bias + sum(weight * value for weight, value in zip(row, hidden)) for row, bias in zip(self.w2, self.b2)]
        return pre_hidden, hidden, logits

    def probabilities(self, features: Sequence[float]) -> list[float]:
        _, _, logits = self._forward(features)
        offset = max(logits)
        exp_logits = [math.exp(value - offset) for value in logits]
        total = sum(exp_logits)
        return [value / total for value in exp_logits]

    def predict(self, features: Sequence[float]) -> tuple[int, float]:
        probabilities = self.probabilities(features)
        index = max(range(len(probabilities)), key=probabilities.__getitem__)
        return index, probabilities[index]

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
        values = [float(value) for value in features]
        pre_hidden, hidden, _ = self._forward(values)
        probabilities = self.probabilities(values)
        loss = -math.log(max(probabilities[target_index], 1e-12))
        output_grad = list(probabilities)
        output_grad[target_index] -= 1.0
        hidden_grad = [sum(output_grad[out] * self.w2[out][unit] for out in range(self.output_dim)) for unit in range(self.hidden_dim)]
        pre_hidden_grad = [gradient if activation > 0.0 else 0.0 for gradient, activation in zip(hidden_grad, pre_hidden)]
        input_gradient = [
            sum(self.w1[unit][index] * gradient for unit, gradient in enumerate(pre_hidden_grad))
            for index in range(self.input_dim)
        ]

        for out in range(self.output_dim):
            for unit in range(self.hidden_dim):
                self.w2[out][unit] -= learning_rate * output_grad[out] * hidden[unit]
            self.b2[out] -= learning_rate * output_grad[out]
        for unit in range(self.hidden_dim):
            for index in range(self.input_dim):
                self.w1[unit][index] -= learning_rate * pre_hidden_grad[unit] * values[index]
            self.b1[unit] -= learning_rate * pre_hidden_grad[unit]
        return loss, input_gradient


class NumpyMLPReadout:
    """Fast numpy-backed MLP readout for bAbI experiments.

    Same architecture as QueryConditionedMLPReadout (one ReLU hidden layer)
    but uses numpy vectorised operations for ~50x speedup on the forward and
    backward passes.  Accepts np.ndarray features directly.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128,
                 seed: int = 0, lr: float = 0.01):
        import numpy as np
        rng = np.random.RandomState(seed)
        scale1 = 1.0 / np.sqrt(input_dim)
        scale2 = 1.0 / np.sqrt(hidden_dim)
        self.w1 = rng.randn(hidden_dim, input_dim).astype(np.float32) * scale1
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.w2 = rng.randn(output_dim, hidden_dim).astype(np.float32) * scale2
        self.b2 = np.zeros(output_dim, dtype=np.float32)
        self.lr = lr

    def _forward(self, x):
        import numpy as np
        self._last_x = x
        self._pre_h = self.w1 @ x + self.b1
        self._h = np.maximum(0, self._pre_h)
        logits = self.w2 @ self._h + self.b2
        # softmax
        logits -= logits.max()
        exp_l = np.exp(logits)
        probs = exp_l / exp_l.sum()
        self._last_probs = probs
        return probs

    def train_step(self, features, target_index: int):
        import numpy as np
        if isinstance(features, list):
            features = np.array(features, dtype=np.float32)
        probs = self._forward(features)
        loss = -np.log(max(probs[target_index], 1e-12))

        # Backward
        grad_logits = probs.copy()
        grad_logits[target_index] -= 1.0  # softmax + cross-entropy gradient

        # w2, b2
        grad_w2 = np.outer(grad_logits, self._h)
        grad_b2 = grad_logits

        # Hidden
        grad_h = self.w2.T @ grad_logits
        grad_pre_h = grad_h * (self._pre_h > 0).astype(np.float32)

        # w1, b1
        grad_w1 = np.outer(grad_pre_h, self._last_x)
        grad_b1 = grad_pre_h

        # SGD update
        self.w2 -= self.lr * grad_w2
        self.b2 -= self.lr * grad_b2
        self.w1 -= self.lr * grad_w1
        self.b1 -= self.lr * grad_b1
        return loss

    def predict(self, features):
        import numpy as np
        if isinstance(features, list):
            features = np.array(features, dtype=np.float32)
        probs = self._forward(features)
        idx = int(np.argmax(probs))
        return idx, float(probs[idx])

    def probabilities(self, features):
        import numpy as np
        if isinstance(features, list):
            features = np.array(features, dtype=np.float32)
        return self._forward(features).tolist()


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

    memory = combined_readout_vector(state, config)
    query_values = [float(value) for value in query]
    # Fixed scaling improves optimisation without erasing evidence magnitude.
    # Per-episode normalization would turn vanishing residual traces into a
    # full-strength feature and invalidate long-gap ablations.
    normalized_memory = [100.0 * value for value in memory]
    gated = [query_value * memory_value for query_value in query_values for memory_value in normalized_memory]
    return [*normalized_memory, *query_values, *gated]


def query_conditioned_features_np(
    state: DigitalMemoryState,
    config: DigitalMemoryConfig,
    query,
):
    """NumPy version of query_conditioned_features — returns np.ndarray."""
    import numpy as np
    memory = combined_readout_vector(state, config)
    query_values = [float(v) for v in query]
    norm_mem = [100.0 * v for v in memory]
    gated = [q * m for q in query_values for m in norm_mem]
    return np.array([*norm_mem, *query_values, *gated], dtype=np.float32)
