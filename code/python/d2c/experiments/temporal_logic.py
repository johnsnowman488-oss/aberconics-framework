"""D3 delayed-feedback temporal-rule experiment over frozen SOE memory."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import math
import random
import sys
from pathlib import Path

if __package__ in {None, ""}:
    python_root = Path(__file__).resolve().parents[2]
    if str(python_root) not in sys.path:
        sys.path.insert(0, str(python_root))

try:
    from ..digital import (
        DigitalDirector,
        DigitalEpisode,
        DigitalMemoryState,
        DigitalStabilityContract,
        QueryConditionedMLPReadout,
        TokenForcingBridge,
        Vocabulary,
        combined_readout_vector,
        query_conditioned_features,
    )
    from .symbolic_induction import memory_config_for_variant
except ImportError:  # pragma: no cover - direct script execution fallback
    from d2c.digital import DigitalDirector, DigitalEpisode, DigitalMemoryState, DigitalStabilityContract, QueryConditionedMLPReadout, TokenForcingBridge, Vocabulary, combined_readout_vector, query_conditioned_features
    from d2c.experiments.symbolic_induction import memory_config_for_variant


_CASES = (
    ("OPEN", "KEY", "UNLOCK"),
    ("OPEN", "LOCK", "KEEP_LOCKED"),
    ("CLOSED", "KEY", "KNOCK"),
    ("CLOSED", "LOCK", "WAIT"),
)
_ACTIONS = ["UNLOCK", "KEEP_LOCKED", "KNOCK", "WAIT"]
_ADAPTIVE_STABILITY_CONTRACT = DigitalStabilityContract()


def adaptive_memory_config_for_variant(variant: str):
    """Matched adaptive kernels: same total stability budget, distinct horizons."""

    if variant == "full":
        return _ADAPTIVE_STABILITY_CONTRACT.config(
            gamma=[2.0, 0.45, 0.035], ratio_allocation=[0.05, 0.15, 0.80],
        )
    if variant == "no_slow":
        return _ADAPTIVE_STABILITY_CONTRACT.config(
            gamma=[2.0, 0.9], ratio_allocation=[0.25, 0.75],
        )
    if variant == "collapsed_gamma":
        return _ADAPTIVE_STABILITY_CONTRACT.config(
            gamma=[0.6, 0.6, 0.6], ratio_allocation=[0.2, 0.3, 0.5],
        )
    raise ValueError(f"unsupported adaptive variant: {variant}")


@dataclass(slots=True)
class TemporalLogicConfig:
    variant: str = "full"
    credit_enabled: bool = True
    gap: int = 32
    train_count: int = 128
    test_count: int = 64
    epochs: int = 30
    hidden_dim: int = 32
    learning_rate: float = 0.01
    pulse_steps: int = 2
    silence_steps: int = 1
    forced_distractors: bool = False
    task_conditioned_eligibility: bool = False
    learned_eligibility: bool = False
    soft_eligibility: bool = False
    nonlinear_eligibility: bool = False
    predictive_credit_eligibility: bool = False
    persistent_predictive_credit_eligibility: bool = False
    eligibility_learning_rate: float = 0.03
    predictive_learning_rate: float = 0.02
    compatibility_hidden_dim: int = 8
    compatibility_parameter_budget: int = 384
    readout_consolidation_epochs: int = 20
    adaptive_kernel: bool = False
    critic_learning_rate: float = 0.05
    consolidation_interval: int = 16
    consolidation_rate: float = 0.05
    seed: int = 41

    def __post_init__(self) -> None:
        if self.variant not in {"full", "no_slow", "collapsed_gamma"}:
            raise ValueError("unsupported memory variant")
        if min(self.gap, self.train_count, self.test_count, self.epochs, self.hidden_dim) <= 0:
            raise ValueError("gap and count/dimension values must be positive")
        if self.learning_rate <= 0.0 or self.pulse_steps <= 0 or self.silence_steps < 0:
            raise ValueError("invalid learning or schedule settings")
        if sum((
            self.task_conditioned_eligibility,
            self.learned_eligibility,
            self.soft_eligibility,
            self.nonlinear_eligibility,
            self.predictive_credit_eligibility,
            self.persistent_predictive_credit_eligibility,
        )) > 1:
            raise ValueError("eligibility modes are mutually exclusive")
        if self.eligibility_learning_rate <= 0.0:
            raise ValueError("eligibility_learning_rate must be positive")
        if self.predictive_learning_rate <= 0.0:
            raise ValueError("predictive_learning_rate must be positive")
        if self.compatibility_hidden_dim <= 0 or self.compatibility_parameter_budget <= 0:
            raise ValueError("compatibility hidden dimension and parameter budget must be positive")
        if self.readout_consolidation_epochs < 0:
            raise ValueError("readout_consolidation_epochs must be non-negative")
        if self.critic_learning_rate <= 0.0 or self.consolidation_interval <= 0:
            raise ValueError("invalid critic or consolidation settings")
        if not 0.0 <= self.consolidation_rate <= 1.0:
            raise ValueError("consolidation_rate must be in [0, 1]")

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)


def _vocabulary() -> Vocabulary:
    return Vocabulary(["OPEN", "CLOSED", "KEY", "LOCK", "FILLER", "DISTRACTOR_A", "DISTRACTOR_B"])


def _episode(config: TemporalLogicConfig, index: int, bridge: TokenForcingBridge) -> tuple[DigitalEpisode, int]:
    premise, trigger, action = _CASES[index % len(_CASES)]
    gap_tokens = [
        ("DISTRACTOR_A" if position % 2 == 0 else "DISTRACTOR_B")
        if config.forced_distractors else "FILLER"
        for position in range(config.gap)
    ]
    tokens = [premise, *gap_tokens, trigger]
    forcings: list[list[float]] = []
    trace_tokens: list[str | None] = []
    for token in tokens:
        # D3's neutral control has trace-visible but zero-forcing filler.
        # Forced-distractor runs use distinct competing token forcings.
        forcing = bridge.zero_forcing() if token == "FILLER" else bridge.forcing_for_token(token)
        forcings.extend([forcing] * config.pulse_steps)
        trace_tokens.extend([token] * config.pulse_steps)
        forcings.extend([bridge.zero_forcing()] * config.silence_steps)
        trace_tokens.extend([None] * config.silence_steps)
    query = bridge.forcing_for_token(trigger)
    return DigitalEpisode(
        forcings=forcings,
        tokens=trace_tokens,
        target=action,
        query=query,
        metadata={"example_index": index, "premise": premise, "trigger": trigger, "gap": config.gap, "action": action},
    ), _ACTIONS.index(action)


def _channel_activity(channels: list[list[float]]) -> list[float]:
    return [sum(abs(value) for value in channel) / len(channel) for channel in channels]


def _event_trace_credit(
    before_channels: list[list[float]],
    after_channels: list[list[float]],
    td_errors: list[float],
) -> float:
    """Eligibility-style terminal credit assigned to one event transition."""

    return sum(
        abs(td_error) * sum(abs(after - before) for before, after in zip(before_channel, after_channel)) / len(after_channel)
        for before_channel, after_channel, td_error in zip(before_channels, after_channels, td_errors)
    )


def _features_for_result(result, memory, *, task_conditioned_eligibility: bool) -> list[float]:
    """Readout features with an optional task-gated premise eligibility trace.

    The gate is an explicit first control: it is supplied by the task's
    premise-token contract, not learned from reward. It snapshots only the
    existing continuous D2C state when a premise event arrives; distractor
    states never enter this eligibility feature.
    """

    features = query_conditioned_features(result.final_state, memory, result.query)
    if not task_conditioned_eligibility:
        return features
    premise_step = next(step for step in reversed(result.trace.steps) if step.token in {"OPEN", "CLOSED"})
    return _features_with_eligibility_snapshot(result, memory, premise_step)


def _features_with_eligibility_snapshot(result, memory, eligibility_step) -> list[float]:
    """Append one selected augmented-state snapshot to standard readout features."""

    features = query_conditioned_features(result.final_state, memory, result.query)
    # Reconstruct the state shape used by the readout without storing a second
    # mutable memory system in the task. This is equivalent to latching the
    # selected step's augmented state online.
    premise_state = DigitalMemoryState(
        u=list(eligibility_step.u), chi=[list(channel) for channel in eligibility_step.chi], t=eligibility_step.t,
    )
    eligibility = [100.0 * value for value in combined_readout_vector(premise_state, memory)]
    gated_eligibility = [query_value * value for query_value in result.query for value in eligibility]
    return [*features, *eligibility, *gated_eligibility]


def _features_with_eligibility_vector(result, memory, eligibility: list[float]) -> list[float]:
    """Append a supplied continuous eligibility trace to standard features."""

    features = query_conditioned_features(result.final_state, memory, result.query)
    gated_eligibility = [query_value * value for query_value in result.query for value in eligibility]
    return [*features, *eligibility, *gated_eligibility]


@dataclass(slots=True)
class SoftEligibilityTrace:
    """Query- and state-conditioned differentiable attention over snapshots.

    Each candidate's logit combines the terminal query, its token identity,
    and a normalized D2C state descriptor captured at that event. This keeps
    the selector causal with respect to the completed episode while avoiding
    a single global ``OPEN``/``CLOSED`` preference shared by every query.
    """

    vocabulary: Vocabulary
    learning_rate: float
    query_token_scores: list[list[float]] = field(init=False)
    query_state_weights: list[list[float]] = field(init=False)

    def __post_init__(self) -> None:
        size = len(self.vocabulary.tokens)
        self.query_token_scores = [[0.0 for _ in range(size)] for _ in range(size)]
        self.query_state_weights = []

    def _ensure_state_weights(self, state_dim: int) -> None:
        if not self.query_state_weights:
            self.query_state_weights = [
                [0.0 for _ in range(state_dim)] for _ in self.vocabulary.tokens
            ]
        elif len(self.query_state_weights[0]) != state_dim:
            raise ValueError("eligibility state dimension changed within one experiment")

    @staticmethod
    def _state_descriptor(snapshot: list[float]) -> list[float]:
        """Bound state-content influence so attention updates remain stable."""

        norm = math.sqrt(sum(value * value for value in snapshot))
        return [value / max(norm, 1.0) for value in snapshot]

    def _candidate_indices(self, result) -> list[int]:
        # Retain one latest snapshot per token identity. This prevents a
        # repeated distractor from winning simply through event multiplicity.
        latest: dict[str, int] = {}
        for index, step in enumerate(result.trace.steps):
            if step.token is not None:
                latest[step.token] = index
        return list(latest.values())

    def _logit(self, query: list[float], token_id: int, snapshot: list[float]) -> float:
        descriptor = self._state_descriptor(snapshot)
        return sum(
            query_value * (
                self.query_token_scores[query_index][token_id]
                + sum(weight * value for weight, value in zip(self.query_state_weights[query_index], descriptor))
            )
            for query_index, query_value in enumerate(query)
        )

    def features(self, result, memory) -> tuple[list[float], list[int], list[float], list[list[float]]]:
        candidates = self._candidate_indices(result)
        snapshots: list[list[float]] = []
        for index in candidates:
            step = result.trace.steps[index]
            state = DigitalMemoryState(u=list(step.u), chi=[list(channel) for channel in step.chi], t=step.t)
            snapshots.append([100.0 * value for value in combined_readout_vector(state, memory)])
        self._ensure_state_weights(len(snapshots[0]))
        logits = [
            self._logit(result.query, self.vocabulary.token_id(result.trace.steps[index].token), snapshot)
            for index, snapshot in zip(candidates, snapshots)
        ]
        offset = max(logits)
        unnormalized = [math.exp(value - offset) for value in logits]
        total = sum(unnormalized)
        attention = [value / total for value in unnormalized]
        eligibility = [
            sum(weight * snapshot[dimension] for weight, snapshot in zip(attention, snapshots))
            for dimension in range(len(snapshots[0]))
        ]
        return _features_with_eligibility_vector(result, memory, eligibility), candidates, attention, snapshots

    def update(
        self,
        result,
        *,
        candidates: list[int],
        attention: list[float],
        snapshots: list[list[float]],
        input_gradient: list[float],
    ) -> None:
        """Backpropagate terminal action loss into conditional attention logits."""

        eligibility_dim = len(snapshots[0])
        base_dim = len(input_gradient) - eligibility_dim * (1 + len(result.query))
        eligibility_gradient = list(input_gradient[base_dim : base_dim + eligibility_dim])
        gated_start = base_dim + eligibility_dim
        for query_index, query_value in enumerate(result.query):
            offset = gated_start + query_index * eligibility_dim
            for dimension in range(eligibility_dim):
                eligibility_gradient[dimension] += query_value * input_gradient[offset + dimension]
        event_gradients = [
            sum(gradient * value for gradient, value in zip(eligibility_gradient, snapshot))
            for snapshot in snapshots
        ]
        expected = sum(weight * gradient for weight, gradient in zip(attention, event_gradients))
        for index, weight, gradient, snapshot in zip(candidates, attention, event_gradients, snapshots):
            token_id = self.vocabulary.token_id(result.trace.steps[index].token)
            logit_gradient = weight * (gradient - expected)
            descriptor = self._state_descriptor(snapshot)
            for query_index, query_value in enumerate(result.query):
                update = self.learning_rate * query_value * logit_gradient
                self.query_token_scores[query_index][token_id] -= update
                for dimension, value in enumerate(descriptor):
                    self.query_state_weights[query_index][dimension] -= update * value

    def score_diagnostics(self) -> dict[str, object]:
        """Expose the two terminal queries and their learned selector weights."""

        query_tokens = [token for token in ("KEY", "LOCK") if token in self.vocabulary.tokens]
        return {
            "scorer": "terminal_query_and_event_state_conditioned_linear_attention",
            "candidate_mode": "latest_per_token_identity",
            "query_token_scores": {
                query_token: dict(zip(
                    self.vocabulary.tokens,
                    self.query_token_scores[self.vocabulary.token_id(query_token)],
                ))
                for query_token in query_tokens
            },
            "query_state_weight_l2": {
                query_token: math.sqrt(sum(
                    weight * weight for weight in self.query_state_weights[self.vocabulary.token_id(query_token)]
                ))
                for query_token in query_tokens
            },
        }

    def premise_attention(self, result, memory) -> float:
        _, candidates, attention, _ = self.features(result, memory)
        return sum(
            weight
            for index, weight in zip(candidates, attention)
            if result.trace.steps[index].token in {"OPEN", "CLOSED"}
        )


@dataclass(slots=True)
class NonlinearCompatibilityEligibilityTrace:
    """Compact query-event MLP attention over one snapshot per token event."""

    vocabulary: Vocabulary
    learning_rate: float
    hidden_dim: int
    parameter_budget: int
    seed: int
    w1: list[list[float]] = field(init=False)
    b1: list[float] = field(init=False)
    w2: list[float] = field(init=False)
    b2: float = field(init=False)
    _input_dim: int | None = field(init=False, default=None)
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.w1 = []
        self.b1 = []
        self.w2 = []
        self.b2 = 0.0
        self._rng = random.Random(self.seed)

    def _candidate_indices(self, result) -> list[int]:
        """Keep the completed pulse for every token occurrence, including repeats."""

        return [
            index
            for index, step in enumerate(result.trace.steps)
            if step.token is not None
            and (index + 1 == len(result.trace.steps) or result.trace.steps[index + 1].token != step.token)
        ]

    @staticmethod
    def _state_descriptor(snapshot: list[float]) -> list[float]:
        norm = math.sqrt(sum(value * value for value in snapshot))
        return [value / max(norm, 1.0) for value in snapshot]

    def _input(self, query: list[float], token_id: int, snapshot: list[float]) -> list[float]:
        token_code = [0.0 for _ in self.vocabulary.tokens]
        token_code[token_id] = 1.0
        return [*query, *token_code, *self._state_descriptor(snapshot)]

    def _initialize(self, input_dim: int) -> None:
        if self._input_dim is not None:
            if self._input_dim != input_dim:
                raise ValueError("compatibility input dimension changed within one experiment")
            return
        parameter_count = self.hidden_dim * input_dim + self.hidden_dim + self.hidden_dim + 1
        if parameter_count > self.parameter_budget:
            raise ValueError(
                f"compatibility head needs {parameter_count} parameters, exceeding budget {self.parameter_budget}"
            )
        scale = 1.0 / math.sqrt(input_dim)
        self.w1 = [[self._rng.uniform(-scale, scale) for _ in range(input_dim)] for _ in range(self.hidden_dim)]
        self.b1 = [0.0 for _ in range(self.hidden_dim)]
        self.w2 = [self._rng.uniform(-1.0 / math.sqrt(self.hidden_dim), 1.0 / math.sqrt(self.hidden_dim)) for _ in range(self.hidden_dim)]
        self._input_dim = input_dim

    def _forward(self, values: list[float]) -> tuple[list[float], float]:
        self._initialize(len(values))
        hidden = [max(0.0, sum(weight * value for weight, value in zip(row, values)) + bias)
                  for row, bias in zip(self.w1, self.b1)]
        return hidden, sum(weight * value for weight, value in zip(self.w2, hidden)) + self.b2

    def features(self, result, memory) -> tuple[list[float], list[int], list[float], list[list[float]]]:
        candidates = self._candidate_indices(result)
        snapshots: list[list[float]] = []
        inputs: list[list[float]] = []
        for index in candidates:
            step = result.trace.steps[index]
            state = DigitalMemoryState(u=list(step.u), chi=[list(channel) for channel in step.chi], t=step.t)
            snapshot = [100.0 * value for value in combined_readout_vector(state, memory)]
            snapshots.append(snapshot)
            inputs.append(self._input(result.query, self.vocabulary.token_id(step.token), snapshot))
        logits = [self._forward(values)[1] for values in inputs]
        offset = max(logits)
        unnormalized = [math.exp(value - offset) for value in logits]
        total = sum(unnormalized)
        attention = [value / total for value in unnormalized]
        eligibility = [
            sum(weight * snapshot[dimension] for weight, snapshot in zip(attention, snapshots))
            for dimension in range(len(snapshots[0]))
        ]
        return _features_with_eligibility_vector(result, memory, eligibility), candidates, attention, snapshots

    def update(self, result, *, candidates, attention, snapshots, input_gradient) -> None:
        eligibility_dim = len(snapshots[0])
        base_dim = len(input_gradient) - eligibility_dim * (1 + len(result.query))
        eligibility_gradient = list(input_gradient[base_dim : base_dim + eligibility_dim])
        gated_start = base_dim + eligibility_dim
        for query_index, query_value in enumerate(result.query):
            offset = gated_start + query_index * eligibility_dim
            for dimension in range(eligibility_dim):
                eligibility_gradient[dimension] += query_value * input_gradient[offset + dimension]
        event_gradients = [
            sum(gradient * value for gradient, value in zip(eligibility_gradient, snapshot))
            for snapshot in snapshots
        ]
        expected = sum(weight * gradient for weight, gradient in zip(attention, event_gradients))
        for index, weight, gradient, snapshot in zip(candidates, attention, event_gradients, snapshots):
            values = self._input(result.query, self.vocabulary.token_id(result.trace.steps[index].token), snapshot)
            hidden, _ = self._forward(values)
            logit_gradient = weight * (gradient - expected)
            output_weights = list(self.w2)
            for hidden_index, hidden_value in enumerate(hidden):
                self.w2[hidden_index] -= self.learning_rate * logit_gradient * hidden_value
            self.b2 -= self.learning_rate * logit_gradient
            for hidden_index, hidden_value in enumerate(hidden):
                hidden_gradient = logit_gradient * output_weights[hidden_index] * (1.0 if hidden_value > 0.0 else 0.0)
                for input_index, input_value in enumerate(values):
                    self.w1[hidden_index][input_index] -= self.learning_rate * hidden_gradient * input_value
                self.b1[hidden_index] -= self.learning_rate * hidden_gradient

    def score_diagnostics(self) -> dict[str, object]:
        parameter_count = 0 if self._input_dim is None else self.hidden_dim * self._input_dim + 2 * self.hidden_dim + 1
        return {
            "scorer": "terminal_query_event_state_nonlinear_compatibility_mlp",
            "candidate_mode": "per_completed_token_event",
            "parameter_count": parameter_count,
            "parameter_budget": self.parameter_budget,
            "hidden_dim": self.hidden_dim,
        }


@dataclass(slots=True)
class PredictiveCreditEligibilityTrace:
    """Online native-SOE eligibility driven by local forcing prediction error.

    The predictor observes the state *before* an event and predicts that
    event's forcing. Its local error is multiplied by channel activity at the
    completed event. Terminal per-channel TD errors then adjust only the
    persistent channel-credit gains; no post-hoc event selector is learned.
    """

    state_dim: int
    channel_count: int
    learning_rate: float
    credit_learning_rate: float
    persistent_decay: bool = False
    weights: list[list[float]] = field(init=False)
    bias: list[float] = field(init=False)
    channel_credit_gains: list[float] = field(init=False)
    _records: dict[int, list[tuple[int, float, list[float]]]] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.weights = [[0.0 for _ in range(self.state_dim)] for _ in range(self.state_dim)]
        self.bias = [0.0 for _ in range(self.state_dim)]
        self.channel_credit_gains = [1.0 for _ in range(self.channel_count)]

    def _event_indices(self, result) -> list[int]:
        indices = [
            index
            for index, step in enumerate(result.trace.steps)
            if step.token is not None
            and (index + 1 == len(result.trace.steps) or result.trace.steps[index + 1].token != step.token)
        ]
        # The terminal trigger is already supplied separately as the readout
        # query. Persistent eligibility tests which *earlier* event survives
        # to that query, so it must not win by zero-delay recency.
        if self.persistent_decay and indices:
            indices.pop()
        return indices

    def _records_for(self, result, memory, *, learn_predictor: bool) -> list[tuple[int, float, list[float]]]:
        records: list[tuple[int, float, list[float]]] = []
        for index in self._event_indices(result):
            step = result.trace.steps[index]
            before_u = [0.0 for _ in step.u] if index == 0 else result.trace.steps[index - 1].u
            prediction = [
                sum(weight * value for weight, value in zip(row, before_u)) + offset
                for row, offset in zip(self.weights, self.bias)
            ]
            residual = [target - estimate for target, estimate in zip(step.forcing, prediction)]
            error = sum(value * value for value in residual) / len(residual)
            activity = _channel_activity(step.chi)
            records.append((index, error, activity))
            if learn_predictor:
                for output_index, value in enumerate(residual):
                    for input_index, before_value in enumerate(before_u):
                        self.weights[output_index][input_index] += self.learning_rate * value * before_value
                    self.bias[output_index] += self.learning_rate * value
        return records

    def _event_score(self, result, memory, index: int, prediction_error: float, activity: list[float]) -> float:
        if not self.persistent_decay:
            return prediction_error * sum(
                gain * channel_activity
                for gain, channel_activity in zip(self.channel_credit_gains, activity)
            )
        elapsed = result.trace.steps[-1].t - result.trace.steps[index].t
        return prediction_error * sum(
            gain * math.exp(-rate * elapsed)
            for gain, rate in zip(self.channel_credit_gains, memory.gamma)
        )

    def _features_from_records(self, result, memory, records) -> tuple[list[float], list[int], list[float], list[list[float]]]:
        candidates = [index for index, _, _ in records]
        snapshots: list[list[float]] = []
        scores: list[float] = []
        for index, prediction_error, activity in records:
            step = result.trace.steps[index]
            state = DigitalMemoryState(u=list(step.u), chi=[list(channel) for channel in step.chi], t=step.t)
            snapshots.append([100.0 * value for value in combined_readout_vector(state, memory)])
            scores.append(self._event_score(result, memory, index, prediction_error, activity))
        total = sum(scores)
        attention = [score / total for score in scores] if total > 0.0 else [1.0 / len(scores) for _ in scores]
        eligibility = [
            sum(weight * snapshot[dimension] for weight, snapshot in zip(attention, snapshots))
            for dimension in range(len(snapshots[0]))
        ]
        return _features_with_eligibility_vector(result, memory, eligibility), candidates, attention, snapshots

    def learn_features(self, result, memory) -> tuple[list[float], list[int], list[float], list[list[float]]]:
        records = self._records_for(result, memory, learn_predictor=True)
        self._records[id(result)] = records
        return self._features_from_records(result, memory, records)

    def features(self, result, memory) -> tuple[list[float], list[int], list[float], list[list[float]]]:
        return self._features_from_records(result, memory, self._records_for(result, memory, learn_predictor=False))

    def apply_terminal_td(self, result, memory, td_errors: list[float]) -> None:
        records = self._records.pop(id(result), [])
        if not records:
            return
        for channel_index, td_error in enumerate(td_errors):
            if self.persistent_decay:
                trace_strength = sum(
                    error * math.exp(-memory.gamma[channel_index] * (result.trace.steps[-1].t - result.trace.steps[index].t))
                    for index, error, _ in records
                ) / len(records)
            else:
                trace_strength = sum(error * activity[channel_index] for _, error, activity in records) / len(records)
            self.channel_credit_gains[channel_index] = max(
                0.0,
                self.channel_credit_gains[channel_index] + self.credit_learning_rate * td_error * trace_strength,
            )

    def diagnostics(self, rows, memory) -> dict[str, object]:
        premise: list[float] = []
        irrelevant: list[float] = []
        errors: list[float] = []
        for _, _, result in rows:
            records = self._records_for(result, memory, learn_predictor=False)
            for index, error, activity in records:
                credit = self._event_score(result, memory, index, error, activity)
                errors.append(error)
                if result.trace.steps[index].token in {"OPEN", "CLOSED"}:
                    premise.append(credit)
                elif result.trace.steps[index].token in {"DISTRACTOR_A", "DISTRACTOR_B", "FILLER"}:
                    irrelevant.append(credit)
        mean_premise = sum(premise) / len(premise)
        mean_irrelevant = sum(irrelevant) / len(irrelevant)
        return {
            "scorer": (
                "online_next_forcing_error_times_channel_decayed_persistent_eligibility"
                if self.persistent_decay else "online_next_forcing_error_times_native_soe_channel_activity"
            ),
            "candidate_mode": "per_completed_token_event",
            "terminal_trigger_excluded": self.persistent_decay,
            "mean_prediction_error": sum(errors) / len(errors),
            "mean_premise_credit": mean_premise,
            "mean_irrelevant_credit": mean_irrelevant,
            "premise_credit_exceeds_irrelevant": mean_premise > mean_irrelevant,
            "channel_credit_gains": list(self.channel_credit_gains),
        }


@dataclass(slots=True)
class DelayedFeedbackEligibilityGate:
    """REINFORCE-style event selector trained only from terminal reward.

    The policy observes event token identity, not premise labels or targets.
    During learning it samples one non-silent event; terminal correctness
    adjusts the token score using a moving reward baseline. Evaluation uses
    the highest-scoring event deterministically.
    """

    vocabulary: Vocabulary
    seed: int
    learning_rate: float
    scores: list[float] = field(init=False)
    reward_baseline: float = 0.0
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.scores = [0.0 for _ in self.vocabulary.tokens]
        self._rng = random.Random(self.seed)

    def _candidate_indices(self, result) -> list[int]:
        return [index for index, step in enumerate(result.trace.steps) if step.token is not None]

    def _probabilities(self, result) -> tuple[list[int], list[float]]:
        candidates = self._candidate_indices(result)
        logits = [self.scores[self.vocabulary.token_id(result.trace.steps[index].token)] for index in candidates]
        offset = max(logits)
        exp_logits = [math.exp(value - offset) for value in logits]
        total = sum(exp_logits)
        return candidates, [value / total for value in exp_logits]

    def select(self, result, *, sample: bool) -> tuple[int, list[int], list[float]]:
        candidates, probabilities = self._probabilities(result)
        if not sample:
            # Repeated pulse steps share a token score; take the latest pulse
            # so the eligibility snapshot contains the complete event input.
            best = max(probabilities)
            position = max(index for index, probability in enumerate(probabilities) if probability == best)
            return candidates[position], candidates, probabilities
        threshold = self._rng.random()
        cumulative = 0.0
        for position, probability in enumerate(probabilities):
            cumulative += probability
            if threshold <= cumulative:
                return candidates[position], candidates, probabilities
        return candidates[-1], candidates, probabilities

    def update(self, result, *, selected_index: int, candidates: list[int], probabilities: list[float], reward: float) -> None:
        advantage = float(reward) - self.reward_baseline
        self.reward_baseline = 0.95 * self.reward_baseline + 0.05 * float(reward)
        for index, probability in zip(candidates, probabilities):
            token_id = self.vocabulary.token_id(result.trace.steps[index].token)
            indicator = 1.0 if index == selected_index else 0.0
            self.scores[token_id] += self.learning_rate * advantage * (indicator - probability)

    def premise_selection_rate(self, rows) -> float:
        selected = 0
        for _, _, result in rows:
            index, _, _ = self.select(result, sample=False)
            if result.trace.steps[index].token in {"OPEN", "CLOSED"}:
                selected += 1
        return selected / len(rows)


@dataclass(slots=True)
class ChannelValueCritic:
    """One linear terminal-value estimate per SOE channel."""

    coefficients: list[float]
    learning_rate: float
    mean_abs_td: list[float] = field(init=False)
    mean_sq_td: list[float] = field(init=False)
    count: int = 0

    def __post_init__(self) -> None:
        self.mean_abs_td = [0.0 for _ in self.coefficients]
        self.mean_sq_td = [0.0 for _ in self.coefficients]

    def estimate(self, activity: list[float]) -> list[float]:
        return [coefficient * value for coefficient, value in zip(self.coefficients, activity)]

    def update(self, activity: list[float], td_errors: list[float]) -> None:
        self.count += 1
        for index, error in enumerate(td_errors):
            self.mean_abs_td[index] += (abs(error) - self.mean_abs_td[index]) / self.count
            self.mean_sq_td[index] += (error * error - self.mean_sq_td[index]) / self.count
        snr = [
            signal / max(math.sqrt(max(square - signal * signal, 0.0)), 1e-6)
            for signal, square in zip(self.mean_abs_td, self.mean_sq_td)
        ]
        normalizer = max(max(snr), 1e-6)
        for index, (value, error) in enumerate(zip(activity, td_errors)):
            self.coefficients[index] += self.learning_rate * (snr[index] / normalizer) * error * value

    def snr(self) -> list[float]:
        return [
            signal / max(math.sqrt(max(square - signal * signal, 0.0)), 1e-6)
            for signal, square in zip(self.mean_abs_td, self.mean_sq_td)
        ]


@dataclass(slots=True)
class NextForcingPredictiveHead:
    """Online one-parameter next-forcing predictor for the single D3 level."""

    learning_rate: float = 0.02
    gain: float = 0.0

    def observe(self, result) -> float:
        errors: list[float] = []
        for current, following in zip(result.trace.steps, result.trace.steps[1:]):
            prediction = [self.gain * value for value in current.u]
            residual = [target - estimate for target, estimate in zip(following.forcing, prediction)]
            errors.append(sum(value * value for value in residual) / len(residual))
            denominator = sum(value * value for value in current.u) + 1e-8
            self.gain += self.learning_rate * sum(value * error for value, error in zip(current.u, residual)) / denominator
        return sum(errors) / len(errors)


def _dataset(config: TemporalLogicConfig, start: int, count: int):
    memory = adaptive_memory_config_for_variant(config.variant) if config.adaptive_kernel else memory_config_for_variant(config.variant)
    bridge = TokenForcingBridge(vocabulary=_vocabulary(), code_mode="one_hot")
    director = DigitalDirector(memory_config=memory, state_dim=bridge.state_dim)
    rows = []
    for index in range(start, start + count):
        episode, target = _episode(config, index, bridge)
        result = director.run_episode(episode, experiment_name="temporal_logic")
        rows.append((_features_for_result(
            result, memory, task_conditioned_eligibility=config.task_conditioned_eligibility,
        ), target, result))
    return director, rows


def _episode_specs(config: TemporalLogicConfig, *, start: int, count: int, bridge: TokenForcingBridge):
    return [_episode(config, index, bridge) for index in range(start, start + count)]


def _accuracy(readout: QueryConditionedMLPReadout, rows, *, gate=None, soft_trace=None, memory=None) -> float:
    correct = 0
    for features, target, result in rows:
        if soft_trace is not None:
            features, _, _, _ = soft_trace.features(result, memory)
        elif gate is not None:
            index, _, _ = gate.select(result, sample=False)
            features = _features_with_eligibility_snapshot(result, memory, result.trace.steps[index])
        correct += readout.predict(features)[0] == target
    return correct / len(rows)


def _eligibility_attention_diagnostics(eligibility_trace, rows, memory) -> dict[str, object] | None:
    """Summarize learned premise attention against its per-episode null level."""

    if eligibility_trace is None:
        return None
    premise_attention: list[float] = []
    uniform_attention: list[float] = []
    candidate_counts: list[int] = []
    for _, _, result in rows:
        _, candidates, attention, _ = eligibility_trace.features(result, memory)
        premise_attention.append(sum(
            weight
            for index, weight in zip(candidates, attention)
            if result.trace.steps[index].token in {"OPEN", "CLOSED"}
        ))
        # Each episode has one premise candidate, so this is the attention it
        # would receive from an untrained uniform selector.
        uniform_attention.append(1.0 / len(candidates))
        candidate_counts.append(len(candidates))
    mean_premise = sum(premise_attention) / len(premise_attention)
    mean_uniform = sum(uniform_attention) / len(uniform_attention)
    return {
        "mean_premise_attention": mean_premise,
        "mean_uniform_premise_attention": mean_uniform,
        "premise_attention_exceeds_uniform": mean_premise > mean_uniform,
        "mean_candidate_count": sum(candidate_counts) / len(candidate_counts),
        **eligibility_trace.score_diagnostics(),
    }


def run_temporal_logic_experiment(config: TemporalLogicConfig) -> dict[str, object]:
    """Train only the action readout after terminal feedback.

    Each prediction is made after the delayed trigger; correctness is then
    converted to a terminal reward and per-channel TD/update diagnostics.
    Kernel proposals are reported but never applied in this first D3 control.
    """

    director, train = _dataset(config, 0, config.train_count)
    _, test = _dataset(config, config.train_count, config.test_count)
    gate = DelayedFeedbackEligibilityGate(
        vocabulary=_vocabulary(), seed=config.seed, learning_rate=config.eligibility_learning_rate,
    ) if config.learned_eligibility else None
    soft_trace = SoftEligibilityTrace(
        vocabulary=_vocabulary(), learning_rate=config.eligibility_learning_rate,
    ) if config.soft_eligibility else None
    nonlinear_trace = NonlinearCompatibilityEligibilityTrace(
        vocabulary=_vocabulary(), learning_rate=config.eligibility_learning_rate,
        hidden_dim=config.compatibility_hidden_dim,
        parameter_budget=config.compatibility_parameter_budget,
        seed=config.seed,
    ) if config.nonlinear_eligibility else None
    predictive_credit_trace = PredictiveCreditEligibilityTrace(
        state_dim=director.state_dim,
        channel_count=director.memory_config.channel_count,
        learning_rate=config.predictive_learning_rate,
        credit_learning_rate=config.eligibility_learning_rate,
    ) if config.predictive_credit_eligibility else None
    persistent_predictive_credit_trace = PredictiveCreditEligibilityTrace(
        state_dim=director.state_dim,
        channel_count=director.memory_config.channel_count,
        learning_rate=config.predictive_learning_rate,
        credit_learning_rate=config.eligibility_learning_rate,
        persistent_decay=True,
    ) if config.persistent_predictive_credit_eligibility else None
    eligibility_trace = persistent_predictive_credit_trace or predictive_credit_trace or nonlinear_trace or soft_trace
    initial_features = train[0][0]
    if eligibility_trace is not None:
        initial_features, _, _, _ = eligibility_trace.features(train[0][2], director.memory_config)
    elif gate is not None:
        first_index, _, _ = gate.select(train[0][2], sample=False)
        initial_features = _features_with_eligibility_snapshot(
            train[0][2], director.memory_config, train[0][2].trace.steps[first_index],
        )
    readout = QueryConditionedMLPReadout(
        input_dim=len(initial_features), output_dim=len(_ACTIONS), hidden_dim=config.hidden_dim, seed=config.seed,
    )
    losses: list[float] = []
    feedback_rows = []
    initial_weights = list(director.memory_config.w)
    critic = ChannelValueCritic(
        coefficients=[0.0 for _ in director.memory_config.gamma], learning_rate=config.critic_learning_rate,
    )
    predictive_head = NextForcingPredictiveHead()
    slow_weights = list(director.memory_config.w)
    kernel_update_count = 0
    consolidation_count = 0
    rng = random.Random(config.seed)
    for _ in range(config.epochs):
        total_loss = 0.0
        epoch_rows = list(train)
        rng.shuffle(epoch_rows)
        for row_index, (features, target, result) in enumerate(epoch_rows):
            if config.adaptive_kernel:
                # Replay the forcing schedule through the current learned
                # kernel; otherwise the episode result is the frozen control.
                episode, expected_target = _episode(
                    config, int(result.metadata.get("example_index", row_index)),
                    TokenForcingBridge(vocabulary=_vocabulary(), code_mode="one_hot"),
                )
                result = director.run_episode(episode, experiment_name="temporal_logic_adaptive")
                target = expected_target
                features = _features_for_result(result, director.memory_config, task_conditioned_eligibility=False)
            selected_index = candidates = probabilities = None
            if gate is not None:
                selected_index, candidates, probabilities = gate.select(result, sample=True)
                features = _features_with_eligibility_snapshot(result, director.memory_config, result.trace.steps[selected_index])
            elif eligibility_trace is not None:
                if persistent_predictive_credit_trace is not None:
                    features, soft_candidates, soft_attention, soft_snapshots = persistent_predictive_credit_trace.learn_features(
                        result, director.memory_config,
                    )
                elif predictive_credit_trace is not None:
                    features, soft_candidates, soft_attention, soft_snapshots = predictive_credit_trace.learn_features(
                        result, director.memory_config,
                    )
                else:
                    features, soft_candidates, soft_attention, soft_snapshots = eligibility_trace.features(
                        result, director.memory_config,
                    )
            predicted, confidence = readout.predict(features)
            reward = 1.0 if predicted == target else -1.0
            # The terminal action-label error is a denser delayed feedback
            # signal than binary correctness. It never exposes premise labels;
            # it only asks whether the selected event state supports the final
            # action after the episode has completed.
            terminal_error = -math.log(max(readout.probabilities(features)[target], 1e-12))
            activity = _channel_activity(result.final_state.chi)
            predictive_error = predictive_head.observe(result) if config.adaptive_kernel else 1.0 - confidence
            feedback = director.apply_delayed_feedback(
                result,
                reward=reward,
                prediction_error=predictive_error,
                current_values=critic.estimate(activity),
            )
            feedback_rows.append(feedback)
            if config.credit_enabled:
                if eligibility_trace is not None:
                    loss, input_gradient = readout.train_step_with_input_gradient(
                        features, target, learning_rate=config.learning_rate,
                    )
                    if predictive_credit_trace is None and persistent_predictive_credit_trace is None:
                        eligibility_trace.update(
                            result, candidates=soft_candidates, attention=soft_attention,
                            snapshots=soft_snapshots, input_gradient=input_gradient,
                        )
                else:
                    loss = readout.train_step(features, target, learning_rate=config.learning_rate)
                total_loss += loss
                if predictive_credit_trace is not None:
                    predictive_credit_trace.apply_terminal_td(result, director.memory_config, feedback.td_error.errors)
                if persistent_predictive_credit_trace is not None:
                    persistent_predictive_credit_trace.apply_terminal_td(
                        result, director.memory_config, feedback.td_error.errors,
                    )
                if config.adaptive_kernel:
                    critic.update(activity, feedback.td_error.errors)
                    director.commit_kernel_proposal(feedback.kernel_proposal)
                    kernel_update_count += 1
                    if kernel_update_count % config.consolidation_interval == 0:
                        slow_weights = [
                            slow + config.consolidation_rate * (fast - slow)
                            for fast, slow in zip(director.memory_config.w, slow_weights)
                        ]
                        director.memory_config.w = [
                            (1.0 - config.consolidation_rate) * fast + config.consolidation_rate * slow
                            for fast, slow in zip(director.memory_config.w, slow_weights)
                        ]
                        ratio = director.memory_config.stability_ratio()
                        if ratio > 0.9:
                            director.memory_config.w = [
                                weight * 0.9 / ratio for weight in director.memory_config.w
                            ]
                        consolidation_count += 1
                if gate is not None:
                    gate.update(
                        result,
                        selected_index=selected_index,
                        candidates=candidates,
                        probabilities=probabilities,
                        reward=-terminal_error,
                    )
        losses.append(total_loss / len(train) if config.credit_enabled else 0.0)

    # Once the learned policy has settled, consolidate the action readout using
    # its deterministic eligibility choice. This remains delayed supervised
    # feedback; it prevents a late-improving gate from being judged by a
    # readout trained mostly on its earlier exploratory selections.
    if gate is not None and config.credit_enabled:
        for _ in range(config.readout_consolidation_epochs):
            consolidation_rows = list(train)
            rng.shuffle(consolidation_rows)
            for _, target, result in consolidation_rows:
                selected_index, _, _ = gate.select(result, sample=False)
                selected_features = _features_with_eligibility_snapshot(
                    result, director.memory_config, result.trace.steps[selected_index],
                )
                readout.train_step(selected_features, target, learning_rate=config.learning_rate)

    premise_credit: list[float] = []
    irrelevant_credit: list[float] = []
    if config.adaptive_kernel:
        bridge = TokenForcingBridge(vocabulary=_vocabulary(), code_mode="one_hot")
        evaluator = DigitalDirector(
            memory_config=type(director.memory_config).from_mapping(director.memory_config.to_mapping()),
            state_dim=bridge.state_dim,
        )
        adaptive_test = []
        for index in range(config.train_count, config.train_count + config.test_count):
            episode, target = _episode(config, index, bridge)
            result = evaluator.run_episode(episode, experiment_name="temporal_logic_adaptive_evaluate")
            adaptive_test.append((_features_for_result(result, evaluator.memory_config, task_conditioned_eligibility=False), target, result))
        test = adaptive_test

    for features, target, result in test:
        if gate is not None:
            selected_index, _, _ = gate.select(result, sample=False)
            features = _features_with_eligibility_snapshot(result, director.memory_config, result.trace.steps[selected_index])
        elif eligibility_trace is not None:
            features, _, _, _ = eligibility_trace.features(result, director.memory_config)
        predicted, confidence = readout.predict(features)
        feedback = director.apply_delayed_feedback(
            result, reward=1.0 if predicted == target else -1.0, prediction_error=1.0 - confidence, phase="EVALUATE",
        )
        steps = result.trace.steps
        premise_index = next(index for index, step in enumerate(steps) if step.token in {"OPEN", "CLOSED"})
        premise_before = [[0.0 for _ in channel] for channel in steps[premise_index].chi]
        premise_credit.append(_event_trace_credit(
            premise_before, steps[premise_index].chi, feedback.td_error.errors,
        ))
        irrelevant_indices = [
            index for index, step in enumerate(steps)
            if step.token in {"FILLER", "DISTRACTOR_A", "DISTRACTOR_B"}
        ]
        event_credits = [
            _event_trace_credit(steps[index - 1].chi, steps[index].chi, feedback.td_error.errors)
            for index in irrelevant_indices
        ]
        irrelevant_credit.append(sum(event_credits) / len(event_credits))

    memory = director.memory_config
    proposal_rescales = sum(item.kernel_proposal.stability_rescaled for item in feedback_rows)
    soft_attention = _eligibility_attention_diagnostics(soft_trace, test, memory)
    nonlinear_attention = _eligibility_attention_diagnostics(nonlinear_trace, test, memory)
    predictive_credit = None if predictive_credit_trace is None else predictive_credit_trace.diagnostics(test, memory)
    persistent_predictive_credit = (
        None if persistent_predictive_credit_trace is None
        else persistent_predictive_credit_trace.diagnostics(test, memory)
    )
    return {
        "experiment_name": "temporal_logic_delayed_feedback",
        "config": config.to_mapping(),
        "variant": config.variant,
        "credit_enabled": config.credit_enabled,
        "forced_distractors": config.forced_distractors,
        "task_conditioned_eligibility": config.task_conditioned_eligibility,
        "learned_eligibility": config.learned_eligibility,
        "soft_eligibility": config.soft_eligibility,
        "nonlinear_eligibility": config.nonlinear_eligibility,
        "predictive_credit_eligibility": config.predictive_credit_eligibility,
        "persistent_predictive_credit_eligibility": config.persistent_predictive_credit_eligibility,
        "adaptive_kernel": config.adaptive_kernel,
        "frozen_components": (
            ["SOE gamma", "token forcing bridge"] if config.adaptive_kernel
            else ["SOE gamma", "SOE weights", "token forcing bridge"]
        ),
        "learned_component": "query_conditioned_relu_mlp_action_readout" if config.credit_enabled else "none",
        "actions": list(_ACTIONS),
        "train_accuracy": _accuracy(readout, train, gate=gate, soft_trace=eligibility_trace, memory=director.memory_config),
        "test_accuracy": _accuracy(readout, test, gate=gate, soft_trace=eligibility_trace, memory=director.memory_config),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "memory_diagnostics": {"deff": memory.deff(), "stability_ratio": memory.stability_ratio()},
        "adaptive_learning": {
            "kernel_update_count": kernel_update_count,
            "consolidation_count": consolidation_count,
            "initial_weights": initial_weights,
            "final_weights": list(director.memory_config.w),
            "critic_coefficients": list(critic.coefficients),
            "critic_snr": critic.snr(),
            "predictive_head_gain": predictive_head.gain,
        },
        "feedback_diagnostics": {
            "feedback_count": len(feedback_rows),
            "mean_abs_td_error": sum(sum(abs(value) for value in item.td_error.errors) / len(item.td_error.errors) for item in feedback_rows) / len(feedback_rows),
            "proposal_rescale_count": proposal_rescales,
            "mean_premise_trace_credit": sum(premise_credit) / len(premise_credit),
            "mean_irrelevant_trace_credit": sum(irrelevant_credit) / len(irrelevant_credit),
            "premise_credit_exceeds_irrelevant": (
                sum(premise_credit) / len(premise_credit) > sum(irrelevant_credit) / len(irrelevant_credit)
            ),
            "eligibility_gate_selective": config.task_conditioned_eligibility,
            "eligibility_gate_description": (
                "task-provided premise-token gate; distractor events excluded"
                if config.task_conditioned_eligibility else "none"
            ),
            "learned_gate_premise_selection_rate": None if gate is None else gate.premise_selection_rate(test),
            "readout_consolidation_epochs": config.readout_consolidation_epochs if gate is not None else 0,
            "soft_eligibility": soft_attention,
            "nonlinear_eligibility": nonlinear_attention,
            "predictive_credit_eligibility": predictive_credit,
            "persistent_predictive_credit_eligibility": persistent_predictive_credit,
            "kernel_updates_applied": kernel_update_count,
        },
    }


def run_temporal_logic_sweep(
    *,
    gaps: list[int],
    seeds: list[int],
    variants: list[str] | None = None,
    forced_distractors: bool = False,
    task_conditioned_eligibility: bool = False,
    learned_eligibility: bool = False,
    soft_eligibility: bool = False,
    nonlinear_eligibility: bool = False,
    predictive_credit_eligibility: bool = False,
    persistent_predictive_credit_eligibility: bool = False,
    compatibility_hidden_dim: int = 8,
    compatibility_parameter_budget: int = 384,
    train_count: int = 64,
    test_count: int = 32,
    epochs: int = 30,
) -> list[dict[str, object]]:
    """Run deterministic D3 ablations over delay and readout initialisation."""

    selected_variants = variants or ["full", "no_slow", "collapsed_gamma"]
    rows: list[dict[str, object]] = []
    for gap in gaps:
        for variant in selected_variants:
            for seed in seeds:
                result = run_temporal_logic_experiment(TemporalLogicConfig(
                    variant=variant,
                    gap=gap,
                    seed=seed,
                    forced_distractors=forced_distractors,
                    task_conditioned_eligibility=task_conditioned_eligibility,
                    learned_eligibility=learned_eligibility,
                    soft_eligibility=soft_eligibility,
                    nonlinear_eligibility=nonlinear_eligibility,
                    predictive_credit_eligibility=predictive_credit_eligibility,
                    persistent_predictive_credit_eligibility=persistent_predictive_credit_eligibility,
                    compatibility_hidden_dim=compatibility_hidden_dim,
                    compatibility_parameter_budget=compatibility_parameter_budget,
                    train_count=train_count,
                    test_count=test_count,
                    epochs=epochs,
                ))
                rows.append({
                    "gap": gap,
                    "seed": seed,
                    "variant": variant,
                    "forced_distractors": forced_distractors,
                    "task_conditioned_eligibility": task_conditioned_eligibility,
                    "learned_eligibility": learned_eligibility,
                    "soft_eligibility": soft_eligibility,
                    "nonlinear_eligibility": nonlinear_eligibility,
                    "predictive_credit_eligibility": predictive_credit_eligibility,
                    "persistent_predictive_credit_eligibility": persistent_predictive_credit_eligibility,
                    "test_accuracy": result["test_accuracy"],
                    "mean_premise_eligibility_attention": (
                        None if (attention := result["feedback_diagnostics"]["nonlinear_eligibility"] or result["feedback_diagnostics"]["soft_eligibility"]) is None
                        else attention["mean_premise_attention"]
                    ),
                    "uniform_premise_eligibility_attention": (
                        None if (attention := result["feedback_diagnostics"]["nonlinear_eligibility"] or result["feedback_diagnostics"]["soft_eligibility"]) is None
                        else attention["mean_uniform_premise_attention"]
                    ),
                    "premise_attention_exceeds_uniform": (
                        None if (attention := result["feedback_diagnostics"]["nonlinear_eligibility"] or result["feedback_diagnostics"]["soft_eligibility"]) is None
                        else attention["premise_attention_exceeds_uniform"]
                    ),
                    "predictive_credit_premise_exceeds_irrelevant": (
                        None if result["feedback_diagnostics"]["predictive_credit_eligibility"] is None
                        else result["feedback_diagnostics"]["predictive_credit_eligibility"]["premise_credit_exceeds_irrelevant"]
                    ),
                    "persistent_predictive_credit_premise_exceeds_irrelevant": (
                        None if result["feedback_diagnostics"]["persistent_predictive_credit_eligibility"] is None
                        else result["feedback_diagnostics"]["persistent_predictive_credit_eligibility"]["premise_credit_exceeds_irrelevant"]
                    ),
                    "premise_credit_exceeds_irrelevant": result["feedback_diagnostics"]["premise_credit_exceeds_irrelevant"],
                    "mean_premise_trace_credit": result["feedback_diagnostics"]["mean_premise_trace_credit"],
                    "mean_irrelevant_trace_credit": result["feedback_diagnostics"]["mean_irrelevant_trace_credit"],
                })
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the D3 delayed-feedback temporal-rule probe.")
    parser.add_argument("--variant", choices=["full", "no_slow", "collapsed_gamma"], default="full")
    parser.add_argument("--gap", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--no-credit", action="store_true")
    parser.add_argument("--forced-distractors", action="store_true")
    parser.add_argument("--task-conditioned-eligibility", action="store_true")
    parser.add_argument("--learned-eligibility", action="store_true")
    parser.add_argument("--soft-eligibility", action="store_true")
    parser.add_argument("--nonlinear-eligibility", action="store_true")
    parser.add_argument("--predictive-credit-eligibility", action="store_true")
    parser.add_argument("--persistent-predictive-credit-eligibility", action="store_true")
    parser.add_argument("--compatibility-hidden-dim", type=int, default=8)
    parser.add_argument("--compatibility-parameter-budget", type=int, default=384)
    parser.add_argument("--adaptive-kernel", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> dict[str, object]:
    args = build_arg_parser().parse_args(argv)
    result = run_temporal_logic_experiment(TemporalLogicConfig(
        variant=args.variant, gap=args.gap, epochs=args.epochs, credit_enabled=not args.no_credit,
        forced_distractors=args.forced_distractors,
        task_conditioned_eligibility=args.task_conditioned_eligibility,
        learned_eligibility=args.learned_eligibility,
        soft_eligibility=args.soft_eligibility,
        nonlinear_eligibility=args.nonlinear_eligibility,
        predictive_credit_eligibility=args.predictive_credit_eligibility,
        persistent_predictive_credit_eligibility=args.persistent_predictive_credit_eligibility,
        compatibility_hidden_dim=args.compatibility_hidden_dim,
        compatibility_parameter_budget=args.compatibility_parameter_budget,
        adaptive_kernel=args.adaptive_kernel,
    ))
    print(f"variant={result['variant']} credit={result['credit_enabled']} test_accuracy={result['test_accuracy']:.3f}")
    return result


if __name__ == "__main__":
    main()
