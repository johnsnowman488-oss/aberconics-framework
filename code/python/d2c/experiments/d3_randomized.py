"""D3-R: randomized indexed rule retrieval with learned soft eligibility.

This replaces the four canonical temporal-logic streams with independently
sampled episodes.  A terminal query names one slot and a mode (identity or
invert); action learning must recover that slot's earlier bit despite other
slot/value events.  The SOE kernel is deliberately frozen in this first
control.  Only the eligibility head and binary action readout learn from the
terminal cross-entropy signal.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random

from ..digital import (
    DigitalDirector,
    DigitalEpisode,
    DigitalMemoryState,
    QueryConditionedMLPReadout,
    combined_readout_vector,
)
from .symbolic_induction import memory_config_for_variant


@dataclass(frozen=True, slots=True)
class IndexedRuleEpisode:
    """One independently generated D3-R episode specification."""

    mode: int
    assignments: tuple[int, ...]
    order: tuple[int, ...]
    query_slot: int

    @property
    def target(self) -> int:
        return self.assignments[self.query_slot] ^ self.mode


@dataclass(slots=True)
class D3RandomizedConfig:
    variant: str = "full"
    slot_count: int = 6
    gap: int = 32
    train_count: int = 128
    test_count: int = 64
    epochs: int = 40
    hidden_dim: int = 32
    learning_rate: float = 0.01
    eligibility_learning_rate: float = 0.03
    pulse_steps: int = 2
    silence_steps: int = 1
    seed: int = 41
    eligibility_enabled: bool = True
    task_gated: bool = False
    binding_noise_std: float = 0.0
    distractor_count: int = 0

    def __post_init__(self) -> None:
        if self.variant not in {"full", "no_slow", "collapsed_gamma"}:
            raise ValueError("unsupported memory variant")
        if min(self.slot_count, self.gap, self.train_count, self.test_count, self.epochs, self.hidden_dim, self.pulse_steps) <= 0:
            raise ValueError("slot_count, gap, counts, and dimensions must be positive")
        if self.silence_steps < 0 or self.learning_rate <= 0.0 or self.eligibility_learning_rate <= 0.0:
            raise ValueError("invalid training or schedule setting")
        if self.task_gated and not self.eligibility_enabled:
            raise ValueError("task_gated and eligibility_enabled=False are mutually exclusive")
        if self.binding_noise_std < 0.0:
            raise ValueError("binding_noise_std must be non-negative")
        if self.distractor_count < 0:
            raise ValueError("distractor_count must be non-negative")

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)


def _sample_specs(config: D3RandomizedConfig, *, seed: int, count: int) -> list[IndexedRuleEpisode]:
    """Sample a stratified, independent stream with no canonical case cycle."""

    rng = random.Random(seed)
    specs: list[IndexedRuleEpisode] = []
    for index in range(count):
        # Cycle the four rule/action strata while all assignments and event
        # positions remain independently random.  This prevents class balance
        # from becoming a substitute for retrieval.
        mode = index % 2
        desired_action = (index // 2) % 2
        query_slot = rng.randrange(config.slot_count)
        assignments = [rng.randrange(2) for _ in range(config.slot_count)]
        assignments[query_slot] = desired_action ^ mode
        order = list(range(config.slot_count))
        rng.shuffle(order)
        specs.append(IndexedRuleEpisode(mode, tuple(assignments), tuple(order), query_slot))
    rng.shuffle(specs)
    return specs


def _state_dim(config: D3RandomizedConfig) -> int:
    # Two value coordinates per slot plus two persistent mode coordinates.
    return 2 * config.slot_count + 2


def _mode_index(config: D3RandomizedConfig, mode: int) -> int:
    return 2 * config.slot_count + mode


def _event_forcing(config: D3RandomizedConfig, slot: int, value: int) -> list[float]:
    forcing = [0.0] * _state_dim(config)
    forcing[2 * slot + value] = 1.0
    return forcing


def _query(config: D3RandomizedConfig, spec: IndexedRuleEpisode) -> list[float]:
    query = [0.0] * _state_dim(config)
    # Query slot has no value information: both coordinates are active.
    query[2 * spec.query_slot] = 1.0
    query[2 * spec.query_slot + 1] = 1.0
    query[_mode_index(config, spec.mode)] = 1.0
    return query


def _episode_noise_rng(config: D3RandomizedConfig, spec: IndexedRuleEpisode) -> random.Random:
    # Tuple-of-int hashes are deterministic across processes, so the noise
    # stream for an episode is reproducible for a fixed config seed.
    key = hash((spec.mode, spec.assignments, spec.order, spec.query_slot))
    return random.Random(config.seed ^ key)


def _episode(config: D3RandomizedConfig, spec: IndexedRuleEpisode) -> DigitalEpisode:
    noise_rng = _episode_noise_rng(config, spec)

    def noisy(forcing: list[float]) -> list[float]:
        if not config.binding_noise_std:
            return forcing
        return [value + noise_rng.gauss(0.0, config.binding_noise_std) for value in forcing]

    forcings: list[list[float]] = []
    tokens: list[str | None] = []

    def add(token: str, forcing: list[float]) -> None:
        forcings.extend([noisy(forcing)] * config.pulse_steps)
        tokens.extend([token] * config.pulse_steps)
        forcings.extend([[0.0] * _state_dim(config)] * config.silence_steps)
        tokens.extend([None] * config.silence_steps)

    mode_forcing = [0.0] * _state_dim(config)
    mode_forcing[_mode_index(config, spec.mode)] = 1.0
    add("MODE_IDENTITY" if spec.mode == 0 else "MODE_INVERT", mode_forcing)
    for slot in spec.order:
        add(f"SLOT_{slot}_BIT_{spec.assignments[slot]}", _event_forcing(config, slot, spec.assignments[slot]))
    # Deterministic distractor events write conflicting one-hot bindings into
    # the same coordinates as real events.  Their DIST_ prefix keeps them out
    # of every SLOT_-filtered eligibility read.
    for distractor_index in range(config.distractor_count):
        slot = noise_rng.randrange(config.slot_count)
        value = noise_rng.randrange(2)
        add(f"DIST_{distractor_index}", _event_forcing(config, slot, value))
    for _ in range(config.gap):
        forcings.append([0.0] * _state_dim(config))
        tokens.append(None)
    return DigitalEpisode(
        forcings=forcings,
        tokens=tokens,
        target=f"ACTION_{spec.target}",
        query=_query(config, spec),
        metadata={"mode": spec.mode, "query_slot": spec.query_slot, "assignments": list(spec.assignments), "order": list(spec.order)},
    )


def _completed_event_indices(result) -> list[int]:
    return [
        index for index, step in enumerate(result.trace.steps)
        if step.token is not None and (index + 1 == len(result.trace.steps) or result.trace.steps[index + 1].token != step.token)
        and step.token.startswith("SLOT_")
    ]


class QuerySlotEligibility:
    """Terminal-query-conditioned soft selector trained through action loss."""

    def __init__(self, config: D3RandomizedConfig) -> None:
        self.config = config
        self.slot_scores = [[[0.0 for _ in range(config.slot_count)] for _ in range(config.slot_count)] for _ in range(2)]
        self.state_weights: list[list[list[float]]] | None = None

    @staticmethod
    def _descriptor(snapshot: list[float]) -> list[float]:
        norm = math.sqrt(sum(value * value for value in snapshot))
        return [value / max(norm, 1.0) for value in snapshot]

    def _ensure_state_weights(self, state_dim: int) -> None:
        if self.state_weights is None:
            self.state_weights = [[[0.0 for _ in range(state_dim)] for _ in range(self.config.slot_count)] for _ in range(2)]

    def features(self, result, memory) -> tuple[list[float], list[int], list[float], list[list[float]]]:
        indices = _completed_event_indices(result)
        snapshots: list[list[float]] = []
        slots: list[int] = []
        for index in indices:
            step = result.trace.steps[index]
            state = DigitalMemoryState(list(step.u), [list(channel) for channel in step.chi], step.t)
            snapshots.append([100.0 * value for value in combined_readout_vector(state, memory)])
            slots.append(int(step.token.split("_")[1]))
        self._ensure_state_weights(len(snapshots[0]))
        mode = int(result.metadata["mode"])
        query_slot = int(result.metadata["query_slot"])
        logits = [
            self.slot_scores[mode][query_slot][slot] + sum(weight * value for weight, value in zip(self.state_weights[mode][query_slot], self._descriptor(snapshot)))
            for slot, snapshot in zip(slots, snapshots)
        ]
        offset = max(logits)
        unnormalized = [math.exp(logit - offset) for logit in logits]
        total = sum(unnormalized)
        attention = [value / total for value in unnormalized]
        eligibility = [sum(weight * snapshot[d] for weight, snapshot in zip(attention, snapshots)) for d in range(len(snapshots[0]))]
        final = [100.0 * value for value in combined_readout_vector(result.final_state, memory)]
        features = [*final, *result.query, *eligibility, *(q * value for q in result.query for value in eligibility)]
        return features, indices, attention, snapshots

    def update(self, result, *, indices: list[int], attention: list[float], snapshots: list[list[float]], input_gradient: list[float]) -> None:
        eligibility_dim = len(snapshots[0])
        final_dim = eligibility_dim
        query_dim = len(result.query)
        start = final_dim + query_dim
        gradient = list(input_gradient[start : start + eligibility_dim])
        gated_start = start + eligibility_dim
        for query_index, query_value in enumerate(result.query):
            offset = gated_start + query_index * eligibility_dim
            for dimension in range(eligibility_dim):
                gradient[dimension] += query_value * input_gradient[offset + dimension]
        event_gradient = [sum(g * value for g, value in zip(gradient, snapshot)) for snapshot in snapshots]
        expected = sum(weight * value for weight, value in zip(attention, event_gradient))
        mode = int(result.metadata["mode"])
        query_slot = int(result.metadata["query_slot"])
        for index, weight, value, snapshot in zip(indices, attention, event_gradient, snapshots):
            slot = int(result.trace.steps[index].token.split("_")[1])
            logit_gradient = weight * (value - expected)
            self.slot_scores[mode][query_slot][slot] -= self.config.eligibility_learning_rate * logit_gradient
            descriptor = self._descriptor(snapshot)
            for dimension, feature in enumerate(descriptor):
                self.state_weights[mode][query_slot][dimension] -= self.config.eligibility_learning_rate * logit_gradient * feature

    def target_attention(self, result, memory) -> tuple[float, float]:
        _, indices, attention, _ = self.features(result, memory)
        target = int(result.metadata["query_slot"])
        selected = sum(weight for index, weight in zip(indices, attention) if int(result.trace.steps[index].token.split("_")[1]) == target)
        return selected, 1.0 / len(indices)


def _run_rows(config: D3RandomizedConfig, specs: list[IndexedRuleEpisode], director: DigitalDirector):
    return [(spec, director.run_episode(_episode(config, spec), experiment_name="d3_randomized")) for spec in specs]


def run_d3_randomized_experiment(config: D3RandomizedConfig) -> dict[str, object]:
    """Train and evaluate D3-R on disjoint randomly generated streams."""

    memory = memory_config_for_variant(config.variant)
    director = DigitalDirector(memory_config=memory, state_dim=_state_dim(config))
    train_specs = _sample_specs(config, seed=config.seed, count=config.train_count)
    test_specs = _sample_specs(config, seed=config.seed + 1_000_003, count=config.test_count)
    train_rows = _run_rows(config, train_specs, director)
    test_rows = _run_rows(config, test_specs, director)
    eligibility = QuerySlotEligibility(config)

    # Initialize input size from the actual continuous state representation.
    first_features, _, _, _ = eligibility.features(train_rows[0][1], memory)
    if not config.eligibility_enabled:
        first_result = train_rows[0][1]
        first_features = [
            *(100.0 * value for value in combined_readout_vector(first_result.final_state, memory)),
            *first_result.query,
        ]
    readout = QueryConditionedMLPReadout(len(first_features), 2, hidden_dim=config.hidden_dim, seed=config.seed)
    losses: list[float] = []
    for _ in range(config.epochs):
        for spec, result in train_rows:
            if config.task_gated:
                # Explicit upper bound only: latch the queried event snapshot.
                indices = _completed_event_indices(result)
                target_index = next(i for i in indices if int(result.trace.steps[i].token.split("_")[1]) == spec.query_slot)
                step = result.trace.steps[target_index]
                snapshot = [100.0 * value for value in combined_readout_vector(DigitalMemoryState(list(step.u), [list(channel) for channel in step.chi], step.t), memory)]
                final = [100.0 * value for value in combined_readout_vector(result.final_state, memory)]
                features = [*final, *result.query, *snapshot, *(q * value for q in result.query for value in snapshot)]
                losses.append(readout.train_step(features, spec.target, learning_rate=config.learning_rate))
            elif config.eligibility_enabled:
                features, indices, attention, snapshots = eligibility.features(result, memory)
                loss, gradient = readout.train_step_with_input_gradient(features, spec.target, learning_rate=config.learning_rate)
                eligibility.update(result, indices=indices, attention=attention, snapshots=snapshots, input_gradient=gradient)
                losses.append(loss)
            else:
                final = [100.0 * value for value in combined_readout_vector(result.final_state, memory)]
                losses.append(readout.train_step([*final, *result.query], spec.target, learning_rate=config.learning_rate))

    def predict(spec: IndexedRuleEpisode) -> tuple[int, float, float]:
        result = director.run_episode(_episode(config, spec), experiment_name="d3_randomized_eval")
        if config.task_gated:
            indices = _completed_event_indices(result)
            target_index = next(i for i in indices if int(result.trace.steps[i].token.split("_")[1]) == spec.query_slot)
            step = result.trace.steps[target_index]
            snapshot = [100.0 * value for value in combined_readout_vector(DigitalMemoryState(list(step.u), [list(channel) for channel in step.chi], step.t), memory)]
            final = [100.0 * value for value in combined_readout_vector(result.final_state, memory)]
            features = [*final, *result.query, *snapshot, *(q * value for q in result.query for value in snapshot)]
            return (*readout.predict(features), 1.0)
        if config.eligibility_enabled:
            features, _, _, _ = eligibility.features(result, memory)
            prediction, confidence = readout.predict(features)
            attention, uniform = eligibility.target_attention(result, memory)
            return prediction, confidence, attention - uniform
        final = [100.0 * value for value in combined_readout_vector(result.final_state, memory)]
        prediction, confidence = readout.predict([*final, *result.query])
        return prediction, confidence, 0.0

    predictions = [predict(spec) for spec in test_specs]
    accuracy = sum(prediction == spec.target for (prediction, _, _), spec in zip(predictions, test_specs)) / len(test_specs)
    # Causal counterfactuals use the same trained model and episode order.
    target_flips = 0
    distractor_invariant = 0
    for spec, (prediction, _, _) in zip(test_specs, predictions):
        target_values = list(spec.assignments)
        target_values[spec.query_slot] ^= 1
        target_spec = IndexedRuleEpisode(spec.mode, tuple(target_values), spec.order, spec.query_slot)
        target_flips += predict(target_spec)[0] != prediction
        distractor_slot = next(slot for slot in spec.order if slot != spec.query_slot)
        distractor_values = list(spec.assignments)
        distractor_values[distractor_slot] ^= 1
        distractor_spec = IndexedRuleEpisode(spec.mode, tuple(distractor_values), spec.order, spec.query_slot)
        distractor_invariant += predict(distractor_spec)[0] == prediction
    attention_margins = [margin for _, _, margin in predictions]
    return {
        "experiment_name": "d3_randomized_indexed_rule_retrieval",
        "config": config.to_mapping(),
        "variant": config.variant,
        "kernel_updates_applied": 0,
        "train_stream_seed": config.seed,
        "test_stream_seed": config.seed + 1_000_003,
        "independent_streams": True,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "test_accuracy": accuracy,
        "mean_target_attention_margin": sum(attention_margins) / len(attention_margins),
        "target_attention_exceeds_uniform": sum(attention_margins) / len(attention_margins) > 0.0,
        "interventions": {
            "queried_value_action_flip_rate": target_flips / len(test_specs),
            "unqueried_distractor_action_invariance_rate": distractor_invariant / len(test_specs),
        },
        "memory_diagnostics": {"stability_ratio": memory.stability_ratio(), "channel_count": memory.channel_count},
    }
