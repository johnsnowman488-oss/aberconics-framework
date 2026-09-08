"""Stage 2: D3-R-LB (learned binding) experiment.

The pre-existing D3-R task encodes each (slot, value) pair as a single
one-hot pulse on the *content* basis `2*slot + value`.  This is a
*pre-bound* forcing pattern: at the moment the SLOT pulse fires, the
forcing is already an outer-product code, and every kernel variant sees
the same binding-shaped state at query time.  Action accuracy is then
kernel-invariant by construction.

Stage 2 removes that confound.  Forcing is split into two disjoint
coordinate halves:

  - slot coordinates  :  [0, slot_count)       encode slot identity only
  - value coordinates :  [slot_count, 2*slot_count)  encode value identity only

A SLOT_k event forces only slot coordinate k (value coords are zero).
A VALUE_v event forces only value coordinate v (slot coords are zero).
The only way a (slot, value) association enters the SOE state is through
the memory channels integrating both pulses within a window comparable
to 1/gamma_l.  Fast channels lose the co-occurrence across a gap; the
slow channel retains it.  Kernel variants that drop the slow channel
(no_slow) therefore fail the task; variants that keep the slow timescale
(full) succeed.  The kernel is the binding writer.

This module is intentionally a *new* experiment, not a refactor of
d3_randomized.  d3_randomized's pre-bound content code remains
unchanged so its calibrated grid and milestone bundle stay valid.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import random
import statistics
from typing import Iterable

from ..digital import (
    DigitalDirector,
    DigitalEpisode,
    DigitalMemoryState,
    QueryConditionedMLPReadout,
    combined_readout_vector,
)
from .symbolic_induction import memory_config_for_variant


# ---------------------------------------------------------------------------
# Episode specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LearnedBindingSpec:
    """One independently generated learned-binding episode.

    `events` lists (slot, value) in the order they occur in the
    episode.  Each event produces a SLOT pulse, a VALUE pulse, then
    `gap` silent steps before the next event.  The `query_slot` field
    picks which slot the system is asked to recall at the terminal
    query.  The target value is the value from the most recent event
    whose slot matches the query.
    """

    events: tuple[tuple[int, int], ...]
    query_slot: int
    gap: int
    distractor_specs: tuple[tuple[int, int], ...] = ()

    @property
    def target(self) -> int:
        for slot, value in reversed(self.events):
            if slot == self.query_slot:
                return value
        raise ValueError("query_slot did not appear in events")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class LearnedBindingConfig:
    """Configuration for a single learned-binding experiment run."""

    variant: str = "full"
    slot_count: int = 4
    value_count: int = 4
    events_per_episode: int = 3
    gap: int = 8
    train_count: int = 96
    test_count: int = 12
    epochs: int = 96
    hidden_dim: int = 16
    learning_rate: float = 0.01
    pulse_steps: int = 2
    silence_steps: int = 1
    seed: int = 41
    distractor_count: int = 2
    binding_noise_std: float = 0.0
    freeze_query_slot: bool = True

    def __post_init__(self) -> None:
        if self.variant not in {"full", "no_slow", "collapsed_gamma"}:
            raise ValueError("unsupported memory variant")
        for name, value in (
            ("slot_count", self.slot_count),
            ("value_count", self.value_count),
            ("events_per_episode", self.events_per_episode),
            ("gap", self.gap),
            ("train_count", self.train_count),
            ("test_count", self.test_count),
            ("epochs", self.epochs),
            ("hidden_dim", self.hidden_dim),
            ("pulse_steps", self.pulse_steps),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.events_per_episode > self.slot_count:
            raise ValueError("events_per_episode must not exceed slot_count")
        if self.silence_steps < 0 or self.learning_rate <= 0.0:
            raise ValueError("invalid training or schedule setting")
        if self.distractor_count < 0 or self.binding_noise_std < 0.0:
            raise ValueError("distractor_count and binding_noise_std must be non-negative")

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Episode sampling
# ---------------------------------------------------------------------------


def _sample_specs(config: LearnedBindingConfig, *, seed: int, count: int) -> list[LearnedBindingSpec]:
    """Sample stratified, independent learned-binding episodes.

    Each episode draws a non-repeating multiset of slots of length
    `events_per_episode`, assigns a random value (0..value_count-1) to
    each, and pins the query to one of the participating slots so the
    target is always defined.  Distractor (slot, value) pairs are
    sampled independently and injected after the real events; they
    share the slot basis with the real events, so they interfere
    through state overlap, not through selection competition.
    """

    rng = random.Random(seed)
    specs: list[LearnedBindingSpec] = []
    for _ in range(count):
        slots = rng.sample(range(config.slot_count), config.events_per_episode)
        if config.freeze_query_slot:
            query_slot = slots[rng.randrange(config.events_per_episode)]
        else:
            query_slot = rng.randrange(config.slot_count)
        events = tuple((slot, rng.randrange(config.value_count)) for slot in slots)
        distractors = tuple(
            (rng.randrange(config.slot_count), rng.randrange(config.value_count))
            for _ in range(config.distractor_count)
        )
        specs.append(LearnedBindingSpec(events=events, query_slot=query_slot, gap=config.gap,
                                        distractor_specs=distractors))
    return specs


# ---------------------------------------------------------------------------
# Forcing pattern: slot/value halves (the key change vs d3_randomized)
# ---------------------------------------------------------------------------


def _state_dim(config: LearnedBindingConfig) -> int:
    return config.slot_count + config.value_count


def _slot_pulse(config: LearnedBindingConfig, slot: int) -> list[float]:
    forcing = [0.0] * _state_dim(config)
    forcing[slot] = 1.0
    return forcing


def _value_pulse(config: LearnedBindingConfig, value: int) -> list[float]:
    forcing = [0.0] * _state_dim(config)
    forcing[config.slot_count + value] = 1.0
    return forcing


def _query(config: LearnedBindingConfig, spec: LearnedBindingSpec) -> list[float]:
    """Query vector: pin slot coordinate, leave value coordinates empty.

    The readout must extract the value that the slow channel associated
    with `spec.query_slot` during the earlier event.
    """
    query = [0.0] * _state_dim(config)
    query[spec.query_slot] = 1.0
    return query

    query = [0.0] * _state_dim(config)
    query[spec.query_slot] = 1.0
    return query


# ---------------------------------------------------------------------------
# Episode construction
# ---------------------------------------------------------------------------


def _episode(config: LearnedBindingConfig, spec: LearnedBindingSpec) -> DigitalEpisode:
    """Build a single learned-binding episode.

    The forcing schedule emits, in order:

      For each (slot, value) in ``spec.events`` (the binding events):
        a. ``pulse_steps`` copies of the slot forcing  (slot coord only)
        b. ``silence_steps`` silence
        c. ``pulse_steps`` copies of the value forcing (value coord only)
        d. ``silence_steps`` silence
        e. ``spec.gap`` silence steps (the binding window)

      Then for each (slot, value) in ``spec.distractor_specs`` (interfering
      fillers): the same slot-then-value-then-gap structure.  The query slot
      coordinate in each filler differs from the query slot, so the
      filler partially excites the same value-coordinate range as the
      real events and creates state overlap interference.

      Finally, ``pulse_steps`` copies of the query forcing (slot coordinate
      only) so the readout can read off the integrated binding.

    No (slot, value) coordinate is ever excited together; the binding must be
    formed by the SOE memory dynamics.
    """
    noise_rng = random.Random(spec.query_slot * 1009 + len(spec.events) * 31 + spec.gap)

    def noisy(forcing: list[float]) -> list[float]:
        if not config.binding_noise_std:
            return forcing
        return [value + noise_rng.gauss(0.0, config.binding_noise_std) for value in forcing]

    episode_forcings: list[list[float]] = []
    episode_tokens: list[str | None] = []

    def add_pulse(token: str, forcing: list[float]) -> None:
        for _ in range(config.pulse_steps):
            episode_forcings.append(noisy(forcing))
            episode_tokens.append(token)
        for _ in range(config.silence_steps):
            episode_forcings.append([0.0] * _state_dim(config))
            episode_tokens.append(None)

    def add_silence(steps: int) -> None:
        for _ in range(steps):
            episode_forcings.append([0.0] * _state_dim(config))
            episode_tokens.append(None)

    def emit_binding(slot: int, value: int, label: str) -> None:
        add_pulse(f"SLOT_{slot}_{label}", _slot_pulse(config, slot))
        add_pulse(f"VALUE_{value}_{label}", _value_pulse(config, value))
        add_silence(spec.gap)

    for index, (slot, value) in enumerate(spec.events):
        emit_binding(slot, value, f"E{index}")
    for index, (slot, value) in enumerate(spec.distractor_specs):
        emit_binding(slot, value, f"D{index}")
    # Final query pulse
    add_pulse(f"QUERY_{spec.query_slot}", _query(config, spec))

    return DigitalEpisode(
        forcings=episode_forcings,
        tokens=episode_tokens,
        target=f"ACTION_{spec.target}",
        query=_query(config, spec),
        metadata={
            "events": list(spec.events),
            "query_slot": spec.query_slot,
            "gap": spec.gap,
            "distractor_specs": list(spec.distractor_specs),
            "pulse_steps": config.pulse_steps,
            "silence_steps": config.silence_steps,
        },
    )


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------


def _predict_variant(
    config: LearnedBindingConfig,
    spec: LearnedBindingSpec,
    variant: str,
    readout: QueryConditionedMLPReadout,
) -> tuple[int, float]:
    """Re-run a spec through a fresh director for the given variant and predict."""
    memory = memory_config_for_variant(variant)
    director = DigitalDirector(memory_config=memory, state_dim=_state_dim(config))
    result = director.run_episode(_episode(config, spec), experiment_name="d3_lb_eval")
    final = [100.0 * value for value in combined_readout_vector(result.final_state, memory)]
    return readout.predict([*final, *result.query])


def _frozen_slow_predict(
    config: LearnedBindingConfig,
    spec: LearnedBindingSpec,
    variant: str,
    readout: QueryConditionedMLPReadout,
) -> tuple[int, float]:
    """Predict after zeroing the slowest channel of the final state.

    If this still gets the answer right, the slow channel was not load-bearing
    for this spec.  If it gets the answer wrong where the full-state prediction
    was right, the slow channel was necessary.
    """
    memory = memory_config_for_variant(variant)
    director = DigitalDirector(memory_config=memory, state_dim=_state_dim(config))
    result = director.run_episode(_episode(config, spec), experiment_name="d3_lb_frozen")
    chi = [list(channel) for channel in result.final_state.chi]
    chi[0] = [0.0] * _state_dim(config)  # zero slowest channel
    zeroed_state = DigitalMemoryState(
        u=list(result.final_state.u),
        chi=chi,
        t=result.final_state.t,
    )
    final = [100.0 * value for value in combined_readout_vector(zeroed_state, memory)]
    return readout.predict([*final, *result.query])
    final = [100.0 * value for value in combined_readout_vector(zeroed_state, memory)]
    return readout.predict([*final, *result.query])


def run_learned_binding_experiment(config: LearnedBindingConfig) -> dict[str, object]:
    """Train and evaluate the learned-binding task on a single kernel variant.

    The runner:
      1. Samples disjoint train and test streams.
      2. Trains a readout on the train stream's final-state + query.
      3. Evaluates held-out accuracy, distractor-invariance, and a
         frozen-slow-channel control.

    To compare variants, run the experiment once per variant config and
    pair the results by seed.
    """
    train_specs = _sample_specs(config, seed=config.seed, count=config.train_count)
    test_specs = _sample_specs(config, seed=config.seed + 1_000_003, count=config.test_count)

    memory = memory_config_for_variant(config.variant)
    director = DigitalDirector(memory_config=memory, state_dim=_state_dim(config))
    train_rows = [
        (spec, director.run_episode(_episode(config, spec), experiment_name="d3_lb_train"))
        for spec in train_specs
    ]
    readout = QueryConditionedMLPReadout(
        2 * _state_dim(config), config.value_count, hidden_dim=config.hidden_dim, seed=config.seed
    )
    losses: list[float] = []
    for _ in range(config.epochs):
        for spec, result in train_rows:
            final = [100.0 * value for value in combined_readout_vector(result.final_state, memory)]
            features = [*final, *result.query]
            losses.append(readout.train_step(features, spec.target, learning_rate=config.learning_rate))

    predictions = [_predict_variant(config, spec, config.variant, readout) for spec in test_specs]
    accuracy = sum(prediction == spec.target for (prediction, _), spec in zip(predictions, test_specs)) / len(test_specs)
    mean_confidence = sum(confidence for _, confidence in predictions) / len(predictions)

    invariance_hits = 0
    for spec, (prediction, _) in zip(test_specs, predictions):
        shuffled = LearnedBindingSpec(
            events=spec.events,
            query_slot=spec.query_slot,
            gap=spec.gap,
            distractor_specs=tuple(
                ((slot + 1) % config.slot_count, value) for slot, value in spec.distractor_specs
            ),
        )
        shuffled_prediction, _ = _predict_variant(config, shuffled, config.variant, readout)
        if shuffled_prediction == prediction:
            invariance_hits += 1
    invariance_rate = invariance_hits / len(test_specs)

    frozen_predictions = [_frozen_slow_predict(config, spec, config.variant, readout) for spec in test_specs]
    frozen_accuracy = sum(
        prediction == spec.target
        for (prediction, _), spec in zip(frozen_predictions, test_specs)
    ) / len(test_specs)
    load_bearing = sum(
        1
        for spec, (full_pred, _), (frozen_pred, _) in zip(test_specs, predictions, frozen_predictions)
        if full_pred == spec.target and frozen_pred != spec.target
    ) / len(test_specs)

    return {
        "experiment_name": "d3_learned_binding",
        "config": config.to_mapping(),
        "variant": config.variant,
        "train_stream_seed": config.seed,
        "test_stream_seed": config.seed + 1_000_003,
        "independent_streams": True,
        "initial_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "test_accuracy": accuracy,
        "mean_confidence": mean_confidence,
        "distractor_invariance_rate": invariance_rate,
        "frozen_slow_accuracy": frozen_accuracy,
        "slow_channel_load_bearing_rate": load_bearing,
        "memory_diagnostics": {"stability_ratio": memory.stability_ratio(), "channel_count": memory.channel_count},
    }


def run_default_experiment() -> dict[str, object]:
    """Run a small end-to-end learned-binding experiment with sensible defaults."""
    return run_learned_binding_experiment(
        LearnedBindingConfig(
            slot_count=4,
            value_count=4,
            events_per_episode=2,
            train_count=64,
            test_count=32,
            epochs=8,
            gap=4,
            distractor_count=2,
            pulse_steps=1,
            silence_steps=1,
        )
    )
