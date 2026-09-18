"""D3-R-LB on the ABI substrate.

Runs the D3-R-LB learned-binding task through the ABI flat director to
test whether the full-vs-no_slow action separation survives when the
kernel must self-form associations rather than receive pre-bound forcing.

When *director* is ``None`` (the contract default), this replicates the
D3-R-LB v3 results on the Python vector-chi substrate.  When an ABI flat
``HierarchicalDigitalDirector`` is provided, the experiment runs on the
scalar-chi substrate where the slow channel's retention is load-bearing.
"""

from __future__ import annotations

from ..digital import (
    DigitalDirector,
    DigitalMemoryState,
    QueryConditionedMLPReadout,
    combined_readout_vector,
)
from ..digital.hierarchy_director import HierarchicalDigitalDirector
from ..ffi import load_library
from .d3_learned_binding import (
    LearnedBindingConfig,
    LearnedBindingSpec,
    _episode,
    _query,
    _sample_specs,
    _state_dim,
)
from .d4a_probe import _bcast_state, _compat
from .symbolic_induction import memory_config_for_variant


def _episode_state(result, director, state_dim, memory):
    """Extract final readout features from an episode result."""
    if isinstance(director, HierarchicalDigitalDirector):
        fs = _bcast_state(_compat(result).final_state, state_dim)
    else:
        fs = result.final_state
    return [100.0 * v for v in combined_readout_vector(fs, memory)]


def _frozen_slow_predict(config, spec, variant, readout, director=None):
    """Predict after zeroing the slowest channel of the final state."""
    memory = memory_config_for_variant(variant)
    state_dim = _state_dim(config)
    if director is None:
        director = DigitalDirector(memory_config=memory, state_dim=state_dim)
    result = director.run_episode(_episode(config, spec), experiment_name="d3_lb_frozen")
    if isinstance(director, HierarchicalDigitalDirector):
        cr = _compat(result)
        chi = list(cr.final_state.chi)
        chi[0] = [0.0] * state_dim
        zeroed = DigitalMemoryState(u=list(cr.final_state.u), chi=chi, t=cr.final_state.t)
    else:
        chi = [list(ch) for ch in result.final_state.chi]
        chi[0] = [0.0] * state_dim
        zeroed = DigitalMemoryState(u=list(result.final_state.u), chi=chi, t=result.final_state.t)
    final = [100.0 * v for v in combined_readout_vector(zeroed, memory)]
    return readout.predict([*final, *result.query])


def run_d3_lb_abi(config: LearnedBindingConfig, *, director=None) -> dict[str, object]:
    """Run the D3-R-LB task through the specified director.

    When *director* is ``None``, uses the Python contract substrate.  When
    an ABI ``HierarchicalDigitalDirector`` is passed, the same task runs
    through the scalar-chi ABI with chi broadcast for the readout.
    """
    memory = memory_config_for_variant(config.variant)
    state_dim = _state_dim(config)
    if director is None:
        director = DigitalDirector(memory_config=memory, state_dim=state_dim)
    is_abi = isinstance(director, HierarchicalDigitalDirector)

    train_specs = _sample_specs(config, seed=config.seed, count=config.train_count)
    test_specs  = _sample_specs(config, seed=config.seed + 1_000_003, count=config.test_count)
    train_rows = [
        (spec, director.run_episode(_episode(config, spec), experiment_name="d3_lb_train"))
        for spec in train_specs
    ]

    readout = QueryConditionedMLPReadout(
        2 * state_dim, config.value_count, hidden_dim=config.hidden_dim, seed=config.seed,
    )

    losses: list[float] = []
    for _ in range(config.epochs):
        for spec, result in train_rows:
            final = _episode_state(result, director, state_dim, memory)
            features = [*final, *_query(config, spec)]
            losses.append(readout.train_step(features, spec.target, learning_rate=config.learning_rate))

    def predict(spec):
        result = director.run_episode(_episode(config, spec), experiment_name="d3_lb_eval")
        final = _episode_state(result, director, state_dim, memory)
        return readout.predict([*final, *result.query])

    predictions = [predict(spec) for spec in test_specs]
    accuracy = sum(p == spec.target for (p, _), spec in zip(predictions, test_specs)) / len(test_specs)

    invariance_hits = 0
    for spec, (pred, _) in zip(test_specs, predictions):
        shuffled = LearnedBindingSpec(
            events=spec.events, query_slot=spec.query_slot, gap=spec.gap,
            distractor_specs=tuple(((s + 1) % config.slot_count, v) for s, v in spec.distractor_specs),
        )
        shuf_pred, _ = predict(shuffled)
        invariance_hits += (shuf_pred == pred)
    invariance_rate = invariance_hits / len(test_specs)

    frozen = [_frozen_slow_predict(config, spec, config.variant, readout, director) for spec in test_specs]
    frozen_accuracy = sum(p == spec.target for (p, _), spec in zip(frozen, test_specs)) / len(test_specs)
    load_bearing = sum(
        1 for spec, (p, _), (f, _) in zip(test_specs, predictions, frozen)
        if p == spec.target and f != spec.target
    ) / len(test_specs)

    return {
        "experiment_name": "d3_lb_abi" if is_abi else "d3_learned_binding",
        "config": config.to_mapping(),
        "variant": config.variant,
        "substrate": "abi_scalar_chi" if is_abi else "python_vector_chi",
        "train_stream_seed": config.seed,
        "test_stream_seed": config.seed + 1_000_003,
        "initial_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "test_accuracy": accuracy,
        "distractor_invariance_rate": invariance_rate,
        "frozen_slow_accuracy": frozen_accuracy,
        "slow_channel_load_bearing_rate": load_bearing,
        "memory_diagnostics": {"stability_ratio": memory.stability_ratio(), "channel_count": memory.channel_count},
    }