"""D4A re-baseline: run the loaded D3-R task through the ABI flat director.

Establishes the ABI scalar-chi substrate's baseline numbers before wiring
hierarchy arms.  Uses the same eligibility + readout + interventions as D3-R,
but replaces the Python ``DigitalDirector`` with a flat
``HierarchicalDigitalDirector`` (single level, no edges).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..digital import (
    DigitalMemoryState,
    QueryConditionedMLPReadout,
    combined_readout_vector,
)
from ..digital.hierarchy_director import HierarchicalDigitalDirector
from ..digital.traces import DigitalTrace
from ..ffi import load_library
from .d3_randomized import (
    D3RandomizedConfig,
    IndexedRuleEpisode,
    QuerySlotEligibility,
    _episode,
    _sample_specs,
)
from .symbolic_induction import memory_config_for_variant
from gfe_ctypes import (
    GFE_C_COUPLING_FORM_B,
    GFE_C_HIERARCHICAL_RELATION_BOTTOM_UP,
    GFE_C_HIERARCHICAL_RELATION_TOP_DOWN,
)


@dataclass(slots=True)
class _CompatResult:
    """Minimal compat wrapper so D3-R eligibility sees ABI results."""
    final_state: DigitalMemoryState
    trace: DigitalTrace
    target: str | None
    query: list[float]
    metadata: dict[str, object]


def _bcast_state(abi_state: DigitalMemoryState, dim: int) -> DigitalMemoryState:
    u = list(abi_state.u)
    chi = [[c[0]] * dim for c in abi_state.chi]
    return DigitalMemoryState(u=u, chi=chi, t=abi_state.t)


def _bcast_trace(abi_trace: DigitalTrace, dim: int) -> DigitalTrace:
    from ..digital.traces import DigitalTraceStep
    steps = []
    for step in abi_trace.steps:
        chi = [[c[0]] * dim for c in step.chi]
        steps.append(DigitalTraceStep(
            step=step.step, t=step.t, token=step.token,
            forcing=list(step.forcing), u=list(step.u), chi=chi,
            metadata=dict(step.metadata),
        ))
    return DigitalTrace(
        experiment_name=abi_trace.experiment_name,
        steps=steps,
        metadata=dict(abi_trace.metadata),
    )


def _compat(abi_result) -> _CompatResult:
    dim = len(abi_result.final_state.u)
    return _CompatResult(
        final_state=_bcast_state(abi_result.final_state, dim),
        trace=_bcast_trace(abi_result.trace, dim),
        target=abi_result.target,
        query=abi_result.query,
        metadata=abi_result.metadata,
    )


def run_d4a_probe(
    config: D3RandomizedConfig,
    *,
    arm: str = "flat_abi",
    bottom_up_gain: float = 0.4,
    top_down_gain: float = 2.0,
    context_gamma: list[float] | None = None,
    context_leak: float = 0.5,
) -> dict[str, object]:
    """Run the loaded D3-R task through the ABI with the specified hierarchy arm.

    Arms:
      flat_abi   -- single level, no edges (the re-baseline reference).
      hier_bu    -- two levels, bottom-up only (context sees events, gives
                    nothing back).  Level-0 dynamics are identical to
                    flat_abi; this is a control arm.
      hier_bu_td -- two levels, bottom-up + top-down.  Level-0 receives
                    kernel modulation from the context level: the active
                    kernel's w is normalised (stronger aggregate feedback)
                    and gamma is uniformly diluted by exp(-gain*drive).
    """
    memory = memory_config_for_variant(config.variant)
    lib = load_library(None)
    state_dim = 2 * config.slot_count + 2

    level0 = {
        "name": "fast",
        "gamma": list(memory.gamma),
        "w": list(memory.w),
        "u": [0.0] * state_dim,
        "chi": [0.0] * memory.channel_count,
        "dt": memory.dt,
        "linear_decay": [memory.leak_rate] * state_dim,
        "forcing_bias": [0.0] * state_dim,
        "form": GFE_C_COUPLING_FORM_B,
        "coupling_index": 0,
    }

    if arm == "flat_abi":
        levels, edges = [level0], []
    elif arm == "w_boost_flat":
        # Flat arm with the top-down materialisation's constant w-normalisation
        # applied statically (no hierarchy, no gamma dilation).  Isolates the
        # always-on feedback-strength component of hier_bu_td: hier_bu_td's
        # active w is also the normalised spec_w (drive cancels), so the
        # hier_bu_td − w_boost_flat difference is purely the context-adaptive
        # gamma component.
        total_w = sum(memory.w)
        level0 = dict(level0)
        level0["w"] = [wk / total_w for wk in memory.w]
        levels, edges = [level0], []
    elif arm in ("hier_bu", "hier_bu_td"):
        cg = context_gamma or [0.35, 0.08]
        cw = [0.05, 0.02][: len(cg)] or [0.02] * len(cg)
        level1 = {
            "name": "context",
            "gamma": list(cg),
            "w": list(cw),
            "u": [0.0] * state_dim,
            "chi": [0.0] * len(cg),
            "dt": memory.dt,
            "linear_decay": [context_leak] * state_dim,
            "forcing_bias": [0.0] * state_dim,
            "form": GFE_C_COUPLING_FORM_B,
            "coupling_index": 0,
        }
        edges = [{
            "source_level": 0,
            "target_level": 1,
            "relation": GFE_C_HIERARCHICAL_RELATION_BOTTOM_UP,
            "gain": bottom_up_gain,
            "normalize_weights": True,
        }]
        if arm == "hier_bu_td":
            edges.append({
                "source_level": 1,
                "target_level": 0,
                "relation": GFE_C_HIERARCHICAL_RELATION_TOP_DOWN,
                "gain": top_down_gain,
                "normalize_weights": True,
            })
        levels = [level0, level1]
    else:
        raise ValueError(f"unknown arm: {arm}")

    director = HierarchicalDigitalDirector(
        lib=lib, levels=levels, edges=edges, state_dim=state_dim, memory_config=memory,
    )

    train_specs = _sample_specs(config, seed=config.seed, count=config.train_count)
    test_specs = _sample_specs(config, seed=config.seed + 1_000_003, count=config.test_count)

    def run_rows(specs):
        return [
            (s, director.run_episode(_episode(config, s), experiment_name="d4a_rebaseline"))
            for s in specs
        ]

    train_rows = run_rows(train_specs)
    eligibility = QuerySlotEligibility(config)
    first_compat = _compat(train_rows[0][1])
    first_features, _, _, _ = eligibility.features(first_compat, memory)
    readout = QueryConditionedMLPReadout(
        len(first_features), 2, hidden_dim=config.hidden_dim, seed=config.seed,
    )

    losses: list[float] = []
    for _ in range(config.epochs):
        for spec, raw in train_rows:
            result = _compat(raw)
            features, indices, attention, snapshots = eligibility.features(result, memory)
            loss, grad = readout.train_step_with_input_gradient(
                features, spec.target, learning_rate=config.learning_rate,
            )
            eligibility.update(result, indices=indices, attention=attention,
                               snapshots=snapshots, input_gradient=grad)
            losses.append(loss)

    def predict(spec):
        raw = director.run_episode(_episode(config, spec), experiment_name="d4a_rebaseline_eval")
        result = _compat(raw)
        features, _, _, _ = eligibility.features(result, memory)
        prediction, confidence = readout.predict(features)
        attention, uniform = eligibility.target_attention(result, memory)
        return prediction, confidence, attention - uniform

    predictions = [predict(spec) for spec in test_specs]
    accuracy = sum(p == spec.target for (p, _, _), spec in zip(predictions, test_specs)) / len(test_specs)

    target_flips = 0
    distractor_invariant = 0
    for spec, (pred, _, _) in zip(test_specs, predictions):
        tv = list(spec.assignments); tv[spec.query_slot] ^= 1
        target_flips += predict(IndexedRuleEpisode(spec.mode, tuple(tv), spec.order, spec.query_slot))[0] != pred
        ds = next(s for s in spec.order if s != spec.query_slot)
        dv = list(spec.assignments); dv[ds] ^= 1
        distractor_invariant += predict(IndexedRuleEpisode(spec.mode, tuple(dv), spec.order, spec.query_slot))[0] == pred

    margins = [m for _, _, m in predictions]
    return {
        "experiment_name": "d4a_hierarchy_probe",
        "config": config.to_mapping(),
        "variant": config.variant,
        "arm": arm,
        "substrate": "abi_scalar_chi",
        "train_stream_seed": config.seed,
        "test_stream_seed": config.seed + 1_000_003,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "test_accuracy": accuracy,
        "mean_target_attention_margin": sum(margins) / len(margins),
        "target_attention_exceeds_uniform": sum(margins) / len(margins) > 0.0,
        "interventions": {
            "queried_value_action_flip_rate": target_flips / len(test_specs),
            "unqueried_distractor_action_invariance_rate": distractor_invariant / len(test_specs),
        },
        "memory_diagnostics": {"stability_ratio": memory.stability_ratio(), "channel_count": memory.channel_count},
    }


def run_d4a_rebaseline(config: D3RandomizedConfig) -> dict[str, object]:
    """Backward-compat wrapper: flat-ABI arm of the D4A probe."""
    return run_d4a_probe(config, arm="flat_abi")