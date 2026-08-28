"""E0 pilot: empirical comparison of adaptive-kernel initialisation policies.

Arms:
  recalibrate  -- initial w rescaled once to stability ratio 0.9*margin at thaw.
  rescale_once -- historical kernel kept; update rule rescales on first commit.
  frozen       -- matched frozen twin (calls the existing experiment directly).

Adaptive arms recompute episode rows only when w changes (commit block K=8),
apply one three-factor proposal per block from epoch-mean signals, and log
the full adaptation trace. Gap 16, loaded (sigma=0.10, 4 distractors).
"""
import json, math, time

from ..digital import DigitalDirector, DigitalMemoryState, QueryConditionedMLPReadout, combined_readout_vector
from ..learning import ThreeFactorUpdateConfig, propose_three_factor_update
from .d3_randomized import (
    D3RandomizedConfig, IndexedRuleEpisode, QuerySlotEligibility,
    _episode, _state_dim, _sample_specs, _completed_event_indices,
    run_d3_randomized_experiment,
)
from .symbolic_induction import memory_config_for_variant

BASE = dict(slot_count=4, gap=16, train_count=96, test_count=12, epochs=96,
            hidden_dim=16, binding_noise_std=0.10, distractor_count=4)
COMMIT_EVERY = 8

def snapshot_vec(result, memory):
    return [100.0 * v for v in combined_readout_vector(result.final_state, memory)]

def run_adaptive(policy: str, seed: int) -> dict:
    cfg = D3RandomizedConfig(variant="full", seed=seed, **BASE)
    memory = memory_config_for_variant("full")
    margin = ThreeFactorUpdateConfig().stability_margin
    init_w = list(memory.w)
    if policy == "recalibrate":
        scale = 0.9 * margin / memory.stability_ratio()
        memory.w = [v * scale for v in memory.w]
    director = DigitalDirector(memory_config=memory, state_dim=_state_dim(cfg))
    train_specs = _sample_specs(cfg, seed=cfg.seed, count=cfg.train_count)
    test_specs = _sample_specs(cfg, seed=cfg.seed + 1_000_003, count=cfg.test_count)

    def rows(specs):
        return [(s, director.run_episode(_episode(cfg, s), experiment_name="d3r_adaptive")) for s in specs]

    train_rows = rows(train_specs)
    eligibility = QuerySlotEligibility(cfg)
    first_features, _, _, _ = eligibility.features(train_rows[0][1], memory)
    readout = QueryConditionedMLPReadout(len(first_features), 2, hidden_dim=cfg.hidden_dim, seed=cfg.seed)

    w_trace = [{"epoch": 0, "w": list(memory.w), "ratio": memory.stability_ratio()}]
    rescales = commits = 0
    for epoch in range(cfg.epochs):
        if epoch > 0 and epoch % COMMIT_EVERY == 0:
            # One three-factor proposal from mean signals over the last epoch.
            acts = [[sum(abs(v) for v in ch) / len(ch) for ch in r.final_state.chi] for _, r in train_rows]
            mean_act = [sum(a[c] for a in acts) / len(acts) for c in range(memory.channel_count)]
            mean_loss = sum(losses[-cfg.train_count:]) / cfg.train_count
            mean_reward = sum(rewards[-cfg.train_count:]) / cfg.train_count
            proposal = propose_three_factor_update(
                weights=memory.w, gamma=memory.gamma, channel_activity=mean_act,
                prediction_error=mean_loss, td_error=[mean_reward] * memory.channel_count,
                config=ThreeFactorUpdateConfig(dt=memory.dt, leak_rate=memory.leak_rate),
            )
            memory.w = list(proposal.clipped_weights)
            commits += 1
            rescales += int(proposal.stability_rescaled)
            w_trace.append({"epoch": epoch, "w": list(memory.w), "ratio": memory.stability_ratio(),
                            "rescaled": proposal.stability_rescaled})
            train_rows = rows(train_specs)  # substrate changed: recompute
        losses, rewards = [], []
        for spec, result in train_rows:
            features, indices, attention, snapshots = eligibility.features(result, memory)
            loss, gradient = readout.train_step_with_input_gradient(features, spec.target, learning_rate=cfg.learning_rate)
            eligibility.update(result, indices=indices, attention=attention, snapshots=snapshots, input_gradient=gradient)
            losses.append(loss)
            rewards.append(1.0 if readout.predict(features)[0] == spec.target else 0.0)

    def predict(spec):
        result = director.run_episode(_episode(cfg, spec), experiment_name="d3r_adaptive_eval")
        features, _, _, _ = eligibility.features(result, memory)
        prediction, confidence = readout.predict(features)
        attention, uniform = eligibility.target_attention(result, memory)
        return prediction, confidence, attention - uniform

    predictions = [predict(s) for s in test_specs]
    accuracy = sum(p == s.target for (p, _, _), s in zip(predictions, test_specs)) / len(test_specs)
    flips = inv = 0
    for spec, (p, _, _) in zip(test_specs, predictions):
        tv = list(spec.assignments); tv[spec.query_slot] ^= 1
        flips += predict(IndexedRuleEpisode(spec.mode, tuple(tv), spec.order, spec.query_slot))[0] != p
        ds = next(s for s in spec.order if s != spec.query_slot)
        dv = list(spec.assignments); dv[ds] ^= 1
        inv += predict(IndexedRuleEpisode(spec.mode, tuple(dv), spec.order, spec.query_slot))[0] == p
    margins = [m for _, _, m in predictions]
    return {
        "arm": policy, "seed": seed,
        "test_accuracy": accuracy,
        "mean_attention_margin": sum(margins) / len(margins),
        "flip_rate": flips / len(test_specs), "invariance": inv / len(test_specs),
        "commits": commits, "rescales": rescales,
        "init_w": init_w, "final_w": list(memory.w),
        "init_ratio": init_w and None, "final_ratio": memory.stability_ratio(),
        "w_trace": w_trace,
    }

def main(output_dir: str = "/tmp", seeds: range = range(101, 111)) -> None:
    """Run the three-arm E0 pilot, printing one JSON row per run and writing
    per-seed detail (including w_trace) to ``output_dir``."""
    for policy in ("recalibrate", "rescale_once", "frozen"):
        for seed in seeds:
            t0 = time.time()
            if policy == "frozen":
                cfg = D3RandomizedConfig(variant="full", seed=seed, **BASE)
                res = run_d3_randomized_experiment(cfg)
                out = {"arm": policy, "seed": seed, "test_accuracy": res["test_accuracy"],
                       "mean_attention_margin": res["mean_target_attention_margin"],
                       "flip_rate": res["interventions"]["queried_value_action_flip_rate"],
                       "invariance": res["interventions"]["unqueried_distractor_action_invariance_rate"],
                       "commits": 0, "rescales": 0,
                       "final_ratio": res["memory_diagnostics"]["stability_ratio"],
                       "final_w": res["config"].get("w", None)}
            else:
                out = run_adaptive(policy, seed)
            out["seconds"] = round(time.time() - t0, 1)
            print(json.dumps({k: v for k, v in out.items() if k != "w_trace"}), flush=True)
            with open(f"{output_dir}/e0_{policy}_{seed}.json", "w") as fh:
                json.dump(out, fh, indent=1)
    print("E0_PILOT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()