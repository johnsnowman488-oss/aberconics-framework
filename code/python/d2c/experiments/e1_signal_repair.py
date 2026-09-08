"""Stage E1-signal-repair harness: does a repaired three-factor signal move w?

Arms:
  frozen           -- matched frozen twin (existing D3-R experiment, no commits).
  block_scalar     -- legacy E1 adaptive behaviour: one commit every
                      COMMIT_EVERY epochs from epoch-mean unsigned scalar
                      signals (regression anchor for the E1 bounded null).
  sequential_signed -- per-epoch commits against the drifting w, using the
                      D2C reference's signed Hebbian term (chi * centered
                      prediction error) and error-modulated decay
                      (beta = beta0 / (1 + |eps|)).

Pre-declared Stage 1 mechanism gate (accuracy is NOT the gate): the adaptive
arm must produce seed-differentiated, channel-asymmetric weight trajectories.
The E1 bounded null showed identical final w vectors (~0.1654/0.1134/0.0503)
across seeds and arms; if the repaired signal still cannot differentiate, the
three-factor rule needs redesign rather than a new task.
"""

import json
import math
import time

from ..digital import DigitalDirector, QueryConditionedMLPReadout
from ..learning import ThreeFactorUpdateConfig, propose_three_factor_update
from .d3_randomized import (
    D3RandomizedConfig, IndexedRuleEpisode, QuerySlotEligibility,
    _episode, _state_dim, _sample_specs, run_d3_randomized_experiment,
)
from .symbolic_induction import memory_config_for_variant

BASE = dict(slot_count=4, gap=8, train_count=96, test_count=12, epochs=96,
            hidden_dim=16, binding_noise_std=0.10, distractor_count=4)
COMMIT_EVERY = 8


def _channel_activity(result, memory, signed: bool) -> list[float]:
    """Per-channel mean chi across state dimensions for one episode."""
    means = []
    for chi in result.final_state.chi:
        value = sum(chi) / len(chi)
        means.append(value if signed else abs(value))
    return means


def run_adaptive(arm: str, seed: int, gap: int) -> dict:
    if arm not in {"block_scalar", "sequential_signed"}:
        raise ValueError(f"unknown adaptive arm: {arm}")
    cfg = D3RandomizedConfig(variant="full", seed=seed, gap=gap, **{
        k: v for k, v in BASE.items() if k != "gap"})
    memory = memory_config_for_variant("full")
    init_w = list(memory.w)
    director = DigitalDirector(memory_config=memory, state_dim=_state_dim(cfg))
    train_specs = _sample_specs(cfg, seed=cfg.seed, count=cfg.train_count)
    test_specs = _sample_specs(cfg, seed=cfg.seed + 1_000_003, count=cfg.test_count)

    def rows(specs):
        return [(s, director.run_episode(_episode(cfg, s), experiment_name="d3r_adaptive")) for s in specs]

    train_rows = rows(train_specs)
    eligibility = QuerySlotEligibility(cfg)
    first_features, _, _, _ = eligibility.features(train_rows[0][1], memory)
    readout = QueryConditionedMLPReadout(len(first_features), 2, hidden_dim=cfg.hidden_dim, seed=cfg.seed)

    signed = arm == "sequential_signed"
    config = ThreeFactorUpdateConfig(
        dt=memory.dt, leak_rate=memory.leak_rate,
        error_modulated_decay=signed,
    )
    w_trace = [{"epoch": 0, "w": list(memory.w), "ratio": memory.stability_ratio()}]
    rescales = commits = 0
    running_loss = 0.0
    for epoch in range(cfg.epochs):
        commit = epoch > 0 and (epoch % COMMIT_EVERY == 0 if arm == "block_scalar" else True)
        if commit:
            acts = [_channel_activity(r, memory, signed=signed) for _, r in train_rows]
            mean_act = [sum(a[c] for a in acts) / len(acts) for c in range(memory.channel_count)]
            mean_loss = sum(losses[-cfg.train_count:]) / cfg.train_count
            mean_reward = sum(rewards[-cfg.train_count:]) / cfg.train_count
            if signed:
                # Zero-centred signed error: episodes better than the running
                # baseline strengthen channels, worse ones weaken them.
                centred = mean_loss - running_loss
                hebbian = [a * centred for a in mean_act]
            else:
                hebbian = None
            proposal = propose_three_factor_update(
                weights=memory.w, gamma=memory.gamma, channel_activity=mean_act,
                prediction_error=mean_loss, td_error=[mean_reward] * memory.channel_count,
                hebbian_signal=hebbian, config=config,
            )
            memory.w = list(proposal.clipped_weights)
            commits += 1
            rescales += int(proposal.stability_rescaled)
            w_trace.append({"epoch": epoch, "w": list(memory.w), "ratio": memory.stability_ratio(),
                            "rescaled": proposal.stability_rescaled})
            if arm == "block_scalar":
                train_rows = rows(train_specs)  # substrate changed: recompute
        losses, rewards = [], []
        for spec, result in train_rows:
            features, indices, attention, snapshots = eligibility.features(result, memory)
            loss, gradient = readout.train_step_with_input_gradient(features, spec.target, learning_rate=cfg.learning_rate)
            eligibility.update(result, indices=indices, attention=attention, snapshots=snapshots, input_gradient=gradient)
            losses.append(loss)
            rewards.append(1.0 if readout.predict(features)[0] == spec.target else 0.0)
        if signed:
            running_loss = (running_loss * epoch + sum(losses) / len(losses)) / (epoch + 1)
    if signed:
        # Substrate drifted every epoch: refresh once before evaluation.
        train_rows = rows(train_specs)
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
    ratios = [entry["ratio"] for entry in w_trace]
    drift = [b - a for a, b in zip(init_w, memory.w)]
    return {
        "arm": arm, "seed": seed, "gap": gap,
        "test_accuracy": accuracy,
        "mean_attention_margin": sum(margins) / len(margins),
        "flip_rate": flips / len(test_specs), "invariance": inv / len(test_specs),
        "commits": commits, "rescales": rescales,
        "init_w": init_w, "final_w": list(memory.w),
        "weight_drift": drift,
        "final_ratio": memory.stability_ratio(),
        "max_ratio": max(ratios),
        "w_trace": w_trace,
    }


def _summarise(rows: list[dict]) -> dict:
    """Cross-seed dispersion diagnostics per arm (the Stage 1 gate metric)."""
    by_arm: dict[str, list[dict]] = {}
    for row in rows:
        by_arm.setdefault(row["arm"], []).append(row)
    summary = {}
    for arm, group in sorted(by_arm.items()):
        final = [r["final_w"] for r in group if r.get("final_w")]
        if not final:
            continue
        channels = len(final[0])
        mean_w = [sum(w[c] for w in final) / len(final) for c in range(channels)]
        std_w = [math.sqrt(sum((w[c] - mean_w[c]) ** 2 for w in final) / len(final)) for c in range(channels)]
        summary[arm] = {
            "runs": len(group),
            "mean_accuracy": sum(r["test_accuracy"] for r in group) / len(group),
            "mean_final_w": [round(v, 6) for v in mean_w],
            "std_final_w_across_seeds": [round(v, 8) for v in std_w],
            "mean_drift": [
                round(sum(r["weight_drift"][c] for r in group if "weight_drift" in r) / len(group), 6)
                for c in range(channels)
            ],
            "rescales": sum(r["rescales"] for r in group),
        }
    return summary


def main(output_dir: str = "/tmp", seeds: range = range(101, 111),
         arms: tuple[str, ...] = ("frozen", "block_scalar", "sequential_signed"),
         gap: int = BASE["gap"]) -> None:
    """Run the Stage 1 signal-repair comparison, printing one JSON row per run."""
    collected: list[dict] = []
    for arm in arms:
        for seed in seeds:
            t0 = time.time()
            if arm == "frozen":
                cfg = D3RandomizedConfig(variant="full", seed=seed, gap=gap, **{
                    k: v for k, v in BASE.items() if k != "gap"})
                res = run_d3_randomized_experiment(cfg)
                out = {"arm": arm, "seed": seed, "gap": gap,
                       "test_accuracy": res["test_accuracy"],
                       "mean_attention_margin": res["mean_target_attention_margin"],
                       "flip_rate": res["interventions"]["queried_value_action_flip_rate"],
                       "invariance": res["interventions"]["unqueried_distractor_action_invariance_rate"],
                       "commits": 0, "rescales": 0,
                       "final_ratio": res["memory_diagnostics"]["stability_ratio"],
                       "final_w": res["config"].get("w", None)}
            else:
                out = run_adaptive(arm, seed, gap)
            out["seconds"] = round(time.time() - t0, 1)
            collected.append(out)
            print(json.dumps({k: v for k, v in out.items() if k != "w_trace"}), flush=True)
            with open(f"{output_dir}/stage1_{arm}_{seed}.json", "w") as fh:
                json.dump(out, fh, indent=1)
    print(json.dumps({"summary": _summarise(collected)}), flush=True)
    print("STAGE1_SIGNAL_REPAIR_COMPLETE", flush=True)


if __name__ == "__main__":
    main()

