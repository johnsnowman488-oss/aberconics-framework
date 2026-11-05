"""E0 bAbI QA convergence test — numpy MLP readout sweep on QA1.

Runs the tuned experiment that Step_Context.md identified as the next action:
sweep hidden dim, epoch count, and learning rate at dim=16 with the fast
numpy-ized readout, then report seed-level accuracy with CIs.
"""
import math
import random
import sys
import time
from pathlib import Path

from d2c.digital.babi import (
    BabiDataset,
    babi_story_to_episode,
    build_babi_answer_set,
    build_babi_bridge,
    build_babi_memory_config,
    build_babi_vocabulary,
)
from d2c.digital.bridge import seed_memory_state, step_memory
from d2c.digital.readout import NumpyMLPReadout, query_conditioned_features_np

DATA = Path(__file__).resolve().parents[2] / "d2c" / "progress" / "babi_data"


def evaluate(readout, test_ds, bridge, memory_cfg, answer_set, dim):
    correct = 0
    for story in test_ds.stories:
        ep = babi_story_to_episode(story, bridge, answer_set)
        state = seed_memory_state(state_dim=dim, channel_count=memory_cfg.channel_count)
        for f in ep.forcings:
            state = step_memory(state, f, memory_cfg)
        feats = query_conditioned_features_np(state, memory_cfg, ep.query)
        pred, _ = readout.predict(feats)
        if answer_set[pred] == ep.target:
            correct += 1
    return correct / len(test_ds.stories)


def run_one(config_label, dim, hidden_dim, epochs, train_n, lr, n_seeds,
            train_ds, test_ds, bridge, memory_cfg, answer_set, feat_dim):
    """Run one configuration across seeds, return list of final accuracies."""
    seed_accs = []
    for seed in range(n_seeds):
        rng = random.Random(seed)
        readout = NumpyMLPReadout(
            input_dim=feat_dim, output_dim=len(answer_set),
            hidden_dim=hidden_dim, seed=seed, lr=lr,
        )
        train_stories = train_ds.stories[:train_n]
        t0 = time.time()
        for ep_idx in range(epochs):
            stories = list(train_stories)
            rng.shuffle(stories)
            for story in stories:
                ep = babi_story_to_episode(story, bridge, answer_set)
                state = seed_memory_state(state_dim=dim, channel_count=memory_cfg.channel_count)
                for f in ep.forcings:
                    state = step_memory(state, f, memory_cfg)
                feats = query_conditioned_features_np(state, memory_cfg, ep.query)
                target_idx = answer_set.index(ep.target)
                readout.train_step(feats, target_idx)

        final_acc = evaluate(readout, test_ds, bridge, memory_cfg, answer_set, dim)
        elapsed = time.time() - t0
        seed_accs.append(final_acc)
        print(f"  [{config_label}] seed {seed}: {final_acc:.3f} ({elapsed:.0f}s)")

    return seed_accs


def main():
    train_ds = BabiDataset.from_jsonl(DATA / "babi_train.jsonl").filter_task(1)
    test_ds = BabiDataset.from_jsonl(DATA / "babi_test.jsonl").filter_task(1)
    vocab = build_babi_vocabulary(train_ds.stories + test_ds.stories)
    answer_set = build_babi_answer_set(train_ds.stories)
    n_answers = len(answer_set)
    dim = 16
    bridge = build_babi_bridge(vocab, dim=dim)
    memory_cfg = build_babi_memory_config(variant="full")
    feat_dim = dim + dim + dim * dim  # 288

    print(f"QA1: {len(train_ds.stories)} train, {len(test_ds.stories)} test, "
          f"{vocab.size} vocab, {n_answers} answers, feat_dim={feat_dim}")
    print()

    # Sweep configurations
    sweep = [
        # (label, hidden_dim, epochs, train_n, lr, seeds)
        ("hd128_ep20_lr05", 128, 20, 500, 0.05, 5),
        ("hd128_ep30_lr05", 128, 30, 500, 0.05, 5),
        ("hd128_ep20_lr02", 128, 20, 500, 0.02, 5),
        ("hd256_ep20_lr05", 256, 20, 500, 0.05, 5),
        # Try full training set
        ("hd128_ep10_lr05_full900", 128, 10, 900, 0.05, 5),
    ]

    results = {}
    for label, hd, ep, tn, lr, ns in sweep:
        accs = run_one(label, dim, hd, ep, tn, lr, ns,
                       train_ds, test_ds, bridge, memory_cfg, answer_set, feat_dim)
        mean = sum(accs) / len(accs)
        se = (math.sqrt(sum((a - mean) ** 2 for a in accs) / len(accs))
              / math.sqrt(len(accs))) if len(accs) > 1 else 0.0
        results[label] = (mean, se, accs)
        print(f"  RESULT [{label}]: {mean:.3f} +/- {se:.3f}")
        print()

    # Find best config and run ablations
    best_label = max(results, key=lambda k: results[k][0])
    best_mean, best_se, _ = results[best_label]
    print(f"\n=== Best config: {best_label} ({best_mean:.3f} +/- {best_se:.3f}) ===")
    print()

    # Parse best config params
    best_cfg = next(c for c in sweep if c[0] == best_label)
    _, best_hd, best_ep, best_tn, best_lr, best_ns = best_cfg

    # Ablation sweep: full / no_slow / collapsed_gamma
    for variant in ["full", "no_slow", "collapsed_gamma"]:
        ablation_cfg = build_babi_memory_config(variant=variant)
        ablation_bridge = build_babi_bridge(vocab, dim=dim)
        accs = run_one(
            f"ablation_{variant}", dim, best_hd, best_ep, best_tn, best_lr, best_ns,
            train_ds, test_ds, ablation_bridge, ablation_cfg, answer_set, feat_dim,
        )
        mean = sum(accs) / len(accs)
        se = (math.sqrt(sum((a - mean) ** 2 for a in accs) / len(accs))
              / math.sqrt(len(accs))) if len(accs) > 1 else 0.0
        print(f"  ABLATION [{variant}]: {mean:.3f} +/- {se:.3f}")
        print()


if __name__ == "__main__":
    main()