"""E0 bAbI QA1 ablation at dim=32 — does more capacity change the ordering?

Best config from dim=16 sweep: hd=128, ep=20, lr=0.02, 500 train stories, 5 seeds.
Running the same ablation (full / no_slow / collapsed_gamma) at dim=32.
"""
import math
import random
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


def evaluate(readout, test_ds, bridge, memory_cfg, dim):
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


train_ds = BabiDataset.from_jsonl(DATA / "babi_train.jsonl").filter_task(1)
test_ds = BabiDataset.from_jsonl(DATA / "babi_test.jsonl").filter_task(1)
vocab = build_babi_vocabulary(train_ds.stories + test_ds.stories)
answer_set = build_babi_answer_set(train_ds.stories)
n_answers = len(answer_set)

dim = 32
feat_dim = dim + dim + dim * dim  # 32+32+1024 = 1088
bridge = build_babi_bridge(vocab, dim=dim)

print(f"QA1 dim={dim}: {len(train_ds.stories)} train, {len(test_ds.stories)} test, "
      f"{vocab.size} vocab, {n_answers} answers, feat_dim={feat_dim}")
print()

# Config: best from dim=16 sweep
HIDDEN_DIM = 128
EPOCHS = 20
TRAIN_N = 500
LR = 0.02
N_SEEDS = 5

for variant in ["full", "no_slow", "collapsed_gamma"]:
    cfg = build_babi_memory_config(variant=variant)
    accs = []
    for seed in range(N_SEEDS):
        rng = random.Random(seed)
        readout = NumpyMLPReadout(
            input_dim=feat_dim, output_dim=n_answers,
            hidden_dim=HIDDEN_DIM, seed=seed, lr=LR,
        )
        train_stories = train_ds.stories[:TRAIN_N]
        t0 = time.time()
        for _ in range(EPOCHS):
            stories = list(train_stories)
            rng.shuffle(stories)
            for story in stories:
                ep = babi_story_to_episode(story, bridge, answer_set)
                state = seed_memory_state(state_dim=dim, channel_count=cfg.channel_count)
                for f in ep.forcings:
                    state = step_memory(state, f, cfg)
                feats = query_conditioned_features_np(state, cfg, ep.query)
                readout.train_step(feats, answer_set.index(ep.target))
        final = evaluate(readout, test_ds, bridge, cfg, dim)
        elapsed = time.time() - t0
        accs.append(final)
        print(f"  {variant} seed {seed}: {final:.3f} ({elapsed:.0f}s)")
    mean = sum(accs) / len(accs)
    se = math.sqrt(sum((a - mean) ** 2 for a in accs) / len(accs)) / math.sqrt(len(accs))
    print(f"  => {variant}: {mean:.3f} +/- {se:.3f}")
    print()

# Also run dim=16 reference alongside for direct comparison
print("--- dim=16 reference (same config) ---")
dim16 = 16
feat16 = dim16 + dim16 + dim16 * dim16
bridge16 = build_babi_bridge(vocab, dim=dim16)

for variant in ["full", "no_slow"]:
    cfg = build_babi_memory_config(variant=variant)
    accs = []
    for seed in range(N_SEEDS):
        rng = random.Random(seed)
        readout = NumpyMLPReadout(
            input_dim=feat16, output_dim=n_answers,
            hidden_dim=HIDDEN_DIM, seed=seed, lr=LR,
        )
        train_stories = train_ds.stories[:TRAIN_N]
        for _ in range(EPOCHS):
            stories = list(train_stories)
            rng.shuffle(stories)
            for story in stories:
                ep = babi_story_to_episode(story, bridge16, answer_set)
                state = seed_memory_state(state_dim=dim16, channel_count=cfg.channel_count)
                for f in ep.forcings:
                    state = step_memory(state, f, cfg)
                feats = query_conditioned_features_np(state, cfg, ep.query)
                readout.train_step(feats, answer_set.index(ep.target))
        final = evaluate(readout, test_ds, bridge16, cfg, dim16)
        accs.append(final)
    mean = sum(accs) / len(accs)
    se = math.sqrt(sum((a - mean) ** 2 for a in accs) / len(accs)) / math.sqrt(len(accs))
    print(f"  dim16 {variant}: {mean:.3f} +/- {se:.3f}")