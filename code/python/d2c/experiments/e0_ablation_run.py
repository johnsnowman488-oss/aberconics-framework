"""Quick ablation run with best config from sweep."""
import math
import random
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

train_ds = BabiDataset.from_jsonl(DATA / "babi_train.jsonl").filter_task(1)
test_ds = BabiDataset.from_jsonl(DATA / "babi_test.jsonl").filter_task(1)
vocab = build_babi_vocabulary(train_ds.stories + test_ds.stories)
answer_set = build_babi_answer_set(train_ds.stories)
dim = 16
feat_dim = dim + dim + dim * dim


def evaluate(readout, test_ds, bridge, memory_cfg):
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


for variant in ["full", "no_slow", "collapsed_gamma"]:
    cfg = build_babi_memory_config(variant=variant)
    bridge = build_babi_bridge(vocab, dim=dim)
    accs = []
    for seed in range(5):
        rng = random.Random(seed)
        readout = NumpyMLPReadout(
            input_dim=feat_dim, output_dim=len(answer_set),
            hidden_dim=128, seed=seed, lr=0.02,
        )
        for _ in range(20):
            stories = list(train_ds.stories[:500])
            rng.shuffle(stories)
            for story in stories:
                ep = babi_story_to_episode(story, bridge, answer_set)
                state = seed_memory_state(state_dim=dim, channel_count=cfg.channel_count)
                for f in ep.forcings:
                    state = step_memory(state, f, cfg)
                feats = query_conditioned_features_np(state, cfg, ep.query)
                readout.train_step(feats, answer_set.index(ep.target))
        final = evaluate(readout, test_ds, bridge, cfg)
        accs.append(final)
        print(f"  {variant} seed {seed}: {final:.3f}")
    mean = sum(accs) / len(accs)
    se = math.sqrt(sum((a - mean) ** 2 for a in accs) / len(accs)) / math.sqrt(len(accs))
    print(f"  => {variant}: {mean:.3f} +/- {se:.3f}")
    print()