from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path

from d2c.digital.babi import (
    BabiDataset, babi_story_to_episode, build_babi_answer_set,
    build_babi_bridge, build_babi_memory_config, build_babi_vocabulary,
)
from d2c.digital.bridge import seed_memory_state, step_memory
from d2c.digital.readout import QueryConditionedMLPReadout, query_conditioned_features

ROOT = Path('/home/ubuntu/aberconics-framework')
DATA = ROOT / 'code/python/d2c/progress/babi_data'
OUT = ROOT / 'code/python/d2c/progress/babi_qa/readout_diagnostic'


def make_features(stories, bridge, cfg, answers):
    rows = []
    for story in stories:
        ep = babi_story_to_episode(story, bridge, answers)
        state = seed_memory_state(state_dim=bridge.state_dim, channel_count=cfg.channel_count)
        for forcing in ep.forcings:
            state = step_memory(state, forcing, cfg)
        rows.append((query_conditioned_features(state, cfg, ep.query), answers.index(ep.target)))
    return rows


def evaluate(model, rows):
    correct = sum(model.predict(x)[0] == y for x, y in rows)
    return correct / len(rows) if rows else 0.0


def run_cell(train_rows, test_rows, *, input_dim, output_dim, hidden, lr, epochs, seed):
    model = QueryConditionedMLPReadout(input_dim=input_dim, output_dim=output_dim,
                                       hidden_dim=hidden, seed=seed)
    rng = random.Random(seed)
    losses = []
    for _ in range(epochs):
        order = list(range(len(train_rows)))
        rng.shuffle(order)
        total = 0.0
        for i in order:
            x, y = train_rows[i]
            total += model.train_step(x, y, learning_rate=lr)
        losses.append(total / len(train_rows))
    return {
        'train_accuracy': evaluate(model, train_rows),
        'test_accuracy': evaluate(model, test_rows),
        'final_train_loss': losses[-1],
        'loss_start': losses[0],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, default=1)
    p.add_argument('--dim', type=int, default=16)
    p.add_argument('--epochs', default='5,10')
    p.add_argument('--lrs', default='0.005,0.01,0.02,0.05,0.1')
    p.add_argument('--hidden', default='16,32,64')
    p.add_argument('--variants', default='full,no_slow')
    p.add_argument('--max-train', type=int, default=0)
    p.add_argument('--output-dir', type=Path, default=OUT)
    args = p.parse_args()
    epochs_grid = [int(x) for x in args.epochs.split(',')]
    lrs = [float(x) for x in args.lrs.split(',')]
    hidden_grid = [int(x) for x in args.hidden.split(',')]
    variants = [x.strip() for x in args.variants.split(',')]

    train = BabiDataset.from_jsonl(DATA / 'babi_train.jsonl').filter_task(1)
    test = BabiDataset.from_jsonl(DATA / 'babi_test.jsonl').filter_task(1)
    if args.max_train:
        train.stories = train.stories[:args.max_train]
    vocab = build_babi_vocabulary(train.stories + test.stories)
    answers = build_babi_answer_set(train.stories)
    bridge = build_babi_bridge(vocab, dim=args.dim)
    cached = {}
    for variant in variants:
        cfg = build_babi_memory_config(variant=variant)
        cached[variant] = (
            make_features(train.stories, bridge, cfg, answers),
            make_features(test.stories, bridge, cfg, answers),
        )
        print(f'cached variant={variant} train={len(train.stories)} test={len(test.stories)}', flush=True)

    rows = []
    for variant in variants:
        train_rows, test_rows = cached[variant]
        input_dim = len(train_rows[0][0])
        for hidden in hidden_grid:
            for lr in lrs:
                for epochs in epochs_grid:
                    for seed in range(args.seeds):
                        result = run_cell(train_rows, test_rows, input_dim=input_dim,
                                          output_dim=len(answers), hidden=hidden,
                                          lr=lr, epochs=epochs, seed=seed)
                        row = {'variant': variant, 'hidden': hidden, 'lr': lr,
                               'epochs': epochs, 'seed': seed, **result}
                        rows.append(row)
                        print(json.dumps(row), flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f'diagnostic_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")}.json'
    path.write_text(json.dumps({'config': vars(args) | {'output_dir': str(args.output_dir)}, 'results': rows}, indent=2))
    print(f'BUNDLE {path}')


if __name__ == '__main__':
    main()
