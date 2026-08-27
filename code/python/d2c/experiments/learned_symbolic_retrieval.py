"""Train a constrained query-conditioned readout over frozen D2C memory."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import math
import sys
from pathlib import Path


if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[4]
    python_root = repo_root / "code" / "python"
    if str(python_root) not in sys.path:
        sys.path.insert(0, str(python_root))

try:
    from ..digital import (
        DigitalDirector,
        DigitalEpisode,
        KeyValueBindingBridge,
        QueryConditionedMLPReadout,
        SymbolicInductionConfig,
        build_symbolic_induction_vocabulary,
        generate_symbolic_induction_example,
        query_conditioned_features,
    )
    from .symbolic_induction import memory_config_for_variant
except ImportError:  # pragma: no cover - direct script execution fallback
    from d2c.digital import (
        DigitalDirector,
        DigitalEpisode,
        KeyValueBindingBridge,
        QueryConditionedMLPReadout,
        SymbolicInductionConfig,
        build_symbolic_induction_vocabulary,
        generate_symbolic_induction_example,
        query_conditioned_features,
    )
    from d2c.experiments.symbolic_induction import memory_config_for_variant


@dataclass(slots=True)
class LearnedRetrievalConfig:
    task: SymbolicInductionConfig
    variant: str = "full"
    train_count: int = 128
    test_count: int = 64
    epochs: int = 30
    hidden_dim: int = 32
    learning_rate: float = 0.08
    pulse_steps: int = 2
    silence_steps: int = 1
    seed: int = 23

    def __post_init__(self) -> None:
        if self.train_count <= 0 or self.test_count <= 0 or self.epochs <= 0:
            raise ValueError("train_count, test_count, and epochs must be positive")
        if self.hidden_dim <= 0 or self.learning_rate <= 0.0:
            raise ValueError("hidden_dim and learning_rate must be positive")

    def to_mapping(self) -> dict[str, object]:
        data = asdict(self)
        data["task"] = self.task.to_mapping()
        return data


def _episode_for_index(
    config: LearnedRetrievalConfig,
    *,
    example_index: int,
    binding: KeyValueBindingBridge,
) -> DigitalEpisode:
    example = generate_symbolic_induction_example(config.task, example_index=example_index)
    forcings: list[list[float]] = []
    tokens: list[str | None] = []
    for key, value in [*example.pairs, *example.distractor_pairs]:
        for _ in range(config.pulse_steps):
            forcings.append(binding.binding_for_pair(key, value))
            tokens.append(value)
        for _ in range(config.silence_steps):
            forcings.append(binding.zero_forcing())
            tokens.append(None)
    for filler in example.filler_tokens:
        for _ in range(config.pulse_steps):
            forcings.append(binding.zero_forcing())
            tokens.append(filler)
        for _ in range(config.silence_steps):
            forcings.append(binding.zero_forcing())
            tokens.append(None)
    for _ in range(config.pulse_steps):
        forcings.append(binding.zero_forcing())
        tokens.append(example.key_token)
    for _ in range(config.silence_steps):
        forcings.append(binding.zero_forcing())
        tokens.append(None)

    query = [0.0 for _ in binding.key_tokens]
    query[binding.key_tokens.index(example.key_token)] = 1.0
    return DigitalEpisode(
        forcings=forcings,
        tokens=tokens,
        target=example.value_token,
        query=query,
        metadata={"example_index": example_index, "pairs": example.pairs},
    )


def _build_dataset(config: LearnedRetrievalConfig, *, start: int, count: int) -> tuple[list[tuple[list[float], int]], list[str]]:
    vocab = build_symbolic_induction_vocabulary(config.task)
    binding = KeyValueBindingBridge(
        key_tokens=[token for token in vocab.tokens if token.startswith("K")],
        value_tokens=[token for token in vocab.tokens if token.startswith("V")],
    )
    memory = memory_config_for_variant(config.variant)
    director = DigitalDirector(memory_config=memory, state_dim=binding.state_dim)
    examples: list[tuple[list[float], int]] = []
    for index in range(start, start + count):
        result = director.run_episode(
            _episode_for_index(config, example_index=index, binding=binding),
            experiment_name="learned_symbolic_retrieval",
        )
        if result.target is None:
            raise RuntimeError("retrieval episode requires a target")
        examples.append((
            query_conditioned_features(result.final_state, memory, result.query),
            binding.value_tokens.index(result.target),
        ))
    return examples, binding.value_tokens


def _accuracy(readout: QueryConditionedMLPReadout, dataset: list[tuple[list[float], int]]) -> float:
    return sum(readout.predict(features)[0] == target for features, target in dataset) / len(dataset)


def run_learned_symbolic_retrieval(config: LearnedRetrievalConfig) -> dict[str, object]:
    """Fit only the readout; memory kernel and binding writer remain frozen."""

    train, value_tokens = _build_dataset(config, start=0, count=config.train_count)
    test, _ = _build_dataset(config, start=config.train_count, count=config.test_count)
    readout = QueryConditionedMLPReadout(
        input_dim=len(train[0][0]),
        output_dim=len(value_tokens),
        hidden_dim=config.hidden_dim,
        seed=config.seed,
    )
    loss_history: list[float] = []
    for _ in range(config.epochs):
        total_loss = sum(readout.train_step(features, target, learning_rate=config.learning_rate) for features, target in train)
        loss_history.append(total_loss / len(train))
    memory = memory_config_for_variant(config.variant)
    return {
        "experiment_name": "learned_symbolic_retrieval",
        "config": config.to_mapping(),
        "variant": config.variant,
        "frozen_components": ["SOE gamma", "SOE weights", "outer-product binding writer"],
        "learned_component": "query_conditioned_relu_mlp_readout",
        "value_tokens": value_tokens,
        "train_count": len(train),
        "test_count": len(test),
        "train_accuracy": _accuracy(readout, train),
        "test_accuracy": _accuracy(readout, test),
        "initial_loss": loss_history[0],
        "final_loss": loss_history[-1],
        "memory_diagnostics": {"deff": memory.deff(), "stability_ratio": memory.stability_ratio()},
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a query-conditioned readout over frozen D2C memory.")
    parser.add_argument("--variant", choices=["full", "no_slow", "collapsed_gamma"], default="full")
    parser.add_argument("--gap", type=int, default=32)
    parser.add_argument("--train-count", type=int, default=128)
    parser.add_argument("--test-count", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    return parser


def main(argv: list[str] | None = None) -> dict[str, object]:
    args = build_arg_parser().parse_args(argv)
    result = run_learned_symbolic_retrieval(LearnedRetrievalConfig(
        task=SymbolicInductionConfig(gap=args.gap),
        variant=args.variant,
        train_count=args.train_count,
        test_count=args.test_count,
        epochs=args.epochs,
    ))
    print(
        f"variant={result['variant']} train_accuracy={result['train_accuracy']:.3f} "
        f"test_accuracy={result['test_accuracy']:.3f} loss={result['initial_loss']:.6f}->{result['final_loss']:.6f}"
    )
    return result


if __name__ == "__main__":
    main()
