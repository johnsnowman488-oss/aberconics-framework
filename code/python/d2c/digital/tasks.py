"""Synthetic symbolic tasks for digital D2C experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Mapping

from .streams import DigitalStream, regular_token_stream
from .tokens import Vocabulary


@dataclass(slots=True)
class SymbolicInductionExample:
    """Multi-pair query-conditioned symbolic retrieval example."""

    key_token: str
    value_token: str
    pairs: list[tuple[str, str]]
    distractor_pairs: list[tuple[str, str]]
    filler_tokens: list[str]
    stream: DigitalStream

    def to_mapping(self) -> dict[str, object]:
        data = asdict(self)
        data["stream"] = self.stream.to_mapping()
        return data


@dataclass(slots=True)
class SymbolicInductionConfig:
    key_count: int = 4
    value_count: int = 4
    filler_count: int = 3
    pair_count: int = 3
    distractor_pair_count: int = 0
    gap: int = 8
    seed: int = 0

    def __post_init__(self) -> None:
        if self.key_count <= 0 or self.value_count <= 0 or self.filler_count <= 0:
            raise ValueError("token counts must be positive")
        if self.pair_count <= 1:
            raise ValueError("pair_count must be greater than one for a binding task")
        if self.pair_count > self.key_count or self.pair_count > self.value_count:
            raise ValueError("pair_count must not exceed key_count or value_count")
        if self.distractor_pair_count < 0:
            raise ValueError("distractor_pair_count must be non-negative")
        if self.gap < 0:
            raise ValueError("gap must be non-negative")

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "SymbolicInductionConfig":
        return cls(
            key_count=int(data.get("key_count", 4)),
            value_count=int(data.get("value_count", 4)),
            filler_count=int(data.get("filler_count", 3)),
            pair_count=int(data.get("pair_count", 3)),
            distractor_pair_count=int(data.get("distractor_pair_count", 0)),
            gap=int(data.get("gap", 8)),
            seed=int(data.get("seed", 0)),
        )


def build_symbolic_induction_vocabulary(config: SymbolicInductionConfig) -> Vocabulary:
    tokens = (
        [f"K{i}" for i in range(config.key_count)]
        + [f"V{i}" for i in range(config.value_count)]
        + [f"F{i}" for i in range(config.filler_count)]
    )
    return Vocabulary(tokens=tokens)


def generate_symbolic_induction_example(
    config: SymbolicInductionConfig,
    *,
    example_index: int = 0,
) -> SymbolicInductionExample:
    rng = random.Random(config.seed + example_index)
    keys = rng.sample([f"K{i}" for i in range(config.key_count)], config.pair_count)
    values = rng.sample([f"V{i}" for i in range(config.value_count)], config.pair_count)
    pairs = list(zip(keys, values, strict=True))
    key_token, value_token = pairs[rng.randrange(len(pairs))]
    distractor_candidates = [
        (key, value)
        for key in (f"K{i}" for i in range(config.key_count))
        for value in (f"V{i}" for i in range(config.value_count))
        if key != key_token and (key, value) not in pairs
    ]
    if config.distractor_pair_count > len(distractor_candidates):
        raise ValueError("distractor_pair_count exceeds available non-query bindings")
    distractor_pairs = rng.sample(distractor_candidates, config.distractor_pair_count)
    filler_tokens = [f"F{rng.randrange(config.filler_count)}" for _ in range(config.gap)]
    tokens = [token for pair in [*pairs, *distractor_pairs] for token in pair] + [*filler_tokens, key_token]
    stream = regular_token_stream(
        tokens,
        dt=1.0,
        target_token=value_token,
        metadata={
            "task": "symbolic_induction_multi_pair_binding",
            "query_key": key_token,
            "value_token": value_token,
            "pairs": pairs,
            "distractor_pairs": distractor_pairs,
            "gap": config.gap,
        },
    )
    return SymbolicInductionExample(
        key_token=key_token,
        value_token=value_token,
        pairs=pairs,
        distractor_pairs=distractor_pairs,
        filler_tokens=filler_tokens,
        stream=stream,
    )
