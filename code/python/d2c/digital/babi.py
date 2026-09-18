"""bAbI dataset loader and episode builder for D2C Phase E experiments.

Loads JSONL-format bAbI task data (Muennighoff/babi variant) and converts
stories into DigitalEpisode objects for the existing DigitalDirector runtime.

Default data path: code/python/d2c/progress/babi_data/
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .bridge import DigitalMemoryConfig, TokenForcingBridge
from .director import DigitalEpisode
from .tokens import Vocabulary


@dataclass(slots=True)
class BabiStory:
    """One parsed bAbI story: sequence of statements + one question + answer."""

    statements: list[str]
    question: str
    answer: str
    task: int
    raw_passage: str = ""

    @property
    def answer_set(self) -> set[str]:
        return {self.answer}


@dataclass(slots=True)
class BabiDataset:
    """Loaded bAbI split containing stories grouped by task."""

    stories: list[BabiStory]
    task_ids: list[int]

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "BabiDataset":
        stories: list[BabiStory] = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            passage: str = record["passage"]
            statements = [s.strip() for s in passage.strip().split("\n") if s.strip()]
            stories.append(BabiStory(
                statements=statements,
                question=record["question"].strip(),
                answer=record["answer"].strip(),
                task=int(record["task"]),
                raw_passage=passage,
            ))
        task_ids = sorted({s.task for s in stories})
        return cls(stories=stories, task_ids=task_ids)

    def filter_task(self, task_id: int) -> "BabiDataset":
        subset = [s for s in self.stories if s.task == task_id]
        return BabiDataset(stories=subset, task_ids=[task_id])


def build_babi_vocabulary(stories: Sequence[BabiStory]) -> Vocabulary:
    """Collect all unique tokens across passages, questions, and answers."""
    tokens: set[str] = set()
    for story in stories:
        for stmt in story.statements:
            tokens.update(stmt.split())
        tokens.update(story.question.split())
        tokens.add(story.answer)
    return Vocabulary(tokens=sorted(tokens))


def build_babi_answer_set(stories: Sequence[BabiStory]) -> list[str]:
    """Sorted unique answer strings across stories."""
    return sorted({s.answer for s in stories})


def build_babi_bridge(
    vocabulary: Vocabulary,
    *,
    dim: int = 64,
    code_mode: str = "dense",
    seed: int = 17,
) -> TokenForcingBridge:
    """Standard TokenForcingBridge configured for bAbI."""
    return TokenForcingBridge(
        vocabulary=vocabulary,
        dim=dim,
        seed=seed,
        code_mode=code_mode,
    )


def build_babi_memory_config(
    *,
    variant: str = "full",
    dt: float = 0.05,
    leak_rate: float = 1.0,
    form: str = "input_driven",
) -> DigitalMemoryConfig:
    """Validated D1/D2 kernel with configurable ablation variant.

    variant:
        'full'            — gamma=[2.0, 0.5, 0.1], w=[0.04, 0.08, 0.12]
        'no_slow'         — gamma=[2.0, 0.5],      w=[0.04, 0.08]
        'collapsed_gamma' — gamma=[1.0, 1.0, 1.0],  w=[0.04, 0.08, 0.12]
    """
    if variant == "full":
        gamma = [2.0, 0.5, 0.1]
        w = [0.04, 0.08, 0.12]
    elif variant == "no_slow":
        gamma = [2.0, 0.5]
        w = [0.04, 0.08]
    elif variant == "collapsed_gamma":
        gamma = [1.0, 1.0, 1.0]
        w = [0.04, 0.08, 0.12]
    else:
        raise ValueError(f"unknown variant: {variant!r}")
    return DigitalMemoryConfig(
        gamma=gamma, w=w, dt=dt, leak_rate=leak_rate, form=form,
    )


def babi_story_to_episode(
    story: BabiStory,
    bridge: TokenForcingBridge,
    answer_set: Sequence[str],
    *,
    silence_steps: int = 0,
) -> DigitalEpisode:
    """Convert one bAbI story into a DigitalEpisode.

    Each word in each statement becomes a forcing step.  Optionally
    zero-forcing silence steps are inserted between statements.  The
    question words are appended, then one final zero-forcing step
    (the readout step where the answer is decoded).

    The episode query is the sum of question-word forcing vectors
    (bag-of-words encoding).  ``answer_set`` is stored in metadata
    for downstream readout classification.
    """
    forcings: list[list[float]] = []
    tokens: list[str | None] = []

    for stmt in story.statements:
        words = stmt.split()
        for word in words:
            forcings.append(bridge.forcing_for_token(word))
            tokens.append(word)
        for _ in range(silence_steps):
            forcings.append(bridge.zero_forcing())
            tokens.append(None)

    question_words = story.question.split()
    for word in question_words:
        forcings.append(bridge.forcing_for_token(word))
        tokens.append(word)

    # One zero-forcing readout step
    forcings.append(bridge.zero_forcing())
    tokens.append(None)

    # Bag-of-words query encoding
    dim = bridge.state_dim
    query = [0.0] * dim
    for word in question_words:
        wf = bridge.forcing_for_token(word)
        for i in range(dim):
            query[i] += wf[i]

    return DigitalEpisode(
        forcings=forcings,
        tokens=tokens,
        target=story.answer,
        query=query,
        metadata={
            "task": story.task,
            "question": story.question,
            "answer_set": list(answer_set),
        },
    )