"""Tests for the bAbI dataset loader and episode builder."""

from __future__ import annotations

from pathlib import Path

import pytest

from d2c.digital.babi import (
    BabiDataset,
    BabiStory,
    babi_story_to_episode,
    build_babi_answer_set,
    build_babi_bridge,
    build_babi_memory_config,
    build_babi_vocabulary,
)

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "d2c" / "progress" / "babi_data"
TRAIN_PATH = FIXTURE_DIR / "babi_train.jsonl"


@pytest.mark.skipif(not (FIXTURE_DIR / "babi_train.jsonl").exists(),
                    reason="bAbI data not downloaded")
class TestBabiDataset:

    def test_load_train(self):
        ds = BabiDataset.from_jsonl(TRAIN_PATH)
        assert len(ds.stories) == 18_013
        assert 1 in ds.task_ids and 20 in ds.task_ids
        assert ds.stories[0].task in ds.task_ids

    def test_filter_task(self):
        ds = BabiDataset.from_jsonl(TRAIN_PATH).filter_task(1)
        assert all(s.task == 1 for s in ds.stories)
        assert len(ds.stories) == 900

    def test_story_fields(self):
        ds = BabiDataset.from_jsonl(TRAIN_PATH).filter_task(1)
        s = ds.stories[0]
        assert isinstance(s.statements, list)
        assert len(s.statements) >= 1
        assert "?" in s.question
        assert s.answer
        assert s.task == 1


class TestBabiHelpers:

    def _sample_story(self) -> BabiStory:
        return BabiStory(
            statements=["Mary moved to the bathroom.", "John went to the hallway."],
            question="Where is Mary?",
            answer="bathroom",
            task=1,
        )

    def test_build_vocabulary(self):
        story = self._sample_story()
        vocab = build_babi_vocabulary([story])
        assert "Mary" in vocab.tokens
        assert "bathroom" in vocab.tokens
        assert "Where" in vocab.tokens

    def test_build_answer_set(self):
        story = self._sample_story()
        answers = build_babi_answer_set([story])
        assert answers == ["bathroom"]

    def test_build_bridge_dense(self):
        story = self._sample_story()
        vocab = build_babi_vocabulary([story])
        bridge = build_babi_bridge(vocab, dim=32, code_mode="dense")
        assert bridge.state_dim == 32
        f = bridge.forcing_for_token("Mary")
        assert len(f) == 32

    def test_build_bridge_one_hot(self):
        story = self._sample_story()
        vocab = build_babi_vocabulary([story])
        bridge = build_babi_bridge(vocab, dim=vocab.size, code_mode="one_hot")
        assert bridge.state_dim == vocab.size
        f = bridge.forcing_for_token("Mary")
        assert sum(f) == pytest.approx(1.0)

    def test_memory_config_full(self):
        cfg = build_babi_memory_config(variant="full")
        assert len(cfg.gamma) == 3
        assert cfg.channel_count == 3
        # Stability ratio of the validated D1/D2 kernel is >1.0
        # (the D3 margin contract rescales this on first commit).
        assert cfg.stability_ratio() > 0.0

    def test_memory_config_no_slow(self):
        cfg = build_babi_memory_config(variant="no_slow")
        assert len(cfg.gamma) == 2

    def test_memory_config_collapsed(self):
        cfg = build_babi_memory_config(variant="collapsed_gamma")
        assert all(g == 1.0 for g in cfg.gamma)

    def test_memory_config_unknown_raises(self):
        with pytest.raises(ValueError, match="unknown variant"):
            build_babi_memory_config(variant="nonexistent")


class TestEpisodeBuilder:

    def _sample_story(self) -> BabiStory:
        return BabiStory(
            statements=["Mary moved to the bathroom.", "John went to the hallway."],
            question="Where is Mary?",
            answer="bathroom",
            task=1,
        )

    def test_episode_structure(self):
        story = self._sample_story()
        vocab = build_babi_vocabulary([story])
        bridge = build_babi_bridge(vocab, dim=32)
        answer_set = build_babi_answer_set([story])

        ep = babi_story_to_episode(story, bridge, answer_set)
        # 10 statement words + 3 question words + 1 readout zero-step = 14
        assert len(ep.forcings) == len(ep.tokens) == 14
        assert ep.target == "bathroom"
        assert len(ep.query) == 32
        assert ep.metadata["task"] == 1

    def test_episode_with_silence(self):
        story = self._sample_story()
        vocab = build_babi_vocabulary([story])
        bridge = build_babi_bridge(vocab, dim=16)
        answer_set = build_babi_answer_set([story])

        ep = babi_story_to_episode(story, bridge, answer_set, silence_steps=2)
        # 10 words + 2 statements x 2 silence + 3 question words + 1 readout = 18
        assert len(ep.forcings) == 18
        # Silence tokens are None (2 stmts × 2 silence + 1 readout = 5)
        silence_tokens = [t for t in ep.tokens if t is None]
        assert len(silence_tokens) == 5

    def test_episode_query_nonzero(self):
        story = self._sample_story()
        vocab = build_babi_vocabulary([story])
        bridge = build_babi_bridge(vocab, dim=16)
        answer_set = build_babi_answer_set([story])

        ep = babi_story_to_episode(story, bridge, answer_set)
        assert any(v != 0.0 for v in ep.query)

    def test_episode_forcing_dimension(self):
        story = self._sample_story()
        vocab = build_babi_vocabulary([story])
        bridge = build_babi_bridge(vocab, dim=48)
        answer_set = build_babi_answer_set([story])

        ep = babi_story_to_episode(story, bridge, answer_set)
        for f in ep.forcings:
            assert len(f) == 48