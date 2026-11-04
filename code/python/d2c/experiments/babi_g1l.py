"""G1-L: learned, query-conditioned entity/role binding for bAbI QA2.

The input event templates are parsed into typed events, but relation storage,
holder-to-person retrieval, and answer readout are parameterized and trained
end-to-end from answer supervision. This is not a general natural-language
parser; claims are limited to the supported QA2 event templates.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..digital.babi import (
    BabiDataset, build_babi_bridge, build_babi_memory_config, build_babi_vocabulary,
)
from ..digital.relational import RelationalEvent, parse_babi_question, parse_babi_statement
from ..runtime.digital_runtime import DigitalRuntime, DigitalRuntimeConfig

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DATA_DIR = _REPO_ROOT / "code" / "python" / "d2c" / "progress" / "babi_data"


@dataclass(frozen=True, slots=True)
class G1LConfig:
    task: int = 2
    dim: int = 48
    seed: int = 0
    learning_rate: float = 0.02
    epochs: int = 3
    train_limit: int = 200
    eval_limit: int = 100
    confirm_seeds: int = 3

    def __post_init__(self) -> None:
        if self.task != 2:
            raise ValueError("G1-L currently supports bAbI QA2 (task 2) only")
        if self.dim < 8:
            raise ValueError("dim must be at least 8")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if self.epochs < 1 or self.train_limit < 0 or self.eval_limit < 1:
            raise ValueError("epochs/evaluation limits are invalid")
        if self.confirm_seeds < 1 or self.confirm_seeds > 10:
            raise ValueError("confirm_seeds must be between 1 and 10")


@dataclass(frozen=True, slots=True)
class ParsedStory:
    events: tuple[RelationalEvent, ...]
    question: str
    query_object: str
    answer: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_split(dataset: BabiDataset) -> list[ParsedStory]:
    parsed: list[ParsedStory] = []
    for story in dataset.stories:
        events = tuple(parse_babi_statement(statement) for statement in story.statements)
        query_object = parse_babi_question(story.question)
        parsed.append(ParsedStory(events, story.question, query_object, story.answer))
    return parsed


def _entity_names(stories: Iterable[ParsedStory]) -> tuple[list[str], list[str], list[str]]:
    people: set[str] = set()
    objects: set[str] = set()
    locations: set[str] = set()
    for story in stories:
        objects.add(story.query_object)
        for event in story.events:
            people.add(event.actor)
            if event.object:
                objects.add(event.object)
            if event.location:
                locations.add(event.location)
    return sorted(people), sorted(objects), sorted(locations)


class LearnedRelationalBinder:
    """Small NumPy model with learned relation encoders and soft holder retrieval."""

    def __init__(
        self,
        *,
        people: list[str],
        objects: list[str],
        locations: list[str],
        answers: list[str],
        dim: int,
        seed: int,
    ) -> None:
        self.people, self.objects, self.locations, self.answers = people, objects, locations, answers
        self.person_id = {name: i for i, name in enumerate(people)}
        self.object_id = {name: i for i, name in enumerate(objects)}
        self.location_id = {name: i for i, name in enumerate(locations)}
        self.answer_id = {name: i for i, name in enumerate(answers)}
        self.dim = dim
        rng = np.random.default_rng(seed)
        d = dim
        scale = 1.0 / math.sqrt(d)

        self.p: dict[str, np.ndarray] = {
            "E_person": rng.normal(0.0, 0.2, (len(people), d)),
            "E_object": rng.normal(0.0, 0.2, (len(objects), d)),
            "E_location": rng.normal(0.0, 0.2, (len(locations), d)),
            "W_person_location": rng.normal(0.0, scale, (d, d)),
            "b_person_location": np.zeros(d),
            "W_object_holder": rng.normal(0.0, scale, (d, d)),
            "b_object_holder": np.zeros(d),
            "W_object_location": rng.normal(0.0, scale, (d, d)),
            "b_object_location": np.zeros(d),
            "W_query_holder": rng.normal(0.0, scale, (d, d)),
            "W_query_object": rng.normal(0.0, scale, (d, d)),
            "b_query": np.zeros(d),
            "W_person_key": rng.normal(0.0, scale, (d, d)),
            "b_person_key": np.zeros(d),
            "W_object_query": rng.normal(0.0, scale, (d, d)),
            "W_output": rng.normal(0.0, 0.05, (len(answers), d)),
            "b_output": np.zeros(len(answers)),
        }
        self.adam_m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.adam_v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.adam_step = 0

    def _person_value(self, location: str) -> tuple[np.ndarray, int, np.ndarray]:
        li = self.location_id[location]
        e = self.p["E_location"][li]
        x = self.p["W_person_location"] @ e + self.p["b_person_location"]
        value = np.tanh(x)
        return value, li, value

    def _forward(self, story: ParsedStory) -> dict[str, Any]:
        person_state: dict[str, np.ndarray] = {}
        person_state_source: dict[str, tuple[int, np.ndarray]] = {}
        holder_state: dict[str, np.ndarray] = {}
        holder_source: dict[str, str] = {}
        object_location: dict[str, np.ndarray] = {}
        object_location_source: dict[str, tuple[str, int, np.ndarray]] = {}
        held: dict[str, bool] = {}

        for event in story.events:
            if event.kind == "move":
                if event.location is None:
                    raise ValueError("move event is missing its location")
                value, li, _ = self._person_value(event.location)
                person_state[event.actor] = value
                person_state_source[event.actor] = (li, value)
            elif event.kind == "pickup":
                if event.object is None:
                    raise ValueError("pickup event is missing its object")
                pi = self.person_id[event.actor]
                x = self.p["W_object_holder"] @ self.p["E_person"][pi] + self.p["b_object_holder"]
                holder_state[event.object] = np.tanh(x)
                holder_source[event.object] = event.actor
                held[event.object] = True
                object_location.pop(event.object, None)
                object_location_source.pop(event.object, None)
            elif event.kind == "release":
                if event.object is None:
                    raise ValueError("release event is missing its object")
                if event.actor in person_state:
                    person_value = person_state[event.actor]
                    x = self.p["W_object_location"] @ person_value + self.p["b_object_location"]
                    object_location[event.object] = np.tanh(x)
                    li, source_value = person_state_source[event.actor]
                    object_location_source[event.object] = (event.actor, li, source_value)
                else:
                    # G1 can observe a release before this actor's location is
                    # established in the current story. Preserve an explicit
                    # unknown-location state instead of aborting the episode.
                    object_location[event.object] = np.tanh(self.p["b_object_location"])
                    object_location_source[event.object] = None
                holder_state.pop(event.object, None)
                holder_source.pop(event.object, None)
                held[event.object] = False
            else:
                raise ValueError(f"unsupported event kind {event.kind!r}")

        if story.query_object not in self.object_id:
            raise ValueError(f"unknown query object {story.query_object!r}")
        oi = self.object_id[story.query_object]
        object_signal = np.tanh(self.p["W_object_query"] @ self.p["E_object"][oi])
        is_held = held.get(story.query_object, False) and story.query_object in holder_state
        attention: np.ndarray | None = None
        people_order = sorted(person_state)
        keys: list[np.ndarray] = []
        if is_held:
            holder = holder_state[story.query_object]
            q = np.tanh(
                self.p["W_query_holder"] @ holder
                + self.p["W_query_object"] @ self.p["E_object"][oi]
                + self.p["b_query"]
            )
            for person in people_order:
                pi = self.person_id[person]
                keys.append(np.tanh(self.p["W_person_key"] @ self.p["E_person"][pi] + self.p["b_person_key"]))
            if not keys:
                raise ValueError("held object has no person candidates")
            scores = np.asarray(keys) @ q / math.sqrt(self.dim)
            scores -= np.max(scores)
            attention = np.exp(scores)
            attention /= np.sum(attention)
            relational_value = sum(
                attention[i] * person_state[person]
                for i, person in enumerate(people_order)
            )
            target_holder = holder_source.get(story.query_object)
            z = relational_value + 0.1 * object_signal
        else:
            if story.query_object not in object_location:
                raise ValueError(f"object {story.query_object!r} has neither holder nor fixed location")
            holder = None
            q = None
            target_holder = None
            relational_value = object_location[story.query_object]
            z = relational_value + 0.1 * object_signal

        logits = self.p["W_output"] @ z + self.p["b_output"]
        shifted = logits - np.max(logits)
        probabilities = np.exp(shifted)
        probabilities /= np.sum(probabilities)
        predicted = self.answers[int(np.argmax(probabilities))]
        predicted_holder = people_order[int(np.argmax(attention))] if attention is not None else None

        return {
            "logits": logits,
            "probabilities": probabilities,
            "prediction": predicted,
            "is_held": is_held,
            "target_holder": target_holder,
            "predicted_holder": predicted_holder,
            "holder_correct": (predicted_holder == target_holder) if is_held else None,
            "attention": {
                people_order[i]: float(attention[i])
                for i in range(len(people_order))
            } if attention is not None else {},
            "cache": {
                "story": story,
                "object_index": oi,
                "object_signal": object_signal,
                "is_held": is_held,
                "holder": holder_state.get(story.query_object),
                "people_order": people_order,
                "keys": keys,
                "attention": attention,
                "person_state": person_state,
                "person_state_source": person_state_source,
                "object_location": object_location.get(story.query_object),
                "object_location_source": object_location_source.get(story.query_object),
                "relational_value": relational_value,
                "z": z,
            },
        }

    def _train_one(self, story: ParsedStory, learning_rate: float) -> float:
        out = self._forward(story)
        target = self.answer_id.get(story.answer)
        if target is None:
            raise ValueError(f"answer {story.answer!r} is absent from the training answer set")
        probs = out["probabilities"]
        loss = -math.log(max(float(probs[target]), 1e-12))
        cache = out["cache"]
        d = self.dim
        grad = {k: np.zeros_like(v) for k, v in self.p.items()}
        dlogits = probs.copy()
        dlogits[target] -= 1.0
        grad["W_output"] += np.outer(dlogits, cache["z"])
        grad["b_output"] += dlogits
        dz = self.p["W_output"].T @ dlogits

        # Object identity is present in both held and released-object queries.
        dobject_signal = 0.1 * dz
        dobject_pre = dobject_signal * (1.0 - cache["object_signal"] ** 2)
        grad["W_object_query"] += np.outer(dobject_pre, self.p["E_object"][cache["object_index"]])
        grad["E_object"][cache["object_index"]] += self.p["W_object_query"].T @ dobject_pre

        if cache["is_held"]:
            alpha = cache["attention"]
            people = cache["people_order"]
            values = cache["person_state"]
            d_alpha = np.asarray([
                float(np.dot(dz, values[name])) for name in people
            ])
            d_scores = alpha * (d_alpha - float(np.dot(alpha, d_alpha)))
            dq = np.zeros(d)
            for i, name in enumerate(people):
                dq += d_scores[i] * cache["keys"][i] / math.sqrt(d)
                pi = self.person_id[name]
                dkey = d_scores[i] * (1.0 - cache["keys"][i] ** 2)
                grad["W_person_key"] += np.outer(dkey, self.p["E_person"][pi])
                grad["b_person_key"] += dkey
                grad["E_person"][pi] += self.p["W_person_key"].T @ dkey

                dvalue = alpha[i] * dz
                li, person_value = cache["person_state_source"][name]
                dvalue_pre = dvalue * (1.0 - person_value ** 2)
                grad["W_person_location"] += np.outer(
                    dvalue_pre, self.p["E_location"][li]
                )
                grad["b_person_location"] += dvalue_pre
                grad["E_location"][li] += self.p["W_person_location"].T @ dvalue_pre

            holder = cache["holder"]
            object_index = cache["object_index"]
            story = cache["story"]
            object_query = self.p["E_object"][object_index]
            q = np.tanh(
                self.p["W_query_holder"] @ holder
                + self.p["W_query_object"] @ object_query
                + self.p["b_query"]
            )
            dq_pre = dq * (1.0 - q ** 2)
            grad["W_query_holder"] += np.outer(dq_pre, holder)
            grad["W_query_object"] += np.outer(dq_pre, object_query)
            grad["b_query"] += dq_pre
            dholder = self.p["W_query_holder"].T @ dq_pre
            holder_source = next(
                event.actor for event in reversed(story.events)
                if event.kind == "pickup" and event.object == story.query_object
            )
            pi = self.person_id[holder_source]
            holder_vector = np.tanh(
                self.p["W_object_holder"] @ self.p["E_person"][pi]
                + self.p["b_object_holder"]
            )
            dholder_pre = dholder * (1.0 - holder_vector ** 2)
            grad["W_object_holder"] += np.outer(dholder_pre, self.p["E_person"][pi])
            grad["b_object_holder"] += dholder_pre
            grad["E_person"][pi] += self.p["W_object_holder"].T @ dholder_pre
        else:
            object_location = cache["object_location"]
            object_location_pre = dz * (1.0 - object_location ** 2)
            source = cache["object_location_source"]
            if source is None:
                # Unknown release location: learn a stable unknown-state bias.
                grad["W_object_location"] += np.zeros_like(self.p["W_object_location"])
                grad["b_object_location"] += object_location_pre
            else:
                _actor, li, person_value = source
                grad["W_object_location"] += np.outer(object_location_pre, person_value)
                grad["b_object_location"] += object_location_pre
                dperson = self.p["W_object_location"].T @ object_location_pre
                dperson_pre = dperson * (1.0 - person_value ** 2)
                grad["W_person_location"] += np.outer(dperson_pre, self.p["E_location"][li])
                grad["b_person_location"] += dperson_pre
                grad["E_location"][li] += self.p["W_person_location"].T @ dperson_pre

        # Adam keeps the small pilot stable without a costly framework dependency.
        self.adam_step += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        norm = math.sqrt(sum(float(np.sum(g * g)) for g in grad.values()))
        clip = min(1.0, 5.0 / max(norm, 1e-12))
        for name, value in self.p.items():
            g = grad[name] * clip
            self.adam_m[name] = beta1 * self.adam_m[name] + (1.0 - beta1) * g
            self.adam_v[name] = beta2 * self.adam_v[name] + (1.0 - beta2) * (g * g)
            m_hat = self.adam_m[name] / (1.0 - beta1 ** self.adam_step)
            v_hat = self.adam_v[name] / (1.0 - beta2 ** self.adam_step)
            value -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)
        return loss


def _run_f0_companion(raw_splits: list[list[Any]], dim: int) -> dict[str, Any]:
    """Exercise the unchanged full SOE/F0 stream once per story and cache its audit.

    The learned relational optimizer uses typed QA2 events, not the F0 state
    vectors. This sidecar makes runtime settings and stability observable
    without conflating runtime orchestration with learned binding.
    """
    raw_stories = [story for split in raw_splits for story in split]
    vocabulary = build_babi_vocabulary(raw_stories)
    bridge = build_babi_bridge(vocabulary, dim=dim)
    runtime = DigitalRuntime(
        memory_config=build_babi_memory_config(variant="full"),
        state_dim=dim,
        config=DigitalRuntimeConfig(trace_capacity=20_000),
        session_id="babi-g1l-full-f0-companion",
    )
    for index, story in enumerate(raw_stories):
        episode_id = f"g1l-f0-{index}"
        runtime.reset_memory()
        runtime.episode_count += 1
        for statement_index, statement in enumerate(story.statements):
            for token in statement.split():
                runtime.step(
                    bridge.forcing_for_token(token), token=token,
                    episode_id=episode_id,
                    metadata={"statement_index": statement_index},
                )
        for token in story.question.split():
            runtime.step(
                bridge.forcing_for_token(token), token=token,
                episode_id=episode_id, metadata={"query": True},
            )
        runtime.step(
            bridge.zero_forcing(), token=None, episode_id=episode_id,
            metadata={"readout_step": True},
        )
    return {
        "variant": "full",
        "mode": "isolated",
        "soe_kernel_modified": False,
        "learning_path_uses_f0_features": False,
        "stories_exercised": len(raw_stories),
        **runtime.runtime_summary(),
    }


def _run_l0(dim: int, learning_rate: float) -> dict[str, Any]:
    synthetic = [
        ParsedStory((
            RelationalEvent("move", "Mary", location="kitchen"),
            RelationalEvent("move", "John", location="garden"),
            RelationalEvent("pickup", "Mary", object="football"),
            RelationalEvent("move", "Mary", location="hallway"),
        ), "Where is the football?", "football", "hallway"),
        ParsedStory((
            RelationalEvent("move", "John", location="office"),
            RelationalEvent("pickup", "John", object="milk"),
            RelationalEvent("move", "John", location="bedroom"),
        ), "Where is the milk?", "milk", "bedroom"),
        ParsedStory((
            RelationalEvent("move", "Mary", location="kitchen"),
            RelationalEvent("pickup", "Mary", object="football"),
            RelationalEvent("move", "Mary", location="hallway"),
            RelationalEvent("release", "Mary", object="football"),
            RelationalEvent("move", "Mary", location="garden"),
        ), "Where is the football?", "football", "hallway"),
    ]
    people, objects, locations = _entity_names(synthetic)
    answers = sorted({s.answer for s in synthetic})
    model = LearnedRelationalBinder(
        people=people, objects=objects, locations=locations,
        answers=answers, dim=max(8, dim), seed=17,
    )
    initial = model._forward(synthetic[0])
    distinct_people = np.linalg.norm(
        model.p["E_person"][model.person_id["Mary"]]
        - model.p["E_person"][model.person_id["John"]]
    ) > 1e-8
    distinct_objects = np.linalg.norm(
        model.p["E_object"][model.object_id["football"]]
        - model.p["E_object"][model.object_id["milk"]]
    ) > 1e-8
    initial_loss = -math.log(max(float(initial["probabilities"][model.answer_id["hallway"]]), 1e-12))
    losses = [
        model._train_one(synthetic[0], learning_rate)
        for _ in range(8)
    ]
    final = model._forward(synthetic[0])
    final_loss = -math.log(max(float(final["probabilities"][model.answer_id["hallway"]]), 1e-12))
    checks = {
        "typed_event_sequence": len(synthetic[0].events) == 4,
        "person_identity_vectors_distinct": bool(distinct_people),
        "object_identity_vectors_distinct": bool(distinct_objects),
        "holder_attention_exposed": bool(initial["attention"]),
        "gradient_values_finite": all(np.isfinite(v).all() for v in model.p.values()),
        "single_case_loss_decreased": bool(final_loss < initial_loss),
    }
    return {
        "stage": "L0",
        "passed": all(checks.values()),
        "checks": checks,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "updates": len(losses),
        "claim_boundary": "Mechanism and gradient sanity only; not a QA2 accuracy result.",
    }


def _evaluate(
    model: LearnedRelationalBinder,
    stories: list[ParsedStory],
) -> dict[str, Any]:
    correct = 0
    holder_correct = holder_total = 0
    holder_chance_total = 0.0
    predictions: list[dict[str, Any]] = []
    for index, story in enumerate(stories):
        output = model._forward(story)
        is_correct = output["prediction"] == story.answer
        correct += int(is_correct)
        if output["is_held"]:
            holder_total += 1
            holder_correct += int(bool(output["holder_correct"]))
            holder_chance_total += 1.0 / max(len(output["attention"]), 1)
        predictions.append({
            "example_index": index,
            "question": story.question,
            "query_object": story.query_object,
            "target": story.answer,
            "prediction": output["prediction"],
            "correct": is_correct,
            "query_relation": "object_holder_to_person_location" if output["is_held"] else "fixed_object_location",
            "target_holder": output["target_holder"],
            "predicted_holder": output["predicted_holder"],
            "holder_correct": output["holder_correct"],
            "holder_attention": output["attention"],
            "confidence": float(np.max(output["probabilities"])),
        })
    total = len(stories)
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0.0,
        "holder_retrieval_correct": holder_correct,
        "holder_retrieval_total": holder_total,
        "holder_retrieval_accuracy": holder_correct / holder_total if holder_total else None,
        "holder_random_baseline": holder_chance_total / holder_total if holder_total else None,
        "predictions": predictions,
    }


def _train(
    model: LearnedRelationalBinder,
    stories: list[ParsedStory],
    *,
    epochs: int,
    learning_rate: float,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    epoch_losses: list[float] = []
    initial_loss: float | None = None
    for _epoch in range(epochs):
        order = rng.permutation(len(stories))
        losses: list[float] = []
        for index in order:
            loss = model._train_one(stories[int(index)], learning_rate)
            if initial_loss is None:
                initial_loss = loss
            losses.append(loss)
        epoch_losses.append(float(np.mean(losses)) if losses else 0.0)
    return {
        "initial_loss": initial_loss,
        "epoch_mean_losses": epoch_losses,
        "final_epoch_loss": epoch_losses[-1] if epoch_losses else None,
        "loss_decreased": bool(len(epoch_losses) > 1 and epoch_losses[-1] < epoch_losses[0]),
        "optimizer": "Adam",
        "learning_rate": learning_rate,
        "epochs": epochs,
        "training_stories": len(stories),
    }


def run_g1l_experiment(
    *,
    config: G1LConfig | None = None,
    data_dir: Path = _DATA_DIR,
    stage: str = "all",
    exercise_f0: bool = True,
) -> dict[str, Any]:
    cfg = config or G1LConfig()
    if stage not in {"all", "pilot"}:
        raise ValueError("stage must be 'all' or 'pilot'")
    started = time.perf_counter()
    data_dir = Path(data_dir)
    train_path = data_dir / "babi_train.jsonl"
    valid_path = data_dir / "babi_valid.jsonl"
    test_path = data_dir / "babi_test.jsonl"
    for path in (train_path, valid_path, test_path):
        if not path.is_file():
            raise FileNotFoundError(f"G1-L requires {path.name}; run d2c.digital.download_babi first")

    # Read and parse each split once; L1 and L2 reuse these in-memory examples.
    train_ds = BabiDataset.from_jsonl(train_path).filter_task(2)
    valid_ds = BabiDataset.from_jsonl(valid_path).filter_task(2)
    test_ds = BabiDataset.from_jsonl(test_path).filter_task(2)
    train_all = _parse_split(train_ds)
    valid_all = _parse_split(valid_ds)
    test_all = _parse_split(test_ds)
    f0_companion = (
        _run_f0_companion(
            [train_ds.stories, valid_ds.stories, test_ds.stories], cfg.dim
        ) if exercise_f0 else {
            "status": "not_exercised",
            "reason": "exercise_f0=False is intended for isolated unit tests only.",
            "learning_path_uses_f0_features": False,
        }
    )
    # Entity and answer vocabularies are learned from training data only.
    people, objects, locations = _entity_names(train_all)
    answers = sorted({story.answer for story in train_all})
    if not train_all or not valid_all or not test_all or not answers:
        raise ValueError("G1-L needs non-empty QA2 train, validation and test splits")
    train_rng = np.random.default_rng(cfg.seed)
    train_indices = (
        train_rng.permutation(len(train_all))[:cfg.train_limit]
        if cfg.train_limit else np.arange(len(train_all))
    )
    valid_rng = np.random.default_rng(cfg.seed + 1)
    valid_indices = valid_rng.permutation(len(valid_all))[:cfg.eval_limit]
    train_pilot = [train_all[int(index)] for index in train_indices]
    valid_pilot = [valid_all[int(index)] for index in valid_indices]
    if not train_pilot or not valid_pilot:
        raise ValueError("G1-L pilot has no training or validation examples")

    l0 = _run_l0(cfg.dim, cfg.learning_rate)
    l1_started = time.perf_counter()
    pilot = LearnedRelationalBinder(
        people=people, objects=objects, locations=locations,
        answers=answers, dim=cfg.dim, seed=cfg.seed,
    )
    before = _evaluate(pilot, valid_pilot)
    training = _train(
        pilot, train_pilot, epochs=cfg.epochs,
        learning_rate=cfg.learning_rate, seed=cfg.seed,
    )
    after = _evaluate(pilot, valid_pilot)
    random_holder_baseline = after["holder_random_baseline"]
    holder_gate = (
        after["holder_retrieval_accuracy"] is not None
        and random_holder_baseline is not None
        and after["holder_retrieval_accuracy"] > random_holder_baseline + 0.05
    )
    l1_gate = bool(
        l0["passed"]
        and training["loss_decreased"]
        and after["accuracy"] > before["accuracy"]
        and holder_gate
    )
    l1 = {
        "stage": "L1",
        "status": "passed" if l1_gate else "pilot_gate_not_met",
        "gate_passed": l1_gate,
        "training": training,
        "validation_before_training": {
            key: before[key] for key in ("accuracy", "holder_retrieval_accuracy", "correct", "total")
        },
        "validation_after_training": {
            key: after[key] for key in ("accuracy", "holder_retrieval_accuracy", "holder_retrieval_correct", "holder_retrieval_total", "correct", "total")
        },
        "random_holder_baseline": random_holder_baseline,
        "validation_predictions": after["predictions"],
        "duration_seconds": time.perf_counter() - l1_started,
        "gate_definition": "L0 passes; pilot training loss falls; validation answer accuracy improves over initialization; holder retrieval exceeds chance by 5 percentage points.",
    }

    l2: dict[str, Any] = {
        "stage": "L2",
        "status": "skipped",
        "reason": "L1 gate not met; held-out confirmation was not run.",
        "seed_results": [],
        "per_example": [],
    }
    l2_started = time.perf_counter()
    if stage == "all" and l1_gate:
        seed_results: list[dict[str, Any]] = []
        for seed in range(cfg.confirm_seeds):
            model = LearnedRelationalBinder(
                people=people, objects=objects, locations=locations,
                answers=answers, dim=cfg.dim, seed=seed,
            )
            train_stats = _train(
                model, train_all, epochs=cfg.epochs,
                learning_rate=cfg.learning_rate, seed=seed,
            )
            evaluation = _evaluate(model, test_all)
            seed_results.append({
                "seed": seed,
                "training": train_stats,
                "correct": evaluation["correct"],
                "total": evaluation["total"],
                "accuracy": evaluation["accuracy"],
                "holder_retrieval_correct": evaluation["holder_retrieval_correct"],
                "holder_retrieval_total": evaluation["holder_retrieval_total"],
                "holder_retrieval_accuracy": evaluation["holder_retrieval_accuracy"],
                "predictions": evaluation["predictions"],
            })
        accuracies = [row["accuracy"] for row in seed_results]
        mean = float(np.mean(accuracies)) if accuracies else 0.0
        se = float(np.std(accuracies, ddof=1) / math.sqrt(len(accuracies))) if len(accuracies) > 1 else 0.0
        l2 = {
            "stage": "L2",
            "status": "completed",
            "reason": None,
            "seeds": cfg.confirm_seeds,
            "mean_accuracy": mean,
            "standard_error": se,
            "per_seed_accuracy": accuracies,
            "mean_holder_retrieval_accuracy": float(np.mean([
                row["holder_retrieval_accuracy"] or 0.0 for row in seed_results
            ])),
            "seed_results": seed_results,
            "per_example": [
                {"seed": row["seed"], **prediction}
                for row in seed_results for prediction in row["predictions"]
            ],
            "duration_seconds": time.perf_counter() - l2_started,
        }
    elif stage == "pilot":
        l2["reason"] = "Pilot mode requested; test-set confirmation is intentionally disabled."

    return {
        "schema_version": 1,
        "experiment": "babi_g1l_learned_relational_binding",
        "task": 2,
        "method": "learned identity embeddings + role-specific relation encoders + soft holder-to-person attention + supervised location readout",
        "learning_performed": True,
        "claim_boundary": "Learned binding is tested only over typed QA2 event templates. Event parsing is deterministic; identity/role representation, holder retrieval and answer readout are learned. Not a general language parser.",
        "configuration": {
            **{
                "task": cfg.task, "dim": cfg.dim, "seed": cfg.seed,
                "learning_rate": cfg.learning_rate, "epochs": cfg.epochs,
                "pilot_train_limit": cfg.train_limit, "pilot_validation_limit": cfg.eval_limit,
                "confirmation_seeds": cfg.confirm_seeds, "variant": "full",
                "runtime_ablation": "deferred", "hierarchy": "held_fixed",
            },
            "stage": stage,
        },
        "dataset": {
            "train_stories": len(train_all),
            "validation_stories": len(valid_all),
            "test_stories": len(test_all),
            "train_sha256": _sha256(train_path),
            "valid_sha256": _sha256(valid_path),
            "test_sha256": _sha256(test_path),
            "task": 2,
        },
        "entity_vocabulary": {
            "people": len(people), "objects": len(objects), "locations": len(locations),
            "answer_labels": answers,
        },
        "f0_companion": f0_companion,
        "preprocessing": {
            "split_loads": 3,
            "parsed_examples": len(train_all) + len(valid_all) + len(test_all),
            "cache": "one in-memory typed-event representation reused by L0/L1/L2",
        },
        "stages": {"L0": l0, "L1": l1, "L2": l2},
        "duration_seconds": time.perf_counter() - started,
    }


def save_g1l_bundle(result: dict[str, Any], *, output_dir: Path) -> dict[str, str]:
    run_id = f"g1l_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    bundle = Path(output_dir) / run_id
    bundle.mkdir(parents=True, exist_ok=True)
    (bundle / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    stage_l2 = result["stages"]["L2"]
    rows = stage_l2["per_example"] or [
        {"seed": result["configuration"]["seed"], **p}
        for p in result["stages"]["L1"]["validation_predictions"]
    ]
    fields = [
        "seed", "example_index", "question", "query_object", "target",
        "prediction", "correct", "query_relation", "target_holder",
        "predicted_holder", "holder_correct", "confidence",
    ]
    with (bundle / "per_example.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})
    with (bundle / "per_example.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    l0, l1, l2 = (result["stages"][name] for name in ("L0", "L1", "L2"))
    summary_lines = [
        "# G1-L learned relational binding",
        "",
        f"- L0 mechanism checks: {'PASS' if l0['passed'] else 'FAIL'}",
        f"- L1 pilot: {l1['status']}",
        f"- L1 validation accuracy: {l1['validation_after_training']['accuracy']:.4f}",
        f"- L1 holder retrieval accuracy: {l1['validation_after_training']['holder_retrieval_accuracy']}",
        f"- L2 confirmation: {l2['status']} — {l2.get('reason') or 'completed'}",
        f"- L2 per-seed accuracy: {l2.get('per_seed_accuracy')}",
        f"- L2 mean accuracy: {l2.get('mean_accuracy')}",
        f"- Total duration: {result['duration_seconds']:.2f}s",
        "",
        f"**Claim boundary:** {result['claim_boundary']}",
        "",
        "See per_example.csv and per_example.jsonl for held-out or pilot answer-level and holder-retrieval evidence.",
    ]
    (bundle / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    now = datetime.now(timezone.utc).isoformat()
    metadata = {
        "schema_version": 1,
        "run_id": run_id,
        "experiment": result["experiment"],
        "status": "completed",
        "created_at": now,
        "completed_at": now,
        "code_revision": os.environ.get("GITHUB_SHA"),
        "configuration": result["configuration"],
        "dataset": result["dataset"],
        "duration_seconds": result["duration_seconds"],
        "result_files": ["summary.json", "summary.md", "per_example.csv", "per_example.jsonl"],
    }
    (bundle / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (bundle / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (bundle / "config.json").write_text(json.dumps(result["configuration"], indent=2), encoding="utf-8")
    return {"bundle_dir": str(bundle), "summary_json": str(bundle / "summary.json")}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the staged G1-L learned relational-binding experiment.")
    parser.add_argument("--stage", choices=("all", "pilot"), default="all")
    parser.add_argument("--task", type=int, default=2)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-limit", type=int, default=200)
    parser.add_argument("--eval-limit", type=int, default=100)
    parser.add_argument("--confirm-seeds", type=int, default=3)
    parser.add_argument("--data-dir", type=Path, default=_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-f0-companion", action="store_true",
                        help="Unit-test escape hatch; not used by the research workflow.")
    return parser


def main(argv: list[str] | None = None) -> dict[str, str]:
    args = build_arg_parser().parse_args(argv)
    cfg = G1LConfig(
        task=args.task, dim=args.dim, seed=args.seed,
        learning_rate=args.learning_rate, epochs=args.epochs,
        train_limit=args.train_limit, eval_limit=args.eval_limit,
        confirm_seeds=args.confirm_seeds,
    )
    result = run_g1l_experiment(
        config=cfg, data_dir=args.data_dir, stage=args.stage,
        exercise_f0=not args.skip_f0_companion,
    )
    paths = save_g1l_bundle(result, output_dir=args.output_dir)
    stages = result["stages"]
    print(f"G1-L L0={stages['L0']['passed']} L1={stages['L1']['status']} L2={stages['L2']['status']}")
    print(f"L1 validation accuracy={stages['L1']['validation_after_training']['accuracy']:.4f}")
    if stages["L2"]["status"] == "completed":
        print(f"L2 test mean accuracy={stages['L2']['mean_accuracy']:.4f} per_seed={stages['L2']['per_seed_accuracy']}")
    else:
        print(f"L2 skipped: {stages['L2']['reason']}")
    print(f"bundle={paths['bundle_dir']}")
    return paths


if __name__ == "__main__":
    main()
