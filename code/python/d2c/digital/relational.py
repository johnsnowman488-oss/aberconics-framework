"""Inspectable relational world state for bAbI QA2.

This deterministic symbolic diagnostic makes identity, holder, and location
relationships explicit. It is not a learned parser or learned readout.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
import re
from typing import Any

_MOVE = re.compile(r"(?P<actor>[A-Z][a-z]+) (?:(?:went(?: back)?)|moved|travelled|journeyed) to the (?P<location>[a-z]+)\.")
_PICKUP = re.compile(r"(?P<actor>[A-Z][a-z]+) (?:got|grabbed|took|picked up) the (?P<object>[a-z]+) there\.")
_RELEASE = re.compile(r"(?P<actor>[A-Z][a-z]+) (?:left|dropped|discarded|put down) the (?P<object>[a-z]+)(?: there)?\.")
_QUESTION = re.compile(r"Where is the (?P<object>[a-z]+)\?")


@dataclass(frozen=True, slots=True)
class RelationalEvent:
    kind: str
    actor: str
    location: str | None = None
    object: str | None = None

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


def parse_babi_statement(statement: str) -> RelationalEvent:
    text = statement.strip()
    match = _MOVE.fullmatch(text)
    if match:
        return RelationalEvent("move", match.group("actor"), location=match.group("location"))
    match = _PICKUP.fullmatch(text)
    if match:
        return RelationalEvent("pickup", match.group("actor"), object=match.group("object"))
    match = _RELEASE.fullmatch(text)
    if match:
        return RelationalEvent("release", match.group("actor"), object=match.group("object"))
    raise ValueError(f"unrecognized bAbI QA2 statement: {statement!r}")


def parse_babi_question(question: str) -> str:
    match = _QUESTION.fullmatch(question.strip())
    if not match:
        raise ValueError(f"unrecognized bAbI QA2 question: {question!r}")
    return match.group("object")


@dataclass(slots=True)
class RelationalWorldState:
    person_locations: dict[str, str] = field(default_factory=dict)
    object_holders: dict[str, str | None] = field(default_factory=dict)
    object_locations: dict[str, str | None] = field(default_factory=dict)
    event_count: int = 0
    move_count: int = 0
    pickup_count: int = 0
    release_count: int = 0
    event_log: list[dict[str, Any]] = field(default_factory=list)

    def apply(self, event: RelationalEvent) -> None:
        before = self.resolve_object(event.object) if event.object is not None else None
        if event.kind == "move":
            if event.location is None:
                raise ValueError("move event requires a location")
            self.person_locations[event.actor] = event.location
            self.move_count += 1
        elif event.kind == "pickup":
            if event.object is None:
                raise ValueError("pickup event requires an object")
            self.object_holders[event.object] = event.actor
            self.object_locations[event.object] = None
            self.pickup_count += 1
        elif event.kind == "release":
            if event.object is None:
                raise ValueError("release event requires an object")
            self.object_holders[event.object] = None
            self.object_locations[event.object] = self.person_locations.get(event.actor)
            self.release_count += 1
        else:
            raise ValueError(f"unsupported event kind: {event.kind!r}")
        self.event_count += 1
        self.event_log.append({
            **event.to_mapping(),
            "actor_location_after": self.person_locations.get(event.actor),
            "object_location_before": before,
            "object_location_after": self.resolve_object(event.object) if event.object is not None else None,
        })

    def resolve_object(self, object_name: str | None) -> str | None:
        if object_name is None:
            return None
        holder = self.object_holders.get(object_name)
        if holder is not None:
            return self.person_locations.get(holder)
        return self.object_locations.get(object_name)

    def to_mapping(self) -> dict[str, Any]:
        objects = sorted(set(self.object_holders) | set(self.object_locations))
        return {
            "person_locations": dict(sorted(self.person_locations.items())),
            "objects": {
                name: {"held_by": self.object_holders.get(name), "location": self.resolve_object(name)}
                for name in objects
            },
            "event_count": self.event_count,
            "move_count": self.move_count,
            "pickup_count": self.pickup_count,
            "release_count": self.release_count,
        }
