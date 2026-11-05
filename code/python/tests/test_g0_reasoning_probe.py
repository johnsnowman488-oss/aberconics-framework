from d2c.experiments.g0_reasoning_probe import (
    answer_from_world,
    controlled_entity_binding_tests,
    replay_world,
)


def test_qa2_oracle_tracks_carried_object_across_person_movement() -> None:
    world = replay_world([
        "Mary moved to the kitchen.",
        "Mary got the football there.",
        "Mary travelled to the hallway.",
    ])
    assert answer_from_world("Where is the football?", world)[0] == "hallway"


def test_qa2_oracle_handles_pickup_before_initial_location_is_observed() -> None:
    world = replay_world([
        "Mary got the milk there.",
        "Mary moved to the garden.",
    ])
    assert answer_from_world("Where is the milk?", world)[0] == "garden"


def test_drop_releases_object_so_later_person_movement_does_not_move_it() -> None:
    world = replay_world([
        "Mary moved to the kitchen.",
        "Mary got the football there.",
        "Mary went to the hallway.",
        "Mary put down the football.",
        "Mary went to the garden.",
    ])
    assert answer_from_world("Where is the football?", world)[0] == "hallway"


def test_controlled_entity_binding_counterfactuals_all_pass() -> None:
    report = controlled_entity_binding_tests()
    assert report["passed"], report["tests"]
