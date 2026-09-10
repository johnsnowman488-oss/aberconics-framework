"""Parity and plumbing tests for the D4B hierarchy streaming director.

These tests validate the Step-1 goal: does the flat single-level chain through
``gfe_c_hierarchical_step_chain_spec`` reproduce the C++ ABERSOE Form-B
scheme exactly, and where does it agree (and honestly, where does it not
agree) with the Python digital contract stepper?

The exact agreements are asserted with tight tolerance.  The documented
substrate difference (scalar-per-channel explicit-Euler memory in the ABI
vs per-coordinate semi-implicit memory in ``step_memory``) is covered by the
module docstring in ``d2c.digital.hierarchy_director`` and the Context ledger.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from d2c.digital import (
    DigitalDirector,
    DigitalEpisode,
    DigitalMemoryConfig,
)
from d2c.digital.hierarchy_director import HierarchicalDigitalDirector
from gfe_ctypes import (
    GFE_C_COUPLING_FORM_B,
    GFE_C_HIERARCHICAL_RELATION_BOTTOM_UP,
    load_gfe_library,
)


def _lib():
    try:
        return load_gfe_library(None)
    except OSError as exc:
        pytest.skip(f"shared library unavailable: {exc}")


def _flat_level(state_dim, *, gamma, w, dt, leak) -> dict[str, object]:
    return {
        "name": "flat",
        "gamma": [float(value) for value in gamma],
        "w": [float(value) for value in w],
        "u": [0.0] * state_dim,
        "chi": [0.0] * len(gamma),
        "dt": dt,
        "linear_decay": [leak] * state_dim,
        "forcing_bias": [0.0] * state_dim,
        "form": GFE_C_COUPLING_FORM_B,
        "coupling_index": 0,
    }


def _euler_b_reference(initial_u, initial_chi, *, gamma, w, leak, dt, forcings):
    """Local reference for the C++ ABERSOE Form-B scheme.

    Matches ``gfe::step_augmented`` for a one-dimensional state with
    coupling_index 0: feedback uses PRE-step chi and enters u[0] only;
    chi_next is driven by the PRE-step u[0].
    """
    u = [float(value) for value in initial_u]
    chi = [float(value) for value in initial_chi]
    out = []
    for x in forcings:
        feedback = sum(wk * ck for wk, ck in zip(w, chi))
        u_next = [u[0] + dt * (-leak * u[0] - feedback + x[0])]
        chi_next = [ck + dt * (-rk * ck + u[0]) for rk, ck in zip(gamma, chi)]
        out.append({"u": u_next, "chi": chi_next})
        u, chi = u_next, chi_next
    return out


def _episode_from(sequence):
    forcings = [[float(value)] for value in sequence]
    return DigitalEpisode(forcings=forcings, tokens=[str(index) for index in range(len(forcings))])


def test_flat_abi_matches_euler_b_reference_with_feedback():
    """Flat 1-dim chain matches the C++ Form-B scheme exactly (full feedback)."""
    lib = _lib()
    gamma = [2.0, 0.45]
    w = [0.16, 0.108]
    leak = 2.0
    dt = 0.05
    sequence = [1.0, 0.0, 1.0, -0.5, 0.5, 0.0, 0.25]

    level = _flat_level(1, gamma=gamma, w=w, dt=dt, leak=leak)
    director = HierarchicalDigitalDirector(
        lib=lib, levels=[level], edges=[], state_dim=1,
        memory_config=DigitalMemoryConfig(gamma=gamma, w=w, dt=dt, leak_rate=leak, form="negative_feedback"),
    )
    result = director.run_episode(_episode_from(sequence), experiment_name="parity_reference")

    reference = _euler_b_reference([0.0], [0.0, 0.0], gamma=gamma, w=w, leak=leak, dt=dt, forcings=[[x] for x in sequence])

    assert len(result.trace.steps) == len(sequence)
    for index, (step, ref) in enumerate(zip(result.trace.steps, reference)):
        assert step.u[0] == pytest.approx(ref["u"][0], abs=1e-9)
        assert len(step.chi) == len(gamma)
        for channel, ref_chi in zip(step.chi, ref["chi"]):
            assert channel[0] == pytest.approx(ref_chi, abs=1e-9)
        assert step.t == pytest.approx((index + 1) * dt, abs=1e-9)

    # Active kernels should round-trip the spec unchanged for a flat chain.
    for kernels in result.active_kernels:
        assert kernels[0]["gamma"] == pytest.approx(gamma, abs=1e-12)
        assert kernels[0]["w"] == pytest.approx(w, abs=1e-12)
    # Spectral units must be populated for every step.
    for spectral in result.spectral_units:
        assert spectral[0]["deff"] > 0.0
        assert spectral[0]["mcap"] > 0.0


def test_flat_abi_matches_python_director_when_feedback_disabled():
    """With memory feedback disabled both substrates implement the same ODE exactly."""
    lib = _lib()
    gamma = [2.0, 0.45]
    # Feedback "disabled" for the shared-surface comparison: the chain-spec
    # validator rejects an exactly zero-sum weight kernel, so use weights
    # small enough that feedback stays far inside the 1e-9 tolerance.
    w = [1e-12, 1e-12]
    leak = 2.0
    dt = 0.05
    sequence = [1.0, 0.0, 1.0, 0.0, -0.5, 0.0]

    config = DigitalMemoryConfig(gamma=gamma, w=w, dt=dt, leak_rate=leak, form="negative_feedback")
    episode = _episode_from(sequence)

    python_result = DigitalDirector(memory_config=config, state_dim=1).run_episode(episode, experiment_name="parity_py")

    level = _flat_level(1, gamma=gamma, w=w, dt=dt, leak=leak)
    abi_director = HierarchicalDigitalDirector(lib=lib, levels=[level], edges=[], state_dim=1, memory_config=config)
    abi_result = abi_director.run_episode(episode, experiment_name="parity_abi")

    assert len(abi_result.trace.steps) == len(python_result.trace.steps)
    for abi_step, py_step in zip(abi_result.trace.steps, python_result.trace.steps):
        assert abi_step.u[0] == pytest.approx(py_step.u[0], abs=1e-9)
        assert len(abi_step.chi) == len(py_step.chi)
        for abi_channel, py_channel in zip(abi_step.chi, py_step.chi):
            assert abi_channel[0] == pytest.approx(py_channel[0], abs=1e-9)
        assert abi_step.t == pytest.approx(py_step.t, abs=1e-9)


def test_trace_shape_matches_digital_director():
    """Level-0 ABI trace keeps the DigitalDirector field shape for eligibility reuse."""
    lib = _lib()
    gamma = [2.0, 0.45]
    # Negligible feedback weight: chain-spec validator forbids exactly zero sum.
    config = DigitalMemoryConfig(gamma=gamma, w=[1e-12, 1e-12], dt=0.05, leak_rate=2.0, form="negative_feedback")
    episode = _episode_from([1.0, 0.0, 1.0])

    python_result = DigitalDirector(memory_config=config, state_dim=1).run_episode(episode)
    abi_result = HierarchicalDigitalDirector(
        lib=lib,
        levels=[_flat_level(1, gamma=gamma, w=[1e-12, 1e-12], dt=0.05, leak=2.0)],
        edges=[],
        state_dim=1,
    ).run_episode(episode)

    expected_fields = {"step", "t", "token", "forcing", "u", "chi", "metadata"}
    for step in abi_result.trace.steps:
        mapping = step.to_mapping()
        assert set(mapping) == expected_fields
        assert isinstance(mapping["chi"][0], list)

    py_final = python_result.final_state
    assert abi_result.final_state.u == pytest.approx(py_final.u, abs=1e-9)
    assert abi_result.final_state.t == pytest.approx(py_final.t, abs=1e-9)


def test_two_level_bottom_up_coupling_and_diagnostics():
    """A two-level chain steps a context level; coupling and spectral work."""
    lib = _lib()
    level0 = _flat_level(1, gamma=[2.0, 0.45], w=[0.16, 0.108], dt=0.05, leak=2.0)
    level1 = _flat_level(1, gamma=[0.35, 0.08], w=[0.05, 0.02], dt=0.05, leak=0.5)
    level1["name"] = "context"
    edges = [
        {
            "source_level": 0,
            "target_level": 1,
            "relation": GFE_C_HIERARCHICAL_RELATION_BOTTOM_UP,
            "gain": 0.4,
            "normalize_weights": True,
        },
    ]
    director = HierarchicalDigitalDirector(lib=lib, levels=[level0, level1], edges=edges, state_dim=1)

    sequence = [1.0, 0.0, 1.0, 0.0, 0.0]
    result = director.run_episode(_episode_from(sequence), experiment_name="d4b_two_level")

    assert len(result.level_traces) == 2
    context_trace = result.level_traces[1]
    # Bottom-up coupling drives the context level away from its zero start.
    assert any(abs(step.u[0]) > 1e-9 for step in context_trace.steps)
    # Spectral units populated per level.
    assert len(result.spectral_units) == len(sequence)
    for spectral in result.spectral_units:
        assert spectral[0]["deff"] > 0.0
        assert spectral[1]["deff"] > 0.0
    for kernels in result.active_kernels:
        assert kernels[0]["gamma"] == pytest.approx([2.0, 0.45], abs=1e-12)
        assert kernels[1]["gamma"] == pytest.approx([0.35, 0.08], abs=1e-12)


def test_forcing_dimension_mismatch_rejected():
    """Forcing shorter than state_dim is rejected instead of silently under-driving."""
    lib = _lib()
    director = HierarchicalDigitalDirector(
        lib=lib,
        levels=[_flat_level(2, gamma=[2.0, 0.45], w=[0.16, 0.108], dt=0.05, leak=2.0)],
        edges=[],
        state_dim=2,
    )
    episode = DigitalEpisode(forcings=[[1.0]], tokens=["A"])
    with pytest.raises(ValueError, match="state_dim"):
        director.run_episode(episode)


def test_run_is_deterministic():
    """The same episode through the same library yields identical outputs."""
    lib = _lib()
    director = HierarchicalDigitalDirector(
        lib=lib,
        levels=[_flat_level(1, gamma=[2.0, 0.45], w=[0.16, 0.108], dt=0.05, leak=2.0)],
        edges=[],
        state_dim=1,
    )
    episode = _episode_from([1.0, 0.0, 1.0])
    first = director.run_episode(episode, experiment_name="deterministic")
    second = director.run_episode(episode, experiment_name="deterministic")

    assert first.final_state.to_mapping() == second.final_state.to_mapping()
    assert first.trace.to_mapping() == second.trace.to_mapping()
    assert first.active_kernels == second.active_kernels
    assert first.spectral_units == second.spectral_units
