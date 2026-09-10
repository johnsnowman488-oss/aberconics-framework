"""Thin convenience layer over the raw ctypes wrapper."""

from __future__ import annotations

import ctypes
from typing import Sequence

import gfe_ctypes as gfe


def load_library(path: str | None = None) -> ctypes.CDLL:
    """Load the shared library used by the Python D2C layer."""

    return gfe.load_gfe_library(path)


def available_abersoe_scenarios(lib: ctypes.CDLL) -> list[str]:
    """Return the registered single-level scenario names."""

    return gfe.list_abersoe_scenarios(lib)


def available_hierarchical_scenarios(lib: ctypes.CDLL) -> list[str]:
    """Return the registered hierarchical scenario names."""

    return gfe.list_hierarchical_scenarios(lib)


def run_abersoe_raw(
    lib: ctypes.CDLL,
    scenario: int,
    *,
    dt: float = 0.01,
    form: int = gfe.GFE_C_COUPLING_FORM_B,
    steps: int = 100,
    sample_stride: int = 10,
    strict_finite: bool = True,
) -> dict[str, object]:
    return gfe.run_abersoe_scenario(
        lib,
        scenario,
        dt=dt,
        form=form,
        steps=steps,
        sample_stride=sample_stride,
        strict_finite=strict_finite,
    )


def step_abersoe_raw(
    lib: ctypes.CDLL,
    scenario: int,
    dt: float,
    form: int,
    *,
    u: Sequence[float],
    chi: Sequence[float],
    t: float,
    gamma: Sequence[float] | None = None,
    w: Sequence[float] | None = None,
) -> dict[str, object]:
    return gfe.step_abersoe(
        lib,
        scenario,
        dt,
        form,
        u=u,
        chi=chi,
        t=t,
        gamma=gamma,
        w=w,
    )


def run_hierarchical_raw(
    lib: ctypes.CDLL,
    scenario: int,
    *,
    enable_bottom_up: bool = True,
    enable_top_down: bool = True,
    steps: int = 80,
    sample_stride: int = 10,
    strict_finite: bool = True,
) -> dict[str, object]:
    return gfe.run_hierarchical_scenario(
        lib,
        scenario,
        enable_bottom_up=enable_bottom_up,
        enable_top_down=enable_top_down,
        steps=steps,
        sample_stride=sample_stride,
        strict_finite=strict_finite,
    )


def validate_chain_raw(
    lib: ctypes.CDLL,
    levels: list[dict[str, object]],
    edges: list[dict[str, object]],
) -> None:
    gfe.validate_hierarchical_chain_spec(lib, levels, edges)


def run_hierarchical_chain_raw(
    lib: ctypes.CDLL,
    levels: list[dict[str, object]],
    edges: list[dict[str, object]],
    *,
    steps: int = 80,
    sample_stride: int = 10,
    strict_finite: bool = True,
) -> dict[str, object]:
    return gfe.run_hierarchical_chain_spec(
        lib,
        levels,
        edges,
        steps=steps,
        sample_stride=sample_stride,
        strict_finite=strict_finite,
    )


def get_chain_cross_level_report_raw(
    lib: ctypes.CDLL,
    levels: list[dict[str, object]],
    edges: list[dict[str, object]],
    *,
    steps: int = 80,
    sample_stride: int = 10,
    strict_finite: bool = True,
) -> dict[str, object]:
    return gfe.get_hierarchical_cross_level_report_for_chain_spec(
        lib,
        levels,
        edges,
        steps=steps,
        sample_stride=sample_stride,
        strict_finite=strict_finite,
    )


def get_chain_renorm_report_raw(
    lib: ctypes.CDLL,
    levels: list[dict[str, object]],
    edges: list[dict[str, object]],
) -> list[dict[str, object]]:
    return gfe.get_hierarchical_renorm_report_for_chain_spec(lib, levels, edges)


def step_hierarchy_raw(
    lib: ctypes.CDLL,
    levels: list[dict[str, object]],
    edges: list[dict[str, object]],
    *,
    level_states: list[dict[str, object]],
    external_forcing: Sequence[float],
    forcing_level: int = 0,
) -> dict[str, object]:
    """Stateful single-step of the hierarchy with external forcing injection.

    This is the D4B streaming interface: the caller owns the per-level state
    dicts and calls this function once per token event.
    """
    return gfe.step_hierarchical_chain_spec(
        lib,
        levels,
        edges,
        level_states=level_states,
        external_forcing=external_forcing,
        forcing_level=forcing_level,
    )


def get_scenario_cross_level_report_raw(
    lib: ctypes.CDLL,
    scenario: int,
    *,
    enable_bottom_up: bool = True,
    enable_top_down: bool = True,
    steps: int = 80,
    sample_stride: int = 10,
    strict_finite: bool = True,
) -> dict[str, object]:
    return gfe.get_hierarchical_cross_level_report(
        lib,
        scenario,
        enable_bottom_up=enable_bottom_up,
        enable_top_down=enable_top_down,
        steps=steps,
        sample_stride=sample_stride,
        strict_finite=strict_finite,
    )


def get_scenario_renorm_report_raw(
    lib: ctypes.CDLL,
    scenario: int,
    *,
    enable_bottom_up: bool = True,
    enable_top_down: bool = True,
) -> list[dict[str, object]]:
    return gfe.get_hierarchical_renorm_report(
        lib,
        scenario,
        enable_bottom_up=enable_bottom_up,
        enable_top_down=enable_top_down,
    )
