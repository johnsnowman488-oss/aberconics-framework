"""Stateful multi-level digital memory stepping over the hierarchy C ABI.

This is the D4B streaming surface made task-facing.  ``DigitalEpisode``
forcing schedules are stepped through ``gfe_c_hierarchical_step_chain_spec``
one event at a time.  The returned level-0 trace keeps the same shape as a
``DigitalDirector`` trace, so eligibility/readout layers written against the
single-level Python stepper can consume hierarchy dynamics unchanged.

Substrate note (verified against the C++ sources, 2026-09-09)
-------------------------------------------------------------
The chain-spec ABI integrates the C++ ABERSOE Form-B scheme
(``gfe::step_augmented`` + ``abersoe::step``).  Its discrete memory has two
properties that differ from the Python digital contract ``step_memory``:

- ``chi`` is a single *scalar* per memory channel, driven by
  ``u[coupling_index]``, and the memory feedback ``sum_k w_k * chi_k`` enters
  only ``u[coupling_index]``;
- the feedback is evaluated from the *pre-step* ``chi`` (explicit Euler),
  whereas ``step_memory`` evaluates feedback from the *post-step* ``chi``
  (semi-implicit Euler) and applies it elementwise on every coordinate.

Both are first-order Euler-B discretizations of the same continuous model,
and for a one-dimensional state with memory feedback disabled they are
exact equals.  Hierarchy experiments that must be numerically comparable to
the existing D1--E1 baselines therefore need this substrate difference
documented explicitly rather than assuming ABI/Python parity.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field

from ..ffi import step_hierarchy_raw
from .bridge import DigitalMemoryConfig, DigitalMemoryState
from .director import DigitalEpisode
from .traces import DigitalTrace, DigitalTraceStep


def _initial_level_state(level: dict[str, object]) -> dict[str, object]:
    u = level.get("u")
    if not u:
        raise ValueError("hierarchy level must define a non-empty initial u")
    gamma = [float(value) for value in level.get("gamma", [])]
    chi = level.get("chi")
    if chi is None or not chi:
        chi = [0.0 for _ in gamma]
    else:
        chi = [float(value) for value in chi]
    if gamma and len(chi) != len(gamma):
        raise ValueError("level initial chi must match kernel channel count when provided")
    return {
        "u": [float(value) for value in u],
        "chi": chi,
        "t": float(level.get("t", 0.0)),
    }


@dataclass(slots=True)
class HierarchicalEpisodeResult:
    """Outcome of one hierarchy episode with per-level traces and diagnostics."""

    final_state: DigitalMemoryState
    level_traces: list[DigitalTrace]
    active_kernels: list[list[dict[str, object]]]
    spectral_units: list[list[dict[str, float]]]
    target: str | None
    query: list[float]
    metadata: dict[str, object]

    @property
    def trace(self) -> DigitalTrace:
        """Level-0 trace, shaped like a ``DigitalDirector`` trace."""
        return self.level_traces[0]


@dataclass(slots=True)
class HierarchicalDigitalDirector:
    """Stream ``DigitalEpisode`` forcing schedules through the hierarchy C ABI.

    ``levels``/``edges`` use the chain-spec dict format consumed by
    :func:`d2c.ffi.step_hierarchy_raw`.  A single-level flat chain (empty
    ``edges``, one level) reproduces the C++ ABERSOE substrate without any
    hierarchy coupling and doubles as the ABI-vs-Python parity vehicle.

    External forcing is injected at level 0 on every step; silent steps are
    zero vectors of ``state_dim``.
    """

    lib: ctypes.CDLL
    levels: list[dict[str, object]]
    edges: list[dict[str, object]] = field(default_factory=list)
    state_dim: int = 1
    memory_config: DigitalMemoryConfig | None = None

    def __post_init__(self) -> None:
        if not self.levels:
            raise ValueError("at least one hierarchy level is required")
        if self.state_dim <= 0:
            raise ValueError("state_dim must be positive")
        for index, level in enumerate(self.levels):
            u = level.get("u")
            if not u or len(u) != self.state_dim:
                raise ValueError(
                    f"hierarchy level {index} initial u must have dimension state_dim={self.state_dim}"
                )

    def run_episode(
        self,
        episode: DigitalEpisode,
        *,
        experiment_name: str = "hierarchical_digital_episode",
    ) -> HierarchicalEpisodeResult:
        """Run one episode, calling the hierarchy step ABI once per forcing step."""
        if len(episode.forcings) != len(episode.tokens):
            raise ValueError("episode forcings and tokens must have matching lengths")
        if not episode.forcings:
            raise ValueError("episode must contain at least one forcing step")

        level_states = [_initial_level_state(level) for level in self.levels]
        traces = [
            DigitalTrace(
                experiment_name=f"{experiment_name}::level{index}",
                metadata={"level": index},
            )
            for index in range(len(self.levels))
        ]
        kernels_all: list[list[dict[str, object]]] = []
        spectral_all: list[list[dict[str, float]]] = []

        for step_index, (forcing, token) in enumerate(zip(episode.forcings, episode.tokens)):
            forcing_values = [float(value) for value in forcing]
            if len(forcing_values) != self.state_dim:
                raise ValueError("forcing dimension must match director state_dim")

            step_result = step_hierarchy_raw(
                self.lib,
                self.levels,
                self.edges,
                level_states=level_states,
                external_forcing=forcing_values,
                forcing_level=0,
            )
            level_states = step_result["level_states"]
            active_kernels = step_result["active_kernels"]
            spectral_units = step_result["spectral_units"]
            kernels_all.append(active_kernels)
            spectral_all.append(spectral_units)

            for level_index, (level_state, trace) in enumerate(zip(level_states, traces)):
                # The ABI reports chi as one scalar per channel; store each as
                # a single-element channel vector to preserve the
                # DigitalDirector trace convention (channel vectors per state
                # coordinate).
                chi_channels = [[float(value)] for value in level_state["chi"]]
                trace.add_step(
                    DigitalTraceStep(
                        step=step_index,
                        t=float(level_state["t"]),
                        token=token if level_index == 0 else None,
                        forcing=(
                            list(forcing_values)
                            if level_index == 0
                            else [0.0] * len(level_state["u"])
                        ),
                        u=[float(value) for value in level_state["u"]],
                        chi=chi_channels,
                        metadata={
                            "active_kernel": active_kernels[level_index],
                            "spectral": spectral_units[level_index],
                        },
                    )
                )

        level0 = level_states[0]
        final_state = DigitalMemoryState(
            u=[float(value) for value in level0["u"]],
            chi=[[float(value)] for value in level0["chi"]],
            t=float(level0["t"]),
        )
        return HierarchicalEpisodeResult(
            final_state=final_state,
            level_traces=traces,
            active_kernels=kernels_all,
            spectral_units=spectral_all,
            target=episode.target,
            query=list(episode.query),
            metadata=dict(episode.metadata),
        )
