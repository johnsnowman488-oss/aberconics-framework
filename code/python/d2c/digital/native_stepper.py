"""ctypes adapter for the vectorized C++ digital-memory ABI.

DigitalRuntime keeps scheduling, stability policy, trace storage, checkpointing
and readout in Python. This adapter replaces only the numerical memory update.
"""
from __future__ import annotations

import ctypes as ct
from typing import Sequence

import numpy as np

from .bridge import DigitalMemoryConfig, DigitalMemoryState


class _Config(ct.Structure):
    _fields_ = [("state_dim", ct.c_size_t), ("channel_count", ct.c_size_t),
                ("dt", ct.c_double), ("leak_rate", ct.c_double), ("form", ct.c_int)]


class _Kernel(ct.Structure):
    _fields_ = [("gamma", ct.POINTER(ct.c_double)), ("gamma_size", ct.c_size_t),
                ("weights", ct.POINTER(ct.c_double)), ("weights_size", ct.c_size_t)]


class _State(ct.Structure):
    _fields_ = [("u", ct.POINTER(ct.c_double)), ("u_size", ct.c_size_t),
                ("chi", ct.POINTER(ct.c_double)), ("chi_size", ct.c_size_t), ("t", ct.c_double)]


class _StateOut(ct.Structure):
    _fields_ = [("u", ct.POINTER(ct.c_double)), ("u_capacity", ct.c_size_t),
                ("u_size", ct.POINTER(ct.c_size_t)), ("chi", ct.POINTER(ct.c_double)),
                ("chi_capacity", ct.c_size_t), ("chi_size", ct.POINTER(ct.c_size_t)),
                ("t", ct.POINTER(ct.c_double))]


class _TrajectoryOut(ct.Structure):
    _fields_ = [("t", ct.POINTER(ct.c_double)), ("t_capacity", ct.c_size_t),
                ("t_size", ct.POINTER(ct.c_size_t)), ("u", ct.POINTER(ct.c_double)),
                ("u_capacity", ct.c_size_t), ("u_size", ct.POINTER(ct.c_size_t)),
                ("chi", ct.POINTER(ct.c_double)), ("chi_capacity", ct.c_size_t),
                ("chi_size", ct.POINTER(ct.c_size_t))]


class _Diagnostics(ct.Structure):
    _fields_ = [("steps_executed", ct.c_size_t), ("all_finite", ct.c_int),
                ("max_abs_u", ct.c_double), ("max_abs_chi", ct.c_double)]


class CppVectorizedMemoryStepper:
    """Drop-in numerical stepper for DigitalRuntime, plus a sequence API."""

    def __init__(self, memory_config: DigitalMemoryConfig, *, library_path: str | None = None):
        # Lazy loading preserves a usable Python-only path without a native build.
        if library_path is None:
            import gfe_ctypes
            self._lib = gfe_ctypes.load_gfe_library()
        else:
            self._lib = ct.CDLL(library_path)
        self._bind()
        self._signature_cache: tuple[object, ...] | None = None
        self._config: _Config | None = None
        self._gamma: np.ndarray | None = None
        self._weights: np.ndarray | None = None
        self._kernel: _Kernel | None = None
        self._refresh(memory_config)

    def _bind(self) -> None:
        self._lib.gfe_c_digital_step.argtypes = [
            ct.POINTER(_Config), ct.POINTER(_Kernel), ct.POINTER(_State),
            ct.POINTER(ct.c_double), ct.c_size_t, ct.POINTER(_StateOut),
            ct.POINTER(ct.c_char), ct.c_size_t,
        ]
        self._lib.gfe_c_digital_step.restype = ct.c_int
        self._lib.gfe_c_digital_run_sequence.argtypes = [
            ct.POINTER(_Config), ct.POINTER(_Kernel), ct.POINTER(_State),
            ct.POINTER(ct.c_double), ct.c_size_t, ct.c_size_t,
            ct.POINTER(_StateOut), ct.POINTER(_TrajectoryOut),
            ct.POINTER(_Diagnostics), ct.POINTER(ct.c_char), ct.c_size_t,
        ]
        self._lib.gfe_c_digital_run_sequence.restype = ct.c_int

    @staticmethod
    def _signature(config: DigitalMemoryConfig) -> tuple[object, ...]:
        return (tuple(map(float, config.gamma)), tuple(map(float, config.w)),
                float(config.dt), float(config.leak_rate), str(config.form))

    def _refresh(self, config: DigitalMemoryConfig) -> None:
        signature = self._signature(config)
        if signature == self._signature_cache:
            return
        if config.form not in {"input_driven", "negative_feedback"}:
            raise ValueError(f"unsupported digital memory form: {config.form!r}")
        self._gamma = np.ascontiguousarray(config.gamma, dtype=np.float64)
        self._weights = np.ascontiguousarray(config.w, dtype=np.float64)
        self._config = _Config(0, len(config.gamma), float(config.dt), float(config.leak_rate),
                               0 if config.form == "input_driven" else 1)
        double_ptr = ct.POINTER(ct.c_double)
        self._kernel = _Kernel(self._gamma.ctypes.data_as(double_ptr), self._gamma.size,
                               self._weights.ctypes.data_as(double_ptr), self._weights.size)
        self._signature_cache = signature

    @staticmethod
    def _ptr(array: np.ndarray):
        return array.ctypes.data_as(ct.POINTER(ct.c_double))

    @staticmethod
    def _check(status: int, error: ct.Array) -> None:
        if status == 0:
            return
        message = error.value.decode("utf-8", errors="replace") or "unknown native error"
        if status == 1:
            raise ValueError(f"vectorized C++ ABI rejected input ({status}): {message}")
        raise RuntimeError(f"vectorized C++ ABI failed ({status}): {message}")

    def __call__(self, state: DigitalMemoryState, forcing: Sequence[float],
                 config: DigitalMemoryConfig) -> DigitalMemoryState:
        self._refresh(config)
        assert self._config is not None and self._kernel is not None
        dim, channels = len(state.u), len(config.gamma)
        self._config.state_dim = dim
        u_in = np.ascontiguousarray(state.u, dtype=np.float64)
        chi_in = np.ascontiguousarray(state.chi, dtype=np.float64).reshape(-1)
        x = np.ascontiguousarray(forcing, dtype=np.float64)
        if x.ndim != 1 or x.size != dim:
            raise ValueError("forcing dimension must match state.u")
        u_out, chi_out = np.empty(dim), np.empty(channels * dim)
        u_size, chi_size, t_out = ct.c_size_t(), ct.c_size_t(), ct.c_double()
        state_view = _State(self._ptr(u_in), u_in.size, self._ptr(chi_in), chi_in.size, float(state.t))
        out_view = _StateOut(self._ptr(u_out), u_out.size, ct.pointer(u_size),
                             self._ptr(chi_out), chi_out.size, ct.pointer(chi_size), ct.pointer(t_out))
        error = ct.create_string_buffer(512)
        status = self._lib.gfe_c_digital_step(
            ct.byref(self._config), ct.byref(self._kernel), ct.byref(state_view),
            self._ptr(x), x.size, ct.byref(out_view), error, len(error))
        self._check(status, error)
        return DigitalMemoryState(u_out.tolist(), chi_out.reshape(channels, dim).tolist(), float(t_out.value))

    def run_sequence(self, initial_state: DigitalMemoryState,
                     forcings: Sequence[Sequence[float]],
                     config: DigitalMemoryConfig) -> dict[str, object]:
        """Execute contiguous forcings [steps, state_dim] in a single native call."""
        self._refresh(config)
        assert self._config is not None and self._kernel is not None
        dim, channels = len(initial_state.u), len(config.gamma)
        x = np.ascontiguousarray(forcings, dtype=np.float64)
        if x.size == 0 and len(forcings) == 0:
            x = np.empty((0, dim), dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != dim:
            raise ValueError("forcings must have shape [steps, state_dim]")
        steps = x.shape[0]
        self._config.state_dim = dim
        u_in = np.ascontiguousarray(initial_state.u, dtype=np.float64)
        chi_in = np.ascontiguousarray(initial_state.chi, dtype=np.float64).reshape(-1)
        u_out, chi_out = np.empty(dim), np.empty(channels * dim)
        u_size, chi_size, t_out = ct.c_size_t(), ct.c_size_t(), ct.c_double()
        state_view = _State(self._ptr(u_in), u_in.size, self._ptr(chi_in), chi_in.size, float(initial_state.t))
        final_view = _StateOut(self._ptr(u_out), u_out.size, ct.pointer(u_size),
                               self._ptr(chi_out), chi_out.size, ct.pointer(chi_size), ct.pointer(t_out))
        times, traj_u, traj_chi = np.empty(steps + 1), np.empty((steps + 1, dim)), np.empty((steps + 1, channels, dim))
        ts, us, cs = ct.c_size_t(), ct.c_size_t(), ct.c_size_t()
        trajectory = _TrajectoryOut(self._ptr(times), times.size, ct.pointer(ts),
                                    self._ptr(traj_u.reshape(-1)), traj_u.size, ct.pointer(us),
                                    self._ptr(traj_chi.reshape(-1)), traj_chi.size, ct.pointer(cs))
        diagnostics, error = _Diagnostics(), ct.create_string_buffer(512)
        forcing_ptr = self._ptr(x.reshape(-1)) if x.size else None
        status = self._lib.gfe_c_digital_run_sequence(
            ct.byref(self._config), ct.byref(self._kernel), ct.byref(state_view),
            forcing_ptr, steps, x.size, ct.byref(final_view), ct.byref(trajectory),
            ct.byref(diagnostics), error, len(error))
        self._check(status, error)
        return {
            "final_state": DigitalMemoryState(u_out.tolist(), chi_out.reshape(channels, dim).tolist(), float(t_out.value)),
            "times": times.tolist(), "u": traj_u.tolist(), "chi": traj_chi.tolist(),
            "diagnostics": {"steps_executed": int(diagnostics.steps_executed),
                            "all_finite": bool(diagnostics.all_finite),
                            "max_abs_u": float(diagnostics.max_abs_u),
                            "max_abs_chi": float(diagnostics.max_abs_chi)},
        }
