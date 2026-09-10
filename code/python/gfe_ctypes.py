"""Minimal ctypes wrapper for gfe_c_api.h."""

from __future__ import annotations

import ctypes
import os
from csv import writer
from ctypes import POINTER, byref, c_char, c_char_p, c_double, c_int, c_size_t
from pathlib import Path
from typing import Dict, List, Sequence


GFE_C_STATUS_OK = 0
GFE_C_STATUS_INVALID_ARGUMENT = 1
GFE_C_STATUS_RUNTIME_ERROR = 2
GFE_C_STATUS_BUFFER_TOO_SMALL = 3

GFE_C_FIT_BACKEND_NNLS_PG = 0
GFE_C_FIT_BACKEND_PRONY = 1

GFE_C_COMPUTE_BACKEND_CPU_REF = 0
GFE_C_COMPUTE_BACKEND_CPU_BLAS = 1
GFE_C_COMPUTE_BACKEND_GPU_CUDA = 2
GFE_C_COMPUTE_BACKEND_GPU_HIP = 3
GFE_C_COMPUTE_BACKEND_GPU_SYCL = 4

GFE_C_COUPLING_FORM_A = 0
GFE_C_COUPLING_FORM_B = 1
GFE_C_COUPLING_FORM_C = 2

GFE_C_ABERSOE_SCENARIO_LINEAR = 0
GFE_C_ABERSOE_SCENARIO_LORENZ63 = 1
GFE_C_ABERSOE_SCENARIO_RESONANT1D = 2

GFE_C_HIERARCHICAL_SCENARIO_TWO_LEVEL_BIDIRECTIONAL = 0
GFE_C_HIERARCHICAL_SCENARIO_THREE_LEVEL_HOMEOSTATIC = 1

GFE_C_HIERARCHICAL_RELATION_BOTTOM_UP = 0
GFE_C_HIERARCHICAL_RELATION_TOP_DOWN = 1

GFE_C_HIERARCHICAL_WARNING_INFO = 0
GFE_C_HIERARCHICAL_WARNING_WARNING = 1
GFE_C_HIERARCHICAL_WARNING_ERROR = 2

GFE_C_HIERARCHICAL_WARNING_CODE_UNKNOWN = 0
GFE_C_HIERARCHICAL_WARNING_CODE_NON_FINITE_RUN = 1
GFE_C_HIERARCHICAL_WARNING_CODE_KERNEL_SHIFT_DETECTED = 2
GFE_C_HIERARCHICAL_WARNING_CODE_LARGE_KERNEL_SHIFT = 3
GFE_C_HIERARCHICAL_WARNING_CODE_BOTTOM_UP_MCAP_INVERSION = 4
GFE_C_HIERARCHICAL_WARNING_CODE_BOTTOM_UP_MSCALE_INVERSION = 5
GFE_C_HIERARCHICAL_WARNING_CODE_TOP_DOWN_TIMESCALE_SHIFT = 6
GFE_C_HIERARCHICAL_WARNING_CODE_TOP_DOWN_DEFF_SHIFT = 7

GFE_C_HIERARCHICAL_WARNING_METRIC_UNKNOWN = 0
GFE_C_HIERARCHICAL_WARNING_METRIC_ALL_FINITE = 1
GFE_C_HIERARCHICAL_WARNING_METRIC_KERNEL_SHIFT = 2
GFE_C_HIERARCHICAL_WARNING_METRIC_DELTA_MCAP = 3
GFE_C_HIERARCHICAL_WARNING_METRIC_DELTA_MSCALE = 4
GFE_C_HIERARCHICAL_WARNING_METRIC_DELTA_DEFF = 5

GFE_C_HIERARCHICAL_RENORM_DIAGNOSTIC_NONE = 0
GFE_C_HIERARCHICAL_RENORM_DIAGNOSTIC_FIT_INSTABILITY = 1
GFE_C_HIERARCHICAL_RENORM_DIAGNOSTIC_OVER_PRUNING = 2
GFE_C_HIERARCHICAL_RENORM_DIAGNOSTIC_PLAUSIBLE_CHANNEL_MERGING = 3


class GfeCSoeFitOptions(ctypes.Structure):
    _fields_ = [
        ("backend", c_int),
        ("n_basis", c_int),
        ("gamma_min", c_double),
        ("gamma_max", c_double),
        ("threshold", c_double),
        ("lr", c_double),
        ("iters", c_int),
        ("prony_order", c_int),
        ("prony_ridge", c_double),
        ("prony_root_imag_tol", c_double),
        ("prony_root_min", c_double),
        ("prony_root_max", c_double),
        ("prony_enforce_nonnegative_weights", c_int),
        ("prony_max_root_iters", c_int),
        ("prony_root_tol", c_double),
    ]


class GfeCBackendConfig(ctypes.Structure):
    _fields_ = [
        ("preferred", c_int),
        ("allow_fallback", c_int),
        ("fallback", c_int),
    ]


class GfeCSpectralUnitOptions(ctypes.Structure):
    _fields_ = [
        ("deff_policy", c_int),
        ("mcap_policy", c_int),
        ("mres_policy", c_int),
        ("mscale_zero_tol", c_double),
    ]


class GfeCSpectralUnits(ctypes.Structure):
    _fields_ = [
        ("mcap", c_double),
        ("mscale", c_double),
        ("mres", c_double),
        ("hmem", c_double),
        ("hnorm", c_double),
        ("deff", c_double),
    ]


class GfeCAbersoeDiagnostics(ctypes.Structure):
    _fields_ = [
        ("steps_executed", c_size_t),
        ("all_finite", c_int),
        ("max_abs_u", c_double),
        ("max_abs_chi", c_double),
        ("final_u_l2", c_double),
        ("final_chi_l2", c_double),
        ("mean_abs_u", c_double),
        ("mean_abs_chi", c_double),
    ]


class GfeCAbersoeConfigRecord(ctypes.Structure):
    _fields_ = [
        ("memory_channels", c_size_t),
        ("dt", c_double),
        ("coupling_index", c_size_t),
        ("steps", c_size_t),
        ("sample_stride", c_size_t),
        ("strict_finite", c_int),
        ("stochastic_forcing", c_int),
        ("forcing_noise_std", c_double),
        ("stochastic_channel", c_size_t),
        ("seed", c_size_t),
        ("fit_backend", c_int),
        ("hebbian_enabled", c_int),
        ("hebbian_rule", c_int),
        ("hebbian_learning_rate", c_double),
        ("hebbian_decay", c_double),
    ]


class GfeCAbersoeRuntimeConfig(ctypes.Structure):
    _fields_ = [
        ("steps", c_size_t),
        ("sample_stride", c_size_t),
        ("strict_finite", c_int),
        ("stochastic_forcing", c_int),
        ("forcing_noise_std", c_double),
        ("stochastic_channel", c_size_t),
        ("seed", c_size_t),
    ]


class GfeCHierarchicalRuntimeConfig(ctypes.Structure):
    _fields_ = [
        ("steps", c_size_t),
        ("sample_stride", c_size_t),
        ("strict_finite", c_int),
    ]


class GfeCMemoryKernelView(ctypes.Structure):
    _fields_ = [
        ("gamma", POINTER(c_double)),
        ("gamma_size", c_size_t),
        ("w", POINTER(c_double)),
        ("w_size", c_size_t),
    ]


class GfeCMemoryKernelMutView(ctypes.Structure):
    _fields_ = [
        ("gamma", POINTER(c_double)),
        ("gamma_capacity", c_size_t),
        ("gamma_size", POINTER(c_size_t)),
        ("w", POINTER(c_double)),
        ("w_capacity", c_size_t),
        ("w_size", POINTER(c_size_t)),
    ]


class GfeCStateView(ctypes.Structure):
    _fields_ = [
        ("u", POINTER(c_double)),
        ("u_size", c_size_t),
        ("chi", POINTER(c_double)),
        ("chi_size", c_size_t),
        ("t", c_double),
    ]


class GfeCStateMutView(ctypes.Structure):
    _fields_ = [
        ("u", POINTER(c_double)),
        ("u_capacity", c_size_t),
        ("u_size", POINTER(c_size_t)),
        ("chi", POINTER(c_double)),
        ("chi_capacity", c_size_t),
        ("chi_size", POINTER(c_size_t)),
        ("t", POINTER(c_double)),
    ]


class GfeCAbersoeSampleSummaryView(ctypes.Structure):
    _fields_ = [
        ("t", POINTER(c_double)),
        ("t_capacity", c_size_t),
        ("t_size", POINTER(c_size_t)),
        ("u0", POINTER(c_double)),
        ("u0_capacity", c_size_t),
        ("u0_size", POINTER(c_size_t)),
        ("chi0", POINTER(c_double)),
        ("chi0_capacity", c_size_t),
        ("chi0_size", POINTER(c_size_t)),
        ("u_l2", POINTER(c_double)),
        ("u_l2_capacity", c_size_t),
        ("u_l2_size", POINTER(c_size_t)),
        ("chi_l2", POINTER(c_double)),
        ("chi_l2_capacity", c_size_t),
        ("chi_l2_size", POINTER(c_size_t)),
    ]


class GfeCAbersoeTrajectoryView(ctypes.Structure):
    _fields_ = [
        ("t", POINTER(c_double)),
        ("t_capacity", c_size_t),
        ("t_size", POINTER(c_size_t)),
        ("u", POINTER(c_double)),
        ("u_capacity", c_size_t),
        ("u_size", POINTER(c_size_t)),
        ("u_dim", POINTER(c_size_t)),
        ("chi", POINTER(c_double)),
        ("chi_capacity", c_size_t),
        ("chi_size", POINTER(c_size_t)),
        ("chi_dim", POINTER(c_size_t)),
    ]


class GfeCHierarchicalRunDiagnostics(ctypes.Structure):
    _fields_ = [
        ("steps_executed", c_size_t),
        ("all_finite", c_int),
        ("level_count", c_size_t),
    ]


class GfeCHierarchicalCrossLevelContract(ctypes.Structure):
    _fields_ = [
        ("pass_", c_int),
        ("all_finite_pass", c_int),
        ("bottom_up_mcap_pass", c_int),
        ("bottom_up_mscale_pass", c_int),
        ("actionable_warning_count", c_size_t),
    ]


class GfeCHierarchicalChainLevelSpec(ctypes.Structure):
    _fields_ = [
        ("name", c_char_p),
        ("kernel", GfeCMemoryKernelView),
        ("initial_state", GfeCStateView),
        ("dt", c_double),
        ("linear_decay", POINTER(c_double)),
        ("linear_decay_size", c_size_t),
        ("forcing_bias", POINTER(c_double)),
        ("forcing_bias_size", c_size_t),
        ("form", c_int),
        ("coupling_index", c_size_t),
    ]


class GfeCHierarchicalChainEdgeSpec(ctypes.Structure):
    _fields_ = [
        ("source_level", c_size_t),
        ("target_level", c_size_t),
        ("relation", c_int),
        ("gain", c_double),
        ("normalize_weights", c_int),
    ]


class GfeCHierarchicalChainSpecView(ctypes.Structure):
    _fields_ = [
        ("levels", POINTER(GfeCHierarchicalChainLevelSpec)),
        ("level_count", c_size_t),
        ("edges", POINTER(GfeCHierarchicalChainEdgeSpec)),
        ("edge_count", c_size_t),
    ]


class GfeCHierarchicalLevelSummaryView(ctypes.Structure):
    _fields_ = [
        ("t", POINTER(c_double)),
        ("t_capacity", c_size_t),
        ("t_size", POINTER(c_size_t)),
        ("state_dim", POINTER(c_size_t)),
        ("state_dim_capacity", c_size_t),
        ("state_dim_size", POINTER(c_size_t)),
        ("memory_channels", POINTER(c_size_t)),
        ("memory_channels_capacity", c_size_t),
        ("memory_channels_size", POINTER(c_size_t)),
        ("u0", POINTER(c_double)),
        ("u0_capacity", c_size_t),
        ("u0_size", POINTER(c_size_t)),
        ("chi0", POINTER(c_double)),
        ("chi0_capacity", c_size_t),
        ("chi0_size", POINTER(c_size_t)),
        ("w0", POINTER(c_double)),
        ("w0_capacity", c_size_t),
        ("w0_size", POINTER(c_size_t)),
        ("gamma0", POINTER(c_double)),
        ("gamma0_capacity", c_size_t),
        ("gamma0_size", POINTER(c_size_t)),
        ("mcap", POINTER(c_double)),
        ("mcap_capacity", c_size_t),
        ("mcap_size", POINTER(c_size_t)),
        ("mscale", POINTER(c_double)),
        ("mscale_capacity", c_size_t),
        ("mscale_size", POINTER(c_size_t)),
        ("hmem", POINTER(c_double)),
        ("hmem_capacity", c_size_t),
        ("hmem_size", POINTER(c_size_t)),
        ("deff", POINTER(c_double)),
        ("deff_capacity", c_size_t),
        ("deff_size", POINTER(c_size_t)),
    ]


class GfeCHierarchicalKernelShiftView(ctypes.Structure):
    _fields_ = [
        ("level_idx", POINTER(c_size_t)),
        ("level_idx_capacity", c_size_t),
        ("level_idx_size", POINTER(c_size_t)),
        ("delta_mcap", POINTER(c_double)),
        ("delta_mcap_capacity", c_size_t),
        ("delta_mcap_size", POINTER(c_size_t)),
        ("delta_mscale", POINTER(c_double)),
        ("delta_mscale_capacity", c_size_t),
        ("delta_mscale_size", POINTER(c_size_t)),
        ("delta_hmem", POINTER(c_double)),
        ("delta_hmem_capacity", c_size_t),
        ("delta_hmem_size", POINTER(c_size_t)),
        ("delta_deff", POINTER(c_double)),
        ("delta_deff_capacity", c_size_t),
        ("delta_deff_size", POINTER(c_size_t)),
        ("changed", POINTER(c_int)),
        ("changed_capacity", c_size_t),
        ("changed_size", POINTER(c_size_t)),
    ]


class GfeCHierarchicalRelationView(ctypes.Structure):
    _fields_ = [
        ("relation_type", POINTER(c_int)),
        ("relation_type_capacity", c_size_t),
        ("relation_type_size", POINTER(c_size_t)),
        ("source_level", POINTER(c_size_t)),
        ("source_level_capacity", c_size_t),
        ("source_level_size", POINTER(c_size_t)),
        ("target_level", POINTER(c_size_t)),
        ("target_level_capacity", c_size_t),
        ("target_level_size", POINTER(c_size_t)),
        ("delta_mcap", POINTER(c_double)),
        ("delta_mcap_capacity", c_size_t),
        ("delta_mcap_size", POINTER(c_size_t)),
        ("delta_mscale", POINTER(c_double)),
        ("delta_mscale_capacity", c_size_t),
        ("delta_mscale_size", POINTER(c_size_t)),
        ("delta_hmem", POINTER(c_double)),
        ("delta_hmem_capacity", c_size_t),
        ("delta_hmem_size", POINTER(c_size_t)),
        ("delta_deff", POINTER(c_double)),
        ("delta_deff_capacity", c_size_t),
        ("delta_deff_size", POINTER(c_size_t)),
        ("target_mcap_not_lower", POINTER(c_int)),
        ("target_mcap_not_lower_capacity", c_size_t),
        ("target_mcap_not_lower_size", POINTER(c_size_t)),
        ("target_mscale_not_lower", POINTER(c_int)),
        ("target_mscale_not_lower_capacity", c_size_t),
        ("target_mscale_not_lower_size", POINTER(c_size_t)),
    ]


class GfeCHierarchicalWarningView(ctypes.Structure):
    _fields_ = [
        ("severity", POINTER(c_int)),
        ("severity_capacity", c_size_t),
        ("severity_size", POINTER(c_size_t)),
        ("code", POINTER(c_int)),
        ("code_capacity", c_size_t),
        ("code_size", POINTER(c_size_t)),
        ("metric", POINTER(c_int)),
        ("metric_capacity", c_size_t),
        ("metric_size", POINTER(c_size_t)),
        ("observed", POINTER(c_double)),
        ("observed_capacity", c_size_t),
        ("observed_size", POINTER(c_size_t)),
        ("threshold", POINTER(c_double)),
        ("threshold_capacity", c_size_t),
        ("threshold_size", POINTER(c_size_t)),
        ("source_level", POINTER(c_size_t)),
        ("source_level_capacity", c_size_t),
        ("source_level_size", POINTER(c_size_t)),
        ("target_level", POINTER(c_size_t)),
        ("target_level_capacity", c_size_t),
        ("target_level_size", POINTER(c_size_t)),
        ("actionable", POINTER(c_int)),
        ("actionable_capacity", c_size_t),
        ("actionable_size", POINTER(c_size_t)),
    ]


class GfeCHierarchicalRenormAnalysisView(ctypes.Structure):
    _fields_ = [
        ("input_level", POINTER(c_size_t)),
        ("input_level_capacity", c_size_t),
        ("input_level_size", POINTER(c_size_t)),
        ("bath_level", POINTER(c_size_t)),
        ("bath_level_capacity", c_size_t),
        ("bath_level_size", POINTER(c_size_t)),
        ("direct_level", POINTER(c_size_t)),
        ("direct_level_capacity", c_size_t),
        ("direct_level_size", POINTER(c_size_t)),
        ("primary_diagnostic", POINTER(c_int)),
        ("primary_diagnostic_capacity", c_size_t),
        ("primary_diagnostic_size", POINTER(c_size_t)),
        ("raw_output_channels", POINTER(c_size_t)),
        ("raw_output_channels_capacity", c_size_t),
        ("raw_output_channels_size", POINTER(c_size_t)),
        ("output_channels", POINTER(c_size_t)),
        ("output_channels_capacity", c_size_t),
        ("output_channels_size", POINTER(c_size_t)),
        ("pruned_channels", POINTER(c_size_t)),
        ("pruned_channels_capacity", c_size_t),
        ("pruned_channels_size", POINTER(c_size_t)),
        ("retained_weight_fraction", POINTER(c_double)),
        ("retained_weight_fraction_capacity", c_size_t),
        ("retained_weight_fraction_size", POINTER(c_size_t)),
        ("fit_l2_error", POINTER(c_double)),
        ("fit_l2_error_capacity", c_size_t),
        ("fit_l2_error_size", POINTER(c_size_t)),
        ("pruned_fit_l2_error", POINTER(c_double)),
        ("pruned_fit_l2_error_capacity", c_size_t),
        ("pruned_fit_l2_error_size", POINTER(c_size_t)),
        ("fit_linf_error", POINTER(c_double)),
        ("fit_linf_error_capacity", c_size_t),
        ("fit_linf_error_size", POINTER(c_size_t)),
        ("pruned_fit_linf_error", POINTER(c_double)),
        ("pruned_fit_linf_error_capacity", c_size_t),
        ("pruned_fit_linf_error_size", POINTER(c_size_t)),
        ("delta_mcap", POINTER(c_double)),
        ("delta_mcap_capacity", c_size_t),
        ("delta_mcap_size", POINTER(c_size_t)),
        ("delta_mscale", POINTER(c_double)),
        ("delta_mscale_capacity", c_size_t),
        ("delta_mscale_size", POINTER(c_size_t)),
        ("delta_deff", POINTER(c_double)),
        ("delta_deff_capacity", c_size_t),
        ("delta_deff_size", POINTER(c_size_t)),
        ("consistency_pass", POINTER(c_int)),
        ("consistency_pass_capacity", c_size_t),
        ("consistency_pass_size", POINTER(c_size_t)),
        ("mcap_pass", POINTER(c_int)),
        ("mcap_pass_capacity", c_size_t),
        ("mcap_pass_size", POINTER(c_size_t)),
        ("mscale_pass", POINTER(c_int)),
        ("mscale_pass_capacity", c_size_t),
        ("mscale_pass_size", POINTER(c_size_t)),
        ("deff_pass", POINTER(c_int)),
        ("deff_pass_capacity", c_size_t),
        ("deff_pass_size", POINTER(c_size_t)),
    ]


class GfeCStringResultView(ctypes.Structure):
    _fields_ = [
        ("data", POINTER(c_char)),
        ("capacity", c_size_t),
        ("size", POINTER(c_size_t)),
    ]


class GfeCFitResultView(ctypes.Structure):
    _fields_ = [
        ("gamma_fit", POINTER(c_double)),
        ("gamma_fit_capacity", c_size_t),
        ("gamma_fit_size", POINTER(c_size_t)),
        ("w_fit", POINTER(c_double)),
        ("w_fit_capacity", c_size_t),
        ("w_fit_size", POINTER(c_size_t)),
        ("fit", POINTER(c_double)),
        ("fit_capacity", c_size_t),
        ("fit_size", POINTER(c_size_t)),
    ]


def _default_library_path() -> Path:
    ext = {"linux": "so", "darwin": "dylib", "win32": "dll"}
    suffix = ext.get(os.sys.platform, "so")
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "code" / "c++ core" / "build-shared" / f"libgfe_core.{suffix}"


def load_gfe_library(path: str | None = None) -> ctypes.CDLL:
    lib_path = path or os.environ.get("GFE_CORE_LIB")
    if not lib_path:
        lib_path = str(_default_library_path())
    lib = ctypes.CDLL(lib_path)

    lib.gfe_c_default_soe_fit_options.argtypes = []
    lib.gfe_c_default_soe_fit_options.restype = GfeCSoeFitOptions

    lib.gfe_c_default_backend_config.argtypes = []
    lib.gfe_c_default_backend_config.restype = GfeCBackendConfig

    lib.gfe_c_default_spectral_unit_options.argtypes = []
    lib.gfe_c_default_spectral_unit_options.restype = GfeCSpectralUnitOptions

    lib.gfe_c_abersoe_default_runtime_config.argtypes = []
    lib.gfe_c_abersoe_default_runtime_config.restype = GfeCAbersoeRuntimeConfig

    lib.gfe_c_abersoe_scenario_count.argtypes = []
    lib.gfe_c_abersoe_scenario_count.restype = c_size_t

    lib.gfe_c_abersoe_scenario_name.argtypes = [
        c_int,
        POINTER(GfeCStringResultView),
        POINTER(c_char),
        c_size_t,
    ]
    lib.gfe_c_abersoe_scenario_name.restype = c_int

    lib.gfe_c_abersoe_run_scenario.argtypes = [
        c_int,
        c_double,
        c_int,
        POINTER(GfeCAbersoeRuntimeConfig),
        POINTER(GfeCStateMutView),
        POINTER(GfeCAbersoeDiagnostics),
        POINTER(GfeCAbersoeConfigRecord),
        POINTER(GfeCAbersoeSampleSummaryView),
        POINTER(c_char),
        c_size_t,
    ]
    lib.gfe_c_abersoe_run_scenario.restype = c_int

    lib.gfe_c_abersoe_run_scenario_with_overrides.argtypes = [
        c_int,
        c_double,
        c_int,
        POINTER(GfeCAbersoeRuntimeConfig),
        POINTER(GfeCStateView),
        POINTER(GfeCMemoryKernelView),
        POINTER(GfeCStateMutView),
        POINTER(GfeCMemoryKernelMutView),
        POINTER(GfeCAbersoeDiagnostics),
        POINTER(GfeCAbersoeConfigRecord),
        POINTER(GfeCAbersoeSampleSummaryView),
        POINTER(GfeCAbersoeTrajectoryView),
        POINTER(c_char),
        c_size_t,
    ]
    lib.gfe_c_abersoe_run_scenario_with_overrides.restype = c_int

    lib.gfe_c_abersoe_step.argtypes = [
        c_int,
        c_double,
        c_int,
        POINTER(GfeCMemoryKernelView),
        POINTER(GfeCStateMutView),
        POINTER(c_char),
        c_size_t,
    ]
    lib.gfe_c_abersoe_step.restype = c_int

    lib.gfe_c_hierarchical_scenario_count.argtypes = []
    lib.gfe_c_hierarchical_scenario_count.restype = c_size_t

    lib.gfe_c_hierarchical_scenario_name.argtypes = [
        c_int,
        POINTER(GfeCStringResultView),
        POINTER(c_char),
        c_size_t,
    ]
    lib.gfe_c_hierarchical_scenario_name.restype = c_int

    lib.gfe_c_hierarchical_run_scenario.argtypes = [
        c_int,
        c_int,
        c_int,
        POINTER(GfeCHierarchicalRuntimeConfig),
        POINTER(GfeCHierarchicalLevelSummaryView),
        POINTER(GfeCHierarchicalRunDiagnostics),
        POINTER(c_char),
        c_size_t,
    ]
    lib.gfe_c_hierarchical_run_scenario.restype = c_int

    lib.gfe_c_hierarchical_get_cross_level_report.argtypes = [
        c_int,
        c_int,
        c_int,
        POINTER(GfeCHierarchicalRuntimeConfig),
        POINTER(GfeCHierarchicalKernelShiftView),
        POINTER(GfeCHierarchicalRelationView),
        POINTER(GfeCHierarchicalWarningView),
        POINTER(GfeCHierarchicalCrossLevelContract),
        POINTER(c_char),
        c_size_t,
    ]
    lib.gfe_c_hierarchical_get_cross_level_report.restype = c_int

    lib.gfe_c_hierarchical_get_renorm_report.argtypes = [
        c_int,
        c_int,
        c_int,
        POINTER(GfeCHierarchicalRenormAnalysisView),
        POINTER(c_char),
        c_size_t,
    ]
    lib.gfe_c_hierarchical_get_renorm_report.restype = c_int

    lib.gfe_c_validate_hierarchical_chain_spec.argtypes = [
        POINTER(GfeCHierarchicalChainSpecView),
        POINTER(c_char),
        c_size_t,
    ]
    lib.gfe_c_validate_hierarchical_chain_spec.restype = c_int

    lib.gfe_c_hierarchical_run_chain_spec.argtypes = [
        POINTER(GfeCHierarchicalChainSpecView),
        POINTER(GfeCHierarchicalRuntimeConfig),
        POINTER(GfeCHierarchicalLevelSummaryView),
        POINTER(GfeCHierarchicalRunDiagnostics),
        POINTER(c_char),
        c_size_t,
    ]
    lib.gfe_c_hierarchical_run_chain_spec.restype = c_int

    lib.gfe_c_hierarchical_step_chain_spec.argtypes = [
        POINTER(GfeCHierarchicalChainSpecView),
        POINTER(GfeCStateMutView),
        c_size_t,
        POINTER(c_double),
        c_size_t,
        c_size_t,
        POINTER(GfeCMemoryKernelMutView),
        c_size_t,
        POINTER(GfeCSpectralUnits),
        c_size_t,
        POINTER(c_char),
        c_size_t,
    ]
    lib.gfe_c_hierarchical_step_chain_spec.restype = c_int

    lib.gfe_c_hierarchical_get_cross_level_report_for_chain_spec.argtypes = [
        POINTER(GfeCHierarchicalChainSpecView),
        POINTER(GfeCHierarchicalRuntimeConfig),
        POINTER(GfeCHierarchicalKernelShiftView),
        POINTER(GfeCHierarchicalRelationView),
        POINTER(GfeCHierarchicalWarningView),
        POINTER(GfeCHierarchicalCrossLevelContract),
        POINTER(c_char),
        c_size_t,
    ]
    lib.gfe_c_hierarchical_get_cross_level_report_for_chain_spec.restype = c_int

    lib.gfe_c_hierarchical_get_renorm_report_for_chain_spec.argtypes = [
        POINTER(GfeCHierarchicalChainSpecView),
        POINTER(GfeCHierarchicalRenormAnalysisView),
        POINTER(c_char),
        c_size_t,
    ]
    lib.gfe_c_hierarchical_get_renorm_report_for_chain_spec.restype = c_int

    lib.gfe_c_hierarchy_run_csv_header.argtypes = [
        POINTER(GfeCStringResultView),
        POINTER(c_char),
        c_size_t,
    ]
    lib.gfe_c_hierarchy_run_csv_header.restype = c_int

    lib.gfe_c_hierarchy_summary_csv_header.argtypes = [
        POINTER(GfeCStringResultView),
        POINTER(c_char),
        c_size_t,
    ]
    lib.gfe_c_hierarchy_summary_csv_header.restype = c_int

    lib.gfe_c_hierarchy_diagnostics_csv_header.argtypes = [
        POINTER(GfeCStringResultView),
        POINTER(c_char),
        c_size_t,
    ]
    lib.gfe_c_hierarchy_diagnostics_csv_header.restype = c_int

    lib.gfe_c_hierarchy_cross_level_csv_header.argtypes = [
        POINTER(GfeCStringResultView),
        POINTER(c_char),
        c_size_t,
    ]
    lib.gfe_c_hierarchy_cross_level_csv_header.restype = c_int

    lib.gfe_c_hierarchy_renorm_csv_header.argtypes = [
        POINTER(GfeCStringResultView),
        POINTER(c_char),
        c_size_t,
    ]
    lib.gfe_c_hierarchy_renorm_csv_header.restype = c_int

    lib.gfe_c_fit_soe_kernel.argtypes = [
        POINTER(c_double),
        c_size_t,
        POINTER(c_double),
        c_size_t,
        POINTER(GfeCSoeFitOptions),
        POINTER(GfeCBackendConfig),
        POINTER(GfeCFitResultView),
        POINTER(c_char),
        c_size_t,
    ]
    lib.gfe_c_fit_soe_kernel.restype = c_int
    return lib


def _call_string_result(func, *args: object) -> str:
    size = c_size_t(0)
    small = ctypes.create_string_buffer(8)
    view = GfeCStringResultView(small, ctypes.sizeof(small), ctypes.pointer(size))
    err = ctypes.create_string_buffer(512)
    status = func(*args, byref(view), err, ctypes.sizeof(err))
    if status == GFE_C_STATUS_BUFFER_TOO_SMALL and size.value > 0:
        buf = ctypes.create_string_buffer(size.value + 1)
        view = GfeCStringResultView(buf, ctypes.sizeof(buf), ctypes.pointer(c_size_t(0)))
        status = func(*args, byref(view), err, ctypes.sizeof(err))
        if status != GFE_C_STATUS_OK:
            raise RuntimeError(f"{func.__name__} failed ({status}): {err.value.decode('utf-8')}")
        return buf.value.decode("utf-8")
    if status != GFE_C_STATUS_OK:
        raise RuntimeError(f"{func.__name__} failed ({status}): {err.value.decode('utf-8')}")
    return small.value.decode("utf-8")


def fit_soe_kernel(lib: ctypes.CDLL, t: List[float], data: List[float]) -> Dict[str, List[float]]:
    if len(t) != len(data):
        raise ValueError("t and data must have the same length")
    n = len(t)
    if n == 0:
        raise ValueError("t and data must be non-empty")

    t_arr = (c_double * n)(*t)
    d_arr = (c_double * n)(*data)

    opts = lib.gfe_c_default_soe_fit_options()
    opts.backend = GFE_C_FIT_BACKEND_NNLS_PG
    opts.n_basis = 16
    opts.threshold = 0.0
    opts.iters = 5000

    backend = lib.gfe_c_default_backend_config()
    backend.preferred = GFE_C_COMPUTE_BACKEND_GPU_CUDA
    backend.allow_fallback = 1
    backend.fallback = GFE_C_COMPUTE_BACKEND_CPU_REF

    cap = max(64, n)
    gamma_buf = (c_double * cap)()
    w_buf = (c_double * cap)()
    fit_buf = (c_double * n)()
    gamma_size = c_size_t(0)
    w_size = c_size_t(0)
    fit_size = c_size_t(0)

    gamma_size_p = ctypes.pointer(gamma_size)
    w_size_p = ctypes.pointer(w_size)
    fit_size_p = ctypes.pointer(fit_size)

    out = GfeCFitResultView(
        gamma_buf,
        cap,
        gamma_size_p,
        w_buf,
        cap,
        w_size_p,
        fit_buf,
        n,
        fit_size_p,
    )
    err = ctypes.create_string_buffer(512)

    status = lib.gfe_c_fit_soe_kernel(
        t_arr,
        n,
        d_arr,
        n,
        byref(opts),
        byref(backend),
        byref(out),
        err,
        ctypes.sizeof(err),
    )
    if status != GFE_C_STATUS_OK:
        raise RuntimeError(f"gfe_c_fit_soe_kernel failed ({status}): {err.value.decode('utf-8')}")

    g = [gamma_buf[i] for i in range(gamma_size.value)]
    w = [w_buf[i] for i in range(w_size.value)]
    fit = [fit_buf[i] for i in range(fit_size.value)]
    return {"gamma_fit": g, "w_fit": w, "fit": fit}


def list_abersoe_scenarios(lib: ctypes.CDLL) -> List[str]:
    count = lib.gfe_c_abersoe_scenario_count()
    return [abersoe_scenario_name(lib, idx) for idx in range(count)]


def abersoe_scenario_name(lib: ctypes.CDLL, scenario: int) -> str:
    return _call_string_result(lib.gfe_c_abersoe_scenario_name, c_int(scenario))


def default_abersoe_runtime_config(lib: ctypes.CDLL) -> GfeCAbersoeRuntimeConfig:
    return lib.gfe_c_abersoe_default_runtime_config()


def list_hierarchical_scenarios(lib: ctypes.CDLL) -> List[str]:
    count = lib.gfe_c_hierarchical_scenario_count()
    return [hierarchical_scenario_name(lib, idx) for idx in range(count)]


def hierarchical_scenario_name(lib: ctypes.CDLL, scenario: int) -> str:
    return _call_string_result(lib.gfe_c_hierarchical_scenario_name, c_int(scenario))


def hierarchy_run_csv_header(lib: ctypes.CDLL) -> str:
    return _call_string_result(lib.gfe_c_hierarchy_run_csv_header)


def hierarchy_summary_csv_header(lib: ctypes.CDLL) -> str:
    return _call_string_result(lib.gfe_c_hierarchy_summary_csv_header)


def hierarchy_diagnostics_csv_header(lib: ctypes.CDLL) -> str:
    return _call_string_result(lib.gfe_c_hierarchy_diagnostics_csv_header)


def hierarchy_cross_level_csv_header(lib: ctypes.CDLL) -> str:
    return _call_string_result(lib.gfe_c_hierarchy_cross_level_csv_header)


def hierarchy_renorm_csv_header(lib: ctypes.CDLL) -> str:
    return _call_string_result(lib.gfe_c_hierarchy_renorm_csv_header)


def _write_csv(path: str, header: str, rows: Sequence[Sequence[object]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        csv_writer = writer(handle)
        csv_writer.writerow(header.split(","))
        for row in rows:
            csv_writer.writerow(row)


def write_hierarchical_summary_csv(
    lib: ctypes.CDLL,
    path: str,
    run_result: Dict[str, object],
    *,
    level_names: Sequence[str] | None = None,
) -> None:
    header = hierarchy_summary_csv_header(lib)
    levels = run_result["levels"]
    diagnostics = run_result["diagnostics"]
    rows = []
    for idx, level in enumerate(levels):
        name = level_names[idx] if level_names is not None and idx < len(level_names) else f"level_{idx}"
        rows.append(
            [
                idx,
                name,
                level["t"],
                level["u0"],
                level["chi0"],
                abs(level["u0"]),
                abs(level["chi0"]),
                level["w0"],
                level["gamma0"],
                level["mcap"],
                level["mscale"],
                level["hmem"],
                level["deff"],
                diagnostics["steps_executed"],
                1 if diagnostics["all_finite"] else 0,
            ]
        )
    _write_csv(path, header, rows)


def write_hierarchical_cross_level_report_csv(
    lib: ctypes.CDLL,
    path: str,
    report: Dict[str, object],
) -> None:
    header = hierarchy_cross_level_csv_header(lib)
    rows = []
    for relation in report["relations"]:
        rows.append(
            [
                "relation",
                "bottom_up" if relation["relation_type"] == GFE_C_HIERARCHICAL_RELATION_BOTTOM_UP else "top_down",
                relation["source_level"],
                relation["target_level"],
                f"level_{relation['source_level']}",
                f"level_{relation['target_level']}",
                relation["delta_mcap"],
                relation["delta_mscale"],
                relation["delta_hmem"],
                relation["delta_deff"],
                1 if relation["target_mcap_not_lower"] else 0,
                1 if relation["target_mscale_not_lower"] else 0,
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
            ]
        )
    for warning in report["warnings"]:
        rows.append(
            [
                "warning",
                "",
                warning["source_level"],
                warning["target_level"],
                f"level_{warning['source_level']}" if warning["source_level"] is not None else "",
                f"level_{warning['target_level']}" if warning["target_level"] is not None else "",
                "",
                "",
                "",
                "",
                "",
                "",
                warning["severity"],
                warning["code"],
                "",
                warning["metric"],
                warning["observed"],
                warning["threshold"],
                1 if warning["actionable"] else 0,
                "",
            ]
        )
    _write_csv(path, header, rows)


def write_hierarchical_renorm_report_csv(
    lib: ctypes.CDLL,
    path: str,
    analyses: Sequence[Dict[str, object]],
) -> None:
    header = hierarchy_renorm_csv_header(lib)
    rows = []
    for analysis in analyses:
        rows.append(
            [
                analysis["input_level"],
                analysis["bath_level"],
                analysis["direct_level"],
                f"level_{analysis['input_level']}",
                f"level_{analysis['bath_level']}",
                "" if analysis["direct_level"] >= 2**63 else f"level_{analysis['direct_level']}",
                analysis["primary_diagnostic"],
                analysis["raw_output_channels"],
                analysis["output_channels"],
                analysis["pruned_channels"],
                analysis["retained_weight_fraction"],
                analysis["fit_l2_error"],
                analysis["pruned_fit_l2_error"],
                analysis["fit_linf_error"],
                analysis["pruned_fit_linf_error"],
                analysis["delta_mcap"],
                analysis["delta_mscale"],
                analysis["delta_deff"],
                1 if analysis["consistency_pass"] else 0,
                1 if analysis["mcap_pass"] else 0,
                1 if analysis["mscale_pass"] else 0,
                1 if analysis["deff_pass"] else 0,
            ]
        )
    _write_csv(path, header, rows)


def write_hierarchical_export_bundle(
    lib: ctypes.CDLL,
    output_dir: str,
    bundle_name: str,
    run_result: Dict[str, object],
    cross_report: Dict[str, object],
    renorm_report: Sequence[Dict[str, object]],
    *,
    level_names: Sequence[str] | None = None,
) -> Dict[str, str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary_csv": str(out_dir / f"{bundle_name}_summary.csv"),
        "cross_level_csv": str(out_dir / f"{bundle_name}_cross_level.csv"),
        "renorm_csv": str(out_dir / f"{bundle_name}_renorm.csv"),
    }
    write_hierarchical_summary_csv(lib, paths["summary_csv"], run_result, level_names=level_names)
    write_hierarchical_cross_level_report_csv(lib, paths["cross_level_csv"], cross_report)
    write_hierarchical_renorm_report_csv(lib, paths["renorm_csv"], renorm_report)
    return paths


def write_hierarchical_chain_export_bundle(
    lib: ctypes.CDLL,
    output_dir: str,
    bundle_name: str,
    levels: List[Dict[str, object]],
    edges: List[Dict[str, object]],
    *,
    steps: int = 80,
    sample_stride: int = 10,
    strict_finite: bool = True,
) -> Dict[str, str]:
    run_result = run_hierarchical_chain_spec(
        lib,
        levels,
        edges,
        steps=steps,
        sample_stride=sample_stride,
        strict_finite=strict_finite,
    )
    cross_report = get_hierarchical_cross_level_report_for_chain_spec(
        lib,
        levels,
        edges,
        steps=steps,
        sample_stride=sample_stride,
        strict_finite=strict_finite,
    )
    renorm_report = get_hierarchical_renorm_report_for_chain_spec(lib, levels, edges)
    return write_hierarchical_export_bundle(
        lib,
        output_dir,
        bundle_name,
        run_result,
        cross_report,
        renorm_report,
        level_names=[str(level.get("name", f"level_{idx}")) for idx, level in enumerate(levels)],
    )


def _double_array(values: List[float]) -> ctypes.Array[c_double] | None:
    if not values:
        return None
    return (c_double * len(values))(*values)


def _build_state_view(
    u: Sequence[float],
    chi: Sequence[float] | None = None,
    *,
    t: float = 0.0,
) -> tuple[GfeCStateView, List[object]]:
    u_arr = _double_array([float(v) for v in u])
    chi_arr = _double_array([float(v) for v in (chi or [])])
    view = GfeCStateView(
        u_arr if u_arr is not None else POINTER(c_double)(),
        len(u),
        chi_arr if chi_arr is not None else POINTER(c_double)(),
        len(chi or []),
        float(t),
    )
    return view, [u_arr, chi_arr]


def _build_kernel_view(
    gamma: Sequence[float],
    w: Sequence[float],
) -> tuple[GfeCMemoryKernelView, List[object]]:
    gamma_arr = _double_array([float(v) for v in gamma])
    w_arr = _double_array([float(v) for v in w])
    view = GfeCMemoryKernelView(
        gamma_arr if gamma_arr is not None else POINTER(c_double)(),
        len(gamma),
        w_arr if w_arr is not None else POINTER(c_double)(),
        len(w),
    )
    return view, [gamma_arr, w_arr]


def _build_hierarchical_chain_spec(
    levels: List[Dict[str, object]],
    edges: List[Dict[str, object]],
) -> tuple[GfeCHierarchicalChainSpecView, List[object]]:
    keepalive: List[object] = []
    level_structs: List[GfeCHierarchicalChainLevelSpec] = []
    for idx, level in enumerate(levels):
        name_bytes = str(level.get("name", f"level_{idx}")).encode("utf-8")
        gamma = [float(v) for v in level["gamma"]]
        weights = [float(v) for v in level["w"]]
        u = [float(v) for v in level["u"]]
        chi = [float(v) for v in level.get("chi", [])]
        linear_decay = [float(v) for v in level.get("linear_decay", [])]
        forcing_bias = [float(v) for v in level.get("forcing_bias", [])]

        gamma_arr = _double_array(gamma)
        weights_arr = _double_array(weights)
        u_arr = _double_array(u)
        chi_arr = _double_array(chi)
        decay_arr = _double_array(linear_decay)
        force_arr = _double_array(forcing_bias)
        keepalive.extend([name_bytes, gamma_arr, weights_arr, u_arr, chi_arr, decay_arr, force_arr])

        kernel = GfeCMemoryKernelView(
            gamma_arr if gamma_arr is not None else POINTER(c_double)(),
            len(gamma),
            weights_arr if weights_arr is not None else POINTER(c_double)(),
            len(weights),
        )
        initial_state = GfeCStateView(
            u_arr if u_arr is not None else POINTER(c_double)(),
            len(u),
            chi_arr if chi_arr is not None else POINTER(c_double)(),
            len(chi),
            float(level.get("t", 0.0)),
        )
        level_structs.append(
            GfeCHierarchicalChainLevelSpec(
                name_bytes,
                kernel,
                initial_state,
                float(level.get("dt", 0.01)),
                decay_arr if decay_arr is not None else POINTER(c_double)(),
                len(linear_decay),
                force_arr if force_arr is not None else POINTER(c_double)(),
                len(forcing_bias),
                int(level.get("form", GFE_C_COUPLING_FORM_B)),
                int(level.get("coupling_index", 0)),
            )
        )

    edge_structs = [
        GfeCHierarchicalChainEdgeSpec(
            int(edge["source_level"]),
            int(edge["target_level"]),
            int(edge["relation"]),
            float(edge.get("gain", 1.0)),
            1 if edge.get("normalize_weights", True) else 0,
        )
        for edge in edges
    ]

    level_arr = (GfeCHierarchicalChainLevelSpec * len(level_structs))(*level_structs) if level_structs else None
    edge_arr = (GfeCHierarchicalChainEdgeSpec * len(edge_structs))(*edge_structs) if edge_structs else None
    keepalive.extend([level_arr, edge_arr])
    spec = GfeCHierarchicalChainSpecView(
        level_arr if level_arr is not None else POINTER(GfeCHierarchicalChainLevelSpec)(),
        len(level_structs),
        edge_arr if edge_arr is not None else POINTER(GfeCHierarchicalChainEdgeSpec)(),
        len(edge_structs),
    )
    return spec, keepalive


def run_abersoe_scenario(
    lib: ctypes.CDLL,
    scenario: int,
    *,
    dt: float = 0.01,
    form: int = GFE_C_COUPLING_FORM_B,
    steps: int | None = None,
    sample_stride: int | None = None,
    strict_finite: bool | None = None,
) -> Dict[str, object]:
    cfg = default_abersoe_runtime_config(lib)
    if steps is not None:
        cfg.steps = steps
    if sample_stride is not None:
        cfg.sample_stride = sample_stride
    if strict_finite is not None:
        cfg.strict_finite = 1 if strict_finite else 0

    final_u = (c_double * 16)()
    final_chi = (c_double * 16)()
    final_t = c_double(0.0)
    final_u_size = c_size_t(0)
    final_chi_size = c_size_t(0)
    final_state = GfeCStateMutView(
        final_u,
        16,
        ctypes.pointer(final_u_size),
        final_chi,
        16,
        ctypes.pointer(final_chi_size),
        ctypes.pointer(final_t),
    )

    n_samples_cap = max(4, (cfg.steps // max(1, cfg.sample_stride)) + 2)
    sample_t = (c_double * n_samples_cap)()
    sample_u0 = (c_double * n_samples_cap)()
    sample_chi0 = (c_double * n_samples_cap)()
    sample_u_l2 = (c_double * n_samples_cap)()
    sample_chi_l2 = (c_double * n_samples_cap)()
    sample_t_size = c_size_t(0)
    sample_u0_size = c_size_t(0)
    sample_chi0_size = c_size_t(0)
    sample_u_l2_size = c_size_t(0)
    sample_chi_l2_size = c_size_t(0)
    samples = GfeCAbersoeSampleSummaryView(
        sample_t,
        n_samples_cap,
        ctypes.pointer(sample_t_size),
        sample_u0,
        n_samples_cap,
        ctypes.pointer(sample_u0_size),
        sample_chi0,
        n_samples_cap,
        ctypes.pointer(sample_chi0_size),
        sample_u_l2,
        n_samples_cap,
        ctypes.pointer(sample_u_l2_size),
        sample_chi_l2,
        n_samples_cap,
        ctypes.pointer(sample_chi_l2_size),
    )

    diagnostics = GfeCAbersoeDiagnostics()
    config = GfeCAbersoeConfigRecord()
    err = ctypes.create_string_buffer(512)

    status = lib.gfe_c_abersoe_run_scenario(
        scenario,
        dt,
        form,
        byref(cfg),
        byref(final_state),
        byref(diagnostics),
        byref(config),
        byref(samples),
        err,
        ctypes.sizeof(err),
    )
    if status != GFE_C_STATUS_OK:
        raise RuntimeError(f"gfe_c_abersoe_run_scenario failed ({status}): {err.value.decode('utf-8')}")

    return {
        "final_state": {
            "u": [final_u[i] for i in range(final_u_size.value)],
            "chi": [final_chi[i] for i in range(final_chi_size.value)],
            "t": final_t.value,
        },
        "diagnostics": {
            "steps_executed": diagnostics.steps_executed,
            "all_finite": bool(diagnostics.all_finite),
            "max_abs_u": diagnostics.max_abs_u,
            "max_abs_chi": diagnostics.max_abs_chi,
            "final_u_l2": diagnostics.final_u_l2,
            "final_chi_l2": diagnostics.final_chi_l2,
            "mean_abs_u": diagnostics.mean_abs_u,
            "mean_abs_chi": diagnostics.mean_abs_chi,
        },
        "config": {
            "memory_channels": config.memory_channels,
            "dt": config.dt,
            "coupling_index": config.coupling_index,
            "steps": config.steps,
            "sample_stride": config.sample_stride,
            "strict_finite": bool(config.strict_finite),
            "stochastic_forcing": bool(config.stochastic_forcing),
            "forcing_noise_std": config.forcing_noise_std,
            "stochastic_channel": config.stochastic_channel,
            "seed": config.seed,
            "fit_backend": config.fit_backend,
            "hebbian_enabled": bool(config.hebbian_enabled),
            "hebbian_rule": config.hebbian_rule,
            "hebbian_learning_rate": config.hebbian_learning_rate,
            "hebbian_decay": config.hebbian_decay,
        },
        "samples": {
            "t": [sample_t[i] for i in range(sample_t_size.value)],
            "u0": [sample_u0[i] for i in range(sample_u0_size.value)],
            "chi0": [sample_chi0[i] for i in range(sample_chi0_size.value)],
            "u_l2": [sample_u_l2[i] for i in range(sample_u_l2_size.value)],
            "chi_l2": [sample_chi_l2[i] for i in range(sample_chi_l2_size.value)],
        },
    }


def run_abersoe_scenario_with_overrides(
    lib: ctypes.CDLL,
    scenario: int,
    *,
    dt: float = 0.01,
    form: int = GFE_C_COUPLING_FORM_B,
    initial_u: Sequence[float] | None = None,
    initial_chi: Sequence[float] | None = None,
    initial_t: float = 0.0,
    gamma: Sequence[float] | None = None,
    w: Sequence[float] | None = None,
    steps: int | None = None,
    sample_stride: int | None = None,
    strict_finite: bool | None = None,
) -> Dict[str, object]:
    cfg = default_abersoe_runtime_config(lib)
    if steps is not None:
        cfg.steps = steps
    if sample_stride is not None:
        cfg.sample_stride = sample_stride
    if strict_finite is not None:
        cfg.strict_finite = 1 if strict_finite else 0

    initial_state_ptr = None
    kernel_ptr = None
    keepalive: List[object] = []
    if initial_u is not None:
        initial_state, state_keepalive = _build_state_view(initial_u, initial_chi, t=initial_t)
        keepalive.extend(state_keepalive)
        initial_state_ptr = byref(initial_state)
    if (gamma is None) != (w is None):
        raise ValueError("gamma and w must be provided together")
    if gamma is not None and w is not None:
        kernel_view, kernel_keepalive = _build_kernel_view(gamma, w)
        keepalive.extend(kernel_keepalive)
        kernel_ptr = byref(kernel_view)

    final_u = (c_double * 32)()
    final_chi = (c_double * 32)()
    final_t = c_double(0.0)
    final_u_size = c_size_t(0)
    final_chi_size = c_size_t(0)
    final_state = GfeCStateMutView(
        final_u,
        32,
        ctypes.pointer(final_u_size),
        final_chi,
        32,
        ctypes.pointer(final_chi_size),
        ctypes.pointer(final_t),
    )

    active_gamma = (c_double * 32)()
    active_w = (c_double * 32)()
    active_gamma_size = c_size_t(0)
    active_w_size = c_size_t(0)
    active_kernel = GfeCMemoryKernelMutView(
        active_gamma,
        32,
        ctypes.pointer(active_gamma_size),
        active_w,
        32,
        ctypes.pointer(active_w_size),
    )

    n_samples_cap = max(4, (cfg.steps // max(1, cfg.sample_stride)) + 2)
    sample_t = (c_double * n_samples_cap)()
    sample_u0 = (c_double * n_samples_cap)()
    sample_chi0 = (c_double * n_samples_cap)()
    sample_u_l2 = (c_double * n_samples_cap)()
    sample_chi_l2 = (c_double * n_samples_cap)()
    sample_t_size = c_size_t(0)
    sample_u0_size = c_size_t(0)
    sample_chi0_size = c_size_t(0)
    sample_u_l2_size = c_size_t(0)
    sample_chi_l2_size = c_size_t(0)
    samples = GfeCAbersoeSampleSummaryView(
        sample_t,
        n_samples_cap,
        ctypes.pointer(sample_t_size),
        sample_u0,
        n_samples_cap,
        ctypes.pointer(sample_u0_size),
        sample_chi0,
        n_samples_cap,
        ctypes.pointer(sample_chi0_size),
        sample_u_l2,
        n_samples_cap,
        ctypes.pointer(sample_u_l2_size),
        sample_chi_l2,
        n_samples_cap,
        ctypes.pointer(sample_chi_l2_size),
    )

    traj_t = (c_double * n_samples_cap)()
    traj_u_cap = max(16, n_samples_cap * 16)
    traj_chi_cap = max(16, n_samples_cap * 32)
    traj_u = (c_double * traj_u_cap)()
    traj_chi = (c_double * traj_chi_cap)()
    traj_t_size = c_size_t(0)
    traj_u_size = c_size_t(0)
    traj_u_dim = c_size_t(0)
    traj_chi_size = c_size_t(0)
    traj_chi_dim = c_size_t(0)
    trajectory = GfeCAbersoeTrajectoryView(
        traj_t,
        n_samples_cap,
        ctypes.pointer(traj_t_size),
        traj_u,
        traj_u_cap,
        ctypes.pointer(traj_u_size),
        ctypes.pointer(traj_u_dim),
        traj_chi,
        traj_chi_cap,
        ctypes.pointer(traj_chi_size),
        ctypes.pointer(traj_chi_dim),
    )

    diagnostics = GfeCAbersoeDiagnostics()
    config = GfeCAbersoeConfigRecord()
    err = ctypes.create_string_buffer(512)

    status = lib.gfe_c_abersoe_run_scenario_with_overrides(
        scenario,
        dt,
        form,
        byref(cfg),
        initial_state_ptr,
        kernel_ptr,
        byref(final_state),
        byref(active_kernel),
        byref(diagnostics),
        byref(config),
        byref(samples),
        byref(trajectory),
        err,
        ctypes.sizeof(err),
    )
    if status != GFE_C_STATUS_OK:
        raise RuntimeError(
            f"gfe_c_abersoe_run_scenario_with_overrides failed ({status}): {err.value.decode('utf-8')}"
        )

    n_traj = traj_t_size.value
    u_dim = traj_u_dim.value
    chi_dim = traj_chi_dim.value
    return {
        "final_state": {
            "u": [final_u[i] for i in range(final_u_size.value)],
            "chi": [final_chi[i] for i in range(final_chi_size.value)],
            "t": final_t.value,
        },
        "active_kernel": {
            "gamma": [active_gamma[i] for i in range(active_gamma_size.value)],
            "w": [active_w[i] for i in range(active_w_size.value)],
        },
        "diagnostics": {
            "steps_executed": diagnostics.steps_executed,
            "all_finite": bool(diagnostics.all_finite),
            "max_abs_u": diagnostics.max_abs_u,
            "max_abs_chi": diagnostics.max_abs_chi,
            "final_u_l2": diagnostics.final_u_l2,
            "final_chi_l2": diagnostics.final_chi_l2,
            "mean_abs_u": diagnostics.mean_abs_u,
            "mean_abs_chi": diagnostics.mean_abs_chi,
        },
        "config": {
            "memory_channels": config.memory_channels,
            "dt": config.dt,
            "coupling_index": config.coupling_index,
            "steps": config.steps,
            "sample_stride": config.sample_stride,
            "strict_finite": bool(config.strict_finite),
            "stochastic_forcing": bool(config.stochastic_forcing),
            "forcing_noise_std": config.forcing_noise_std,
            "stochastic_channel": config.stochastic_channel,
            "seed": config.seed,
            "fit_backend": config.fit_backend,
            "hebbian_enabled": bool(config.hebbian_enabled),
            "hebbian_rule": config.hebbian_rule,
            "hebbian_learning_rate": config.hebbian_learning_rate,
            "hebbian_decay": config.hebbian_decay,
        },
        "samples": {
            "t": [sample_t[i] for i in range(sample_t_size.value)],
            "u0": [sample_u0[i] for i in range(sample_u0_size.value)],
            "chi0": [sample_chi0[i] for i in range(sample_chi0_size.value)],
            "u_l2": [sample_u_l2[i] for i in range(sample_u_l2_size.value)],
            "chi_l2": [sample_chi_l2[i] for i in range(sample_chi_l2_size.value)],
        },
        "trajectory": {
            "t": [traj_t[i] for i in range(n_traj)],
            "u_dim": u_dim,
            "chi_dim": chi_dim,
            "u": [
                [traj_u[i * u_dim + j] for j in range(u_dim)]
                for i in range(n_traj)
            ],
            "chi": [
                [traj_chi[i * chi_dim + j] for j in range(chi_dim)]
                for i in range(n_traj)
            ],
        },
    }


def step_abersoe(
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
) -> Dict[str, object]:
    """Single-step simulation for a given scenario with optional kernel overrides."""
    u_buf = (c_double * len(u))(*u)
    chi_buf = (c_double * len(chi))(*chi)
    t_val = c_double(t)
    u_size = c_size_t(len(u))
    chi_size = c_size_t(len(chi))

    state = GfeCStateMutView(
        u_buf,
        len(u),
        ctypes.pointer(u_size),
        chi_buf,
        len(chi),
        ctypes.pointer(chi_size),
        ctypes.pointer(t_val),
    )

    kernel_ptr = None
    if gamma is not None and w is not None:
        if len(gamma) != len(w):
            raise ValueError("gamma and w must have same length for kernel override")
        gamma_buf = (c_double * len(gamma))(*gamma)
        w_buf = (c_double * len(w))(*w)
        kernel = GfeCMemoryKernelView(gamma_buf, len(gamma), w_buf, len(w))
        kernel_ptr = ctypes.pointer(kernel)

    err = ctypes.create_string_buffer(512)
    status = lib.gfe_c_abersoe_step(
        scenario,
        dt,
        form,
        kernel_ptr,
        byref(state),
        err,
        ctypes.sizeof(err),
    )
    if status != GFE_C_STATUS_OK:
        raise RuntimeError(f"gfe_c_abersoe_step failed ({status}): {err.value.decode('utf-8')}")

    return {
        "u": [u_buf[i] for i in range(u_size.value)],
        "chi": [chi_buf[i] for i in range(chi_size.value)],
        "t": t_val.value,
    }


def run_hierarchical_scenario(
    lib: ctypes.CDLL,
    scenario: int,
    *,
    enable_bottom_up: bool = True,
    enable_top_down: bool = True,
    steps: int = 80,
    sample_stride: int = 10,
    strict_finite: bool = True,
) -> Dict[str, object]:
    cfg = GfeCHierarchicalRuntimeConfig(steps, sample_stride, 1 if strict_finite else 0)

    cap = 8
    t = (c_double * cap)()
    state_dim = (c_size_t * cap)()
    memory_channels = (c_size_t * cap)()
    u0 = (c_double * cap)()
    chi0 = (c_double * cap)()
    w0 = (c_double * cap)()
    gamma0 = (c_double * cap)()
    mcap = (c_double * cap)()
    mscale = (c_double * cap)()
    hmem = (c_double * cap)()
    deff = (c_double * cap)()
    t_size = c_size_t(0)
    state_dim_size = c_size_t(0)
    memory_channels_size = c_size_t(0)
    u0_size = c_size_t(0)
    chi0_size = c_size_t(0)
    w0_size = c_size_t(0)
    gamma0_size = c_size_t(0)
    mcap_size = c_size_t(0)
    mscale_size = c_size_t(0)
    hmem_size = c_size_t(0)
    deff_size = c_size_t(0)
    levels = GfeCHierarchicalLevelSummaryView(
        t, cap, ctypes.pointer(t_size),
        state_dim, cap, ctypes.pointer(state_dim_size),
        memory_channels, cap, ctypes.pointer(memory_channels_size),
        u0, cap, ctypes.pointer(u0_size),
        chi0, cap, ctypes.pointer(chi0_size),
        w0, cap, ctypes.pointer(w0_size),
        gamma0, cap, ctypes.pointer(gamma0_size),
        mcap, cap, ctypes.pointer(mcap_size),
        mscale, cap, ctypes.pointer(mscale_size),
        hmem, cap, ctypes.pointer(hmem_size),
        deff, cap, ctypes.pointer(deff_size),
    )
    diagnostics = GfeCHierarchicalRunDiagnostics()
    err = ctypes.create_string_buffer(512)
    status = lib.gfe_c_hierarchical_run_scenario(
        scenario,
        1 if enable_bottom_up else 0,
        1 if enable_top_down else 0,
        byref(cfg),
        byref(levels),
        byref(diagnostics),
        err,
        ctypes.sizeof(err),
    )
    if status != GFE_C_STATUS_OK:
        raise RuntimeError(f"gfe_c_hierarchical_run_scenario failed ({status}): {err.value.decode('utf-8')}")

    n = diagnostics.level_count
    return {
        "diagnostics": {
            "steps_executed": diagnostics.steps_executed,
            "all_finite": bool(diagnostics.all_finite),
            "level_count": diagnostics.level_count,
        },
        "levels": [
            {
                "t": t[i],
                "state_dim": state_dim[i],
                "memory_channels": memory_channels[i],
                "u0": u0[i],
                "chi0": chi0[i],
                "w0": w0[i],
                "gamma0": gamma0[i],
                "mcap": mcap[i],
                "mscale": mscale[i],
                "hmem": hmem[i],
                "deff": deff[i],
            }
            for i in range(n)
        ],
    }


def validate_hierarchical_chain_spec(
    lib: ctypes.CDLL,
    levels: List[Dict[str, object]],
    edges: List[Dict[str, object]],
) -> None:
    spec, keepalive = _build_hierarchical_chain_spec(levels, edges)
    _ = keepalive
    err = ctypes.create_string_buffer(512)
    status = lib.gfe_c_validate_hierarchical_chain_spec(byref(spec), err, ctypes.sizeof(err))
    if status != GFE_C_STATUS_OK:
        raise RuntimeError(f"gfe_c_validate_hierarchical_chain_spec failed ({status}): {err.value.decode('utf-8')}")


def run_hierarchical_chain_spec(
    lib: ctypes.CDLL,
    levels: List[Dict[str, object]],
    edges: List[Dict[str, object]],
    *,
    steps: int = 80,
    sample_stride: int = 10,
    strict_finite: bool = True,
) -> Dict[str, object]:
    spec, keepalive = _build_hierarchical_chain_spec(levels, edges)
    _ = keepalive
    cfg = GfeCHierarchicalRuntimeConfig(steps, sample_stride, 1 if strict_finite else 0)

    cap = max(4, len(levels) + 1)
    t = (c_double * cap)()
    state_dim = (c_size_t * cap)()
    memory_channels = (c_size_t * cap)()
    u0 = (c_double * cap)()
    chi0 = (c_double * cap)()
    w0 = (c_double * cap)()
    gamma0 = (c_double * cap)()
    mcap = (c_double * cap)()
    mscale = (c_double * cap)()
    hmem = (c_double * cap)()
    deff = (c_double * cap)()
    t_size = c_size_t(0)
    state_dim_size = c_size_t(0)
    memory_channels_size = c_size_t(0)
    u0_size = c_size_t(0)
    chi0_size = c_size_t(0)
    w0_size = c_size_t(0)
    gamma0_size = c_size_t(0)
    mcap_size = c_size_t(0)
    mscale_size = c_size_t(0)
    hmem_size = c_size_t(0)
    deff_size = c_size_t(0)
    level_view = GfeCHierarchicalLevelSummaryView(
        t, cap, ctypes.pointer(t_size),
        state_dim, cap, ctypes.pointer(state_dim_size),
        memory_channels, cap, ctypes.pointer(memory_channels_size),
        u0, cap, ctypes.pointer(u0_size),
        chi0, cap, ctypes.pointer(chi0_size),
        w0, cap, ctypes.pointer(w0_size),
        gamma0, cap, ctypes.pointer(gamma0_size),
        mcap, cap, ctypes.pointer(mcap_size),
        mscale, cap, ctypes.pointer(mscale_size),
        hmem, cap, ctypes.pointer(hmem_size),
        deff, cap, ctypes.pointer(deff_size),
    )
    diagnostics = GfeCHierarchicalRunDiagnostics()
    err = ctypes.create_string_buffer(512)
    status = lib.gfe_c_hierarchical_run_chain_spec(
        byref(spec),
        byref(cfg),
        byref(level_view),
        byref(diagnostics),
        err,
        ctypes.sizeof(err),
    )
    if status != GFE_C_STATUS_OK:
        raise RuntimeError(f"gfe_c_hierarchical_run_chain_spec failed ({status}): {err.value.decode('utf-8')}")

    n = diagnostics.level_count
    return {
        "diagnostics": {
            "steps_executed": diagnostics.steps_executed,
            "all_finite": bool(diagnostics.all_finite),
            "level_count": diagnostics.level_count,
        },
        "levels": [
            {
                "t": t[i],
                "state_dim": state_dim[i],
                "memory_channels": memory_channels[i],
                "u0": u0[i],
                "chi0": chi0[i],
                "w0": w0[i],
                "gamma0": gamma0[i],
                "mcap": mcap[i],
                "mscale": mscale[i],
                "hmem": hmem[i],
                "deff": deff[i],
            }
            for i in range(n)
        ],
    }


def get_hierarchical_cross_level_report_for_chain_spec(
    lib: ctypes.CDLL,
    levels: List[Dict[str, object]],
    edges: List[Dict[str, object]],
    *,
    steps: int = 80,
    sample_stride: int = 10,
    strict_finite: bool = True,
) -> Dict[str, object]:
    spec, keepalive = _build_hierarchical_chain_spec(levels, edges)
    _ = keepalive
    cfg = GfeCHierarchicalRuntimeConfig(steps, sample_stride, 1 if strict_finite else 0)
    cap = max(8, (len(levels) * 2) + len(edges) + 2)

    level_idx = (c_size_t * cap)()
    delta_mcap = (c_double * cap)()
    delta_mscale = (c_double * cap)()
    delta_hmem = (c_double * cap)()
    delta_deff = (c_double * cap)()
    changed = (c_int * cap)()
    level_idx_size = c_size_t(0)
    delta_mcap_size = c_size_t(0)
    delta_mscale_size = c_size_t(0)
    delta_hmem_size = c_size_t(0)
    delta_deff_size = c_size_t(0)
    changed_size = c_size_t(0)
    kernel_shifts = GfeCHierarchicalKernelShiftView(
        level_idx, cap, ctypes.pointer(level_idx_size),
        delta_mcap, cap, ctypes.pointer(delta_mcap_size),
        delta_mscale, cap, ctypes.pointer(delta_mscale_size),
        delta_hmem, cap, ctypes.pointer(delta_hmem_size),
        delta_deff, cap, ctypes.pointer(delta_deff_size),
        changed, cap, ctypes.pointer(changed_size),
    )

    relation_type = (c_int * cap)()
    source_level = (c_size_t * cap)()
    target_level = (c_size_t * cap)()
    relation_delta_mcap = (c_double * cap)()
    relation_delta_mscale = (c_double * cap)()
    relation_delta_hmem = (c_double * cap)()
    relation_delta_deff = (c_double * cap)()
    target_mcap_not_lower = (c_int * cap)()
    target_mscale_not_lower = (c_int * cap)()
    relation_type_size = c_size_t(0)
    source_level_size = c_size_t(0)
    target_level_size = c_size_t(0)
    relation_delta_mcap_size = c_size_t(0)
    relation_delta_mscale_size = c_size_t(0)
    relation_delta_hmem_size = c_size_t(0)
    relation_delta_deff_size = c_size_t(0)
    target_mcap_not_lower_size = c_size_t(0)
    target_mscale_not_lower_size = c_size_t(0)
    relations = GfeCHierarchicalRelationView(
        relation_type, cap, ctypes.pointer(relation_type_size),
        source_level, cap, ctypes.pointer(source_level_size),
        target_level, cap, ctypes.pointer(target_level_size),
        relation_delta_mcap, cap, ctypes.pointer(relation_delta_mcap_size),
        relation_delta_mscale, cap, ctypes.pointer(relation_delta_mscale_size),
        relation_delta_hmem, cap, ctypes.pointer(relation_delta_hmem_size),
        relation_delta_deff, cap, ctypes.pointer(relation_delta_deff_size),
        target_mcap_not_lower, cap, ctypes.pointer(target_mcap_not_lower_size),
        target_mscale_not_lower, cap, ctypes.pointer(target_mscale_not_lower_size),
    )

    severity = (c_int * cap)()
    code = (c_int * cap)()
    metric = (c_int * cap)()
    observed = (c_double * cap)()
    threshold = (c_double * cap)()
    warning_source_level = (c_size_t * cap)()
    warning_target_level = (c_size_t * cap)()
    actionable = (c_int * cap)()
    severity_size = c_size_t(0)
    code_size = c_size_t(0)
    metric_size = c_size_t(0)
    observed_size = c_size_t(0)
    threshold_size = c_size_t(0)
    warning_source_level_size = c_size_t(0)
    warning_target_level_size = c_size_t(0)
    actionable_size = c_size_t(0)
    warnings = GfeCHierarchicalWarningView(
        severity, cap, ctypes.pointer(severity_size),
        code, cap, ctypes.pointer(code_size),
        metric, cap, ctypes.pointer(metric_size),
        observed, cap, ctypes.pointer(observed_size),
        threshold, cap, ctypes.pointer(threshold_size),
        warning_source_level, cap, ctypes.pointer(warning_source_level_size),
        warning_target_level, cap, ctypes.pointer(warning_target_level_size),
        actionable, cap, ctypes.pointer(actionable_size),
    )

    contract = GfeCHierarchicalCrossLevelContract()
    err = ctypes.create_string_buffer(512)
    status = lib.gfe_c_hierarchical_get_cross_level_report_for_chain_spec(
        byref(spec),
        byref(cfg),
        byref(kernel_shifts),
        byref(relations),
        byref(warnings),
        byref(contract),
        err,
        ctypes.sizeof(err),
    )
    if status != GFE_C_STATUS_OK:
        raise RuntimeError(
            f"gfe_c_hierarchical_get_cross_level_report_for_chain_spec failed ({status}): {err.value.decode('utf-8')}"
        )

    n_shifts = level_idx_size.value
    n_rel = relation_type_size.value
    n_warn = severity_size.value
    return {
        "contract": {
            "pass": bool(contract.pass_),
            "all_finite_pass": bool(contract.all_finite_pass),
            "bottom_up_mcap_pass": bool(contract.bottom_up_mcap_pass),
            "bottom_up_mscale_pass": bool(contract.bottom_up_mscale_pass),
            "actionable_warning_count": contract.actionable_warning_count,
        },
        "kernel_shifts": [
            {
                "level_idx": level_idx[i],
                "delta_mcap": delta_mcap[i],
                "delta_mscale": delta_mscale[i],
                "delta_hmem": delta_hmem[i],
                "delta_deff": delta_deff[i],
                "changed": bool(changed[i]),
            }
            for i in range(n_shifts)
        ],
        "relations": [
            {
                "relation_type": relation_type[i],
                "source_level": source_level[i],
                "target_level": target_level[i],
                "delta_mcap": relation_delta_mcap[i],
                "delta_mscale": relation_delta_mscale[i],
                "delta_hmem": relation_delta_hmem[i],
                "delta_deff": relation_delta_deff[i],
                "target_mcap_not_lower": bool(target_mcap_not_lower[i]),
                "target_mscale_not_lower": bool(target_mscale_not_lower[i]),
            }
            for i in range(n_rel)
        ],
        "warnings": [
            {
                "severity": severity[i],
                "code": code[i],
                "metric": metric[i],
                "observed": observed[i],
                "threshold": threshold[i],
                "source_level": warning_source_level[i],
                "target_level": warning_target_level[i],
                "actionable": bool(actionable[i]),
            }
            for i in range(n_warn)
        ],
    }


def get_hierarchical_renorm_report_for_chain_spec(
    lib: ctypes.CDLL,
    levels: List[Dict[str, object]],
    edges: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    spec, keepalive = _build_hierarchical_chain_spec(levels, edges)
    _ = keepalive
    cap = max(4, len(levels) + 1)

    input_level = (c_size_t * cap)()
    bath_level = (c_size_t * cap)()
    direct_level = (c_size_t * cap)()
    primary_diagnostic = (c_int * cap)()
    raw_output_channels = (c_size_t * cap)()
    output_channels = (c_size_t * cap)()
    pruned_channels = (c_size_t * cap)()
    retained_weight_fraction = (c_double * cap)()
    fit_l2_error = (c_double * cap)()
    pruned_fit_l2_error = (c_double * cap)()
    fit_linf_error = (c_double * cap)()
    pruned_fit_linf_error = (c_double * cap)()
    delta_mcap = (c_double * cap)()
    delta_mscale = (c_double * cap)()
    delta_deff = (c_double * cap)()
    consistency_pass = (c_int * cap)()
    mcap_pass = (c_int * cap)()
    mscale_pass = (c_int * cap)()
    deff_pass = (c_int * cap)()

    input_level_size = c_size_t(0)
    bath_level_size = c_size_t(0)
    direct_level_size = c_size_t(0)
    primary_diagnostic_size = c_size_t(0)
    raw_output_channels_size = c_size_t(0)
    output_channels_size = c_size_t(0)
    pruned_channels_size = c_size_t(0)
    retained_weight_fraction_size = c_size_t(0)
    fit_l2_error_size = c_size_t(0)
    pruned_fit_l2_error_size = c_size_t(0)
    fit_linf_error_size = c_size_t(0)
    pruned_fit_linf_error_size = c_size_t(0)
    delta_mcap_size = c_size_t(0)
    delta_mscale_size = c_size_t(0)
    delta_deff_size = c_size_t(0)
    consistency_pass_size = c_size_t(0)
    mcap_pass_size = c_size_t(0)
    mscale_pass_size = c_size_t(0)
    deff_pass_size = c_size_t(0)

    analyses = GfeCHierarchicalRenormAnalysisView(
        input_level, cap, ctypes.pointer(input_level_size),
        bath_level, cap, ctypes.pointer(bath_level_size),
        direct_level, cap, ctypes.pointer(direct_level_size),
        primary_diagnostic, cap, ctypes.pointer(primary_diagnostic_size),
        raw_output_channels, cap, ctypes.pointer(raw_output_channels_size),
        output_channels, cap, ctypes.pointer(output_channels_size),
        pruned_channels, cap, ctypes.pointer(pruned_channels_size),
        retained_weight_fraction, cap, ctypes.pointer(retained_weight_fraction_size),
        fit_l2_error, cap, ctypes.pointer(fit_l2_error_size),
        pruned_fit_l2_error, cap, ctypes.pointer(pruned_fit_l2_error_size),
        fit_linf_error, cap, ctypes.pointer(fit_linf_error_size),
        pruned_fit_linf_error, cap, ctypes.pointer(pruned_fit_linf_error_size),
        delta_mcap, cap, ctypes.pointer(delta_mcap_size),
        delta_mscale, cap, ctypes.pointer(delta_mscale_size),
        delta_deff, cap, ctypes.pointer(delta_deff_size),
        consistency_pass, cap, ctypes.pointer(consistency_pass_size),
        mcap_pass, cap, ctypes.pointer(mcap_pass_size),
        mscale_pass, cap, ctypes.pointer(mscale_pass_size),
        deff_pass, cap, ctypes.pointer(deff_pass_size),
    )

    err = ctypes.create_string_buffer(512)
    status = lib.gfe_c_hierarchical_get_renorm_report_for_chain_spec(
        byref(spec),
        byref(analyses),
        err,
        ctypes.sizeof(err),
    )
    if status != GFE_C_STATUS_OK:
        raise RuntimeError(
            f"gfe_c_hierarchical_get_renorm_report_for_chain_spec failed ({status}): {err.value.decode('utf-8')}"
        )

    n = input_level_size.value
    return [
        {
            "input_level": input_level[i],
            "bath_level": bath_level[i],
            "direct_level": direct_level[i],
            "primary_diagnostic": primary_diagnostic[i],
            "raw_output_channels": raw_output_channels[i],
            "output_channels": output_channels[i],
            "pruned_channels": pruned_channels[i],
            "retained_weight_fraction": retained_weight_fraction[i],
            "fit_l2_error": fit_l2_error[i],
            "pruned_fit_l2_error": pruned_fit_l2_error[i],
            "fit_linf_error": fit_linf_error[i],
            "pruned_fit_linf_error": pruned_fit_linf_error[i],
            "delta_mcap": delta_mcap[i],
            "delta_mscale": delta_mscale[i],
            "delta_deff": delta_deff[i],
            "consistency_pass": bool(consistency_pass[i]),
            "mcap_pass": bool(mcap_pass[i]),
            "mscale_pass": bool(mscale_pass[i]),
            "deff_pass": bool(deff_pass[i]),
        }
        for i in range(n)
    ]


def get_hierarchical_cross_level_report(
    lib: ctypes.CDLL,
    scenario: int,
    *,
    enable_bottom_up: bool = True,
    enable_top_down: bool = True,
    steps: int = 80,
    sample_stride: int = 10,
    strict_finite: bool = True,
) -> Dict[str, object]:
    cfg = GfeCHierarchicalRuntimeConfig(steps, sample_stride, 1 if strict_finite else 0)
    cap = 16

    level_idx = (c_size_t * cap)()
    delta_mcap = (c_double * cap)()
    delta_mscale = (c_double * cap)()
    delta_hmem = (c_double * cap)()
    delta_deff = (c_double * cap)()
    changed = (c_int * cap)()
    level_idx_size = c_size_t(0)
    delta_mcap_size = c_size_t(0)
    delta_mscale_size = c_size_t(0)
    delta_hmem_size = c_size_t(0)
    delta_deff_size = c_size_t(0)
    changed_size = c_size_t(0)
    kernel_shifts = GfeCHierarchicalKernelShiftView(
        level_idx, cap, ctypes.pointer(level_idx_size),
        delta_mcap, cap, ctypes.pointer(delta_mcap_size),
        delta_mscale, cap, ctypes.pointer(delta_mscale_size),
        delta_hmem, cap, ctypes.pointer(delta_hmem_size),
        delta_deff, cap, ctypes.pointer(delta_deff_size),
        changed, cap, ctypes.pointer(changed_size),
    )

    relation_type = (c_int * cap)()
    source_level = (c_size_t * cap)()
    target_level = (c_size_t * cap)()
    relation_delta_mcap = (c_double * cap)()
    relation_delta_mscale = (c_double * cap)()
    relation_delta_hmem = (c_double * cap)()
    relation_delta_deff = (c_double * cap)()
    target_mcap_not_lower = (c_int * cap)()
    target_mscale_not_lower = (c_int * cap)()
    relation_type_size = c_size_t(0)
    source_level_size = c_size_t(0)
    target_level_size = c_size_t(0)
    relation_delta_mcap_size = c_size_t(0)
    relation_delta_mscale_size = c_size_t(0)
    relation_delta_hmem_size = c_size_t(0)
    relation_delta_deff_size = c_size_t(0)
    target_mcap_not_lower_size = c_size_t(0)
    target_mscale_not_lower_size = c_size_t(0)
    relations = GfeCHierarchicalRelationView(
        relation_type, cap, ctypes.pointer(relation_type_size),
        source_level, cap, ctypes.pointer(source_level_size),
        target_level, cap, ctypes.pointer(target_level_size),
        relation_delta_mcap, cap, ctypes.pointer(relation_delta_mcap_size),
        relation_delta_mscale, cap, ctypes.pointer(relation_delta_mscale_size),
        relation_delta_hmem, cap, ctypes.pointer(relation_delta_hmem_size),
        relation_delta_deff, cap, ctypes.pointer(relation_delta_deff_size),
        target_mcap_not_lower, cap, ctypes.pointer(target_mcap_not_lower_size),
        target_mscale_not_lower, cap, ctypes.pointer(target_mscale_not_lower_size),
    )

    severity = (c_int * cap)()
    code = (c_int * cap)()
    metric = (c_int * cap)()
    observed = (c_double * cap)()
    threshold = (c_double * cap)()
    warning_source_level = (c_size_t * cap)()
    warning_target_level = (c_size_t * cap)()
    actionable = (c_int * cap)()
    severity_size = c_size_t(0)
    code_size = c_size_t(0)
    metric_size = c_size_t(0)
    observed_size = c_size_t(0)
    threshold_size = c_size_t(0)
    warning_source_level_size = c_size_t(0)
    warning_target_level_size = c_size_t(0)
    actionable_size = c_size_t(0)
    warnings = GfeCHierarchicalWarningView(
        severity, cap, ctypes.pointer(severity_size),
        code, cap, ctypes.pointer(code_size),
        metric, cap, ctypes.pointer(metric_size),
        observed, cap, ctypes.pointer(observed_size),
        threshold, cap, ctypes.pointer(threshold_size),
        warning_source_level, cap, ctypes.pointer(warning_source_level_size),
        warning_target_level, cap, ctypes.pointer(warning_target_level_size),
        actionable, cap, ctypes.pointer(actionable_size),
    )

    contract = GfeCHierarchicalCrossLevelContract()
    err = ctypes.create_string_buffer(512)
    status = lib.gfe_c_hierarchical_get_cross_level_report(
        scenario,
        1 if enable_bottom_up else 0,
        1 if enable_top_down else 0,
        byref(cfg),
        byref(kernel_shifts),
        byref(relations),
        byref(warnings),
        byref(contract),
        err,
        ctypes.sizeof(err),
    )
    if status != GFE_C_STATUS_OK:
        raise RuntimeError(
            f"gfe_c_hierarchical_get_cross_level_report failed ({status}): {err.value.decode('utf-8')}"
        )

    n_shifts = level_idx_size.value
    n_rel = relation_type_size.value
    n_warn = severity_size.value
    return {
        "contract": {
            "pass": bool(contract.pass_),
            "all_finite_pass": bool(contract.all_finite_pass),
            "bottom_up_mcap_pass": bool(contract.bottom_up_mcap_pass),
            "bottom_up_mscale_pass": bool(contract.bottom_up_mscale_pass),
            "actionable_warning_count": contract.actionable_warning_count,
        },
        "kernel_shifts": [
            {
                "level_idx": level_idx[i],
                "delta_mcap": delta_mcap[i],
                "delta_mscale": delta_mscale[i],
                "delta_hmem": delta_hmem[i],
                "delta_deff": delta_deff[i],
                "changed": bool(changed[i]),
            }
            for i in range(n_shifts)
        ],
        "relations": [
            {
                "relation_type": relation_type[i],
                "source_level": source_level[i],
                "target_level": target_level[i],
                "delta_mcap": relation_delta_mcap[i],
                "delta_mscale": relation_delta_mscale[i],
                "delta_hmem": relation_delta_hmem[i],
                "delta_deff": relation_delta_deff[i],
                "target_mcap_not_lower": bool(target_mcap_not_lower[i]),
                "target_mscale_not_lower": bool(target_mscale_not_lower[i]),
            }
            for i in range(n_rel)
        ],
        "warnings": [
            {
                "severity": severity[i],
                "code": code[i],
                "metric": metric[i],
                "observed": observed[i],
                "threshold": threshold[i],
                "source_level": warning_source_level[i],
                "target_level": warning_target_level[i],
                "actionable": bool(actionable[i]),
            }
            for i in range(n_warn)
        ],
    }


def get_hierarchical_renorm_report(
    lib: ctypes.CDLL,
    scenario: int,
    *,
    enable_bottom_up: bool = True,
    enable_top_down: bool = True,
) -> List[Dict[str, object]]:
    cap = 16

    input_level = (c_size_t * cap)()
    bath_level = (c_size_t * cap)()
    direct_level = (c_size_t * cap)()
    primary_diagnostic = (c_int * cap)()
    raw_output_channels = (c_size_t * cap)()
    output_channels = (c_size_t * cap)()
    pruned_channels = (c_size_t * cap)()
    retained_weight_fraction = (c_double * cap)()
    fit_l2_error = (c_double * cap)()
    pruned_fit_l2_error = (c_double * cap)()
    fit_linf_error = (c_double * cap)()
    pruned_fit_linf_error = (c_double * cap)()
    delta_mcap = (c_double * cap)()
    delta_mscale = (c_double * cap)()
    delta_deff = (c_double * cap)()
    consistency_pass = (c_int * cap)()
    mcap_pass = (c_int * cap)()
    mscale_pass = (c_int * cap)()
    deff_pass = (c_int * cap)()

    input_level_size = c_size_t(0)
    bath_level_size = c_size_t(0)
    direct_level_size = c_size_t(0)
    primary_diagnostic_size = c_size_t(0)
    raw_output_channels_size = c_size_t(0)
    output_channels_size = c_size_t(0)
    pruned_channels_size = c_size_t(0)
    retained_weight_fraction_size = c_size_t(0)
    fit_l2_error_size = c_size_t(0)
    pruned_fit_l2_error_size = c_size_t(0)
    fit_linf_error_size = c_size_t(0)
    pruned_fit_linf_error_size = c_size_t(0)
    delta_mcap_size = c_size_t(0)
    delta_mscale_size = c_size_t(0)
    delta_deff_size = c_size_t(0)
    consistency_pass_size = c_size_t(0)
    mcap_pass_size = c_size_t(0)
    mscale_pass_size = c_size_t(0)
    deff_pass_size = c_size_t(0)

    analyses = GfeCHierarchicalRenormAnalysisView(
        input_level, cap, ctypes.pointer(input_level_size),
        bath_level, cap, ctypes.pointer(bath_level_size),
        direct_level, cap, ctypes.pointer(direct_level_size),
        primary_diagnostic, cap, ctypes.pointer(primary_diagnostic_size),
        raw_output_channels, cap, ctypes.pointer(raw_output_channels_size),
        output_channels, cap, ctypes.pointer(output_channels_size),
        pruned_channels, cap, ctypes.pointer(pruned_channels_size),
        retained_weight_fraction, cap, ctypes.pointer(retained_weight_fraction_size),
        fit_l2_error, cap, ctypes.pointer(fit_l2_error_size),
        pruned_fit_l2_error, cap, ctypes.pointer(pruned_fit_l2_error_size),
        fit_linf_error, cap, ctypes.pointer(fit_linf_error_size),
        pruned_fit_linf_error, cap, ctypes.pointer(pruned_fit_linf_error_size),
        delta_mcap, cap, ctypes.pointer(delta_mcap_size),
        delta_mscale, cap, ctypes.pointer(delta_mscale_size),
        delta_deff, cap, ctypes.pointer(delta_deff_size),
        consistency_pass, cap, ctypes.pointer(consistency_pass_size),
        mcap_pass, cap, ctypes.pointer(mcap_pass_size),
        mscale_pass, cap, ctypes.pointer(mscale_pass_size),
        deff_pass, cap, ctypes.pointer(deff_pass_size),
    )

    err = ctypes.create_string_buffer(512)
    status = lib.gfe_c_hierarchical_get_renorm_report(
        scenario,
        1 if enable_bottom_up else 0,
        1 if enable_top_down else 0,
        byref(analyses),
        err,
        ctypes.sizeof(err),
    )
    if status != GFE_C_STATUS_OK:
        raise RuntimeError(
            f"gfe_c_hierarchical_get_renorm_report failed ({status}): {err.value.decode('utf-8')}"
        )

    n = input_level_size.value
    return [
        {
            "input_level": input_level[i],
            "bath_level": bath_level[i],
            "direct_level": direct_level[i],
            "primary_diagnostic": primary_diagnostic[i],
            "raw_output_channels": raw_output_channels[i],
            "output_channels": output_channels[i],
            "pruned_channels": pruned_channels[i],
            "retained_weight_fraction": retained_weight_fraction[i],
            "fit_l2_error": fit_l2_error[i],
            "pruned_fit_l2_error": pruned_fit_l2_error[i],
            "fit_linf_error": fit_linf_error[i],
            "pruned_fit_linf_error": pruned_fit_linf_error[i],
            "delta_mcap": delta_mcap[i],
            "delta_mscale": delta_mscale[i],
            "delta_deff": delta_deff[i],
            "consistency_pass": bool(consistency_pass[i]),
            "mcap_pass": bool(mcap_pass[i]),
            "mscale_pass": bool(mscale_pass[i]),
            "deff_pass": bool(deff_pass[i]),
        }
        for i in range(n)
    ]


def step_hierarchical_chain_spec(
    lib: ctypes.CDLL,
    levels: List[Dict[str, object]],
    edges: List[Dict[str, object]],
    *,
    level_states: List[Dict[str, object]],
    external_forcing: Sequence[float],
    forcing_level: int = 0,
) -> Dict[str, object]:
    """Stateful single-step of the hierarchy with external forcing injection.

    Parameters
    ----------
    levels, edges : chain-spec definition (same format as run_hierarchical_chain_spec).
    level_states : list of ``{"u": [...], "chi": [...], "t": float}`` dicts,
        one per level, holding the current state.
    external_forcing : forcing vector injected at ``forcing_level``.
    forcing_level : which level receives the external forcing (default 0).

    Returns
    -------
    dict with keys ``"level_states"`` (updated states), ``"active_kernels"``,
    and ``"spectral_units"`` (per-level diagnostics).
    """
    spec, keepalive = _build_hierarchical_chain_spec(levels, edges)
    _ = keepalive

    n_levels = len(level_states)
    if n_levels != len(levels):
        raise ValueError("level_states length must match levels length")

    # Build mutable state views for each level.
    state_keepalive: List[object] = []
    state_structs: List[GfeCStateMutView] = []
    for st in level_states:
        u = [float(v) for v in st["u"]]
        chi = [float(v) for v in st.get("chi", [])]
        u_buf = (c_double * len(u))(*u)
        chi_buf = (c_double * len(chi))(*chi) if chi else (c_double * 1)()
        u_size = c_size_t(len(u))
        chi_size = c_size_t(len(chi))
        t_val = c_double(float(st.get("t", 0.0)))
        state_keepalive.extend([u_buf, chi_buf, u_size, chi_size, t_val])
        state_structs.append(GfeCStateMutView(
            u_buf, len(u), ctypes.pointer(u_size),
            chi_buf, len(chi), ctypes.pointer(chi_size),
            ctypes.pointer(t_val),
        ))
    state_arr = (GfeCStateMutView * n_levels)(*state_structs)

    # External forcing buffer.
    ext = (c_double * len(external_forcing))(*[float(v) for v in external_forcing])

    # Build mutable kernel views for output.
    kernel_keepalive: List[object] = []
    kernel_structs: List[GfeCMemoryKernelMutView] = []
    for lev in levels:
        n_ch = len(lev["gamma"])
        g_buf = (c_double * n_ch)()
        w_buf = (c_double * n_ch)()
        g_size = c_size_t(0)
        w_size = c_size_t(0)
        kernel_keepalive.extend([g_buf, w_buf, g_size, w_size])
        kernel_structs.append(GfeCMemoryKernelMutView(
            g_buf, n_ch, ctypes.pointer(g_size),
            w_buf, n_ch, ctypes.pointer(w_size),
        ))
    kernel_arr = (GfeCMemoryKernelMutView * n_levels)(*kernel_structs)

    # Spectral units output.
    spectral_arr = (GfeCSpectralUnits * n_levels)()

    err = ctypes.create_string_buffer(512)
    status = lib.gfe_c_hierarchical_step_chain_spec(
        byref(spec),
        state_arr,
        n_levels,
        ext,
        len(external_forcing),
        forcing_level,
        kernel_arr,
        n_levels,
        spectral_arr,
        n_levels,
        err,
        ctypes.sizeof(err),
    )
    if status != GFE_C_STATUS_OK:
        raise RuntimeError(
            f"gfe_c_hierarchical_step_chain_spec failed ({status}): "
            f"{err.value.decode('utf-8')}"
        )

    # Extract results.
    result_states = []
    for i, st in enumerate(level_states):
        sv = state_structs[i]
        result_states.append({
            "u": [sv.u[j] for j in range(len(st["u"]))],
            "chi": [sv.chi[j] for j in range(len(st.get("chi", [])))],
            "t": sv.t[0],
        })

    result_kernels = []
    for i, lev in enumerate(levels):
        n_ch = len(lev["gamma"])
        kv = kernel_structs[i]
        result_kernels.append({
            "gamma": [kv.gamma[j] for j in range(n_ch)],
            "w": [kv.w[j] for j in range(n_ch)],
        })

    result_spectral = []
    for i in range(n_levels):
        s = spectral_arr[i]
        result_spectral.append({
            "mcap": s.mcap,
            "mscale": s.mscale,
            "mres": s.mres,
            "hmem": s.hmem,
            "hnorm": s.hnorm,
            "deff": s.deff,
        })

    return {
        "level_states": result_states,
        "active_kernels": result_kernels,
        "spectral_units": result_spectral,
    }


if __name__ == "__main__":
    lib = load_gfe_library()
    t_vals = [0.1 * i for i in range(81)]
    y_vals = [0.65 * (2.718281828459045 ** (-0.8 * t)) + 0.35 * (2.718281828459045 ** (-0.12 * t)) for t in t_vals]
    result = fit_soe_kernel(lib, t_vals, y_vals)
    print("ctypes call ok")
    print(f"modes={len(result['gamma_fit'])} fit_len={len(result['fit'])}")
    print(f"abersoe_scenarios={list_abersoe_scenarios(lib)}")
