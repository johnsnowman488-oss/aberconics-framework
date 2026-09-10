#include "gfe/gfe_c_api.h"

#include "abersoe/abersoe_config.hpp"
#include "abersoe/abersoe_diagnostics.hpp"
#include "abersoe/abersoe_learning.hpp"
#include "abersoe/abersoe_scenarios.hpp"
#include "abersoe/hierarchical_min.hpp"
#include "abersoe/hierarchical_min_diagnostics.hpp"
#include "gfe/gfe_backend.hpp"
#include "gfe/gfe_core.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>

namespace {

void write_error(char* error_msg, std::size_t error_msg_capacity, const std::string& msg) {
    if (!error_msg || error_msg_capacity == 0) {
        return;
    }
    const std::size_t n = std::min(error_msg_capacity - 1, msg.size());
    std::memcpy(error_msg, msg.data(), n);
    error_msg[n] = '\0';
}

template <typename T>
void write_required_size(T* size_out, std::size_t size) {
    if (size_out) {
        *size_out = size;
    }
}

int write_string_result(const std::string& value, gfe_c_string_result_view* out) {
    if (!out) {
        return GFE_C_STATUS_INVALID_ARGUMENT;
    }
    write_required_size(out->size, value.size());
    if (!out->data) {
        return GFE_C_STATUS_INVALID_ARGUMENT;
    }
    if (out->capacity <= value.size()) {
        return GFE_C_STATUS_BUFFER_TOO_SMALL;
    }
    std::memcpy(out->data, value.data(), value.size());
    out->data[value.size()] = '\0';
    return GFE_C_STATUS_OK;
}

gfe::FitBackend to_fit_backend(int value) {
    switch (value) {
        case GFE_C_FIT_BACKEND_NNLS_PG:
            return gfe::FitBackend::NnlsProjectedGradient;
        case GFE_C_FIT_BACKEND_PRONY:
            return gfe::FitBackend::Prony;
        default:
            throw std::invalid_argument("Unknown C fit backend");
    }
}

gfe::ComputeBackend to_compute_backend(int value) {
    switch (value) {
        case GFE_C_COMPUTE_BACKEND_CPU_REF:
            return gfe::ComputeBackend::CpuReference;
        case GFE_C_COMPUTE_BACKEND_CPU_BLAS:
            return gfe::ComputeBackend::CpuBlas;
        case GFE_C_COMPUTE_BACKEND_GPU_CUDA:
            return gfe::ComputeBackend::GpuCuda;
        case GFE_C_COMPUTE_BACKEND_GPU_HIP:
            return gfe::ComputeBackend::GpuHip;
        case GFE_C_COMPUTE_BACKEND_GPU_SYCL:
            return gfe::ComputeBackend::GpuSycl;
        default:
            throw std::invalid_argument("Unknown C compute backend");
    }
}

gfe::CouplingForm to_coupling_form(int value) {
    switch (value) {
        case GFE_C_COUPLING_FORM_A:
            return gfe::CouplingForm::InputDriven;
        case GFE_C_COUPLING_FORM_B:
            return gfe::CouplingForm::NegativeFeedback;
        case GFE_C_COUPLING_FORM_C:
            return gfe::CouplingForm::ResonantSecondOrder;
        default:
            throw std::invalid_argument("Unknown C coupling form");
    }
}

abersoe::AberSOEScenario to_abersoe_scenario(int value) {
    switch (value) {
        case GFE_C_ABERSOE_SCENARIO_LINEAR:
            return abersoe::AberSOEScenario::LinearScalar;
        case GFE_C_ABERSOE_SCENARIO_LORENZ63:
            return abersoe::AberSOEScenario::Lorenz63;
        case GFE_C_ABERSOE_SCENARIO_RESONANT1D:
            return abersoe::AberSOEScenario::Resonant1D;
        default:
            throw std::invalid_argument("Unknown C ABERSOE scenario");
    }
}

abersoe::HierarchicalScenario to_hierarchical_scenario(int value) {
    switch (value) {
        case GFE_C_HIERARCHICAL_SCENARIO_TWO_LEVEL_BIDIRECTIONAL:
            return abersoe::HierarchicalScenario::TwoLevelBidirectional;
        case GFE_C_HIERARCHICAL_SCENARIO_THREE_LEVEL_HOMEOSTATIC:
            return abersoe::HierarchicalScenario::ThreeLevelHomeostatic;
        default:
            throw std::invalid_argument("Unknown C hierarchical scenario");
    }
}

int to_relation_type_id(const std::string& relation) {
    if (relation == "bottom_up") {
        return GFE_C_HIERARCHICAL_RELATION_BOTTOM_UP;
    }
    if (relation == "top_down") {
        return GFE_C_HIERARCHICAL_RELATION_TOP_DOWN;
    }
    return -1;
}

int to_warning_severity_id(const std::string& severity) {
    if (severity == "info") {
        return GFE_C_HIERARCHICAL_WARNING_INFO;
    }
    if (severity == "warning") {
        return GFE_C_HIERARCHICAL_WARNING_WARNING;
    }
    if (severity == "error") {
        return GFE_C_HIERARCHICAL_WARNING_ERROR;
    }
    return -1;
}

int to_warning_code_id(const std::string& code) {
    if (code == "non_finite_run") {
        return GFE_C_HIERARCHICAL_WARNING_CODE_NON_FINITE_RUN;
    }
    if (code == "kernel_shift_detected") {
        return GFE_C_HIERARCHICAL_WARNING_CODE_KERNEL_SHIFT_DETECTED;
    }
    if (code == "large_kernel_shift") {
        return GFE_C_HIERARCHICAL_WARNING_CODE_LARGE_KERNEL_SHIFT;
    }
    if (code == "bottom_up_mcap_inversion") {
        return GFE_C_HIERARCHICAL_WARNING_CODE_BOTTOM_UP_MCAP_INVERSION;
    }
    if (code == "bottom_up_mscale_inversion") {
        return GFE_C_HIERARCHICAL_WARNING_CODE_BOTTOM_UP_MSCALE_INVERSION;
    }
    if (code == "top_down_timescale_shift") {
        return GFE_C_HIERARCHICAL_WARNING_CODE_TOP_DOWN_TIMESCALE_SHIFT;
    }
    if (code == "top_down_deff_shift") {
        return GFE_C_HIERARCHICAL_WARNING_CODE_TOP_DOWN_DEFF_SHIFT;
    }
    return GFE_C_HIERARCHICAL_WARNING_CODE_UNKNOWN;
}

int to_warning_metric_id(const std::string& metric) {
    if (metric == "all_finite") {
        return GFE_C_HIERARCHICAL_WARNING_METRIC_ALL_FINITE;
    }
    if (metric == "kernel_shift") {
        return GFE_C_HIERARCHICAL_WARNING_METRIC_KERNEL_SHIFT;
    }
    if (metric == "delta_mcap") {
        return GFE_C_HIERARCHICAL_WARNING_METRIC_DELTA_MCAP;
    }
    if (metric == "delta_mscale") {
        return GFE_C_HIERARCHICAL_WARNING_METRIC_DELTA_MSCALE;
    }
    if (metric == "delta_deff") {
        return GFE_C_HIERARCHICAL_WARNING_METRIC_DELTA_DEFF;
    }
    return GFE_C_HIERARCHICAL_WARNING_METRIC_UNKNOWN;
}

int to_renorm_diagnostic_id(abersoe::HierarchicalRenormDiagnosticCause cause) {
    switch (cause) {
        case abersoe::HierarchicalRenormDiagnosticCause::None:
            return GFE_C_HIERARCHICAL_RENORM_DIAGNOSTIC_NONE;
        case abersoe::HierarchicalRenormDiagnosticCause::FitInstability:
            return GFE_C_HIERARCHICAL_RENORM_DIAGNOSTIC_FIT_INSTABILITY;
        case abersoe::HierarchicalRenormDiagnosticCause::OverPruning:
            return GFE_C_HIERARCHICAL_RENORM_DIAGNOSTIC_OVER_PRUNING;
        case abersoe::HierarchicalRenormDiagnosticCause::PlausibleChannelMerging:
            return GFE_C_HIERARCHICAL_RENORM_DIAGNOSTIC_PLAUSIBLE_CHANNEL_MERGING;
    }
    return GFE_C_HIERARCHICAL_RENORM_DIAGNOSTIC_NONE;
}

gfe::DeffPolicy to_deff_policy(int value) {
    switch (value) {
        case GFE_C_DEFF_EFFECTIVE_RANK_ONLY:
            return gfe::DeffPolicy::EffectiveRankOnly;
        case GFE_C_DEFF_PAPER_EFFECTIVE_DIMENSION:
            return gfe::DeffPolicy::PaperEffectiveDimension;
        default:
            throw std::invalid_argument("Unknown C Deff policy");
    }
}

gfe::McapPolicy to_mcap_policy(int value) {
    switch (value) {
        case GFE_C_MCAP_LEGACY_WEIGHTED_TIMESCALE:
            return gfe::McapPolicy::LegacyWeightedTimescale;
        case GFE_C_MCAP_PAPER_MEAN_DEPTH:
            return gfe::McapPolicy::PaperMeanDepth;
        default:
            throw std::invalid_argument("Unknown C Mcap policy");
    }
}

gfe::MresPolicy to_mres_policy(int value) {
    switch (value) {
        case GFE_C_MRES_RETURN_NAN:
            return gfe::MresPolicy::ReturnNaN;
        case GFE_C_MRES_RETURN_INFINITY:
            return gfe::MresPolicy::ReturnInfinity;
        case GFE_C_MRES_THROW_ON_ZERO_SCALE:
            return gfe::MresPolicy::ThrowOnZeroScale;
        default:
            throw std::invalid_argument("Unknown C Mres policy");
    }
}

gfe::SoeFitOptions to_cpp_options(const gfe_c_soe_fit_options& c) {
    gfe::SoeFitOptions out;
    out.backend = to_fit_backend(c.backend);
    out.n_basis = c.n_basis;
    out.gamma_min = c.gamma_min;
    out.gamma_max = c.gamma_max;
    out.threshold = c.threshold;
    out.lr = c.lr;
    out.iters = c.iters;
    out.prony.order = c.prony_order;
    out.prony.ridge = c.prony_ridge;
    out.prony.root_imag_tol = c.prony_root_imag_tol;
    out.prony.root_min = c.prony_root_min;
    out.prony.root_max = c.prony_root_max;
    out.prony.enforce_nonnegative_weights = (c.prony_enforce_nonnegative_weights != 0);
    out.prony.max_root_iters = c.prony_max_root_iters;
    out.prony.root_tol = c.prony_root_tol;
    return out;
}

gfe::BackendConfig to_cpp_backend(const gfe_c_backend_config& c) {
    gfe::BackendConfig out;
    out.preferred = to_compute_backend(c.preferred);
    out.allow_fallback = (c.allow_fallback != 0);
    out.fallback = to_compute_backend(c.fallback);
    return out;
}

gfe::SpectralUnitOptions to_cpp_spectral_options(const gfe_c_spectral_unit_options& c) {
    gfe::SpectralUnitOptions out;
    out.deff_policy = to_deff_policy(c.deff_policy);
    out.mcap_policy = to_mcap_policy(c.mcap_policy);
    out.mres_policy = to_mres_policy(c.mres_policy);
    out.mscale_zero_tol = c.mscale_zero_tol;
    return out;
}

gfe_c_spectral_units to_c_spectral_units(const gfe::SpectralUnits& s) {
    gfe_c_spectral_units out{};
    out.mcap = s.Mcap;
    out.mscale = s.Mscale;
    out.mres = s.Mres;
    out.hmem = s.Hmem;
    out.hnorm = s.Hnorm;
    out.deff = s.Deff;
    return out;
}

gfe::MemoryKernelParams to_cpp_kernel(const gfe_c_memory_kernel_view& kernel) {
    if (!kernel.gamma || !kernel.w) {
        throw std::invalid_argument("Kernel gamma/w pointers must not be null");
    }
    if (kernel.gamma_size == 0 || kernel.w_size == 0) {
        throw std::invalid_argument("Kernel gamma/w must be non-empty");
    }
    if (kernel.gamma_size != kernel.w_size) {
        throw std::invalid_argument("Kernel gamma/w size mismatch");
    }
    gfe::MemoryKernelParams out;
    out.gamma.assign(kernel.gamma, kernel.gamma + kernel.gamma_size);
    out.w.assign(kernel.w, kernel.w + kernel.w_size);
    return out;
}

abersoe::AberSOEState to_cpp_state(const gfe_c_state_view& state) {
    if (!state.u) {
        throw std::invalid_argument("State u pointer must not be null");
    }
    if (state.chi_size > 0 && !state.chi) {
        throw std::invalid_argument("State chi pointer must not be null when chi_size > 0");
    }
    abersoe::AberSOEState out;
    out.u.assign(state.u, state.u + state.u_size);
    if (state.chi && state.chi_size > 0) {
        out.chi.assign(state.chi, state.chi + state.chi_size);
    }
    out.t = state.t;
    return out;
}

abersoe::AberSOEState to_cpp_state(const gfe_c_state_mut_view& state) {
    if (!state.u || !state.u_size) {
        throw std::invalid_argument("State u/u_size pointer must not be null");
    }
    if (state.chi_capacity > 0 && (!state.chi || !state.chi_size)) {
        throw std::invalid_argument("State chi/chi_size pointer must not be null when chi_capacity > 0");
    }
    if (!state.t) {
        throw std::invalid_argument("State t pointer must not be null");
    }
    abersoe::AberSOEState out;
    out.u.assign(state.u, state.u + *state.u_size);
    if (state.chi && state.chi_size && *state.chi_size > 0) {
        out.chi.assign(state.chi, state.chi + *state.chi_size);
    }
    out.t = *state.t;
    return out;
}

abersoe::HierarchicalChainRelationType to_chain_relation_type(int value) {
    switch (value) {
        case GFE_C_HIERARCHICAL_RELATION_BOTTOM_UP:
            return abersoe::HierarchicalChainRelationType::BottomUp;
        case GFE_C_HIERARCHICAL_RELATION_TOP_DOWN:
            return abersoe::HierarchicalChainRelationType::TopDown;
        default:
            throw std::invalid_argument("Unknown C hierarchical chain relation type");
    }
}

abersoe::HierarchicalChainSpec to_cpp_hierarchical_chain_spec(const gfe_c_hierarchical_chain_spec_view& spec) {
    if (!spec.levels && spec.level_count > 0) {
        throw std::invalid_argument("Hierarchical chain levels pointer must not be null");
    }
    if (!spec.edges && spec.edge_count > 0) {
        throw std::invalid_argument("Hierarchical chain edges pointer must not be null");
    }

    abersoe::HierarchicalChainSpec out;
    out.levels.reserve(spec.level_count);
    out.edges.reserve(spec.edge_count);

    for (std::size_t i = 0; i < spec.level_count; ++i) {
        const auto& level = spec.levels[i];
        abersoe::HierarchicalChainLevelSpec cpp_level;
        cpp_level.name = level.name ? std::string(level.name) : ("level_" + std::to_string(i));
        cpp_level.kernel = to_cpp_kernel(level.kernel);
        cpp_level.initial_state = to_cpp_state(level.initial_state);
        cpp_level.dt = level.dt;
        if (level.linear_decay_size > 0) {
            if (!level.linear_decay) {
                throw std::invalid_argument("Hierarchical chain linear_decay pointer must not be null when size > 0");
            }
            cpp_level.linear_decay.assign(level.linear_decay, level.linear_decay + level.linear_decay_size);
        }
        if (level.forcing_bias_size > 0) {
            if (!level.forcing_bias) {
                throw std::invalid_argument("Hierarchical chain forcing_bias pointer must not be null when size > 0");
            }
            cpp_level.forcing_bias.assign(level.forcing_bias, level.forcing_bias + level.forcing_bias_size);
        }
        cpp_level.form = to_coupling_form(level.form);
        cpp_level.coupling_index = level.coupling_index;
        out.levels.push_back(cpp_level);
    }

    for (std::size_t i = 0; i < spec.edge_count; ++i) {
        const auto& edge = spec.edges[i];
        abersoe::HierarchicalChainEdgeSpec cpp_edge;
        cpp_edge.source_level = edge.source_level;
        cpp_edge.target_level = edge.target_level;
        cpp_edge.relation = to_chain_relation_type(edge.relation);
        cpp_edge.gain = edge.gain;
        cpp_edge.normalize_weights = (edge.normalize_weights != 0);
        out.edges.push_back(cpp_edge);
    }

    return out;
}

std::string fit_backend_name(int value) {
    switch (value) {
        case GFE_C_FIT_BACKEND_NNLS_PG:
            return "nnls_pg";
        case GFE_C_FIT_BACKEND_PRONY:
            return "prony";
        default:
            throw std::invalid_argument("Unknown C fit backend");
    }
}

std::string compute_backend_name(int value) {
    switch (value) {
        case GFE_C_COMPUTE_BACKEND_CPU_REF:
            return "cpu_ref";
        case GFE_C_COMPUTE_BACKEND_CPU_BLAS:
            return "cpu_blas";
        case GFE_C_COMPUTE_BACKEND_GPU_CUDA:
            return "gpu_cuda";
        case GFE_C_COMPUTE_BACKEND_GPU_HIP:
            return "gpu_hip";
        case GFE_C_COMPUTE_BACKEND_GPU_SYCL:
            return "gpu_sycl";
        default:
            throw std::invalid_argument("Unknown C compute backend");
    }
}

int to_fit_backend_id(const std::string& name) {
    if (name == "nnls_pg") {
        return GFE_C_FIT_BACKEND_NNLS_PG;
    }
    if (name == "prony") {
        return GFE_C_FIT_BACKEND_PRONY;
    }
    return -1;
}

int to_hebbian_rule_id(const std::string& name) {
    if (name == abersoe::hebbian_rule_name(abersoe::HebbianRule::OjaNormalized)) {
        return GFE_C_HEBBIAN_RULE_OJA_NORMALIZED;
    }
    if (name == abersoe::hebbian_rule_name(abersoe::HebbianRule::PlainHebbian)) {
        return GFE_C_HEBBIAN_RULE_PLAIN_HEBBIAN;
    }
    return GFE_C_HEBBIAN_RULE_UNKNOWN;
}

gfe_c_abersoe_diagnostics to_c_abersoe_diagnostics(const abersoe::AberSOEDiagnostics& d) {
    gfe_c_abersoe_diagnostics out{};
    out.steps_executed = d.steps_executed;
    out.all_finite = d.all_finite ? 1 : 0;
    out.max_abs_u = d.max_abs_u;
    out.max_abs_chi = d.max_abs_chi;
    out.final_u_l2 = d.final_u_l2;
    out.final_chi_l2 = d.final_chi_l2;
    out.mean_abs_u = d.mean_abs_u;
    out.mean_abs_chi = d.mean_abs_chi;
    return out;
}

gfe_c_abersoe_config_record to_c_abersoe_config(const abersoe::AberSOEConfigRecord& c) {
    gfe_c_abersoe_config_record out{};
    out.memory_channels = c.memory_channels;
    out.dt = c.dt;
    out.coupling_index = c.coupling_index;
    out.steps = c.steps;
    out.sample_stride = c.sample_stride;
    out.strict_finite = c.strict_finite ? 1 : 0;
    out.stochastic_forcing = c.stochastic_forcing ? 1 : 0;
    out.forcing_noise_std = c.forcing_noise_std;
    out.stochastic_channel = c.stochastic_channel;
    out.seed = c.seed;
    out.fit_backend = to_fit_backend_id(c.fit_backend);
    out.hebbian_enabled = c.hebbian_enabled ? 1 : 0;
    out.hebbian_rule = to_hebbian_rule_id(c.hebbian_rule);
    out.hebbian_learning_rate = c.hebbian_learning_rate;
    out.hebbian_decay = c.hebbian_decay;
    return out;
}

abersoe::AberSOERuntimeConfig to_cpp_abersoe_runtime_config(const gfe_c_abersoe_runtime_config& c) {
    abersoe::AberSOERuntimeConfig out;
    out.steps = c.steps;
    out.sample_stride = c.sample_stride;
    out.strict_finite = (c.strict_finite != 0);
    out.stochastic_forcing = (c.stochastic_forcing != 0);
    out.forcing_noise_std = c.forcing_noise_std;
    out.stochastic_channel = c.stochastic_channel;
    out.seed = c.seed;
    return out;
}

void write_state_sizes(gfe_c_state_mut_view& out, const abersoe::AberSOEState& state) {
    write_required_size(out.u_size, state.u.size());
    write_required_size(out.chi_size, state.chi.size());
}

bool can_copy_state(const gfe_c_state_mut_view& out, const abersoe::AberSOEState& state) {
    return out.u && out.chi && out.t && out.u_capacity >= state.u.size() && out.chi_capacity >= state.chi.size();
}

void copy_state(gfe_c_state_mut_view& out, const abersoe::AberSOEState& state) {
    std::copy(state.u.begin(), state.u.end(), out.u);
    std::copy(state.chi.begin(), state.chi.end(), out.chi);
    if (out.u_size) {
        *out.u_size = state.u.size();
    }
    if (out.chi_size) {
        *out.chi_size = state.chi.size();
    }
    *out.t = state.t;
}

void write_sample_sizes(gfe_c_abersoe_sample_summary_view& out, std::size_t n) {
    write_required_size(out.t_size, n);
    write_required_size(out.u0_size, n);
    write_required_size(out.chi0_size, n);
    write_required_size(out.u_l2_size, n);
    write_required_size(out.chi_l2_size, n);
}

bool can_copy_samples(const gfe_c_abersoe_sample_summary_view& out, std::size_t n) {
    return out.t && out.u0 && out.chi0 && out.u_l2 && out.chi_l2 &&
           out.t_capacity >= n && out.u0_capacity >= n && out.chi0_capacity >= n &&
           out.u_l2_capacity >= n && out.chi_l2_capacity >= n;
}

void copy_samples(gfe_c_abersoe_sample_summary_view& out, const abersoe::AberSOERunResult& result) {
    const std::size_t n = result.samples.size();
    for (std::size_t i = 0; i < n; ++i) {
        const auto& s = result.samples[i];
        out.t[i] = s.t;
        out.u0[i] = s.u.empty() ? 0.0 : s.u[0];
        out.chi0[i] = s.chi.empty() ? 0.0 : s.chi[0];
        double u_l2 = 0.0;
        for (double v : s.u) {
            u_l2 += v * v;
        }
        double chi_l2 = 0.0;
        for (double v : s.chi) {
            chi_l2 += v * v;
        }
        out.u_l2[i] = std::sqrt(u_l2);
        out.chi_l2[i] = std::sqrt(chi_l2);
    }
    if (out.t_size) {
        *out.t_size = n;
    }
    if (out.u0_size) {
        *out.u0_size = n;
    }
    if (out.chi0_size) {
        *out.chi0_size = n;
    }
    if (out.u_l2_size) {
        *out.u_l2_size = n;
    }
    if (out.chi_l2_size) {
        *out.chi_l2_size = n;
    }
}

void write_kernel_sizes(gfe_c_memory_kernel_mut_view& out, const gfe::MemoryKernelParams& kernel) {
    write_required_size(out.gamma_size, kernel.gamma.size());
    write_required_size(out.w_size, kernel.w.size());
}

bool can_copy_kernel(const gfe_c_memory_kernel_mut_view& out, const gfe::MemoryKernelParams& kernel) {
    return out.gamma && out.w &&
           out.gamma_capacity >= kernel.gamma.size() &&
           out.w_capacity >= kernel.w.size();
}

void copy_kernel(gfe_c_memory_kernel_mut_view& out, const gfe::MemoryKernelParams& kernel) {
    std::copy(kernel.gamma.begin(), kernel.gamma.end(), out.gamma);
    std::copy(kernel.w.begin(), kernel.w.end(), out.w);
    if (out.gamma_size) {
        *out.gamma_size = kernel.gamma.size();
    }
    if (out.w_size) {
        *out.w_size = kernel.w.size();
    }
}

void write_trajectory_sizes(gfe_c_abersoe_trajectory_view& out, const abersoe::AberSOERunResult& result) {
    const std::size_t n = result.samples.size();
    const std::size_t u_dim = n == 0 ? 0 : result.samples.front().u.size();
    const std::size_t chi_dim = n == 0 ? 0 : result.samples.front().chi.size();
    write_required_size(out.t_size, n);
    write_required_size(out.u_size, n * u_dim);
    write_required_size(out.chi_size, n * chi_dim);
    write_required_size(out.u_dim, u_dim);
    write_required_size(out.chi_dim, chi_dim);
}

bool can_copy_trajectory(const gfe_c_abersoe_trajectory_view& out, const abersoe::AberSOERunResult& result) {
    const std::size_t n = result.samples.size();
    const std::size_t u_dim = n == 0 ? 0 : result.samples.front().u.size();
    const std::size_t chi_dim = n == 0 ? 0 : result.samples.front().chi.size();
    return out.t && out.u && out.u_dim && out.chi && out.chi_dim &&
           out.t_capacity >= n &&
           out.u_capacity >= n * u_dim &&
           out.chi_capacity >= n * chi_dim;
}

void copy_trajectory(gfe_c_abersoe_trajectory_view& out, const abersoe::AberSOERunResult& result) {
    const std::size_t n = result.samples.size();
    const std::size_t u_dim = n == 0 ? 0 : result.samples.front().u.size();
    const std::size_t chi_dim = n == 0 ? 0 : result.samples.front().chi.size();
    for (std::size_t i = 0; i < n; ++i) {
        const auto& sample = result.samples[i];
        out.t[i] = sample.t;
        for (std::size_t j = 0; j < u_dim; ++j) {
            out.u[i * u_dim + j] = sample.u[j];
        }
        for (std::size_t j = 0; j < chi_dim; ++j) {
            out.chi[i * chi_dim + j] = sample.chi[j];
        }
    }
    if (out.t_size) {
        *out.t_size = n;
    }
    if (out.u_size) {
        *out.u_size = n * u_dim;
    }
    if (out.u_dim) {
        *out.u_dim = u_dim;
    }
    if (out.chi_size) {
        *out.chi_size = n * chi_dim;
    }
    if (out.chi_dim) {
        *out.chi_dim = chi_dim;
    }
}

void write_level_summary_sizes(gfe_c_hierarchical_level_summary_view& out, std::size_t n) {
    write_required_size(out.t_size, n);
    write_required_size(out.state_dim_size, n);
    write_required_size(out.memory_channels_size, n);
    write_required_size(out.u0_size, n);
    write_required_size(out.chi0_size, n);
    write_required_size(out.w0_size, n);
    write_required_size(out.gamma0_size, n);
    write_required_size(out.mcap_size, n);
    write_required_size(out.mscale_size, n);
    write_required_size(out.hmem_size, n);
    write_required_size(out.deff_size, n);
}

bool can_copy_level_summaries(const gfe_c_hierarchical_level_summary_view& out, std::size_t n) {
    return out.t && out.state_dim && out.memory_channels && out.u0 && out.chi0 && out.w0 &&
           out.gamma0 && out.mcap && out.mscale && out.hmem && out.deff &&
           out.t_capacity >= n && out.state_dim_capacity >= n && out.memory_channels_capacity >= n &&
           out.u0_capacity >= n && out.chi0_capacity >= n && out.w0_capacity >= n &&
           out.gamma0_capacity >= n && out.mcap_capacity >= n && out.mscale_capacity >= n &&
           out.hmem_capacity >= n && out.deff_capacity >= n;
}

void copy_level_summaries(gfe_c_hierarchical_level_summary_view& out, const abersoe::HierarchicalRunResult& result) {
    const std::size_t n = result.final_levels.size();
    for (std::size_t i = 0; i < n; ++i) {
        const auto& level = result.final_levels[i];
        out.t[i] = level.state.t;
        out.state_dim[i] = level.state.u.size();
        out.memory_channels[i] = level.state.chi.size();
        out.u0[i] = level.state.u.empty() ? 0.0 : level.state.u[0];
        out.chi0[i] = level.state.chi.empty() ? 0.0 : level.state.chi[0];
        out.w0[i] = level.active_kernel.w.empty() ? 0.0 : level.active_kernel.w[0];
        out.gamma0[i] = level.active_kernel.gamma.empty() ? 0.0 : level.active_kernel.gamma[0];
        out.mcap[i] = level.spectral.Mcap;
        out.mscale[i] = level.spectral.Mscale;
        out.hmem[i] = level.spectral.Hmem;
        out.deff[i] = level.spectral.Deff;
    }
    if (out.t_size) *out.t_size = n;
    if (out.state_dim_size) *out.state_dim_size = n;
    if (out.memory_channels_size) *out.memory_channels_size = n;
    if (out.u0_size) *out.u0_size = n;
    if (out.chi0_size) *out.chi0_size = n;
    if (out.w0_size) *out.w0_size = n;
    if (out.gamma0_size) *out.gamma0_size = n;
    if (out.mcap_size) *out.mcap_size = n;
    if (out.mscale_size) *out.mscale_size = n;
    if (out.hmem_size) *out.hmem_size = n;
    if (out.deff_size) *out.deff_size = n;
}

void write_kernel_shift_sizes(gfe_c_hierarchical_kernel_shift_view& out, std::size_t n) {
    write_required_size(out.level_idx_size, n);
    write_required_size(out.delta_mcap_size, n);
    write_required_size(out.delta_mscale_size, n);
    write_required_size(out.delta_hmem_size, n);
    write_required_size(out.delta_deff_size, n);
    write_required_size(out.changed_size, n);
}

bool can_copy_kernel_shifts(const gfe_c_hierarchical_kernel_shift_view& out, std::size_t n) {
    return out.level_idx && out.delta_mcap && out.delta_mscale && out.delta_hmem && out.delta_deff && out.changed &&
           out.level_idx_capacity >= n && out.delta_mcap_capacity >= n && out.delta_mscale_capacity >= n &&
           out.delta_hmem_capacity >= n && out.delta_deff_capacity >= n && out.changed_capacity >= n;
}

void copy_kernel_shifts(gfe_c_hierarchical_kernel_shift_view& out, const abersoe::HierarchicalCrossLevelReport& report) {
    const std::size_t n = report.kernel_shifts.size();
    for (std::size_t i = 0; i < n; ++i) {
        const auto& shift = report.kernel_shifts[i];
        out.level_idx[i] = shift.level_idx;
        out.delta_mcap[i] = shift.delta_mcap;
        out.delta_mscale[i] = shift.delta_mscale;
        out.delta_hmem[i] = shift.delta_hmem;
        out.delta_deff[i] = shift.delta_deff;
        out.changed[i] = shift.changed ? 1 : 0;
    }
    if (out.level_idx_size) *out.level_idx_size = n;
    if (out.delta_mcap_size) *out.delta_mcap_size = n;
    if (out.delta_mscale_size) *out.delta_mscale_size = n;
    if (out.delta_hmem_size) *out.delta_hmem_size = n;
    if (out.delta_deff_size) *out.delta_deff_size = n;
    if (out.changed_size) *out.changed_size = n;
}

void write_relation_sizes(gfe_c_hierarchical_relation_view& out, std::size_t n) {
    write_required_size(out.relation_type_size, n);
    write_required_size(out.source_level_size, n);
    write_required_size(out.target_level_size, n);
    write_required_size(out.delta_mcap_size, n);
    write_required_size(out.delta_mscale_size, n);
    write_required_size(out.delta_hmem_size, n);
    write_required_size(out.delta_deff_size, n);
    write_required_size(out.target_mcap_not_lower_size, n);
    write_required_size(out.target_mscale_not_lower_size, n);
}

bool can_copy_relations(const gfe_c_hierarchical_relation_view& out, std::size_t n) {
    return out.relation_type && out.source_level && out.target_level && out.delta_mcap && out.delta_mscale &&
           out.delta_hmem && out.delta_deff && out.target_mcap_not_lower && out.target_mscale_not_lower &&
           out.relation_type_capacity >= n && out.source_level_capacity >= n && out.target_level_capacity >= n &&
           out.delta_mcap_capacity >= n && out.delta_mscale_capacity >= n && out.delta_hmem_capacity >= n &&
           out.delta_deff_capacity >= n && out.target_mcap_not_lower_capacity >= n &&
           out.target_mscale_not_lower_capacity >= n;
}

void copy_relations(gfe_c_hierarchical_relation_view& out, const abersoe::HierarchicalCrossLevelReport& report) {
    const std::size_t n = report.relations.size();
    for (std::size_t i = 0; i < n; ++i) {
        const auto& rel = report.relations[i];
        out.relation_type[i] = to_relation_type_id(rel.relation);
        out.source_level[i] = rel.source_level;
        out.target_level[i] = rel.target_level;
        out.delta_mcap[i] = rel.delta_mcap;
        out.delta_mscale[i] = rel.delta_mscale;
        out.delta_hmem[i] = rel.delta_hmem;
        out.delta_deff[i] = rel.delta_deff;
        out.target_mcap_not_lower[i] = rel.target_mcap_not_lower ? 1 : 0;
        out.target_mscale_not_lower[i] = rel.target_mscale_not_lower ? 1 : 0;
    }
    if (out.relation_type_size) *out.relation_type_size = n;
    if (out.source_level_size) *out.source_level_size = n;
    if (out.target_level_size) *out.target_level_size = n;
    if (out.delta_mcap_size) *out.delta_mcap_size = n;
    if (out.delta_mscale_size) *out.delta_mscale_size = n;
    if (out.delta_hmem_size) *out.delta_hmem_size = n;
    if (out.delta_deff_size) *out.delta_deff_size = n;
    if (out.target_mcap_not_lower_size) *out.target_mcap_not_lower_size = n;
    if (out.target_mscale_not_lower_size) *out.target_mscale_not_lower_size = n;
}

void write_warning_sizes(gfe_c_hierarchical_warning_view& out, std::size_t n) {
    write_required_size(out.severity_size, n);
    write_required_size(out.code_size, n);
    write_required_size(out.metric_size, n);
    write_required_size(out.observed_size, n);
    write_required_size(out.threshold_size, n);
    write_required_size(out.source_level_size, n);
    write_required_size(out.target_level_size, n);
    write_required_size(out.actionable_size, n);
}

bool can_copy_warnings(const gfe_c_hierarchical_warning_view& out, std::size_t n) {
    return out.severity && out.code && out.metric && out.observed && out.threshold &&
           out.source_level && out.target_level && out.actionable &&
           out.severity_capacity >= n && out.code_capacity >= n && out.metric_capacity >= n &&
           out.observed_capacity >= n && out.threshold_capacity >= n && out.source_level_capacity >= n &&
           out.target_level_capacity >= n && out.actionable_capacity >= n;
}

void copy_warnings(gfe_c_hierarchical_warning_view& out, const abersoe::HierarchicalCrossLevelReport& report) {
    const std::size_t n = report.warnings.size();
    for (std::size_t i = 0; i < n; ++i) {
        const auto& warning = report.warnings[i];
        out.severity[i] = to_warning_severity_id(warning.severity);
        out.code[i] = to_warning_code_id(warning.code);
        out.metric[i] = to_warning_metric_id(warning.metric);
        out.observed[i] = warning.observed;
        out.threshold[i] = warning.threshold;
        out.source_level[i] = warning.source_level;
        out.target_level[i] = warning.target_level;
        out.actionable[i] = warning.actionable ? 1 : 0;
    }
    if (out.severity_size) *out.severity_size = n;
    if (out.code_size) *out.code_size = n;
    if (out.metric_size) *out.metric_size = n;
    if (out.observed_size) *out.observed_size = n;
    if (out.threshold_size) *out.threshold_size = n;
    if (out.source_level_size) *out.source_level_size = n;
    if (out.target_level_size) *out.target_level_size = n;
    if (out.actionable_size) *out.actionable_size = n;
}

void write_renorm_sizes(gfe_c_hierarchical_renorm_analysis_view& out, std::size_t n) {
    write_required_size(out.input_level_size, n);
    write_required_size(out.bath_level_size, n);
    write_required_size(out.direct_level_size, n);
    write_required_size(out.primary_diagnostic_size, n);
    write_required_size(out.raw_output_channels_size, n);
    write_required_size(out.output_channels_size, n);
    write_required_size(out.pruned_channels_size, n);
    write_required_size(out.retained_weight_fraction_size, n);
    write_required_size(out.fit_l2_error_size, n);
    write_required_size(out.pruned_fit_l2_error_size, n);
    write_required_size(out.fit_linf_error_size, n);
    write_required_size(out.pruned_fit_linf_error_size, n);
    write_required_size(out.delta_mcap_size, n);
    write_required_size(out.delta_mscale_size, n);
    write_required_size(out.delta_deff_size, n);
    write_required_size(out.consistency_pass_size, n);
    write_required_size(out.mcap_pass_size, n);
    write_required_size(out.mscale_pass_size, n);
    write_required_size(out.deff_pass_size, n);
}

bool can_copy_renorm(const gfe_c_hierarchical_renorm_analysis_view& out, std::size_t n) {
    return out.input_level && out.bath_level && out.direct_level && out.primary_diagnostic &&
           out.raw_output_channels && out.output_channels && out.pruned_channels &&
           out.retained_weight_fraction && out.fit_l2_error && out.pruned_fit_l2_error &&
           out.fit_linf_error && out.pruned_fit_linf_error && out.delta_mcap && out.delta_mscale &&
           out.delta_deff && out.consistency_pass && out.mcap_pass && out.mscale_pass && out.deff_pass &&
           out.input_level_capacity >= n && out.bath_level_capacity >= n && out.direct_level_capacity >= n &&
           out.primary_diagnostic_capacity >= n && out.raw_output_channels_capacity >= n &&
           out.output_channels_capacity >= n && out.pruned_channels_capacity >= n &&
           out.retained_weight_fraction_capacity >= n && out.fit_l2_error_capacity >= n &&
           out.pruned_fit_l2_error_capacity >= n && out.fit_linf_error_capacity >= n &&
           out.pruned_fit_linf_error_capacity >= n && out.delta_mcap_capacity >= n &&
           out.delta_mscale_capacity >= n && out.delta_deff_capacity >= n &&
           out.consistency_pass_capacity >= n && out.mcap_pass_capacity >= n &&
           out.mscale_pass_capacity >= n && out.deff_pass_capacity >= n;
}

void copy_renorm(gfe_c_hierarchical_renorm_analysis_view& out,
                 const std::vector<abersoe::HierarchicalAdjacentRenormAnalysis>& analyses) {
    const std::size_t n = analyses.size();
    for (std::size_t i = 0; i < n; ++i) {
        const auto& a = analyses[i];
        out.input_level[i] = a.input_level;
        out.bath_level[i] = a.bath_level;
        out.direct_level[i] = a.direct_level;
        out.primary_diagnostic[i] = to_renorm_diagnostic_id(a.report.primary_diagnostic);
        out.raw_output_channels[i] = a.report.raw_output_channels;
        out.output_channels[i] = a.report.output_channels;
        out.pruned_channels[i] = a.report.pruned_channels;
        out.retained_weight_fraction[i] = a.report.retained_weight_fraction;
        out.fit_l2_error[i] = a.report.fit_l2_error;
        out.pruned_fit_l2_error[i] = a.report.pruned_fit_l2_error;
        out.fit_linf_error[i] = a.report.fit_linf_error;
        out.pruned_fit_linf_error[i] = a.report.pruned_fit_linf_error;
        out.delta_mcap[i] = a.report.input_to_output_pruned_drift.delta_mcap;
        out.delta_mscale[i] = a.report.input_to_output_pruned_drift.delta_mscale;
        out.delta_deff[i] = a.report.input_to_output_pruned_drift.delta_deff;
        out.consistency_pass[i] = a.report.consistency.pass ? 1 : 0;
        out.mcap_pass[i] = a.report.consistency.mcap_nondecrease_pass ? 1 : 0;
        out.mscale_pass[i] = a.report.consistency.mscale_nondecrease_pass ? 1 : 0;
        out.deff_pass[i] = a.report.consistency.deff_bound_pass ? 1 : 0;
    }
    if (out.input_level_size) *out.input_level_size = n;
    if (out.bath_level_size) *out.bath_level_size = n;
    if (out.direct_level_size) *out.direct_level_size = n;
    if (out.primary_diagnostic_size) *out.primary_diagnostic_size = n;
    if (out.raw_output_channels_size) *out.raw_output_channels_size = n;
    if (out.output_channels_size) *out.output_channels_size = n;
    if (out.pruned_channels_size) *out.pruned_channels_size = n;
    if (out.retained_weight_fraction_size) *out.retained_weight_fraction_size = n;
    if (out.fit_l2_error_size) *out.fit_l2_error_size = n;
    if (out.pruned_fit_l2_error_size) *out.pruned_fit_l2_error_size = n;
    if (out.fit_linf_error_size) *out.fit_linf_error_size = n;
    if (out.pruned_fit_linf_error_size) *out.pruned_fit_linf_error_size = n;
    if (out.delta_mcap_size) *out.delta_mcap_size = n;
    if (out.delta_mscale_size) *out.delta_mscale_size = n;
    if (out.delta_deff_size) *out.delta_deff_size = n;
    if (out.consistency_pass_size) *out.consistency_pass_size = n;
    if (out.mcap_pass_size) *out.mcap_pass_size = n;
    if (out.mscale_pass_size) *out.mscale_pass_size = n;
    if (out.deff_pass_size) *out.deff_pass_size = n;
}

bool can_copy(const gfe_c_fit_result_view& out, const gfe::FitResult& fit) {
    return out.gamma_fit_capacity >= fit.gamma_fit.size() && out.w_fit_capacity >= fit.w_fit.size() &&
           out.fit_capacity >= fit.fit.size();
}

void copy_fit(gfe_c_fit_result_view& out, const gfe::FitResult& fit) {
    std::copy(fit.gamma_fit.begin(), fit.gamma_fit.end(), out.gamma_fit);
    std::copy(fit.w_fit.begin(), fit.w_fit.end(), out.w_fit);
    std::copy(fit.fit.begin(), fit.fit.end(), out.fit);
}

void write_sizes(gfe_c_fit_result_view& out, const gfe::FitResult& fit) {
    if (out.gamma_fit_size) {
        *out.gamma_fit_size = fit.gamma_fit.size();
    }
    if (out.w_fit_size) {
        *out.w_fit_size = fit.w_fit.size();
    }
    if (out.fit_size) {
        *out.fit_size = fit.fit.size();
    }
}

} // namespace

extern "C" {

gfe_c_soe_fit_options gfe_c_default_soe_fit_options(void) {
    const gfe::SoeFitOptions base;
    gfe_c_soe_fit_options out{};
    out.backend = GFE_C_FIT_BACKEND_NNLS_PG;
    out.n_basis = base.n_basis;
    out.gamma_min = base.gamma_min;
    out.gamma_max = base.gamma_max;
    out.threshold = base.threshold;
    out.lr = base.lr;
    out.iters = base.iters;
    out.prony_order = base.prony.order;
    out.prony_ridge = base.prony.ridge;
    out.prony_root_imag_tol = base.prony.root_imag_tol;
    out.prony_root_min = base.prony.root_min;
    out.prony_root_max = base.prony.root_max;
    out.prony_enforce_nonnegative_weights = base.prony.enforce_nonnegative_weights ? 1 : 0;
    out.prony_max_root_iters = base.prony.max_root_iters;
    out.prony_root_tol = base.prony.root_tol;
    return out;
}

gfe_c_backend_config gfe_c_default_backend_config(void) {
    const gfe::BackendConfig base;
    gfe_c_backend_config out{};
    out.preferred = static_cast<int>(base.preferred);
    out.allow_fallback = base.allow_fallback ? 1 : 0;
    out.fallback = static_cast<int>(base.fallback);
    return out;
}

gfe_c_spectral_unit_options gfe_c_default_spectral_unit_options(void) {
    const gfe::SpectralUnitOptions base;
    gfe_c_spectral_unit_options out{};
    out.deff_policy = static_cast<int>(base.deff_policy);
    out.mcap_policy = static_cast<int>(base.mcap_policy);
    out.mres_policy = static_cast<int>(base.mres_policy);
    out.mscale_zero_tol = base.mscale_zero_tol;
    return out;
}

gfe_c_abersoe_runtime_config gfe_c_abersoe_default_runtime_config(void) {
    const abersoe::AberSOERuntimeConfig base;
    gfe_c_abersoe_runtime_config out{};
    out.steps = base.steps;
    out.sample_stride = base.sample_stride;
    out.strict_finite = base.strict_finite ? 1 : 0;
    out.stochastic_forcing = base.stochastic_forcing ? 1 : 0;
    out.forcing_noise_std = base.forcing_noise_std;
    out.stochastic_channel = base.stochastic_channel;
    out.seed = base.seed;
    return out;
}

gfe_c_hierarchical_runtime_config gfe_c_hierarchical_default_runtime_config(void) {
    const abersoe::HierarchicalRuntimeConfig base;
    gfe_c_hierarchical_runtime_config out{};
    out.steps = base.steps;
    out.sample_stride = base.sample_stride;
    out.strict_finite = base.strict_finite ? 1 : 0;
    return out;
}

size_t gfe_c_abersoe_scenario_count(void) {
    return abersoe::list_scenarios().size();
}

int gfe_c_abersoe_scenario_name(int scenario,
                                gfe_c_string_result_view* out,
                                char* error_msg,
                                size_t error_msg_capacity) {
    try {
        const auto name = abersoe::scenario_name(to_abersoe_scenario(scenario));
        const int rc = write_string_result(name, out);
        if (rc == GFE_C_STATUS_BUFFER_TOO_SMALL) {
            write_error(error_msg, error_msg_capacity, "Output buffer too small");
        } else if (rc == GFE_C_STATUS_INVALID_ARGUMENT) {
            write_error(error_msg, error_msg_capacity, "Invalid string output buffer");
        }
        return rc;
    } catch (const std::invalid_argument& e) {
        write_error(error_msg, error_msg_capacity, e.what());
        return GFE_C_STATUS_INVALID_ARGUMENT;
    } catch (const std::exception& e) {
        write_error(error_msg, error_msg_capacity, e.what());
        return GFE_C_STATUS_RUNTIME_ERROR;
    } catch (...) {
        write_error(error_msg, error_msg_capacity, "Unknown C API error");
        return GFE_C_STATUS_RUNTIME_ERROR;
    }
}

int gfe_c_abersoe_run_scenario(int scenario,
                               double dt,
                               int form,
                               const gfe_c_abersoe_runtime_config* cfg,
                               gfe_c_state_mut_view* final_state,
                               gfe_c_abersoe_diagnostics* diagnostics,
                               gfe_c_abersoe_config_record* config,
                               gfe_c_abersoe_sample_summary_view* samples,
                               char* error_msg,
                               size_t error_msg_capacity) {
    try {
        if (!cfg || !final_state || !diagnostics || !config || !samples) {
            write_error(error_msg, error_msg_capacity, "Null pointer argument");
            return GFE_C_STATUS_INVALID_ARGUMENT;
        }
        const auto cpp_scenario = to_abersoe_scenario(scenario);
        const auto cpp_form = to_coupling_form(form);
        auto model = abersoe::make_scenario_model(cpp_scenario, dt, cpp_form);
        auto state0 = abersoe::make_scenario_initial_state(cpp_scenario);
        const auto cpp_cfg = to_cpp_abersoe_runtime_config(*cfg);
        const auto result = abersoe::run(model, state0, cpp_cfg);

        write_state_sizes(*final_state, result.final_state);
        write_sample_sizes(*samples, result.samples.size());
        if (!can_copy_state(*final_state, result.final_state) || !can_copy_samples(*samples, result.samples.size())) {
            write_error(error_msg, error_msg_capacity, "Output buffer too small");
            return GFE_C_STATUS_BUFFER_TOO_SMALL;
        }

        copy_state(*final_state, result.final_state);
        copy_samples(*samples, result);
        *diagnostics = to_c_abersoe_diagnostics(result.diagnostics);
        *config = to_c_abersoe_config(result.config);
        return GFE_C_STATUS_OK;
    } catch (const std::invalid_argument& e) {
        write_error(error_msg, error_msg_capacity, e.what());
        return GFE_C_STATUS_INVALID_ARGUMENT;
    } catch (const std::exception& e) {
        write_error(error_msg, error_msg_capacity, e.what());
        return GFE_C_STATUS_RUNTIME_ERROR;
    } catch (...) {
        write_error(error_msg, error_msg_capacity, "Unknown C API error");
        return GFE_C_STATUS_RUNTIME_ERROR;
    }
}

int gfe_c_abersoe_run_scenario_with_overrides(
    int scenario,
    double dt,
    int form,
    const gfe_c_abersoe_runtime_config* cfg,
    const gfe_c_state_view* initial_state_override,
    const gfe_c_memory_kernel_view* kernel_override,
    gfe_c_state_mut_view* final_state,
    gfe_c_memory_kernel_mut_view* active_kernel,
    gfe_c_abersoe_diagnostics* diagnostics,
    gfe_c_abersoe_config_record* config,
    gfe_c_abersoe_sample_summary_view* samples,
    gfe_c_abersoe_trajectory_view* trajectory,
    char* error_msg,
    size_t error_msg_capacity) {
    try {
        if (!cfg || !final_state || !active_kernel || !diagnostics || !config || !samples || !trajectory) {
            write_error(error_msg, error_msg_capacity, "Null pointer argument");
            return GFE_C_STATUS_INVALID_ARGUMENT;
        }

        const auto cpp_scenario = to_abersoe_scenario(scenario);
        const auto cpp_form = to_coupling_form(form);
        auto model = abersoe::make_scenario_model(cpp_scenario, dt, cpp_form);
        auto state0 = initial_state_override ? to_cpp_state(*initial_state_override)
                                             : abersoe::make_scenario_initial_state(cpp_scenario);
        if (kernel_override) {
            model.kernel = to_cpp_kernel(*kernel_override);
        }

        const auto cpp_cfg = to_cpp_abersoe_runtime_config(*cfg);
        const auto result = abersoe::run(model, state0, cpp_cfg);

        gfe::MemoryKernelParams active_cpp_kernel;
        active_cpp_kernel.gamma = model.kernel.gamma;
        active_cpp_kernel.w = result.final_kernel_w.empty() ? model.kernel.w : result.final_kernel_w;

        write_state_sizes(*final_state, result.final_state);
        write_kernel_sizes(*active_kernel, active_cpp_kernel);
        write_sample_sizes(*samples, result.samples.size());
        write_trajectory_sizes(*trajectory, result);
        if (!can_copy_state(*final_state, result.final_state) ||
            !can_copy_kernel(*active_kernel, active_cpp_kernel) ||
            !can_copy_samples(*samples, result.samples.size()) ||
            !can_copy_trajectory(*trajectory, result)) {
            write_error(error_msg, error_msg_capacity, "Output buffer too small");
            return GFE_C_STATUS_BUFFER_TOO_SMALL;
        }

        copy_state(*final_state, result.final_state);
        copy_kernel(*active_kernel, active_cpp_kernel);
        copy_samples(*samples, result);
        copy_trajectory(*trajectory, result);
        *diagnostics = to_c_abersoe_diagnostics(result.diagnostics);
        *config = to_c_abersoe_config(result.config);
        return GFE_C_STATUS_OK;
    } catch (const std::invalid_argument& e) {
        write_error(error_msg, error_msg_capacity, e.what());
        return GFE_C_STATUS_INVALID_ARGUMENT;
    } catch (const std::exception& e) {
        write_error(error_msg, error_msg_capacity, e.what());
        return GFE_C_STATUS_RUNTIME_ERROR;
    } catch (...) {
        write_error(error_msg, error_msg_capacity, "Unknown C API error");
        return GFE_C_STATUS_RUNTIME_ERROR;
    }
}

int gfe_c_abersoe_step(
    int scenario,
    double dt,
    int form,
    const gfe_c_memory_kernel_view* kernel,
    gfe_c_state_mut_view* state,
    char* error_msg,
    size_t error_msg_capacity) {
    try {
        if (!state) {
            write_error(error_msg, error_msg_capacity, "Null pointer argument");
            return GFE_C_STATUS_INVALID_ARGUMENT;
        }

        const auto cpp_scenario = to_abersoe_scenario(scenario);
        const auto cpp_form = to_coupling_form(form);
        auto model = abersoe::make_scenario_model(cpp_scenario, dt, cpp_form);
        if (kernel) {
            model.kernel = to_cpp_kernel(*kernel);
        }

        auto current_state = to_cpp_state(*state);
        const auto next_state = abersoe::step(model, current_state);

        write_state_sizes(*state, next_state);
        if (!can_copy_state(*state, next_state)) {
            write_error(error_msg, error_msg_capacity, "Output buffer too small");
            return GFE_C_STATUS_BUFFER_TOO_SMALL;
        }

        copy_state(*state, next_state);
        return GFE_C_STATUS_OK;
    } catch (const std::invalid_argument& e) {
        write_error(error_msg, error_msg_capacity, e.what());
        return GFE_C_STATUS_INVALID_ARGUMENT;
    } catch (const std::exception& e) {
        write_error(error_msg, error_msg_capacity, e.what());
        return GFE_C_STATUS_RUNTIME_ERROR;
    } catch (...) {
        write_error(error_msg, error_msg_capacity, "Unknown C API error");
        return GFE_C_STATUS_RUNTIME_ERROR;
    }
}

size_t gfe_c_hierarchical_scenario_count(void) {
    return abersoe::list_hierarchical_scenarios().size();
}

int gfe_c_hierarchical_scenario_name(int scenario,
                                     gfe_c_string_result_view* out,
                                     char* error_msg,
                                     size_t error_msg_capacity) {
    try {
        const auto name = abersoe::hierarchical_scenario_name(to_hierarchical_scenario(scenario));
        const int rc = write_string_result(name, out);
        if (rc == GFE_C_STATUS_BUFFER_TOO_SMALL) {
            write_error(error_msg, error_msg_capacity, "Output buffer too small");
        } else if (rc == GFE_C_STATUS_INVALID_ARGUMENT) {
            write_error(error_msg, error_msg_capacity, "Invalid string output buffer");
        }
        return rc;
    } catch (const std::invalid_argument& e) {
        write_error(error_msg, error_msg_capacity, e.what());
        return GFE_C_STATUS_INVALID_ARGUMENT;
    } catch (const std::exception& e) {
        write_error(error_msg, error_msg_capacity, e.what());
        return GFE_C_STATUS_RUNTIME_ERROR;
    } catch (...) {
        write_error(error_msg, error_msg_capacity, "Unknown C API error");
        return GFE_C_STATUS_RUNTIME_ERROR;
    }
}

int gfe_c_hierarchical_run_scenario(int scenario,
                                    int enable_bottom_up,
                                    int enable_top_down,
                                    const gfe_c_hierarchical_runtime_config* cfg,
                                    gfe_c_hierarchical_level_summary_view* levels,
                                    gfe_c_hierarchical_run_diagnostics* diagnostics,
                                    char* error_msg,
                                    size_t error_msg_capacity) {
    try {
        if (!cfg || !levels || !diagnostics) {
            write_error(error_msg, error_msg_capacity, "Null pointer argument");
            return GFE_C_STATUS_INVALID_ARGUMENT;
        }
        const auto cpp_scenario = to_hierarchical_scenario(scenario);
        const auto model = abersoe::make_hierarchical_scenario_model(
            cpp_scenario, enable_bottom_up != 0, enable_top_down != 0);
        const auto initial_states = abersoe::make_hierarchical_scenario_initial_states(cpp_scenario);
        const auto state0 = abersoe::initialize_hierarchy_state(model, initial_states);

        abersoe::HierarchicalRuntimeConfig cpp_cfg;
        cpp_cfg.steps = cfg->steps;
        cpp_cfg.sample_stride = cfg->sample_stride;
        cpp_cfg.strict_finite = (cfg->strict_finite != 0);
        const auto result = abersoe::run(model, state0, cpp_cfg);

        write_level_summary_sizes(*levels, result.final_levels.size());
        if (!can_copy_level_summaries(*levels, result.final_levels.size())) {
            write_error(error_msg, error_msg_capacity, "Output buffer too small");
            return GFE_C_STATUS_BUFFER_TOO_SMALL;
        }

        copy_level_summaries(*levels, result);
        diagnostics->steps_executed = result.diagnostics.steps_executed;
        diagnostics->all_finite = result.diagnostics.all_finite ? 1 : 0;
        diagnostics->level_count = result.final_levels.size();
        return GFE_C_STATUS_OK;
    } catch (const std::invalid_argument& e) {
        write_error(error_msg, error_msg_capacity, e.what());
        return GFE_C_STATUS_INVALID_ARGUMENT;
    } catch (const std::exception& e) {
        write_error(error_msg, error_msg_capacity, e.what());
        return GFE_C_STATUS_RUNTIME_ERROR;
    } catch (...) {
        write_error(error_msg, error_msg_capacity, "Unknown C API error");
        return GFE_C_STATUS_RUNTIME_ERROR;
    }
}

int gfe_c_hierarchical_get_cross_level_report(int scenario,
                                              int enable_bottom_up,
                                              int enable_top_down,
                                              const gfe_c_hierarchical_runtime_config* cfg,
                                              gfe_c_hierarchical_kernel_shift_view* kernel_shifts,
                                              gfe_c_hierarchical_relation_view* relations,
                                              gfe_c_hierarchical_warning_view* warnings,
                                              gfe_c_hierarchical_cross_level_contract* contract,
                                              char* error_msg,
                                              size_t error_msg_capacity) {
    try {
        if (!cfg || !kernel_shifts || !relations || !warnings || !contract) {
            write_error(error_msg, error_msg_capacity, "Null pointer argument");
            return GFE_C_STATUS_INVALID_ARGUMENT;
        }
        const auto cpp_scenario = to_hierarchical_scenario(scenario);
        const auto model = abersoe::make_hierarchical_scenario_model(
            cpp_scenario, enable_bottom_up != 0, enable_top_down != 0);
        const auto initial_states = abersoe::make_hierarchical_scenario_initial_states(cpp_scenario);
        const auto state0 = abersoe::initialize_hierarchy_state(model, initial_states);
        abersoe::HierarchicalRuntimeConfig cpp_cfg;
        cpp_cfg.steps = cfg->steps;
        cpp_cfg.sample_stride = cfg->sample_stride;
        cpp_cfg.strict_finite = (cfg->strict_finite != 0);
        const auto run = abersoe::run(model, state0, cpp_cfg);
        const auto report = abersoe::make_hierarchical_cross_level_report(model, run);

        write_kernel_shift_sizes(*kernel_shifts, report.kernel_shifts.size());
        write_relation_sizes(*relations, report.relations.size());
        write_warning_sizes(*warnings, report.warnings.size());
        if (!can_copy_kernel_shifts(*kernel_shifts, report.kernel_shifts.size()) ||
            !can_copy_relations(*relations, report.relations.size()) ||
            !can_copy_warnings(*warnings, report.warnings.size())) {
            write_error(error_msg, error_msg_capacity, "Output buffer too small");
            return GFE_C_STATUS_BUFFER_TOO_SMALL;
        }

        copy_kernel_shifts(*kernel_shifts, report);
        copy_relations(*relations, report);
        copy_warnings(*warnings, report);
        contract->pass = report.contract.pass ? 1 : 0;
        contract->all_finite_pass = report.contract.all_finite_pass ? 1 : 0;
        contract->bottom_up_mcap_pass = report.contract.bottom_up_mcap_pass ? 1 : 0;
        contract->bottom_up_mscale_pass = report.contract.bottom_up_mscale_pass ? 1 : 0;
        contract->actionable_warning_count = report.contract.actionable_warning_count;
        return GFE_C_STATUS_OK;
    } catch (const std::invalid_argument& e) {
        write_error(error_msg, error_msg_capacity, e.what());
        return GFE_C_STATUS_INVALID_ARGUMENT;
    } catch (const std::exception& e) {
        write_error(error_msg, error_msg_capacity, e.what());
        return GFE_C_STATUS_RUNTIME_ERROR;
    } catch (...) {
        write_error(error_msg, error_msg_capacity, "Unknown C API error");
        return GFE_C_STATUS_RUNTIME_ERROR;
    }
}

int gfe_c_hierarchical_get_renorm_report(int scenario,
                                         int enable_bottom_up,
                                         int enable_top_down,
                                         gfe_c_hierarchical_renorm_analysis_view* analyses,
                                         char* error_msg,
                                         size_t error_msg_capacity) {
    try {
        if (!analyses) {
            write_error(error_msg, error_msg_capacity, "Null pointer argument");
            return GFE_C_STATUS_INVALID_ARGUMENT;
        }
        const auto cpp_scenario = to_hierarchical_scenario(scenario);
        const auto model = abersoe::make_hierarchical_scenario_model(
            cpp_scenario, enable_bottom_up != 0, enable_top_down != 0);
        const auto report = abersoe::make_hierarchical_adjacent_renorm_analyses(model);
        write_renorm_sizes(*analyses, report.size());
        if (!can_copy_renorm(*analyses, report.size())) {
            write_error(error_msg, error_msg_capacity, "Output buffer too small");
            return GFE_C_STATUS_BUFFER_TOO_SMALL;
        }
        copy_renorm(*analyses, report);
        return GFE_C_STATUS_OK;
    } catch (const std::invalid_argument& e) {
        write_error(error_msg, error_msg_capacity, e.what());
        return GFE_C_STATUS_INVALID_ARGUMENT;
    } catch (const std::exception& e) {
        write_error(error_msg, error_msg_capacity, e.what());
        return GFE_C_STATUS_RUNTIME_ERROR;
    } catch (...) {
        write_error(error_msg, error_msg_capacity, "Unknown C API error");
        return GFE_C_STATUS_RUNTIME_ERROR;
    }
}

int gfe_c_validate_hierarchical_chain_spec(const gfe_c_hierarchical_chain_spec_view* spec,
                                           char* error_msg,
                                           size_t error_msg_capacity) {
    try {
        if (!spec) {
            write_error(error_msg, error_msg_capacity, "Null pointer argument");
            return GFE_C_STATUS_INVALID_ARGUMENT;
        }
        const auto cpp_spec = to_cpp_hierarchical_chain_spec(*spec);
        std::string validation_error;
        if (!abersoe::validate_hierarchical_chain_spec(cpp_spec, &validation_error)) {
            write_error(error_msg, error_msg_capacity, validation_error);
            return GFE_C_STATUS_INVALID_ARGUMENT;
        }
        return GFE_C_STATUS_OK;
    } catch (const std::invalid_argument& e) {
        write_error(error_msg, error_msg_capacity, e.what());
        return GFE_C_STATUS_INVALID_ARGUMENT;
    } catch (const std::exception& e) {
        write_error(error_msg, error_msg_capacity, e.what());
        return GFE_C_STATUS_RUNTIME_ERROR;
    } catch (...) {
        write_error(error_msg, error_msg_capacity, "Unknown C API error");
        return GFE_C_STATUS_RUNTIME_ERROR;
    }
}

int gfe_c_hierarchical_run_chain_spec(const gfe_c_hierarchical_chain_spec_view* spec,
                                      const gfe_c_hierarchical_runtime_config* cfg,
                                      gfe_c_hierarchical_level_summary_view* levels,
                                      gfe_c_hierarchical_run_diagnostics* diagnostics,
                                      char* error_msg,
                                      size_t error_msg_capacity) {
    try {
        if (!spec || !cfg || !levels || !diagnostics) {
            write_error(error_msg, error_msg_capacity, "Null pointer argument");
            return GFE_C_STATUS_INVALID_ARGUMENT;
        }
        const auto cpp_spec = to_cpp_hierarchical_chain_spec(*spec);
        std::string validation_error;
        if (!abersoe::validate_hierarchical_chain_spec(cpp_spec, &validation_error)) {
            write_error(error_msg, error_msg_capacity, validation_error);
            return GFE_C_STATUS_INVALID_ARGUMENT;
        }
        const auto model = abersoe::make_hierarchical_model_from_spec(cpp_spec);
        const auto initial_states = abersoe::make_hierarchical_initial_states_from_spec(cpp_spec);
        const auto state0 = abersoe::initialize_hierarchy_state(model, initial_states);
        abersoe::HierarchicalRuntimeConfig cpp_cfg;
        cpp_cfg.steps = cfg->steps;
        cpp_cfg.sample_stride = cfg->sample_stride;
        cpp_cfg.strict_finite = (cfg->strict_finite != 0);
        const auto result = abersoe::run(model, state0, cpp_cfg);
        write_level_summary_sizes(*levels, result.final_levels.size());
        if (!can_copy_level_summaries(*levels, result.final_levels.size())) {
            write_error(error_msg, error_msg_capacity, "Output buffer too small");
            return GFE_C_STATUS_BUFFER_TOO_SMALL;
        }
        copy_level_summaries(*levels, result);
        diagnostics->steps_executed = result.diagnostics.steps_executed;
        diagnostics->all_finite = result.diagnostics.all_finite ? 1 : 0;
        diagnostics->level_count = result.final_levels.size();
        return GFE_C_STATUS_OK;
    } catch (const std::invalid_argument& e) {
        write_error(error_msg, error_msg_capacity, e.what());
        return GFE_C_STATUS_INVALID_ARGUMENT;
    } catch (const std::exception& e) {
        write_error(error_msg, error_msg_capacity, e.what());
        return GFE_C_STATUS_RUNTIME_ERROR;
    } catch (...) {
        write_error(error_msg, error_msg_capacity, "Unknown C API error");
        return GFE_C_STATUS_RUNTIME_ERROR;
    }
}

int gfe_c_hierarchical_get_cross_level_report_for_chain_spec(
    const gfe_c_hierarchical_chain_spec_view* spec,
    const gfe_c_hierarchical_runtime_config* cfg,
    gfe_c_hierarchical_kernel_shift_view* kernel_shifts,
    gfe_c_hierarchical_relation_view* relations,
    gfe_c_hierarchical_warning_view* warnings,
    gfe_c_hierarchical_cross_level_contract* contract,
    char* error_msg,
    size_t error_msg_capacity) {
    try {
        if (!spec || !cfg || !kernel_shifts || !relations || !warnings || !contract) {
            write_error(error_msg, error_msg_capacity, "Null pointer argument");
            return GFE_C_STATUS_INVALID_ARGUMENT;
        }
        const auto cpp_spec = to_cpp_hierarchical_chain_spec(*spec);
        std::string validation_error;
        if (!abersoe::validate_hierarchical_chain_spec(cpp_spec, &validation_error)) {
            write_error(error_msg, error_msg_capacity, validation_error);
            return GFE_C_STATUS_INVALID_ARGUMENT;
        }
        const auto model = abersoe::make_hierarchical_model_from_spec(cpp_spec);
        const auto initial_states = abersoe::make_hierarchical_initial_states_from_spec(cpp_spec);
        const auto state0 = abersoe::initialize_hierarchy_state(model, initial_states);
        abersoe::HierarchicalRuntimeConfig cpp_cfg;
        cpp_cfg.steps = cfg->steps;
        cpp_cfg.sample_stride = cfg->sample_stride;
        cpp_cfg.strict_finite = (cfg->strict_finite != 0);
        const auto run = abersoe::run(model, state0, cpp_cfg);
        const auto report = abersoe::make_hierarchical_cross_level_report(model, run);

        write_kernel_shift_sizes(*kernel_shifts, report.kernel_shifts.size());
        write_relation_sizes(*relations, report.relations.size());
        write_warning_sizes(*warnings, report.warnings.size());
        if (!can_copy_kernel_shifts(*kernel_shifts, report.kernel_shifts.size()) ||
            !can_copy_relations(*relations, report.relations.size()) ||
            !can_copy_warnings(*warnings, report.warnings.size())) {
            write_error(error_msg, error_msg_capacity, "Output buffer too small");
            return GFE_C_STATUS_BUFFER_TOO_SMALL;
        }

        copy_kernel_shifts(*kernel_shifts, report);
        copy_relations(*relations, report);
        copy_warnings(*warnings, report);
        contract->pass = report.contract.pass ? 1 : 0;
        contract->all_finite_pass = report.contract.all_finite_pass ? 1 : 0;
        contract->bottom_up_mcap_pass = report.contract.bottom_up_mcap_pass ? 1 : 0;
        contract->bottom_up_mscale_pass = report.contract.bottom_up_mscale_pass ? 1 : 0;
        contract->actionable_warning_count = report.contract.actionable_warning_count;
        return GFE_C_STATUS_OK;
    } catch (const std::invalid_argument& e) {
        write_error(error_msg, error_msg_capacity, e.what());
        return GFE_C_STATUS_INVALID_ARGUMENT;
    } catch (const std::exception& e) {
        write_error(error_msg, error_msg_capacity, e.what());
        return GFE_C_STATUS_RUNTIME_ERROR;
    } catch (...) {
        write_error(error_msg, error_msg_capacity, "Unknown C API error");
        return GFE_C_STATUS_RUNTIME_ERROR;
    }
}

int gfe_c_hierarchical_get_renorm_report_for_chain_spec(
    const gfe_c_hierarchical_chain_spec_view* spec,
    gfe_c_hierarchical_renorm_analysis_view* analyses,
    char* error_msg,
    size_t error_msg_capacity) {
    try {
        if (!spec || !analyses) {
            write_error(error_msg, error_msg_capacity, "Null pointer argument");
            return GFE_C_STATUS_INVALID_ARGUMENT;
        }
        const auto cpp_spec = to_cpp_hierarchical_chain_spec(*spec);
        std::string validation_error;
        if (!abersoe::validate_hierarchical_chain_spec(cpp_spec, &validation_error)) {
            write_error(error_msg, error_msg_capacity, validation_error);
            return GFE_C_STATUS_INVALID_ARGUMENT;
        }
        const auto model = abersoe::make_hierarchical_model_from_spec(cpp_spec);
        const auto report = abersoe::make_hierarchical_adjacent_renorm_analyses(model);
        write_renorm_sizes(*analyses, report.size());
        if (!can_copy_renorm(*analyses, report.size())) {
            write_error(error_msg, error_msg_capacity, "Output buffer too small");
            return GFE_C_STATUS_BUFFER_TOO_SMALL;
        }
        copy_renorm(*analyses, report);
        return GFE_C_STATUS_OK;
    } catch (const std::invalid_argument& e) {
        write_error(error_msg, error_msg_capacity, e.what());
        return GFE_C_STATUS_INVALID_ARGUMENT;
    } catch (const std::exception& e) {
        write_error(error_msg, error_msg_capacity, e.what());
        return GFE_C_STATUS_RUNTIME_ERROR;
    } catch (...) {
        write_error(error_msg, error_msg_capacity, "Unknown C API error");
        return GFE_C_STATUS_RUNTIME_ERROR;
    }
}

int gfe_c_hierarchical_step_chain_spec(
    const gfe_c_hierarchical_chain_spec_view* spec,
    gfe_c_state_mut_view* level_states,
    size_t level_count,
    const double* external_forcing,
    size_t external_forcing_size,
    size_t forcing_level,
    gfe_c_memory_kernel_mut_view* active_kernels,
    size_t active_kernel_count,
    gfe_c_spectral_units* spectral_units,
    size_t spectral_count,
    char* error_msg,
    size_t error_msg_capacity) {
    try {
        if (!spec || !level_states || !active_kernels || !spectral_units) {
            write_error(error_msg, error_msg_capacity, "Null pointer argument");
            return GFE_C_STATUS_INVALID_ARGUMENT;
        }
        const auto cpp_spec = to_cpp_hierarchical_chain_spec(*spec);
        std::string validation_error;
        if (!abersoe::validate_hierarchical_chain_spec(cpp_spec, &validation_error)) {
            write_error(error_msg, error_msg_capacity, validation_error);
            return GFE_C_STATUS_INVALID_ARGUMENT;
        }
        if (level_count != spec->level_count ||
            active_kernel_count != spec->level_count ||
            spectral_count != spec->level_count) {
            write_error(error_msg, error_msg_capacity, "Array counts must equal level_count");
            return GFE_C_STATUS_INVALID_ARGUMENT;
        }
        if (forcing_level >= spec->level_count) {
            write_error(error_msg, error_msg_capacity, "forcing_level out of range");
            return GFE_C_STATUS_INVALID_ARGUMENT;
        }

        const auto model = abersoe::make_hierarchical_model_from_spec(cpp_spec);

        // Build current hierarchy state from the per-level mutable views.
        abersoe::HierarchicalState current(level_count);
        for (std::size_t i = 0; i < level_count; ++i) {
            current[i].state = to_cpp_state(level_states[i]);
            current[i].active_kernel = model.levels[i].model.kernel;
            if (current[i].state.chi.empty()) {
                current[i].state.chi.assign(current[i].active_kernel.gamma.size(), 0.0);
            }
            current[i].spectral = gfe::spectral_units(
                current[i].active_kernel.w, current[i].active_kernel.gamma);
        }

        // Build external forcing vector.
        gfe::State ext(external_forcing, external_forcing + external_forcing_size);

        // Step with external forcing injection.
        const auto next = abersoe::step_with_external_forcing(
            model, current, forcing_level, ext);

        // Validate all output buffers BEFORE writing anything, following the
        // two-phase write-sizes/validate/copy pattern used by the other ABI
        // entry points.  Prevents partial writes and buffer overruns when a
        // caller supplies undersized buffers.
        for (std::size_t i = 0; i < level_count; ++i) {
            write_state_sizes(level_states[i], next[i].state);
            write_kernel_sizes(active_kernels[i], next[i].active_kernel);
        }
        for (std::size_t i = 0; i < level_count; ++i) {
            if (!can_copy_state(level_states[i], next[i].state) ||
                !can_copy_kernel(active_kernels[i], next[i].active_kernel)) {
                write_error(error_msg, error_msg_capacity, "Output buffer too small");
                return GFE_C_STATUS_BUFFER_TOO_SMALL;
            }
        }

        // Write back updated states, active kernels, and spectral units.
        for (std::size_t i = 0; i < level_count; ++i) {
            copy_state(level_states[i], next[i].state);
            copy_kernel(active_kernels[i], next[i].active_kernel);
            spectral_units[i].mcap  = next[i].spectral.Mcap;
            spectral_units[i].mscale = next[i].spectral.Mscale;
            spectral_units[i].mres  = next[i].spectral.Mres;
            spectral_units[i].hmem  = next[i].spectral.Hmem;
            spectral_units[i].hnorm = next[i].spectral.Hnorm;
            spectral_units[i].deff  = next[i].spectral.Deff;
        }
        return GFE_C_STATUS_OK;
    } catch (const std::invalid_argument& e) {
        write_error(error_msg, error_msg_capacity, e.what());
        return GFE_C_STATUS_INVALID_ARGUMENT;
    } catch (const std::exception& e) {
        write_error(error_msg, error_msg_capacity, e.what());
        return GFE_C_STATUS_RUNTIME_ERROR;
    } catch (...) {
        write_error(error_msg, error_msg_capacity, "Unknown C API error");
        return GFE_C_STATUS_RUNTIME_ERROR;
    }
}

int gfe_c_validate_memory_kernel(const gfe_c_memory_kernel_view* kernel,
                                 char* error_msg,
                                 size_t error_msg_capacity) {
    try {
        if (!kernel) {
            write_error(error_msg, error_msg_capacity, "Null kernel argument");
            return GFE_C_STATUS_INVALID_ARGUMENT;
        }
        const auto cpp_kernel = to_cpp_kernel(*kernel);
        gfe::validate_decay_ordering(cpp_kernel.gamma);
        return GFE_C_STATUS_OK;
    } catch (const std::invalid_argument& e) {
        write_error(error_msg, error_msg_capacity, e.what());
        return GFE_C_STATUS_INVALID_ARGUMENT;
    } catch (const std::exception& e) {
        write_error(error_msg, error_msg_capacity, e.what());
        return GFE_C_STATUS_RUNTIME_ERROR;
    } catch (...) {
        write_error(error_msg, error_msg_capacity, "Unknown C API error");
        return GFE_C_STATUS_RUNTIME_ERROR;
    }
}

int gfe_c_spectral_units_from_kernel(const gfe_c_memory_kernel_view* kernel,
                                     const gfe_c_spectral_unit_options* options,
                                     gfe_c_spectral_units* out,
                                     char* error_msg,
                                     size_t error_msg_capacity) {
    try {
        if (!kernel || !options || !out) {
            write_error(error_msg, error_msg_capacity, "Null pointer argument");
            return GFE_C_STATUS_INVALID_ARGUMENT;
        }
        const auto cpp_kernel = to_cpp_kernel(*kernel);
        const auto cpp_options = to_cpp_spectral_options(*options);
        const auto spectral = gfe::spectral_units(cpp_kernel.w, cpp_kernel.gamma, cpp_options);
        *out = to_c_spectral_units(spectral);
        return GFE_C_STATUS_OK;
    } catch (const std::invalid_argument& e) {
        write_error(error_msg, error_msg_capacity, e.what());
        return GFE_C_STATUS_INVALID_ARGUMENT;
    } catch (const std::exception& e) {
        write_error(error_msg, error_msg_capacity, e.what());
        return GFE_C_STATUS_RUNTIME_ERROR;
    } catch (...) {
        write_error(error_msg, error_msg_capacity, "Unknown C API error");
        return GFE_C_STATUS_RUNTIME_ERROR;
    }
}

int gfe_c_fit_backend_name(int backend,
                           gfe_c_string_result_view* out,
                           char* error_msg,
                           size_t error_msg_capacity) {
    try {
        const auto name = fit_backend_name(backend);
        const int rc = write_string_result(name, out);
        if (rc == GFE_C_STATUS_BUFFER_TOO_SMALL) {
            write_error(error_msg, error_msg_capacity, "Output buffer too small");
        } else if (rc == GFE_C_STATUS_INVALID_ARGUMENT) {
            write_error(error_msg, error_msg_capacity, "Invalid string output buffer");
        }
        return rc;
    } catch (const std::invalid_argument& e) {
        write_error(error_msg, error_msg_capacity, e.what());
        return GFE_C_STATUS_INVALID_ARGUMENT;
    } catch (const std::exception& e) {
        write_error(error_msg, error_msg_capacity, e.what());
        return GFE_C_STATUS_RUNTIME_ERROR;
    } catch (...) {
        write_error(error_msg, error_msg_capacity, "Unknown C API error");
        return GFE_C_STATUS_RUNTIME_ERROR;
    }
}

int gfe_c_compute_backend_name(int backend,
                               gfe_c_string_result_view* out,
                               char* error_msg,
                               size_t error_msg_capacity) {
    try {
        const auto name = compute_backend_name(backend);
        const int rc = write_string_result(name, out);
        if (rc == GFE_C_STATUS_BUFFER_TOO_SMALL) {
            write_error(error_msg, error_msg_capacity, "Output buffer too small");
        } else if (rc == GFE_C_STATUS_INVALID_ARGUMENT) {
            write_error(error_msg, error_msg_capacity, "Invalid string output buffer");
        }
        return rc;
    } catch (const std::invalid_argument& e) {
        write_error(error_msg, error_msg_capacity, e.what());
        return GFE_C_STATUS_INVALID_ARGUMENT;
    } catch (const std::exception& e) {
        write_error(error_msg, error_msg_capacity, e.what());
        return GFE_C_STATUS_RUNTIME_ERROR;
    } catch (...) {
        write_error(error_msg, error_msg_capacity, "Unknown C API error");
        return GFE_C_STATUS_RUNTIME_ERROR;
    }
}

int gfe_c_hierarchy_run_csv_header(gfe_c_string_result_view* out,
                                   char* error_msg,
                                   size_t error_msg_capacity) {
    try {
        const int rc = write_string_result(abersoe::hierarchy_run_csv_header(), out);
        if (rc == GFE_C_STATUS_BUFFER_TOO_SMALL) {
            write_error(error_msg, error_msg_capacity, "Output buffer too small");
        } else if (rc == GFE_C_STATUS_INVALID_ARGUMENT) {
            write_error(error_msg, error_msg_capacity, "Invalid string output buffer");
        }
        return rc;
    } catch (const std::exception& e) {
        write_error(error_msg, error_msg_capacity, e.what());
        return GFE_C_STATUS_RUNTIME_ERROR;
    } catch (...) {
        write_error(error_msg, error_msg_capacity, "Unknown C API error");
        return GFE_C_STATUS_RUNTIME_ERROR;
    }
}

int gfe_c_hierarchy_summary_csv_header(gfe_c_string_result_view* out,
                                       char* error_msg,
                                       size_t error_msg_capacity) {
    try {
        const int rc = write_string_result(abersoe::hierarchy_summary_csv_header(), out);
        if (rc == GFE_C_STATUS_BUFFER_TOO_SMALL) {
            write_error(error_msg, error_msg_capacity, "Output buffer too small");
        } else if (rc == GFE_C_STATUS_INVALID_ARGUMENT) {
            write_error(error_msg, error_msg_capacity, "Invalid string output buffer");
        }
        return rc;
    } catch (const std::exception& e) {
        write_error(error_msg, error_msg_capacity, e.what());
        return GFE_C_STATUS_RUNTIME_ERROR;
    } catch (...) {
        write_error(error_msg, error_msg_capacity, "Unknown C API error");
        return GFE_C_STATUS_RUNTIME_ERROR;
    }
}

int gfe_c_hierarchy_diagnostics_csv_header(gfe_c_string_result_view* out,
                                           char* error_msg,
                                           size_t error_msg_capacity) {
    try {
        const int rc = write_string_result(abersoe::hierarchy_diagnostics_csv_header(), out);
        if (rc == GFE_C_STATUS_BUFFER_TOO_SMALL) {
            write_error(error_msg, error_msg_capacity, "Output buffer too small");
        } else if (rc == GFE_C_STATUS_INVALID_ARGUMENT) {
            write_error(error_msg, error_msg_capacity, "Invalid string output buffer");
        }
        return rc;
    } catch (const std::exception& e) {
        write_error(error_msg, error_msg_capacity, e.what());
        return GFE_C_STATUS_RUNTIME_ERROR;
    } catch (...) {
        write_error(error_msg, error_msg_capacity, "Unknown C API error");
        return GFE_C_STATUS_RUNTIME_ERROR;
    }
}

int gfe_c_hierarchy_cross_level_csv_header(gfe_c_string_result_view* out,
                                           char* error_msg,
                                           size_t error_msg_capacity) {
    try {
        const int rc = write_string_result(abersoe::hierarchy_cross_level_csv_header(), out);
        if (rc == GFE_C_STATUS_BUFFER_TOO_SMALL) {
            write_error(error_msg, error_msg_capacity, "Output buffer too small");
        } else if (rc == GFE_C_STATUS_INVALID_ARGUMENT) {
            write_error(error_msg, error_msg_capacity, "Invalid string output buffer");
        }
        return rc;
    } catch (const std::exception& e) {
        write_error(error_msg, error_msg_capacity, e.what());
        return GFE_C_STATUS_RUNTIME_ERROR;
    } catch (...) {
        write_error(error_msg, error_msg_capacity, "Unknown C API error");
        return GFE_C_STATUS_RUNTIME_ERROR;
    }
}

int gfe_c_hierarchy_renorm_csv_header(gfe_c_string_result_view* out,
                                      char* error_msg,
                                      size_t error_msg_capacity) {
    try {
        const int rc = write_string_result(abersoe::hierarchy_renorm_csv_header(), out);
        if (rc == GFE_C_STATUS_BUFFER_TOO_SMALL) {
            write_error(error_msg, error_msg_capacity, "Output buffer too small");
        } else if (rc == GFE_C_STATUS_INVALID_ARGUMENT) {
            write_error(error_msg, error_msg_capacity, "Invalid string output buffer");
        }
        return rc;
    } catch (const std::exception& e) {
        write_error(error_msg, error_msg_capacity, e.what());
        return GFE_C_STATUS_RUNTIME_ERROR;
    } catch (...) {
        write_error(error_msg, error_msg_capacity, "Unknown C API error");
        return GFE_C_STATUS_RUNTIME_ERROR;
    }
}

int gfe_c_fit_soe_kernel(const double* t,
                         size_t t_len,
                         const double* data,
                         size_t data_len,
                         const gfe_c_soe_fit_options* options,
                         const gfe_c_backend_config* backend_cfg,
                         gfe_c_fit_result_view* out,
                         char* error_msg,
                         size_t error_msg_capacity) {
    try {
        if (!t || !data || !options || !backend_cfg || !out) {
            write_error(error_msg, error_msg_capacity, "Null pointer argument");
            return GFE_C_STATUS_INVALID_ARGUMENT;
        }
        if (t_len == 0 || data_len == 0 || t_len != data_len) {
            write_error(error_msg, error_msg_capacity, "t and data must have same non-zero length");
            return GFE_C_STATUS_INVALID_ARGUMENT;
        }
        if (!out->gamma_fit || !out->w_fit || !out->fit) {
            write_error(error_msg, error_msg_capacity, "Output buffers must not be null");
            return GFE_C_STATUS_INVALID_ARGUMENT;
        }

        const gfe::Vec t_vec(t, t + t_len);
        const gfe::Vec d_vec(data, data + data_len);
        const gfe::SoeFitOptions cpp_opt = to_cpp_options(*options);
        const gfe::BackendConfig cpp_backend = to_cpp_backend(*backend_cfg);

        const auto fit = gfe::fit_soe_kernel(t_vec, d_vec, cpp_opt, cpp_backend);
        write_sizes(*out, fit);

        if (!can_copy(*out, fit)) {
            write_error(error_msg, error_msg_capacity, "Output buffer too small");
            return GFE_C_STATUS_BUFFER_TOO_SMALL;
        }

        copy_fit(*out, fit);
        return GFE_C_STATUS_OK;
    } catch (const std::invalid_argument& e) {
        write_error(error_msg, error_msg_capacity, e.what());
        return GFE_C_STATUS_INVALID_ARGUMENT;
    } catch (const std::exception& e) {
        write_error(error_msg, error_msg_capacity, e.what());
        return GFE_C_STATUS_RUNTIME_ERROR;
    } catch (...) {
        write_error(error_msg, error_msg_capacity, "Unknown C API error");
        return GFE_C_STATUS_RUNTIME_ERROR;
    }
}

} // extern "C"
