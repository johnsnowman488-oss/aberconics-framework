#pragma once

#include <stddef.h>

#if defined(_WIN32)
#  if defined(GFE_CORE_EXPORTS)
#    define GFE_C_API __declspec(dllexport)
#  else
#    define GFE_C_API __declspec(dllimport)
#  endif
#else
#  define GFE_C_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum gfe_c_status {
    GFE_C_STATUS_OK = 0,
    GFE_C_STATUS_INVALID_ARGUMENT = 1,
    GFE_C_STATUS_RUNTIME_ERROR = 2,
    GFE_C_STATUS_BUFFER_TOO_SMALL = 3
} gfe_c_status;

typedef enum gfe_c_bool {
    GFE_C_FALSE = 0,
    GFE_C_TRUE = 1
} gfe_c_bool;

typedef enum gfe_c_fit_backend {
    GFE_C_FIT_BACKEND_NNLS_PG = 0,
    GFE_C_FIT_BACKEND_PRONY = 1
} gfe_c_fit_backend;

typedef enum gfe_c_compute_backend {
    GFE_C_COMPUTE_BACKEND_CPU_REF = 0,
    GFE_C_COMPUTE_BACKEND_CPU_BLAS = 1,
    GFE_C_COMPUTE_BACKEND_GPU_CUDA = 2,
    GFE_C_COMPUTE_BACKEND_GPU_HIP = 3,
    GFE_C_COMPUTE_BACKEND_GPU_SYCL = 4
} gfe_c_compute_backend;

typedef enum gfe_c_coupling_form {
    GFE_C_COUPLING_FORM_A = 0,
    GFE_C_COUPLING_FORM_B = 1,
    GFE_C_COUPLING_FORM_C = 2
} gfe_c_coupling_form;

typedef enum gfe_c_abersoe_scenario {
    GFE_C_ABERSOE_SCENARIO_LINEAR = 0,
    GFE_C_ABERSOE_SCENARIO_LORENZ63 = 1,
    GFE_C_ABERSOE_SCENARIO_RESONANT1D = 2
} gfe_c_abersoe_scenario;

typedef enum gfe_c_hierarchical_scenario {
    GFE_C_HIERARCHICAL_SCENARIO_TWO_LEVEL_BIDIRECTIONAL = 0,
    GFE_C_HIERARCHICAL_SCENARIO_THREE_LEVEL_HOMEOSTATIC = 1
} gfe_c_hierarchical_scenario;

typedef enum gfe_c_hierarchical_relation_type {
    GFE_C_HIERARCHICAL_RELATION_BOTTOM_UP = 0,
    GFE_C_HIERARCHICAL_RELATION_TOP_DOWN = 1
} gfe_c_hierarchical_relation_type;

typedef enum gfe_c_hierarchical_warning_severity {
    GFE_C_HIERARCHICAL_WARNING_INFO = 0,
    GFE_C_HIERARCHICAL_WARNING_WARNING = 1,
    GFE_C_HIERARCHICAL_WARNING_ERROR = 2
} gfe_c_hierarchical_warning_severity;

typedef enum gfe_c_hierarchical_warning_code {
    GFE_C_HIERARCHICAL_WARNING_CODE_UNKNOWN = 0,
    GFE_C_HIERARCHICAL_WARNING_CODE_NON_FINITE_RUN = 1,
    GFE_C_HIERARCHICAL_WARNING_CODE_KERNEL_SHIFT_DETECTED = 2,
    GFE_C_HIERARCHICAL_WARNING_CODE_LARGE_KERNEL_SHIFT = 3,
    GFE_C_HIERARCHICAL_WARNING_CODE_BOTTOM_UP_MCAP_INVERSION = 4,
    GFE_C_HIERARCHICAL_WARNING_CODE_BOTTOM_UP_MSCALE_INVERSION = 5,
    GFE_C_HIERARCHICAL_WARNING_CODE_TOP_DOWN_TIMESCALE_SHIFT = 6,
    GFE_C_HIERARCHICAL_WARNING_CODE_TOP_DOWN_DEFF_SHIFT = 7
} gfe_c_hierarchical_warning_code;

typedef enum gfe_c_hierarchical_warning_metric {
    GFE_C_HIERARCHICAL_WARNING_METRIC_UNKNOWN = 0,
    GFE_C_HIERARCHICAL_WARNING_METRIC_ALL_FINITE = 1,
    GFE_C_HIERARCHICAL_WARNING_METRIC_KERNEL_SHIFT = 2,
    GFE_C_HIERARCHICAL_WARNING_METRIC_DELTA_MCAP = 3,
    GFE_C_HIERARCHICAL_WARNING_METRIC_DELTA_MSCALE = 4,
    GFE_C_HIERARCHICAL_WARNING_METRIC_DELTA_DEFF = 5
} gfe_c_hierarchical_warning_metric;

typedef enum gfe_c_hierarchical_renorm_diagnostic {
    GFE_C_HIERARCHICAL_RENORM_DIAGNOSTIC_NONE = 0,
    GFE_C_HIERARCHICAL_RENORM_DIAGNOSTIC_FIT_INSTABILITY = 1,
    GFE_C_HIERARCHICAL_RENORM_DIAGNOSTIC_OVER_PRUNING = 2,
    GFE_C_HIERARCHICAL_RENORM_DIAGNOSTIC_PLAUSIBLE_CHANNEL_MERGING = 3
} gfe_c_hierarchical_renorm_diagnostic;

typedef enum gfe_c_hebbian_rule {
    GFE_C_HEBBIAN_RULE_UNKNOWN = 0,
    GFE_C_HEBBIAN_RULE_OJA_NORMALIZED = 1,
    GFE_C_HEBBIAN_RULE_PLAIN_HEBBIAN = 2
} gfe_c_hebbian_rule;

typedef enum gfe_c_deff_policy {
    GFE_C_DEFF_EFFECTIVE_RANK_ONLY = 0,
    GFE_C_DEFF_PAPER_EFFECTIVE_DIMENSION = 1
} gfe_c_deff_policy;

typedef enum gfe_c_mcap_policy {
    GFE_C_MCAP_LEGACY_WEIGHTED_TIMESCALE = 0,
    GFE_C_MCAP_PAPER_MEAN_DEPTH = 1
} gfe_c_mcap_policy;

typedef enum gfe_c_mres_policy {
    GFE_C_MRES_RETURN_NAN = 0,
    GFE_C_MRES_RETURN_INFINITY = 1,
    GFE_C_MRES_THROW_ON_ZERO_SCALE = 2
} gfe_c_mres_policy;

typedef struct gfe_c_soe_fit_options {
    int backend;
    int n_basis;
    double gamma_min;
    double gamma_max;
    double threshold;
    double lr;
    int iters;

    int prony_order;
    double prony_ridge;
    double prony_root_imag_tol;
    double prony_root_min;
    double prony_root_max;
    int prony_enforce_nonnegative_weights;
    int prony_max_root_iters;
    double prony_root_tol;
} gfe_c_soe_fit_options;

typedef struct gfe_c_backend_config {
    int preferred;
    int allow_fallback;
    int fallback;
} gfe_c_backend_config;

typedef struct gfe_c_spectral_unit_options {
    int deff_policy;
    int mcap_policy;
    int mres_policy;
    double mscale_zero_tol;
} gfe_c_spectral_unit_options;

typedef struct gfe_c_spectral_units {
    double mcap;
    double mscale;
    double mres;
    double hmem;
    double hnorm;
    double deff;
} gfe_c_spectral_units;

typedef struct gfe_c_memory_kernel_view {
    const double* gamma;
    size_t gamma_size;
    const double* w;
    size_t w_size;
} gfe_c_memory_kernel_view;

typedef struct gfe_c_memory_kernel_mut_view {
    double* gamma;
    size_t gamma_capacity;
    size_t* gamma_size;
    double* w;
    size_t w_capacity;
    size_t* w_size;
} gfe_c_memory_kernel_mut_view;

typedef struct gfe_c_state_view {
    const double* u;
    size_t u_size;
    const double* chi;
    size_t chi_size;
    double t;
} gfe_c_state_view;

typedef struct gfe_c_state_mut_view {
    double* u;
    size_t u_capacity;
    size_t* u_size;
    double* chi;
    size_t chi_capacity;
    size_t* chi_size;
    double* t;
} gfe_c_state_mut_view;

typedef struct gfe_c_abersoe_diagnostics {
    size_t steps_executed;
    int all_finite;
    double max_abs_u;
    double max_abs_chi;
    double final_u_l2;
    double final_chi_l2;
    double mean_abs_u;
    double mean_abs_chi;
} gfe_c_abersoe_diagnostics;

typedef struct gfe_c_abersoe_config_record {
    size_t memory_channels;
    double dt;
    size_t coupling_index;
    size_t steps;
    size_t sample_stride;
    int strict_finite;
    int stochastic_forcing;
    double forcing_noise_std;
    size_t stochastic_channel;
    unsigned long long seed;
    int fit_backend;
    int hebbian_enabled;
    int hebbian_rule;
    double hebbian_learning_rate;
    double hebbian_decay;
} gfe_c_abersoe_config_record;

typedef struct gfe_c_abersoe_runtime_config {
    size_t steps;
    size_t sample_stride;
    int strict_finite;
    int stochastic_forcing;
    double forcing_noise_std;
    size_t stochastic_channel;
    unsigned long long seed;
} gfe_c_abersoe_runtime_config;

typedef struct gfe_c_hierarchical_runtime_config {
    size_t steps;
    size_t sample_stride;
    int strict_finite;
} gfe_c_hierarchical_runtime_config;

typedef struct gfe_c_abersoe_sample_summary_view {
    double* t;
    size_t t_capacity;
    size_t* t_size;
    double* u0;
    size_t u0_capacity;
    size_t* u0_size;
    double* chi0;
    size_t chi0_capacity;
    size_t* chi0_size;
    double* u_l2;
    size_t u_l2_capacity;
    size_t* u_l2_size;
    double* chi_l2;
    size_t chi_l2_capacity;
    size_t* chi_l2_size;
} gfe_c_abersoe_sample_summary_view;

typedef struct gfe_c_abersoe_trajectory_view {
    double* t;
    size_t t_capacity;
    size_t* t_size;
    double* u;
    size_t u_capacity;
    size_t* u_size;
    size_t* u_dim;
    double* chi;
    size_t chi_capacity;
    size_t* chi_size;
    size_t* chi_dim;
} gfe_c_abersoe_trajectory_view;

typedef struct gfe_c_hierarchical_run_diagnostics {
    size_t steps_executed;
    int all_finite;
    size_t level_count;
} gfe_c_hierarchical_run_diagnostics;

typedef struct gfe_c_hierarchical_cross_level_contract {
    int pass;
    int all_finite_pass;
    int bottom_up_mcap_pass;
    int bottom_up_mscale_pass;
    size_t actionable_warning_count;
} gfe_c_hierarchical_cross_level_contract;

typedef struct gfe_c_hierarchical_chain_level_spec {
    const char* name;
    gfe_c_memory_kernel_view kernel;
    gfe_c_state_view initial_state;
    double dt;
    const double* linear_decay;
    size_t linear_decay_size;
    const double* forcing_bias;
    size_t forcing_bias_size;
    int form;
    size_t coupling_index;
} gfe_c_hierarchical_chain_level_spec;

typedef struct gfe_c_hierarchical_chain_edge_spec {
    size_t source_level;
    size_t target_level;
    int relation;
    double gain;
    int normalize_weights;
} gfe_c_hierarchical_chain_edge_spec;

typedef struct gfe_c_hierarchical_chain_spec_view {
    const gfe_c_hierarchical_chain_level_spec* levels;
    size_t level_count;
    const gfe_c_hierarchical_chain_edge_spec* edges;
    size_t edge_count;
} gfe_c_hierarchical_chain_spec_view;

typedef struct gfe_c_hierarchical_level_summary_view {
    double* t;
    size_t t_capacity;
    size_t* t_size;
    size_t* state_dim;
    size_t state_dim_capacity;
    size_t* state_dim_size;
    size_t* memory_channels;
    size_t memory_channels_capacity;
    size_t* memory_channels_size;
    double* u0;
    size_t u0_capacity;
    size_t* u0_size;
    double* chi0;
    size_t chi0_capacity;
    size_t* chi0_size;
    double* w0;
    size_t w0_capacity;
    size_t* w0_size;
    double* gamma0;
    size_t gamma0_capacity;
    size_t* gamma0_size;
    double* mcap;
    size_t mcap_capacity;
    size_t* mcap_size;
    double* mscale;
    size_t mscale_capacity;
    size_t* mscale_size;
    double* hmem;
    size_t hmem_capacity;
    size_t* hmem_size;
    double* deff;
    size_t deff_capacity;
    size_t* deff_size;
} gfe_c_hierarchical_level_summary_view;

typedef struct gfe_c_hierarchical_kernel_shift_view {
    size_t* level_idx;
    size_t level_idx_capacity;
    size_t* level_idx_size;
    double* delta_mcap;
    size_t delta_mcap_capacity;
    size_t* delta_mcap_size;
    double* delta_mscale;
    size_t delta_mscale_capacity;
    size_t* delta_mscale_size;
    double* delta_hmem;
    size_t delta_hmem_capacity;
    size_t* delta_hmem_size;
    double* delta_deff;
    size_t delta_deff_capacity;
    size_t* delta_deff_size;
    int* changed;
    size_t changed_capacity;
    size_t* changed_size;
} gfe_c_hierarchical_kernel_shift_view;

typedef struct gfe_c_hierarchical_relation_view {
    int* relation_type;
    size_t relation_type_capacity;
    size_t* relation_type_size;
    size_t* source_level;
    size_t source_level_capacity;
    size_t* source_level_size;
    size_t* target_level;
    size_t target_level_capacity;
    size_t* target_level_size;
    double* delta_mcap;
    size_t delta_mcap_capacity;
    size_t* delta_mcap_size;
    double* delta_mscale;
    size_t delta_mscale_capacity;
    size_t* delta_mscale_size;
    double* delta_hmem;
    size_t delta_hmem_capacity;
    size_t* delta_hmem_size;
    double* delta_deff;
    size_t delta_deff_capacity;
    size_t* delta_deff_size;
    int* target_mcap_not_lower;
    size_t target_mcap_not_lower_capacity;
    size_t* target_mcap_not_lower_size;
    int* target_mscale_not_lower;
    size_t target_mscale_not_lower_capacity;
    size_t* target_mscale_not_lower_size;
} gfe_c_hierarchical_relation_view;

typedef struct gfe_c_hierarchical_warning_view {
    int* severity;
    size_t severity_capacity;
    size_t* severity_size;
    int* code;
    size_t code_capacity;
    size_t* code_size;
    int* metric;
    size_t metric_capacity;
    size_t* metric_size;
    double* observed;
    size_t observed_capacity;
    size_t* observed_size;
    double* threshold;
    size_t threshold_capacity;
    size_t* threshold_size;
    size_t* source_level;
    size_t source_level_capacity;
    size_t* source_level_size;
    size_t* target_level;
    size_t target_level_capacity;
    size_t* target_level_size;
    int* actionable;
    size_t actionable_capacity;
    size_t* actionable_size;
} gfe_c_hierarchical_warning_view;

typedef struct gfe_c_hierarchical_renorm_analysis_view {
    size_t* input_level;
    size_t input_level_capacity;
    size_t* input_level_size;
    size_t* bath_level;
    size_t bath_level_capacity;
    size_t* bath_level_size;
    size_t* direct_level;
    size_t direct_level_capacity;
    size_t* direct_level_size;
    int* primary_diagnostic;
    size_t primary_diagnostic_capacity;
    size_t* primary_diagnostic_size;
    size_t* raw_output_channels;
    size_t raw_output_channels_capacity;
    size_t* raw_output_channels_size;
    size_t* output_channels;
    size_t output_channels_capacity;
    size_t* output_channels_size;
    size_t* pruned_channels;
    size_t pruned_channels_capacity;
    size_t* pruned_channels_size;
    double* retained_weight_fraction;
    size_t retained_weight_fraction_capacity;
    size_t* retained_weight_fraction_size;
    double* fit_l2_error;
    size_t fit_l2_error_capacity;
    size_t* fit_l2_error_size;
    double* pruned_fit_l2_error;
    size_t pruned_fit_l2_error_capacity;
    size_t* pruned_fit_l2_error_size;
    double* fit_linf_error;
    size_t fit_linf_error_capacity;
    size_t* fit_linf_error_size;
    double* pruned_fit_linf_error;
    size_t pruned_fit_linf_error_capacity;
    size_t* pruned_fit_linf_error_size;
    double* delta_mcap;
    size_t delta_mcap_capacity;
    size_t* delta_mcap_size;
    double* delta_mscale;
    size_t delta_mscale_capacity;
    size_t* delta_mscale_size;
    double* delta_deff;
    size_t delta_deff_capacity;
    size_t* delta_deff_size;
    int* consistency_pass;
    size_t consistency_pass_capacity;
    size_t* consistency_pass_size;
    int* mcap_pass;
    size_t mcap_pass_capacity;
    size_t* mcap_pass_size;
    int* mscale_pass;
    size_t mscale_pass_capacity;
    size_t* mscale_pass_size;
    int* deff_pass;
    size_t deff_pass_capacity;
    size_t* deff_pass_size;
} gfe_c_hierarchical_renorm_analysis_view;

typedef struct gfe_c_fit_result_view {
    double* gamma_fit;
    size_t gamma_fit_capacity;
    size_t* gamma_fit_size;

    double* w_fit;
    size_t w_fit_capacity;
    size_t* w_fit_size;

    double* fit;
    size_t fit_capacity;
    size_t* fit_size;
} gfe_c_fit_result_view;

typedef struct gfe_c_string_result_view {
    char* data;
    size_t capacity;
    size_t* size;
} gfe_c_string_result_view;

GFE_C_API gfe_c_soe_fit_options gfe_c_default_soe_fit_options(void);
GFE_C_API gfe_c_backend_config gfe_c_default_backend_config(void);
GFE_C_API gfe_c_spectral_unit_options gfe_c_default_spectral_unit_options(void);
GFE_C_API gfe_c_abersoe_runtime_config gfe_c_abersoe_default_runtime_config(void);
GFE_C_API gfe_c_hierarchical_runtime_config gfe_c_hierarchical_default_runtime_config(void);

GFE_C_API size_t gfe_c_abersoe_scenario_count(void);

GFE_C_API int gfe_c_abersoe_scenario_name(int scenario,
                                          gfe_c_string_result_view* out,
                                          char* error_msg,
                                          size_t error_msg_capacity);

GFE_C_API int gfe_c_abersoe_run_scenario(int scenario,
                                         double dt,
                                         int form,
                                         const gfe_c_abersoe_runtime_config* cfg,
                                         gfe_c_state_mut_view* final_state,
                                         gfe_c_abersoe_diagnostics* diagnostics,
                                         gfe_c_abersoe_config_record* config,
                                         gfe_c_abersoe_sample_summary_view* samples,
                                         char* error_msg,
                                         size_t error_msg_capacity);

GFE_C_API int gfe_c_abersoe_run_scenario_with_overrides(
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
    size_t error_msg_capacity);

GFE_C_API int gfe_c_abersoe_step(
    int scenario,
    double dt,
    int form,
    const gfe_c_memory_kernel_view* kernel,
    gfe_c_state_mut_view* state,
    char* error_msg,
    size_t error_msg_capacity);

GFE_C_API size_t gfe_c_hierarchical_scenario_count(void);

GFE_C_API int gfe_c_hierarchical_scenario_name(int scenario,
                                               gfe_c_string_result_view* out,
                                               char* error_msg,
                                               size_t error_msg_capacity);

GFE_C_API int gfe_c_hierarchical_run_scenario(int scenario,
                                              int enable_bottom_up,
                                              int enable_top_down,
                                              const gfe_c_hierarchical_runtime_config* cfg,
                                              gfe_c_hierarchical_level_summary_view* levels,
                                              gfe_c_hierarchical_run_diagnostics* diagnostics,
                                              char* error_msg,
                                              size_t error_msg_capacity);

GFE_C_API int gfe_c_hierarchical_get_cross_level_report(int scenario,
                                                        int enable_bottom_up,
                                                        int enable_top_down,
                                                        const gfe_c_hierarchical_runtime_config* cfg,
                                                        gfe_c_hierarchical_kernel_shift_view* kernel_shifts,
                                                        gfe_c_hierarchical_relation_view* relations,
                                                        gfe_c_hierarchical_warning_view* warnings,
                                                        gfe_c_hierarchical_cross_level_contract* contract,
                                                        char* error_msg,
                                                        size_t error_msg_capacity);

GFE_C_API int gfe_c_hierarchical_get_renorm_report(int scenario,
                                                   int enable_bottom_up,
                                                   int enable_top_down,
                                                   gfe_c_hierarchical_renorm_analysis_view* analyses,
                                                   char* error_msg,
                                                   size_t error_msg_capacity);

GFE_C_API int gfe_c_validate_hierarchical_chain_spec(const gfe_c_hierarchical_chain_spec_view* spec,
                                                     char* error_msg,
                                                     size_t error_msg_capacity);

GFE_C_API int gfe_c_hierarchical_run_chain_spec(const gfe_c_hierarchical_chain_spec_view* spec,
                                                const gfe_c_hierarchical_runtime_config* cfg,
                                                gfe_c_hierarchical_level_summary_view* levels,
                                                gfe_c_hierarchical_run_diagnostics* diagnostics,
                                                char* error_msg,
                                                size_t error_msg_capacity);

GFE_C_API int gfe_c_hierarchical_get_cross_level_report_for_chain_spec(
    const gfe_c_hierarchical_chain_spec_view* spec,
    const gfe_c_hierarchical_runtime_config* cfg,
    gfe_c_hierarchical_kernel_shift_view* kernel_shifts,
    gfe_c_hierarchical_relation_view* relations,
    gfe_c_hierarchical_warning_view* warnings,
    gfe_c_hierarchical_cross_level_contract* contract,
    char* error_msg,
    size_t error_msg_capacity);

GFE_C_API int gfe_c_hierarchical_get_renorm_report_for_chain_spec(
    const gfe_c_hierarchical_chain_spec_view* spec,
    gfe_c_hierarchical_renorm_analysis_view* analyses,
    char* error_msg,
    size_t error_msg_capacity);

// Stateful single-step of the hierarchy with external forcing at a specified
// level.  This is the D4B streaming interface: the caller owns the per-level
// state buffers and calls this function once per token event.
//
// ``spec``            -- chain-spec defining the hierarchy (levels, edges).
// ``level_states``    -- array of mutable state views, one per level.
//                         On entry these hold the current (u, chi, t);
//                         on exit they are updated to the post-step values.
// ``level_count``     -- must equal spec.level_count.
// ``external_forcing``-- forcing vector added to level ``forcing_level``
//                         (typically level 0) for this step only.
// ``external_forcing_size`` -- dimension of external_forcing (must match
//                              level forcing_level's state dimension).
// ``forcing_level``   -- which level receives the external forcing (usually 0).
// ``active_kernels``  -- array of mutable kernel views, one per level.
//                         On exit, filled with each level's post-step
//                         active kernel (w, gamma).
// ``active_kernel_count`` -- must equal level_count.
// ``spectral_units``  -- array of spectral-unit structs, one per level.
//                         On exit, filled with each level's post-step
//                         spectral diagnostics.
// ``spectral_count``  -- must equal level_count.
GFE_C_API int gfe_c_hierarchical_step_chain_spec(
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
    size_t error_msg_capacity);

GFE_C_API int gfe_c_validate_memory_kernel(const gfe_c_memory_kernel_view* kernel,
                                           char* error_msg,
                                           size_t error_msg_capacity);

GFE_C_API int gfe_c_spectral_units_from_kernel(const gfe_c_memory_kernel_view* kernel,
                                               const gfe_c_spectral_unit_options* options,
                                               gfe_c_spectral_units* out,
                                               char* error_msg,
                                               size_t error_msg_capacity);

GFE_C_API int gfe_c_fit_backend_name(int backend,
                                     gfe_c_string_result_view* out,
                                     char* error_msg,
                                     size_t error_msg_capacity);

GFE_C_API int gfe_c_compute_backend_name(int backend,
                                         gfe_c_string_result_view* out,
                                         char* error_msg,
                                         size_t error_msg_capacity);

GFE_C_API int gfe_c_hierarchy_run_csv_header(gfe_c_string_result_view* out,
                                             char* error_msg,
                                             size_t error_msg_capacity);

GFE_C_API int gfe_c_hierarchy_summary_csv_header(gfe_c_string_result_view* out,
                                                 char* error_msg,
                                                 size_t error_msg_capacity);

GFE_C_API int gfe_c_hierarchy_diagnostics_csv_header(gfe_c_string_result_view* out,
                                                     char* error_msg,
                                                     size_t error_msg_capacity);

GFE_C_API int gfe_c_hierarchy_cross_level_csv_header(gfe_c_string_result_view* out,
                                                     char* error_msg,
                                                     size_t error_msg_capacity);

GFE_C_API int gfe_c_hierarchy_renorm_csv_header(gfe_c_string_result_view* out,
                                                char* error_msg,
                                                size_t error_msg_capacity);

GFE_C_API int gfe_c_fit_soe_kernel(const double* t,
                                   size_t t_len,
                                   const double* data,
                                   size_t data_len,
                                   const gfe_c_soe_fit_options* options,
                                   const gfe_c_backend_config* backend_cfg,
                                   gfe_c_fit_result_view* out,
                                   char* error_msg,
                                   size_t error_msg_capacity);

#ifdef __cplusplus
} // extern "C"
#endif
