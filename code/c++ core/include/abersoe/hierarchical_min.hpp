#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

#include "abersoe/abersoe_runtime.hpp"

namespace abersoe {

using HierarchicalObservable = gfe::State;

struct HierarchicalLevelSpec {
    std::string name;
    AberSOEModelSpec model;
    std::function<HierarchicalObservable(const AberSOEState&)> observable;
};

struct HierarchicalLevelState {
    AberSOEState state;
    gfe::MemoryKernelParams active_kernel;
    gfe::SpectralUnits spectral{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
};

using HierarchicalState = std::vector<HierarchicalLevelState>;

struct PositiveKernelModulation {
    gfe::Vec alpha;
    gfe::Vec beta;
    bool normalize_weights = true;
};

struct BottomUpCoupling {
    std::size_t source_level = 0;
    std::size_t target_level = 0;
    std::function<HierarchicalObservable(const AberSOEState&)> source_observable;
    std::function<gfe::State(const HierarchicalObservable&, double)> forcing;
};

struct TopDownCoupling {
    std::size_t source_level = 0;
    std::size_t target_level = 0;
    std::function<HierarchicalObservable(const AberSOEState&)> source_observable;
    std::function<PositiveKernelModulation(const HierarchicalObservable&)> modulation;
};

struct HierarchicalMINModel {
    std::vector<HierarchicalLevelSpec> levels;
    std::vector<BottomUpCoupling> bottom_up_couplings;
    std::vector<TopDownCoupling> top_down_couplings;
};

enum class HierarchicalChainRelationType : int {
    BottomUp,
    TopDown
};

struct HierarchicalChainLevelSpec {
    std::string name;
    gfe::MemoryKernelParams kernel;
    AberSOEState initial_state;
    double dt = 0.01;
    gfe::State linear_decay;
    gfe::State forcing_bias;
    gfe::CouplingForm form = gfe::CouplingForm::NegativeFeedback;
    std::size_t coupling_index = 0;
};

struct HierarchicalChainEdgeSpec {
    std::size_t source_level = 0;
    std::size_t target_level = 0;
    HierarchicalChainRelationType relation = HierarchicalChainRelationType::BottomUp;
    double gain = 1.0;
    bool normalize_weights = true;
};

struct HierarchicalChainSpec {
    std::vector<HierarchicalChainLevelSpec> levels;
    std::vector<HierarchicalChainEdgeSpec> edges;
};

struct HierarchicalRuntimeConfig {
    std::size_t steps = 1000;
    std::size_t sample_stride = 1;
    bool strict_finite = true;
};

struct HierarchicalDiagnostics {
    std::size_t steps_executed = 0;
    bool all_finite = true;
    std::vector<AberSOEDiagnostics> per_level;
};

struct HierarchicalSnapshot {
    HierarchicalState levels;
};

struct HierarchicalRunResult {
    HierarchicalState final_levels;
    std::vector<HierarchicalSnapshot> samples;
    HierarchicalDiagnostics diagnostics;
};

enum class HierarchicalScenario : int {
    TwoLevelBidirectional,
    ThreeLevelHomeostatic
};

bool validate_hierarchical_model(const HierarchicalMINModel& model, std::string* error = nullptr);
bool validate_hierarchical_chain_spec(const HierarchicalChainSpec& spec, std::string* error = nullptr);
bool validate_hierarchical_runtime_config(const HierarchicalRuntimeConfig& cfg, std::string* error = nullptr);
HierarchicalMINModel make_hierarchical_model_from_spec(const HierarchicalChainSpec& spec);
std::vector<AberSOEState> make_hierarchical_initial_states_from_spec(const HierarchicalChainSpec& spec);

gfe::MemoryKernelParams materialize_positive_kernel(const PositiveKernelModulation& modulation);
HierarchicalState initialize_hierarchy_state(
    const HierarchicalMINModel& model,
    const std::vector<AberSOEState>& initial_states);

HierarchicalState step(const HierarchicalMINModel& model, const HierarchicalState& state);

// Stateful single-step with external forcing injected at a specific level.
// ``external_forcing`` is added to level ``forcing_level``'s base forcing
// for this step only; other levels are stepped as usual.
HierarchicalState step_with_external_forcing(
    const HierarchicalMINModel& model,
    const HierarchicalState& state,
    std::size_t forcing_level,
    const gfe::State& external_forcing);

HierarchicalRunResult run(
    const HierarchicalMINModel& model,
    const HierarchicalState& state0,
    const HierarchicalRuntimeConfig& cfg = {});

std::vector<std::string> list_hierarchical_scenarios();
std::string hierarchical_scenario_name(HierarchicalScenario scenario);
HierarchicalScenario parse_hierarchical_scenario(const std::string& name);
HierarchicalMINModel make_hierarchical_scenario_model(
    HierarchicalScenario scenario,
    bool enable_bottom_up = true,
    bool enable_top_down = true);
std::vector<AberSOEState> make_hierarchical_scenario_initial_states(HierarchicalScenario scenario);

} // namespace abersoe

#include "abersoe/hierarchical_min_coupling.hpp"
#include "abersoe/hierarchical_min_diagnostics.hpp"
#include "abersoe/hierarchical_min_renorm.hpp"
