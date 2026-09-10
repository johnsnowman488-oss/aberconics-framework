#include "abersoe/hierarchical_min.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>

namespace abersoe {
namespace {

bool finite_vec(const std::vector<double>& x) {
    for (double v : x) {
        if (!std::isfinite(v)) {
            return false;
        }
    }
    return true;
}

gfe::State zeros_like(std::size_t n) {
    return gfe::State(n, 0.0);
}

gfe::State ones_like(std::size_t n) {
    return gfe::State(n, 1.0);
}

HierarchicalObservable default_observable(const AberSOEState& state) {
    return state.u;
}

HierarchicalObservable project_observable(
    const AberSOEState& state,
    const std::function<HierarchicalObservable(const AberSOEState&)>& projector) {
    if (projector) {
        return projector(state);
    }
    return default_observable(state);
}

gfe::SpectralUnits spectral_from_kernel(const gfe::MemoryKernelParams& kernel) {
    return gfe::spectral_units(kernel.w, kernel.gamma);
}

void ensure_matching_dimensions(const gfe::State& actual, std::size_t expected, const std::string& name) {
    if (actual.size() != expected) {
        throw std::invalid_argument(name + " returned invalid state dimension");
    }
}

void validate_level_index(std::size_t idx, std::size_t n_levels, const std::string& label, std::string* error) {
    if (idx >= n_levels && error) {
        *error = label + " level index out of range";
    }
}

HierarchicalState normalize_state(const HierarchicalMINModel& model, const HierarchicalState& state) {
    if (state.size() != model.levels.size()) {
        throw std::invalid_argument("hierarchy state size must match number of levels");
    }

    HierarchicalState out = state;
    for (std::size_t i = 0; i < out.size(); ++i) {
        auto& level = out[i];
        if (level.active_kernel.gamma.empty() || level.active_kernel.w.empty()) {
            level.active_kernel = model.levels[i].model.kernel;
        }
        if (level.state.chi.empty()) {
            level.state.chi.assign(level.active_kernel.gamma.size(), 0.0);
        }
        if (level.state.chi.size() != level.active_kernel.gamma.size()) {
            throw std::invalid_argument("level state chi size must match active kernel channel count");
        }
        level.spectral = spectral_from_kernel(level.active_kernel);
    }
    return out;
}

} // namespace

bool validate_hierarchical_model(const HierarchicalMINModel& model, std::string* error) {
    if (model.levels.empty()) {
        if (error) {
            *error = "hierarchy must contain at least one level";
        }
        return false;
    }

    const double dt0 = model.levels.front().model.dynamics.dt;
    for (std::size_t i = 0; i < model.levels.size(); ++i) {
        std::string level_error;
        if (!validate_model_spec(model.levels[i].model, &level_error)) {
            if (error) {
                *error = "invalid level[" + std::to_string(i) + "]: " + level_error;
            }
            return false;
        }
        if (!model.levels[i].model.operators.apply_L) {
            if (error) {
                *error = "level[" + std::to_string(i) + "] must define operators.apply_L";
            }
            return false;
        }
        if (std::fabs(model.levels[i].model.dynamics.dt - dt0) > 1e-12) {
            if (error) {
                *error = "all hierarchy levels must currently share the same dynamics.dt";
            }
            return false;
        }
    }

    std::vector<std::size_t> top_down_targets(model.levels.size(), 0);
    for (const auto& edge : model.bottom_up_couplings) {
        if (edge.source_level >= model.levels.size()) {
            validate_level_index(edge.source_level, model.levels.size(), "bottom-up source", error);
            return false;
        }
        if (edge.target_level >= model.levels.size()) {
            validate_level_index(edge.target_level, model.levels.size(), "bottom-up target", error);
            return false;
        }
        if (edge.source_level == edge.target_level) {
            if (error) {
                *error = "bottom-up coupling cannot target the same level";
            }
            return false;
        }
        if (!edge.forcing) {
            if (error) {
                *error = "bottom-up coupling must define a forcing mapper";
            }
            return false;
        }
    }

    for (const auto& edge : model.top_down_couplings) {
        if (edge.source_level >= model.levels.size()) {
            validate_level_index(edge.source_level, model.levels.size(), "top-down source", error);
            return false;
        }
        if (edge.target_level >= model.levels.size()) {
            validate_level_index(edge.target_level, model.levels.size(), "top-down target", error);
            return false;
        }
        if (edge.source_level == edge.target_level) {
            if (error) {
                *error = "top-down coupling cannot target the same level";
            }
            return false;
        }
        if (!edge.modulation) {
            if (error) {
                *error = "top-down coupling must define a modulation mapper";
            }
            return false;
        }
        top_down_targets[edge.target_level] += 1;
        if (top_down_targets[edge.target_level] > 1) {
            if (error) {
                *error = "phase-1 hierarchy supports at most one top-down coupling per target level";
            }
            return false;
        }
    }

    return true;
}

bool validate_hierarchical_chain_spec(const HierarchicalChainSpec& spec, std::string* error) {
    if (spec.levels.empty()) {
        if (error) {
            *error = "hierarchical chain spec must contain at least one level";
        }
        return false;
    }

    const double dt0 = spec.levels.front().dt;
    for (std::size_t i = 0; i < spec.levels.size(); ++i) {
        const auto& level = spec.levels[i];
        if (level.initial_state.u.empty()) {
            if (error) {
                *error = "chain level[" + std::to_string(i) + "] must define a non-empty initial_state.u";
            }
            return false;
        }
        if (!std::isfinite(level.dt) || level.dt <= 0.0) {
            if (error) {
                *error = "chain level[" + std::to_string(i) + "] must define a positive finite dt";
            }
            return false;
        }
        if (std::fabs(level.dt - dt0) > 1e-12) {
            if (error) {
                *error = "all chain levels must currently share the same dt";
            }
            return false;
        }
        if (level.kernel.gamma.empty() || level.kernel.w.empty()) {
            if (error) {
                *error = "chain level[" + std::to_string(i) + "] must define a non-empty kernel";
            }
            return false;
        }
        if (level.kernel.gamma.size() != level.kernel.w.size()) {
            if (error) {
                *error = "chain level[" + std::to_string(i) + "] kernel gamma/w size mismatch";
            }
            return false;
        }
        for (std::size_t k = 0; k < level.kernel.gamma.size(); ++k) {
            if (!std::isfinite(level.kernel.gamma[k]) || level.kernel.gamma[k] <= 0.0 ||
                !std::isfinite(level.kernel.w[k]) || level.kernel.w[k] < 0.0) {
                if (error) {
                    *error = "chain level[" + std::to_string(i) + "] kernel entries must be finite with positive gamma and non-negative w";
                }
                return false;
            }
        }
        for (std::size_t k = 1; k < level.kernel.gamma.size(); ++k) {
            if (level.kernel.gamma[k - 1] < level.kernel.gamma[k]) {
                if (error) {
                    *error = "chain level[" + std::to_string(i) + "] kernel gamma must be sorted in descending order";
                }
                return false;
            }
        }
        if (!level.initial_state.chi.empty() && level.initial_state.chi.size() != level.kernel.gamma.size()) {
            if (error) {
                *error = "chain level[" + std::to_string(i) + "] initial_state.chi must match kernel channel count when provided";
            }
            return false;
        }
        if (!finite_vec(level.initial_state.u) || !finite_vec(level.initial_state.chi) ||
            !std::isfinite(level.initial_state.t)) {
            if (error) {
                *error = "chain level[" + std::to_string(i) + "] initial state must be finite";
            }
            return false;
        }
        if (!level.linear_decay.empty() && level.linear_decay.size() != level.initial_state.u.size()) {
            if (error) {
                *error = "chain level[" + std::to_string(i) + "] linear_decay must match state dimension when provided";
            }
            return false;
        }
        if (!level.forcing_bias.empty() && level.forcing_bias.size() != level.initial_state.u.size()) {
            if (error) {
                *error = "chain level[" + std::to_string(i) + "] forcing_bias must match state dimension when provided";
            }
            return false;
        }
        if ((!level.linear_decay.empty() && !finite_vec(level.linear_decay)) ||
            (!level.forcing_bias.empty() && !finite_vec(level.forcing_bias))) {
            if (error) {
                *error = "chain level[" + std::to_string(i) + "] linear_decay and forcing_bias must be finite when provided";
            }
            return false;
        }
        if (level.coupling_index >= level.initial_state.u.size()) {
            if (error) {
                *error = "chain level[" + std::to_string(i) + "] coupling_index must be within state dimension";
            }
            return false;
        }
    }

    std::set<std::tuple<std::size_t, std::size_t, HierarchicalChainRelationType>> seen_edges;
    for (std::size_t i = 0; i < spec.edges.size(); ++i) {
        const auto& edge = spec.edges[i];
        if (edge.source_level >= spec.levels.size() || edge.target_level >= spec.levels.size()) {
            if (error) {
                *error = "chain edge[" + std::to_string(i) + "] level index out of range";
            }
            return false;
        }
        if (edge.source_level == edge.target_level) {
            if (error) {
                *error = "chain edge[" + std::to_string(i) + "] cannot connect a level to itself";
            }
            return false;
        }
        const std::size_t hi = std::max(edge.source_level, edge.target_level);
        const std::size_t lo = std::min(edge.source_level, edge.target_level);
        if (hi - lo != 1) {
            if (error) {
                *error = "chain edge[" + std::to_string(i) + "] must connect adjacent levels";
            }
            return false;
        }
        if (edge.relation == HierarchicalChainRelationType::BottomUp &&
            edge.target_level != edge.source_level + 1) {
            if (error) {
                *error = "chain bottom-up edge[" + std::to_string(i) + "] must point from level n to level n+1";
            }
            return false;
        }
        if (edge.relation == HierarchicalChainRelationType::TopDown &&
            edge.source_level != edge.target_level + 1) {
            if (error) {
                *error = "chain top-down edge[" + std::to_string(i) + "] must point from level n to level n-1";
            }
            return false;
        }
        if (!std::isfinite(edge.gain)) {
            if (error) {
                *error = "chain edge[" + std::to_string(i) + "] gain must be finite";
            }
            return false;
        }
        const std::size_t source_dim = spec.levels[edge.source_level].initial_state.u.size();
        const std::size_t target_dim = spec.levels[edge.target_level].initial_state.u.size();
        if (edge.relation == HierarchicalChainRelationType::BottomUp &&
            !(source_dim == target_dim || source_dim == 1)) {
            if (error) {
                *error = "chain bottom-up edge[" + std::to_string(i) + "] requires source_dim == target_dim or source_dim == 1";
            }
            return false;
        }
        const auto key = std::make_tuple(edge.source_level, edge.target_level, edge.relation);
        if (!seen_edges.insert(key).second) {
            if (error) {
                *error = "chain edge[" + std::to_string(i) + "] duplicates an existing relation";
            }
            return false;
        }
    }

    return true;
}

HierarchicalMINModel make_hierarchical_model_from_spec(const HierarchicalChainSpec& spec) {
    std::string error;
    if (!validate_hierarchical_chain_spec(spec, &error)) {
        throw std::invalid_argument("Invalid hierarchical chain spec: " + error);
    }

    HierarchicalMINModel model;
    model.levels.reserve(spec.levels.size());
    for (std::size_t i = 0; i < spec.levels.size(); ++i) {
        const auto& level = spec.levels[i];
        const std::size_t state_dim = level.initial_state.u.size();
        const gfe::State linear_decay = level.linear_decay.empty() ? ones_like(state_dim) : level.linear_decay;
        const gfe::State forcing_bias = level.forcing_bias.empty() ? zeros_like(state_dim) : level.forcing_bias;

        AberSOEModelSpec model_spec;
        model_spec.kernel = level.kernel;
        model_spec.dynamics.dt = level.dt;
        model_spec.dynamics.form = level.form;
        model_spec.dynamics.coupling_index = level.coupling_index;
        model_spec.operators.apply_L = [linear_decay](const gfe::State& u, double) {
            gfe::State out(u.size(), 0.0);
            for (std::size_t k = 0; k < u.size(); ++k) {
                out[k] = -linear_decay[k] * u[k];
            }
            return out;
        };
        model_spec.operators.apply_N = [](const gfe::State& u, double) { return gfe::State(u.size(), 0.0); };
        model_spec.operators.forcing = [forcing_bias](double) { return forcing_bias; };
        model.levels.push_back({level.name, model_spec, {}});
    }

    for (const auto& edge : spec.edges) {
        if (edge.relation == HierarchicalChainRelationType::BottomUp) {
            BottomUpCoupling coupling;
            coupling.source_level = edge.source_level;
            coupling.target_level = edge.target_level;
            const std::size_t target_dim = spec.levels[edge.target_level].initial_state.u.size();
            coupling.forcing = [gain = edge.gain, target_dim](const HierarchicalObservable& obs, double) {
                if (obs.size() == target_dim) {
                    gfe::State out(target_dim, 0.0);
                    for (std::size_t k = 0; k < target_dim; ++k) {
                        out[k] = gain * obs[k];
                    }
                    return out;
                }
                if (obs.size() == 1) {
                    return gfe::State(target_dim, gain * obs[0]);
                }
                throw std::invalid_argument("generic bottom-up mapper received incompatible observable dimension");
            };
            model.bottom_up_couplings.push_back(coupling);
        } else {
            TopDownCoupling coupling;
            coupling.source_level = edge.source_level;
            coupling.target_level = edge.target_level;
            const auto base_kernel = spec.levels[edge.target_level].kernel;
            coupling.modulation = [gain = edge.gain, normalize = edge.normalize_weights, base_kernel](
                                      const HierarchicalObservable& obs) {
                double drive = 0.0;
                for (double v : obs) {
                    drive += std::fabs(v);
                }
                drive /= static_cast<double>(obs.empty() ? 1 : obs.size());
                PositiveKernelModulation m;
                m.normalize_weights = normalize;
                m.alpha.resize(base_kernel.w.size());
                m.beta.resize(base_kernel.gamma.size());
                for (std::size_t k = 0; k < base_kernel.w.size(); ++k) {
                    const double base_w = std::max(base_kernel.w[k], 1e-12);
                    m.alpha[k] = std::log(base_w) + gain * drive;
                    m.beta[k] = std::log(base_kernel.gamma[k]) - gain * drive;
                }
                return m;
            };
            model.top_down_couplings.push_back(coupling);
        }
    }

    return model;
}

std::vector<AberSOEState> make_hierarchical_initial_states_from_spec(const HierarchicalChainSpec& spec) {
    std::string error;
    if (!validate_hierarchical_chain_spec(spec, &error)) {
        throw std::invalid_argument("Invalid hierarchical chain spec: " + error);
    }

    std::vector<AberSOEState> out;
    out.reserve(spec.levels.size());
    for (const auto& level : spec.levels) {
        out.push_back(level.initial_state);
    }
    return out;
}

bool validate_hierarchical_runtime_config(const HierarchicalRuntimeConfig& cfg, std::string* error) {
    if (cfg.sample_stride == 0) {
        if (error) {
            *error = "sample_stride must be >= 1";
        }
        return false;
    }
    return true;
}

gfe::MemoryKernelParams materialize_positive_kernel(const PositiveKernelModulation& modulation) {
    if (modulation.alpha.empty() || modulation.beta.empty()) {
        throw std::invalid_argument("kernel modulation alpha and beta must be non-empty");
    }
    if (modulation.alpha.size() != modulation.beta.size()) {
        throw std::invalid_argument("kernel modulation alpha and beta must have the same size");
    }

    gfe::MemoryKernelParams out;
    out.gamma.resize(modulation.beta.size());
    out.w.resize(modulation.alpha.size());

    double w_sum = 0.0;
    for (std::size_t i = 0; i < modulation.alpha.size(); ++i) {
        out.w[i] = std::exp(modulation.alpha[i]);
        out.gamma[i] = std::exp(modulation.beta[i]);
        w_sum += out.w[i];
    }

    if (modulation.normalize_weights) {
        for (double& w : out.w) {
            w /= w_sum;
        }
    }

    std::vector<std::size_t> order(out.gamma.size(), 0);
    for (std::size_t i = 0; i < order.size(); ++i) {
        order[i] = i;
    }
    std::sort(order.begin(), order.end(), [&out](std::size_t a, std::size_t b) {
        return out.gamma[a] > out.gamma[b];
    });

    gfe::MemoryKernelParams sorted;
    sorted.gamma.reserve(order.size());
    sorted.w.reserve(order.size());
    for (std::size_t idx : order) {
        sorted.gamma.push_back(out.gamma[idx]);
        sorted.w.push_back(out.w[idx]);
    }
    return sorted;
}

HierarchicalState initialize_hierarchy_state(
    const HierarchicalMINModel& model,
    const std::vector<AberSOEState>& initial_states) {
    if (initial_states.size() != model.levels.size()) {
        throw std::invalid_argument("initial_states size must match number of hierarchy levels");
    }

    HierarchicalState out(model.levels.size());
    for (std::size_t i = 0; i < initial_states.size(); ++i) {
        out[i].state = initial_states[i];
        out[i].active_kernel = model.levels[i].model.kernel;
        if (out[i].state.chi.empty()) {
            out[i].state.chi.assign(out[i].active_kernel.gamma.size(), 0.0);
        }
        out[i].spectral = spectral_from_kernel(out[i].active_kernel);
    }
    return out;
}

HierarchicalState step(const HierarchicalMINModel& model, const HierarchicalState& state) {
    std::string error;
    if (!validate_hierarchical_model(model, &error)) {
        throw std::invalid_argument("Invalid hierarchical model: " + error);
    }

    const HierarchicalState current = normalize_state(model, state);
    std::vector<HierarchicalObservable> observables(current.size());
    for (std::size_t i = 0; i < current.size(); ++i) {
        observables[i] = project_observable(current[i].state, model.levels[i].observable);
    }

    HierarchicalState next(current.size());
    for (std::size_t i = 0; i < current.size(); ++i) {
        AberSOEModelSpec local_model = model.levels[i].model;
        gfe::MemoryKernelParams active_kernel = current[i].active_kernel;

        for (const auto& edge : model.top_down_couplings) {
            if (edge.target_level != i) {
                continue;
            }
            const HierarchicalObservable source_obs =
                project_observable(current[edge.source_level].state, edge.source_observable);
            active_kernel = materialize_positive_kernel(edge.modulation(source_obs));
            if (active_kernel.gamma.size() != current[i].state.chi.size()) {
                throw std::invalid_argument("top-down modulation must preserve target memory channel count in phase 1");
            }
        }
        local_model.kernel = active_kernel;

        const auto base_forcing = local_model.operators.forcing;
        const std::size_t target_dim = current[i].state.u.size();
        local_model.operators.forcing = [&, i, base_forcing, target_dim](double t) {
            gfe::State total = base_forcing ? base_forcing(t) : gfe::State(target_dim, 0.0);
            ensure_matching_dimensions(total, target_dim, "base forcing");
            for (const auto& edge : model.bottom_up_couplings) {
                if (edge.target_level != i) {
                    continue;
                }
                const HierarchicalObservable source_obs =
                    project_observable(current[edge.source_level].state, edge.source_observable);
                gfe::State contribution = edge.forcing(source_obs, t);
                ensure_matching_dimensions(contribution, target_dim, "bottom-up forcing");
                for (std::size_t k = 0; k < target_dim; ++k) {
                    total[k] += contribution[k];
                }
            }
            return total;
        };

        next[i].state = abersoe::step(local_model, current[i].state);
        next[i].active_kernel = local_model.kernel;
        next[i].spectral = spectral_from_kernel(next[i].active_kernel);
    }
    return next;
}

HierarchicalState step_with_external_forcing(
    const HierarchicalMINModel& model,
    const HierarchicalState& state,
    std::size_t forcing_level,
    const gfe::State& external_forcing) {
    std::string error;
    if (!validate_hierarchical_model(model, &error)) {
        throw std::invalid_argument("Invalid hierarchical model: " + error);
    }
    if (forcing_level >= model.levels.size()) {
        throw std::invalid_argument("forcing_level out of range");
    }

    // Build a shallow-copied model with the target level's forcing overridden.
    HierarchicalMINModel mod_model = model;
    const auto original_forcing = mod_model.levels[forcing_level].model.operators.forcing;
    mod_model.levels[forcing_level].model.operators.forcing =
        [original_forcing, external_forcing](double t) {
            gfe::State total = original_forcing ? original_forcing(t)
                                                : gfe::State(external_forcing.size(), 0.0);
            for (std::size_t k = 0; k < external_forcing.size() && k < total.size(); ++k) {
                total[k] += external_forcing[k];
            }
            return total;
        };

    return step(mod_model, state);
}

HierarchicalRunResult run(
    const HierarchicalMINModel& model,
    const HierarchicalState& state0,
    const HierarchicalRuntimeConfig& cfg) {
    std::string error;
    if (!validate_hierarchical_model(model, &error)) {
        throw std::invalid_argument("Invalid hierarchical model: " + error);
    }
    if (!validate_hierarchical_runtime_config(cfg, &error)) {
        throw std::invalid_argument("Invalid hierarchical runtime config: " + error);
    }

    HierarchicalRunResult out;
    HierarchicalState state = normalize_state(model, state0);
    out.diagnostics.all_finite = true;
    out.diagnostics.per_level.assign(model.levels.size(), AberSOEDiagnostics{});
    out.samples.push_back(HierarchicalSnapshot{state});

    for (std::size_t step_idx = 0; step_idx < cfg.steps; ++step_idx) {
        state = step(model, state);
        out.diagnostics.steps_executed += 1;
        for (std::size_t level_idx = 0; level_idx < state.size(); ++level_idx) {
            update_diagnostics(out.diagnostics.per_level[level_idx], state[level_idx].state.u, state[level_idx].state.chi);
            const bool level_finite =
                finite_vec(state[level_idx].state.u) &&
                finite_vec(state[level_idx].state.chi) &&
                std::isfinite(state[level_idx].state.t) &&
                std::isfinite(state[level_idx].spectral.Mcap) &&
                std::isfinite(state[level_idx].spectral.Mscale) &&
                std::isfinite(state[level_idx].spectral.Hmem) &&
                std::isfinite(state[level_idx].spectral.Hnorm) &&
                std::isfinite(state[level_idx].spectral.Deff);
            out.diagnostics.all_finite = out.diagnostics.all_finite && level_finite;
        }
        if (cfg.strict_finite && !out.diagnostics.all_finite) {
            throw std::runtime_error("hierarchical run became non-finite");
        }
        if ((step_idx + 1) % cfg.sample_stride == 0) {
            out.samples.push_back(HierarchicalSnapshot{state});
        }
    }

    out.final_levels = state;
    return out;
}

} // namespace abersoe
