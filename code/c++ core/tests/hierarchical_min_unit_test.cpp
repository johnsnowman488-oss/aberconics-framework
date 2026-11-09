#include "abersoe/hierarchical_min.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace {

abersoe::AberSOEModelSpec make_level(double decay, double forcing0) {
    abersoe::AberSOEModelSpec model;
    model.kernel = gfe::MemoryKernelParams{gfe::Vec{1.0, 0.5}, gfe::Vec{0.6, 0.4}};
    model.dynamics.dt = 0.01;
    model.dynamics.form = gfe::CouplingForm::NegativeFeedback;
    model.dynamics.coupling_index = 0;
    model.operators.apply_L = [decay](const gfe::State& u, double) { return gfe::State{-decay * u[0]}; };
    model.operators.apply_N = [](const gfe::State&, double) { return gfe::State{0.0}; };
    model.operators.forcing = [forcing0](double) { return gfe::State{forcing0}; };
    return model;
}

} // namespace

int main() {
    try {
        {
            if (abersoe::hierarchy_run_csv_header() !=
                    "sample_idx,level_idx,level_name,t,u0,chi0,u_l2,chi_l2,w0,gamma0,Mcap,Mscale,Hmem,Deff" ||
                abersoe::hierarchy_summary_csv_header() !=
                    "level_idx,level_name,t,u0,chi0,u_l2,chi_l2,w0,gamma0,Mcap,Mscale,Hmem,Deff,steps_executed,all_finite" ||
                abersoe::hierarchy_diagnostics_csv_header() !=
                    "level_idx,level_name,steps_executed,all_finite,max_abs_u,max_abs_chi,final_u_l2,final_chi_l2,mean_abs_u,mean_abs_chi,w0,gamma0,Mcap,Mscale,Hmem,Deff" ||
                abersoe::hierarchy_cross_level_csv_header() !=
                    "record_type,relation,source_level,target_level,source_name,target_name,delta_mcap,delta_mscale,delta_hmem,delta_deff,status_a,status_b,severity,code,message,metric,observed,threshold,actionable,suggested_action" ||
                abersoe::hierarchy_renorm_csv_header() !=
                    "input_level,bath_level,direct_level,input_name,bath_name,direct_name,primary_diagnostic,raw_output_channels,output_channels,pruned_channels,retained_weight_fraction,fit_l2_error,pruned_fit_l2_error,fit_linf_error,pruned_fit_linf_error,delta_mcap,delta_mscale,delta_deff,consistency_pass,mcap_pass,mscale_pass,deff_pass" ||
                abersoe::hierarchy_regression_snapshot_csv_header() !=
                    "level_idx,levels,total_state_dim,total_memory_channels,steps_executed,all_finite,final_u_l2,final_chi_l2,first_weight,first_rate,Deff,Mcap,Mscale") {
                std::cerr << "CSV schema headers changed unexpectedly\n";
                return 1;
            }
        }

        {
            const auto names = abersoe::list_hierarchical_scenarios();
            if (names.size() < 2) {
                std::cerr << "hierarchical scenario registry should expose at least two scenarios\n";
                return 1;
            }
            if (abersoe::parse_hierarchical_scenario("three_level_homeostatic") !=
                abersoe::HierarchicalScenario::ThreeLevelHomeostatic) {
                std::cerr << "3-level scenario parse mismatch\n";
                return 1;
            }
        }

        {
            abersoe::HierarchicalRuntimeConfig cfg;
            cfg.sample_stride = 0;
            std::string err;
            if (abersoe::validate_hierarchical_runtime_config(cfg, &err)) {
                std::cerr << "sample_stride=0 should fail validation\n";
                return 1;
            }
        }

        {
            abersoe::HierarchicalChainSpec spec;
            spec.levels.push_back({"custom_fast",
                                   gfe::MemoryKernelParams{gfe::Vec{1.0, 0.5}, gfe::Vec{0.6, 0.4}},
                                   abersoe::AberSOEState{gfe::State{1.0}, gfe::Vec{0.0, 0.0}, 0.0},
                                   0.01,
                                   gfe::State{0.2},
                                   gfe::State{0.0},
                                   gfe::CouplingForm::NegativeFeedback,
                                   0});
            spec.levels.push_back({"custom_slow",
                                   gfe::MemoryKernelParams{gfe::Vec{0.8, 0.2}, gfe::Vec{0.7, 0.3}},
                                   abersoe::AberSOEState{gfe::State{0.2}, gfe::Vec{0.0, 0.0}, 0.0},
                                   0.01,
                                   gfe::State{0.1},
                                   gfe::State{0.05},
                                   gfe::CouplingForm::NegativeFeedback,
                                   0});
            spec.edges.push_back({0, 1, abersoe::HierarchicalChainRelationType::BottomUp, 0.5, true});
            spec.edges.push_back({1, 0, abersoe::HierarchicalChainRelationType::TopDown, 0.25, true});
            std::string err;
            if (!abersoe::validate_hierarchical_chain_spec(spec, &err)) {
                std::cerr << "valid hierarchical chain spec should pass validation: " << err << "\n";
                return 1;
            }
        }

        {
            abersoe::HierarchicalChainSpec spec;
            spec.levels.push_back({"custom_fast",
                                   gfe::MemoryKernelParams{gfe::Vec{0.5, 1.0}, gfe::Vec{0.6, 0.4}},
                                   abersoe::AberSOEState{gfe::State{1.0}, gfe::Vec{0.0, 0.0}, 0.0},
                                   0.01});
            spec.levels.push_back({"custom_slow",
                                   gfe::MemoryKernelParams{gfe::Vec{0.8, 0.2}, gfe::Vec{0.7, 0.3}},
                                   abersoe::AberSOEState{gfe::State{0.2}, gfe::Vec{0.0, 0.0}, 0.0},
                                   0.01});
            std::string err;
            if (abersoe::validate_hierarchical_chain_spec(spec, &err)) {
                std::cerr << "unsorted chain kernel should fail validation\n";
                return 1;
            }
        }

        {
            abersoe::HierarchicalChainSpec spec;
            spec.levels.push_back({"l0",
                                   gfe::MemoryKernelParams{gfe::Vec{1.0, 0.5}, gfe::Vec{0.6, 0.4}},
                                   abersoe::AberSOEState{gfe::State{1.0}, gfe::Vec{0.0, 0.0}, 0.0},
                                   0.01});
            spec.levels.push_back({"l1",
                                   gfe::MemoryKernelParams{gfe::Vec{0.8, 0.2}, gfe::Vec{0.7, 0.3}},
                                   abersoe::AberSOEState{gfe::State{0.2}, gfe::Vec{0.0, 0.0}, 0.0},
                                   0.01});
            spec.levels.push_back({"l2",
                                   gfe::MemoryKernelParams{gfe::Vec{0.6, 0.1}, gfe::Vec{0.8, 0.2}},
                                   abersoe::AberSOEState{gfe::State{0.1}, gfe::Vec{0.0, 0.0}, 0.0},
                                   0.01});
            spec.edges.push_back({0, 2, abersoe::HierarchicalChainRelationType::BottomUp, 1.0, true});
            std::string err;
            if (abersoe::validate_hierarchical_chain_spec(spec, &err)) {
                std::cerr << "non-adjacent chain edge should fail validation\n";
                return 1;
            }
        }

        {
            abersoe::HierarchicalChainSpec spec;
            spec.levels.push_back({"builder_fast",
                                   gfe::MemoryKernelParams{gfe::Vec{1.0, 0.5}, gfe::Vec{0.6, 0.4}},
                                   abersoe::AberSOEState{gfe::State{1.0}, gfe::Vec{0.0, 0.0}, 0.0},
                                   0.01,
                                   gfe::State{0.3},
                                   gfe::State{0.0},
                                   gfe::CouplingForm::NegativeFeedback,
                                   0});
            spec.levels.push_back({"builder_slow",
                                   gfe::MemoryKernelParams{gfe::Vec{0.8, 0.2}, gfe::Vec{0.7, 0.3}},
                                   abersoe::AberSOEState{gfe::State{0.2}, gfe::Vec{0.0, 0.0}, 0.0},
                                   0.01,
                                   gfe::State{0.15},
                                   gfe::State{0.02},
                                   gfe::CouplingForm::NegativeFeedback,
                                   0});
            spec.edges.push_back({0, 1, abersoe::HierarchicalChainRelationType::BottomUp, 0.4, true});
            spec.edges.push_back({1, 0, abersoe::HierarchicalChainRelationType::TopDown, 0.2, true});
            const auto model = abersoe::make_hierarchical_model_from_spec(spec);
            if (model.levels.size() != 2 || model.bottom_up_couplings.size() != 1 || model.top_down_couplings.size() != 1) {
                std::cerr << "generic model builder should preserve level/edge counts\n";
                return 1;
            }
            const auto initial_states = abersoe::make_hierarchical_initial_states_from_spec(spec);
            if (initial_states.size() != spec.levels.size() ||
                initial_states[0].u != spec.levels[0].initial_state.u ||
                initial_states[1].chi != spec.levels[1].initial_state.chi) {
                std::cerr << "initial-state helper should preserve per-level initial states\n";
                return 1;
            }
        }

        {
            abersoe::HierarchicalMINModel model;
            model.levels.push_back({"a", make_level(0.2, 0.0), {}});
            auto b = make_level(0.1, 0.0);
            b.dynamics.dt = 0.02;
            model.levels.push_back({"b", b, {}});
            std::string err;
            if (abersoe::validate_hierarchical_model(model, &err)) {
                std::cerr << "mixed dt hierarchy should fail validation\n";
                return 1;
            }
        }

        {
            abersoe::HierarchicalMINModel model;
            model.levels.push_back({"fast", make_level(0.2, 0.0), {}});
            model.levels.push_back({"slow", make_level(0.1, 0.0), {}});
            abersoe::BottomUpCoupling edge;
            edge.source_level = 0;
            edge.target_level = 1;
            model.bottom_up_couplings.push_back(edge);
            std::string err;
            if (abersoe::validate_hierarchical_model(model, &err)) {
                std::cerr << "missing forcing mapper should fail validation\n";
                return 1;
            }
        }

        {
            abersoe::HierarchicalMINModel model;
            model.levels.push_back({"fast", make_level(0.2, 0.0), {}});
            model.levels.push_back({"slow", make_level(0.1, 0.0), {}});
            abersoe::TopDownCoupling e1;
            e1.source_level = 1;
            e1.target_level = 0;
            e1.modulation = [](const gfe::State&) {
                abersoe::PositiveKernelModulation m;
                m.alpha = gfe::Vec{0.0, 0.0};
                m.beta = gfe::Vec{0.0, -1.0};
                return m;
            };
            auto e2 = e1;
            model.top_down_couplings.push_back(e1);
            model.top_down_couplings.push_back(e2);
            std::string err;
            if (abersoe::validate_hierarchical_model(model, &err)) {
                std::cerr << "multiple top-down edges to same target should fail validation\n";
                return 1;
            }
        }

        {
            abersoe::HierarchicalMINModel default_model;
            default_model.levels.push_back({"fast", make_level(0.2, 0.0), {}});
            default_model.levels.push_back({"slow", make_level(0.1, 0.0), {}});
            abersoe::BottomUpCoupling edge_default;
            edge_default.source_level = 0;
            edge_default.target_level = 1;
            edge_default.forcing = [](const gfe::State& obs, double) { return gfe::State{obs[0]}; };
            default_model.bottom_up_couplings.push_back(edge_default);

            abersoe::HierarchicalMINModel projected_model = default_model;
            projected_model.bottom_up_couplings[0].source_observable = [](const abersoe::AberSOEState& state) {
                return gfe::State{2.0 * state.u[0]};
            };

            std::vector<abersoe::AberSOEState> init(2);
            init[0].u = gfe::State{1.0};
            init[0].chi = gfe::Vec{0.0, 0.0};
            init[1].u = gfe::State{0.0};
            init[1].chi = gfe::Vec{0.0, 0.0};

            const auto default_state = abersoe::initialize_hierarchy_state(default_model, init);
            const auto projected_state = abersoe::initialize_hierarchy_state(projected_model, init);
            const auto default_next = abersoe::step(default_model, default_state);
            const auto projected_next = abersoe::step(projected_model, projected_state);

            if (!(projected_next[1].state.u[0] > default_next[1].state.u[0])) {
                std::cerr << "edge source observable projection should affect bottom-up forcing\n";
                return 1;
            }
        }

        {
            abersoe::HierarchicalMINModel model;
            model.levels.push_back({"fast", make_level(0.2, 0.0), {}});
            model.levels.push_back({"slow", make_level(0.1, 0.0), {}});
            abersoe::TopDownCoupling edge;
            edge.source_level = 1;
            edge.target_level = 0;
            edge.modulation = [](const gfe::State&) {
                abersoe::PositiveKernelModulation m;
                m.alpha = gfe::Vec{0.0};
                m.beta = gfe::Vec{0.0};
                return m;
            };
            model.top_down_couplings.push_back(edge);

            std::vector<abersoe::AberSOEState> init(2);
            init[0].u = gfe::State{1.0};
            init[0].chi = gfe::Vec{0.0, 0.0};
            init[1].u = gfe::State{0.1};
            init[1].chi = gfe::Vec{0.0, 0.0};
            const auto state0 = abersoe::initialize_hierarchy_state(model, init);
            bool threw = false;
            try {
                (void)abersoe::step(model, state0);
            } catch (const std::invalid_argument&) {
                threw = true;
            }
            if (!threw) {
                std::cerr << "top-down channel-count mismatch should throw\n";
                return 1;
            }
        }

        {
            const auto model = abersoe::make_hierarchical_scenario_model(
                abersoe::HierarchicalScenario::ThreeLevelHomeostatic, true, true);
            const auto initial_states =
                abersoe::make_hierarchical_scenario_initial_states(abersoe::HierarchicalScenario::ThreeLevelHomeostatic);
            abersoe::HierarchicalRuntimeConfig cfg;
            cfg.steps = 24;
            cfg.sample_stride = 6;
            const auto state0 = abersoe::initialize_hierarchy_state(model, initial_states);
            const auto run = abersoe::run(model, state0, cfg);
            const auto cross_level = abersoe::make_hierarchical_cross_level_report(model, run);
            if (cross_level.kernel_shifts.size() != model.levels.size()) {
                std::cerr << "cross-level report should include one kernel shift per level\n";
                return 1;
            }
            if (cross_level.relations.size() !=
                model.bottom_up_couplings.size() + model.top_down_couplings.size()) {
                std::cerr << "cross-level report should include one relation per configured coupling\n";
                return 1;
            }
            if (!cross_level.contract.pass) {
                std::cerr << "reference 3-level cross-level contract should pass\n";
                return 1;
            }
            const auto cross_snapshot = abersoe::make_cross_level_regression_snapshot(cross_level);
            if (cross_snapshot.warning_codes.size() != cross_level.warnings.size()) {
                std::cerr << "cross-level regression snapshot should preserve warning-code shape\n";
                return 1;
            }
            bool saw_actionable = false;
            for (const auto& warning : cross_level.warnings) {
                if (warning.actionable) {
                    saw_actionable = true;
                    if (warning.suggested_action.empty() || warning.metric.empty()) {
                        std::cerr << "actionable warnings should carry metric and suggested action\n";
                        return 1;
                    }
                }
            }
            if (!saw_actionable) {
                std::cerr << "cross-level report should emit at least one actionable warning for the reference scenario\n";
                return 1;
            }

            const std::string cross_path =
                (std::filesystem::temp_directory_path() / "hierarchical_cross_level_schema_test.csv").string();
            abersoe::write_hierarchy_cross_level_csv(cross_path, cross_level);
            std::ifstream in(cross_path);
            std::string header;
            std::getline(in, header);
            if (header != abersoe::hierarchy_cross_level_csv_header()) {
                std::cerr << "cross-level CSV header should match the published schema header\n";
                return 1;
            }
            const auto renorm = abersoe::make_hierarchical_adjacent_renorm_analyses(model);
            if (renorm.size() != model.levels.size() - 1) {
                std::cerr << "adjacent renorm analyses should cover neighboring level pairs\n";
                return 1;
            }
            const auto renorm_snapshot = abersoe::make_renorm_regression_snapshot(renorm);
            if (renorm_snapshot.analysis_count != renorm.size() ||
                renorm_snapshot.diagnostic_names.size() != renorm.size()) {
                std::cerr << "renorm regression snapshot should preserve analysis shape\n";
                return 1;
            }
            bool saw_named_diagnostic = false;
            for (const auto& analysis : renorm) {
                const auto name =
                    abersoe::hierarchical_renorm_diagnostic_cause_name(analysis.report.primary_diagnostic);
                if (!name.empty() && name != "unknown") {
                    saw_named_diagnostic = true;
                }
            }
            if (!saw_named_diagnostic) {
                std::cerr << "renorm analyses should expose named diagnostic causes\n";
                return 1;
            }

            const std::string renorm_path =
                (std::filesystem::temp_directory_path() / "hierarchical_renorm_schema_test.csv").string();
            abersoe::write_hierarchy_renorm_csv(renorm_path, renorm);
            std::ifstream renorm_in(renorm_path);
            std::string renorm_header;
            std::getline(renorm_in, renorm_header);
            if (renorm_header != abersoe::hierarchy_renorm_csv_header()) {
                std::cerr << "renorm CSV header should match the published schema header\n";
                return 1;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "hierarchical_min_unit_test failed: " << e.what() << "\n";
        return 1;
    }

    std::cout << "hierarchical_min_unit_test passed\n";
    return 0;
}
