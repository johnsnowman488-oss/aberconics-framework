from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from d2c.digital import (
    DigitalMemoryConfig,
    DigitalTrace,
    KeyValueBindingBridge,
    SymbolicInductionConfig,
    TokenForcingBridge,
    Vocabulary,
    build_forcing_schedule,
    build_symbolic_induction_vocabulary,
    generate_symbolic_induction_example,
    regular_token_stream,
    seed_memory_state,
    step_memory,
)
from d2c.digital.readout import decode_key_value_binding
from d2c.experiments.symbolic_induction import (
    SymbolicInductionRunConfig,
    run_symbolic_induction_experiment,
    save_symbolic_induction_bundle,
)
from d2c.experiments.thinking_between_tokens import (
    ThinkingBetweenTokensConfig,
    run_thinking_between_tokens_experiment,
)
from d2c.experiments.learned_symbolic_retrieval import LearnedRetrievalConfig, run_learned_symbolic_retrieval
from d2c.experiments.temporal_logic import TemporalLogicConfig, run_temporal_logic_experiment, run_temporal_logic_sweep
from d2c.experiments.d3_randomized import D3RandomizedConfig, run_d3_randomized_experiment


def test_vocabulary_stream_and_forcing_schedule_are_deterministic():
    vocab = Vocabulary(["A", "B", "F"])
    bridge = TokenForcingBridge(vocabulary=vocab, dim=4, seed=3)
    stream = regular_token_stream(["A", "B"], dt=0.5, target_token="B")
    schedule = build_forcing_schedule(stream, bridge, pulse_steps=2, silence_steps=1)

    assert vocab.token_id("B") == 1
    assert stream.tokens() == ["A", "B"]
    assert len(schedule.forcings) == 6
    assert schedule.tokens == ["A", "A", None, "B", "B", None]
    assert bridge.forcing_for_token("A") == bridge.forcing_for_token("A")


def test_digital_memory_step_and_trace_roundtrip():
    cfg = DigitalMemoryConfig(gamma=[1.0, 0.1], w=[0.2, 0.4], dt=0.05, leak_rate=2.0)
    state = seed_memory_state(state_dim=3, channel_count=2)
    next_state = step_memory(state, [1.0, 0.0, -1.0], cfg)

    assert next_state.t == 0.05
    assert next_state.u != state.u
    assert cfg.deff() > 1.0
    assert cfg.stability_ratio() < 3.0

    trace = DigitalTrace(experiment_name="roundtrip")
    data = trace.to_mapping()
    restored = DigitalTrace.from_mapping(data)
    assert restored.experiment_name == "roundtrip"


def test_key_value_binding_readout_is_query_conditioned():
    cfg = DigitalMemoryConfig(gamma=[1.0], w=[0.2], dt=0.05, leak_rate=2.0)
    binding = KeyValueBindingBridge(key_tokens=["K0", "K1"], value_tokens=["V0", "V1"])
    state = seed_memory_state(state_dim=binding.state_dim, channel_count=1)
    state.u = binding.binding_for_pair("K0", "V1")

    predicted, score = decode_key_value_binding(state, cfg, binding, "K0")

    assert predicted == "V1"
    assert score > 0.0


def test_symbolic_induction_task_generation():
    cfg = SymbolicInductionConfig(
        key_count=3,
        value_count=3,
        filler_count=1,
        pair_count=2,
        distractor_pair_count=2,
        gap=5,
        seed=11,
    )
    vocab = build_symbolic_induction_vocabulary(cfg)
    example = generate_symbolic_induction_example(cfg)

    assert len(vocab.tokens) == 7
    assert example.stream.tokens()[:4] == [token for pair in example.pairs for token in pair]
    assert example.stream.tokens()[-1] == example.key_token
    assert example.stream.target_token == example.value_token
    assert len(example.pairs) == 2
    assert len(example.distractor_pairs) == 2
    assert all(key != example.key_token for key, _ in example.distractor_pairs)
    assert dict(example.pairs)[example.key_token] == example.value_token


def test_symbolic_induction_experiment_and_bundle(tmp_path):
    cfg = SymbolicInductionRunConfig(
        task=SymbolicInductionConfig(key_count=4, value_count=4, filler_count=2, pair_count=3, gap=8, seed=5),
        variant="full",
        seed_count=4,
        baseline_window=3,
        binding_noise_std=0.02,
    )
    result = run_symbolic_induction_experiment(cfg)
    paths = save_symbolic_induction_bundle(result, output_dir=tmp_path, bundle_name="quick")

    assert result["summary"]["count"] == 4
    assert result["memory_diagnostics"]["deff"] > 1.0
    assert result["baseline"]["accuracy"] == 0.0
    assert result["threshold_calibration"] is not None
    assert result["minimum_binding_score"] >= 0.001
    assert Path(paths["summary_json"]).exists()
    assert Path(paths["report_txt"]).exists()


def test_thinking_between_tokens_probe_reports_matched_conditions():
    result = run_thinking_between_tokens_experiment(
        ThinkingBetweenTokensConfig(
            task=SymbolicInductionConfig(key_count=3, value_count=3, filler_count=1, pair_count=2, gap=4, seed=7),
            seed_count=2,
            silence_steps=2,
        )
    )

    assert result["zero_interval"]["mean_silent_steps"] == 0.0
    assert result["frozen_state"]["mean_silent_steps"] == result["free_evolution"]["mean_silent_steps"]
    assert result["free_evolution"]["mean_silent_steps"] > 0.0
    assert result["interval_timing"]["frozen_state"]["accuracy"] == 0.5
    assert result["interval_timing"]["free_evolution"]["accuracy"] == 1.0
    assert result["memory_diagnostics"]["stability_ratio"] < 3.0


def test_learned_query_conditioned_readout_reduces_held_out_loss():
    result = run_learned_symbolic_retrieval(LearnedRetrievalConfig(
        task=SymbolicInductionConfig(key_count=3, value_count=3, filler_count=1, pair_count=2, gap=8, seed=3),
        train_count=24,
        test_count=12,
        epochs=12,
        hidden_dim=16,
    ))

    assert result["final_loss"] < result["initial_loss"]
    assert 0.0 <= result["test_accuracy"] <= 1.0


def test_temporal_logic_delayed_feedback_learns_only_the_readout():
    result = run_temporal_logic_experiment(TemporalLogicConfig(
        gap=8, train_count=32, test_count=16, epochs=30, hidden_dim=16,
    ))

    assert result["final_loss"] < result["initial_loss"]
    assert result["test_accuracy"] > 0.75
    assert result["feedback_diagnostics"]["feedback_count"] == 32 * 30
    assert result["feedback_diagnostics"]["kernel_updates_applied"] == 0


def test_d3_randomized_uses_disjoint_streams_and_reports_causal_interventions():
    result = run_d3_randomized_experiment(D3RandomizedConfig(
        slot_count=4, gap=4, train_count=12, test_count=8, epochs=4, hidden_dim=12,
    ))

    assert result["experiment_name"] == "d3_randomized_indexed_rule_retrieval"
    assert result["independent_streams"] is True
    assert result["train_stream_seed"] != result["test_stream_seed"]
    assert result["final_loss"] < result["initial_loss"]
    assert result["kernel_updates_applied"] == 0
    assert 0.0 <= result["test_accuracy"] <= 1.0
    assert -1.0 <= result["mean_target_attention_margin"] <= 1.0
    assert 0.0 <= result["interventions"]["queried_value_action_flip_rate"] <= 1.0
    assert 0.0 <= result["interventions"]["unqueried_distractor_action_invariance_rate"] <= 1.0


def test_d3_randomized_binding_noise_and_distractors_reproducible():
    def run(seed: int) -> dict:
        return run_d3_randomized_experiment(D3RandomizedConfig(
            slot_count=4, gap=4, train_count=8, test_count=6, epochs=3,
            hidden_dim=12, seed=seed, binding_noise_std=0.05, distractor_count=2,
        ))

    first = run(7)
    assert first["config"]["binding_noise_std"] == 0.05
    assert first["config"]["distractor_count"] == 2
    assert -1.0 <= first["mean_target_attention_margin"] <= 1.0
    # Fixed config seed must reproduce the episode stream exactly.
    assert run(7)["final_loss"] == first["final_loss"]


def test_temporal_logic_sweep_retains_delay_and_attribution_metadata():
    rows = run_temporal_logic_sweep(
        gaps=[4], seeds=[3], variants=["full"], forced_distractors=True,
        train_count=16, test_count=8, epochs=8,
    )

    assert len(rows) == 1
    assert rows[0]["forced_distractors"] is True
    assert 0.0 <= rows[0]["test_accuracy"] <= 1.0
    assert isinstance(rows[0]["premise_credit_exceeds_irrelevant"], bool)


def test_task_conditioned_eligibility_excludes_forced_distractors_from_readout_trace():
    result = run_temporal_logic_experiment(TemporalLogicConfig(
        gap=16, forced_distractors=True, task_conditioned_eligibility=True,
        train_count=24, test_count=12, epochs=20, hidden_dim=16,
    ))

    assert result["test_accuracy"] > 0.75
    assert result["feedback_diagnostics"]["eligibility_gate_selective"] is True


def test_learned_eligibility_reports_terminal_feedback_selection_diagnostics():
    result = run_temporal_logic_experiment(TemporalLogicConfig(
        gap=8, forced_distractors=True, learned_eligibility=True,
        train_count=16, test_count=8, epochs=8, hidden_dim=12,
    ))

    rate = result["feedback_diagnostics"]["learned_gate_premise_selection_rate"]
    assert 0.0 <= rate <= 1.0
    assert result["learned_eligibility"] is True


def test_soft_eligibility_is_reported_by_experiment_and_sweep():
    result = run_temporal_logic_experiment(TemporalLogicConfig(
        gap=4, forced_distractors=True, soft_eligibility=True,
        train_count=12, test_count=8, epochs=4, hidden_dim=12,
    ))

    attention = result["feedback_diagnostics"]["soft_eligibility"]
    assert result["soft_eligibility"] is True
    assert attention is not None
    assert 0.0 <= attention["mean_premise_attention"] <= 1.0
    assert attention["mean_uniform_premise_attention"] > 0.0
    assert attention["scorer"] == "terminal_query_and_event_state_conditioned_linear_attention"
    assert set(attention["query_token_scores"]) == {"KEY", "LOCK"}
    assert set(attention["query_token_scores"]["KEY"]) == {
        "OPEN", "CLOSED", "KEY", "LOCK", "FILLER", "DISTRACTOR_A", "DISTRACTOR_B",
    }
    assert attention["query_state_weight_l2"]["KEY"] > 0.0

    rows = run_temporal_logic_sweep(
        gaps=[4], seeds=[3], variants=["full"], forced_distractors=True,
        soft_eligibility=True, train_count=8, test_count=4, epochs=2,
    )
    assert rows[0]["soft_eligibility"] is True
    assert 0.0 <= rows[0]["mean_premise_eligibility_attention"] <= 1.0
    assert rows[0]["uniform_premise_eligibility_attention"] > 0.0
    assert isinstance(rows[0]["premise_attention_exceeds_uniform"], bool)


def test_nonlinear_eligibility_uses_per_event_candidates_within_its_budget():
    result = run_temporal_logic_experiment(TemporalLogicConfig(
        gap=4, forced_distractors=True, nonlinear_eligibility=True,
        train_count=12, test_count=8, epochs=4, hidden_dim=12,
        compatibility_hidden_dim=8, compatibility_parameter_budget=384,
    ))

    attention = result["feedback_diagnostics"]["nonlinear_eligibility"]
    assert result["nonlinear_eligibility"] is True
    assert attention is not None
    assert attention["scorer"] == "terminal_query_event_state_nonlinear_compatibility_mlp"
    assert attention["candidate_mode"] == "per_completed_token_event"
    assert attention["mean_candidate_count"] == 6.0
    assert attention["parameter_count"] <= attention["parameter_budget"] == 384

    rows = run_temporal_logic_sweep(
        gaps=[4], seeds=[3], variants=["full"], forced_distractors=True,
        nonlinear_eligibility=True, train_count=8, test_count=4, epochs=2,
    )
    assert rows[0]["nonlinear_eligibility"] is True
    assert 0.0 <= rows[0]["mean_premise_eligibility_attention"] <= 1.0


def test_predictive_credit_eligibility_reports_native_trace_diagnostics():
    result = run_temporal_logic_experiment(TemporalLogicConfig(
        gap=4, forced_distractors=True, predictive_credit_eligibility=True,
        train_count=12, test_count=8, epochs=4, hidden_dim=12,
    ))

    credit = result["feedback_diagnostics"]["predictive_credit_eligibility"]
    assert result["predictive_credit_eligibility"] is True
    assert credit is not None
    assert credit["scorer"] == "online_next_forcing_error_times_native_soe_channel_activity"
    assert credit["candidate_mode"] == "per_completed_token_event"
    assert credit["mean_prediction_error"] >= 0.0
    assert len(credit["channel_credit_gains"]) == 3

    rows = run_temporal_logic_sweep(
        gaps=[4], seeds=[3], variants=["full"], forced_distractors=True,
        predictive_credit_eligibility=True, train_count=8, test_count=4, epochs=2,
    )
    assert rows[0]["predictive_credit_eligibility"] is True
    assert isinstance(rows[0]["predictive_credit_premise_exceeds_irrelevant"], bool)


def test_persistent_predictive_credit_excludes_terminal_trigger_from_eligibility():
    result = run_temporal_logic_experiment(TemporalLogicConfig(
        gap=4, forced_distractors=True, persistent_predictive_credit_eligibility=True,
        train_count=12, test_count=8, epochs=4, hidden_dim=12,
    ))

    credit = result["feedback_diagnostics"]["persistent_predictive_credit_eligibility"]
    assert result["persistent_predictive_credit_eligibility"] is True
    assert credit is not None
    assert credit["scorer"] == "online_next_forcing_error_times_channel_decayed_persistent_eligibility"
    assert credit["terminal_trigger_excluded"] is True
    assert credit["candidate_mode"] == "per_completed_token_event"

    rows = run_temporal_logic_sweep(
        gaps=[4], seeds=[3], variants=["full"], forced_distractors=True,
        persistent_predictive_credit_eligibility=True, train_count=8, test_count=4, epochs=2,
    )
    assert rows[0]["persistent_predictive_credit_eligibility"] is True
    assert isinstance(rows[0]["persistent_predictive_credit_premise_exceeds_irrelevant"], bool)


def test_adaptive_kernel_commits_stable_d3_feedback_proposals():
    result = run_temporal_logic_experiment(TemporalLogicConfig(
        gap=4, adaptive_kernel=True, train_count=12, test_count=8, epochs=3,
        hidden_dim=12, consolidation_interval=4,
    ))

    assert result["adaptive_learning"]["kernel_update_count"] == 12 * 3
    assert result["feedback_diagnostics"]["kernel_updates_applied"] == 12 * 3
    assert result["memory_diagnostics"]["stability_ratio"] <= 0.9
