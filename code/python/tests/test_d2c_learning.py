from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from d2c import (
    ConsolidationConfig,
    ConsolidationState,
    LastValuePredictor,
    LinearDeltaPredictor,
    LorenzTraceWindow,
    ThreeFactorUpdateConfig,
    TraceStore,
    analyze_trace_store_windows,
    channel_discounts,
    consolidate_weights,
    evaluate_next_state_prediction,
    per_channel_td_errors,
    propose_three_factor_update,
    state_norm_reward,
)
from d2c.runtime import LorenzTraceRecord


def _window() -> LorenzTraceWindow:
    return LorenzTraceWindow(
        ic_index=0,
        source="memory",
        start_idx=0,
        end_idx=2,
        sample_t=[0.0, 0.1, 0.2],
        sample_u=[
            [1.0, 0.0, 2.0],
            [1.2, 0.1, 2.1],
            [1.5, 0.2, 2.4],
        ],
        sample_chi=[
            [0.10, 0.05, 0.01],
            [0.20, 0.08, 0.03],
            [0.30, 0.13, 0.05],
        ],
        next_u=[1.9, 0.3, 2.8],
        metadata={"window_size": 3, "stride": 1, "has_memory_channels": True},
    )


def test_predictors_evaluate_trace_window_next_state():
    window = _window()

    last = evaluate_next_state_prediction(window, LastValuePredictor())
    linear = evaluate_next_state_prediction(window, LinearDeltaPredictor())

    assert last.predictor_name == "last_value"
    assert last.error_l2 is not None
    assert linear.predictor_name == "linear_delta"
    assert linear.error_l2 is not None
    assert linear.error_l2 < last.error_l2


def test_critic_helpers_compute_channel_discounts_and_td_errors():
    gamma = [0.2, 0.05, 0.01]
    discounts = channel_discounts(gamma, dt=0.1)
    
    # Negative reward when above target_norm
    reward_neg = state_norm_reward([40.0, 0.0, 0.0], target_norm=28.0, scale=10.0)
    assert reward_neg < 0.0
    
    # Positive reward when below target_norm
    reward_pos = state_norm_reward([10.0, 0.0, 0.0], target_norm=28.0, scale=10.0)
    assert reward_pos > 0.0

    td = per_channel_td_errors(
        reward=reward_neg,
        current_values=[0.2, 0.4, 0.8],
        next_values=[0.25, 0.45, 0.85],
        gamma=gamma,
        dt=0.1,
    )

    assert len(discounts) == 3
    assert all(0.0 < value < 1.0 for value in discounts)
    assert len(td.errors) == 3
    assert td.discounts == discounts


def test_three_factor_update_proposal_is_stability_bounded():
    window = _window()
    prediction = evaluate_next_state_prediction(window, LinearDeltaPredictor())
    channel_activity = [
        sum(abs(row[index]) for row in window.sample_chi) / len(window.sample_chi)
        for index in range(3)
    ]

    # Test with unsigned absolute signals (default)
    proposal_abs = propose_three_factor_update(
        weights=[0.5, 0.2, 0.1],
        gamma=[0.2, 0.05, 0.01],
        channel_activity=channel_activity,
        prediction_error=prediction.error,
        td_error=[0.1, -0.05, 0.02],
        config=ThreeFactorUpdateConfig(
            eta_pred=0.05,
            eta_val=0.02,
            decay=0.001,
            dt=0.1,
            leak_rate=20.0,
            stability_margin=0.9,
        ),
    )

    assert len(proposal_abs.clipped_weights) == 3
    assert all(value >= 0.0 for value in proposal_abs.clipped_weights)
    assert proposal_abs.stability_ratio_after <= 0.9

    # Test with signed hebbian signals
    hebbian_signal = [0.1, -0.2, 0.05]
    proposal_hebb = propose_three_factor_update(
        weights=[0.5, 0.2, 0.1],
        gamma=[0.2, 0.05, 0.01],
        channel_activity=[0.0, 0.0, 0.0], # Should be ignored
        prediction_error=0.0,            # Should be ignored
        hebbian_signal=hebbian_signal,
        td_error=0.0,
        config=ThreeFactorUpdateConfig(
            eta_pred=0.1,
            decay=0.01,
            dt=1.0,
            leak_rate=10.0,
        )
    )
    
    # delta_w = eta_pred * hebb - decay * w
    # w0 = 0.5, hebb = 0.1 -> delta = 0.1 * 0.1 - 0.01 * 0.5 = 0.01 - 0.005 = 0.005
    # w1 = 0.2, hebb = -0.2 -> delta = 0.1 * -0.2 - 0.01 * 0.2 = -0.02 - 0.002 = -0.022
    assert proposal_hebb.delta[0] == pytest.approx(0.005)
    assert proposal_hebb.delta[1] == pytest.approx(-0.022)
    assert proposal_hebb.proposed_weights[1] < 0.2


def test_default_three_factor_behavior_is_unchanged():
    """Signed/error-modulated flags default off; legacy path must be stable."""
    proposal = propose_three_factor_update(
        weights=[0.16, 0.108, 0.0448],
        gamma=[2.0, 0.45, 0.035],
        channel_activity=[0.5, 0.3, 0.1],
        prediction_error=0.4,
        td_error=0.1,
        config=ThreeFactorUpdateConfig(dt=1.0, eta_pred=0.01, eta_val=0.01, decay=0.001,
                                       leak_rate=4.0),
    )
    # Unsigned hebbian: delta = 0.01 * |act| * |eps| + 0.01 * 0.1 - 0.001 * w
    for weight, activity, delta in zip([0.16, 0.108, 0.0448], [0.5, 0.3, 0.1], proposal.delta):
        expected = 0.01 * abs(activity) * 0.4 + 0.01 * 0.1 - 0.001 * weight
        assert delta == pytest.approx(expected)
    assert proposal.notes == []


def test_signed_hebbian_produces_channel_selective_updates():
    """Signed chi * eps weakens channels when error is negative, proportionally."""
    activity = [0.5, 0.3, 0.1]
    base = dict(
        weights=[0.16, 0.108, 0.0448],
        gamma=[2.0, 0.45, 0.035],
        channel_activity=activity,
        td_error=0.0,
        config=ThreeFactorUpdateConfig(dt=1.0, eta_pred=0.1, eta_val=0.01, decay=0.0,
                                       signed_hebbian=True),
    )
    positive = propose_three_factor_update(prediction_error=0.4, **base)
    negative = propose_three_factor_update(prediction_error=-0.4, **base)
    for delta_pos, delta_neg, act in zip(positive.delta, negative.delta, activity):
        assert delta_pos == pytest.approx(0.1 * act * 0.4)
        assert delta_neg == pytest.approx(-0.1 * act * 0.4)
        # channel selectivity: larger activity -> larger magnitude delta
    assert abs(positive.delta[0]) > abs(positive.delta[2])


def test_error_modulated_decay_scales_forgetting_with_error():
    """Larger prediction error slows decay (beta = beta0 / (1 + |eps|))."""
    base = dict(
        weights=[0.16, 0.108, 0.0448],
        gamma=[2.0, 0.45, 0.035],
        channel_activity=[0.5, 0.3, 0.1],
        td_error=0.0,
        config=ThreeFactorUpdateConfig(dt=1.0, eta_pred=0.0, eta_val=0.0, decay=0.01,
                                       error_modulated_decay=True),
    )
    accurate = propose_three_factor_update(prediction_error=0.1, **base)
    inaccurate = propose_three_factor_update(prediction_error=1.0, **base)
    # Pure decay: delta = -beta * w with beta = 0.01 / (1 + |eps|)
    for delta_small, delta_large, weight in zip(accurate.delta, inaccurate.delta, base["weights"]):
        assert delta_small == pytest.approx(-0.01 / 1.1 * weight)
        assert delta_large == pytest.approx(-0.01 / 2.0 * weight)
    assert any("error-modulated decay" in note for note in accurate.notes)


def test_sequential_commits_differ_from_aggregated_commit():
    """Error-modulated decay makes sequential commits nonlinear in signals."""
    weights = [0.16, 0.108, 0.0448]
    gamma = [2.0, 0.45, 0.035]
    activity = [0.5, 0.3, 0.1]
    config = ThreeFactorUpdateConfig(dt=1.0, eta_pred=0.0, eta_val=0.0, decay=0.01,
                                     error_modulated_decay=True)

    # Sequential: two commits with different per-episode errors.
    first = propose_three_factor_update(
        weights=weights, gamma=gamma, channel_activity=activity,
        prediction_error=0.2, config=config)
    second = propose_three_factor_update(
        weights=first.clipped_weights, gamma=gamma, channel_activity=activity,
        prediction_error=1.0, config=config)
    sequential = second.clipped_weights

    # Aggregate: one commit with the mean error.
    aggregate = propose_three_factor_update(
        weights=weights, gamma=gamma, channel_activity=activity,
        prediction_error=0.6, config=config).clipped_weights

    for seq, agg in zip(sequential, aggregate):
        assert seq != pytest.approx(agg)


def test_consolidation_moves_slow_weights_toward_fast_weights():
    state = ConsolidationState(fast_weights=[0.5, 0.2, 0.1], slow_weights=[0.3, 0.15, 0.08])
    result = consolidate_weights(state, ConsolidationConfig(rate=0.25))

    assert result.state.slow_weights[0] > state.slow_weights[0]
    assert result.state.fast_weights[0] < state.fast_weights[0]
    assert len(result.state.total_weights) == 3


def test_trace_store_learning_analysis_builds_update_proposals():
    trace_store = TraceStore(experiment_name="learning_analysis_test")
    trace_store.add_record(
        LorenzTraceRecord(
            ic_index=0,
            initial_state=[1.0, 0.0, 2.0],
            baseline_trace_t=[0.0, 0.1, 0.2, 0.3],
            baseline_trace_u=[
                [1.0, 0.0, 2.0],
                [1.1, 0.1, 2.1],
                [1.3, 0.1, 2.3],
                [1.6, 0.2, 2.6],
            ],
            memory_trace_t=[0.0, 0.1, 0.2, 0.3],
            memory_trace_u=[
                [1.0, 0.0, 2.0],
                [1.2, 0.1, 2.1],
                [1.5, 0.2, 2.4],
                [1.9, 0.3, 2.8],
            ],
            memory_trace_chi=[
                [0.10, 0.05, 0.01],
                [0.20, 0.08, 0.03],
                [0.30, 0.13, 0.05],
                [0.45, 0.20, 0.08],
            ],
            memory_kernel_gamma=[0.2, 0.05, 0.01],
            memory_kernel_weights=[0.5, 0.2, 0.1],
            baseline_stability={"passed": True},
            memory_stability={"passed": True},
        )
    )

    analysis = analyze_trace_store_windows(trace_store, source="memory", window_size=3, stride=1)

    assert analysis.window_count == 2
    assert analysis.mean_prediction_error_l2 >= 0.0
    assert analysis.windows[0].td_error is not None
    assert analysis.windows[0].update_proposal is not None
    assert analysis.windows[0].update_proposal.stability_ratio_after <= 0.9


def test_learning_analysis_report_is_formatted():
    from d2c import format_learning_analysis_report

    trace_store = TraceStore(experiment_name="report_test")
    trace_store.add_record(
        LorenzTraceRecord(
            ic_index=0,
            initial_state=[1.0, 0.0, 2.0],
            baseline_trace_t=[0.0, 0.1, 0.2, 0.3],
            baseline_trace_u=[[1.0, 0.0, 2.0], [1.1, 0.1, 2.1], [1.3, 0.1, 2.3], [1.6, 0.2, 2.6]],
            memory_trace_t=[0.0, 0.1, 0.2, 0.3],
            memory_trace_u=[[1.0, 0.0, 2.0], [1.2, 0.1, 2.1], [1.5, 0.2, 2.4], [1.9, 0.3, 2.8]],
            memory_trace_chi=[[0.10, 0.05, 0.01], [0.20, 0.08, 0.03], [0.30, 0.13, 0.05], [0.45, 0.20, 0.08]],
            memory_kernel_gamma=[0.2, 0.05, 0.01],
            memory_kernel_weights=[0.5, 0.2, 0.1],
            baseline_stability={"passed": True},
            memory_stability={"passed": True},
        )
    )

    analysis = analyze_trace_store_windows(trace_store, source="memory", window_size=3, stride=1)
    report = format_learning_analysis_report(analysis)

    assert "Learning Analysis Report" in report
    assert "Experiment: report_test" in report
    assert "Mean Prediction Error" in report
    assert "Window 0" in report
    assert "Stability:" in report
