from pathlib import Path
import sys
import tempfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gfe_ctypes as gfe

from gfe_ctypes import (
    GFE_C_ABERSOE_SCENARIO_LINEAR,
    GFE_C_ABERSOE_SCENARIO_LORENZ63,
    GFE_C_COUPLING_FORM_B,
    GFE_C_HIERARCHICAL_RELATION_BOTTOM_UP,
    GFE_C_HIERARCHICAL_RELATION_TOP_DOWN,
    GFE_C_HIERARCHICAL_SCENARIO_THREE_LEVEL_HOMEOSTATIC,
    GFE_C_HIERARCHICAL_WARNING_CODE_UNKNOWN,
    fit_soe_kernel,
    get_hierarchical_cross_level_report,
    get_hierarchical_cross_level_report_for_chain_spec,
    get_hierarchical_renorm_report,
    get_hierarchical_renorm_report_for_chain_spec,
    hierarchy_cross_level_csv_header,
    hierarchy_renorm_csv_header,
    hierarchy_summary_csv_header,
    list_abersoe_scenarios,
    list_hierarchical_scenarios,
    load_gfe_library,
    run_abersoe_scenario,
    run_abersoe_scenario_with_overrides,
    run_hierarchical_chain_spec,
    run_hierarchical_scenario,
    validate_hierarchical_chain_spec,
    write_hierarchical_chain_export_bundle,
    write_hierarchical_cross_level_report_csv,
    write_hierarchical_export_bundle,
    write_hierarchical_renorm_report_csv,
    write_hierarchical_summary_csv,
)


def _lib_path() -> str | None:
    return None


@pytest.mark.smoke
def test_ctypes_fit_call():
    try:
        lib = load_gfe_library(_lib_path())
    except OSError as exc:
        pytest.skip(f"shared library unavailable: {exc}")

    t = [0.1 * i for i in range(81)]
    y = [0.65 * (2.718281828459045 ** (-0.8 * x)) + 0.35 * (2.718281828459045 ** (-0.12 * x)) for x in t]
    out = fit_soe_kernel(lib, t, y)

    assert len(out["fit"]) == len(t)
    assert len(out["gamma_fit"]) > 0
    assert len(out["w_fit"]) == len(out["gamma_fit"])
    assert all(v == v for v in out["fit"])  # NaN check without extra deps


@pytest.mark.smoke
def test_ctypes_abersoe_scenario_run():
    try:
        lib = load_gfe_library(_lib_path())
    except OSError as exc:
        pytest.skip(f"shared library unavailable: {exc}")

    scenarios = list_abersoe_scenarios(lib)
    assert "linear" in scenarios
    assert "lorenz63" in scenarios

    out = run_abersoe_scenario(
        lib,
        GFE_C_ABERSOE_SCENARIO_LINEAR,
        dt=0.01,
        steps=20,
        sample_stride=5,
        strict_finite=True,
    )

    assert len(out["final_state"]["u"]) == 1
    assert len(out["final_state"]["chi"]) == 3
    assert out["final_state"]["t"] > 0.0
    assert out["diagnostics"]["steps_executed"] == 20
    assert out["diagnostics"]["all_finite"] is True
    assert out["config"]["steps"] == 20
    assert out["config"]["sample_stride"] == 5
    assert len(out["samples"]["t"]) > 0
    assert len(out["samples"]["t"]) == len(out["samples"]["u0"])
    assert len(out["samples"]["t"]) == len(out["samples"]["chi0"])


@pytest.mark.smoke
def test_ctypes_abersoe_lorenz_overrides():
    try:
        lib = load_gfe_library(_lib_path())
    except OSError as exc:
        pytest.skip(f"shared library unavailable: {exc}")

    out = run_abersoe_scenario_with_overrides(
        lib,
        GFE_C_ABERSOE_SCENARIO_LORENZ63,
        dt=0.01,
        steps=20,
        sample_stride=1,
        strict_finite=True,
        initial_u=[2.0, -1.0, 8.0],
        initial_chi=[0.1, -0.2, 0.05],
        initial_t=0.25,
        gamma=[1.4, 0.6, 0.2],
        w=[0.45, 0.35, 0.2],
    )

    assert len(out["final_state"]["u"]) == 3
    assert len(out["final_state"]["chi"]) == 3
    assert out["active_kernel"]["gamma"] == pytest.approx([1.4, 0.6, 0.2])
    assert out["active_kernel"]["w"] == pytest.approx([0.45, 0.35, 0.2])
    assert out["trajectory"]["u_dim"] == 3
    assert out["trajectory"]["chi_dim"] == 3
    assert len(out["trajectory"]["t"]) == 21
    assert out["trajectory"]["t"][0] == pytest.approx(0.25)
    assert out["trajectory"]["u"][0] == pytest.approx([2.0, -1.0, 8.0])
    assert out["trajectory"]["chi"][0] == pytest.approx([0.1, -0.2, 0.05])


@pytest.mark.smoke
def test_ctypes_hierarchical_scenario_run():
    try:
        lib = load_gfe_library(_lib_path())
    except OSError as exc:
        pytest.skip(f"shared library unavailable: {exc}")

    scenarios = list_hierarchical_scenarios(lib)
    assert "two_level_bidirectional" in scenarios
    assert "three_level_homeostatic" in scenarios

    out = run_hierarchical_scenario(
        lib,
        GFE_C_HIERARCHICAL_SCENARIO_THREE_LEVEL_HOMEOSTATIC,
        enable_bottom_up=True,
        enable_top_down=True,
        steps=24,
        sample_stride=6,
        strict_finite=True,
    )

    assert out["diagnostics"]["steps_executed"] == 24
    assert out["diagnostics"]["all_finite"] is True
    assert out["diagnostics"]["level_count"] == 3
    assert len(out["levels"]) == 3
    assert out["levels"][0]["t"] > 0.0
    assert out["levels"][0]["state_dim"] == 1
    assert out["levels"][0]["memory_channels"] == 2
    assert out["levels"][0]["deff"] > 0.0


@pytest.mark.smoke
def test_ctypes_hierarchical_report_access():
    try:
        lib = load_gfe_library(_lib_path())
    except OSError as exc:
        pytest.skip(f"shared library unavailable: {exc}")

    cross = get_hierarchical_cross_level_report(
        lib,
        GFE_C_HIERARCHICAL_SCENARIO_THREE_LEVEL_HOMEOSTATIC,
        enable_bottom_up=True,
        enable_top_down=True,
        steps=24,
        sample_stride=6,
        strict_finite=True,
    )
    assert cross["contract"]["all_finite_pass"] is True
    assert len(cross["kernel_shifts"]) == 3
    assert len(cross["relations"]) >= 2
    assert cross["contract"]["actionable_warning_count"] >= 0
    if cross["warnings"]:
        assert cross["warnings"][0]["code"] >= GFE_C_HIERARCHICAL_WARNING_CODE_UNKNOWN

    renorm = get_hierarchical_renorm_report(
        lib,
        GFE_C_HIERARCHICAL_SCENARIO_THREE_LEVEL_HOMEOSTATIC,
        enable_bottom_up=True,
        enable_top_down=True,
    )
    assert len(renorm) == 2
    assert renorm[0]["raw_output_channels"] >= renorm[0]["output_channels"]
    assert 0.0 <= renorm[0]["retained_weight_fraction"] <= 1.0


@pytest.mark.smoke
def test_ctypes_hierarchy_schema_headers():
    try:
        lib = load_gfe_library(_lib_path())
    except OSError as exc:
        pytest.skip(f"shared library unavailable: {exc}")

    cross_header = hierarchy_cross_level_csv_header(lib)
    renorm_header = hierarchy_renorm_csv_header(lib)
    assert cross_header.startswith("record_type,relation,source_level,target_level")
    assert renorm_header.startswith("input_level,bath_level,direct_level")


@pytest.mark.smoke
def test_ctypes_hierarchical_chain_spec_run():
    try:
        lib = load_gfe_library(_lib_path())
    except OSError as exc:
        pytest.skip(f"shared library unavailable: {exc}")

    levels = [
        {
            "name": "custom_fast",
            "gamma": [1.0, 0.4],
            "w": [0.65, 0.35],
            "u": [1.0],
            "chi": [0.0, 0.0],
            "dt": 0.01,
            "linear_decay": [0.25],
            "forcing_bias": [0.0],
            "form": GFE_C_COUPLING_FORM_B,
            "coupling_index": 0,
        },
        {
            "name": "custom_slow",
            "gamma": [0.7, 0.15],
            "w": [0.6, 0.4],
            "u": [0.1],
            "chi": [0.0, 0.0],
            "dt": 0.01,
            "linear_decay": [0.08],
            "forcing_bias": [0.01],
            "form": GFE_C_COUPLING_FORM_B,
            "coupling_index": 0,
        },
    ]
    edges = [
        {
            "source_level": 0,
            "target_level": 1,
            "relation": GFE_C_HIERARCHICAL_RELATION_BOTTOM_UP,
            "gain": 0.35,
            "normalize_weights": True,
        },
        {
            "source_level": 1,
            "target_level": 0,
            "relation": GFE_C_HIERARCHICAL_RELATION_TOP_DOWN,
            "gain": 0.18,
            "normalize_weights": True,
        },
    ]

    validate_hierarchical_chain_spec(lib, levels, edges)
    out = run_hierarchical_chain_spec(lib, levels, edges, steps=18, sample_stride=6, strict_finite=True)
    assert out["diagnostics"]["steps_executed"] == 18
    assert out["diagnostics"]["all_finite"] is True
    assert out["diagnostics"]["level_count"] == 2
    assert len(out["levels"]) == 2
    assert out["levels"][0]["deff"] > 0.0

    cross = get_hierarchical_cross_level_report_for_chain_spec(
        lib, levels, edges, steps=18, sample_stride=6, strict_finite=True
    )
    assert cross["contract"]["all_finite_pass"] is True
    assert len(cross["kernel_shifts"]) == 2
    assert len(cross["relations"]) == 2

    renorm = get_hierarchical_renorm_report_for_chain_spec(lib, levels, edges)
    assert len(renorm) == 1
    assert 0.0 <= renorm[0]["retained_weight_fraction"] <= 1.0

    with tempfile.TemporaryDirectory() as tmpdir:
        summary_path = str(Path(tmpdir) / "summary.csv")
        cross_path = str(Path(tmpdir) / "cross.csv")
        renorm_path = str(Path(tmpdir) / "renorm.csv")
        write_hierarchical_summary_csv(lib, summary_path, out, level_names=[level["name"] for level in levels])
        write_hierarchical_cross_level_report_csv(lib, cross_path, cross)
        write_hierarchical_renorm_report_csv(lib, renorm_path, renorm)

        assert Path(summary_path).read_text(encoding="utf-8").splitlines()[0] == hierarchy_summary_csv_header(lib)
        assert Path(cross_path).read_text(encoding="utf-8").splitlines()[0] == hierarchy_cross_level_csv_header(lib)
        assert Path(renorm_path).read_text(encoding="utf-8").splitlines()[0] == hierarchy_renorm_csv_header(lib)

        bundle_paths = write_hierarchical_export_bundle(
            lib,
            tmpdir,
            "custom_case",
            out,
            cross,
            renorm,
            level_names=[level["name"] for level in levels],
        )
        assert Path(bundle_paths["summary_csv"]).exists()
        assert Path(bundle_paths["cross_level_csv"]).exists()
        assert Path(bundle_paths["renorm_csv"]).exists()

        chain_bundle_paths = write_hierarchical_chain_export_bundle(
            lib,
            tmpdir,
            "custom_case_direct",
            levels,
            edges,
            steps=18,
            sample_stride=6,
            strict_finite=True,
        )
        assert Path(chain_bundle_paths["summary_csv"]).exists()
        assert Path(chain_bundle_paths["cross_level_csv"]).exists()
        assert Path(chain_bundle_paths["renorm_csv"]).exists()


def test_ctypes_hierarchical_step_chain_spec():
    """D4B smoke test: stateful single-step with external forcing injection."""
    try:
        lib = load_gfe_library(_lib_path())
    except OSError as exc:
        pytest.skip(f"shared library unavailable: {exc}")

    levels = [
        {
            "name": "fast",
            "gamma": [1.0, 0.4],
            "w": [0.65, 0.35],
            "u": [0.0],
            "chi": [0.0, 0.0],
            "dt": 0.01,
            "linear_decay": [0.25],
            "forcing_bias": [0.0],
            "form": GFE_C_COUPLING_FORM_B,
            "coupling_index": 0,
        },
        {
            "name": "slow",
            "gamma": [0.7, 0.15],
            "w": [0.6, 0.4],
            "u": [0.0],
            "chi": [0.0, 0.0],
            "dt": 0.01,
            "linear_decay": [0.08],
            "forcing_bias": [0.0],
            "form": GFE_C_COUPLING_FORM_B,
            "coupling_index": 0,
        },
    ]
    edges = [
        {
            "source_level": 0,
            "target_level": 1,
            "relation": GFE_C_HIERARCHICAL_RELATION_BOTTOM_UP,
            "gain": 0.35,
            "normalize_weights": True,
        },
        {
            "source_level": 1,
            "target_level": 0,
            "relation": GFE_C_HIERARCHICAL_RELATION_TOP_DOWN,
            "gain": 0.18,
            "normalize_weights": True,
        },
    ]

    # Initial states: zeroed out.
    level_states = [
        {"u": [0.0], "chi": [0.0, 0.0], "t": 0.0},
        {"u": [0.0], "chi": [0.0, 0.0], "t": 0.0},
    ]

    # Step once with a non-zero forcing pulse at level 0.
    result = gfe.step_hierarchical_chain_spec(
        lib, levels, edges,
        level_states=level_states,
        external_forcing=[1.0],
        forcing_level=0,
    )

    # Basic structural checks.
    assert "level_states" in result
    assert "active_kernels" in result
    assert "spectral_units" in result
    assert len(result["level_states"]) == 2
    assert len(result["active_kernels"]) == 2
    assert len(result["spectral_units"]) == 2

    # After one step with forcing, level 0 state should have moved.
    assert result["level_states"][0]["u"][0] != 0.0
    # Time should have advanced by dt.
    assert result["level_states"][0]["t"] == pytest.approx(0.01)
    assert result["level_states"][1]["t"] == pytest.approx(0.01)

    # Active kernels should match the spec.
    assert len(result["active_kernels"][0]["gamma"]) == 2
    assert len(result["active_kernels"][0]["w"]) == 2
    assert result["active_kernels"][0]["gamma"] == [1.0, 0.4]

    # Spectral units should be populated.
    assert result["spectral_units"][0]["deff"] > 0.0

    # Step again using the updated states — stateful progression.
    result2 = gfe.step_hierarchical_chain_spec(
        lib, levels, edges,
        level_states=result["level_states"],
        external_forcing=[0.5],
        forcing_level=0,
    )
    assert result2["level_states"][0]["t"] == pytest.approx(0.02)
    # State should have changed further.
    assert result2["level_states"][0]["u"][0] != result["level_states"][0]["u"][0]

    # Step with zero forcing — silent step.
    result3 = gfe.step_hierarchical_chain_spec(
        lib, levels, edges,
        level_states=result2["level_states"],
        external_forcing=[0.0],
        forcing_level=0,
    )
    assert result3["level_states"][0]["t"] == pytest.approx(0.03)
