from __future__ import annotations

import math

import pytest

from d2c.experiments.flat_c_capacity_sweep import (
    BASE_KERNEL_MASS,
    BASE_STABILITY_RATIO,
    build_channel_config,
)


@pytest.mark.parametrize("channels", [3, 4, 6])
def test_channel_capacity_preserves_mass_and_stability(channels):
    config = build_channel_config(channels)
    assert config.channel_count == channels
    assert math.isclose(sum(config.w), BASE_KERNEL_MASS, abs_tol=1e-10)
    assert math.isclose(config.stability_ratio(), BASE_STABILITY_RATIO, abs_tol=1e-10)
    assert all(weight > 0 for weight in config.w)


def test_three_channel_configuration_is_exact_g2_flat_baseline():
    config = build_channel_config(3)
    assert config.gamma == [2.0, 0.5, 0.1]
    assert config.w == [0.04, 0.08, 0.12]
    assert config.dt == 0.05
    assert config.leak_rate == 1.0
    assert config.form == "input_driven"


def test_unsupported_channel_count_is_rejected():
    with pytest.raises(ValueError, match="3, 4, or 6"):
        build_channel_config(5)
