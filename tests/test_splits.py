"""Synthetic timestamp-only checks; no recordings or movement patterns."""

from dataclasses import replace

import numpy as np
import pytest

from neurodecode.splits import (
    Interval,
    TimingConfig,
    plan_splits,
    prediction_support,
    read_config,
    select_behavior_samples,
)


def test_order_disjointness_purge_and_elapsed_proportions() -> None:
    plan = plan_splits(np.arange(2501) * 0.004, [-1, 11], TimingConfig())
    parts = plan["partitions"]
    assert [p["name"] for p in parts] == ["train", "validation", "test"]
    assert [p["nominal_duration_s"] for p in parts] == pytest.approx([7, 1.5, 1.5])
    assert [p["purge_discarded_s"] for p in parts] == pytest.approx([0.2, 0.4, 0.2])
    for a, b in zip(parts, parts[1:]):
        assert b["retained"]["start_s"] - a["retained"]["end_s"] == pytest.approx(0.4)
    assert parts[1]["continuous_segments"][0]["warmup_s"] == 0.2
    assert plan["eligible_feature_indices"] is None
    assert plan["bin_grid"]["origin_s"] == 0


def test_support_and_half_open_ownership() -> None:
    segment = Interval(10, 11)
    config = TimingConfig()
    support = prediction_support(10.25, segment, config)
    assert support["feature_interval_s"] == pytest.approx([10.05, 10.25])
    assert support["target_nominal_interval_s"] == pytest.approx([10.23, 10.25])
    assert support["target_source_envelope_s"] == pytest.approx([10.224, 10.25])
    assert support["prediction_available_at_or_after_s"] == 10.25
    assert segment.contains(10)
    assert not segment.contains(11)
    with pytest.raises(ValueError, match="warm-up"):
        prediction_support(10.1, segment, config)
    with pytest.raises(ValueError, match="half-open"):
        prediction_support(11, segment, config)


def test_past_sample_selection_actual_times_and_no_future_tolerance() -> None:
    times = np.arange(251) * 0.004
    indices = select_behavior_samples(times, 0.222, Interval(0, 1), TimingConfig())
    np.testing.assert_allclose(times[list(indices)], [0.200, 0.220])
    just_before = np.nextafter(0.220, -np.inf)
    a, b = select_behavior_samples(times, just_before, Interval(0, 1), TimingConfig())
    assert times[a] <= just_before - 0.02
    assert times[b] < 0.220  # Even a tiny future timestamp must not be selected.


def test_target_support_can_exceed_neural_history() -> None:
    config = TimingConfig(history_bins=1, max_behavior_age_ms=30)
    assert config.lookback_s == pytest.approx(0.050)
    with pytest.raises(ValueError, match="warm-up"):
        prediction_support(0.04, Interval(0, 1), config)
    plan = plan_splits(np.arange(2501) * 0.004, [0, 10], config)
    assert plan["purge_per_side_s"] == pytest.approx(0.050)


def test_gaps_split_segments_without_moving_boundaries_and_reset_warmup() -> None:
    times = np.arange(2501) * 0.004
    times = times[(times <= 2) | (times >= 2.4)]
    plan = plan_splits(times, [0, 10], TimingConfig())
    train = plan["partitions"][0]
    assert train["nominal"]["end_s"] == 7
    assert plan["behavior_gaps_s"][0] == pytest.approx([2, 2.4])
    assert len(train["continuous_segments"]) == 2
    assert train["gap_discarded_inside_retained_s"] == pytest.approx(0.4)
    assert train["continuous_segments"][1][
        "earliest_support_complete_end_s"
    ] == pytest.approx(2.6)
    with pytest.raises(ValueError, match="gap crosses"):
        select_behavior_samples(times, 2.5, Interval(0, 6.8), TimingConfig())
    with pytest.raises(ValueError, match="warm-up"):
        prediction_support(2.5, Interval(2.4, 6.8), TimingConfig())


def test_stale_behavior_and_cross_partition_samples_fail() -> None:
    times = np.array([0.0, 0.1, 0.2, 0.4, 0.5])
    with pytest.raises(ValueError, match="too old"):
        select_behavior_samples(times, 0.25, Interval(0, 1), TimingConfig())
    config = TimingConfig(history_bins=1)
    with pytest.raises(ValueError, match="warm-up"):
        select_behavior_samples(np.arange(251) * 0.004, 0.221, Interval(0.2, 1), config)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bin_ms": 0},
        {"bin_ms": float("nan")},
        {"history_bins": 0},
        {"history_bins": 2.5},
        {"max_behavior_age_ms": -1},
        {"behavior_gap_ms": 0},
        {"additional_lag_ms": 1},
        {"forecast_horizon_ms": 20},
    ],
)
def test_impossible_config_fails(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        replace(TimingConfig(), **kwargs)


@pytest.mark.parametrize(
    "fractions", [(0.7, 0.2, 0.2), (0.7, 0, 0.3), (1,), (np.nan, 0.15, 0.15)]
)
def test_invalid_fractions_fail(fractions: tuple) -> None:
    with pytest.raises(ValueError, match="fractions"):
        plan_splits(np.arange(2501) * 0.004, [0, 10], TimingConfig(), fractions)


def test_short_recording_or_fragmented_partition_fails() -> None:
    with pytest.raises(ValueError, match="Insufficient"):
        plan_splits(np.arange(126) * 0.004, [0, 1], TimingConfig())
    times = np.arange(2501) * 0.004
    times = times[(times < 7.21) | (times > 8.29)]
    with pytest.raises(ValueError, match="continuous duration.*validation"):
        plan_splits(times, [0, 10], TimingConfig())


@pytest.mark.parametrize(
    "times",
    [np.array([0, 0, 1]), np.array([1, 0]), np.array([0, np.nan]), np.zeros((2, 2))],
)
def test_invalid_timestamps_fail(times: np.ndarray) -> None:
    with pytest.raises(ValueError, match="timestamps"):
        plan_splits(times, [0, 10], TimingConfig())


def test_absent_or_disjoint_neural_coverage_fails() -> None:
    times = np.arange(2501) * 0.004
    with pytest.raises(ValueError, match="coverage required"):
        plan_splits(times, None, TimingConfig())
    with pytest.raises(ValueError, match="positive duration"):
        plan_splits(times, [20, 30], TimingConfig())


def test_project_config_matches_contract() -> None:
    config, fractions = read_config("configs/base.toml")
    assert config == TimingConfig()
    assert fractions == (0.70, 0.15, 0.15)


def test_age_tolerance_is_included_in_support_guard() -> None:
    config = TimingConfig(history_bins=1)
    assert config.lookback_s == pytest.approx(0.026000001, abs=1e-15, rel=0)
    end = 0.25
    segment = Interval(end - 0.0260000005, 1)
    with pytest.raises(ValueError, match="warm-up"):
        prediction_support(end, segment, config)
