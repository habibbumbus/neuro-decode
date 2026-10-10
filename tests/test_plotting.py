"""Test selection and shared time conversion using visibly synthetic raw arrays."""

from dataclasses import replace

import numpy as np
import pytest

from neurodecode.data.loaders import Recording
from neurodecode.plotting import select_plot_interval


def synthetic_recording() -> Recording:
    times = np.arange(100.0, 141.0)
    trains = (
        (np.array([99.0, 100.0, 109.9, 110.0, 139.0]), np.array([])),
        (np.array([100.5, 109.0]), np.array([101.0])),
        (np.array([101.5]), np.array([])),
    )
    return Recording(
        times,
        np.column_stack((times, -times)),
        ("M1 002", "M1 001", "M1 003"),
        trains,
        {"spike_range_s": [99.0, 139.0], "gap_threshold_s": 1.5},
    )


def test_half_open_interval_and_shared_offset_preserve_identity() -> None:
    recording = synthetic_recording()
    interval = select_plot_interval(recording, channel_limit=2)
    assert (interval.start_s, interval.end_s) == (100.0, 110.0)
    np.testing.assert_array_equal(interval.behavior_time_relative_s, np.arange(10))
    np.testing.assert_array_equal(
        interval.cursor_position_mm, recording.cursor_position_mm[:10]
    )
    assert [(t.channel_id, t.unit_index) for t in interval.trains] == [
        ("M1 001", 0),
        ("M1 001", 1),
        ("M1 002", 0),
        ("M1 002", 1),
    ]
    np.testing.assert_allclose(interval.trains[0].time_relative_s, [0.5, 9.0])
    np.testing.assert_allclose(interval.trains[2].time_relative_s, [0.0, 9.9])
    assert interval.trains[3].time_relative_s.size == 0
    assert interval.trains[3].label == "M1 002 / U2 (sorted slot)"
    assert "unsorted" in interval.trains[2].label


def test_selection_does_not_depend_on_movement_or_later_values() -> None:
    recording = synthetic_recording()
    changed = replace(
        recording, cursor_position_mm=np.full_like(recording.cursor_position_mm, 999.0)
    )
    original = select_plot_interval(recording)
    selected = select_plot_interval(changed)
    assert (selected.start_s, selected.end_s) == (original.start_s, original.end_s)
    assert [(t.channel_id, t.unit_index) for t in selected.trains] == [
        (t.channel_id, t.unit_index) for t in original.trains
    ]
    for a, b in zip(original.trains, selected.trains, strict=True):
        np.testing.assert_array_equal(a.time_relative_s, b.time_relative_s)


def test_starts_at_first_behavior_sample_inside_overlap() -> None:
    recording = synthetic_recording()
    recording.diagnostics["spike_range_s"] = [100.5, 139.0]
    interval = select_plot_interval(recording)
    assert interval.start_s == 101.0
    assert interval.end_s == 111.0
    assert interval.behavior_time_relative_s[0] == 0.0


def test_refuses_gap_without_searching_for_another_clip() -> None:
    recording = synthetic_recording()
    times = np.concatenate((np.arange(100.0, 103.0), np.arange(110.0, 141.0)))
    recording = replace(
        recording,
        behavior_times_s=times,
        cursor_position_mm=np.column_stack((times, times)),
    )
    with pytest.raises(ValueError, match="gap intersects fixed interval"):
        select_plot_interval(recording)


def test_rejects_clip_entering_provisional_later_portion() -> None:
    recording = synthetic_recording()
    times = np.arange(100.0, 113.0)
    recording = replace(
        recording,
        behavior_times_s=times,
        cursor_position_mm=recording.cursor_position_mm[:13],
    )
    with pytest.raises(ValueError, match="provisional training coverage"):
        select_plot_interval(recording)


def test_rejects_absent_neural_coverage() -> None:
    recording = synthetic_recording()
    recording.diagnostics["spike_range_s"] = None
    with pytest.raises(ValueError, match="No neural events"):
        select_plot_interval(recording)


@pytest.mark.parametrize("duration", [0.0, -1.0, np.nan])
def test_rejects_invalid_duration(duration: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        select_plot_interval(synthetic_recording(), duration_s=duration)
