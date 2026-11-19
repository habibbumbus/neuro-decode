"""Synthetic endpoint alignment, integration, anchoring and explicit gap rejection."""

import numpy as np
import pytest

from neurodecode.demo import audit_tracker_alignment, integrate_velocity


def arguments():
    return (
        np.array([[100.0, -50.0], [200.0, 20.0], [-10.0, 40.0]]),
        np.array([1.0, 1.02, 1.05]),
        np.array([1.02, 1.05, 1.06]),
        np.array([10, 11, 12]),
        np.array([0, 0, 0]),
        np.array([7.0, -3.0]),
    )


def test_actual_variable_durations_and_completed_bin_timestamps():
    values, starts, ends, ids, segments, initial = arguments()
    result = integrate_velocity(values, starts, ends, ids, segments, initial)
    np.testing.assert_array_equal(result.time_s, [1.0, 1.02, 1.05, 1.06])
    np.testing.assert_allclose(
        result.position_mm, [[7, -3], [9, -4], [15, -3.4], [14.9, -3]]
    )
    # First estimate belongs at its bin END, not the anchor or next bin.
    np.testing.assert_allclose(
        result.position_mm[1] - initial, values[0] * (ends[0] - starts[0])
    )


def test_shared_initial_anchor_and_no_reference_corrections_or_input_mutation():
    args = arguments()
    first = integrate_velocity(*args)
    second = integrate_velocity(args[0] * 2, *args[1:])
    np.testing.assert_array_equal(first.position_mm[0], second.position_mm[0])
    np.testing.assert_array_equal(args[-1], [7, -3])
    args[-1][:] = 999
    np.testing.assert_array_equal(first.position_mm[0], [7, -3])
    assert not np.array_equal(first.position_mm[1:], second.position_mm[1:])


def test_future_velocity_changes_do_not_change_earlier_positions():
    args = arguments()
    before = integrate_velocity(*args)
    changed = args[0].copy()
    changed[-1] += 1000
    after = integrate_velocity(changed, *args[1:])
    np.testing.assert_array_equal(before.position_mm[:-1], after.position_mm[:-1])


@pytest.mark.parametrize(
    "case",
    ["missing", "nan", "gap", "segment", "repeated_id", "backwards", "zero_duration"],
)
def test_missing_and_discontinuous_input_fails_without_bridging_or_reset(case):
    args = list(arguments())
    available = np.ones(3, dtype=bool)
    if case == "missing":
        available[1] = False
    elif case == "nan":
        args[0][1, 0] = np.nan
    elif case == "gap":
        args[1][1] += 0.001
    elif case == "segment":
        args[4][1:] = 1
    elif case == "repeated_id":
        args[3][1] = args[3][0]
    elif case == "backwards":
        args[2][1] = args[1][1] - 0.01
    else:
        args[2][1] = args[1][1]
    with pytest.raises(ValueError):
        integrate_velocity(*args, available=available)


def test_exact_tracker_alignment_is_distinct_from_velocity_integration_label():
    starts, ends = np.array([1.0, 1.02]), np.array([1.02, 1.04])
    source_times = np.column_stack((starts, ends))
    positions = np.array([[[0.0, 0.0], [2.0, 1.0]], [[2.0, 1.0], [3.0, 3.0]]])
    velocities = np.diff(positions, axis=1)[:, 0] / np.diff(source_times, axis=1)
    audit = audit_tracker_alignment(
        starts, ends, velocities, source_times, np.array([[0, 1], [1, 2]]), positions
    )
    assert audit["exact_tracker_grid_alignment"]
    assert audit["reference_label"] == "tracker-derived reference velocity integration"


def test_behavior_duration_mismatch_is_detected_not_silently_called_tracker_position():
    starts, ends = np.array([1.0, 1.02]), np.array([1.02, 1.04])
    times = np.array([[1.0, 1.016], [1.016, 1.04]])
    positions = np.array([[[0.0, 0.0], [2.0, 1.0]], [[2.0, 1.0], [3.0, 3.0]]])
    velocities = np.diff(positions, axis=1)[:, 0] / np.diff(times, axis=1)
    audit = audit_tracker_alignment(
        starts, ends, velocities, times, np.array([[0, 1], [1, 2]]), positions
    )
    assert not audit["exact_tracker_grid_alignment"]
    assert audit["duration_mismatches_over_1ns"] == 2
    assert audit["source_endpoint_chain_breaks"] == 0
    assert (
        audit["maximum_target_integration_vs_sampled_tracker_coordinate_difference_mm"]
        > 0
    )


def test_future_tracker_sample_and_incorrect_target_values_rejected():
    starts, ends = np.array([1.0]), np.array([1.02])
    positions = np.array([[[0.0, 0.0], [2.0, 1.0]]])
    times = np.array([[1.0, 1.02]])
    values = (positions[:, 1] - positions[:, 0]) / np.diff(times, axis=1)
    with pytest.raises(ValueError, match="after"):
        audit_tracker_alignment(
            starts, ends, values, times + 0.001, np.array([[0, 1]]), positions
        )
    with pytest.raises(AssertionError):
        audit_tracker_alignment(
            starts, ends, values + 1, times, np.array([[0, 1]]), positions
        )
