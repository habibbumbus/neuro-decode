"""Synthetic target alignment: units, actual elapsed time, causality and identity."""

from dataclasses import asdict, replace

import numpy as np
import pytest

from neurodecode.data.binning import bin_partition
from neurodecode.data.loaders import Recording
from neurodecode.data.targets import align_velocity_targets
from neurodecode.splits import TimingConfig


def inputs(times=None, config=None, segments=None):
    config = config or TimingConfig()
    times = 1 + np.arange(251) * 0.004 if times is None else times
    positions = np.column_stack((10 + 125 * (times - 1), 5 - 62.5 * (times - 1)))
    source = Recording(
        times,
        positions,
        ("M1 001",),
        ((np.array([1.013, 1.023, 1.235]),),),
        {"spike_range_s": [0.9, 2.1]},
    )
    spec = {
        "schema_version": 1,
        "timing": asdict(config),
        "overlap": {"start_s": 1.0, "end_s": 2.0},
        "bin_grid": {"origin_s": 1.0, "width_s": config.bin_s},
        "partitions": [
            {
                "name": "train",
                "retained": {"start_s": 1.0, "end_s": 2.0},
                "continuous_segments": [
                    {"start_s": a, "end_s": b} for a, b in (segments or [(1, 2)])
                ],
            }
        ],
    }
    bins = bin_partition(source, spec, config)
    return source, bins, spec, config


def test_constant_velocity_sign_units_pairing_and_warmup() -> None:
    source, bins, spec, config = inputs()
    labels = align_velocity_targets(source, bins, spec, config)
    assert labels.bins is bins
    assert labels.velocity_mm_s.shape == (bins.counts.shape[0], 2)
    np.testing.assert_allclose(
        labels.velocity_mm_s[labels.valid_mask],
        np.tile([125, -62.5], (len(labels.valid_rows), 1)),
    )
    assert labels.valid_rows[0] == 9
    assert labels.reasons[:9] == ("warmup",) * 9
    assert np.isnan(labels.velocity_mm_s[:9]).all()
    row = labels.valid_rows[0]
    assert bins.bin_end_s[row] == 1.2
    np.testing.assert_allclose(labels.source_time_s[row], [1.18, 1.2])
    assert labels.elapsed_s[row] == pytest.approx(0.020)
    assert bins.counts[labels.valid_mask].shape[0] == labels.valid_rows.size


def test_irregular_sampling_uses_actual_denominator_not_nominal_width() -> None:
    times = 1 + np.arange(251) * 0.004
    times[55] = 1.222  # End boundary 1.220 must use the earlier 1.216 sample.
    source, bins, spec, config = inputs(times)
    labels = align_velocity_targets(source, bins, spec, config)
    row = int(np.flatnonzero(bins.bin_end_s == 1.22)[0])
    assert labels.valid_mask[row]
    np.testing.assert_allclose(labels.source_time_s[row], [1.2, 1.216])
    assert labels.elapsed_s[row] == pytest.approx(0.016)
    assert labels.elapsed_deviation_s[row] == pytest.approx(-0.004)
    np.testing.assert_allclose(labels.velocity_mm_s[row], [125, -62.5])
    i0, i1 = labels.source_indices[row]
    wrong = (
        source.cursor_position_mm[i1] - source.cursor_position_mm[i0]
    ) / config.bin_s
    assert not np.allclose(wrong, labels.velocity_mm_s[row])


def test_exact_boundary_included_but_one_ulp_future_sample_excluded() -> None:
    source, bins, spec, config = inputs()
    original = align_velocity_targets(source, bins, spec, config)
    row = int(np.flatnonzero(bins.bin_end_s == 1.22)[0])
    assert original.source_time_s[row, 1] == 1.22
    times = source.behavior_times_s.copy()
    times[55] = np.nextafter(1.22, np.inf)
    changed_source, changed_bins, changed_spec, changed_config = inputs(times)
    changed = align_velocity_targets(
        changed_source, changed_bins, changed_spec, changed_config
    )
    assert changed.source_time_s[row, 1] < bins.bin_end_s[row]
    assert changed.source_indices[row, 1] == 54


def test_future_behavior_mutation_cannot_change_earlier_targets() -> None:
    source, bins, spec, config = inputs()
    positions = source.cursor_position_mm.copy()
    positions[source.behavior_times_s > 1.22] += [100000, -100000]
    changed = replace(source, cursor_position_mm=positions)
    a = align_velocity_targets(source, bins, spec, config)
    b = align_velocity_targets(changed, bins, spec, config)
    earlier = bins.bin_end_s <= 1.22
    np.testing.assert_array_equal(a.valid_mask[earlier], b.valid_mask[earlier])
    np.testing.assert_array_equal(a.source_time_s[earlier], b.source_time_s[earlier])
    np.testing.assert_array_equal(a.velocity_mm_s[earlier], b.velocity_mm_s[earlier])


def test_stale_samples_have_reasons_and_no_velocity() -> None:
    config = TimingConfig(behavior_gap_ms=100)
    times = 1 + np.arange(251) * 0.004
    times = times[(times <= 1.228) | (times >= 1.252)]
    source, bins, spec, config = inputs(times, config)
    labels = align_velocity_targets(source, bins, spec, config)
    row = int(np.flatnonzero(bins.bin_end_s == 1.24)[0])
    assert labels.reasons[row] == "stale_sample"
    assert not labels.valid_mask[row]
    assert np.isnan(labels.velocity_mm_s[row]).all()
    assert labels.sample_age_s[row, 1] > 0.006


def test_gap_segments_reset_support_and_never_interpolate() -> None:
    times = 1 + np.arange(251) * 0.004
    times = times[(times <= 1.396) | (times >= 1.5)]
    source, bins, spec, config = inputs(times, segments=[(1, 1.396), (1.5, 2)])
    labels = align_velocity_targets(source, bins, spec, config)
    second = np.flatnonzero(bins.segment_ids == 1)
    assert not labels.valid_mask[second[:9]].any()
    assert all(labels.reasons[r] == "warmup" for r in second[:9])
    assert labels.valid_mask[second[9]]
    assert (labels.source_time_s[second[9:]] >= 1.5).all()
    assert not np.any((bins.bin_start_s < 1.5) & (bins.bin_end_s > 1.396))


def test_stale_bins_crossing_new_gap_fail_clearly() -> None:
    source, bins, spec, config = inputs()
    keep = (source.behavior_times_s <= 1.396) | (source.behavior_times_s >= 1.5)
    changed = replace(
        source,
        behavior_times_s=source.behavior_times_s[keep],
        cursor_position_mm=source.cursor_position_mm[keep],
    )
    with pytest.raises(ValueError, match="gap"):
        align_velocity_targets(changed, bins, spec, config)


def test_partition_start_cannot_borrow_a_position_from_purge_time() -> None:
    config = TimingConfig(history_bins=1)
    times = 1 + np.arange(251) * 0.004
    times[10] = np.nextafter(1.04, np.inf)
    source, _, spec, _ = inputs(times, config)
    spec["partitions"] = [
        {
            "name": "train",
            "retained": {"start_s": 1, "end_s": 1.035},
            "continuous_segments": [{"start_s": 1, "end_s": 1.035}],
        },
        {
            "name": "validation",
            "retained": {"start_s": 1.039, "end_s": 2},
            "continuous_segments": [{"start_s": 1.039, "end_s": 2}],
        },
    ]
    bins = bin_partition(source, spec, config, "validation")
    labels = align_velocity_targets(source, bins, spec, config)
    assert labels.reasons[0] == "cross_segment_sample"
    assert labels.source_time_s[0, 0] == 1.036
    changed_positions = source.cursor_position_mm.copy()
    changed_positions[times < 1.039] += 100000
    changed = align_velocity_targets(
        replace(source, cursor_position_mm=changed_positions), bins, spec, config
    )
    np.testing.assert_array_equal(labels.velocity_mm_s, changed.velocity_mm_s)
    assert np.all(labels.source_time_s[labels.valid_mask] >= 1.039)


def test_nonfinite_selected_positions_are_row_exclusions_not_repaired() -> None:
    source, bins, spec, config = inputs()
    positions = source.cursor_position_mm.copy()
    positions[50, 0] = np.nan
    labels = align_velocity_targets(
        replace(source, cursor_position_mm=positions), bins, spec, config
    )
    assert labels.reasons[9] == "nonfinite_position"
    assert labels.reasons[10] == "nonfinite_position"
    assert not labels.valid_mask[9:11].any()
    assert labels.valid_mask[11]
    assert np.isnan(labels.velocity_mm_s[9:11]).all()


def test_nonfinite_future_position_does_not_invalidate_past_labels() -> None:
    source, bins, spec, config = inputs()
    positions = source.cursor_position_mm.copy()
    positions[200] = np.nan
    a = align_velocity_targets(source, bins, spec, config)
    b = align_velocity_targets(
        replace(source, cursor_position_mm=positions), bins, spec, config
    )
    past = bins.bin_end_s < source.behavior_times_s[200]
    np.testing.assert_array_equal(a.velocity_mm_s[past], b.velocity_mm_s[past])


def test_same_selected_sample_rejects_nonpositive_elapsed() -> None:
    config = TimingConfig(history_bins=1, max_behavior_age_ms=100, behavior_gap_ms=1000)
    source, bins, spec, config = inputs(np.array([1.0, 1.1, 1.2, 2.0]), config)
    labels = align_velocity_targets(source, bins, spec, config)
    assert "nonpositive_elapsed" in labels.reasons
    assert not np.any(labels.valid_mask[labels.elapsed_s <= 0])


@pytest.mark.parametrize(
    "case", ["duplicate", "descending", "nonfinite", "positions_shape"]
)
def test_invalid_recording_structure_fails_without_sorting(case: str) -> None:
    source, bins, spec, config = inputs()
    times = source.behavior_times_s.copy()
    if case == "duplicate":
        times[50] = times[49]
    elif case == "descending":
        times[50] = times[49] - 0.01
    elif case == "nonfinite":
        times[50] = np.nan
    else:
        source = replace(source, cursor_position_mm=source.cursor_position_mm[:-1])
    source = replace(source, behavior_times_s=times)
    with pytest.raises(ValueError):
        align_velocity_targets(source, bins, spec, config)


@pytest.mark.parametrize(
    "case",
    [
        "counts_length",
        "reversed",
        "timestamp",
        "grid_id",
        "segment_id",
        "channel_id",
        "counts_content",
    ],
)
def test_mismatched_bin_identity_never_trimmed_or_repaired(case: str) -> None:
    source, bins, spec, config = inputs()
    if case == "counts_length":
        bins = replace(bins, counts=bins.counts[:-1])
    elif case == "reversed":
        bins = replace(bins, bin_end_s=bins.bin_end_s[::-1])
    elif case == "timestamp":
        ends = bins.bin_end_s.copy()
        ends[20] = np.nextafter(ends[20], np.inf)
        bins = replace(bins, bin_end_s=ends)
    elif case == "grid_id":
        bins = replace(bins, grid_end_indices=bins.grid_end_indices + 1)
    elif case == "segment_id":
        bins = replace(bins, segment_ids=bins.segment_ids + 1)
    elif case == "channel_id":
        bins = replace(bins, channel_ids=("M1 999",))
    else:
        counts = bins.counts.copy()
        counts[20, 0] += 1
        bins = replace(bins, counts=counts)
    with pytest.raises(ValueError, match="Mismatched"):
        align_velocity_targets(source, bins, spec, config)


def test_binless_fragment_keeps_empty_alignment_shapes() -> None:
    source, bins, spec, config = inputs(segments=[(1, 1.01)])
    labels = align_velocity_targets(source, bins, spec, config)
    assert labels.velocity_mm_s.shape == (0, 2)
    assert labels.source_time_s.shape == (0, 2)
    assert labels.reasons == ()
    assert labels.valid_rows.size == 0


def test_finite_positions_with_overflowing_difference_reject_velocity() -> None:
    source, bins, spec, config = inputs()
    positions = source.cursor_position_mm.copy()
    positions[45, 0] = -np.finfo(np.float64).max
    positions[50, 0] = np.finfo(np.float64).max
    labels = align_velocity_targets(
        replace(source, cursor_position_mm=positions), bins, spec, config
    )
    assert labels.reasons[9] == "nonfinite_velocity"
    assert not labels.valid_mask[9]
    assert np.isnan(labels.velocity_mm_s[9]).all()
