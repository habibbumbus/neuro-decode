"""Hand-calculated and synthetic-only binning checks; no real data fixtures."""

from copy import deepcopy
from dataclasses import asdict, replace

import numpy as np
import pytest

from neurodecode.data.binning import bin_partition, segment_grid
from neurodecode.data.loaders import Recording
from neurodecode.splits import Interval, TimingConfig, plan_splits


def recording() -> Recording:
    times = 1 + np.arange(251) * 0.004
    return Recording(
        times,
        np.zeros((times.size, 2)),
        ("S1 001", "M1 002", "M1 001", "M1 003"),
        (
            (np.array([1.01]),),
            (np.array([1.0, 1.02, 1.025]), np.array([1.02, 1.04]), np.array([])),
            (
                np.array([1.01, 1.02, 1.02, 1.04, 1.08]),
                np.array([1.015, 1.025]),
                np.array([]),
            ),
            (np.array([]), np.array([]), np.array([])),
        ),
        {"spike_range_s": [0.9, 2.1]},
    )


def specification(segments: list[tuple] | None = None) -> dict:
    segments = segments or [(1.0, 1.081)]
    return {
        "schema_version": 1,
        "timing": asdict(TimingConfig()),
        "overlap": {"start_s": 1.0, "end_s": 2.0},
        "bin_grid": {"origin_s": 1.0, "width_s": 0.02},
        "partitions": [
            {
                "name": "train",
                "retained": {"start_s": 1.0, "end_s": 1.081},
                "continuous_segments": [
                    {"start_s": a, "end_s": b} for a, b in segments
                ],
            }
        ],
    }


def test_hand_counts_edges_units_empty_slots_and_identity_order() -> None:
    data = bin_partition(recording(), specification(), TimingConfig())
    assert data.channel_ids == ("M1 001", "M1 002", "M1 003")
    assert data.source_channel_indices == (2, 1, 3)
    assert data.unit_indices == ((0, 1, 2),) * 3
    # Duplicate events are separate list entries; right-edge spikes belong once.
    np.testing.assert_array_equal(
        data.counts, [[4, 2, 0], [2, 2, 0], [0, 0, 0], [1, 0, 0]]
    )
    assert data.counts.dtype == np.int64
    assert (data.counts >= 0).all()
    np.testing.assert_array_equal(data.bin_start_s[1:], data.bin_end_s[:-1])
    assert data.partition == "train"
    np.testing.assert_array_equal(data.segment_ids, [0, 0, 0, 0])


def test_each_event_counted_once_and_conserved_against_original_lists() -> None:
    source = recording()
    binned = bin_partition(source, specification(), TimingConfig())
    for column, channel in enumerate(binned.source_channel_indices):
        independent = sum(
            np.count_nonzero((events > 1.0) & (events <= 1.08))
            for events in source.spike_times_s[channel]
        )
        assert binned.counts[:, column].sum() == independent
        for row, (start, end) in enumerate(
            zip(binned.bin_start_s, binned.bin_end_s, strict=True)
        ):
            expected = sum(
                np.count_nonzero((events > start) & (events <= end))
                for events in source.spike_times_s[channel]
            )
            assert binned.counts[row, column] == expected


def test_one_ulp_future_event_not_pulled_into_previous_bin() -> None:
    source = recording()
    edge = 1.02
    events = np.array([edge, np.nextafter(edge, np.inf)])
    source = replace(source, channel_ids=("M1 001",), spike_times_s=((events,),))
    binned = bin_partition(source, specification(), TimingConfig())
    np.testing.assert_array_equal(binned.counts[:, 0], [1, 1, 0, 0])


def test_mutating_events_after_end_does_not_change_earlier_counts() -> None:
    source = recording()
    changed = replace(
        source,
        spike_times_s=(
            source.spike_times_s[0],
            source.spike_times_s[1],
            (
                np.array([1.01, 1.02, 1.02, 1.04, 1.045, 1.06, 1.079]),
                np.array([1.015, 1.025]),
                np.array([]),
            ),
            source.spike_times_s[3],
        ),
    )
    original = bin_partition(source, specification(), TimingConfig())
    future_changed = bin_partition(changed, specification(), TimingConfig())
    np.testing.assert_array_equal(original.counts[:2], future_changed.counts[:2])
    assert not np.array_equal(original.counts[2:], future_changed.counts[2:])


def test_channel_reordering_preserves_outputs_and_no_m1_fallback_is_explicit() -> None:
    source = recording()
    order = [3, 2, 0, 1]
    reordered = replace(
        source,
        channel_ids=tuple(source.channel_ids[i] for i in order),
        spike_times_s=tuple(source.spike_times_s[i] for i in order),
    )
    np.testing.assert_array_equal(
        bin_partition(source, specification(), TimingConfig()).counts,
        bin_partition(reordered, specification(), TimingConfig()).counts,
    )
    fallback = replace(source, channel_ids=("S1 004", "S1 003", "S1 002", "S1 001"))
    selected = bin_partition(fallback, specification(), TimingConfig())
    assert selected.channel_ids == ("S1 001", "S1 002", "S1 003", "S1 004")
    assert "no M1" in selected.channel_policy


def test_incomplete_bins_and_half_open_segment_end_are_excluded() -> None:
    indices, starts, ends = segment_grid(Interval(1.005, 1.065), 1.0, 0.02)
    np.testing.assert_allclose(starts, [1.02, 1.04])
    np.testing.assert_allclose(ends, [1.04, 1.06])
    np.testing.assert_array_equal(indices, [2, 3])
    _, _, closed = segment_grid(Interval(1.0, 1.04), 1.0, 0.02)
    np.testing.assert_array_equal(closed, [1.02])
    _, _, marginal = segment_grid(
        Interval(np.nextafter(1.02, np.inf), 1.081), 1.0, 0.02
    )
    assert marginal[0] == 1.06  # Do not expand boundary to retain a marginal bin.


def test_grid_uses_integer_indices_without_cumulative_drift() -> None:
    indices, starts, ends = segment_grid(Interval(1288, 1662), 1288, 0.02)
    np.testing.assert_array_equal(ends, 1288 + indices * 0.02)
    np.testing.assert_array_equal(starts, 1288 + (indices - 1) * 0.02)
    np.testing.assert_array_equal(starts[1:], ends[:-1])
    assert ends[-1] < 1662


def test_invalid_gap_is_omitted_not_zero_filled_and_segment_ids_preserved() -> None:
    source = recording()
    keep = (source.behavior_times_s <= 1.028) | (source.behavior_times_s >= 1.06)
    source = replace(
        source,
        behavior_times_s=source.behavior_times_s[keep],
        cursor_position_mm=source.cursor_position_mm[keep],
    )
    binned = bin_partition(
        source, specification([(1, 1.028), (1.06, 1.081)]), TimingConfig()
    )
    np.testing.assert_allclose(binned.bin_end_s, [1.02, 1.08])
    np.testing.assert_array_equal(binned.segment_ids, [0, 1])
    assert binned.counts.shape == (2, 3)
    assert binned.segment_metadata[0]["tail_excluded_s"] == pytest.approx(0.008)


def test_stale_segment_crossing_actual_gap_fails() -> None:
    source = recording()
    times = source.behavior_times_s
    keep = (times <= 1.028) | (times >= 1.06)
    source = replace(
        source,
        behavior_times_s=times[keep],
        cursor_position_mm=source.cursor_position_mm[keep],
    )
    with pytest.raises(ValueError, match="behavioral gap"):
        bin_partition(source, specification(), TimingConfig())


def test_all_empty_trains_produce_valid_zero_rows_not_missingness() -> None:
    source = recording()
    source = replace(
        source,
        spike_times_s=tuple(
            tuple(np.array([]) for _ in c) for c in source.spike_times_s
        ),
    )
    binned = bin_partition(source, specification(), TimingConfig())
    np.testing.assert_array_equal(binned.counts, np.zeros((4, 3), dtype=np.int64))
    # Declared coverage is supplied by the fixture; do not infer it from silence.


def test_binless_fragment_returns_empty_rows_with_exclusion_metadata() -> None:
    binned = bin_partition(recording(), specification([(1, 1.01)]), TimingConfig())
    assert binned.counts.shape == (0, 3)
    assert binned.segment_metadata[0]["binless"]
    assert binned.segment_metadata[0]["head_excluded_s"] == pytest.approx(0.01)


def test_all_partitions_follow_saved_raw_time_and_purge_gap_rules() -> None:
    times = np.arange(2501) * 0.004
    source = Recording(
        times,
        np.zeros((len(times), 2)),
        ("M1 001",),
        ((np.arange(1001) * 0.01,),),
        {"spike_range_s": [-1, 11]},
    )
    plan = plan_splits(times, [-1, 11], TimingConfig())
    outputs = []
    for part in plan["partitions"]:
        binned = bin_partition(source, plan, TimingConfig(), part["name"])
        assert (binned.bin_start_s >= part["retained"]["start_s"]).all()
        assert (binned.bin_end_s < part["retained"]["end_s"]).all()
        outputs.append(binned)
    for a, b in zip(outputs, outputs[1:]):
        assert a.bin_end_s[-1] < b.bin_start_s[0]


@pytest.mark.parametrize(
    "case",
    [
        "timing",
        "grid",
        "segments",
        "channel_identity",
        "descending",
        "nonfinite",
        "coverage",
    ],
)
def test_concrete_invalid_inputs_fail_without_repair(case: str) -> None:
    source, spec = recording(), deepcopy(specification())
    if case == "timing":
        spec["timing"]["bin_ms"] = 10
    elif case == "grid":
        spec["bin_grid"]["origin_s"] = 0
    elif case == "segments":
        spec["partitions"][0]["continuous_segments"] = [
            {"start_s": 0.99, "end_s": 1.04}
        ]
    elif case == "channel_identity":
        source = replace(source, channel_ids=("M1 001",) * 4)
    elif case in ("descending", "nonfinite"):
        values = np.array([1.02, 1.01]) if case == "descending" else np.array([np.nan])
        source = replace(
            source,
            spike_times_s=(
                source.spike_times_s[0],
                (values,),
                *source.spike_times_s[2:],
            ),
        )
    else:
        spec["overlap"]["end_s"] = 3
    with pytest.raises(ValueError):
        bin_partition(source, spec, TimingConfig())


def test_disjoint_segments_conserve_only_included_original_events() -> None:
    source = recording()
    keep = (source.behavior_times_s <= 1.028) | (source.behavior_times_s >= 1.06)
    source = replace(
        source,
        behavior_times_s=source.behavior_times_s[keep],
        cursor_position_mm=source.cursor_position_mm[keep],
    )
    binned = bin_partition(
        source, specification([(1, 1.028), (1.06, 1.081)]), TimingConfig()
    )
    for column, channel in enumerate(binned.source_channel_indices):
        expected = sum(
            np.count_nonzero(
                ((events > 1) & (events <= 1.02)) | ((events > 1.06) & (events <= 1.08))
            )
            for events in source.spike_times_s[channel]
        )
        assert binned.counts[:, column].sum() == expected


@pytest.mark.parametrize("which", ["partitions", "segments"])
def test_overlapping_partition_or_segment_specs_fail(which: str) -> None:
    spec = specification()
    if which == "partitions":
        spec["partitions"].append(
            {
                "name": "validation",
                "retained": {"start_s": 1.06, "end_s": 1.1},
                "continuous_segments": [{"start_s": 1.06, "end_s": 1.1}],
            }
        )
    else:
        spec["partitions"][0]["continuous_segments"] = [
            {"start_s": 1, "end_s": 1.04},
            {"start_s": 1.02, "end_s": 1.081},
        ]
    with pytest.raises(ValueError, match="overlap"):
        bin_partition(recording(), spec, TimingConfig())
