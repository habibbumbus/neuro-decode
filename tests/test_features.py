"""History order, boundaries, identity and training-only normalization."""

from dataclasses import asdict, replace

import numpy as np
import pytest

from neurodecode.data.binning import BinnedSpikes
from neurodecode.data.features import (
    build_history_windows,
    fit_count_scaler,
    flatten_features,
    unflatten_features,
)
from neurodecode.data.targets import AlignedTargets
from neurodecode.splits import TimingConfig


def inputs(partition="train", counts=None, gap=False):
    config = TimingConfig(history_bins=3)
    spec = {
        "source": {"sha256": "a" * 64},
        "configuration": {"sha256": "b" * 64},
        "timing": asdict(config),
        "bin_grid": {"origin_s": 0.0, "width_s": config.bin_s},
        "partitions": [
            {
                "name": name,
                "retained": {"start_s": offset, "end_s": offset + 0.17},
                "continuous_segments": (
                    [
                        {"start_s": offset, "end_s": offset + 0.061},
                        {"start_s": offset + 0.1, "end_s": offset + 0.161},
                    ]
                    if gap
                    else [{"start_s": offset, "end_s": offset + 0.13}]
                ),
            }
            for name, offset in (("train", 0.0), ("validation", 0.2), ("test", 0.4))
        ],
    }
    part = next(p for p in spec["partitions"] if p["name"] == partition)
    offset = {"train": 0, "validation": 10, "test": 20}[partition]
    ids = np.array([1, 2, 3, 6, 7, 8] if gap else range(1, 7)) + offset
    segments = np.array([0, 0, 0, 1, 1, 1] if gap else [0] * 6)
    metadata = tuple(
        dict(
            s,
            segment_id=i,
            row_start=i * 3 if gap else 0,
            row_stop=(i + 1) * 3 if gap else 6,
        )
        for i, s in enumerate(part["continuous_segments"])
    )
    bins = BinnedSpikes(
        np.arange(12).reshape(6, 2) if counts is None else np.array(counts),
        (ids - 1) * config.bin_s,
        ids * config.bin_s,
        ("M1 001", "M1 002"),
        (0, 1),
        ((0, 1), (0,)),
        partition,
        segments,
        ids,
        metadata,
        "M1 lexical; all unit slots",
    )
    reasons = (
        ("warmup", "warmup", "valid") * 2
        if gap
        else ("warmup", "warmup", "valid", "valid", "valid", "valid")
    )
    valid = np.array([r == "valid" for r in reasons])
    velocity = np.column_stack((np.arange(6), -np.arange(6))).astype(float)
    velocity[~valid] = np.nan
    targets = AlignedTargets(
        bins,
        velocity,
        valid,
        reasons,
        np.zeros((6, 2), dtype=np.int64),
        np.zeros((6, 2)),
        np.zeros((6, 2)),
        np.full(6, 0.02),
        np.zeros(6),
        {"version": 1, "nominal_bin_s": config.bin_s},
    )
    return targets, spec, config


def test_hand_calculated_order_and_final_target_identity():
    targets, spec, config = inputs()
    w = build_history_windows(targets, spec, config)
    assert w.X_raw.shape == (4, 3, 2)
    np.testing.assert_array_equal(w.X_raw[0], [[0, 1], [2, 3], [4, 5]])
    np.testing.assert_array_equal(w.y[0], [2, -2])
    np.testing.assert_array_equal(w.source_bin_end_indices[0], [1, 2, 3])
    np.testing.assert_array_equal(w.source_rows[0], [0, 1, 2])
    assert w.target_rows[0] == 2
    assert w.target_bin_end_indices[0] == 3
    assert w.target_time_s[0] == 0.06
    assert w.summary["warmup_windows"] == 2
    assert w.channel_ids == targets.bins.channel_ids


def test_flattening_is_reversible_channels_fastest():
    targets, spec, config = inputs()
    w = build_history_windows(targets, spec, config)
    flat = flatten_features(w.X_raw)
    np.testing.assert_array_equal(flat[0], [0, 1, 2, 3, 4, 5])
    np.testing.assert_array_equal(unflatten_features(flat, 3, 2), w.X_raw)
    assert flatten_features(np.zeros((0, 3, 2))).shape == (0, 6)
    with pytest.raises(ValueError):
        flatten_features(flat)
    with pytest.raises(ValueError):
        unflatten_features(flat, 2, 2)


def test_zero_counts_valid_but_explicit_invalid_bin_breaks_history():
    targets, spec, config = inputs(counts=np.zeros((6, 2), dtype=np.int64))
    assert build_history_windows(targets, spec, config).X_raw.shape == (4, 3, 2)
    mask = np.ones(6, dtype=bool)
    mask[2] = False
    w = build_history_windows(targets, spec, config, mask)
    np.testing.assert_array_equal(w.target_rows, [5])
    assert w.summary["rejection_reasons"]["invalid_history_bin"] == 3


def test_invalid_target_also_breaks_later_histories_without_filtering_rows():
    targets, spec, config = inputs()
    valid = targets.valid_mask.copy()
    valid[2] = False
    velocity = targets.velocity_mm_s.copy()
    velocity[2] = np.nan
    targets = replace(
        targets,
        valid_mask=valid,
        velocity_mm_s=velocity,
        reasons=("warmup", "warmup", "nonfinite_position", "valid", "valid", "valid"),
    )
    w = build_history_windows(targets, spec, config)
    np.testing.assert_array_equal(w.target_rows, [5])
    assert w.summary["rejection_reasons"] == {
        "warmup": 2,
        "invalid_target": 1,
        "invalid_history_bin": 2,
    }


def test_missing_grid_bin_cannot_be_hidden_by_adjacent_array_rows():
    targets, spec, config = inputs()
    keep = np.array([0, 1, 3, 4, 5])
    bins = replace(
        targets.bins,
        counts=targets.bins.counts[keep],
        bin_start_s=targets.bins.bin_start_s[keep],
        bin_end_s=targets.bins.bin_end_s[keep],
        grid_end_indices=targets.bins.grid_end_indices[keep],
        segment_ids=targets.bins.segment_ids[keep],
        segment_metadata=(dict(targets.bins.segment_metadata[0], row_stop=5),),
    )
    targets = replace(
        targets,
        bins=bins,
        velocity_mm_s=targets.velocity_mm_s[keep],
        valid_mask=targets.valid_mask[keep],
        reasons=tuple(targets.reasons[i] for i in keep),
        source_indices=targets.source_indices[keep],
        source_time_s=targets.source_time_s[keep],
        sample_age_s=targets.sample_age_s[keep],
        elapsed_s=targets.elapsed_s[keep],
        elapsed_deviation_s=targets.elapsed_deviation_s[keep],
    )
    w = build_history_windows(targets, spec, config)
    np.testing.assert_array_equal(w.source_bin_end_indices, [[4, 5, 6]])
    assert w.summary["rejection_reasons"]["discontinuous_history"] == 2


def test_gap_segments_reset_warmup_and_never_cross():
    targets, spec, config = inputs(gap=True)
    w = build_history_windows(targets, spec, config)
    np.testing.assert_array_equal(w.source_bin_end_indices, [[1, 2, 3], [6, 7, 8]])
    np.testing.assert_array_equal(w.segment_ids, [0, 1])
    assert w.summary["warmup_windows"] == 4


def test_partition_windows_use_only_their_own_sources():
    for partition in ("train", "validation", "test"):
        targets, spec, config = inputs(partition)
        w = build_history_windows(targets, spec, config)
        part = next(p for p in spec["partitions"] if p["name"] == partition)
        assert np.all(w.source_bin_start_s >= part["retained"]["start_s"])
        assert np.all(w.source_bin_end_s < part["retained"]["end_s"])
        assert w.partition == partition
        if partition != "train":
            with pytest.raises(ValueError, match="training partition"):
                fit_count_scaler(targets, spec, config)


def test_future_counts_do_not_change_past_raw_or_frozen_transformed_windows():
    targets, spec, config = inputs()
    old = build_history_windows(targets, spec, config)
    scaler = fit_count_scaler(targets, spec, config)
    counts = targets.bins.counts.copy()
    counts[5] += 999
    changed = replace(targets, bins=replace(targets.bins, counts=counts))
    new = build_history_windows(changed, spec, config)
    np.testing.assert_array_equal(old.X_raw[:-1], new.X_raw[:-1])
    np.testing.assert_array_equal(
        scaler.transform(old)[:-1], scaler.transform(new)[:-1]
    )
    # A new OFFLINE fit on changed training data may legitimately differ.
    assert not np.array_equal(
        scaler.means, fit_count_scaler(changed, spec, config).means
    )


def test_unique_training_bins_once_including_warmup_not_overlap_weighted():
    targets, spec, config = inputs(
        counts=np.column_stack(([0, 1, 2, 3, 4, 100], [0] * 6))
    )
    scaler = fit_count_scaler(targets, spec, config)
    w = build_history_windows(targets, spec, config)
    np.testing.assert_allclose(scaler.means, targets.bins.counts.mean(axis=0))
    np.testing.assert_allclose(scaler.scales[0], targets.bins.counts[:, 0].std(ddof=0))
    assert scaler.fitted_bin_count == 6
    assert scaler.means[0] != w.X_raw[:, :, 0].mean()


def test_constant_channels_and_heldout_values_keep_frozen_fit_and_order():
    train, spec, config = inputs(counts=np.column_stack((np.arange(6), [0] * 6)))
    scaler = fit_count_scaler(train, spec, config)
    means, scales = scaler.means.copy(), scaler.scales.copy()
    assert scaler.constant_channels.tolist() == [False, True]
    assert scaler.means[1] == 0 and scaler.scales[1] == 1
    with pytest.raises(ValueError):
        scaler.means[0] = 5
    for partition in ("validation", "test"):
        heldout, _, _ = inputs(partition, counts=np.full((6, 2), 1000, dtype=np.int64))
        w = build_history_windows(heldout, spec, config)
        transformed = scaler.transform(w)
        np.testing.assert_array_equal(transformed[:, :, 1], 1000)
        assert transformed.shape == (4, 3, 2)
    np.testing.assert_array_equal(scaler.means, means)
    np.testing.assert_array_equal(scaler.scales, scales)
    np.testing.assert_array_equal(fit_count_scaler(train, spec, config).means, means)


def test_transform_keeps_raw_counts_labels_and_metadata():
    targets, spec, config = inputs()
    w = build_history_windows(targets, spec, config)
    original = w.X_raw.copy()
    scaler = fit_count_scaler(targets, spec, config)
    x = scaler.transform(w)
    assert not np.shares_memory(x, w.X_raw)
    np.testing.assert_array_equal(w.X_raw, original)
    np.testing.assert_array_equal(w.y, targets.velocity_mm_s[w.target_rows])
    metadata = scaler.to_metadata()
    assert metadata["means"] == scaler.means.tolist()
    assert metadata["channel_ids"] == list(w.channel_ids)
    assert metadata["unit_indices"] == [[0, 1], [0]]
    assert metadata["identity"]["source_sha256"] == "a" * 64


@pytest.mark.parametrize("case", ["channel_order", "identity", "history"])
def test_frozen_scaler_rejects_context_mismatch(case):
    targets, spec, config = inputs()
    w = build_history_windows(targets, spec, config)
    scaler = fit_count_scaler(targets, spec, config)
    if case == "channel_order":
        w = replace(w, channel_ids=w.channel_ids[::-1])
    elif case == "identity":
        w = replace(w, identity=(("source_sha256", "wrong"),))
    else:
        w = replace(w, X_raw=w.X_raw[:, :-1])
    with pytest.raises(ValueError, match="identity mismatch"):
        scaler.transform(w)


@pytest.mark.parametrize(
    "case", ["length", "timestamp", "target", "reason", "segment", "mapping"]
)
def test_invalid_inputs_fail_without_trimming_or_repair(case):
    targets, spec, config = inputs()
    if case == "length":
        targets = replace(targets, velocity_mm_s=targets.velocity_mm_s[:-1])
    elif case == "timestamp":
        ends = targets.bins.bin_end_s.copy()
        ends[2] = np.nextafter(ends[2], np.inf)
        targets = replace(targets, bins=replace(targets.bins, bin_end_s=ends))
    elif case == "target":
        velocity = targets.velocity_mm_s.copy()
        velocity[2] = np.nan
        targets = replace(targets, velocity_mm_s=velocity)
    elif case == "reason":
        targets = replace(targets, reasons=("valid",) * 6)
    elif case == "segment":
        spec["partitions"][0]["retained"]["end_s"] = 0.1
    else:
        targets = replace(targets, bins=replace(targets.bins, unit_indices=((0,),)))
    with pytest.raises(ValueError):
        build_history_windows(targets, spec, config)


def test_no_eligible_windows_retains_shapes_and_no_valid_fit_fails():
    targets, spec, config = inputs()
    mask = np.zeros(6, dtype=bool)
    w = build_history_windows(targets, spec, config, mask)
    assert w.X_raw.shape == (0, 3, 2)
    assert w.y.shape == (0, 2)
    with pytest.raises(ValueError, match="No valid training bins"):
        fit_count_scaler(targets, spec, config, mask)


def test_segment_identity_prevents_crossing_even_if_targets_claim_valid():
    targets, spec, config = inputs(gap=True)
    valid = np.array([False, False, True, True, True, True])
    velocity = np.column_stack((np.arange(6), -np.arange(6))).astype(float)
    velocity[:2] = np.nan
    targets = replace(
        targets,
        valid_mask=valid,
        velocity_mm_s=velocity,
        reasons=("warmup", "warmup", "valid", "valid", "valid", "valid"),
    )
    w = build_history_windows(targets, spec, config)
    np.testing.assert_array_equal(w.source_bin_end_indices, [[1, 2, 3], [6, 7, 8]])
    assert w.summary["rejection_reasons"]["discontinuous_history"] == 2


def test_explicit_invalid_bins_are_excluded_from_scaler_fit():
    targets, spec, config = inputs()
    mask = np.ones(6, dtype=bool)
    mask[5] = False
    scaler = fit_count_scaler(targets, spec, config, mask)
    np.testing.assert_allclose(scaler.means, targets.bins.counts[:5].mean(axis=0))
    assert scaler.fitted_bin_count == 5


def test_nonzero_constant_channel_uses_unit_scale_without_deletion():
    targets, spec, config = inputs(counts=np.full((6, 2), 7, dtype=np.int64))
    scaler = fit_count_scaler(targets, spec, config)
    np.testing.assert_array_equal(scaler.means, [7, 7])
    np.testing.assert_array_equal(scaler.scales, [1, 1])
    w = build_history_windows(targets, spec, config)
    np.testing.assert_array_equal(scaler.transform(w), np.zeros((4, 3, 2)))
