"""Backward-difference velocity labels paired to verified causal count-bin identities."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from neurodecode.data.binning import BinnedSpikes, bin_partition
from neurodecode.data.loaders import Recording
from neurodecode.splits import COMPARISON_TOLERANCE_S, Interval, TimingConfig


@dataclass(frozen=True)
class AlignedTargets:
    """Audit-preserving labels for every original count row.

    bins retains all original counts, channel IDs, bin timestamps, partition and
    segment identity. velocity_mm_s is [T, 2], x/y mm/s, NaN on invalid rows.
    valid_mask is [T]; reasons is [T], "valid" or the first rejection reason.
    source_indices/source_time_s/sample_age_s are [T, 2], start/end selections.
    Unavailable source indices are -1 and corresponding times/ages are NaN.
    elapsed_s and elapsed_deviation_s are [T], actual b1-b0 and that minus the
    configured width. They remain recorded on rejected rows where available.
    Filter counts and velocities with the same valid_mask; never trim to fit.
    """

    bins: BinnedSpikes
    velocity_mm_s: NDArray[np.float64]
    valid_mask: NDArray[np.bool_]
    reasons: tuple[str, ...]
    source_indices: NDArray[np.int64]
    source_time_s: NDArray[np.float64]
    sample_age_s: NDArray[np.float64]
    elapsed_s: NDArray[np.float64]
    elapsed_deviation_s: NDArray[np.float64]
    policy: dict[str, object]

    @property
    def valid_rows(self) -> NDArray[np.int64]:
        """Original row indices, preserving the validated bin identities."""
        return np.flatnonzero(self.valid_mask)


def align_velocity_targets(
    recording: Recording,
    bins: BinnedSpikes,
    specification: dict[str, object],
    config: TimingConfig,
) -> AlignedTargets:
    """Select past-or-present positions and divide by their actual elapsed seconds.

    Revalidate bins using existing binning logic, including original event counts,
    shape, order, grid timestamps and segment identities. Malformed/stale inputs
    fail rather than being repaired. Legitimate row exclusions retain reasons.
    No interpolation, smoothing, normalization, histories, or model fitting.
    """
    positions = recording.cursor_position_mm
    times = recording.behavior_times_s
    if positions.shape != (times.size, 2):
        raise ValueError("Position shape must be [behavior samples, 2]; no trimming")
    if bins.counts.dtype.kind not in "iu" or np.any(bins.counts < 0):
        raise ValueError("Counts must be nonnegative integers")
    for values in (bins.grid_end_indices, bins.segment_ids):
        if values.dtype.kind not in "iu":
            raise ValueError("Bin/segment identities must be integers")
    expected = bin_partition(recording, specification, config, bins.partition)
    for field in (
        "counts",
        "bin_start_s",
        "bin_end_s",
        "grid_end_indices",
        "segment_ids",
    ):
        if not np.array_equal(getattr(bins, field), getattr(expected, field)):
            raise ValueError(f"Mismatched bin identity/content: {field}; no trimming")
    for field in (
        "channel_ids",
        "source_channel_indices",
        "unit_indices",
        "segment_metadata",
        "channel_policy",
    ):
        if getattr(bins, field) != getattr(expected, field):
            raise ValueError(f"Mismatched bin metadata: {field}")
    row_count = bins.bin_end_s.size
    indices = np.column_stack(
        (
            np.searchsorted(times, bins.bin_start_s, side="right") - 1,
            np.searchsorted(times, bins.bin_end_s, side="right") - 1,
        )
    ).astype(np.int64)
    source_time = np.full((row_count, 2), np.nan)
    available = indices >= 0
    source_time[available] = times[indices[available]]
    boundaries = np.column_stack((bins.bin_start_s, bins.bin_end_s))
    ages = boundaries - source_time
    elapsed = source_time[:, 1] - source_time[:, 0]
    deviation = elapsed - config.bin_s
    velocity = np.full((row_count, 2), np.nan)
    valid = np.zeros(row_count, dtype=bool)
    reasons = ["valid"] * row_count
    age_limit = config.max_behavior_age_ms / 1000 + COMPARISON_TOLERANCE_S
    # Age differences bound separation error; a second ns covers nominal-grid round-off.
    deviation_limit = config.max_behavior_age_ms / 1000 + 2 * COMPARISON_TOLERANCE_S
    origin = specification["bin_grid"]["origin_s"]
    for item in bins.segment_metadata:
        segment = Interval(item["start_s"], item["end_s"])
        first, stop = item["row_start"], item["row_stop"]
        for row in range(first, stop):
            i0, i1 = indices[row]
            if i0 < 0 or i1 < 0:
                reasons[row] = "missing_past_sample"
            elif not all(segment.contains(float(s)) for s in source_time[row]):
                reasons[row] = "cross_segment_sample"
            elif np.any(ages[row] < 0) or np.any(ages[row] > age_limit):
                reasons[row] = "stale_sample"
            elif elapsed[row] <= 0:
                reasons[row] = "nonpositive_elapsed"
            elif abs(deviation[row]) > deviation_limit:
                reasons[row] = "elapsed_deviation"
            else:
                history_start = (
                    origin
                    + (int(bins.grid_end_indices[row]) - config.history_bins)
                    * config.bin_s
                )
                earliest_source = min(history_start, bins.bin_start_s[row] - age_limit)
                if (
                    row - first < config.history_bins - 1
                    or earliest_source < segment.start_s
                ):
                    reasons[row] = "warmup"
                elif not np.isfinite(positions[[i0, i1]]).all():
                    reasons[row] = "nonfinite_position"
                else:
                    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                        value = (
                            positions[i1].astype(np.float64)
                            - positions[i0].astype(np.float64)
                        ) / elapsed[row]
                    if not np.isfinite(value).all():
                        reasons[row] = "nonfinite_velocity"
                    else:
                        velocity[row] = value
                        valid[row] = True
    return AlignedTargets(
        bins,
        velocity,
        valid,
        tuple(reasons),
        indices,
        source_time,
        ages,
        elapsed,
        deviation,
        {
            "version": 1,
            "denominator": "actual selected behavioral elapsed seconds",
            "nominal_bin_s": config.bin_s,
            "max_sample_age_s": age_limit,
            "max_elapsed_deviation_s": deviation_limit,
            "boundary_rule": "latest sample <= stored shared grid start/end; no future tolerance",
            "warmup": "full configured history/support inside same retained segment; no histories built",
            "invalid_velocity": "NaN with explicit valid_mask and first rejection reason",
        },
    )
