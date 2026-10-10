"""Causal channel counts on the saved global grid, without targets or histories."""

from dataclasses import asdict, dataclass
from math import ceil, floor, isfinite

import numpy as np
from numpy.typing import NDArray

from neurodecode.data.loaders import Recording
from neurodecode.splits import COMPARISON_TOLERANCE_S, Interval, TimingConfig


@dataclass(frozen=True)
class BinnedSpikes:
    """One partition's valid bins, ordered chronologically.

    counts: [T, C] int64, events per bin, not rates or single-neuron activity.
    bin_start_s/bin_end_s: [T] seconds, with (start, end] event inclusion.
    channel_ids: [C], lexical M1 identifiers if present, otherwise all identifiers.
    unit_indices: [C][included slots], zero-based source slots; slot 0 is residual
        unsorted activity. Empty slots/channels remain included.
    segment_ids: [T] original continuous-segment indices within partition.
    grid_end_indices: [T] integer k for origin + k * bin width.
    segment_metadata: edge exclusions and row ranges, including binless fragments.
    Invalid intervals have no rows, rather than zero-filled counts.
    """

    counts: NDArray[np.int64]
    bin_start_s: NDArray[np.float64]
    bin_end_s: NDArray[np.float64]
    channel_ids: tuple[str, ...]
    source_channel_indices: tuple[int, ...]
    unit_indices: tuple[tuple[int, ...], ...]
    partition: str
    segment_ids: NDArray[np.int64]
    grid_end_indices: NDArray[np.int64]
    segment_metadata: tuple[dict[str, object], ...]
    channel_policy: str


def segment_grid(
    segment: Interval, origin_s: float, width_s: float
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]:
    """Return complete bins with start >= segment start and end < segment end.

    Derive each edge from an integer index, never cumulative floating addition.
    Estimate a generous index range, then compare actual float edges strictly.
    No epsilon, source timestamp rounding, boundary snapping, or partial bins.
    """
    if not isfinite(origin_s) or not isfinite(width_s) or width_s <= 0:
        raise ValueError("Grid origin must be finite and width finite/positive")
    if segment.start_s < origin_s:
        raise ValueError("Segment precedes grid origin")
    first = max(1, floor((segment.start_s - origin_s) / width_s))
    last = ceil((segment.end_s - origin_s) / width_s) + 1
    indices = np.arange(first, last + 1, dtype=np.int64)
    starts = origin_s + (indices - 1) * width_s
    ends = origin_s + indices * width_s
    if np.any(ends <= starts):
        raise ValueError("Bin width cannot be represented at this timestamp magnitude")
    keep = (starts >= segment.start_s) & (ends < segment.end_s)
    return indices[keep], starts[keep], ends[keep]


def bin_partition(
    recording: Recording,
    specification: dict[str, object],
    config: TimingConfig,
    partition: str = "train",
) -> BinnedSpikes:
    """Aggregate every included source unit once into its channel's complete bins.

    Reuse loader timestamps/identities; use only declared coverage and timestamp
    gaps, never spike silence or movement values. Source/checksum and full saved
    plan identity are verified by the reproducible script before this call.
    """
    if specification["schema_version"] != 1 or specification["timing"] != asdict(
        config
    ):
        raise ValueError("Unsupported split schema or timing/configuration mismatch")
    grid = specification["bin_grid"]
    origin, width = grid["origin_s"], grid["width_s"]
    overlap = Interval(**specification["overlap"])
    if origin != overlap.start_s or width != config.bin_s:
        raise ValueError("Grid disagrees with saved overlap/configuration")
    times = recording.behavior_times_s
    if (
        times.ndim != 1
        or times.size < 2
        or not np.isfinite(times).all()
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError("Behavior timestamps must be finite and strictly increasing")
    neural = recording.diagnostics["spike_range_s"]
    if (
        neural is None
        or overlap.start_s < max(times[0], neural[0])
        or overlap.end_s > min(times[-1], neural[1])
    ):
        raise ValueError("Saved overlap exceeds declared neural/behavior coverage")
    previous_partition_end = overlap.start_s
    for item in specification["partitions"]:
        bounds = Interval(**item["retained"])
        if bounds.start_s < previous_partition_end or bounds.end_s > overlap.end_s:
            raise ValueError(
                "Retained partitions overlap, are unordered, or exceed overlap"
            )
        previous_partition_end = bounds.end_s
    matches = [p for p in specification["partitions"] if p["name"] == partition]
    if len(matches) != 1:
        raise ValueError("Requested partition must exist exactly once")
    retained = Interval(**matches[0]["retained"])
    if retained.start_s < overlap.start_s or retained.end_s > overlap.end_s:
        raise ValueError("Retained partition exceeds overlap")
    names = recording.channel_ids
    if (
        not names
        or len(names) != len(set(names))
        or len(names) != len(recording.spike_times_s)
    ):
        raise ValueError("Channel identities must uniquely match spike structures")
    m1 = [i for i, name in enumerate(names) if name.startswith("M1 ")]
    selected = tuple(sorted(m1 or range(len(names)), key=lambda i: names[i]))
    unit_indices = tuple(
        tuple(range(len(recording.spike_times_s[c]))) for c in selected
    )
    for c in selected:
        if not recording.spike_times_s[c]:
            raise ValueError("Channel must have at least one explicit unit slot")
        for events in recording.spike_times_s[c]:
            if (
                events.ndim != 1
                or not np.isfinite(events).all()
                or np.any(np.diff(events) < 0)
            ):
                raise ValueError(
                    "Spike vectors must be finite and nondecreasing; no repair"
                )
    batches, starts_all, ends_all, indices_all, ids_all, metadata = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    previous_end = retained.start_s
    row_start = 0
    for segment_id, item in enumerate(matches[0]["continuous_segments"]):
        segment = Interval(item["start_s"], item["end_s"])
        if segment.start_s < previous_end or segment.end_s > retained.end_s:
            raise ValueError(
                "Segments overlap, are unordered, or exceed retained partition"
            )
        previous_end = segment.end_s
        crossing = (times[:-1] < segment.end_s) & (times[1:] > segment.start_s)
        if np.any(
            crossing
            & (np.diff(times) > config.behavior_gap_ms / 1000 + COMPARISON_TOLERANCE_S)
        ):
            raise ValueError(
                "Saved segment crosses a behavioral gap; replan before binning"
            )
        indices, starts, ends = segment_grid(segment, origin, width)
        counts = np.zeros((ends.size, len(selected)), dtype=np.int64)
        for column, channel in enumerate(selected):
            for events in recording.spike_times_s[channel]:
                # Number of events <= end minus number <= start implements (start, end].
                counts[:, column] += np.searchsorted(
                    events, ends, side="right"
                ) - np.searchsorted(events, starts, side="right")
        metadata.append(
            {
                "segment_id": segment_id,
                "start_s": segment.start_s,
                "end_s": segment.end_s,
                "row_start": row_start,
                "row_stop": row_start + int(ends.size),
                "head_excluded_s": float(starts[0] - segment.start_s)
                if ends.size
                else segment.duration_s,
                "tail_excluded_s": float(segment.end_s - ends[-1])
                if ends.size
                else 0.0,
                "binless": not bool(ends.size),
            }
        )
        row_start += int(ends.size)
        batches.append(counts)
        starts_all.append(starts)
        ends_all.append(ends)
        indices_all.append(indices)
        ids_all.append(np.full(ends.size, segment_id, dtype=np.int64))
    if not batches:
        raise ValueError("Partition has no valid continuous segments")
    return BinnedSpikes(
        np.concatenate(batches),
        np.concatenate(starts_all),
        np.concatenate(ends_all),
        tuple(names[c] for c in selected),
        selected,
        unit_indices,
        partition,
        np.concatenate(ids_all),
        np.concatenate(indices_all),
        tuple(metadata),
        "lexical M1 channels"
        if m1
        else "no M1 identifiers: all channels in lexical order",
    )
