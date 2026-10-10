"""Read the documented Zenodo reaching schema from MATLAB 7.3/HDF5 files.

This module preserves event timestamps, channel order, and unit slots. It does
not load waveform snippets or perform feature preparation. Units are supplied by
the dataset's source documentation, not inferred from numeric magnitudes.
"""

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from numpy.typing import NDArray


class RecordingFormatError(ValueError):
    """The input does not satisfy the supported recording schema."""


@dataclass(frozen=True)
class Recording:
    """Raw recording arrays with their original time origin and identities.

    behavior_times_s: [T], strictly increasing behavioral sample times, seconds.
    cursor_position_mm: [T, 2], matching x/y samples, millimeters.
    channel_ids: [C], source channel identifier strings, in source order.
    spike_times_s: [C][U], each entry a variable-length [events] array, seconds.
        Python unit index 0 is MATLAB unit 1 (residual unsorted activity).
        Empty slots remain empty; units are not aggregated or treated as stable
        single neurons across sessions.
    diagnostics: timing and coverage observations; no samples have been repaired.
    """

    behavior_times_s: NDArray[np.float64]
    cursor_position_mm: NDArray[np.float64]
    channel_ids: tuple[str, ...]
    spike_times_s: tuple[tuple[NDArray[np.float64], ...], ...]
    diagnostics: dict[str, object]


def _dataset(file: h5py.File, name: str, matlab_class: str) -> h5py.Dataset:
    value = file.get(name)
    if not isinstance(value, h5py.Dataset):
        raise RecordingFormatError(f"{name}: required dataset is missing or is a group")
    if value.attrs.get("MATLAB_class") != matlab_class.encode():
        raise RecordingFormatError(f"{name}: expected MATLAB_class={matlab_class}")
    return value


def _double(dataset: h5py.Dataset, label: str) -> NDArray[np.float64]:
    if dataset.attrs.get("MATLAB_class") != b"double" or dataset.dtype != np.dtype(
        "f8"
    ):
        raise RecordingFormatError(f"{label}: expected a real float64 MATLAB double")
    values = dataset[()]
    if not np.isfinite(values).all():
        raise RecordingFormatError(f"{label}: nonfinite values")
    return values


def _references(dataset: h5py.Dataset, label: str) -> NDArray:
    if h5py.check_dtype(ref=dataset.dtype) is not h5py.Reference:
        raise RecordingFormatError(f"{label}: expected an object-reference cell array")
    return dataset[()]


def _target(file: h5py.File, reference: h5py.Reference, label: str) -> h5py.Dataset:
    if not reference:
        raise RecordingFormatError(
            f"{label}: null reference is not an explicit empty train"
        )
    try:
        value = file[reference]
    except (KeyError, ValueError) as error:
        raise RecordingFormatError(f"{label}: invalid object reference") from error
    if not isinstance(value, h5py.Dataset):
        raise RecordingFormatError(f"{label}: reference must point to a dataset")
    return value


def _spike_vector(dataset: h5py.Dataset, label: str) -> NDArray[np.float64]:
    if "MATLAB_empty" in dataset.attrs:
        # MATLAB encodes an empty array using its dimension vector, not events.
        dimensions = dataset[()]
        if (
            dataset.attrs["MATLAB_empty"] != 1
            or dataset.attrs.get("MATLAB_class") != b"double"
            or dataset.dtype.kind != "u"
            or dimensions.shape != (2,)
            or not np.any(dimensions == 0)
        ):
            raise RecordingFormatError(f"{label}: unsupported MATLAB empty encoding")
        return np.empty(0, dtype=np.float64)
    if dataset.ndim != 2 or dataset.shape[0] != 1:
        raise RecordingFormatError(f"{label}: expected stored spike vector [1, events]")
    values = _double(dataset, label)[0, :]
    if np.any(np.diff(values) < 0):
        raise RecordingFormatError(
            f"{label}: descending spike timestamps; no sorting performed"
        )
    return values


def load_recording(path: str | Path) -> Recording:
    """Load only the supported reaching-recording HDF5 schema, without repair.

    Reject missing/wrong structures, nonfinite values, duplicate or decreasing
    behavior times, decreasing spike times, and inconsistent channel mapping.
    Preserve empty trains, duplicate spike events, behavioral gaps, and events
    outside behavior coverage; report these in diagnostics. A gap is an interval
    greater than 1.5 times the observed median, a reporting heuristic only.
    Earlier MATLAB formats require a different reader and are not supported.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if not h5py.is_hdf5(path):
        raise RecordingFormatError(
            "Expected MATLAB 7.3/HDF5; earlier MAT formats unsupported"
        )
    with h5py.File(path, "r") as file:
        time_dataset = _dataset(file, "t", "double")
        if time_dataset.ndim != 2 or time_dataset.shape[0] != 1:
            raise RecordingFormatError("t: expected stored shape [1, T]")
        times = _double(time_dataset, "t")[0, :]
        if times.size < 2:
            raise RecordingFormatError("t: at least two behavioral samples required")
        intervals = np.diff(times)
        duplicate_count = int(np.count_nonzero(intervals == 0))
        descending_count = int(np.count_nonzero(intervals < 0))
        if duplicate_count or descending_count:
            raise RecordingFormatError(
                f"t: {duplicate_count} duplicate and {descending_count} descending intervals"
            )
        position_dataset = _dataset(file, "cursor_pos", "double")
        if position_dataset.shape != (2, times.size):
            raise RecordingFormatError(
                "cursor_pos: expected stored shape [2, T] matching t"
            )
        # Source schema [T, 2] + stored [2, T] + matching t establish orientation.
        positions = _double(position_dataset, "cursor_pos").T
        names_dataset = _dataset(file, "chan_names", "cell")
        if names_dataset.ndim != 2 or names_dataset.shape[0] != 1:
            raise RecordingFormatError("chan_names: expected stored shape [1, C]")
        channel_count = names_dataset.shape[1]
        if channel_count == 0:
            raise RecordingFormatError("chan_names: no channels")
        names = []
        for channel, reference in enumerate(
            _references(names_dataset, "chan_names")[0]
        ):
            label = f"chan_names[{channel}]"
            dataset = _target(file, reference, label)
            if (
                dataset.attrs.get("MATLAB_class") != b"char"
                or dataset.dtype != np.dtype("uint16")
                or dataset.ndim != 2
                or dataset.shape[1] != 1
            ):
                raise RecordingFormatError(
                    f"{label}: expected stored UTF-16 column vector"
                )
            try:
                name = dataset[()].astype("<u2").tobytes().decode("utf-16-le")
            except UnicodeDecodeError as error:
                raise RecordingFormatError(f"{label}: invalid UTF-16") from error
            if not name.strip() or "\x00" in name:
                raise RecordingFormatError(
                    f"{label}: empty or null-containing identifier"
                )
            names.append(name)
        if len(set(names)) != channel_count:
            raise RecordingFormatError("chan_names: duplicate channel identifiers")
        spikes_dataset = _dataset(file, "spikes", "cell")
        if (
            spikes_dataset.ndim != 2
            or spikes_dataset.shape[1] != channel_count
            or spikes_dataset.shape[0] == 0
        ):
            raise RecordingFormatError(
                "spikes: expected stored shape [U, C] matching chan_names"
            )
        references = _references(spikes_dataset, "spikes")
        trains = tuple(
            tuple(
                _spike_vector(
                    _target(
                        file, references[unit, channel], f"spikes[{channel}][{unit}]"
                    ),
                    f"spikes[{channel}][{unit}]",
                )
                for unit in range(references.shape[0])
            )
            for channel in range(channel_count)
        )
    counts = [[int(train.size) for train in channel] for channel in trains]
    nonempty = [train for channel in trains for train in channel if train.size]
    duplicate_spikes = [
        {
            "channel_id": names[c],
            "unit_index": u,
            "count": int(np.count_nonzero(np.diff(x) == 0)),
        }
        for c, channel in enumerate(trains)
        for u, x in enumerate(channel)
        if np.any(np.diff(x) == 0)
    ]
    median = float(np.median(intervals))
    diagnostics = {
        "behavior_range_s": [float(times[0]), float(times[-1])],
        "behavior_duration_s": float(times[-1] - times[0]),
        "sample_interval_s": {
            "min": float(intervals.min()),
            "median": median,
            "mean": float(intervals.mean()),
            "max": float(intervals.max()),
            "std": float(intervals.std()),
        },
        "gap_threshold_s": 1.5 * median,
        "gap_after_sample_indices": np.flatnonzero(intervals > 1.5 * median).tolist(),
        "behavior_duplicate_intervals": duplicate_count,
        "spike_duplicate_events": duplicate_spikes,
        "spike_counts_by_channel_unit": counts,
        "total_spike_events": sum(sum(channel) for channel in counts),
        "empty_spike_trains": sum(
            count == 0 for channel in counts for count in channel
        ),
        "spike_range_s": (
            [float(min(x[0] for x in nonempty)), float(max(x[-1] for x in nonempty))]
            if nonempty
            else None
        ),
        "events_before_behavior": sum(
            int(np.count_nonzero(x < times[0])) for x in nonempty
        ),
        "events_after_behavior": sum(
            int(np.count_nonzero(x > times[-1])) for x in nonempty
        ),
    }
    return Recording(times, positions, tuple(names), trains, diagnostics)


def summarize_recording(recording: Recording) -> dict[str, object]:
    """Return a concise JSON-compatible summary; full counts remain in diagnostics."""
    diagnostics = recording.diagnostics
    return {
        "behavior_samples": int(recording.behavior_times_s.size),
        "cursor_position_shape": list(recording.cursor_position_mm.shape),
        "channel_count": len(recording.channel_ids),
        "first_channel": recording.channel_ids[0],
        "last_channel": recording.channel_ids[-1],
        "unit_slots_per_channel": len(recording.spike_times_s[0]),
        **{
            key: value
            for key, value in diagnostics.items()
            if key != "spike_counts_by_channel_unit"
        },
    }
