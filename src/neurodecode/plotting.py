"""Deterministic exploratory views of raw recorded spikes and cursor positions."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from neurodecode.data.loaders import Recording


@dataclass(frozen=True)
class RasterTrain:
    """One identified source unit slot; event times relative to the shared start."""

    channel_id: str
    unit_index: int
    time_relative_s: NDArray[np.float64]

    @property
    def label(self) -> str:
        kind = "unsorted" if self.unit_index == 0 else "sorted slot"
        return f"{self.channel_id} / U{self.unit_index + 1} ({kind})"


@dataclass(frozen=True)
class PlotInterval:
    """All panels share the half-open original-time interval [start_s, end_s)."""

    start_s: float
    end_s: float
    provisional_training_end_s: float
    behavior_time_relative_s: NDArray[np.float64]
    cursor_position_mm: NDArray[np.float64]
    trains: tuple[RasterTrain, ...]


def select_plot_interval(
    recording: Recording,
    duration_s: float = 10.0,
    channel_limit: int = 4,
    unit_limit: int = 2,
) -> PlotInterval:
    """Select earliest shared coverage, constrained to the provisional first 70%.

    Start at the first behavioral sample at/after the global spike coverage start.
    Use [start, start + duration), with no quality-based search for another clip.
    Channel IDs are sorted lexically; select their first unit slots, even if empty.
    Reject gaps intersecting the interval using the loader's reporting threshold;
    do not interpolate, shift signals, or retry on a better-looking interval.
    This guard reserves later data; it does not generate evaluation partitions.
    """
    if not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("duration_s must be finite and positive")
    if channel_limit < 1 or unit_limit < 1:
        raise ValueError("channel_limit and unit_limit must be positive")
    times = recording.behavior_times_s
    spike_range = recording.diagnostics["spike_range_s"]
    if spike_range is None:
        raise ValueError("No neural events establish shared coverage")
    overlap_start = max(float(times[0]), spike_range[0])
    overlap_end = min(float(times[-1]), spike_range[1])
    index = int(np.searchsorted(times, overlap_start, side="left"))
    if index >= times.size:
        raise ValueError("No shared neural/behavior interval")
    start = float(times[index])
    end = start + duration_s
    training_end = float(times[0] + 0.70 * (times[-1] - times[0]))
    if end > min(overlap_end, training_end):
        raise ValueError(
            "Requested early interval exceeds shared or provisional training coverage"
        )
    crossing = (times[:-1] < end) & (times[1:] > start)
    gaps = np.flatnonzero(
        crossing & (np.diff(times) > recording.diagnostics["gap_threshold_s"])
    )
    if gaps.size:
        raise ValueError(
            f"Behavior gap intersects fixed interval after sample indices {gaps.tolist()}"
        )
    mask = (times >= start) & (times < end)
    if np.count_nonzero(mask) < 2:
        raise ValueError("Interval contains fewer than two behavior samples")
    selected_channels = sorted(
        range(len(recording.channel_ids)), key=lambda c: recording.channel_ids[c]
    )[:channel_limit]
    trains = []
    for channel in selected_channels:
        for unit, events in enumerate(recording.spike_times_s[channel][:unit_limit]):
            selected = events[(events >= start) & (events < end)]
            trains.append(
                RasterTrain(recording.channel_ids[channel], unit, selected - start)
            )
    return PlotInterval(
        start,
        end,
        training_end,
        times[mask] - start,
        recording.cursor_position_mm[mask],
        tuple(trains),
    )


def save_recorded_figure(
    interval: PlotInterval, session: str, output: str | Path
) -> None:
    """Save a raster, shared-time positions, and equal-scale trajectory without a GUI."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(14, 8), layout="constrained")
    FigureCanvasAgg(figure)
    grid = figure.add_gridspec(2, 2, width_ratios=(1.7, 1), height_ratios=(1.4, 1))
    raster = figure.add_subplot(grid[0, 0])
    position = figure.add_subplot(grid[1, 0], sharex=raster)
    trajectory = figure.add_subplot(grid[:, 1])
    duration = interval.end_s - interval.start_s
    raster.eventplot(
        [train.time_relative_s for train in interval.trains],
        lineoffsets=np.arange(len(interval.trains)),
        linelengths=0.7,
        linewidths=0.7,
        colors="#263f61",
    )
    raster.set_yticks(
        range(len(interval.trains)), [train.label for train in interval.trains]
    )
    raster.set_ylim(len(interval.trains) - 0.5, -0.5)
    raster.set_xlim(0, duration)
    raster.set_title("A. Spike events · one channel/unit slot per row", loc="left")
    raster.set_ylabel("Identified spike train")
    raster.tick_params(axis="x", labelbottom=False)
    for row, train in enumerate(interval.trains):
        if train.time_relative_s.size == 0:
            raster.text(
                duration * 0.98,
                row,
                "No events in interval",
                ha="right",
                va="center",
                fontsize=8,
                color="#6b7280",
            )
    colors = ("#1768ac", "#b34f00")
    for axis, name in enumerate(("x", "y")):
        position.plot(
            interval.behavior_time_relative_s,
            interval.cursor_position_mm[:, axis],
            label=f"Cursor {name}",
            color=colors[axis],
            linewidth=1.2,
        )
    position.set_title("B. Recorded cursor position", loc="left")
    position.set_xlabel(f"Time since interval start ({interval.start_s:.3f} s) [s]")
    position.set_ylabel("Position [mm]")
    position.legend(loc="best", frameon=False, ncols=2)
    xy = interval.cursor_position_mm
    trajectory.plot(xy[:, 0], xy[:, 1], color="#6b7280", linewidth=1)
    trajectory.scatter(
        *xy[0], s=90, marker="o", color="#16803b", label="First sample", zorder=3
    )
    trajectory.scatter(
        *xy[-1], s=110, marker="X", color="#c62828", label="Last sample", zorder=4
    )
    trajectory.set_aspect("equal", adjustable="box")
    trajectory.set_xlabel("Cursor x [mm]")
    trajectory.set_ylabel("Cursor y [mm]")
    trajectory.set_title("C. x-y trajectory · same interval", loc="left")
    trajectory.legend(loc="best", frameon=False)
    for axis in (raster, position, trajectory):
        axis.grid(alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        f"Recorded neural activity and cursor movement | {session}\n"
        f"Source interval [{interval.start_s:.3f}, {interval.end_s:.3f}) s · "
        "early training-intended exploration · no predictions",
        fontsize=13,
    )
    figure.text(
        0.5,
        -0.015,
        "No signal shift, smoothing, or gap interpolation. Temporal co-occurrence is not decoding evidence.",
        ha="center",
        fontsize=9,
        color="#4b5563",
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, bbox_inches="tight", facecolor="white")
