"""Shared velocity scoring with explicit timestamps, identities and masks."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from neurodecode.data.features import HistoryWindows


@dataclass(frozen=True)
class ExampleIndex:
    """Label-free prediction index; mask is fixed by preparation, never errors."""

    time_s: NDArray[np.float64]
    bin_ids: NDArray[np.int64]
    segment_ids: NDArray[np.int64]
    score_mask: NDArray[np.bool_]
    partition: str
    channel_ids: tuple[str, ...]
    unit_indices: tuple[tuple[int, ...], ...]
    identity: tuple[tuple[str, str], ...]

    def validate(self) -> None:
        """Reject malformed index metadata without sorting or selecting rows."""
        n = self.time_s.size
        if self.time_s.shape != (n,) or not np.isfinite(self.time_s).all():
            raise ValueError("Timestamps must be finite [N]")
        for values in (self.bin_ids, self.segment_ids):
            if values.shape != (n,) or values.dtype.kind not in "iu":
                raise ValueError("Bin/segment identities must be integer [N]")
        if self.score_mask.shape != (n,) or self.score_mask.dtype != np.dtype(bool):
            raise ValueError("Scoring mask must be boolean [N]")
        if np.any(np.diff(self.time_s) <= 0) or np.any(
            self.bin_ids[1:] <= self.bin_ids[:-1]
        ):
            raise ValueError("Timestamps and bin identities must be strictly ordered")
        if np.any(self.segment_ids < 0) or np.any(np.diff(self.segment_ids) < 0):
            raise ValueError("Segment identities must be nonnegative and ordered")
        if (
            not self.channel_ids
            or len(set(self.channel_ids)) != len(self.channel_ids)
            or len(self.unit_indices) != len(self.channel_ids)
        ):
            raise ValueError("Channel/unit identities are inconsistent")
        if not self.partition or not self.identity:
            raise ValueError("Partition and preparation identity are required")


@dataclass(frozen=True)
class VelocitySeries:
    """[N,2] vx/vy in mm/s, paired with an explicit prediction/target index."""

    values_mm_s: NDArray[np.float64]
    index: ExampleIndex

    def validate(self) -> None:
        self.index.validate()
        if self.values_mm_s.shape != (self.index.time_s.size, 2):
            raise ValueError("Velocity shape must be [N,2], matching timestamps")
        if not np.isfinite(self.values_mm_s).all():
            raise ValueError(
                "All velocity values must be finite, even outside scoring mask"
            )


def index_from_windows(windows: HistoryWindows) -> ExampleIndex:
    """Reuse Phase 2D eligibility; copy only label-free metadata for prediction."""
    n = windows.target_time_s.size
    if (
        windows.X_raw.ndim != 3
        or windows.X_raw.shape[0] != n
        or windows.X_raw.shape[2] != len(windows.channel_ids)
    ):
        raise ValueError("Prepared feature shape disagrees with example identities")
    if (
        windows.source_bin_end_indices.shape != windows.X_raw.shape[:2]
        or windows.source_bin_end_s.shape != windows.X_raw.shape[:2]
    ):
        raise ValueError("Prepared source identities disagree with feature shape")
    if (
        windows.X_raw.shape[1] < 1
        or not np.array_equal(
            windows.source_bin_end_indices[:, -1], windows.target_bin_end_indices
        )
        or not np.array_equal(windows.source_bin_end_s[:, -1], windows.target_time_s)
    ):
        raise ValueError("Prepared target must match the final source bin")
    index = ExampleIndex(
        windows.target_time_s.copy(),
        windows.target_bin_end_indices.copy(),
        windows.segment_ids.copy(),
        np.ones(n, dtype=bool),
        windows.partition,
        windows.channel_ids,
        windows.unit_indices,
        windows.identity,
    )
    index.validate()
    return index


def targets_from_windows(windows: HistoryWindows) -> VelocitySeries:
    """Scoring/training labels only; never pass this series to held-out predict."""
    result = VelocitySeries(windows.y.copy(), index_from_windows(windows))
    result.validate()
    return result


def require_matching_indices(a: ExampleIndex, b: ExampleIndex) -> None:
    """Require exact row identity/order and identical predeclared scoring masks."""
    a.validate()
    b.validate()
    for name in ("time_s", "bin_ids", "segment_ids", "score_mask"):
        if not np.array_equal(getattr(a, name), getattr(b, name)):
            raise ValueError(f"Prediction/target index mismatch: {name}")
    for name in ("partition", "channel_ids", "unit_indices", "identity"):
        if getattr(a, name) != getattr(b, name):
            raise ValueError(f"Prediction/target context mismatch: {name}")


def evaluate_velocity(targets: VelocitySeries, predictions: VelocitySeries) -> dict:
    """Per-axis RMSE/R²; overall RMSE averages BOTH axes before square root.

    All inputs must be finite and exact identities/masks must match. Undefined
    R² is None with a reason; mean R² is undefined if either axis is undefined.
    No model-specific exclusions, automatic finite filtering or flattering fill.
    """
    targets.validate()
    predictions.validate()
    require_matching_indices(targets.index, predictions.index)
    mask = targets.index.score_mask
    n = int(mask.sum())
    if n == 0:
        raise ValueError("No scored examples")
    y, predicted = targets.values_mm_s[mask], predictions.values_mm_s[mask]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        squared = np.square(predicted - y)
        mse = squared.mean(axis=0)
        sse = squared.sum(axis=0)
        sst = np.square(y - y.mean(axis=0)).sum(axis=0)
    if not all(np.isfinite(v).all() for v in (mse, sse, sst)):
        raise ValueError("Metric arithmetic overflowed; no score emitted")
    r2, status = [], []
    for axis in range(2):
        if n < 2:
            r2.append(None)
            status.append("insufficient_scored_examples")
        elif np.all(y[:, axis] == y[0, axis]) or sst[axis] == 0:
            r2.append(None)
            status.append("constant_target")
        else:
            value = float(1 - sse[axis] / sst[axis])
            if not np.isfinite(value):
                raise ValueError("R² arithmetic is nonfinite")
            r2.append(value)
            status.append("defined")
    times = targets.index.time_s[mask]
    return {
        "scored_examples": n,
        "first_target_time_s": float(times[0]),
        "last_target_time_s": float(times[-1]),
        "target_time_span_s": float(times[-1] - times[0]),
        "rmse_mm_s": np.sqrt(mse).tolist(),
        "overall_rmse_mm_s": float(np.sqrt(mse.mean())),
        "r2": r2,
        "mean_r2": None if any(v is None for v in r2) else sum(r2) / 2,
        "r2_status": status,
    }
