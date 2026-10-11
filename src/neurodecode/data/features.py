"""Causal oldest-to-newest histories and frozen training-only channel standardization."""

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray

from neurodecode.data.targets import AlignedTargets
from neurodecode.splits import TimingConfig


def _identity(
    specification: dict, targets: AlignedTargets
) -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            {
                "source_sha256": specification["source"]["sha256"],
                "split_sha256_canonical_json": hashlib.sha256(
                    json.dumps(
                        specification,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode()
                ).hexdigest(),
                "configuration_sha256_utf8_lf": specification["configuration"][
                    "sha256"
                ],
                "target_policy_sha256": hashlib.sha256(
                    json.dumps(
                        targets.policy,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode()
                ).hexdigest(),
            }.items()
        )
    )


def _validate(
    targets: AlignedTargets,
    specification: dict,
    config: TimingConfig,
    bin_valid_mask: NDArray[np.bool_] | None,
) -> NDArray[np.bool_]:
    """Validate the reused structures without reading/realigning raw recordings."""
    bins = targets.bins
    t, c = bins.counts.shape if bins.counts.ndim == 2 else (-1, -1)
    if c < 1 or c != len(bins.channel_ids) or len(set(bins.channel_ids)) != c:
        raise ValueError("Counts must be [T, C] with unique fixed channel identifiers")
    if len(bins.unit_indices) != c or len(bins.source_channel_indices) != c:
        raise ValueError("Channel/unit mapping length disagrees with counts")
    if bins.counts.dtype.kind not in "iu" or np.any(bins.counts < 0):
        raise ValueError("Raw counts must be nonnegative integers")
    if (
        specification["timing"] != asdict(config)
        or specification["bin_grid"]["width_s"] != config.bin_s
    ):
        raise ValueError("Configuration disagrees with saved timing/grid")
    for name in ("bin_start_s", "bin_end_s", "grid_end_indices", "segment_ids"):
        if getattr(bins, name).shape != (t,):
            raise ValueError(f"Mismatched bin length: {name}")
    if (
        bins.grid_end_indices.dtype.kind not in "iu"
        or bins.segment_ids.dtype.kind not in "iu"
    ):
        raise ValueError("Bin and segment identities must be integers")
    if np.any(np.diff(bins.grid_end_indices) <= 0):
        raise ValueError(
            "Bin identities must be strictly increasing, never sorted here"
        )
    origin = specification["bin_grid"]["origin_s"]
    if not np.array_equal(
        bins.bin_start_s, origin + (bins.grid_end_indices - 1) * config.bin_s
    ) or not np.array_equal(
        bins.bin_end_s, origin + bins.grid_end_indices * config.bin_s
    ):
        raise ValueError("Bin timestamps disagree with exact saved grid identities")
    if (
        targets.velocity_mm_s.shape != (t, 2)
        or targets.valid_mask.shape != (t,)
        or len(targets.reasons) != t
    ):
        raise ValueError("Targets, reasons and bins must retain identical row lengths")
    for name in ("source_indices", "source_time_s", "sample_age_s"):
        if getattr(targets, name).shape != (t, 2):
            raise ValueError(f"Mismatched target provenance length: {name}")
    for name in ("elapsed_s", "elapsed_deviation_s"):
        if getattr(targets, name).shape != (t,):
            raise ValueError(f"Mismatched target provenance length: {name}")
    if targets.valid_mask.dtype != np.dtype(bool):
        raise ValueError("Target validity must be an explicit boolean mask")
    if not np.array_equal(
        targets.valid_mask, np.array([r == "valid" for r in targets.reasons])
    ):
        raise ValueError("Target mask/reasons disagree")
    if not np.isfinite(targets.velocity_mm_s[targets.valid_mask]).all():
        raise ValueError("A target declared valid is nonfinite")
    parts = [p for p in specification["partitions"] if p["name"] == bins.partition]
    if len(parts) != 1:
        raise ValueError("Partition must match saved specification")
    expected_segments = parts[0]["continuous_segments"]
    if len(expected_segments) != len(bins.segment_metadata):
        raise ValueError("Segment metadata disagrees with specification")
    retained = parts[0]["retained"]
    covered = 0
    previous_end = retained["start_s"]
    for index, (segment, metadata) in enumerate(
        zip(expected_segments, bins.segment_metadata, strict=True)
    ):
        if (
            not previous_end
            <= segment["start_s"]
            < segment["end_s"]
            <= retained["end_s"]
        ):
            raise ValueError("Segments must be ordered inside their retained partition")
        previous_end = segment["end_s"]
        start, stop = metadata["row_start"], metadata["row_stop"]
        if (
            start != covered
            or not start <= stop <= t
            or metadata["segment_id"] != index
        ):
            raise ValueError("Segment row ranges are not ordered and complete")
        if (
            metadata["start_s"] != segment["start_s"]
            or metadata["end_s"] != segment["end_s"]
        ):
            raise ValueError("Segment bounds changed")
        if not np.all(bins.segment_ids[start:stop] == index):
            raise ValueError("Rows do not match segment identity")
        if np.any(bins.bin_start_s[start:stop] < segment["start_s"]) or np.any(
            bins.bin_end_s[start:stop] >= segment["end_s"]
        ):
            raise ValueError("Source bins cross segment boundaries")
        covered = stop
    if covered != t:
        raise ValueError("Segment metadata does not cover all rows")
    valid = np.array([r in ("valid", "warmup") for r in targets.reasons], dtype=bool)
    if bin_valid_mask is not None:
        if bin_valid_mask.shape != (t,) or bin_valid_mask.dtype != np.dtype(bool):
            raise ValueError("Bin validity must be a boolean [T] mask")
        valid &= bin_valid_mask
    return valid


@dataclass(frozen=True)
class HistoryWindows:
    """Raw [N,L,C] counts, [N,2] mm/s labels and explicit original-bin provenance."""

    X_raw: NDArray[np.int64]
    y: NDArray[np.float64]
    source_rows: NDArray[np.int64]
    source_bin_end_indices: NDArray[np.int64]
    source_bin_start_s: NDArray[np.float64]
    source_bin_end_s: NDArray[np.float64]
    target_rows: NDArray[np.int64]
    target_bin_end_indices: NDArray[np.int64]
    target_time_s: NDArray[np.float64]
    segment_ids: NDArray[np.int64]
    channel_ids: tuple[str, ...]
    unit_indices: tuple[tuple[int, ...], ...]
    partition: str
    identity: tuple[tuple[str, str], ...]
    summary: dict[str, object]


def build_history_windows(
    targets: AlignedTargets,
    specification: dict,
    config: TimingConfig,
    bin_valid_mask: NDArray[np.bool_] | None = None,
) -> HistoryWindows:
    """Never window filtered rows: require L valid contiguous bins and a valid target.

    Warm-up target exclusions still supply valid counts. Other target-quality
    failures conservatively invalidate their source-count row as well. An explicit
    additional mask marks known missing/invalid observations; silence never does.
    """
    source_valid = _validate(targets, specification, config, bin_valid_mask)
    bins = targets.bins
    length = config.history_bins
    accepted, reasons = [], {}
    for row in range(bins.counts.shape[0]):
        first = row - length + 1
        if first < 0 or targets.reasons[row] == "warmup":
            reason = "warmup"
        elif not targets.valid_mask[row]:
            reason = "invalid_target"
        elif (
            np.any(bins.segment_ids[first : row + 1] != bins.segment_ids[row])
            or np.any(np.diff(bins.grid_end_indices[first : row + 1]) != 1)
            or not np.array_equal(
                bins.bin_start_s[first + 1 : row + 1], bins.bin_end_s[first:row]
            )
        ):
            reason = "discontinuous_history"
        elif not source_valid[first : row + 1].all():
            reason = "invalid_history_bin"
        else:
            accepted.append(row)
            continue
        reasons[reason] = reasons.get(reason, 0) + 1
    target_rows = np.array(accepted, dtype=np.int64)
    rows = target_rows[:, None] - np.arange(length - 1, -1, -1)
    return HistoryWindows(
        bins.counts[rows].copy(),
        targets.velocity_mm_s[target_rows].copy(),
        rows,
        bins.grid_end_indices[rows],
        bins.bin_start_s[rows],
        bins.bin_end_s[rows],
        target_rows,
        bins.grid_end_indices[target_rows],
        bins.bin_end_s[target_rows],
        bins.segment_ids[target_rows],
        bins.channel_ids,
        bins.unit_indices,
        bins.partition,
        _identity(specification, targets),
        {
            "count_rows": int(bins.counts.shape[0]),
            "valid_source_bins": int(source_valid.sum()),
            "valid_targets": int(targets.valid_mask.sum()),
            "retained_windows": int(target_rows.size),
            "warmup_windows": reasons.get("warmup", 0),
            "invalid_windows": sum(v for k, v in reasons.items() if k != "warmup"),
            "rejection_reasons": reasons,
        },
    )


def flatten_features(X: NDArray) -> NDArray:
    """[N,L,C] -> [N,L*C], oldest lag first, channels fastest, C-order."""
    if X.ndim != 3:
        raise ValueError("Expected [N,L,C]")
    return X.reshape(X.shape[0], X.shape[1] * X.shape[2], order="C")


def unflatten_features(flat: NDArray, history_bins: int, channels: int) -> NDArray:
    """Inverse of flatten_features, with explicit dimension checks."""
    if (
        flat.ndim != 2
        or history_bins < 1
        or channels < 1
        or flat.shape[1] != history_bins * channels
    ):
        raise ValueError("Flattened dimensions do not match L*C")
    return flat.reshape(flat.shape[0], history_bins, channels, order="C")


@dataclass(frozen=True)
class FrozenCountScaler:
    """Per-channel training means/population scales; zero training std uses scale 1."""

    means: NDArray[np.float64]
    scales: NDArray[np.float64]
    constant_channels: NDArray[np.bool_]
    channel_ids: tuple[str, ...]
    unit_indices: tuple[tuple[int, ...], ...]
    history_bins: int
    identity: tuple[tuple[str, str], ...]
    fitted_bin_count: int
    fitted_bin_identity_sha256: str
    fitted_counts_sha256: str

    def transform(self, windows: HistoryWindows) -> NDArray[np.float64]:
        """Return new normalized values; preserve raw windows/counts and frozen state."""
        if (
            windows.identity != self.identity
            or windows.channel_ids != self.channel_ids
            or windows.unit_indices != self.unit_indices
            or windows.X_raw.shape[1:] != (self.history_bins, len(self.channel_ids))
        ):
            raise ValueError(
                "Scaler source/split/config/channel/window identity mismatch"
            )
        return (windows.X_raw.astype(np.float64) - self.means) / self.scales

    def to_metadata(self) -> dict:
        """Small replay-relevant fit metadata; bulk windows remain unsaved."""
        return {
            "schema_version": 1,
            "identity": dict(self.identity),
            "channel_ids": list(self.channel_ids),
            "unit_indices": [list(u) for u in self.unit_indices],
            "history_bins": self.history_bins,
            "means": self.means.tolist(),
            "scales": self.scales.tolist(),
            "constant_channels": self.constant_channels.tolist(),
            "fitted_bin_count": self.fitted_bin_count,
            "fitted_bin_identity_sha256": self.fitted_bin_identity_sha256,
            "fitted_counts_sha256": self.fitted_counts_sha256,
            "fit_basis": "unique valid training bins, including warm-up; each original bin once",
            "standard_deviation": "population ddof=0; exact zero std replaced by scale=1",
            "formula": "(raw count - training channel mean) / frozen training scale",
            "window_order": "oldest to newest; channels fastest in C-order flattening",
            "units": "means/scales in events per bin; normalized features dimensionless; y remains mm/s",
            "source_validity": "valid or warmup reasons; other quality failures and explicit invalid-bin masks excluded",
        }


def fit_count_scaler(
    training: AlignedTargets,
    specification: dict,
    config: TimingConfig,
    bin_valid_mask: NDArray[np.bool_] | None = None,
) -> FrozenCountScaler:
    """Offline fit over unique training rows only, never repeated window entries.

    Refitting on changed training data may change statistics. Once fitted, future
    observations/held-out values cannot update them; transform has no fit path.
    """
    if training.bins.partition != "train":
        raise ValueError("Scaler must be fitted on training partition only")
    valid = _validate(training, specification, config, bin_valid_mask)
    if not valid.any():
        raise ValueError("No valid training bins to fit")
    values = training.bins.counts[valid].astype(np.float64)
    means = values.mean(axis=0)
    std = values.std(axis=0, ddof=0)
    constant = std == 0
    scales = np.where(constant, 1.0, std)
    for array in (means, scales, constant):
        array.setflags(write=False)
    identities = np.column_stack(
        (training.bins.segment_ids[valid], training.bins.grid_end_indices[valid])
    )
    return FrozenCountScaler(
        means,
        scales,
        constant,
        training.bins.channel_ids,
        training.bins.unit_indices,
        config.history_bins,
        _identity(specification, training),
        int(valid.sum()),
        hashlib.sha256(identities.astype("<i8").tobytes()).hexdigest(),
        hashlib.sha256(training.bins.counts[valid].astype("<i8").tobytes()).hexdigest(),
    )
