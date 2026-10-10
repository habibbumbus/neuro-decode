"""Plan raw chronological partitions and support bounds; no feature/target arrays."""

import tomllib
from dataclasses import asdict, dataclass
from math import isclose, isfinite
from pathlib import Path

import numpy as np

COMPARISON_TOLERANCE_S = 1e-9


@dataclass(frozen=True)
class TimingConfig:
    bin_ms: float = 20
    history_bins: int = 10
    additional_lag_ms: float = 0
    forecast_horizon_ms: float = 0
    max_behavior_age_ms: float = 6
    behavior_gap_ms: float = 6

    def __post_init__(self) -> None:
        for name in ("bin_ms", "max_behavior_age_ms", "behavior_gap_ms"):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if type(self.history_bins) is not int or self.history_bins < 1:
            raise ValueError("history_bins must be a positive integer")
        if self.additional_lag_ms != 0 or self.forecast_horizon_ms != 0:
            raise ValueError(
                "Phase 2A supports zero additional lag and forecast horizon"
            )

    @property
    def bin_s(self) -> float:
        return self.bin_ms / 1000

    @property
    def history_s(self) -> float:
        return self.history_bins * self.bin_s

    @property
    def lookback_s(self) -> float:
        """Union of neural history and worst-case behavioral target source support."""
        return max(
            self.history_s,
            self.bin_s + self.max_behavior_age_ms / 1000 + COMPARISON_TOLERANCE_S,
        )


@dataclass(frozen=True)
class Interval:
    """All partition, segment, and prediction-ownership intervals are [start, end)."""

    start_s: float
    end_s: float

    def __post_init__(self) -> None:
        if not all(map(isfinite, (self.start_s, self.end_s))):
            raise ValueError("Interval endpoints must be finite")
        if self.end_s <= self.start_s:
            raise ValueError("Interval must have positive duration")

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def contains(self, timestamp_s: float) -> bool:
        return self.start_s <= timestamp_s < self.end_s


def read_config(path: str | Path) -> tuple[TimingConfig, tuple[float, ...]]:
    with Path(path).open("rb") as file:
        config = tomllib.load(file)
    return TimingConfig(**config["timing"]), tuple(
        config["split"][name]
        for name in ("train_fraction", "validation_fraction", "test_fraction")
    )


def _validate_times(times: np.ndarray) -> None:
    if times.ndim != 1 or times.size < 2 or not np.isfinite(times).all():
        raise ValueError("Behavior timestamps must be a finite vector with >=2 samples")
    if np.any(np.diff(times) <= 0):
        raise ValueError("Behavior timestamps must be strictly increasing; no repair")


def prediction_support(
    end_s: float, segment: Interval, config: TimingConfig
) -> dict[str, object]:
    """Conservative support envelope, not an eligible index or computed velocity."""
    if not isfinite(end_s) or not segment.contains(end_s):
        raise ValueError("Prediction end must belong to its half-open segment")
    earliest = end_s - config.lookback_s
    if earliest < segment.start_s:
        raise ValueError("Incomplete warm-up: source support crosses segment start")
    return {
        "feature_interval_s": [end_s - config.history_s, end_s],
        "feature_endpoint_convention": "(start, end]",
        "target_nominal_interval_s": [end_s - config.bin_s, end_s],
        "target_source_envelope_s": [
            end_s
            - config.bin_s
            - config.max_behavior_age_ms / 1000
            - COMPARISON_TOLERANCE_S,
            end_s,
        ],
        "prediction_available_at_or_after_s": end_s,
    }


def select_behavior_samples(
    times: np.ndarray, end_s: float, segment: Interval, config: TimingConfig
) -> tuple[int, int]:
    """Select two past-or-present samples only; do not read positions or derive velocity.

    The 1 ns tolerance applies to age/gap comparisons only, never to selecting a
    future sample or extending partition ownership. Reject gaps in all support.
    """
    _validate_times(times)
    prediction_support(end_s, segment, config)
    boundaries = (end_s - config.bin_s, end_s)
    selected = tuple(
        int(np.searchsorted(times, b, side="right")) - 1 for b in boundaries
    )
    for index, boundary in zip(selected, boundaries, strict=True):
        if index < 0 or not segment.contains(float(times[index])):
            raise ValueError("Behavior sample unavailable inside the same segment")
        if (
            boundary - times[index]
            > config.max_behavior_age_ms / 1000 + COMPARISON_TOLERANCE_S
        ):
            raise ValueError("Behavior sample is too old")
    left = end_s - config.lookback_s
    crossing = (times[:-1] < end_s) & (times[1:] > left)
    if np.any(
        crossing
        & (np.diff(times) > config.behavior_gap_ms / 1000 + COMPARISON_TOLERANCE_S)
    ):
        raise ValueError("Behavior gap crosses prediction support; reset required")
    return selected


def plan_splits(
    times: np.ndarray,
    neural_range_s: tuple[float, float] | list[float],
    config: TimingConfig,
    fractions: tuple[float, ...] = (0.70, 0.15, 0.15),
) -> dict[str, object]:
    """Use timestamp/coverage metadata only; proportions are elapsed overlap time.

    Gaps do not move nominal boundaries. Every continuous segment is intersected
    with each purged partition. Future construction must revalidate actual sample
    ages and source support; this plan deliberately contains no eligible indices.
    Conservatively require one segment longer than lookback plus one bin in every
    partition, guaranteeing temporal room regardless of grid phase (not eligibility).
    """
    _validate_times(times)
    if len(fractions) != 3 or any(not isfinite(f) or f <= 0 for f in fractions):
        raise ValueError("Three finite positive split fractions required")
    if not isclose(sum(fractions), 1, rel_tol=0, abs_tol=1e-12):
        raise ValueError("Split fractions must sum to one")
    if neural_range_s is None or len(neural_range_s) != 2:
        raise ValueError("Neural event coverage required; silence is not missing data")
    neural = Interval(*neural_range_s)
    overlap = Interval(
        max(float(times[0]), neural.start_s), min(float(times[-1]), neural.end_s)
    )
    cut1 = overlap.start_s + fractions[0] * overlap.duration_s
    cut2 = overlap.start_s + (fractions[0] + fractions[1]) * overlap.duration_s
    edges = [overlap.start_s, cut1, cut2, overlap.end_s]
    purge = config.lookback_s
    gap_indices = np.flatnonzero(
        np.diff(times) > config.behavior_gap_ms / 1000 + COMPARISON_TOLERANCE_S
    )
    continuous = []
    first = 0
    gaps = []
    for index in gap_indices:
        if times[index] > times[first]:
            continuous.append(Interval(float(times[first]), float(times[index])))
        gaps.append([float(times[index]), float(times[index + 1])])
        first = int(index + 1)
    if times[-1] > times[first]:
        continuous.append(Interval(float(times[first]), float(times[-1])))
    partitions = []
    for i, name in enumerate(("train", "validation", "test")):
        nominal = Interval(edges[i], edges[i + 1])
        start = nominal.start_s + (purge if i else 0)
        end = nominal.end_s - (purge if i < 2 else 0)
        if end <= start:
            raise ValueError(f"Insufficient recording length after purge: {name}")
        retained = Interval(start, end)
        segments = []
        for raw_segment in continuous:
            a = max(raw_segment.start_s, retained.start_s)
            b = min(raw_segment.end_s, retained.end_s)
            if b > a:
                segment = Interval(a, b)
                segments.append(
                    {
                        **asdict(segment),
                        "warmup_s": config.lookback_s,
                        "earliest_support_complete_end_s": a + config.lookback_s,
                        "has_room_for_grid_window": segment.duration_s
                        > config.lookback_s + config.bin_s,
                    }
                )
        if not any(s["has_room_for_grid_window"] for s in segments):
            raise ValueError(
                f"Insufficient continuous duration for complete windows: {name}"
            )
        continuous_duration = sum(s["end_s"] - s["start_s"] for s in segments)
        partitions.append(
            {
                "name": name,
                "nominal": asdict(nominal),
                "retained": asdict(retained),
                "nominal_duration_s": nominal.duration_s,
                "purge_discarded_s": nominal.duration_s - retained.duration_s,
                "gap_discarded_inside_retained_s": retained.duration_s
                - continuous_duration,
                "continuous_segments": segments,
            }
        )
    return {
        "schema_version": 1,
        "timing": asdict(config),
        "age_gap_comparison_tolerance_s": COMPARISON_TOLERANCE_S,
        "fractions": dict(zip(("train", "validation", "test"), fractions, strict=True)),
        "proportion_basis": "elapsed global neural/behavior overlap; gaps do not move boundaries",
        "overlap": asdict(overlap),
        "bin_grid": {
            "origin_s": overlap.start_s,
            "width_s": config.bin_s,
            "end_rule": "origin + k * width, integer k >= 1",
        },
        "endpoint_convention": "Partitions/segments [start, end); spike bins (start, end]; terminal overlap endpoint excluded",
        "purge_per_side_s": purge,
        "purge_derivation": "max(history_bins * bin_s, bin_s + max_behavior_age_s + age_tolerance_s) on EACH side of each nominal boundary",
        "purge_intervals_s": [
            [cut1 - purge, cut1 + purge],
            [cut2 - purge, cut2 + purge],
        ],
        "behavior_gaps_s": gaps,
        "gap_policy": "Intervals > fixed behavior_gap_ms + 1 ns split segments; exclude [last_before, first_after), no bridging or repair",
        "reset_policy": "Reset history and future recurrent/Kalman state at session, partition, and segment starts; no held-out behavioral initialization",
        "support_rule": "All spike events and selected behavioral samples inside same retained partition and continuous segment; full conservative support envelope required",
        "behavior_selection": "Latest stored timestamp <= each boundary via searchsorted(side=right)-1; age <= max_behavior_age_ms + 1 ns; no future sample tolerance",
        "warmup_s": config.lookback_s,
        "prediction_availability": "At or after bin end plus actual computation delay; completed-interval velocity estimation, not forecasting",
        "eligible_feature_indices": None,
        "eligibility_status": "Pending binning/target construction and actual source-support checks; segment warm-up bounds are not feature indices",
        "partitions": partitions,
    }
