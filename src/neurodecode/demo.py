"""Validation cursor presentation: audited endpoint integration, never inference."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class IntegratedPath:
    """Anchor at first bin start, then one point at each completed-bin end."""

    time_s: NDArray[np.float64]
    position_mm: NDArray[np.float64]


def integrate_velocity(
    velocity_mm_s: NDArray,
    bin_start_s: NDArray,
    bin_end_s: NDArray,
    bin_ids: NDArray,
    segment_ids: NDArray,
    initial_position_mm: NDArray,
    available: NDArray | None = None,
) -> IntegratedPath:
    """Integrate each velocity over its own actual bin; fail on any gap/missingness.

    No reference position is accepted, so decoded paths cannot be corrected by it.
    Inputs must already share the verified source/partition/channel context.
    """
    n = len(bin_end_s)
    if n == 0 or velocity_mm_s.shape != (n, 2):
        raise ValueError("Need nonempty velocities [N,2]")
    for values in (bin_start_s, bin_end_s, bin_ids, segment_ids):
        if values.shape != (n,):
            raise ValueError("Timeline/identity arrays must be [N]")
    if bin_ids.dtype.kind not in "iu" or segment_ids.dtype.kind not in "iu":
        raise ValueError("Bin and segment identities must be integers")
    if initial_position_mm.shape != (2,) or not np.isfinite(initial_position_mm).all():
        raise ValueError("Need a finite common initial position [2]")
    if (
        not np.isfinite(velocity_mm_s).all()
        or not np.isfinite(bin_start_s).all()
        or not np.isfinite(bin_end_s).all()
    ):
        raise ValueError("Missing/nonfinite velocity or timestamp; no interpolation")
    if available is not None and (
        available.shape != (n,)
        or available.dtype != np.dtype(bool)
        or not available.all()
    ):
        raise ValueError("Unavailable bins cannot be silently integrated")
    duration_s = bin_end_s - bin_start_s
    if (
        np.any(duration_s <= 0)
        or np.any(np.diff(bin_ids) != 1)
        or np.any(segment_ids != segment_ids[0])
        or not np.array_equal(bin_start_s[1:], bin_end_s[:-1])
    ):
        raise ValueError(
            "Discontinuous/unordered bins: refuse clip, no bridge or reset"
        )
    positions = np.vstack(
        (
            initial_position_mm.copy(),
            initial_position_mm
            + np.cumsum(velocity_mm_s * duration_s[:, None], axis=0),
        )
    )
    if not np.isfinite(positions).all():
        raise ValueError("Nonfinite integrated position")
    return IntegratedPath(np.concatenate((bin_start_s[:1], bin_end_s)), positions)


def audit_tracker_alignment(
    starts: NDArray,
    ends: NDArray,
    target_velocity: NDArray,
    source_times: NDArray,
    source_indices: NDArray,
    source_positions: NDArray,
) -> dict:
    """Check sampled tracker differences against labels and neural-bin durations."""
    n = len(ends)
    if (
        starts.shape != (n,)
        or target_velocity.shape != (n, 2)
        or source_times.shape != (n, 2)
        or source_indices.shape != (n, 2)
        or source_positions.shape != (n, 2, 2)
        or not all(
            np.isfinite(v).all()
            for v in (starts, ends, target_velocity, source_times, source_positions)
        )
    ):
        raise ValueError("Malformed tracker alignment audit inputs")
    neural_dt, behavior_dt = ends - starts, np.diff(source_times, axis=1)[:, 0]
    if np.any(neural_dt <= 0) or np.any(behavior_dt <= 0):
        raise ValueError("Nonpositive bin/source duration")
    if np.any(source_times[:, 0] > starts) or np.any(source_times[:, 1] > ends):
        raise ValueError("Tracker sample comes from after its stated bin edge")
    reconstructed = (source_positions[:, 1] - source_positions[:, 0]) / behavior_dt[
        :, None
    ]
    np.testing.assert_array_equal(reconstructed, target_velocity)
    chain_breaks = int(
        np.count_nonzero(source_indices[1:, 0] != source_indices[:-1, 1])
    )
    exact = bool(
        np.array_equal(source_times[:, 0], starts)
        and np.array_equal(source_times[:, 1], ends)
        and chain_breaks == 0
    )
    integrated = source_positions[0, 0] + np.cumsum(
        target_velocity * neural_dt[:, None], axis=0
    )
    return {
        "samples": n,
        "target_velocity_reconstruction": "exact equality",
        "exact_tracker_grid_alignment": exact,
        "behavior_elapsed_range_s": [
            float(behavior_dt.min()),
            float(behavior_dt.max()),
        ],
        "neural_bin_duration_range_s": [float(neural_dt.min()), float(neural_dt.max())],
        "duration_mismatches_over_1ns": int(
            np.count_nonzero(np.abs(behavior_dt - neural_dt) > 1e-9)
        ),
        "maximum_duration_difference_s": float(np.abs(behavior_dt - neural_dt).max()),
        "maximum_source_edge_age_s": float(
            np.max(np.column_stack((starts, ends)) - source_times)
        ),
        "source_endpoint_chain_breaks": chain_breaks,
        "maximum_target_integration_vs_sampled_tracker_coordinate_difference_mm": float(
            np.abs(integrated - source_positions[:, 1]).max()
        ),
        "reference_label": "tracker-derived reference velocity integration",
        "reference_policy": "always display saved target velocity integration over neural-bin durations; do not present it as original tracker positions",
    }
