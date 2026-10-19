"""Training-only 2-D Kalman parameter estimation; no filtering loop."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from neurodecode.data.features import FrozenCountScaler, HistoryWindows
from neurodecode.evaluation import (
    ExampleIndex,
    VelocitySeries,
    index_from_windows,
    require_matching_indices,
)
from neurodecode.splits import TimingConfig


def normalization_sha256(scaler: FrozenCountScaler) -> str:
    """Same canonical metadata encoding used by the existing ridge artifact."""
    return hashlib.sha256(
        json.dumps(
            scaler.to_metadata(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


@dataclass(frozen=True)
class KalmanFitConfig:
    """Fixed fitting assumptions; state variance floor is in (mm/s)^2."""

    observation: str = "standardized_current_bin"
    observation_covariance: str = "diagonal"
    state_variance_floor_mm2_s2: float = 1e-6
    observation_variance_floor: float = 1e-8
    relative_variance_floor: float = 1e-8

    def validate(self) -> None:
        if (
            self.observation != "standardized_current_bin"
            or self.observation_covariance != "diagonal"
        ):
            raise ValueError("Unsupported Kalman observation/covariance policy")
        values = (
            self.state_variance_floor_mm2_s2,
            self.observation_variance_floor,
            self.relative_variance_floor,
        )
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError("Variance floors must be finite and positive")


@dataclass(frozen=True)
class CurrentBinObservations:
    """Label-free [N,C] current-bin observations with explicit temporal identity."""

    values: NDArray[np.float64]
    index: ExampleIndex
    bin_start_s: NDArray[np.float64]
    session_ids: tuple[str, ...]
    grid_origin_s: float
    bin_width_s: float
    normalization_sha256: str

    def validate(self, scaler: FrozenCountScaler) -> None:
        self.index.validate()
        n, c = self.index.time_s.size, len(self.index.channel_ids)
        if self.values.shape != (n, c) or not np.isfinite(self.values).all():
            raise ValueError("Neural observations must be finite [N,C]")
        if len(self.session_ids) != n or any(
            not isinstance(s, str) or not s for s in self.session_ids
        ):
            raise ValueError("Every observation requires an explicit session identity")
        if (
            not np.isfinite(self.grid_origin_s)
            or not np.isfinite(self.bin_width_s)
            or self.bin_width_s <= 0
        ):
            raise ValueError("Grid origin/width must be finite and width positive")
        ids = self.index.bin_ids
        if not np.array_equal(
            self.index.time_s, self.grid_origin_s + ids * self.bin_width_s
        ) or not np.array_equal(
            self.bin_start_s, self.grid_origin_s + (ids - 1) * self.bin_width_s
        ):
            raise ValueError(
                "Observation edges disagree with exact saved grid identities"
            )
        if (
            self.index.identity != scaler.identity
            or self.index.channel_ids != scaler.channel_ids
            or self.index.unit_indices != scaler.unit_indices
            or self.normalization_sha256 != normalization_sha256(scaler)
        ):
            raise ValueError("Observation source/channel/transform identity mismatch")


def current_bin_observations(
    windows: HistoryWindows,
    scaler: FrozenCountScaler,
    session: str,
    grid_origin_s: float,
    bin_width_s: float,
) -> CurrentBinObservations:
    """Reuse existing frozen normalization, selecting the final bin, never labels."""
    normalized = scaler.transform(windows)
    result = CurrentBinObservations(
        normalized[:, -1, :].copy(),
        index_from_windows(windows),
        windows.source_bin_start_s[:, -1].copy(),
        (session,) * windows.X_raw.shape[0],
        grid_origin_s,
        bin_width_s,
        normalization_sha256(scaler),
    )
    result.validate(scaler)
    return result


def consecutive_training_pairs(
    observations: CurrentBinObservations,
) -> NDArray[np.int64]:
    """Return original previous/current row indices, never stitch filtered rows."""
    index = observations.index
    index.validate()
    if index.partition != "train":
        raise ValueError("Kalman estimation accepts training partition only")
    n = index.time_s.size
    if len(observations.session_ids) != n or observations.bin_start_s.shape != (n,):
        raise ValueError("Session/edge metadata length mismatch")
    valid = (
        index.score_mask[:-1]
        & index.score_mask[1:]
        & (index.bin_ids[1:] == index.bin_ids[:-1] + 1)
        & (index.segment_ids[1:] == index.segment_ids[:-1])
        & (observations.bin_start_s[1:] == index.time_s[:-1])
        & np.array(
            [
                a == b
                for a, b in zip(
                    observations.session_ids[:-1],
                    observations.session_ids[1:],
                    strict=True,
                )
            ],
            dtype=bool,
        )
    )
    current = np.flatnonzero(valid) + 1
    return np.column_stack((current - 1, current)).astype(np.int64)


def covariance_diagnostics(matrix: NDArray[np.float64]) -> dict:
    """Report finite/symmetry/eigen/rank evidence without filling undefined condition."""
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or not np.isfinite(matrix).all()
    ):
        raise ValueError("Covariance must be finite and square")
    symmetry = float(np.max(np.abs(matrix - matrix.T)))
    eigenvalues = np.linalg.eigvalsh((matrix + matrix.T) / 2)
    tolerance = (
        32
        * np.finfo(float).eps
        * matrix.shape[0]
        * max(1.0, float(np.max(np.abs(eigenvalues))))
    )
    if symmetry > tolerance or eigenvalues[0] < -tolerance:
        raise ValueError(
            "Covariance is not symmetric positive semidefinite within roundoff"
        )
    return {
        "shape": list(matrix.shape),
        "finite": True,
        "symmetry_max_abs": symmetry,
        "minimum_eigenvalue": float(eigenvalues[0]),
        "maximum_eigenvalue": float(eigenvalues[-1]),
        "numerical_rank": int(np.count_nonzero(eigenvalues > tolerance)),
        "psd_tolerance": float(tolerance),
        "positive_semidefinite": True,
        "condition_number": None
        if eigenvalues[0] <= tolerance
        else float(eigenvalues[-1] / eigenvalues[0]),
    }


def _covariance(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Centered population residual covariance, denominator M (ddof=0)."""
    centered = values - values.mean(axis=0)
    with np.errstate(over="ignore", invalid="ignore"):
        covariance = centered.T @ centered / len(values)
    covariance = (covariance + covariance.T) / 2
    covariance_diagnostics(covariance)
    return covariance


def _regularize_state(
    covariance: NDArray[np.float64], config: KalmanFitConfig
) -> tuple[NDArray, dict]:
    before = covariance_diagnostics(covariance)
    floor = max(
        config.state_variance_floor_mm2_s2,
        config.relative_variance_floor * float(np.trace(covariance)) / len(covariance),
    )
    jitter = max(0.0, floor - before["minimum_eigenvalue"])
    result = covariance + jitter * np.eye(len(covariance))
    after = covariance_diagnostics(result)
    if after["minimum_eigenvalue"] < floor - after["psd_tolerance"]:
        raise ValueError("State covariance regularization failed")
    return result, {
        "raw": before,
        "variance_floor": floor,
        "diagonal_jitter_added": jitter,
        "regularized": after,
    }


def _fit_affine(
    x: NDArray, y: NDArray, name: str
) -> tuple[NDArray, NDArray, NDArray, dict]:
    """Library least squares on centered two-state predictors; intercept separate."""
    x_mean, y_mean = x.mean(axis=0), y.mean(axis=0)
    coefficient, _, rank, singular_values = np.linalg.lstsq(
        x - x_mean, y - y_mean, rcond=None
    )
    if rank != 2:
        raise ValueError(
            f"{name} needs rank-2 velocity predictors; observed rank {rank}"
        )
    intercept = y_mean - x_mean @ coefficient
    residual = y - (x @ coefficient + intercept)
    if not all(np.isfinite(a).all() for a in (coefficient, intercept, residual)):
        raise ValueError(f"{name} fitting produced nonfinite values")
    return (
        coefficient.T,
        intercept,
        residual,
        {
            "centered_predictor_rank": int(rank),
            "singular_values": singular_values.tolist(),
            "condition_number": float(singular_values[0] / singular_values[-1]),
            "residual_mean": residual.mean(axis=0).tolist(),
        },
    )


@dataclass(frozen=True)
class KalmanParameters:
    """Estimated matrices in column-state convention, with training-only prior."""

    A: NDArray[np.float64]
    a: NDArray[np.float64]
    H: NDArray[np.float64]
    d: NDArray[np.float64]
    Q: NDArray[np.float64]
    R: NDArray[np.float64]
    initial_mean: NDArray[np.float64]
    initial_covariance: NDArray[np.float64]
    scaler: FrozenCountScaler
    config: KalmanFitConfig
    diagnostics: dict

    def validate(self) -> None:
        self.config.validate()
        c = len(self.scaler.channel_ids)
        shapes = {
            "A": (2, 2),
            "a": (2,),
            "H": (c, 2),
            "d": (c,),
            "Q": (2, 2),
            "R": (c, c),
            "initial_mean": (2,),
            "initial_covariance": (2, 2),
        }
        for name, shape in shapes.items():
            value = getattr(self, name)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"Invalid {name} shape/value")
        for name in ("Q", "R", "initial_covariance"):
            diagnostic = covariance_diagnostics(getattr(self, name))
            if diagnostic["minimum_eigenvalue"] <= 0:
                raise ValueError(
                    f"{name} must be positive definite after regularization"
                )

    def save(self, path: str | Path) -> None:
        """Save explicit fitted parameters/configuration; no filtering state or pickle."""
        self.validate()
        metadata = {
            "schema_version": 1,
            "model": "kalman_parameters_only",
            "matrices": {
                name: getattr(self, name).tolist()
                for name in (
                    "A",
                    "a",
                    "H",
                    "d",
                    "Q",
                    "R",
                    "initial_mean",
                    "initial_covariance",
                )
            },
            "state": "completed-bin [vx,vy] in mm/s",
            "observation": "frozen-training standardized current-bin channel counts",
            "matrix_convention": "column state: s_t=A@s_previous+a+w; z_t=H@s_t+d+epsilon",
            "covariance_units": {
                "Q": "(mm/s)^2",
                "R": "standardized count squared",
                "initial_covariance": "(mm/s)^2",
            },
            "initial_state_policy": {
                "mean": "eligible training velocity mean",
                "covariance": "eligible training velocity population covariance plus declared floor if needed",
                "first_bin": "training prior at first eligible bin before its observation; no extra transition",
                "reset": "reuse this prior at every session/partition/segment start and eligibility discontinuity",
                "held_out_behavior_input": False,
            },
            "normalization": self.scaler.to_metadata(),
            "normalization_sha256": normalization_sha256(self.scaler),
            "configuration": asdict(self.config),
            "diagnostics": self.diagnostics,
        }
        with Path(path).open("x", encoding="utf-8") as file:
            file.write(json.dumps(metadata, indent=2, allow_nan=False) + chr(10))


def fit_kalman_parameters(
    observations: CurrentBinObservations,
    states: VelocitySeries,
    scaler: FrozenCountScaler,
    timing: TimingConfig,
    config: KalmanFitConfig,
) -> KalmanParameters:
    """Fit A/a/H/d/Q/R only, on matched valid training rows/pairs."""
    config.validate()
    observations.validate(scaler)
    states.validate()
    require_matching_indices(observations.index, states.index)
    if (
        observations.bin_width_s != timing.bin_s
        or scaler.history_bins != timing.history_bins
    ):
        raise ValueError("Kalman timing disagrees with preparation")
    pairs = consecutive_training_pairs(observations)
    valid = observations.index.score_mask
    if np.count_nonzero(valid) < 4 or len(pairs) < 4:
        raise ValueError(
            "Need at least four valid training states and consecutive transitions"
        )
    previous, current = pairs.T
    A, a, transition_residual, transition_fit = _fit_affine(
        states.values_mm_s[previous], states.values_mm_s[current], "Transition"
    )
    H, d, observation_residual, observation_fit = _fit_affine(
        states.values_mm_s[valid], observations.values[valid], "Observation"
    )
    Q, q_diagnostics = _regularize_state(_covariance(transition_residual), config)
    raw_r = _covariance(observation_residual)
    raw_variance = np.diag(raw_r)
    r_floor = max(
        config.observation_variance_floor,
        config.relative_variance_floor * float(raw_variance.mean()),
    )
    variance = np.maximum(raw_variance, r_floor)
    R = np.diag(variance)
    initial_mean = states.values_mm_s[valid].mean(axis=0)
    initial_covariance, p_diagnostics = _regularize_state(
        _covariance(states.values_mm_s[valid]), config
    )
    index = observations.index
    pair_identities = np.column_stack(
        (index.bin_ids[previous], index.bin_ids[current], index.segment_ids[current])
    ).astype("<i8")
    diagnostics = {
        "valid_training_observations": int(valid.sum()),
        "candidate_adjacent_rows": max(0, len(valid) - 1),
        "eligible_training_transitions": int(len(pairs)),
        "excluded_adjacent_rows": max(0, len(valid) - 1) - int(len(pairs)),
        "sessions": sorted(set(observations.session_ids)),
        "first_pair_bin_ids": pair_identities[0, :2].tolist(),
        "last_pair_bin_ids": pair_identities[-1, :2].tolist(),
        "first_training_time_s": float(index.time_s[valid][0]),
        "last_training_time_s": float(index.time_s[valid][-1]),
        "training_index_sha256": hashlib.sha256(
            np.column_stack((index.bin_ids[valid], index.segment_ids[valid]))
            .astype("<i8")
            .tobytes()
            + index.time_s[valid].astype("<f8").tobytes()
            + json.dumps(
                [s for s, v in zip(observations.session_ids, valid, strict=True) if v]
            ).encode()
        ).hexdigest(),
        "transition_pair_identity_sha256": hashlib.sha256(
            pair_identities.tobytes()
            + json.dumps([observations.session_ids[i] for i in current]).encode()
        ).hexdigest(),
        "transition_fit": transition_fit,
        "observation_fit": observation_fit,
        "A_spectral_radius": float(np.max(np.abs(np.linalg.eigvals(A)))),
        "Q": q_diagnostics,
        "R": {
            "raw_full_residual_covariance": covariance_diagnostics(raw_r),
            "assumption": "diagonal covariance; cross-channel residual covariance not retained",
            "discarded_offdiagonal_frobenius_norm": float(
                np.linalg.norm(raw_r - np.diag(raw_variance))
            ),
            "variance_floor": r_floor,
            "floored_channels": [
                channel
                for channel, raw in zip(index.channel_ids, raw_variance, strict=True)
                if raw < r_floor
            ],
            "maximum_variance_added": float(np.max(variance - raw_variance)),
            "regularized": covariance_diagnostics(R),
        },
        "initial_covariance": p_diagnostics,
        "covariance_denominator": "population M (ddof=0), centered residuals; no degrees-of-freedom correction",
        "grid_origin_s": observations.grid_origin_s,
        "bin_width_s": observations.bin_width_s,
        "no_filtering_or_decoding_scores": True,
    }
    result = KalmanParameters(
        A, a, H, d, Q, R, initial_mean, initial_covariance, scaler, config, diagnostics
    )
    result.validate()
    for name in ("A", "a", "H", "d", "Q", "R", "initial_mean", "initial_covariance"):
        getattr(result, name).setflags(write=False)
    return result
