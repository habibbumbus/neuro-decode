"""Train-only library ridge fitting and portable, identity-bound parameters."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sklearn
from numpy.typing import NDArray
from sklearn.linear_model import Ridge

from neurodecode.data.features import (
    FrozenCountScaler,
    HistoryWindows,
    flatten_features,
)
from neurodecode.evaluation import (
    ExampleIndex,
    VelocitySeries,
    index_from_windows,
    require_matching_indices,
)

FEATURE_ORDER = "C-order: oldest bin first; channels fastest"


def scaler_sha256(scaler: FrozenCountScaler) -> str:
    """Hash full fit metadata, including ordered means/scales and training-row hashes."""
    return hashlib.sha256(
        json.dumps(
            scaler.to_metadata(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _validate_scaler(scaler: FrozenCountScaler) -> None:
    c = len(scaler.channel_ids)
    if c < 1 or scaler.history_bins < 1 or scaler.fitted_bin_count < 1:
        raise ValueError("Scaler dimensions/fit count must be positive")
    if len(set(scaler.channel_ids)) != c or len(scaler.unit_indices) != c:
        raise ValueError("Scaler channel/unit mapping is invalid")
    for array in (scaler.means, scaler.scales, scaler.constant_channels):
        if array.shape != (c,):
            raise ValueError("Scaler arrays must have shape [C]")
    if (
        not np.isfinite(scaler.means).all()
        or not np.isfinite(scaler.scales).all()
        or np.any(scaler.scales <= 0)
        or scaler.constant_channels.dtype != np.dtype(bool)
    ):
        raise ValueError("Scaler values must be finite, scales positive, flags boolean")
    if np.any(scaler.scales[scaler.constant_channels] != 1):
        raise ValueError("Constant-channel scaler must retain scale 1")


@dataclass(frozen=True)
class RidgeInputs:
    """Label-free normalized [N,L*C] features with exact prediction identities."""

    values: NDArray[np.float64]
    index: ExampleIndex
    history_bins: int
    normalization_sha256: str
    feature_order: str = FEATURE_ORDER

    def validate(self, scaler: FrozenCountScaler) -> None:
        self.index.validate()
        _validate_scaler(scaler)
        if (
            self.values.shape
            != (self.index.time_s.size, scaler.history_bins * len(scaler.channel_ids))
            or not np.isfinite(self.values).all()
        ):
            raise ValueError("Ridge features must be finite [N,L*C]")
        if (
            self.history_bins != scaler.history_bins
            or self.feature_order != FEATURE_ORDER
            or self.index.channel_ids != scaler.channel_ids
            or self.index.unit_indices != scaler.unit_indices
            or self.index.identity != scaler.identity
            or self.normalization_sha256 != scaler_sha256(scaler)
        ):
            raise ValueError("Ridge feature/channel/transform identity mismatch")


def ridge_inputs(windows: HistoryWindows, scaler: FrozenCountScaler) -> RidgeInputs:
    """Use the existing frozen transform and flattening; never read window labels."""
    result = RidgeInputs(
        flatten_features(scaler.transform(windows)),
        index_from_windows(windows),
        scaler.history_bins,
        scaler_sha256(scaler),
    )
    result.validate(scaler)
    return result


@dataclass(frozen=True)
class RidgeDecoder:
    """Weights [L*C,2], intercept [2] mm/s, and the frozen training transform."""

    weights: NDArray[np.float64]
    intercept: NDArray[np.float64]
    alpha: float
    scaler: FrozenCountScaler
    fitted_examples: int
    fit_library_version: str

    @classmethod
    def fit(
        cls,
        features: RidgeInputs,
        targets: VelocitySeries,
        scaler: FrozenCountScaler,
        alpha: float,
    ) -> "RidgeDecoder":
        """Fit training rows only using sklearn SVD; intercept is unpenalized."""
        features.validate(scaler)
        targets.validate()
        require_matching_indices(features.index, targets.index)
        if features.index.partition != "train":
            raise ValueError("Ridge weights must be fitted on training only")
        if features.values.shape[0] == 0 or not features.index.score_mask.all():
            raise ValueError(
                "Fit requires all eligible training rows without extra masks"
            )
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("Ridge alpha must be finite and positive")
        estimator = Ridge(alpha=alpha, solver="svd", fit_intercept=True, copy_X=True)
        estimator.fit(features.values, targets.values_mm_s)
        weights = estimator.coef_.T.copy()
        intercept = estimator.intercept_.copy()
        for array in (weights, intercept):
            array.setflags(write=False)
        result = cls(
            weights,
            intercept,
            float(alpha),
            scaler,
            features.values.shape[0],
            sklearn.__version__,
        )
        result.validate()
        return result

    def validate(self) -> None:
        _validate_scaler(self.scaler)
        d = self.scaler.history_bins * len(self.scaler.channel_ids)
        if self.weights.shape != (d, 2) or self.intercept.shape != (2,):
            raise ValueError("Ridge parameter shape mismatch")
        if not np.isfinite(self.weights).all() or not np.isfinite(self.intercept).all():
            raise ValueError("Ridge parameters must be finite")
        if not np.isfinite(self.alpha) or self.alpha <= 0 or self.fitted_examples < 1:
            raise ValueError("Invalid ridge alpha/fit count")

    def predict(self, features: RidgeInputs) -> VelocitySeries:
        """Predict with frozen parameters; no labels, model selection or refit."""
        self.validate()
        features.validate(self.scaler)
        prediction = VelocitySeries(
            features.values @ self.weights + self.intercept, features.index
        )
        prediction.validate()
        return prediction

    def save(self, path: str | Path) -> None:
        """Exclusive JSON write: explicit numeric parameters, no pickle execution."""
        self.validate()
        metadata = {
            "schema_version": 1,
            "model": "ridge",
            "alpha": self.alpha,
            "solver": "svd",
            "fit_intercept": True,
            "intercept_penalized": False,
            "feature_order": FEATURE_ORDER,
            "weights_layout": "[L*C,2]; columns vx,vy",
            "output_units": "mm/s",
            "weights": self.weights.tolist(),
            "intercept": self.intercept.tolist(),
            "scaler": self.scaler.to_metadata(),
            "normalization_sha256": scaler_sha256(self.scaler),
            "fitted_examples": self.fitted_examples,
            "fit_library": "scikit-learn",
            "fit_library_version": self.fit_library_version,
        }
        with Path(path).open("x", encoding="utf-8") as file:
            file.write(json.dumps(metadata, indent=2, allow_nan=False) + "\n")

    @classmethod
    def load(
        cls, path: str | Path, expected_scaler: FrozenCountScaler | None = None
    ) -> "RidgeDecoder":
        """Load full parameters/transform; optionally require exact external fit identity."""
        metadata = json.loads(Path(path).read_text(encoding="utf-8"))
        if (
            metadata["schema_version"] != 1
            or metadata["model"] != "ridge"
            or metadata["feature_order"] != FEATURE_ORDER
            or metadata["weights_layout"] != "[L*C,2]; columns vx,vy"
            or metadata["output_units"] != "mm/s"
            or metadata["solver"] != "svd"
            or metadata["fit_intercept"] is not True
            or metadata["intercept_penalized"] is not False
        ):
            raise ValueError("Unsupported ridge artifact schema/order/units")
        m = metadata["scaler"]
        scaler = FrozenCountScaler.from_metadata(m)
        _validate_scaler(scaler)
        if (
            scaler.to_metadata() != m
            or scaler_sha256(scaler) != metadata["normalization_sha256"]
        ):
            raise ValueError("Saved transform metadata/hash mismatch")
        if expected_scaler is not None and scaler_sha256(scaler) != scaler_sha256(
            expected_scaler
        ):
            raise ValueError("Incompatible expected transform/channel identity")
        weights = np.array(metadata["weights"], dtype=float)
        intercept = np.array(metadata["intercept"], dtype=float)
        for array in (
            weights,
            intercept,
            scaler.means,
            scaler.scales,
            scaler.constant_channels,
        ):
            array.setflags(write=False)
        result = cls(
            weights,
            intercept,
            metadata["alpha"],
            scaler,
            metadata["fitted_examples"],
            metadata["fit_library_version"],
        )
        result.validate()
        return result


def select_ridge_alpha(candidates: list[dict], tolerance_mm_s: float = 1e-9) -> float:
    """Min overall validation RMSE; largest alpha within absolute tolerance of min."""
    if not candidates or not np.isfinite(tolerance_mm_s) or tolerance_mm_s < 0:
        raise ValueError("Candidates and finite nonnegative tie tolerance are required")
    alphas = [c["alpha"] for c in candidates]
    errors = [c["metrics"]["overall_rmse_mm_s"] for c in candidates]
    if (
        len(set(alphas)) != len(alphas)
        or not np.isfinite(alphas).all()
        or min(alphas) <= 0
        or not np.isfinite(errors).all()
        or min(errors) < 0
    ):
        raise ValueError("Candidate alphas/errors must be unique, finite and valid")
    minimum = min(errors)
    return float(
        max(
            a
            for a, error in zip(alphas, errors, strict=True)
            if error <= minimum + tolerance_mm_s
        )
    )
