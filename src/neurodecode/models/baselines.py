"""Trivial predictors: no neural-pattern fit and no held-out label input."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from neurodecode.data.features import HistoryWindows
from neurodecode.evaluation import ExampleIndex, VelocitySeries, targets_from_windows


def predict_zero(index: ExampleIndex) -> VelocitySeries:
    """Predict [0,0] mm/s using a label-free index."""
    index.validate()
    return VelocitySeries(np.zeros((index.time_s.size, 2)), index)


@dataclass(frozen=True)
class TrainingMeanVelocity:
    """Frozen two-axis mean over eligible training targets only."""

    mean_mm_s: NDArray[np.float64]
    fitted_examples: int
    channel_ids: tuple[str, ...]
    unit_indices: tuple[tuple[int, ...], ...]
    identity: tuple[tuple[str, str], ...]

    @classmethod
    def fit(cls, training: HistoryWindows) -> "TrainingMeanVelocity":
        if training.partition != "train":
            raise ValueError("Mean velocity must be fitted on training only")
        targets = targets_from_windows(training)
        if targets.values_mm_s.shape[0] == 0:
            raise ValueError("No eligible training targets")
        mean = targets.values_mm_s.mean(axis=0)
        if not np.isfinite(mean).all():
            raise ValueError("Training mean arithmetic is nonfinite")
        mean.setflags(write=False)
        return cls(
            mean,
            targets.values_mm_s.shape[0],
            training.channel_ids,
            training.unit_indices,
            training.identity,
        )

    def predict(self, index: ExampleIndex) -> VelocitySeries:
        """Broadcast frozen [vx,vy]; cannot receive labels or refit here."""
        index.validate()
        if (
            index.identity != self.identity
            or index.channel_ids != self.channel_ids
            or index.unit_indices != self.unit_indices
        ):
            raise ValueError("Training-mean preparation/channel identity mismatch")
        return VelocitySeries(np.tile(self.mean_mm_s, (index.time_s.size, 1)), index)
