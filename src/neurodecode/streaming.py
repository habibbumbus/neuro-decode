"""Label-free completed-bin replay with the existing frozen prediction APIs."""

from collections import deque
from dataclasses import dataclass
from time import perf_counter_ns
from typing import Iterator

import numpy as np
from numpy.typing import NDArray

from neurodecode.data.features import (
    FrozenCountScaler,
    HistoryWindows,
    _identity,
    _validate,
)
from neurodecode.data.targets import AlignedTargets
from neurodecode.models.gru import GRUDecoder, gru_inputs
from neurodecode.models.kalman import (
    KalmanParameters,
    current_bin_observations,
    kalman_step,
)
from neurodecode.models.ridge import RidgeDecoder, ridge_inputs
from neurodecode.splits import TimingConfig


@dataclass(frozen=True)
class ReplayBin:
    """One completed raw count bin and fixed quality metadata; no movement labels."""

    counts: NDArray[np.int64]
    source_row: int
    bin_id: int
    start_s: float
    end_s: float
    segment_id: int
    partition: str
    session: str
    channel_ids: tuple[str, ...]
    unit_indices: tuple[tuple[int, ...], ...]
    identity: tuple[tuple[str, str], ...]
    source_valid: bool
    target_valid: bool


def iter_replay_bins(
    targets: AlignedTargets,
    specification: dict,
    timing: TimingConfig,
    bin_valid_mask: NDArray[np.bool_] | None = None,
) -> Iterator[ReplayBin]:
    """Adapt verified preparation to packets; reuse its exact quality/identity rules.

    This adapter runs outside timing. _validate checks existing prepared structure,
    including label finiteness, but only fixed quality flags enter replay packets.
    No movement values enter the replay buffer or decoder. Counts were binned by
    the established offline binning code, not by an alternate event processor.
    """
    source_valid = _validate(targets, specification, timing, bin_valid_mask)
    identity = _identity(specification, targets)
    bins = targets.bins
    for row in range(len(bins.counts)):
        yield ReplayBin(
            bins.counts[row].copy(),
            row,
            int(bins.grid_end_indices[row]),
            float(bins.bin_start_s[row]),
            float(bins.bin_end_s[row]),
            int(bins.segment_ids[row]),
            bins.partition,
            specification["source"].get("session", "synthetic"),
            bins.channel_ids,
            bins.unit_indices,
            identity,
            bool(source_valid[row]),
            bool(targets.valid_mask[row]),
        )


class HistoryBuffer:
    """Retain at most L raw packets; gaps/invalid sources discard history."""

    def __init__(self, scaler: FrozenCountScaler, origin_s: float, bin_s: float):
        if not np.isfinite([origin_s, bin_s]).all() or bin_s <= 0:
            raise ValueError("Need finite grid origin and positive bin width")
        self.scaler = FrozenCountScaler.from_metadata(scaler.to_metadata())
        self.origin_s, self.bin_s = origin_s, bin_s
        self._history = deque(maxlen=scaler.history_bins)
        self._previous = None
        self.context_reset = True

    def push(self, packet: ReplayBin) -> HistoryWindows | None:
        """Build exactly one oldest-to-newest history from already revealed bins."""
        s = self.scaler
        if (
            packet.channel_ids != s.channel_ids
            or packet.unit_indices != s.unit_indices
            or packet.identity != s.identity
        ):
            raise ValueError("Replay channel/unit/source/split identity mismatch")
        if (
            packet.counts.shape != (len(s.channel_ids),)
            or packet.counts.dtype.kind not in "iu"
            or np.any(packet.counts < 0)
        ):
            raise ValueError("Raw replay counts must be nonnegative integer [C]")
        if (
            type(packet.bin_id) is not int
            or type(packet.segment_id) is not int
            or type(packet.source_row) is not int
            or packet.source_row < 0
            or not packet.partition
            or not packet.session
            or type(packet.source_valid) is not bool
            or type(packet.target_valid) is not bool
            or packet.start_s != self.origin_s + (packet.bin_id - 1) * self.bin_s
            or packet.end_s != self.origin_s + packet.bin_id * self.bin_s
            or packet.end_s <= packet.start_s
        ):
            raise ValueError("Invalid replay metadata or exact grid edges")
        previous = self._previous
        same_context = previous is not None and (
            packet.partition == previous.partition
            and packet.session == previous.session
        )
        if same_context and packet.bin_id <= previous.bin_id:
            raise ValueError("Replay bins must arrive in strictly increasing order")
        contiguous = same_context and (
            packet.segment_id == previous.segment_id
            and packet.bin_id == previous.bin_id + 1
            and packet.start_s == previous.end_s
        )
        self.context_reset = not contiguous
        if not contiguous or not packet.source_valid:
            self._history.clear()
        self._previous = packet
        if not packet.source_valid:
            return None
        # Own the current count vector; later caller mutation cannot change history.
        owned = ReplayBin(**{**packet.__dict__, "counts": packet.counts.copy()})
        self._history.append(owned)
        if not packet.target_valid or len(self._history) < s.history_bins:
            return None
        history = list(self._history)
        ids = np.array([[p.bin_id for p in history]], dtype=np.int64)
        return HistoryWindows(
            np.stack([p.counts for p in history])[None],
            np.full((1, 2), np.nan),  # deliberately unusable scoring placeholder
            np.array([[p.source_row for p in history]], dtype=np.int64),
            ids,
            np.array([[p.start_s for p in history]]),
            np.array([[p.end_s for p in history]]),
            np.array([packet.source_row]),
            np.array([packet.bin_id]),
            np.array([packet.end_s]),
            np.array([packet.segment_id]),
            s.channel_ids,
            s.unit_indices,
            packet.partition,
            s.identity,
            {},
        )


@dataclass(frozen=True)
class ReplayPrediction:
    """One emitted estimate plus exact provenance and optional nested timings."""

    bin_id: int
    time_s: float
    segment_id: int
    partition: str
    session: str
    source_bin_ids: tuple[int, ...]
    velocity_mm_s: NDArray[np.float64]
    reset: bool
    model_ns: int | None
    preparation_model_ns: int | None


class SequentialDecoder:
    """One model's independent stream; zero labels and no test/scoring interface."""

    def __init__(
        self,
        model: RidgeDecoder | GRUDecoder | KalmanParameters,
        origin_s: float,
        bin_s: float,
    ):
        if not isinstance(model, (RidgeDecoder, GRUDecoder, KalmanParameters)):
            raise ValueError("Unsupported replay model")
        if isinstance(model, (RidgeDecoder, KalmanParameters)):
            model.validate()
        if isinstance(model, KalmanParameters) and (
            origin_s != model.diagnostics["grid_origin_s"]
            or bin_s != model.diagnostics["bin_width_s"]
        ):
            raise ValueError("Replay grid differs from fitted Kalman grid")
        self.model = model
        self.history = HistoryBuffer(model.scaler, origin_s, bin_s)
        self._active = False
        self._mean = self._covariance = None

    def step(
        self, packet: ReplayBin, *, measure: bool = False
    ) -> ReplayPrediction | None:
        """Reveal one bin, prepare through existing adapters, then call frozen model.

        Timed total starts at entry and ends after inference/state update. Model
        time includes the checked prediction API, not only matrix/GRU kernels.
        Packet creation and returned-record packaging are outside both boundaries.
        """
        started = perf_counter_ns() if measure else 0
        windows = self.history.push(packet)
        if windows is None:
            self._active = False
            return None
        if self.history.context_reset:
            self._active = False
        reset = not self._active
        model = self.model
        if isinstance(model, RidgeDecoder):
            features = ridge_inputs(windows, model.scaler)
        elif isinstance(model, GRUDecoder):
            features = gru_inputs(windows, model.scaler)
        else:
            features = current_bin_observations(
                windows,
                model.scaler,
                packet.session,
                self.history.origin_s,
                self.history.bin_s,
            )
        prepared = perf_counter_ns() if measure else 0
        if isinstance(model, RidgeDecoder):
            velocity = model.predict(features).values_mm_s[0].copy()
        elif isinstance(model, GRUDecoder):
            velocity = model.predict(features, batch_size=1).values_mm_s[0].copy()
        else:
            mean = self._mean if self._active else model.initial_mean
            covariance = self._covariance if self._active else model.initial_covariance
            step = kalman_step(
                model, mean, covariance, features.values[0], predict=self._active
            )
            self._mean, self._covariance = step.mean, step.covariance
            velocity = step.mean.copy()
        self._active = True
        ended = perf_counter_ns() if measure else 0
        return ReplayPrediction(
            packet.bin_id,
            packet.end_s,
            packet.segment_id,
            packet.partition,
            packet.session,
            tuple(int(i) for i in windows.source_bin_end_indices[0]),
            velocity,
            reset,
            ended - prepared if measure else None,
            ended - started if measure else None,
        )
