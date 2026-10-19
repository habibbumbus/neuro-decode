"""One-layer causal GRU over independently reset, identity-bound histories."""

import json
import platform
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from neurodecode.data.features import FrozenCountScaler, HistoryWindows
from neurodecode.evaluation import ExampleIndex, VelocitySeries, index_from_windows
from neurodecode.models.ridge import scaler_sha256

WINDOW_POLICY = "oldest to newest; zero hidden state for each independent window"


@dataclass(frozen=True)
class GRUConfig:
    """One unidirectional layer; dimensions come from the frozen scaler."""

    hidden_size: int = 32

    def validate(self) -> None:
        if type(self.hidden_size) is not int or self.hidden_size < 1:
            raise ValueError("GRU hidden_size must be a positive integer")


@dataclass(frozen=True)
class GRUInputs:
    """Label-free normalized float32 [N,L,C], with unchanged prepared row identity."""

    values: NDArray[np.float32]
    index: ExampleIndex
    history_bins: int
    normalization_sha256: str
    window_policy: str = WINDOW_POLICY

    def validate(self, scaler: FrozenCountScaler) -> None:
        self.index.validate()
        if (
            self.values.shape
            != (self.index.time_s.size, scaler.history_bins, len(scaler.channel_ids))
            or self.values.dtype != np.dtype(np.float32)
            or not np.isfinite(self.values).all()
        ):
            raise ValueError("GRU inputs must be finite float32 [N,L,C]")
        if (
            self.history_bins != scaler.history_bins
            or self.index.channel_ids != scaler.channel_ids
            or self.index.unit_indices != scaler.unit_indices
            or self.index.identity != scaler.identity
            or self.normalization_sha256 != scaler_sha256(scaler)
            or self.window_policy != WINDOW_POLICY
        ):
            raise ValueError("GRU channel/transform/source/window identity mismatch")


def gru_inputs(windows: HistoryWindows, scaler: FrozenCountScaler) -> GRUInputs:
    """Apply the existing frozen transform, without reading labels or fitting it."""
    index = index_from_windows(windows)
    source_ids = windows.source_bin_end_indices
    if (
        windows.source_bin_start_s.shape != source_ids.shape
        or np.any(np.diff(source_ids, axis=1) != 1)
        or not np.array_equal(
            windows.source_bin_start_s[:, 1:], windows.source_bin_end_s[:, :-1]
        )
    ):
        raise ValueError("GRU requires unchanged consecutive prepared history bins")
    result = GRUInputs(
        np.ascontiguousarray(scaler.transform(windows), dtype=np.float32),
        index,
        scaler.history_bins,
        scaler_sha256(scaler),
    )
    result.validate(scaler)
    return result


class GRUDecoder(nn.Module):
    """CPU float32 [B,L,C] -> [B,2] mm/s; no persistent hidden state."""

    def __init__(self, scaler: FrozenCountScaler, config: GRUConfig = GRUConfig()):
        super().__init__()
        config.validate()
        # Reuse the existing scaler's strict metadata validation, never refit.
        self.scaler = FrozenCountScaler.from_metadata(scaler.to_metadata())
        self.config = config
        self.recurrent = nn.GRU(
            input_size=len(scaler.channel_ids),
            hidden_size=config.hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
            dropout=0.0,
            device="cpu",
            dtype=torch.float32,
        )
        self.head = nn.Linear(config.hidden_size, 2, device="cpu", dtype=torch.float32)

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Read oldest-to-newest; initialize h0=[1,B,H] to zero on every call."""
        if (
            values.ndim != 3
            or values.shape[1:]
            != (self.scaler.history_bins, len(self.scaler.channel_ids))
            or values.shape[0] < 1
            or values.dtype != torch.float32
            or values.device.type != "cpu"
            or not torch.isfinite(values).all()
        ):
            raise ValueError("Forward requires finite CPU float32 [B,L,C]")
        hidden = values.new_zeros((1, values.shape[0], self.config.hidden_size))
        _, final_hidden = self.recurrent(values, hidden)
        result = self.head(final_hidden[-1])
        if not torch.isfinite(result).all():
            raise ValueError("GRU produced nonfinite velocities")
        return result

    def predict(self, features: GRUInputs, batch_size: int = 256) -> VelocitySeries:
        """Frozen eval/inference mode; labels are absent and row order is preserved."""
        features.validate(self.scaler)
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("Prediction batch_size must be a positive integer")
        self.eval()
        result = np.empty((features.index.time_s.size, 2), dtype=np.float64)
        with torch.inference_mode():
            for start in range(0, len(result), batch_size):
                stop = min(start + batch_size, len(result))
                x = torch.from_numpy(features.values[start:stop].copy())
                result[start:stop] = self(x).numpy()
        prediction = VelocitySeries(result, features.index)
        prediction.validate()
        return prediction

    def save(self, path: str | Path, training_metadata: dict) -> None:
        """Exclusive state_dict + primitive metadata checkpoint, without live hidden state."""
        metadata = {
            "schema_version": 1,
            "model": "bounded_window_gru",
            "architecture": {
                "layers": 1,
                "bidirectional": False,
                "dropout": 0.0,
                "input_channels": len(self.scaler.channel_ids),
                "history_bins": self.scaler.history_bins,
                "output_dimensions": 2,
                **asdict(self.config),
            },
            "window_policy": WINDOW_POLICY,
            "output_units": "mm/s",
            "dtype": "float32",
            "normalization": self.scaler.to_metadata(),
            "normalization_sha256": scaler_sha256(self.scaler),
            "parameter_count": self.parameter_count,
            "training": training_metadata,
            "versions": {
                "python": platform.python_version(),
                **{name: version(name) for name in ("torch", "numpy", "neurodecode")},
            },
        }
        # JSON-compatible metadata excludes executable objects, NaN and tensors.
        metadata = json.loads(json.dumps(metadata, allow_nan=False))
        state = {
            name: value.detach().cpu().clone()
            for name, value in self.state_dict().items()
        }
        if not all(torch.isfinite(value).all() for value in state.values()):
            raise ValueError("Cannot save nonfinite GRU parameters")
        with Path(path).open("xb") as file:
            torch.save({"metadata": metadata, "state_dict": state}, file)

    @classmethod
    def load(
        cls, path: str | Path, expected_scaler: FrozenCountScaler | None = None
    ) -> "GRUDecoder":
        """Load tensors/primitive metadata on CPU with weights_only=True and strict keys."""
        payload = torch.load(path, map_location="cpu", weights_only=True)
        metadata = payload["metadata"]
        if (
            metadata["schema_version"] != 1
            or metadata["model"] != "bounded_window_gru"
            or metadata["window_policy"] != WINDOW_POLICY
            or metadata["dtype"] != "float32"
            or metadata["output_units"] != "mm/s"
        ):
            raise ValueError(
                "Unsupported GRU checkpoint schema/order/units/reset policy"
            )
        scaler = FrozenCountScaler.from_metadata(metadata["normalization"])
        if scaler_sha256(scaler) != metadata["normalization_sha256"] or (
            expected_scaler is not None
            and scaler_sha256(scaler) != scaler_sha256(expected_scaler)
        ):
            raise ValueError("Incompatible GRU transform/channel identity")
        architecture = metadata["architecture"]
        config = GRUConfig(architecture["hidden_size"])
        expected = {
            "layers": 1,
            "bidirectional": False,
            "dropout": 0.0,
            "input_channels": len(scaler.channel_ids),
            "history_bins": scaler.history_bins,
            "output_dimensions": 2,
            **asdict(config),
        }
        if architecture != expected:
            raise ValueError("Incompatible GRU architecture/input dimensions")
        # Loading should not consume the caller's random initialization sequence.
        with torch.random.fork_rng(devices=[]):
            model = cls(scaler, config)
        if model.parameter_count != metadata["parameter_count"]:
            raise ValueError("GRU checkpoint parameter count mismatch")
        state = payload["state_dict"]
        if not all(
            isinstance(v, torch.Tensor)
            and v.dtype == torch.float32
            and torch.isfinite(v).all()
            for v in state.values()
        ):
            raise ValueError("Invalid GRU checkpoint tensor dtype/value")
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError as error:
            raise ValueError("GRU checkpoint parameter shape/key mismatch") from error
        model.eval()
        return model
