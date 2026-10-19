"""Minimal CPU GRU training on existing independent prepared histories."""

from contextlib import contextmanager
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from neurodecode.data.features import FrozenCountScaler
from neurodecode.evaluation import (
    VelocitySeries,
    evaluate_velocity,
    require_matching_indices,
)
from neurodecode.models.gru import GRUConfig, GRUDecoder, GRUInputs


@dataclass(frozen=True)
class GRUTrainingConfig:
    """Component defaults, not a declaration of real-data experiment candidates."""

    seed: int = 2026
    batch_size: int = 256
    max_epochs: int = 50
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    gradient_clip_norm: float = 1.0
    patience: int = 8
    minimum_improvement_mm_s: float = 0.0

    def validate(self) -> None:
        for name in ("batch_size", "max_epochs", "patience"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0,2**32)")
        for name in ("learning_rate", "adam_epsilon", "gradient_clip_norm"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("weight_decay", "minimum_improvement_mm_s"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if any(
            not np.isfinite(v) or not 0 <= v < 1
            for v in (self.adam_beta1, self.adam_beta2)
        ):
            raise ValueError("Adam betas must be in [0,1)")


@contextmanager
def cpu_training_runtime(seed: int):
    """Seed CPU randomness/use deterministic ops and one thread, restoring caller state."""
    previous_threads = torch.get_num_threads()
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        try:
            yield
        finally:
            torch.use_deterministic_algorithms(
                previous_determinism, warn_only=previous_warn_only
            )
            torch.set_num_threads(previous_threads)


def make_adam(model: GRUDecoder, config: GRUTrainingConfig) -> torch.optim.Adam:
    """Explicit Adam settings; no scheduler, optimizer search or target scaling."""
    config.validate()
    return torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon,
        weight_decay=config.weight_decay,
    )


def gru_training_step(
    model: GRUDecoder,
    optimizer: torch.optim.Adam,
    x: torch.Tensor,
    y: torch.Tensor,
    clip_norm: float,
) -> dict:
    """One MSE/backprop/clip/Adam step; report gradient norm BEFORE clipping."""
    if (
        y.shape != (x.shape[0], 2)
        or y.dtype != torch.float32
        or y.device.type != "cpu"
        or not torch.isfinite(y).all()
    ):
        raise ValueError("Training targets must be finite CPU float32 [B,2]")
    if not np.isfinite(clip_norm) or clip_norm <= 0:
        raise ValueError("Gradient clip norm must be finite and positive")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = nn.functional.mse_loss(model(x), y, reduction="mean")
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite training loss")
    loss.backward()
    norm = nn.utils.clip_grad_norm_(
        model.parameters(), clip_norm, error_if_nonfinite=True
    )
    optimizer.step()
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("Optimizer produced nonfinite weights")
    return {
        "mse_mm2_s2": float(loss.detach()),
        "gradient_norm_before_clip": float(norm),
    }


@dataclass(frozen=True)
class GRUTrainingResult:
    model: GRUDecoder
    history: tuple[dict, ...]
    best_epoch: int
    config: GRUTrainingConfig
    training_examples: int
    validation_examples: int

    def checkpoint_metadata(self) -> dict:
        """Complete training choices/history; source/transform are stored with the model."""
        return {
            "configuration": asdict(self.config),
            "best_epoch": self.best_epoch,
            "training_examples": self.training_examples,
            "validation_examples": self.validation_examples,
            "selection": "minimum validation overall RMSE; improvement must exceed configured threshold; earliest tie",
            "loss": "mean squared error across examples and both unscaled mm/s axes",
            "window_batches": "seeded permutation of training windows; chronological bins within each window",
            "runtime": "CPU float32; one thread; deterministic algorithms; fresh zero h0 per window",
            "optimizer_resume_supported": False,
            "history": list(self.history),
        }


def fit_gru(
    training: GRUInputs,
    training_targets: VelocitySeries,
    validation: GRUInputs,
    validation_targets: VelocitySeries,
    scaler: FrozenCountScaler,
    model_config: GRUConfig,
    config: GRUTrainingConfig,
) -> GRUTrainingResult:
    """Fit training only; validation selects a snapshot but never receives gradients.

    Phase 5A exercises this function only with explicitly synthetic data.
    Real-recording experiments and their predeclared candidate list are Phase 5B.
    """
    config.validate()
    model_config.validate()
    for features, labels, name in (
        (training, training_targets, "train"),
        (validation, validation_targets, "validation"),
    ):
        features.validate(scaler)
        labels.validate()
        require_matching_indices(features.index, labels.index)
        if features.index.partition != name or features.index.time_s.size == 0:
            raise ValueError("Need nonempty train/validation partitions in that order")
        if not features.index.score_mask.all():
            raise ValueError(
                "Training requires existing eligible rows, without extra masks"
            )
    if training.index.time_s[-1] >= validation.index.time_s[0]:
        raise ValueError("Training must precede validation chronologically")
    # Frozen transforms/context have already been checked, not fitted here.
    with cpu_training_runtime(config.seed):
        model = GRUDecoder(scaler, model_config)
        optimizer = make_adam(model, config)
        generator = torch.Generator(device="cpu").manual_seed(config.seed)
        x = torch.from_numpy(training.values.copy())
        y = torch.from_numpy(training_targets.values_mm_s.astype(np.float32))
        if not torch.isfinite(y).all():
            raise ValueError("Targets overflowed CPU float32")
        best_score, best_state, best_epoch, stale = float("inf"), None, 0, 0
        history = []
        for epoch in range(1, config.max_epochs + 1):
            order = torch.randperm(len(x), generator=generator)
            total_loss, max_norm = 0.0, 0.0
            for start in range(0, len(x), config.batch_size):
                rows = order[start : start + config.batch_size]
                step = gru_training_step(
                    model, optimizer, x[rows], y[rows], config.gradient_clip_norm
                )
                total_loss += step["mse_mm2_s2"] * len(rows)
                max_norm = max(max_norm, step["gradient_norm_before_clip"])
            # predict() uses eval/inference mode and accepts no labels.
            predictions = model.predict(validation, config.batch_size)
            metrics = evaluate_velocity(validation_targets, predictions)
            score = metrics["overall_rmse_mm_s"]
            history.append(
                {
                    "epoch": epoch,
                    "training_online_mse_mm2_s2": total_loss / len(x),
                    "validation_metrics": metrics,
                    "maximum_gradient_norm_before_clip": max_norm,
                }
            )
            if score < best_score - config.minimum_improvement_mm_s:
                best_score, best_epoch, stale = score, epoch, 0
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
            else:
                stale += 1
            if stale >= config.patience:
                break
        model.load_state_dict(best_state, strict=True)
        model.eval()
    return GRUTrainingResult(
        model, tuple(history), best_epoch, config, len(x), len(validation.values)
    )
