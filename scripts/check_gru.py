"""Phase 5A synthetic-only GRU overfit/installation smoke check; no recording access."""

import argparse
import hashlib
import json
import platform
import time
import tomllib
from dataclasses import asdict, replace
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from evaluate_baselines import git_state

from neurodecode.data.features import FrozenCountScaler, HistoryWindows
from neurodecode.evaluation import evaluate_velocity, targets_from_windows
from neurodecode.models.gru import GRUConfig, GRUDecoder, gru_inputs
from neurodecode.training import GRUTrainingConfig, cpu_training_runtime, fit_gru


def main() -> None:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", help="New directory under ignored runs/")
    args = parser.parse_args()
    if not args.run_id or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for c in args.run_id
    ):
        raise ValueError("Run ID must use letters, digits, hyphens or underscores")
    output = Path("runs") / args.run_id
    if output.exists():
        raise ValueError(f"Existing run preserved: {output}; choose a new run ID")
    settings = tomllib.loads(Path("configs/gru.toml").read_text(encoding="utf-8"))
    architecture = GRUConfig(**settings["gru"])
    default_training = GRUTrainingConfig(**settings["training"])
    default_training.validate()
    # Fixed toy optimization budget; not a real-data candidate or tuning result.
    training_config = replace(
        default_training,
        max_epochs=250,
        batch_size=16,
        learning_rate=0.02,
        patience=250,
    )
    n, length, channels = 16, 10, 96
    raw = (
        np.random.default_rng(53)
        .poisson(1.0, size=(n, length, channels))
        .astype(np.int64)
    )
    y = np.column_stack(
        (
            0.4 * raw[:, -1, 0] + 0.2 * raw[:, 0, 1],
            -0.3 * raw[:, -1, 1] + 0.1 * raw[:, 0, 0],
        )
    )
    identity = tuple(
        sorted(
            {
                "source_sha256": "synthetic-only",
                "split_sha256_canonical_json": "synthetic-only",
                "configuration_sha256_utf8_lf": "synthetic-only",
                "target_policy_sha256": "synthetic-only",
            }.items()
        )
    )
    scaler = FrozenCountScaler(
        np.zeros(channels),
        np.ones(channels),
        np.zeros(channels, dtype=bool),
        tuple(f"M1 {i + 1:03d}" for i in range(channels)),
        ((0,),) * channels,
        length,
        identity,
        n * length,
        "synthetic-only",
        "synthetic-only",
    )

    def windows(partition, offset):
        ids = offset + length + np.arange(n, dtype=np.int64) * (length + 1)
        source_ids = ids[:, None] - np.arange(length - 1, -1, -1)
        return HistoryWindows(
            raw.copy(),
            y.copy(),
            source_ids - 1,
            source_ids,
            (source_ids - 1) * 0.02,
            source_ids * 0.02,
            ids - 1,
            ids,
            ids * 0.02,
            np.arange(n, dtype=np.int64),
            scaler.channel_ids,
            scaler.unit_indices,
            partition,
            identity,
            {"synthetic_only": True},
        )

    train, val = windows("train", 0), windows("validation", 10000)
    train_inputs, val_inputs = gru_inputs(train, scaler), gru_inputs(val, scaler)
    result = fit_gru(
        train_inputs,
        targets_from_windows(train),
        val_inputs,
        targets_from_windows(val),
        scaler,
        architecture,
        training_config,
    )
    predictions = result.model.predict(train_inputs)
    metrics = evaluate_velocity(targets_from_windows(train), predictions)
    if metrics["overall_rmse_mm_s"] ** 2 >= 0.002:
        raise AssertionError(
            "Tiny synthetic overfit check failed; not recording performance"
        )
    with cpu_training_runtime(training_config.seed):
        seen = []
        hook = result.model.recurrent.register_forward_hook(
            lambda _, args, out: seen.append(
                {
                    "initial_hidden": list(args[1].shape),
                    "recurrent_output": list(out[0].shape),
                    "final_hidden": list(out[1].shape),
                }
            )
        )
        with torch.inference_mode():
            one_output = result.model(torch.from_numpy(train_inputs.values[:1].copy()))
        hook.remove()
    output.mkdir(parents=True)
    metadata = result.checkpoint_metadata()
    metadata.update(
        {
            "synthetic_only": True,
            "fixture_seed": 53,
            "validation_fixture": "reindexed copies for overfit wiring, not independent generalization",
        }
    )
    result.model.save(output / "gru.pt", metadata)
    restored = GRUDecoder.load(output / "gru.pt", expected_scaler=scaler)
    np.testing.assert_array_equal(
        predictions.values_mm_s, restored.predict(train_inputs).values_mm_s
    )
    implementation_files = [
        "scripts/check_gru.py",
        "src/neurodecode/models/gru.py",
        "src/neurodecode/training.py",
        "src/neurodecode/data/features.py",
        "src/neurodecode/evaluation.py",
        "src/neurodecode/models/ridge.py",
    ]
    summary = {
        "synthetic_only": True,
        "real_recording_accessed": False,
        "parameter_count": result.model.parameter_count,
        "shapes": {
            "input": [1, length, channels],
            **seen[0],
            "head_input": [1, architecture.hidden_size],
            "output": list(one_output.shape),
        },
        "synthetic_training_rmse_mm_s": metrics["overall_rmse_mm_s"],
        "best_epoch": result.best_epoch,
        "epochs_run": len(result.history),
        "roundtrip": "exact eval predictions on same input/batch",
        "one_synthetic_prediction_mm_s": one_output.numpy()[0].tolist(),
    }
    report = {
        "checkpoint": "Phase 5A synthetic-only verification",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "configuration": asdict(architecture),
        "training": metadata,
        "normalization": scaler.to_metadata(),
        "git": git_state(),
        "versions": {
            "python": platform.python_version(),
            **{p: version(p) for p in ("torch", "numpy", "neurodecode")},
        },
        "platform": platform.platform(),
        "device": "cpu",
        "cuda_build": torch.version.cuda,
        "implementation_sha256_utf8_lf": {
            p: hashlib.sha256(Path(p).read_text(encoding="utf-8").encode()).hexdigest()
            for p in implementation_files
        },
        "configuration_sha256_utf8_lf": hashlib.sha256(
            Path("configs/gru.toml").read_text(encoding="utf-8").encode()
        ).hexdigest(),
        "checkpoint_sha256": hashlib.sha256(
            (output / "gru.pt").read_bytes()
        ).hexdigest(),
        "duration_s": time.perf_counter() - started,
        "duration_scope": "whole synthetic check, not inference latency",
        "interpretation": "memorization/training-path smoke check only; no scientific generalization claim",
    }
    with (output / "run.json").open("x", encoding="utf-8") as file:
        file.write(json.dumps(report, indent=2, allow_nan=False) + chr(10))
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
