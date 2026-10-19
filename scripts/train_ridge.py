"""Phase 3B: fixed-grid ridge selection on validation; never evaluate test."""

import argparse
import hashlib
import json
import os
import platform
import time
import tomllib
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
from bin_recording import load_verified_inputs
from evaluate_baselines import git_state, write_predictions
from threadpoolctl import threadpool_info, threadpool_limits

from neurodecode.data.features import (
    FrozenCountScaler,
    fit_count_scaler,
    prepare_partition,
)
from neurodecode.evaluation import (
    evaluate_velocity,
    index_from_windows,
    targets_from_windows,
)
from neurodecode.models.baselines import TrainingMeanVelocity, predict_zero
from neurodecode.models.ridge import (
    FEATURE_ORDER,
    RidgeDecoder,
    ridge_inputs,
    scaler_sha256,
    select_ridge_alpha,
)
from neurodecode.plotting import save_baseline_velocity_figure


def main() -> None:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", help="New directory under ignored runs/")
    args = parser.parse_args()
    if (
        not args.run_id
        or args.run_id in (".", "..")
        or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
            for c in args.run_id
        )
    ):
        raise ValueError("Run ID must use letters, digits, hyphens or underscores")
    output = Path("runs") / args.run_id
    if output.exists():
        raise ValueError(f"Existing run preserved: {output}; choose a new run ID")
    ridge_path = Path("configs/ridge.toml")
    settings = tomllib.loads(ridge_path.read_text(encoding="utf-8"))["ridge"]
    expected = {
        "solver": "svd",
        "fit_intercept": True,
        "primary_metric": "overall_rmse_mm_s",
        "tie_break": "largest_alpha_within_tolerance_of_minimum",
    }
    if any(settings.get(k) != v for k, v in expected.items()):
        raise ValueError("Unsupported ridge selection/solver policy")
    alphas = settings["alphas"]
    if (
        not alphas
        or sorted(set(alphas)) != alphas
        or not np.isfinite(alphas).all()
        or min(alphas) <= 0
    ):
        raise ValueError("Alpha grid must be increasing, unique, finite and positive")
    tolerance = settings["tie_tolerance_mm_s"]
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Tie tolerance must be finite/nonnegative")
    # Declare the saved rule before fitting or inspecting any candidate scores.
    print(json.dumps({"predeclared_ridge_settings": settings}))
    recording, specification, config = load_verified_inputs()
    train_targets, training = prepare_partition(
        recording, specification, config, "train"
    )
    fitted_scaler = fit_count_scaler(train_targets, specification, config)
    scaler = fitted_scaler
    prior_scaler = Path("runs/phase2d/count_scaler.json")
    normalization_provenance = "existing Phase 2D train-only fit rule, recomputed"
    if prior_scaler.exists():
        saved = json.loads(prior_scaler.read_text(encoding="utf-8"))
        core = {key: saved[key] for key in fitted_scaler.to_metadata()}
        scaler = FrozenCountScaler.from_metadata(core)
        if scaler_sha256(scaler) != scaler_sha256(fitted_scaler):
            raise ValueError(
                "Saved Phase 2D normalization differs from verified training fit"
            )
        normalization_provenance = (
            "restored Phase 2D frozen metadata, verified against training-only fit"
        )
    train_features = ridge_inputs(training, scaler)
    train_labels = targets_from_windows(training)
    # No test targets/windows are constructed. Validation is for selection only.
    _, validation = prepare_partition(recording, specification, config, "validation")
    validation_features = ridge_inputs(validation, scaler)
    validation_index = index_from_windows(validation)
    validation_labels = targets_from_windows(validation)
    candidates, models = [], {}
    with threadpool_limits(limits=1):
        for alpha in alphas:
            model = RidgeDecoder.fit(train_features, train_labels, scaler, alpha)
            prediction = model.predict(validation_features)
            score = evaluate_velocity(validation_labels, prediction)
            candidates.append({"alpha": alpha, "metrics": score})
            models[alpha] = model
            print(
                f"alpha={alpha:g} validation overall RMSE={score['overall_rmse_mm_s']:.6f}"
            )
        selected_alpha = select_ridge_alpha(candidates, tolerance)
        selected = models[selected_alpha]
        predictions = {
            "Zero velocity": predict_zero(validation_index),
            "Training mean": TrainingMeanVelocity.fit(training).predict(
                validation_index
            ),
            "Ridge": selected.predict(validation_features),
        }
        metrics = {
            name: evaluate_velocity(validation_labels, p)
            for name, p in predictions.items()
        }
        training_metrics = evaluate_velocity(
            train_labels, selected.predict(train_features)
        )
        pool_metadata = threadpool_info()
    output.mkdir(parents=True)
    selected.save(output / "ridge.json")
    restored = RidgeDecoder.load(output / "ridge.json", expected_scaler=scaler)
    with threadpool_limits(limits=1):
        reloaded_prediction = restored.predict(
            ridge_inputs(validation, restored.scaler)
        )
    np.testing.assert_array_equal(
        predictions["Ridge"].values_mm_s, reloaded_prediction.values_mm_s
    )
    write_predictions(
        output / "predictions.csv", validation, validation_labels, predictions
    )
    os.environ.setdefault("MPLCONFIGDIR", str(Path("runs/.matplotlib").resolve()))
    plot = save_baseline_velocity_figure(
        validation_labels,
        predictions,
        specification["source"]["session"],
        output / "validation_velocity.png",
    )
    first_prediction = predictions["Ridge"].values_mm_s[0]
    details = {
        "time_s": float(validation_index.time_s[0]),
        "bin_id": int(validation_index.bin_ids[0]),
        "input_shape": list(validation.X_raw[0].shape),
        "flattened_shape": list(validation_features.values[0].shape),
        "weights_shape": list(selected.weights.shape),
        "intercept_mm_s": selected.intercept.tolist(),
        "prediction_mm_s": first_prediction.tolist(),
        "actual_mm_s": validation_labels.values_mm_s[0].tolist(),
        "error_mm_s": (first_prediction - validation_labels.values_mm_s[0]).tolist(),
        "first_feature": {
            "channel": validation.channel_ids[0],
            "raw_count": int(validation.X_raw[0, 0, 0]),
            "training_mean": float(scaler.means[0]),
            "training_scale": float(scaler.scales[0]),
            "normalized": float(validation_features.values[0, 0]),
            "weights_vx_vy": selected.weights[0].tolist(),
            "contribution_vx_vy": (
                validation_features.values[0, 0] * selected.weights[0]
            ).tolist(),
        },
    }
    implementation_files = [
        "scripts/train_ridge.py",
        "scripts/evaluate_baselines.py",
        "scripts/bin_recording.py",
        "src/neurodecode/models/ridge.py",
        "src/neurodecode/models/baselines.py",
        "src/neurodecode/data/features.py",
        "src/neurodecode/data/targets.py",
        "src/neurodecode/data/binning.py",
        "src/neurodecode/data/loaders.py",
        "src/neurodecode/splits.py",
        "src/neurodecode/evaluation.py",
        "src/neurodecode/plotting.py",
    ]
    identity_bytes = (
        np.column_stack((validation_index.segment_ids, validation_index.bin_ids))
        .astype("<i8")
        .tobytes()
        + validation_index.time_s.astype("<f8").tobytes()
    )
    metadata = {
        "schema_version": 1,
        "checkpoint": "Phase 3B ridge validation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": specification["source"],
        "identity": dict(validation.identity),
        "split_specification": specification,
        "timing": asdict(config),
        "ridge_settings": settings,
        "selected_alpha": selected_alpha,
        "ridge_configuration_sha256_utf8_lf": hashlib.sha256(
            ridge_path.read_text(encoding="utf-8").encode()
        ).hexdigest(),
        "feature_order": FEATURE_ORDER,
        "channel_ids": list(validation.channel_ids),
        "unit_indices": [list(u) for u in validation.unit_indices],
        "normalization": scaler.to_metadata(),
        "normalization_sha256": scaler_sha256(scaler),
        "normalization_provenance": normalization_provenance,
        "train_summary": training.summary,
        "validation_summary": validation.summary,
        "training_mean_mm_s": TrainingMeanVelocity.fit(training).mean_mm_s.tolist(),
        "candidates": candidates,
        "metrics": metrics,
        "selected_training_metrics": training_metrics,
        "first_validation_example": details,
        "plot": plot,
        "eligible_validation_identity_sha256": hashlib.sha256(
            identity_bytes
        ).hexdigest(),
        "score_mask_rule": "all unchanged Phase 2D eligible validation examples",
        "test_evaluation": "not scored/predicted/plotted; no test targets/windows constructed",
        "model_roundtrip": "exact validation prediction equality",
        "git": git_state(),
        "device": "CPU",
        "seed": None,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                p: version(p)
                for p in (
                    "numpy",
                    "scipy",
                    "scikit-learn",
                    "h5py",
                    "matplotlib",
                    "threadpoolctl",
                )
            },
            "blas_thread_limit": 1,
            "native_libraries": [
                {
                    k: p.get(k)
                    for k in (
                        "internal_api",
                        "prefix",
                        "num_threads",
                        "version",
                        "architecture",
                    )
                }
                for p in pool_metadata
            ],
        },
        "implementation_sha256_utf8_lf": {
            p: hashlib.sha256(Path(p).read_text(encoding="utf-8").encode()).hexdigest()
            for p in implementation_files
        },
        "artifact_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                output / "ridge.json",
                output / "predictions.csv",
                output / "validation_velocity.png",
            )
        },
        "duration_s": time.perf_counter() - started,
        "duration_scope": "whole preparation/grid selection/export; not inference latency",
    }
    for filename, value in (("run.json", metadata), ("metrics.json", metrics)):
        with (output / filename).open("x", encoding="utf-8") as file:
            file.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
    print(
        f"Selected alpha={selected_alpha:g}; {validation_index.time_s.size} validation examples"
    )
    for name, score in metrics.items():
        print(json.dumps({"predictor": name, **score}))
    print(
        json.dumps(
            {"first_validation_example": details, "run_directory": output.as_posix()},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
