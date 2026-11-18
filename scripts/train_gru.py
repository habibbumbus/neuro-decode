"""Phase 5B round 1: bounded CPU training and matched validation, test closed."""

import argparse
import csv
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
from evaluate_kalman import verified_artifact
from threadpoolctl import threadpool_limits

from neurodecode.data.features import (
    FrozenCountScaler,
    fit_count_scaler,
    prepare_partition,
)
from neurodecode.evaluation import (
    VelocitySeries,
    evaluate_velocity,
    targets_from_windows,
)
from neurodecode.models.gru import GRUConfig, GRUDecoder, gru_inputs
from neurodecode.models.ridge import RidgeDecoder, ridge_inputs, scaler_sha256
from neurodecode.plotting import save_baseline_velocity_figure
from neurodecode.training import GRUTrainingConfig, cpu_training_runtime, fit_gru


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as file:
        file.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def select_learning_rate(candidates, tolerance):
    """Minimum validation RMSE; lower rate within the declared absolute tolerance."""
    if not candidates or not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Need candidates and a finite nonnegative tolerance")
    scores = [c["validation_metrics"]["overall_rmse_mm_s"] for c in candidates]
    if not np.isfinite(scores).all():
        raise ValueError("Candidate scores must be finite")
    best = min(scores)
    return min(
        c["learning_rate"]
        for c, score in zip(candidates, scores, strict=True)
        if score <= best + tolerance
    )


def reference_predictions(directory, windows, labels, specification, scaler):
    """Verify saved Phase 4B bytes and every scored identity/target before reuse."""
    path, metadata = verified_artifact(directory, "predictions.csv", "artifact_sha256")
    if (
        metadata["source"] != specification["source"]
        or metadata["split_specification"] != specification
        or metadata["normalization_sha256"] != scaler_sha256(scaler)
        or metadata["channel_ids"] != list(scaler.channel_ids)
        or metadata["unit_indices"] != [list(u) for u in scaler.unit_indices]
    ):
        raise ValueError("Reference source/split/channel/transform changed")
    index = labels.index
    identity_hash = hashlib.sha256(
        np.column_stack((index.segment_ids, index.bin_ids)).astype("<i8").tobytes()
        + index.time_s.astype("<f8").tobytes()
    ).hexdigest()
    if identity_hash != metadata["eligible_validation_identity_sha256"]:
        raise ValueError("Reference eligible identity changed")
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != len(index.time_s) or len(rows) != 2775:
        raise ValueError("Expected exactly the original 2775 validation rows")
    for i, row in enumerate(rows):
        if (
            row["partition"] != index.partition
            or int(row["segment_id"]) != index.segment_ids[i]
            or int(row["bin_id"]) != index.bin_ids[i]
            or float(row["time_s"]) != index.time_s[i]
            or row["score_mask"] != str(bool(index.score_mask[i]))
            or int(row["first_source_bin_id"]) != windows.source_bin_end_indices[i, 0]
            or int(row["last_source_bin_id"]) != windows.source_bin_end_indices[i, -1]
        ):
            raise ValueError("Reference row/window identity changed")
        for axis, name in enumerate(("x", "y")):
            if float(row[f"actual_v{name}_mm_s"]) != labels.values_mm_s[i, axis]:
                raise ValueError("Reference target changed")
    predictions = {}
    for title, prefix in (
        ("Zero velocity", "zero"),
        ("Training mean", "mean"),
        ("Ridge", "ridge"),
        ("Kalman", "kalman"),
    ):
        prediction = VelocitySeries(
            np.array(
                [
                    [float(row[f"{prefix}_v{a}_mm_s"]) for a in ("x", "y")]
                    for row in rows
                ]
            ),
            index,
        )
        metrics = evaluate_velocity(labels, prediction)
        if metrics != metadata["metrics"][title]:
            raise ValueError("Reference shared metrics changed")
        predictions[title] = prediction
    return predictions, {
        "path": path.as_posix(),
        "sha256": sha(path),
        "identity_sha256": identity_hash,
    }


def save_learning_curves(history, best_epoch, title, output):
    """End-of-epoch scores share frozen weights; online batch loss is separate."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(10, 5), layout="constrained")
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    epochs = [h["epoch"] for h in history]
    for key, label in (
        ("training_metrics", "Training, end of epoch"),
        ("validation_metrics", "Validation"),
    ):
        axis.plot(epochs, [h[key]["overall_rmse_mm_s"] for h in history], label=label)
    axis.plot(
        epochs,
        [np.sqrt(h["training_online_mse_mm2_s2"]) for h in history],
        label="Training, online batch loss sqrt",
        linestyle=":",
        alpha=0.7,
    )
    axis.axvline(
        best_epoch, color="black", linestyle="--", label=f"Selected epoch {best_epoch}"
    )
    axis.set(xlabel="Epoch", ylabel="Overall RMSE [mm/s]", title=title)
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.savefig(output, dpi=150, facecolor="white")


def diagnostics(labels, prediction):
    return {
        "bias_mm_s": (prediction.values_mm_s - labels.values_mm_s)
        .mean(axis=0)
        .tolist(),
        "prediction_mean_mm_s": prediction.values_mm_s.mean(axis=0).tolist(),
        "prediction_std_ddof0_mm_s": prediction.values_mm_s.std(axis=0).tolist(),
        "target_mean_mm_s": labels.values_mm_s.mean(axis=0).tolist(),
        "target_std_ddof0_mm_s": labels.values_mm_s.std(axis=0).tolist(),
    }


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    args = parser.parse_args()
    if not args.run_id or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for c in args.run_id
    ):
        raise ValueError("Run ID must use letters, digits, hyphens or underscores")
    output = Path("runs") / args.run_id
    if output.exists():
        raise ValueError(f"Existing run preserved: {output}")
    plan = tomllib.loads(
        Path("configs/gru_experiment.toml").read_text(encoding="utf-8")
    )
    defaults = tomllib.loads(Path("configs/gru.toml").read_text(encoding="utf-8"))
    model_config = GRUConfig(**defaults["gru"])
    base = GRUTrainingConfig(**defaults["training"])
    if (
        model_config != GRUConfig()
        or base != GRUTrainingConfig()
        or plan["learning_rates"] != [0.001, 0.003]
        or plan["selection_seed"] != 2026
        or plan["additional_seeds"] != [2027, 2028]
        or plan["representative_seed"] != 2026
        or plan["primary_metric"] != "overall_rmse_mm_s"
        or plan["setting_tie_break"] != "lower learning rate"
        or plan["setting_tie_tolerance_mm_s"] != 1e-9
        or plan["max_runs"] != 4
        or plan["max_total_epochs"] != 200
        or plan["max_optimizer_updates"] != 10400
        or plan["training_budget_s"] != 1200
        or plan["target_scaling"] != "none; unscaled mm/s"
    ):
        raise ValueError("Runner implements only the documented round 1 plan/defaults")
    output.mkdir(parents=True)
    os.environ.setdefault("MPLCONFIGDIR", str(Path("runs/.matplotlib").resolve()))
    implementation_files = [p for p in Path("src/neurodecode").rglob("*.py")]
    implementation_files += [
        Path("scripts") / p
        for p in (
            "train_gru.py",
            "bin_recording.py",
            "evaluate_baselines.py",
            "evaluate_kalman.py",
        )
    ]
    implementation_files += [
        Path("configs") / p for p in ("base.toml", "gru.toml", "gru_experiment.toml")
    ]
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_state(),
        "plan": plan,
        "component_defaults": defaults,
        "implementation_sha256_utf8_lf": {
            p.as_posix(): hashlib.sha256(
                p.read_text(encoding="utf-8").encode()
            ).hexdigest()
            for p in implementation_files
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "device": "CPU",
            "dtype": "float32",
            "threads": 1,
            "packages": {
                name: version(name)
                for name in (
                    "torch",
                    "numpy",
                    "scipy",
                    "scikit-learn",
                    "h5py",
                    "matplotlib",
                    "threadpoolctl",
                )
            },
        },
    }
    # This immutable record exists before any recording GRU fit.
    write_json(output / "predeclared_plan.json", provenance)
    print(json.dumps({"predeclared_plan": plan}), flush=True)
    recording, specification, timing = load_verified_inputs()
    train_targets, training = prepare_partition(
        recording, specification, timing, "train"
    )
    fitted = fit_count_scaler(train_targets, specification, timing)
    saved = json.loads(
        Path("runs/phase2d/count_scaler.json").read_text(encoding="utf-8")
    )
    scaler = FrozenCountScaler.from_metadata(
        {k: saved[k] for k in fitted.to_metadata()}
    )
    if scaler_sha256(scaler) != scaler_sha256(fitted):
        raise ValueError("Phase 2D scaler differs from training-only fit")
    _, validation = prepare_partition(recording, specification, timing, "validation")
    train_x, val_x = gru_inputs(training, scaler), gru_inputs(validation, scaler)
    train_y, val_y = targets_from_windows(training), targets_from_windows(validation)
    if train_x.values.shape != (13070, 10, 96) or val_x.values.shape != (2775, 10, 96):
        raise ValueError("Prepared populations changed")
    references, reference_provenance = reference_predictions(
        Path("runs/phase4b-kalman-audited"), validation, val_y, specification, scaler
    )
    ridge_path, _ = verified_artifact(
        Path("runs/phase3b-ridge"), "ridge.json", "artifact_sha256"
    )
    with threadpool_limits(limits=1):
        ridge = RidgeDecoder.load(ridge_path, expected_scaler=scaler)
        np.testing.assert_array_equal(
            references["Ridge"].values_mm_s,
            ridge.predict(ridge_inputs(validation, scaler)).values_mm_s,
        )
    if not np.array_equal(
        references["Training mean"].values_mm_s[0], train_y.values_mm_s.mean(axis=0)
    ):
        raise ValueError("Saved training mean changed")
    deadline = time.monotonic() + plan["training_budget_s"]
    records, candidates, predictions_by_run = [], [], {}
    selected_rate = None

    def run(rate, seed, role):
        if time.monotonic() >= deadline:
            records.append(
                {
                    "learning_rate": rate,
                    "seed": seed,
                    "role": role,
                    "status": "not run: deadline",
                }
            )
            return None
        config = GRUTrainingConfig(
            **{**defaults["training"], "seed": seed, "learning_rate": rate}
        )
        run_name = f"lr-{rate:g}-seed-{seed}"
        directory = output / run_name
        directory.mkdir()
        print(f"Starting {run_name} ({role})", flush=True)
        run_started = time.monotonic()

        def progress(h):
            if h["epoch"] == 1 or h["epoch"] % 10 == 0:
                print(
                    f"{run_name} epoch {h['epoch']} train {h['training_metrics']['overall_rmse_mm_s']:.6f} validation {h['validation_metrics']['overall_rmse_mm_s']:.6f}",
                    flush=True,
                )

        with threadpool_limits(limits=1), cpu_training_runtime(seed):
            result = fit_gru(
                train_x,
                train_y,
                val_x,
                val_y,
                scaler,
                model_config,
                config,
                deadline_monotonic=deadline,
                epoch_callback=progress,
            )
            prediction = result.model.predict(val_x, config.batch_size)
            train_prediction = result.model.predict(train_x, config.batch_size)
            result.model.save(directory / "gru.pt", result.checkpoint_metadata())
            restored = GRUDecoder.load(directory / "gru.pt", expected_scaler=scaler)
            np.testing.assert_array_equal(
                prediction.values_mm_s,
                restored.predict(val_x, config.batch_size).values_mm_s,
            )
        history = list(result.history)
        last = history[-1]["epoch"]
        best_score = history[result.best_epoch - 1]["validation_metrics"][
            "overall_rmse_mm_s"
        ]
        stale = last - result.best_epoch
        complete = last == config.max_epochs or stale >= config.patience
        stop = (
            "maximum epochs"
            if last == config.max_epochs
            else "patience"
            if stale >= config.patience
            else "deadline"
        )
        record = {
            "run": run_name,
            "role": role,
            "status": "complete" if complete else "incomplete: deadline",
            "learning_rate": rate,
            "seed": seed,
            "configuration": asdict(config),
            "epochs": last,
            "optimizer_updates": last * 52,
            "stop_reason": stop,
            "best_epoch": result.best_epoch,
            "parameter_count": result.model.parameter_count,
            "validation_metrics": evaluate_velocity(val_y, prediction),
            "training_metrics": evaluate_velocity(train_y, train_prediction),
            "training_diagnostics": diagnostics(train_y, train_prediction),
            "validation_diagnostics": diagnostics(val_y, prediction),
            "checkpoint_reload": "exact validation prediction equality",
            "checkpoint_sha256": sha(directory / "gru.pt"),
            "duration_s": time.monotonic() - run_started,
            "first_validation_example": {
                "time_s": float(val_y.index.time_s[0]),
                "bin_id": int(val_y.index.bin_ids[0]),
                "history_start_s": float(validation.source_bin_start_s[0, 0]),
                "input_shape": [1, 10, 96],
                "hidden_shape": [1, 1, 32],
                "first_channel_raw_counts": validation.X_raw[0, :, 0].tolist(),
                "first_channel_normalized": val_x.values[0, :, 0].tolist(),
                "prediction_mm_s": prediction.values_mm_s[0].tolist(),
                "actual_mm_s": val_y.values_mm_s[0].tolist(),
                "error_mm_s": (
                    prediction.values_mm_s[0] - val_y.values_mm_s[0]
                ).tolist(),
            },
        }
        if record["validation_metrics"]["overall_rmse_mm_s"] != best_score:
            raise ValueError("Restored checkpoint differs from selected epoch score")
        write_json(directory / "run.json", record)
        write_json(directory / "history.json", history)
        with (directory / "learning_curve.csv").open(
            "x", newline="", encoding="utf-8"
        ) as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "epoch",
                    "online_training_mse_mm2_s2",
                    "train_rmse_mm_s",
                    "validation_rmse_mm_s",
                    "validation_mse_mm2_s2",
                    "max_gradient_norm_before_clip",
                ]
            )
            for h in history:
                writer.writerow(
                    [
                        h["epoch"],
                        h["training_online_mse_mm2_s2"],
                        h["training_metrics"]["overall_rmse_mm_s"],
                        h["validation_metrics"]["overall_rmse_mm_s"],
                        h["validation_metrics"]["overall_rmse_mm_s"] ** 2,
                        h["maximum_gradient_norm_before_clip"],
                    ]
                )
        write_predictions(
            directory / "predictions.csv", validation, val_y, {"GRU": prediction}
        )
        save_learning_curves(
            history,
            result.best_epoch,
            f"GRU validation development | lr={rate:g}, seed={seed}",
            directory / "learning_curves.png",
        )
        records.append(record)
        predictions_by_run[run_name] = prediction
        print(
            json.dumps(
                {
                    "completed": run_name,
                    "best_epoch": result.best_epoch,
                    "validation_rmse": best_score,
                    "status": record["status"],
                }
            ),
            flush=True,
        )
        return record

    for rate in plan["learning_rates"]:
        candidate = run(rate, plan["selection_seed"], "setting selection")
        if candidate is not None and candidate["status"] == "complete":
            candidates.append(candidate)
    if len(candidates) == len(plan["learning_rates"]):
        selected_rate = select_learning_rate(
            candidates, plan["setting_tie_tolerance_mm_s"]
        )
        write_json(
            output / "selection.json",
            {
                "learning_rate": selected_rate,
                "seed": 2026,
                "candidates": candidates,
                "rule": plan,
            },
        )
        for seed in plan["additional_seeds"]:
            run(selected_rate, seed, "seed repeat; not setting selection")
    metrics, seed_summary, plot = {}, {}, None
    if selected_rate is not None:
        selected = [
            r
            for r in records
            if r.get("status") == "complete" and r["learning_rate"] == selected_rate
        ]
        representative = next(
            r for r in selected if r["seed"] == plan["representative_seed"]
        )
        predictions = {**references, "GRU": predictions_by_run[representative["run"]]}
        metrics = {name: evaluate_velocity(val_y, p) for name, p in predictions.items()}
        for key in ("overall_rmse_mm_s", "rmse_mm_s", "r2", "mean_r2"):
            values = np.array([r["validation_metrics"][key] for r in selected])
            seed_summary[key] = {
                "mean": values.mean(axis=0).tolist(),
                "std_ddof0": values.std(axis=0).tolist(),
            }
        seed_summary["seeds"] = [r["seed"] for r in selected]
        write_predictions(output / "predictions.csv", validation, val_y, predictions)
        plot = save_baseline_velocity_figure(
            val_y,
            predictions,
            specification["source"]["session"],
            output / "validation_velocity.png",
        )
    total_epochs = sum(r.get("epochs", 0) for r in records)
    if (
        total_epochs > plan["max_total_epochs"]
        or sum(r.get("optimizer_updates", 0) for r in records)
        > plan["max_optimizer_updates"]
    ):
        raise ValueError("Declared epoch/update budget exceeded")
    metadata = {
        **provenance,
        "source": specification["source"],
        "split_specification": specification,
        "timing": asdict(timing),
        "normalization": scaler.to_metadata(),
        "normalization_sha256": scaler_sha256(scaler),
        "reference_predictions": reference_provenance,
        "training_shape": list(train_x.values.shape),
        "validation_shape": list(val_x.values.shape),
        "selected_learning_rate": selected_rate,
        "records": records,
        "total_epochs": total_epochs,
        "metrics": metrics,
        "seed_summary": seed_summary,
        "plot": plot,
        "test_evaluation": "no test targets/windows/predictions/metrics/plots constructed; common loader reads source/global metadata",
        "validation_usage": "selection/scoring only; absent from inputs, optimizer and transform fitting",
        "duration_s": time.monotonic() - started,
        "duration_scope": "whole experiment, not inference latency",
        "artifact_sha256": {
            p.relative_to(output).as_posix(): sha(p)
            for p in output.rglob("*")
            if p.is_file()
        },
    }
    write_json(output / "metrics.json", metrics)
    write_json(output / "run.json", metadata)
    print(
        json.dumps(
            {
                "selected_learning_rate": selected_rate,
                "metrics": metrics,
                "seed_summary": seed_summary,
                "total_epochs": total_epochs,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
