"""Phase 6B round 1: fixed validation-only disruptions on verified replay."""

import argparse
import csv
import hashlib
import json
import os
import platform
import tomllib
from dataclasses import asdict, replace
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
from audit_gru import read_rows
from bin_recording import load_verified_inputs
from evaluate_baselines import git_state
from replay_validation import sha, write_json
from threadpoolctl import threadpool_limits

from neurodecode.data.features import FrozenCountScaler, prepare_partition
from neurodecode.evaluation import (
    VelocitySeries,
    evaluate_velocity,
    targets_from_windows,
)
from neurodecode.models.gru import GRUDecoder
from neurodecode.models.kalman import KalmanParameters
from neurodecode.models.ridge import RidgeDecoder
from neurodecode.robustness import perturb_packet, planned_conditions, ranked_channels
from neurodecode.streaming import HistoryBuffer, SequentialDecoder, iter_replay_bins
from neurodecode.training import cpu_training_runtime


def packet_hash(packet) -> str:
    payload = b"missing" if packet.missing else packet.counts.astype("<i8").tobytes()
    return hashlib.sha256(payload).hexdigest()


def metrics(labels, values, mask) -> dict:
    index = replace(labels.index, score_mask=mask.copy())
    return evaluate_velocity(
        VelocitySeries(labels.values_mm_s, index), VelocitySeries(values, index)
    )


def save_figures(output, conditions, records, predictions, labels, settings):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    colors = {"ridge": "#b52b2d", "gru": "#008b8b", "kalman": "#7b3fb2"}
    by_name = {r["condition"]["name"]: r for r in records if r["status"] == "complete"}
    figure = Figure(figsize=(11, 8), layout="constrained")
    FigureCanvasAgg(figure)
    for axis, kind, xlabel in zip(
        figure.subplots(2, 2).flat,
        ("channel_loss", "thinning", "poisson", "missing"),
        (
            "Disabled physical channels [%]",
            "Removed event probability",
            "Added rate / training mean",
            "Unavailable burst [bins]",
        ),
        strict=True,
    ):
        candidates = [c for c in conditions if c.kind == kind and c.name in by_name]

        def xvalue(c):
            return (
                c.severity * 100
                if kind == "channel_loss"
                else 1 - c.severity
                if kind == "thinning"
                else c.severity
            )

        candidates.sort(key=xvalue)
        for name, color in colors.items():
            xs = [0] + [xvalue(c) for c in candidates]
            ys = [by_name["clean"]["primary_metrics"][name]["overall_rmse_mm_s"]] + [
                by_name[c.name]["primary_metrics"][name]["overall_rmse_mm_s"]
                for c in candidates
            ]
            axis.plot(xs, ys, marker="o", label=name, color=color)
        axis.set(xlabel=xlabel, ylabel="Overall RMSE [mm/s]")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    figure.suptitle("Validation disruption severity | common predeclared scoring rows")
    figure.savefig(output / "severity.png", dpi=150, facecolor="white")
    for condition_name in settings["trace_conditions"]:
        if condition_name not in predictions:
            continue
        figure = Figure(figsize=(13, 9), layout="constrained")
        FigureCanvasAgg(figure)
        axes = figure.subplots(3, 2, sharex=True)
        times = labels.index.time_s
        start, stop = settings["trace_interval_s"]
        rows = (times >= start) & (times < stop)
        for i, name in enumerate(colors):
            for a, axis_name in enumerate(("x", "y")):
                axis = axes[i, a]
                axis.plot(
                    times[rows] - start,
                    labels.values_mm_s[rows, a],
                    label="Actual",
                    color="#333333",
                    linewidth=1,
                )
                axis.plot(
                    times[rows] - start,
                    predictions["clean"][name][rows, a],
                    label="Clean",
                    color="#1768ac",
                    linewidth=1,
                )
                axis.plot(
                    times[rows] - start,
                    predictions[condition_name][name][rows, a],
                    label="Disrupted",
                    color="#d95f02",
                    linewidth=1.1,
                )
                if condition_name == "missing-5":
                    axis.axvspan(
                        4,
                        4.28,
                        color="#d95f02",
                        alpha=0.12,
                        label="No emission / recovery",
                    )
                else:
                    axis.axvspan(2, 8, color="#d95f02", alpha=0.06)
                axis.set(
                    title=f"{name}: v{axis_name}",
                    ylabel="Velocity [mm/s]",
                    xlim=(0, 10),
                )
                axis.grid(alpha=0.15)
                axis.legend(frameon=False, fontsize=8)
        for axis in axes[-1]:
            axis.set_xlabel(f"Time since {start:.3f} s [s]")
        figure.suptitle(
            f"Validation | {condition_name} | fixed [{start:.3f},{stop:.3f}) s; no smoothing"
        )
        figure.savefig(
            output / f"trace_{condition_name}.png", dpi=150, facecolor="white"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    args = parser.parse_args()
    if not args.run_id or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for c in args.run_id
    ):
        raise ValueError("Use a new simple run ID")
    output = Path("runs") / args.run_id
    if output.exists():
        raise ValueError(f"Existing run preserved: {output}")
    settings = tomllib.loads(
        Path("configs/robustness.toml").read_text(encoding="utf-8")
    )
    if (
        settings["channel_counts"] != [10, 24, 48]
        or settings["thinning_keep_probabilities"] != [0.8, 0.5]
        or settings["poisson_training_mean_multipliers"] != [0.25, 1.0]
        or settings["missing_lengths_bins"] != [1, 5]
    ):
        raise ValueError("This runner implements only the documented fixed round 1")
    reference_dir = Path("runs/phase6a-validation-timing")
    reference = json.loads((reference_dir / "run.json").read_text(encoding="utf-8"))
    for filename, expected in reference["artifact_sha256"].items():
        if sha(reference_dir / filename) != expected:
            raise ValueError(f"Changed Phase 6A artifact: {filename}")
    verification = json.loads(
        (reference_dir / "verification.json").read_text(encoding="utf-8")
    )
    if sha(reference_dir / "run.json") != verification["run_json_sha256"]:
        raise ValueError("Phase 6A metadata changed after audit")
    if (
        reference["source"]["sha256"] != settings["source_sha256"]
        or settings["partition"] != "validation"
    ):
        raise ValueError("Wrong source or partition")
    scaler = FrozenCountScaler.from_metadata(reference["normalization"])
    if len(scaler.channel_ids) != settings["expected_channels"]:
        raise ValueError("Physical channel population changed")
    conditions = planned_conditions(settings, scaler.channel_ids)
    recording, specification, timing = load_verified_inputs()
    if specification != reference["split_specification"]:
        raise ValueError("Existing split changed")
    aligned, windows = prepare_partition(recording, specification, timing, "validation")
    labels = targets_from_windows(windows)
    original = list(iter_replay_bins(aligned, specification, timing))
    saved_rows = read_rows(reference_dir / "predictions.csv")
    if (
        len(saved_rows) != len(labels.index.bin_ids)
        or len(saved_rows) != settings["expected_base_rows"]
    ):
        raise ValueError("Original eligible population changed")
    for i, row in enumerate(saved_rows):
        if (
            int(row["bin_id"]) != labels.index.bin_ids[i]
            or float(row["time_s"]) != labels.index.time_s[i]
            or int(row["segment_id"]) != labels.index.segment_ids[i]
            or row["partition"] != "validation"
            or row["score_mask"] != "True"
            or int(row["first_source_bin_id"]) != windows.source_bin_end_indices[i, 0]
            or int(row["last_source_bin_id"]) != windows.source_bin_end_indices[i, -1]
            or any(
                float(row[f"actual_v{a}_mm_s"]) != labels.values_mm_s[i, axis]
                for axis, a in enumerate(("x", "y"))
            )
        ):
            raise ValueError("Original scored identity/target changed")
    saved_predictions = {
        name: np.array(
            [
                [float(row[f"{name}_v{a}_mm_s"]) for a in ("x", "y")]
                for row in saved_rows
            ]
        )
        for name in ("ridge", "gru", "kalman")
    }
    models = {}
    classes = {"ridge": RidgeDecoder, "gru": GRUDecoder, "kalman": KalmanParameters}
    for name, artifact in reference["model_artifacts"].items():
        path = Path(artifact["path"]).resolve()
        if (
            not path.is_relative_to(Path("runs").resolve())
            or sha(path) != artifact["sha256"]
        ):
            raise ValueError("Trusted model artifact changed")
        models[name] = classes[name].load(path, expected_scaler=scaler)
    origin = specification["bin_grid"]["origin_s"]
    base_ids = labels.index.bin_ids
    variants, availability = {}, {}
    for condition in conditions:
        packets = [perturb_packet(p, condition, scaler.means) for p in original]
        variants[condition.name] = packets
        buffer = HistoryBuffer(scaler, origin, timing.bin_s)
        # Derive support only, with no weights, labels or prediction errors.
        emitted_ids = {p.bin_id for p in packets if buffer.push(p) is not None}
        availability[condition.name] = np.array(
            [int(i) in emitted_ids for i in base_ids]
        )
    primary_mask = np.logical_and.reduce(list(availability.values()))
    if primary_mask.sum() != settings["expected_primary_rows"]:
        raise ValueError("Declared shared scoring population changed")
    output.mkdir(parents=True)
    os.environ.setdefault("MPLCONFIGDIR", str(Path("runs/.matplotlib").resolve()))
    paths = list(Path("src/neurodecode").rglob("*.py")) + [
        Path(__file__),
        Path("configs/robustness.toml"),
        Path("scripts/bin_recording.py"),
    ]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_state(),
        "settings": settings,
        "source": reference["source"],
        "split_specification": specification,
        "identity": dict(windows.identity),
        "channel_ids": list(scaler.channel_ids),
        "unit_indices": [list(u) for u in scaler.unit_indices],
        "normalization": scaler.to_metadata(),
        "model_artifacts": reference["model_artifacts"],
        "phase6a_predictions_sha256": sha(reference_dir / "predictions.csv"),
        "conditions": [asdict(c) for c in conditions],
        "channel_ranking": ranked_channels(
            scaler.channel_ids, settings["source_sha256"], settings["channel_seed"]
        ),
        "affected_raw_bin_ids": {
            c.name: [
                p.bin_id
                for p in original
                if c.kind != "clean" and c.first_bin_id <= p.bin_id < c.stop_bin_id
            ]
            for c in conditions
        },
        "available_scoring_bin_ids": {
            name: base_ids[mask].tolist() for name, mask in availability.items()
        },
        "primary_scoring_bin_ids": base_ids[primary_mask].tolist(),
        "primary_excluded_bin_ids": base_ids[~primary_mask].tolist(),
        "input_packet_hashes": {
            name: [packet_hash(p) for p in packets]
            for name, packets in variants.items()
        },
        "poisson_lambda_by_multiplier": {
            str(rate): (rate * scaler.means).tolist()
            for rate in settings["poisson_training_mean_multipliers"]
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "device": "CPU",
            "threads": 1,
            "packages": {
                name: version(name)
                for name in ("numpy", "torch", "matplotlib", "threadpoolctl")
            },
        },
        "implementation_sha256_utf8_lf": {
            p.as_posix(): hashlib.sha256(
                p.read_text(encoding="utf-8").encode()
            ).hexdigest()
            for p in paths
        },
    }
    # Exact channel/bin/mask choices and rates exist before the first model replay.
    write_json(output / "manifest.json", manifest)
    records, predictions, failures = [], {}, []
    base_positions = {int(i): row for row, i in enumerate(base_ids)}
    with threadpool_limits(limits=1), cpu_training_runtime(2026):
        for condition in conditions:
            directory = output / condition.name
            directory.mkdir()
            print(f"Starting declared condition {condition.name}", flush=True)
            packets = variants[condition.name]
            with (directory / "affected_inputs.csv").open(
                "x", newline="", encoding="utf-8"
            ) as file:
                writer = csv.writer(file)
                writer.writerow(
                    ["bin_id", "time_s", "missing", "raw_before_json", "raw_after_json"]
                )
                for old, new in zip(original, packets, strict=True):
                    if (
                        condition.kind != "clean"
                        and condition.first_bin_id <= old.bin_id < condition.stop_bin_id
                    ):
                        writer.writerow(
                            [
                                old.bin_id,
                                old.end_s,
                                new.missing,
                                json.dumps(old.counts.tolist()),
                                json.dumps(
                                    None if new.counts is None else new.counts.tolist()
                                ),
                            ]
                        )
            record = {
                "condition": asdict(condition),
                "status": "complete",
                "primary_metrics": {},
                "available_metrics": {},
                "paired_clean_metrics": {},
                "resets": {},
            }
            condition_predictions = {}
            try:
                for name, model in models.items():
                    stream = SequentialDecoder(model, origin, timing.bin_s)
                    values = np.full((len(base_ids), 2), np.nan)
                    seen, resets = [], []
                    for p in packets:
                        result = stream.step(p)
                        if result is None:
                            continue
                        row = base_positions[result.bin_id]
                        if result.time_s != labels.index.time_s[
                            row
                        ] or result.source_bin_ids != tuple(
                            windows.source_bin_end_indices[row]
                        ):
                            raise ValueError(
                                "Disrupted replay changed emitted identity"
                            )
                        values[row] = result.velocity_mm_s
                        seen.append(result.bin_id)
                        if result.reset:
                            resets.append(result.bin_id)
                    available = availability[condition.name]
                    if seen != base_ids[available].tolist():
                        raise ValueError(
                            "Model emissions differ from declared common support"
                        )
                    if condition.kind == "clean":
                        np.testing.assert_array_equal(values, saved_predictions[name])
                        if metrics(labels, values, available) != metrics(
                            labels, saved_predictions[name], available
                        ):
                            raise ValueError("Clean Phase 6A metrics changed")
                    condition_predictions[name] = values
                    # Finite placeholders satisfy the scorer outside its explicit mask;
                    # unavailable outputs remain blank/NaN in exported evidence.
                    finite = np.nan_to_num(values, nan=0.0)
                    record["primary_metrics"][name] = metrics(
                        labels, finite, primary_mask
                    )
                    record["available_metrics"][name] = metrics(
                        labels, finite, available
                    )
                    record["paired_clean_metrics"][name] = metrics(
                        labels, saved_predictions[name], available
                    )
                    record["resets"][name] = resets
                    print(
                        f"{condition.name} {name}: N={int(primary_mask.sum())}, RMSE={record['primary_metrics'][name]['overall_rmse_mm_s']:.6f}",
                        flush=True,
                    )
            except (ValueError, AssertionError, RuntimeError) as error:
                record["status"] = "failed"
                record["error"] = f"{type(error).__name__}: {error}"
                failures.append({"condition": condition.name, "error": record["error"]})
                print(json.dumps(failures[-1]), flush=True)
                if condition.kind == "clean":
                    write_json(directory / "metrics.json", record)
                    raise ValueError(
                        "Clean reference failed; no disrupted inference allowed"
                    ) from error
            records.append(record)
            if record["status"] == "complete":
                predictions[condition.name] = condition_predictions
                available = availability[condition.name]
                record["emitted_rows"] = int(available.sum())
                record["coverage_fraction"] = float(available.mean())
                record["excluded_from_available_ids"] = base_ids[~available].tolist()
                with (directory / "predictions.csv").open(
                    "x", newline="", encoding="utf-8"
                ) as file:
                    writer = csv.writer(file)
                    writer.writerow(
                        [
                            "partition",
                            "segment_id",
                            "bin_id",
                            "time_s",
                            "first_source_bin_id",
                            "last_source_bin_id",
                            "primary_score",
                            "emitted",
                            "exclusion_reason",
                            "actual_vx_mm_s",
                            "actual_vy_mm_s",
                            "ridge_vx_mm_s",
                            "ridge_vy_mm_s",
                            "gru_vx_mm_s",
                            "gru_vy_mm_s",
                            "kalman_vx_mm_s",
                            "kalman_vy_mm_s",
                        ]
                    )
                    for i, bin_id in enumerate(base_ids):
                        reason = (
                            ""
                            if primary_mask[i]
                            else (
                                "missing_packet"
                                if condition.first_bin_id
                                <= bin_id
                                < condition.stop_bin_id
                                else "history_recovery"
                            )
                            if not available[i]
                            else "excluded_for_shared_scoring_with_missing_conditions"
                        )
                        writer.writerow(
                            [
                                "validation",
                                int(windows.segment_ids[i]),
                                int(bin_id),
                                float(labels.index.time_s[i]),
                                int(windows.source_bin_end_indices[i, 0]),
                                int(bin_id),
                                bool(primary_mask[i]),
                                bool(available[i]),
                                reason,
                                *labels.values_mm_s[i],
                                *[
                                    v if np.isfinite(v) else ""
                                    for name in ("ridge", "gru", "kalman")
                                    for v in condition_predictions[name][i]
                                ],
                            ]
                        )
            write_json(directory / "metrics.json", record)
    with (output / "metrics.csv").open("x", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "condition",
                "model",
                "scored_examples",
                "emitted_examples",
                "rmse_x_mm_s",
                "rmse_y_mm_s",
                "overall_rmse_mm_s",
                "r2_x",
                "r2_y",
                "mean_r2",
                "status",
            ]
        )
        for record in records:
            for name, m in record["primary_metrics"].items():
                writer.writerow(
                    [
                        record["condition"]["name"],
                        name,
                        m["scored_examples"],
                        record.get("emitted_rows", ""),
                        *m["rmse_mm_s"],
                        m["overall_rmse_mm_s"],
                        *m["r2"],
                        m["mean_r2"],
                        record["status"],
                    ]
                )
    save_figures(output, conditions, records, predictions, labels, settings)
    example_ids = [
        settings["affected_first_bin_id"] + timing.history_bins - 1,
        settings["missing_first_bin_id"],
        settings["missing_first_bin_id"] + 5 + timing.history_bins - 1,
    ]
    metadata = {
        "manifest_sha256": sha(output / "manifest.json"),
        "records": records,
        "failures": failures,
        "primary_rows": int(primary_mask.sum()),
        "base_rows": len(base_ids),
        "clean_phase6a_equality": "exact predictions and metrics on all original rows",
        "examples": [
            {
                "bin_id": bin_id,
                "time_s": float(labels.index.time_s[base_positions[bin_id]]),
                "actual_mm_s_scoring_only": labels.values_mm_s[
                    base_positions[bin_id]
                ].tolist(),
                "conditions": {
                    c: {
                        name: values[base_positions[bin_id]].tolist()
                        if np.isfinite(values[base_positions[bin_id]]).all()
                        else None
                        for name, values in series.items()
                    }
                    for c, series in predictions.items()
                },
            }
            for bin_id in example_ids
        ],
        "test": "no test windows, targets, predictions or metrics; common source/global metadata loader unchanged",
        "artifact_sha256": {
            p.relative_to(output).as_posix(): sha(p)
            for p in output.rglob("*")
            if p.is_file()
        },
    }
    write_json(output / "run.json", metadata)
    print(
        json.dumps(
            {
                "completed_conditions": len(predictions),
                "failures": failures,
                "primary_rows": int(primary_mask.sum()),
                "run": output.as_posix(),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
