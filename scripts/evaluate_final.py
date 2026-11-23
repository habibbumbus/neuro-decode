"""Frozen evaluation: preflight/validation are safe; test requires a committed freeze."""

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
from audit_gru import read_rows
from bin_recording import load_verified_inputs
from evaluate_baselines import git_state
from render_cursor_replay import export_animation
from replay_validation import sha, write_json
from threadpoolctl import threadpool_limits

from neurodecode.data.features import FrozenCountScaler, prepare_partition
from neurodecode.demo import audit_tracker_alignment, integrate_velocity
from neurodecode.final_evaluation import (
    MODEL_ORDER,
    authorize_operation,
    evaluate_prepared,
)
from neurodecode.models.baselines import TrainingMeanVelocity
from neurodecode.models.gru import GRUDecoder
from neurodecode.models.kalman import KalmanParameters
from neurodecode.models.ridge import RidgeDecoder
from neurodecode.streaming import iter_replay_bins
from neurodecode.training import cpu_training_runtime

FREEZE_PATH = Path("configs/final_evaluation.json")


def canonical_sha(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def preflight(freeze):
    """Only metadata/file integrity and model loading; no recording/partition loader."""
    if freeze["schema_version"] != 1 or freeze["model_order"] != list(MODEL_ORDER):
        raise ValueError("Unsupported frozen protocol")
    for relative, expected in freeze["implementation_sha256_utf8_lf"].items():
        if (
            hashlib.sha256(
                Path(relative).read_text(encoding="utf-8").encode()
            ).hexdigest()
            != expected
        ):
            raise ValueError(f"Frozen implementation changed: {relative}")
    for relative, expected in freeze["json_inputs_canonical_sha256"].items():
        if (
            canonical_sha(json.loads(Path(relative).read_text(encoding="utf-8")))
            != expected
        ):
            raise ValueError(f"Frozen source/split metadata changed: {relative}")
    for name, expected in freeze["packages"].items():
        if version(name) != expected:
            raise ValueError(f"Frozen package version changed: {name}")
    if platform.python_version() != freeze["python"]:
        raise ValueError("Frozen Python version changed")
    for tool, expected in freeze["export_tools"].items():
        binary = shutil.which(tool)
        if (
            binary is None
            or subprocess.run(
                [binary, "-version"], capture_output=True, text=True, check=True
            ).stdout.splitlines()[0]
            != expected
        ):
            raise ValueError(f"Frozen export tool changed/unavailable: {tool}")
    for item in list(freeze["artifacts"].values()) + list(
        freeze["validation_references"].values()
    ):
        path = Path(item["path"]).resolve()
        if (
            not path.is_relative_to(Path("runs").resolve())
            or sha(path) != item["sha256"]
        ):
            raise ValueError(f"Frozen trusted artifact changed: {path.name}")
    scaler = FrozenCountScaler.from_metadata(freeze["normalization"])
    models = {
        "ridge": RidgeDecoder.load(
            freeze["artifacts"]["ridge"]["path"], expected_scaler=scaler
        ),
        "kalman": KalmanParameters.load(
            freeze["artifacts"]["kalman"]["path"], expected_scaler=scaler
        ),
        "gru": GRUDecoder.load(
            freeze["artifacts"]["gru"]["path"], expected_scaler=scaler
        ),
    }
    baseline = json.loads(
        Path(freeze["artifacts"]["mean"]["path"]).read_text(encoding="utf-8")
    )
    if (
        baseline["training_mean_mm_s"] != freeze["mean_mm_s"]
        or baseline["mean_fit_examples"] != 13070
        or baseline["identity"] != dict(scaler.identity)
        or models["ridge"].alpha != 1000
        or models["ridge"].fitted_examples != 13070
        or models["gru"].parameter_count != 12546
        or not np.array_equal(models["kalman"].initial_mean, freeze["mean_mm_s"])
    ):
        raise ValueError("Fitted-model/training-mean provenance mismatch")
    mean_model = TrainingMeanVelocity(
        np.array(freeze["mean_mm_s"]),
        13070,
        scaler.channel_ids,
        scaler.unit_indices,
        scaler.identity,
    )
    return models, mean_model


def read_committed_utf8(relative):
    """Read HEAD's bytes, decode strict UTF-8, normalize CRLF/CR to LF.

    This matches Path.read_text(encoding="utf-8") used by local preflight.
    Never use locale-dependent subprocess text decoding for a frozen identity.
    Preserve all other characters, including any BOM; invalid UTF-8 fails closed.
    """
    prefix = ["git", "-c", f"safe.directory={Path.cwd().as_posix()}"]
    raw = subprocess.run(
        prefix + ["show", f"HEAD:{relative}"],
        capture_output=True,
        check=True,
    ).stdout
    return (
        raw.decode("utf-8", errors="strict").replace("\r\n", "\n").replace("\r", "\n")
    )


def verify_committed_implementation(freeze):
    """Compare every declared HEAD source identity, never just working files."""
    for relative, expected in freeze["implementation_sha256_utf8_lf"].items():
        committed_code = read_committed_utf8(relative)
        if hashlib.sha256(committed_code.encode("utf-8")).hexdigest() != expected:
            raise ValueError(
                f"Frozen implementation is not committed at HEAD: {relative}"
            )


def require_committed_freeze(freeze):
    committed = read_committed_utf8(FREEZE_PATH.as_posix())
    if canonical_sha(json.loads(committed)) != canonical_sha(freeze):
        raise ValueError("Current freeze is not committed exactly at HEAD")
    verify_committed_implementation(freeze)
    evidence = Path("runs") / freeze["validation_run_id"]
    run = json.loads((evidence / "run.json").read_text(encoding="utf-8"))
    verification = json.loads(
        (evidence / "verification.json").read_text(encoding="utf-8")
    )
    if (
        run["status"] != "passed"
        or run["partition"] != "validation"
        or run["freeze_sha256"] != canonical_sha(freeze)
        or verification["run_json_sha256"] != sha(evidence / "run.json")
    ):
        raise ValueError(
            "Passing audited validation rehearsal for this exact freeze is required"
        )


def select_clip(windows, frozen_clip):
    rows = np.flatnonzero(
        (windows.target_time_s >= frozen_clip["start_s"])
        & (windows.target_time_s < frozen_clip["end_s_exclusive"])
    )
    if (
        len(rows) != 500
        or not np.array_equal(
            windows.target_bin_end_indices[rows],
            np.arange(frozen_clip["first_bin_id"], frozen_clip["first_bin_id"] + 500),
        )
        or np.any(windows.segment_ids[rows] != windows.segment_ids[rows[0]])
    ):
        raise ValueError(
            "Timestamp-fixed clip lacks 500 contiguous eligible rows; no replacement clip"
        )
    return rows


def compare_validation_references(result, windows, freeze):
    for key, models in (
        ("five_model", MODEL_ORDER),
        ("sequential", ("ridge", "kalman", "gru")),
    ):
        rows = read_rows(Path(freeze["validation_references"][key]["path"]))
        labels = result["labels"]
        if len(rows) != len(windows.target_rows) or len(rows) != 2775:
            raise ValueError("Established validation population changed")
        for i, row in enumerate(rows):
            if (
                int(row["bin_id"]) != labels.index.bin_ids[i]
                or float(row["time_s"]) != labels.index.time_s[i]
                or int(row["segment_id"]) != labels.index.segment_ids[i]
                or row["partition"] != "validation"
                or row["score_mask"] != "True"
                or int(row["first_source_bin_id"])
                != windows.source_bin_end_indices[i, 0]
                or int(row["last_source_bin_id"])
                != windows.source_bin_end_indices[i, -1]
                or any(
                    float(row[f"actual_v{a}_mm_s"]) != labels.values_mm_s[i, axis]
                    for axis, a in enumerate(("x", "y"))
                )
            ):
                raise ValueError(
                    "Established validation source/target identity changed"
                )
        for name in models:
            expected = np.array(
                [[float(row[f"{name}_v{a}_mm_s"]) for a in ("x", "y")] for row in rows]
            )
            tolerance = freeze["tolerances"][name]
            np.testing.assert_allclose(
                result["sequential"][name].values_mm_s,
                expected,
                atol=tolerance["atol_mm_s"],
                rtol=tolerance["rtol"],
            )


def save_figures(output, result, rows, partition):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    labels = result["labels"]
    colors = ("#d95f02", "#16803b", "#b52b2d", "#7b3fb2", "#008b8b")
    figure = Figure(figsize=(12, 6), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 1, sharex=True)
    t = labels.index.time_s[rows]
    for axis, name in enumerate(("x", "y")):
        axes[axis].plot(
            t - t[0],
            labels.values_mm_s[rows, axis],
            label="Tracker-derived target",
            color="#172536",
            linewidth=1,
        )
        for model, color in zip(MODEL_ORDER, colors, strict=True):
            axes[axis].plot(
                t - t[0],
                result["sequential"][model].values_mm_s[rows, axis],
                label=model,
                color=color,
                linewidth=1,
            )
        axes[axis].set_ylabel(f"v{name} [mm/s]")
        axes[axis].grid(alpha=0.2)
        axes[axis].legend(frameon=False, ncols=3)
    axes[-1].set(xlabel=f"Seconds since {t[0]:.3f} s", xlim=(0, 10))
    figure.suptitle(
        f"{partition.title()} velocity | frozen timestamp-selected first 10 seconds"
    )
    figure.savefig(output / "velocity.png", dpi=150, facecolor="white")
    figure = Figure(figsize=(10, 5), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2)
    axes[0].bar(
        MODEL_ORDER,
        [result["metrics"][n]["overall_rmse_mm_s"] for n in MODEL_ORDER],
        color=colors,
    )
    axes[0].set_ylabel("Overall RMSE [mm/s]")
    axes[1].bar(
        MODEL_ORDER,
        [
            np.nan
            if result["metrics"][n]["mean_r2"] is None
            else result["metrics"][n]["mean_r2"]
            for n in MODEL_ORDER
        ],
        color=colors,
    )
    axes[1].set_ylabel("Mean axis R²; undefined values omitted")
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle(
        f"Frozen {partition} evaluation | every model on the same eligible rows"
    )
    figure.savefig(output / "metrics.png", dpi=150, facecolor="white")


def execute(operation, run_id, token):
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    freeze_digest = canonical_sha(freeze)
    authorize_operation(operation, token, freeze_digest)
    if operation == "test":
        require_committed_freeze(freeze)
    models, mean = preflight(freeze)
    if operation == "preflight":
        print(
            json.dumps(
                {
                    "preflight": "passed; no data loader called",
                    "freeze_sha256": freeze_digest,
                    "test_access": False,
                },
                indent=2,
            )
        )
        return
    expected_run = (
        freeze["validation_run_id"]
        if operation == "validation"
        else freeze["test_run_id"]
    )
    if run_id != expected_run:
        raise ValueError(f"Frozen run ID required: {expected_run}")
    output = Path("runs") / run_id
    if output.exists():
        raise ValueError(f"Existing evaluation/access evidence preserved: {output}")
    output.mkdir(parents=True)
    os.environ.setdefault("MPLCONFIGDIR", str(Path("runs/.matplotlib").resolve()))
    protocol = {
        "freeze_sha256": freeze_digest,
        "freeze": freeze,
        "git": git_state(),
        "operation": operation,
    }
    write_json(output / "protocol.json", protocol)
    write_json(
        output / "access_event.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "operation": operation,
            "test_opening_requested": operation == "test",
            "freeze_sha256": freeze_digest,
            "stage": "authorization/preflight passed; before recording load and partition construction",
        },
    )
    stage = "load source for explicitly selected partition"
    try:
        recording, specification, timing = load_verified_inputs()
        if specification["source"] != freeze["source"]:
            raise ValueError("Source identity changed")
        stage = f"prepare {operation} only"
        targets, windows = prepare_partition(
            recording, specification, timing, operation
        )
        rows = select_clip(windows, freeze["clips"][operation])
        stage = "frozen offline/replay integrity and independent metrics"
        with threadpool_limits(limits=1), cpu_training_runtime(2026):
            result = evaluate_prepared(
                targets,
                windows,
                specification,
                timing,
                models,
                mean,
                freeze["tolerances"],
            )
        if operation == "validation":
            compare_validation_references(result, windows, freeze)
        stage = "export common rows and tables"
        labels = result["labels"]
        with (output / "predictions.csv").open(
            "x", newline="", encoding="utf-8"
        ) as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "partition",
                    "segment_id",
                    "bin_id",
                    "time_s",
                    "bin_start_s",
                    "first_source_bin_id",
                    "first_source_start_s",
                    "behavior_start_s",
                    "behavior_end_s",
                    "actual_vx_mm_s",
                    "actual_vy_mm_s",
                    *[
                        f"{prefix}{name}_v{a}_mm_s"
                        for prefix in ("", "offline_")
                        for name in MODEL_ORDER
                        for a in ("x", "y")
                    ],
                ]
            )
            for i, raw in enumerate(windows.target_rows):
                writer.writerow(
                    [
                        operation,
                        int(windows.segment_ids[i]),
                        int(labels.index.bin_ids[i]),
                        float(labels.index.time_s[i]),
                        float(targets.bins.bin_start_s[raw]),
                        int(windows.source_bin_end_indices[i, 0]),
                        float(windows.source_bin_start_s[i, 0]),
                        *targets.source_time_s[raw],
                        *labels.values_mm_s[i],
                        *[
                            v
                            for key in ("sequential", "offline")
                            for name in MODEL_ORDER
                            for v in result[key][name].values_mm_s[i]
                        ],
                    ]
                )
        with (output / "metrics.csv").open("x", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "model",
                    "N",
                    "first_time_s",
                    "last_time_s",
                    "rmse_x_mm_s",
                    "rmse_y_mm_s",
                    "overall_rmse_mm_s",
                    "r2_x",
                    "r2_y",
                    "mean_r2",
                    "r2_status",
                ]
            )
            for name in MODEL_ORDER:
                m = result["metrics"][name]
                writer.writerow(
                    [
                        name,
                        m["scored_examples"],
                        m["first_target_time_s"],
                        m["last_target_time_s"],
                        *m["rmse_mm_s"],
                        m["overall_rmse_mm_s"],
                        *m["r2"],
                        m["mean_r2"],
                        json.dumps(m["r2_status"]),
                    ]
                )
        table = f"# Frozen {operation} evaluation\n\n| Model | N | RMSE x | RMSE y | Overall RMSE | R² x | R² y | Mean R² |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        for name in MODEL_ORDER:
            m = result["metrics"][name]
            values = [*m["rmse_mm_s"], m["overall_rmse_mm_s"], *m["r2"], m["mean_r2"]]
            table += (
                f"| {name} | {m['scored_examples']} | "
                + " | ".join("undefined" if v is None else f"{v:.6f}" for v in values)
                + " |\n"
            )
        (output / "metrics.md").write_text(table, encoding="utf-8")
        emitted = set(int(i) for i in labels.index.bin_ids)
        with (output / "exclusions.csv").open(
            "x", newline="", encoding="utf-8"
        ) as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "partition",
                    "segment_id",
                    "bin_id",
                    "time_s",
                    "emitted_all_models",
                    "reason",
                ]
            )
            for p in iter_replay_bins(targets, specification, timing):
                reason = (
                    "eligible"
                    if p.bin_id in emitted
                    else targets.reasons[p.source_row]
                    if targets.reasons[p.source_row] != "valid"
                    else "incomplete_or_invalid_history"
                )
                writer.writerow(
                    [
                        p.partition,
                        p.segment_id,
                        p.bin_id,
                        p.end_s,
                        p.bin_id in emitted,
                        reason,
                    ]
                )
        write_json(
            output / "quality_summary.json",
            {
                "prepared": windows.summary,
                "target_reasons": dict(Counter(targets.reasons)),
                "segments": targets.bins.segment_metadata,
                "coverage_of_common_eligible_rows": {n: 1.0 for n in MODEL_ORDER},
            },
        )
        write_json(output / "integrity.json", result["integrity"])
        save_figures(output, result, rows, operation)
        stage = "fixed cursor replay and encoded-video integrity"
        raw = windows.target_rows[rows]
        starts, ends = targets.bins.bin_start_s[raw], targets.bins.bin_end_s[raw]
        audit = audit_tracker_alignment(
            starts,
            ends,
            labels.values_mm_s[rows],
            targets.source_time_s[raw],
            targets.source_indices[raw],
            recording.cursor_position_mm[targets.source_indices[raw]],
        )
        paths = {
            "reference": integrate_velocity(
                labels.values_mm_s[rows],
                starts,
                ends,
                labels.index.bin_ids[rows],
                windows.segment_ids[rows],
                np.zeros(2),
            )
        }
        for name in ("ridge", "gru", "kalman"):
            paths[name] = integrate_velocity(
                result["sequential"][name].values_mm_s[rows],
                starts,
                ends,
                labels.index.bin_ids[rows],
                windows.segment_ids[rows],
                np.zeros(2),
            )
        settings = {
            **freeze["export"],
            "interval_start_s": freeze["clips"][operation]["start_s"],
        }
        visual = export_animation(
            output, paths, ends, settings, audit, result["metrics"], partition=operation
        )
        probe = json.loads(
            subprocess.run(
                [
                    shutil.which("ffprobe"),
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height,nb_frames,avg_frame_rate,duration",
                    "-of",
                    "json",
                    str(output / "cursor_replay.mp4"),
                ],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        )
        stream = probe["streams"][0]
        if (
            stream["nb_frames"] != "500"
            or stream["avg_frame_rate"] != "50/1"
            or float(stream["duration"]) != 10
            or stream["width"] != 1600
            or stream["height"] != 900
        ):
            raise ValueError("Frozen video properties failed")
        subprocess.run(
            [
                shutil.which("ffmpeg"),
                "-v",
                "error",
                "-i",
                str(output / "cursor_replay.mp4"),
                "-vf",
                r"select=eq(n\,0)+eq(n\,250)+eq(n\,499)",
                "-fps_mode",
                "vfr",
                "-frames:v",
                "3",
                str(output / "encoded_frame_%02d.png"),
            ],
            check=True,
        )
        write_json(
            output / "run.json",
            {
                "status": "passed",
                "partition": operation,
                "freeze_sha256": freeze_digest,
                "git": git_state(),
                "source": freeze["source"],
                "identity": dict(windows.identity),
                "metrics": result["metrics"],
                "clip": freeze["clips"][operation],
                "alignment_audit": audit,
                "visual": visual,
                "video_probe": probe,
                "validation_references_checked": operation == "validation",
                "test_examples_constructed": operation == "test",
                "artifact_sha256": {
                    p.name: sha(p) for p in output.iterdir() if p.is_file()
                },
            },
        )
        print(
            json.dumps(
                {
                    "operation": operation,
                    "status": "passed; audit command still required",
                    "N": len(windows.target_rows),
                    "output": output.as_posix(),
                    "freeze_sha256": freeze_digest,
                },
                indent=2,
            )
        )
    except Exception as error:
        write_json(
            output / "failure.json",
            {
                "status": "failed; stop, no model/clip substitution",
                "stage": stage,
                "exception": f"{type(error).__name__}: {error}",
                "test_access_requested": operation == "test",
            },
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["preflight", "validation", "test"])
    parser.add_argument("--run-id")
    parser.add_argument("--open-test-sha256")
    args = parser.parse_args()
    execute(args.operation, args.run_id, args.open_test_sha256)


if __name__ == "__main__":
    main()
