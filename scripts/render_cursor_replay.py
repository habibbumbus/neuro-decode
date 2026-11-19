"""Phase 6C: fixed validation cursor animation from audited sequential predictions."""

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tomllib
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
from audit_gru import read_rows
from bin_recording import load_verified_inputs
from evaluate_baselines import git_state
from replay_validation import sha, write_json

from neurodecode.data.features import prepare_partition
from neurodecode.demo import audit_tracker_alignment, integrate_velocity
from neurodecode.evaluation import (
    VelocitySeries,
    evaluate_velocity,
    targets_from_windows,
)


def export_animation(output, paths, clip_times, settings, audit, validation_metrics):
    """Fixed axes and synchronized 50-fps endpoints; no inference or corrections."""
    from matplotlib.animation import FFMpegWriter
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    figure = Figure(
        figsize=(
            settings["width_px"] / settings["dpi"],
            settings["height_px"] / settings["dpi"],
        ),
        dpi=settings["dpi"],
        facecolor="#ffffff",
    )
    FigureCanvasAgg(figure)
    figure.text(
        0.055,
        0.945,
        "Offline replay of recorded intracortical activity",
        fontsize=23,
        weight="bold",
        color="#172536",
    )
    figure.text(
        0.055,
        0.905,
        "VALIDATION ONLY  |  indy_20161005_06  |  fixed first 10 seconds  |  not live brain control",
        fontsize=12,
        color="#4b5563",
    )
    axis = figure.add_axes([0.06, 0.15, 0.48, 0.70])
    all_points = np.concatenate([p.position_mm for p in paths.values()])
    low, high = all_points.min(axis=0), all_points.max(axis=0)
    center = (low + high) / 2
    half = max(float(np.max(high - low)), 20.0) * 0.56
    limits = {
        "x": [float(center[0] - half), float(center[0] + half)],
        "y": [float(center[1] - half), float(center[1] + half)],
    }
    axis.set(
        xlim=limits["x"],
        ylim=limits["y"],
        xlabel="Relative cursor x [mm]",
        ylabel="Relative cursor y [mm]",
    )
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.20)
    axis.spines[["top", "right"]].set_visible(False)
    axis.scatter([0], [0], marker="x", color="#6b7280", s=60, linewidth=1.5, zorder=6)
    styles = {
        "reference": (
            "#172536",
            "-",
            "tracker-derived reference\nvelocity integration",
        ),
        "ridge": ("#1671bd", "-", "Ridge"),
        "gru": ("#d66a00", "--", "GRU, seed 2026"),
        "kalman": ("#803da1", "-.", "Kalman"),
    }
    lines, heads, handles = {}, {}, []
    for name, (color, style, label) in styles.items():
        (lines[name],) = axis.plot(
            [],
            [],
            color=color,
            linestyle=style,
            linewidth=2.0 if name == "reference" else 1.8,
            zorder=5 if name == "reference" else 3,
        )
        (heads[name],) = axis.plot(
            [],
            [],
            marker="o",
            color=color,
            markersize=7 if name == "reference" else 5,
            linestyle="none",
            zorder=7,
        )
        handles.append(
            Line2D(
                [],
                [],
                color=color,
                linestyle=style,
                linewidth=2,
                marker="o",
                markersize=5,
                label=label,
            )
        )
    clock = figure.text(0.59, 0.815, "", fontsize=20, weight="bold", color="#172536")
    figure.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.582, 0.78),
        frameon=False,
        fontsize=12,
        labelspacing=0.9,
        handlelength=3,
    )
    figure.text(
        0.59,
        0.525,
        "All paths share one initial anchor",
        fontsize=15,
        weight="bold",
        color="#172536",
    )
    figure.text(
        0.59,
        0.450,
        "(0, 0) mm at 1550.180 s, before the clip.\n"
        "Position = anchor + sum(velocity x neural-bin duration).\n"
        "Actual durations; no smoothing or position corrections.",
        fontsize=11,
        linespacing=1.6,
        color="#374151",
    )
    figure.text(
        0.59,
        0.395,
        "Full-validation velocity RMSE [mm/s]",
        fontsize=14,
        weight="bold",
        color="#172536",
    )
    ranking = "   |   ".join(
        f"{('GRU' if name == 'gru' else name.title())} {validation_metrics[name]['overall_rmse_mm_s']:.3f}"
        for name in ("ridge", "gru", "kalman")
    )
    figure.text(0.59, 0.360, ranking, fontsize=12, color="#374151")
    figure.text(
        0.59,
        0.325,
        "Ridge has the lowest validation velocity RMSE.\nAnimation does not establish accuracy.",
        fontsize=11,
        color="#374151",
    )
    figure.text(
        0.59,
        0.265,
        "Current distance from integrated reference [mm]",
        fontsize=12,
        color="#172536",
    )
    drift = figure.text(0.59, 0.230, "", fontsize=12, color="#374151")
    figure.text(
        0.59,
        0.145,
        "Reference integrates saved tracker-derived velocities.\n"
        f"{audit['duration_mismatches_over_1ns']}/500 tracker intervals differ from neural-bin durations;\n"
        "this line is not the original sampled tracker-position path.",
        fontsize=11,
        linespacing=1.5,
        color="#4b5563",
    )
    figure.text(
        0.055,
        0.045,
        "1x playback (50 fps) is separate from processing latency.  Test partition remains closed.",
        fontsize=11,
        color="#4b5563",
    )
    figure.text(
        0.055,
        0.018,
        "Recorded nonhuman-primate reaching data: O'Doherty et al., Zenodo 583331 v1.  Common fixed axes include Kalman drift.",
        fontsize=10,
        color="#6b7280",
    )

    writer = FFMpegWriter(
        fps=settings["fps"],
        codec=settings["codec"],
        extra_args=[
            "-crf",
            str(settings["crf"]),
            "-preset",
            settings["preset"],
            "-pix_fmt",
            settings["pixel_format"],
            "-movflags",
            "+faststart",
        ],
        metadata={
            "title": "NeuroDecode validation cursor replay",
            "comment": "Offline replay of recorded intracortical activity; not live brain control; no trajectory corrections",
        },
    )
    inspections = {}
    with writer.saving(figure, output / "cursor_replay.mp4", dpi=settings["dpi"]):
        for frame, time_s in enumerate(clip_times):
            for name, path in paths.items():
                xy = path.position_mm[: frame + 2]
                lines[name].set_data(xy[:, 0], xy[:, 1])
                heads[name].set_data([xy[-1, 0]], [xy[-1, 1]])
            clock.set_text(
                f"Elapsed {time_s - settings['interval_start_s']:05.2f} s   |   bin end {time_s:.3f} s"
            )
            reference = paths["reference"].position_mm[frame + 1]
            drift.set_text(
                "   |   ".join(
                    f"{('GRU' if name == 'gru' else name.title())} {np.linalg.norm(paths[name].position_mm[frame + 1] - reference):.1f}"
                    for name in ("ridge", "gru", "kalman")
                )
            )
            assert (
                list(axis.get_xlim()) == limits["x"]
                and list(axis.get_ylim()) == limits["y"]
            )
            if frame in settings["inspection_frame_indices"]:
                filename = {
                    0: "frame_first.png",
                    250: "frame_middle.png",
                    499: "frame_last.png",
                }[frame]
                figure.savefig(
                    output / filename, dpi=settings["dpi"], facecolor="white"
                )
                inspections[str(frame)] = {
                    "bin_end_s": float(time_s),
                    "clip_elapsed_s": float(time_s - settings["interval_start_s"]),
                    "axis_limits_mm": limits,
                }
                if frame == settings["still_frame_index"]:
                    shutil.copyfile(output / filename, output / "still.png")
            writer.grab_frame(facecolor="white")
            if frame % 125 == 0 or frame == len(clip_times) - 1:
                print(f"Exported frame {frame + 1}/{len(clip_times)}", flush=True)
    return {"common_axis_limits_mm": limits, "inspection_frames": inspections}


def main():
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
        raise ValueError(f"Existing media preserved: {output}")
    settings = tomllib.loads(Path("configs/demo.toml").read_text(encoding="utf-8"))
    if (
        settings["partition"] != "validation"
        or settings["interval_start_s"] != 1550.2
        or settings["interval_end_s_exclusive"] != 1560.2
        or settings["expected_samples"] != 500
        or settings["models"] != ["ridge", "gru", "kalman"]
        or settings["gru_seed"] != 2026
        or settings["fps"] != 50
        or settings["initial_position_mm"] != [0.0, 0.0]
        or settings["still_frame_index"] != 250
        or settings["inspection_frame_indices"] != [0, 250, 499]
    ):
        raise ValueError("This export implements only the fixed Phase 6C presentation")
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise RuntimeError(
            "Existing ffmpeg and ffprobe executables are required on PATH"
        )
    os.environ.setdefault("MPLCONFIGDIR", str(Path("runs/.matplotlib").resolve()))
    directory = Path(settings["prediction_run"])
    reference_metadata = json.loads(
        (directory / "run.json").read_text(encoding="utf-8")
    )
    verification = json.loads(
        (directory / "verification.json").read_text(encoding="utf-8")
    )
    if sha(directory / "run.json") != verification["run_json_sha256"]:
        raise ValueError("Phase 6A metadata changed after verification")
    for filename, expected in reference_metadata["artifact_sha256"].items():
        if sha(directory / filename) != expected:
            raise ValueError(f"Changed Phase 6A evidence: {filename}")
    for name, artifact in reference_metadata["model_artifacts"].items():
        path = Path(artifact["path"]).resolve()
        if (
            not path.is_relative_to(Path("runs").resolve())
            or sha(path) != artifact["sha256"]
        ):
            raise ValueError(f"Changed trusted model identity: {name}")
    recording, specification, timing = load_verified_inputs()
    if specification != reference_metadata["split_specification"]:
        raise ValueError("Source/split changed")
    targets, windows = prepare_partition(recording, specification, timing, "validation")
    if dict(windows.identity) != reference_metadata["identity"]:
        raise ValueError("Preparation identity changed")
    labels = targets_from_windows(windows)
    saved = read_rows(directory / "predictions.csv")
    if len(saved) != len(labels.index.bin_ids):
        raise ValueError("Saved prediction row population changed")
    for i, row in enumerate(saved):
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
            raise ValueError(
                "Saved prediction/target timestamp or source identity mismatch"
            )
    selected = np.flatnonzero(
        (labels.index.time_s >= settings["interval_start_s"])
        & (labels.index.time_s < settings["interval_end_s_exclusive"])
    )
    if len(selected) != settings["expected_samples"]:
        raise ValueError(
            "Fixed interval must retain exactly 500 rows; no alternate clip"
        )
    raw_rows = windows.target_rows[selected]
    starts, ends = targets.bins.bin_start_s[raw_rows], targets.bins.bin_end_s[raw_rows]
    ids, segments = labels.index.bin_ids[selected], labels.index.segment_ids[selected]
    positions = recording.cursor_position_mm[targets.source_indices[raw_rows]]
    audit = audit_tracker_alignment(
        starts,
        ends,
        labels.values_mm_s[selected],
        targets.source_time_s[raw_rows],
        targets.source_indices[raw_rows],
        positions,
    )
    velocities = {"reference": labels.values_mm_s[selected].copy()}
    validation_metrics = {}
    for name in settings["models"]:
        values = np.array(
            [
                [float(row[f"{name}_v{axis}_mm_s"]) for axis in ("x", "y")]
                for row in saved
            ]
        )
        velocities[name] = values[selected]
        validation_metrics[name] = evaluate_velocity(
            labels, VelocitySeries(values, labels.index)
        )
    paths = {
        name: integrate_velocity(
            values,
            starts,
            ends,
            ids,
            segments,
            np.array(settings["initial_position_mm"]),
            labels.index.score_mask[selected],
        )
        for name, values in velocities.items()
    }
    for path in paths.values():
        np.testing.assert_array_equal(path.time_s[1:], ends)
        np.testing.assert_array_equal(
            path.position_mm[0], settings["initial_position_mm"]
        )
    output.mkdir(parents=True)
    sources = list(Path("src/neurodecode").rglob("*.py")) + [
        Path(__file__),
        Path("scripts/bin_recording.py"),
        Path("configs/demo.toml"),
    ]
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_state(),
        "settings": settings,
        "source": specification["source"],
        "split_specification": specification,
        "identity": dict(windows.identity),
        "prediction_run_json_sha256": sha(directory / "run.json"),
        "predictions_csv_sha256": sha(directory / "predictions.csv"),
        "model_artifacts": reference_metadata["model_artifacts"],
        "channel_ids": reference_metadata["channel_ids"],
        "normalization_sha256": reference_metadata["normalization_sha256"],
        "alignment_audit": audit,
        "initial_anchor": {
            "time_s": float(starts[0]),
            "position_mm": settings["initial_position_mm"],
            "meaning": "shared relative origin; no subsequent reference correction",
        },
        "selected_bin_ids": ids.tolist(),
        "selected_bin_start_s": starts.tolist(),
        "selected_bin_end_s": ends.tolist(),
        "selected_velocity_input_sha256": {
            name: hashlib.sha256(value.astype("<f8").tobytes()).hexdigest()
            for name, value in velocities.items()
        },
        "full_validation_metrics": validation_metrics,
        "integration_rule": settings["integration"],
        "neural_duration_sum_s": float((ends - starts).sum()),
        "frame_timing": "500 bin-end frames at 50 fps; first elapsed 0, last elapsed 9.98; 10-second encoded video; anchor is 0.02 s before clip start",
        "test": "validation only; no model inference/refit or test windows/scores; common source/global metadata loader unchanged",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                name: version(name) for name in ("numpy", "matplotlib", "pillow")
            },
        },
        "export_tools": {
            tool: {
                "path": path,
                "version": subprocess.run(
                    [path, "-version"], capture_output=True, text=True, check=True
                ).stdout.splitlines()[0],
            }
            for tool, path in (("ffmpeg", ffmpeg), ("ffprobe", ffprobe))
        },
        "implementation_sha256_utf8_lf": {
            p.as_posix(): hashlib.sha256(
                p.read_text(encoding="utf-8").encode()
            ).hexdigest()
            for p in sources
        },
    }
    write_json(output / "inputs.json", provenance)
    with (output / "positions.csv").open("x", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "time_s",
                "clip_elapsed_s",
                "bin_id",
                "is_initial_anchor",
                *[f"{name}_{a}_mm" for name in paths for a in ("x", "y")],
            ]
        )
        for i, time_s in enumerate(paths["reference"].time_s):
            writer.writerow(
                [
                    float(time_s),
                    float(time_s - settings["interval_start_s"]),
                    "" if i == 0 else int(ids[i - 1]),
                    i == 0,
                    *[
                        float(p.position_mm[i, a])
                        for p in paths.values()
                        for a in range(2)
                    ],
                ]
            )
    with (output / "alignment.csv").open("x", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "bin_id",
                "grid_start_s",
                "grid_end_s",
                "neural_duration_s",
                "behavior_start_s",
                "behavior_end_s",
                "behavior_duration_s",
                "tracker_start_x_mm",
                "tracker_start_y_mm",
                "tracker_end_x_mm",
                "tracker_end_y_mm",
            ]
        )
        for i, raw in enumerate(raw_rows):
            writer.writerow(
                [
                    int(ids[i]),
                    float(starts[i]),
                    float(ends[i]),
                    float(ends[i] - starts[i]),
                    *targets.source_time_s[raw],
                    float(targets.elapsed_s[raw]),
                    *positions[i, 0],
                    *positions[i, 1],
                ]
            )
    visual = export_animation(output, paths, ends, settings, audit, validation_metrics)
    probe = json.loads(
        subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,nb_frames,avg_frame_rate,duration,codec_name,pix_fmt",
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
        int(stream["nb_frames"]) != 500
        or int(stream["width"]) != settings["width_px"]
        or int(stream["height"]) != settings["height_px"]
        or stream["avg_frame_rate"] != "50/1"
        or abs(float(stream["duration"]) - 10) > 1e-6
    ):
        raise ValueError("Encoded video differs from declared frame/timing settings")
    subprocess.run(
        [
            ffmpeg,
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
    metadata = {
        "inputs_json_sha256": sha(output / "inputs.json"),
        "visual": visual,
        "encoded_video_probe": probe,
        "terminal_distance_from_integrated_reference_mm": {
            name: float(
                np.linalg.norm(
                    path.position_mm[-1] - paths["reference"].position_mm[-1]
                )
            )
            for name, path in paths.items()
            if name != "reference"
        },
        "terminal_positions_mm": {
            name: path.position_mm[-1].tolist() for name, path in paths.items()
        },
        "artifact_sha256": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
    }
    write_json(output / "run.json", metadata)
    print(
        json.dumps(
            {
                "output": output.as_posix(),
                "alignment": audit,
                "probe": probe,
                "terminal_distance_mm": metadata[
                    "terminal_distance_from_integrated_reference_mm"
                ],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
