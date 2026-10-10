"""Generate the fixed early recorded-data figure from the acquisition manifest."""

import hashlib
import json
import os
import sys
from pathlib import Path

from neurodecode.data.loaders import load_recording
from neurodecode.plotting import save_recorded_figure, select_plot_interval

if __name__ == "__main__":
    if len(sys.argv) > 2:
        raise SystemExit("Usage: python scripts/plot_recording.py [RUN_FOLDER_NAME]")
    folder = sys.argv[1] if len(sys.argv) == 2 else "phase1c"
    if not folder or Path(folder).name != folder or folder in (".", ".."):
        raise SystemExit("Run folder must be a single directory name under runs/")
    os.environ.setdefault("MPLCONFIGDIR", str(Path("runs/.matplotlib").resolve()))
    manifest = json.loads(Path("data/manifest.json").read_text(encoding="utf-8"))
    if len(manifest["recordings"]) != 1:
        raise SystemExit("This checkpoint expects exactly one acquired recording")
    source = manifest["recordings"][0]
    path = Path(source["local_path"])
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != source["local_checksums"]["SHA256"]:
        raise SystemExit(
            "Recording differs from the verified manifest; figure not generated"
        )
    interval = select_plot_interval(load_recording(path))
    output = (Path("runs") / folder) / f"{source['session']}_recorded.png"
    if output.exists() or output.with_suffix(".json").exists():
        raise SystemExit(
            f"Output already exists; preserve or rename it before rerunning: {output}"
        )
    save_recorded_figure(interval, source["session"], output)
    metadata = {
        "session": source["session"],
        "source_record": manifest["dataset"]["record_id"],
        "source_version": manifest["dataset"]["version"],
        "source_sha256": digest.hexdigest(),
        "interval_s": [interval.start_s, interval.end_s],
        "interval_convention": "[start, end)",
        "provisional_training_end_s": interval.provisional_training_end_s,
        "selection_rule": "Earliest 10 seconds in overlap; first four lexical channel IDs, first two unit slots",
        "gap_policy": "Fail if loader-defined behavior gap intersects interval; no retry or interpolation",
        "behavior_samples": int(interval.behavior_time_relative_s.size),
        "trains": [
            {
                "channel_id": train.channel_id,
                "unit_index": train.unit_index,
                "interval_events": int(train.time_relative_s.size),
            }
            for train in interval.trains
        ],
        "figure": output.as_posix(),
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))
