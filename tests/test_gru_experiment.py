"""Round 1 selection and saved-reference alignment guards; synthetic only."""

import csv
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from neurodecode.evaluation import (
    VelocitySeries,
    evaluate_velocity,
    targets_from_windows,
)
from neurodecode.models.ridge import scaler_sha256
from tests.test_gru import synthetic


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("train_gru")


def test_setting_tie_rule_is_independent_of_candidate_order(runner):
    candidates = [
        {"learning_rate": 0.003, "validation_metrics": {"overall_rmse_mm_s": 50.0}},
        {
            "learning_rate": 0.001,
            "validation_metrics": {"overall_rmse_mm_s": 50.0 + 5e-10},
        },
    ]
    assert runner.select_learning_rate(candidates, 1e-9) == 0.001
    assert runner.select_learning_rate(candidates[::-1], 1e-9) == 0.001
    assert runner.select_learning_rate(candidates, 0.0) == 0.003
    candidates[0]["validation_metrics"]["overall_rmse_mm_s"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        runner.select_learning_rate(candidates, 1e-9)


@pytest.mark.parametrize(
    "mutation", [None, "timestamp", "target", "mask", "window", "bytes"]
)
def test_frozen_reference_reuse_requires_exact_rows_targets_and_bytes(
    runner, tmp_path, mutation
):
    scaler, _, windows = synthetic(n=2775)
    labels = targets_from_windows(windows)
    predictions = {
        name: VelocitySeries(labels.values_mm_s.copy(), labels.index)
        for name in ("Zero velocity", "Training mean", "Ridge", "Kalman")
    }
    path = tmp_path / "predictions.csv"
    runner.write_predictions(path, windows, labels, predictions)
    specification = {"source": {"session": "synthetic"}}
    index = labels.index
    metadata = {
        "source": specification["source"],
        "split_specification": specification,
        "normalization_sha256": scaler_sha256(scaler),
        "channel_ids": list(scaler.channel_ids),
        "unit_indices": [list(u) for u in scaler.unit_indices],
        "eligible_validation_identity_sha256": hashlib.sha256(
            np.column_stack((index.segment_ids, index.bin_ids)).astype("<i8").tobytes()
            + index.time_s.astype("<f8").tobytes()
        ).hexdigest(),
        "metrics": {
            name: evaluate_velocity(labels, p) for name, p in predictions.items()
        },
        "artifact_sha256": {"predictions.csv": runner.sha(path)},
    }
    if mutation:
        with path.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        field, value = {
            "timestamp": ("time_s", "1.0"),
            "target": ("actual_vx_mm_s", "999.0"),
            "mask": ("score_mask", "False"),
            "window": ("first_source_bin_id", "-1"),
            "bytes": ("ridge_vx_mm_s", "999.0"),
        }[mutation]
        rows[0][field] = value
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        if mutation != "bytes":
            metadata["artifact_sha256"]["predictions.csv"] = runner.sha(path)
    (tmp_path / "run.json").write_text(json.dumps(metadata), encoding="utf-8")
    if mutation:
        with pytest.raises(ValueError):
            runner.reference_predictions(
                tmp_path, windows, labels, specification, scaler
            )
    else:
        actual, _ = runner.reference_predictions(
            tmp_path, windows, labels, specification, scaler
        )
        for name in predictions:
            np.testing.assert_array_equal(
                actual[name].values_mm_s, predictions[name].values_mm_s
            )
