"""Synthetic-only freeze gates, independent metrics, common support and no fitting."""

import importlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from neurodecode.data.binning import bin_partition
from neurodecode.data.features import build_history_windows, fit_count_scaler
from neurodecode.data.targets import align_velocity_targets
from neurodecode.evaluation import (
    VelocitySeries,
    evaluate_velocity,
    targets_from_windows,
)
from neurodecode.final_evaluation import (
    authorize_operation,
    check_metrics,
    evaluate_prepared,
    independent_metrics,
    verify_support,
)
from neurodecode.models.baselines import TrainingMeanVelocity
from neurodecode.models.gru import GRUConfig, GRUDecoder
from neurodecode.models.ridge import RidgeDecoder, ridge_inputs
from neurodecode.splits import TimingConfig
from neurodecode.training import cpu_training_runtime
from tests.test_kalman_filter import system
from tests.test_targets import inputs


def fixture():
    source, _, spec, timing = inputs(config=TimingConfig(history_bins=3))
    source = replace(
        source,
        cursor_position_mm=np.column_stack(
            (
                10 * np.sin(8 * source.behavior_times_s),
                5 * np.cos(6 * source.behavior_times_s),
            )
        ),
    )
    spec.update(
        source={"sha256": "synthetic-only", "session": "synthetic"},
        configuration={"sha256": "synthetic-config"},
    )
    spec["partitions"] = [
        {
            "name": name,
            "retained": {"start_s": a, "end_s": b},
            "continuous_segments": [{"start_s": a, "end_s": b}],
        }
        for name, a, b in (("train", 1.0, 1.4), ("validation", 1.6, 2.0))
    ]
    train_targets = align_velocity_targets(
        source, bin_partition(source, spec, timing, "train"), spec, timing
    )
    train = build_history_windows(train_targets, spec, timing)
    scaler = fit_count_scaler(train_targets, spec, timing)
    ridge = RidgeDecoder.fit(
        ridge_inputs(train, scaler), targets_from_windows(train), scaler, 1.0
    )
    p, _, _ = system(2)
    kalman = replace(
        p,
        H=np.array([[1.0, 0.5]]),
        d=np.zeros(1),
        R=np.eye(1),
        scaler=scaler,
        diagnostics={"grid_origin_s": 1.0, "bin_width_s": 0.02},
    )
    with cpu_training_runtime(7):
        gru = GRUDecoder(scaler, GRUConfig(8))
    targets = align_velocity_targets(
        source, bin_partition(source, spec, timing, "validation"), spec, timing
    )
    windows = build_history_windows(targets, spec, timing)
    tolerances = {
        n: {"atol_mm_s": 1e-6, "rtol": 1e-6}
        for n in ("zero", "mean", "ridge", "kalman", "gru")
    }
    return (
        targets,
        windows,
        spec,
        timing,
        {"ridge": ridge, "kalman": kalman, "gru": gru},
        TrainingMeanVelocity.fit(train),
        tolerances,
    )


def test_no_fitting_common_rows_independent_metrics_and_label_mutation(monkeypatch):
    args = fixture()

    def forbidden(*a, **k):
        raise AssertionError("Fitting is forbidden during frozen evaluation")

    monkeypatch.setattr(RidgeDecoder, "fit", forbidden)
    monkeypatch.setattr(TrainingMeanVelocity, "fit", forbidden)
    with cpu_training_runtime(7):
        first = evaluate_prepared(*args)
        t, w, *rest = args
        values = t.velocity_mm_s.copy()
        values[t.valid_mask] += 1000
        second = evaluate_prepared(
            replace(t, velocity_mm_s=values), replace(w, y=w.y + 1000), *rest
        )
    for name in first["sequential"]:
        np.testing.assert_array_equal(
            first["sequential"][name].values_mm_s,
            second["sequential"][name].values_mm_s,
        )
        assert first["metrics"][name]["scored_examples"] == len(w.y)
    assert (
        first["integrity"]["parameter_digests_before"]
        == second["integrity"]["parameter_digests_after"]
    )
    assert first["metrics"]["zero"] != second["metrics"]["zero"]


@pytest.mark.parametrize(
    "case",
    ["partition", "neural_support", "behavior_support", "future", "count_alignment"],
)
def test_cross_partition_future_and_misaligned_support_rejected(case):
    t, w, spec, timing, *_ = fixture()
    if case == "partition":
        w = replace(w, partition="train")
    elif case == "neural_support":
        values = w.source_bin_start_s.copy()
        values[0, 0] = 1.0
        w = replace(w, source_bin_start_s=values)
    elif case == "behavior_support":
        values = t.source_time_s.copy()
        values[w.target_rows[0], 0] = 1.0
        t = replace(t, source_time_s=values)
    elif case == "future":
        values = t.source_time_s.copy()
        values[w.target_rows, 1] = w.target_time_s + 0.001
        t = replace(t, source_time_s=values)
    else:
        values = w.X_raw.copy()
        values[0, 0, 0] += 1
        w = replace(w, X_raw=values)
    with pytest.raises((ValueError, AssertionError)):
        verify_support(t, w, spec, timing, "validation")


@pytest.mark.parametrize("n", [1, 3])
def test_independent_constant_axis_status_is_not_fabricated(n):
    y = np.tile([3.0, 4.0], (n, 1))
    p = np.zeros((n, 2))
    metrics = independent_metrics(y, p)
    assert metrics["r2"] == [None, None] and metrics["mean_r2"] is None
    assert metrics["overall_rmse_mm_s"] == pytest.approx(np.sqrt(12.5))


def test_independent_metric_check_detects_wrong_result():
    args = fixture()
    labels = targets_from_windows(args[1])
    p = VelocitySeries(np.zeros_like(labels.values_mm_s), labels.index)
    shared = evaluate_velocity(labels, p)
    wrong = {**shared, "overall_rmse_mm_s": 0.0}
    with pytest.raises(ValueError, match="metric"):
        check_metrics(labels.values_mm_s, p.values_mm_s, wrong)


@pytest.mark.parametrize("token", [None, "wrong"])
def test_test_mode_rejected_before_recording_loader_or_model_preflight(
    tmp_path, monkeypatch, token
):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    runner = importlib.import_module("evaluate_final")
    path = tmp_path / "synthetic-freeze.json"
    path.write_text(json.dumps({"synthetic_only": True}))
    monkeypatch.setattr(runner, "FREEZE_PATH", path)

    def forbidden(*args, **kwargs):
        raise AssertionError("No data/model loading permitted")

    monkeypatch.setattr(runner, "load_verified_inputs", forbidden)
    monkeypatch.setattr(runner, "preflight", forbidden)
    with pytest.raises(ValueError, match="Test access denied"):
        runner.execute("test", "never-created", token)
    assert not (tmp_path / "runs").exists()


def test_synthetic_token_requires_committed_freeze_before_data_loader(
    tmp_path, monkeypatch
):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    runner = importlib.import_module("evaluate_final")
    freeze = {"synthetic_only": True}
    path = tmp_path / "freeze.json"
    path.write_text(json.dumps(freeze))
    monkeypatch.setattr(runner, "FREEZE_PATH", path)

    def denied(*args):
        raise ValueError("not committed")

    monkeypatch.setattr(runner, "require_committed_freeze", denied)
    monkeypatch.setattr(
        runner, "load_verified_inputs", lambda: pytest.fail("Data loader called")
    )
    with pytest.raises(ValueError, match="not committed"):
        runner.execute("test", "never-created", runner.canonical_sha(freeze))


def test_artifact_mutation_fails_before_parameter_or_source_loading(
    tmp_path, monkeypatch
):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    runner = importlib.import_module("evaluate_final")
    monkeypatch.chdir(tmp_path)
    folder = tmp_path / "runs"
    folder.mkdir()
    path = folder / "model.json"
    path.write_text("original")
    digest = runner.sha(path)
    path.write_text("corrupted")
    freeze = {
        "schema_version": 1,
        "model_order": ["zero", "mean", "ridge", "kalman", "gru"],
        "implementation_sha256_utf8_lf": {},
        "json_inputs_canonical_sha256": {},
        "packages": {},
        "python": sys.version.split()[0],
        "export_tools": {},
        "artifacts": {"ridge": {"path": str(path), "sha256": digest}},
        "validation_references": {},
    }
    with pytest.raises(ValueError, match="artifact"):
        runner.preflight(freeze)


def test_operation_gate_has_no_implicit_test_default():
    authorize_operation("validation", None, "fixed")
    authorize_operation("preflight", None, "fixed")
    with pytest.raises(ValueError):
        authorize_operation("validation", "fixed", "fixed")
