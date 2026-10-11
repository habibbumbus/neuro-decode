"""Small known-answer metrics and label-free baseline checks."""

from dataclasses import replace

import numpy as np
import pytest

from neurodecode.data.features import HistoryWindows
from neurodecode.evaluation import (
    ExampleIndex,
    VelocitySeries,
    evaluate_velocity,
    index_from_windows,
    targets_from_windows,
)
from neurodecode.models.baselines import TrainingMeanVelocity, predict_zero
from neurodecode.plotting import select_validation_interval


def windows(partition="train"):
    ids = np.arange(2, 5, dtype=np.int64)
    offset = 0 if partition == "train" else 0.2
    times = offset + ids * 0.02
    source_ids = np.column_stack((ids - 1, ids))
    return HistoryWindows(
        np.arange(12).reshape(3, 2, 2),
        np.array([[0.0, 0.0], [2.0, 4.0], [4.0, 8.0]]),
        source_ids - 1,
        source_ids,
        offset + (source_ids - 1) * 0.02,
        offset + source_ids * 0.02,
        ids - 1,
        ids,
        times,
        np.zeros(3, dtype=np.int64),
        ("M1 001", "M1 002"),
        ((0,), (0, 1)),
        partition,
        (("source_sha256", "synthetic"),),
        {"retained_windows": 3},
    )


def series(values, partition="validation"):
    values = np.asarray(values, dtype=float)
    n = len(values)
    index = ExampleIndex(
        np.arange(n) * 0.02,
        np.arange(1, n + 1),
        np.zeros(n, dtype=np.int64),
        np.ones(n, dtype=bool),
        partition,
        ("M1 001",),
        ((0,),),
        (("source_sha256", "synthetic"),),
    )
    return VelocitySeries(values, index)


def test_hand_metrics_both_axes_and_overall_formula():
    truth = series([[0, 0], [2, 4]])
    score = evaluate_velocity(truth, predict_zero(truth.index))
    np.testing.assert_allclose(score["rmse_mm_s"], [np.sqrt(2), np.sqrt(8)])
    assert score["overall_rmse_mm_s"] == pytest.approx(np.sqrt(5))
    assert score["overall_rmse_mm_s"] != pytest.approx(np.mean(score["rmse_mm_s"]))
    assert score["r2"] == [-1.0, -1.0]
    assert score["mean_r2"] == -1
    assert score["scored_examples"] == 2


def test_perfect_prediction_zero_rmse_unit_r2_nonconstant_axes():
    truth = series([[0, 1], [2, -1], [4, 5]])
    score = evaluate_velocity(
        truth, VelocitySeries(truth.values_mm_s.copy(), truth.index)
    )
    assert score["rmse_mm_s"] == [0, 0]
    assert score["overall_rmse_mm_s"] == 0
    assert score["r2"] == [1, 1]
    assert score["mean_r2"] == 1


@pytest.mark.parametrize("perfect", [True, False])
def test_constant_axis_explicitly_undefined_even_if_perfect(perfect):
    truth = series([[7, 0], [7, 2], [7, 4]])
    predictions = (
        VelocitySeries(truth.values_mm_s.copy(), truth.index)
        if perfect
        else predict_zero(truth.index)
    )
    score = evaluate_velocity(truth, predictions)
    assert score["r2"][0] is None
    assert score["r2_status"][0] == "constant_target"
    assert score["mean_r2"] is None
    assert score["r2"][1] is not None


def test_single_scored_example_r2_undefined():
    truth = series([[1, 2]])
    score = evaluate_velocity(truth, predict_zero(truth.index))
    assert score["r2"] == [None, None]
    assert score["r2_status"] == ["insufficient_scored_examples"] * 2


def test_frozen_training_mean_and_zero_ignore_changed_validation_labels():
    training = windows()
    model = TrainingMeanVelocity.fit(training)
    np.testing.assert_array_equal(model.mean_mm_s, [2, 4])
    validation = windows("validation")
    a = model.predict(index_from_windows(validation))
    changed = replace(validation, y=validation.y + 100000)
    b = model.predict(index_from_windows(changed))
    np.testing.assert_array_equal(a.values_mm_s, [[2, 4]] * 3)
    np.testing.assert_array_equal(a.values_mm_s, b.values_mm_s)
    np.testing.assert_array_equal(predict_zero(a.index).values_mm_s, np.zeros((3, 2)))
    np.testing.assert_array_equal(model.mean_mm_s, [2, 4])
    with pytest.raises(ValueError):
        model.mean_mm_s[0] = 5
    with pytest.raises(ValueError, match="training only"):
        TrainingMeanVelocity.fit(validation)


def test_train_mean_rejects_empty_and_nonfinite_training_targets():
    training = windows()
    with pytest.raises(ValueError, match="finite"):
        TrainingMeanVelocity.fit(replace(training, y=np.full((3, 2), np.nan)))
    empty = replace(
        training,
        X_raw=training.X_raw[:0],
        y=training.y[:0],
        source_bin_end_indices=training.source_bin_end_indices[:0],
        source_bin_end_s=training.source_bin_end_s[:0],
        target_time_s=training.target_time_s[:0],
        target_bin_end_indices=training.target_bin_end_indices[:0],
        segment_ids=training.segment_ids[:0],
    )
    with pytest.raises(ValueError, match="No eligible"):
        TrainingMeanVelocity.fit(empty)


@pytest.mark.parametrize(
    "case",
    [
        "shape",
        "time",
        "order",
        "bin_id",
        "segment",
        "mask",
        "partition",
        "identity",
        "channel_order",
    ],
)
def test_shape_timestamp_identity_and_mask_mismatch_fail(case):
    truth = targets_from_windows(windows("validation"))
    prediction = predict_zero(truth.index)
    index = prediction.index
    if case == "shape":
        prediction = replace(prediction, values_mm_s=prediction.values_mm_s[:-1])
    elif case == "time":
        times = index.time_s.copy()
        times[1] = np.nextafter(times[1], np.inf)
        index = replace(index, time_s=times)
    elif case == "order":
        index = replace(index, time_s=index.time_s[::-1])
    elif case == "bin_id":
        index = replace(index, bin_ids=index.bin_ids + 1)
    elif case == "segment":
        index = replace(index, segment_ids=index.segment_ids + 1)
    elif case == "mask":
        index = replace(index, score_mask=np.array([True, False, True]))
    elif case == "partition":
        index = replace(index, partition="train")
    elif case == "identity":
        index = replace(index, identity=(("source_sha256", "different"),))
    else:
        index = replace(index, channel_ids=index.channel_ids[::-1])
    prediction = replace(prediction, index=index)
    with pytest.raises(ValueError):
        evaluate_velocity(truth, prediction)


@pytest.mark.parametrize("who", ["target", "prediction"])
def test_nonfinite_inputs_fail_instead_of_silent_masking(who):
    truth = series([[0, 0], [2, 4]])
    prediction = predict_zero(truth.index)
    values = np.array([[np.nan, 0], [0, 0]])
    if who == "target":
        truth = replace(truth, values_mm_s=values)
    else:
        prediction = replace(prediction, values_mm_s=values)
    with pytest.raises(ValueError, match="finite"):
        evaluate_velocity(truth, prediction)


def test_shared_mask_scores_same_rows_without_automatic_finite_filtering():
    truth = series([[0, 0], [2, 4], [1000, 1000]])
    index = replace(truth.index, score_mask=np.array([True, True, False]))
    truth = replace(truth, index=index)
    score = evaluate_velocity(truth, predict_zero(index))
    assert score["scored_examples"] == 2
    assert score["overall_rmse_mm_s"] == pytest.approx(np.sqrt(5))
    bad = truth.values_mm_s.copy()
    bad[-1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        evaluate_velocity(replace(truth, values_mm_s=bad), predict_zero(index))
    index = replace(index, score_mask=np.zeros(3, dtype=bool))
    with pytest.raises(ValueError, match="No scored"):
        evaluate_velocity(replace(truth, index=index), predict_zero(index))


def test_training_mean_rejects_different_preparation_identity():
    model = TrainingMeanVelocity.fit(windows())
    index = index_from_windows(windows("validation"))
    with pytest.raises(ValueError, match="identity mismatch"):
        model.predict(replace(index, identity=(("source_sha256", "different"),)))


def test_prepared_target_must_match_last_source_bin():
    validation = windows("validation")
    with pytest.raises(ValueError, match="final source"):
        index_from_windows(
            replace(
                validation, target_bin_end_indices=validation.target_bin_end_indices + 1
            )
        )


def test_plot_rule_fixed_by_timestamps_not_labels():
    truth = series(np.column_stack((np.arange(20), -np.arange(20))))
    selected = select_validation_interval(truth.index, duration_s=0.1)
    assert selected == slice(0, 5)
    changed = replace(truth, values_mm_s=truth.values_mm_s[::-1])
    assert select_validation_interval(changed.index, duration_s=0.1) == selected


@pytest.mark.parametrize("case", ["gap", "segment", "mask", "coverage", "partition"])
def test_plot_rejects_first_interval_problems_without_searching(case):
    truth = series(np.zeros((20, 2)))
    index = truth.index
    if case == "gap":
        ids = index.bin_ids.copy()
        ids[3:] += 1
        index = replace(index, bin_ids=ids)
    elif case == "segment":
        ids = index.segment_ids.copy()
        ids[3:] = 1
        index = replace(index, segment_ids=ids)
    elif case == "mask":
        mask = index.score_mask.copy()
        mask[3] = False
        index = replace(index, score_mask=mask)
    elif case == "partition":
        index = replace(
            index, partition="test"
        )  # Synthetic metadata, never real test data.
    else:
        index = replace(
            index,
            time_s=index.time_s[:3],
            bin_ids=index.bin_ids[:3],
            segment_ids=index.segment_ids[:3],
            score_mask=index.score_mask[:3],
        )
    with pytest.raises(ValueError):
        select_validation_interval(index, duration_s=0.1)


def test_metric_overflow_fails_without_emitting_infinity():
    truth = series([[1e308, 0], [-1e308, 2]])
    with pytest.raises(ValueError, match="overflow"):
        evaluate_velocity(truth, predict_zero(truth.index))
