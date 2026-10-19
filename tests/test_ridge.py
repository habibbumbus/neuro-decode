"""Synthetic linear recovery, feature order, frozen transforms and safe reload."""

import json
from dataclasses import asdict, replace

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from neurodecode.data.features import (
    FrozenCountScaler,
    fit_count_scaler,
    prepare_partition,
)
from neurodecode.data.loaders import Recording
from neurodecode.evaluation import targets_from_windows
from neurodecode.models.ridge import (
    FEATURE_ORDER,
    RidgeDecoder,
    ridge_inputs,
    scaler_sha256,
    select_ridge_alpha,
)
from neurodecode.splits import TimingConfig


def inputs():
    rng = np.random.default_rng(42)
    config = TimingConfig(history_bins=2)
    times = 1 + np.arange(1001) * 0.004
    recording = Recording(
        times,
        np.column_stack((125 * times, -60 * times)),
        ("M1 001", "M1 002"),
        tuple((np.sort(rng.uniform(1, 5, 900)),) for _ in range(2)),
        {"spike_range_s": [1.0, 5.0]},
    )
    spec = {
        "schema_version": 1,
        "overlap": {"start_s": 1.0, "end_s": 5.0},
        "source": {"sha256": "synthetic"},
        "configuration": {"sha256": "synthetic"},
        "timing": asdict(config),
        "bin_grid": {"origin_s": 1.0, "width_s": 0.02},
        "partitions": [
            {
                "name": name,
                "retained": {"start_s": start, "end_s": end},
                "continuous_segments": [{"start_s": start, "end_s": end}],
            }
            for name, start, end in (("train", 1.0, 3.0), ("validation", 3.2, 5.0))
        ],
    }
    aligned, training = prepare_partition(recording, spec, config, "train")
    _, validation = prepare_partition(recording, spec, config, "validation")
    scaler = fit_count_scaler(aligned, spec, config)
    weights = np.array([[4.0, -0.5], [-2.0, 2.0], [1.5, -3.0], [3.0, 1.0]])
    intercept = np.array([7.0, -4.0])
    # Labels here are visibly synthetic, constructed for known-answer regression.
    training = replace(
        training, y=ridge_inputs(training, scaler).values @ weights + intercept
    )
    validation = replace(
        validation, y=ridge_inputs(validation, scaler).values @ weights + intercept
    )
    return training, validation, scaler, weights, intercept, aligned, spec, config


def fit_small(alpha=1e-6):
    training, validation, scaler, weights, intercept, *_ = inputs()
    features = ridge_inputs(training, scaler)
    labels = targets_from_windows(training)
    return (
        RidgeDecoder.fit(features, labels, scaler, alpha),
        training,
        validation,
        scaler,
        weights,
        intercept,
    )


def test_recovers_known_two_output_mapping_and_matches_library():
    model, training, validation, scaler, weights, intercept = fit_small()
    np.testing.assert_allclose(model.weights, weights, atol=1e-6)
    np.testing.assert_allclose(model.intercept, intercept, atol=1e-6)
    predicted = model.predict(ridge_inputs(validation, scaler)).values_mm_s
    np.testing.assert_allclose(predicted, validation.y, atol=2e-6)
    library = Ridge(alpha=1e-6, solver="svd", fit_intercept=True).fit(
        ridge_inputs(training, scaler).values, training.y
    )
    np.testing.assert_allclose(
        predicted, library.predict(ridge_inputs(validation, scaler).values), atol=1e-12
    )


def test_actual_model_path_uses_oldest_bin_then_channel_order():
    training, _, scaler, *_ = inputs()
    features = ridge_inputs(training, scaler)
    raw = training.X_raw[0]
    expected = np.array(
        [
            (raw[0, 0] - scaler.means[0]) / scaler.scales[0],
            (raw[0, 1] - scaler.means[1]) / scaler.scales[1],
            (raw[1, 0] - scaler.means[0]) / scaler.scales[0],
            (raw[1, 1] - scaler.means[1]) / scaler.scales[1],
        ]
    )
    np.testing.assert_array_equal(features.values[0], expected)
    weights = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
    model = RidgeDecoder(weights, np.array([2.0, -3.0]), 1.0, scaler, 1, "synthetic")
    np.testing.assert_array_equal(
        model.predict(features).values_mm_s[0], [expected[0] + 2, expected[3] - 3]
    )


def test_intercept_is_unpenalized_for_constant_targets_at_large_alpha():
    training, _, scaler, *_ = inputs()
    labels = targets_from_windows(
        replace(training, y=np.tile([17.0, -8.0], (len(training.y), 1)))
    )
    model = RidgeDecoder.fit(ridge_inputs(training, scaler), labels, scaler, 1e12)
    np.testing.assert_allclose(model.weights, 0, atol=1e-12)
    np.testing.assert_allclose(model.intercept, [17, -8], atol=1e-12)


def test_roundtrip_preserves_predictions_and_complete_frozen_transform(tmp_path):
    model, _, validation, scaler, *_ = fit_small()
    path = tmp_path / "ridge.json"
    model.save(path)
    loaded = RidgeDecoder.load(path, expected_scaler=scaler)
    np.testing.assert_array_equal(
        model.predict(ridge_inputs(validation, scaler)).values_mm_s,
        loaded.predict(ridge_inputs(validation, loaded.scaler)).values_mm_s,
    )
    assert loaded.scaler.to_metadata() == scaler.to_metadata()
    assert loaded.scaler.means.flags.writeable is False
    assert loaded.weights.flags.writeable is False
    with pytest.raises(FileExistsError):
        model.save(path)


@pytest.mark.parametrize(
    "case", ["channel_order", "means", "scales", "history", "identity"]
)
def test_loaded_model_rejects_incompatible_expected_transform(tmp_path, case):
    model, _, _, scaler, *_ = fit_small()
    path = tmp_path / "ridge.json"
    model.save(path)
    if case == "channel_order":
        changed = replace(scaler, channel_ids=scaler.channel_ids[::-1])
    elif case == "means":
        changed = replace(scaler, means=scaler.means + 1)
    elif case == "scales":
        changed = replace(scaler, scales=scaler.scales * 2)
    elif case == "history":
        changed = replace(scaler, history_bins=scaler.history_bins + 1)
    else:
        changed = replace(scaler, identity=(("source_sha256", "other"),))
    with pytest.raises(ValueError, match="Incompatible"):
        RidgeDecoder.load(path, expected_scaler=changed)


@pytest.mark.parametrize(
    "case", ["channel_order", "transform", "order", "shape", "nonfinite", "identity"]
)
def test_inference_rejects_incompatible_features(case):
    model, _, validation, scaler, *_ = fit_small()
    features = ridge_inputs(validation, scaler)
    if case == "channel_order":
        features = replace(
            features,
            index=replace(features.index, channel_ids=features.index.channel_ids[::-1]),
        )
    elif case == "transform":
        features = replace(features, normalization_sha256="wrong")
    elif case == "order":
        features = replace(features, feature_order="newest first")
    elif case == "shape":
        features = replace(features, values=features.values[:, :-1])
    elif case == "nonfinite":
        values = features.values.copy()
        values[0, 0] = np.nan
        features = replace(features, values=values)
    else:
        features = replace(
            features,
            index=replace(features.index, identity=(("source_sha256", "other"),)),
        )
    with pytest.raises(ValueError):
        model.predict(features)


def test_validation_labels_cannot_enter_candidate_fit_or_normalization():
    training, validation, scaler, _, _, aligned, spec, config = inputs()
    train_features, labels = (
        ridge_inputs(training, scaler),
        targets_from_windows(training),
    )
    first = RidgeDecoder.fit(train_features, labels, scaler, 10)
    changed_validation = replace(validation, y=np.full_like(validation.y, np.nan))
    # Feature construction and inference cannot read validation targets, even NaNs.
    features = ridge_inputs(changed_validation, scaler)
    assert np.isfinite(first.predict(features).values_mm_s).all()
    second = RidgeDecoder.fit(train_features, labels, scaler, 10)
    np.testing.assert_array_equal(first.weights, second.weights)
    np.testing.assert_array_equal(first.intercept, second.intercept)
    repeated_scaler = fit_count_scaler(aligned, spec, config)
    assert scaler_sha256(repeated_scaler) == scaler_sha256(scaler)
    with pytest.raises(ValueError, match="training only"):
        RidgeDecoder.fit(
            ridge_inputs(validation, scaler),
            targets_from_windows(validation),
            scaler,
            10,
        )


def test_larger_regularization_shrinks_weight_norm():
    a, training, _, scaler, *_ = fit_small(alpha=0.1)
    b = RidgeDecoder.fit(
        ridge_inputs(training, scaler), targets_from_windows(training), scaler, 1e5
    )
    assert np.linalg.norm(b.weights) < np.linalg.norm(a.weights)


def test_predeclared_selection_uses_overall_rmse_and_largest_alpha_tie():
    candidates = [
        {"alpha": 0.1, "metrics": {"overall_rmse_mm_s": 5.0}},
        {"alpha": 1.0, "metrics": {"overall_rmse_mm_s": 4.0}},
        {"alpha": 10.0, "metrics": {"overall_rmse_mm_s": 4.0 + 5e-10}},
        {"alpha": 100.0, "metrics": {"overall_rmse_mm_s": 4.0 + 2e-9}},
    ]
    assert select_ridge_alpha(candidates, 1e-9) == 10
    assert select_ridge_alpha(candidates, 0) == 1


@pytest.mark.parametrize("alphas", [[], [0], [-1], [1, 1], [float("nan")]])
def test_invalid_selection_grid_fails(alphas):
    candidates = [{"alpha": a, "metrics": {"overall_rmse_mm_s": 1.0}} for a in alphas]
    with pytest.raises(ValueError):
        select_ridge_alpha(candidates)


@pytest.mark.parametrize("case", ["order", "normalization", "weights"])
def test_corrupted_artifact_fails_clearly(tmp_path, case):
    model, *_ = fit_small()
    path = tmp_path / "ridge.json"
    model.save(path)
    metadata = json.loads(path.read_text())
    if case == "order":
        metadata["feature_order"] = "wrong"
    elif case == "normalization":
        metadata["scaler"]["means"][0] += 1
    else:
        metadata["weights"] = [[1, 2]]
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        RidgeDecoder.load(path)


@pytest.mark.parametrize("case", ["scale", "constant_flag", "policy"])
def test_saved_normalizer_rejects_invalid_values_and_changed_policy(case):
    _, _, scaler, *_ = inputs()
    metadata = scaler.to_metadata()
    if case == "scale":
        metadata["scales"][0] = 0
    elif case == "constant_flag":
        metadata["constant_channels"][0] = True
    else:
        metadata["fit_basis"] = "validation fitted"
    with pytest.raises(ValueError):
        FrozenCountScaler.from_metadata(metadata)


def test_saved_feature_order_is_explicit(tmp_path):
    model, *_ = fit_small()
    path = tmp_path / "ridge.json"
    model.save(path)
    metadata = json.loads(path.read_text())
    assert metadata["feature_order"] == FEATURE_ORDER
    assert metadata["weights_layout"] == "[L*C,2]; columns vx,vy"
    assert metadata["intercept_penalized"] is False
