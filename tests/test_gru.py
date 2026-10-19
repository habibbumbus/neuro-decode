"""Synthetic bounded-window GRU mechanics; no recording training or scoring."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from neurodecode.data.features import FrozenCountScaler, HistoryWindows
from neurodecode.evaluation import targets_from_windows
from neurodecode.models.gru import GRUConfig, GRUDecoder, gru_inputs
from neurodecode.training import (
    GRUTrainingConfig,
    cpu_training_runtime,
    fit_gru,
    gru_training_step,
    make_adam,
)


def synthetic(n=12, channels=3, length=10):
    rng = np.random.default_rng(53)
    raw = rng.poisson(1.0, size=(n, length, channels)).astype(np.int64)
    y = np.column_stack(
        (
            0.4 * raw[:, -1, 0] + 0.2 * raw[:, 0, 1],
            -0.3 * raw[:, -1, 1] + 0.1 * raw[:, 0, 0],
        )
    )
    scaler = FrozenCountScaler(
        np.zeros(channels),
        np.ones(channels),
        np.zeros(channels, dtype=bool),
        tuple(f"M1 {i + 1:03d}" for i in range(channels)),
        ((0,),) * channels,
        length,
        (("source_sha256", "synthetic-only"),),
        n * length,
        "synthetic",
        "synthetic",
    )

    def windows(partition, offset):
        ids = offset + length + np.arange(n, dtype=np.int64) * (length + 1)
        source_ids = ids[:, None] - np.arange(length - 1, -1, -1)
        return HistoryWindows(
            raw.copy(),
            y.copy(),
            source_ids - 1,
            source_ids,
            (source_ids - 1) * 0.02,
            source_ids * 0.02,
            ids - 1,
            ids,
            ids * 0.02,
            np.arange(n, dtype=np.int64),
            scaler.channel_ids,
            scaler.unit_indices,
            partition,
            scaler.identity,
            {},
        )

    # Reindexed copies only for the tiny overfit wiring check, not generalization.
    return scaler, windows("train", 0), windows("validation", 10000)


def fit_fixture(config=None, channels=3):
    scaler, training, validation = synthetic(channels=channels)
    return (
        fit_gru(
            gru_inputs(training, scaler),
            targets_from_windows(training),
            gru_inputs(validation, scaler),
            targets_from_windows(validation),
            scaler,
            GRUConfig(8),
            config or GRUTrainingConfig(max_epochs=2),
        ),
        scaler,
        training,
        validation,
    )


def test_actual_architecture_shapes_and_parameter_count():
    scaler, training, _ = synthetic(channels=96)
    with cpu_training_runtime(1):
        model = GRUDecoder(scaler)
        x = torch.from_numpy(gru_inputs(training, scaler).values)
        seen = []
        hook = model.recurrent.register_forward_hook(
            lambda _, args, out: seen.append((args[1].clone(), out))
        )
        prediction = model(x)
        hook.remove()
    assert prediction.shape == (12, 2) and model.parameter_count == 12546
    assert seen[0][0].shape == (1, 12, 32) and torch.count_nonzero(seen[0][0]) == 0
    assert seen[0][1][0].shape == (12, 10, 32)
    assert seen[0][1][1].shape == (1, 12, 32)
    assert model.recurrent.num_layers == 1 and not model.recurrent.bidirectional


def test_hand_controlled_sequence_order_weights_recent_bins_differently():
    scaler, _, _ = synthetic(channels=2, length=2)
    model = GRUDecoder(scaler, GRUConfig(1))
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        model.recurrent.weight_ih_l0[2, 0] = 1.0
        model.head.weight[0, 0] = 1.0
    x = torch.tensor([[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 0.0]]])
    expected = torch.tanh(torch.tensor(1.0))
    prediction = model(x)
    torch.testing.assert_close(
        prediction[:, 0], torch.stack((expected / 4, expected / 2))
    )
    torch.testing.assert_close(prediction[:, 1], torch.zeros(2))


def test_windows_and_calls_do_not_share_hidden_state():
    scaler, training, _ = synthetic()
    with cpu_training_runtime(4):
        model = GRUDecoder(scaler)
        features = gru_inputs(training, scaler)
        before = model.predict(features).values_mm_s
        changed = features.values.copy()
        changed[0] += 1000
        after = model.predict(replace(features, values=changed)).values_mm_s
        np.testing.assert_array_equal(before[1:], after[1:])
        np.testing.assert_array_equal(before, model.predict(features).values_mm_s)
        reordered = (
            model(torch.from_numpy(features.values[::-1].copy())).detach().numpy()[::-1]
        )
        np.testing.assert_allclose(before, reordered, atol=1e-7)
        assert not model.training


def test_gradient_clipping_and_one_adam_step_update_finite_weights():
    scaler, training, _ = synthetic()
    with cpu_training_runtime(5):
        model = GRUDecoder(scaler)
        optimizer = make_adam(model, GRUTrainingConfig())
        before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        info = gru_training_step(
            model,
            optimizer,
            torch.from_numpy(gru_inputs(training, scaler).values),
            torch.from_numpy(training.y.astype(np.float32)) * 100,
            0.01,
        )
        norm = torch.sqrt(sum(torch.sum(p.grad**2) for p in model.parameters()))
        assert torch.isfinite(norm) and norm <= 0.010001
        assert info["gradient_norm_before_clip"] > 0.01
        assert model.training and all(
            torch.isfinite(p.grad).all() for p in model.parameters()
        )
        assert any(not torch.equal(before[k], v) for k, v in model.state_dict().items())


def test_tiny_synthetic_mapping_overfits_and_roundtrip_is_exact(tmp_path):
    result, scaler, training, validation = fit_fixture(
        GRUTrainingConfig(
            max_epochs=250, batch_size=12, learning_rate=0.02, patience=250
        )
    )
    prediction = result.model.predict(gru_inputs(training, scaler)).values_mm_s
    assert np.mean((prediction - training.y) ** 2) < 0.002
    path = tmp_path / "gru.pt"
    result.model.save(path, result.checkpoint_metadata())
    loaded = GRUDecoder.load(path, expected_scaler=scaler)
    expected = result.model.predict(gru_inputs(validation, scaler)).values_mm_s
    np.testing.assert_array_equal(
        expected, loaded.predict(gru_inputs(validation, scaler)).values_mm_s
    )
    assert not loaded.training
    payload = torch.load(path, weights_only=True)
    assert payload["metadata"]["training"]["configuration"]["seed"] == 2026
    assert "hidden_state" not in payload
    with pytest.raises(FileExistsError):
        result.model.save(path, {})


def test_seeded_training_repeat_and_caller_runtime_are_preserved():
    previous_threads = torch.get_num_threads()
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    random_state = torch.random.get_rng_state().clone()
    first, _, _, _ = fit_fixture()
    second, _, _, _ = fit_fixture()
    for name, value in first.model.state_dict().items():
        torch.testing.assert_close(
            value, second.model.state_dict()[name], rtol=0, atol=0
        )
    assert first.history == second.history
    assert torch.get_num_threads() == previous_threads
    assert torch.are_deterministic_algorithms_enabled() == previous_determinism
    assert torch.equal(torch.random.get_rng_state(), random_state)


def test_validation_labels_cannot_change_one_epoch_training_weights_or_transform():
    scaler, training, validation = synthetic()
    args = (
        gru_inputs(training, scaler),
        targets_from_windows(training),
        gru_inputs(validation, scaler),
    )
    config = GRUTrainingConfig(max_epochs=1)
    first = fit_gru(
        *args, targets_from_windows(validation), scaler, GRUConfig(8), config
    )
    changed = replace(validation, y=validation.y + 1000)
    second = fit_gru(*args, targets_from_windows(changed), scaler, GRUConfig(8), config)
    for name, value in first.model.state_dict().items():
        torch.testing.assert_close(
            value, second.model.state_dict()[name], rtol=0, atol=0
        )
    assert first.model.scaler.to_metadata() == second.model.scaler.to_metadata()
    assert (
        first.history[0]["validation_metrics"]
        != second.history[0]["validation_metrics"]
    )


def test_earliest_best_validation_snapshot_is_restored_and_patience_stops(monkeypatch):
    import neurodecode.training as training_module

    snapshots = []

    class TrackingModel(GRUDecoder):
        def predict(self, *args, **kwargs):
            snapshots.append({k: v.clone() for k, v in self.state_dict().items()})
            return super().predict(*args, **kwargs)

    original = training_module.evaluate_velocity
    scores = iter([3.0, 1.0, 1.0])

    def score(*args):
        metrics = original(*args)
        metrics["overall_rmse_mm_s"] = next(scores)
        return metrics

    monkeypatch.setattr(training_module, "GRUDecoder", TrackingModel)
    monkeypatch.setattr(training_module, "evaluate_velocity", score)
    result, _, _, _ = fit_fixture(GRUTrainingConfig(max_epochs=10, patience=1))
    assert result.best_epoch == 2 and len(result.history) == 3
    for name, value in result.model.state_dict().items():
        torch.testing.assert_close(value, snapshots[1][name], rtol=0, atol=0)


@pytest.mark.parametrize(
    "case", ["channel", "transform", "source", "shape", "nonfinite", "order_policy"]
)
def test_incompatible_inputs_fail(case):
    scaler, training, _ = synthetic()
    features = gru_inputs(training, scaler)
    if case == "channel":
        features = replace(
            features,
            index=replace(features.index, channel_ids=features.index.channel_ids[::-1]),
        )
    elif case == "transform":
        features = replace(features, normalization_sha256="wrong")
    elif case == "source":
        features = replace(
            features,
            index=replace(features.index, identity=(("source_sha256", "wrong"),)),
        )
    elif case == "shape":
        features = replace(features, values=features.values[:, :-1])
    elif case == "order_policy":
        features = replace(features, window_policy="newest first")
    else:
        values = features.values.copy()
        values[0, 0, 0] = np.nan
        features = replace(features, values=values)
    with pytest.raises(ValueError):
        GRUDecoder(scaler).predict(features)


@pytest.mark.parametrize(
    "case", ["channel", "transform", "reset", "dimensions", "nonfinite_weight"]
)
def test_checkpoint_mismatch_rejected(tmp_path, case):
    scaler, _, _ = synthetic()
    model = GRUDecoder(scaler)
    path = tmp_path / "gru.pt"
    model.save(path, {"synthetic": True})
    payload = torch.load(path, weights_only=True)
    m = payload["metadata"]
    if case == "channel":
        m["normalization"]["channel_ids"].reverse()
    elif case == "transform":
        m["normalization_sha256"] = "wrong"
    elif case == "reset":
        m["window_policy"] = "persistent"
    elif case == "dimensions":
        m["architecture"]["input_channels"] += 1
    else:
        payload["state_dict"]["head.bias"][0] = float("nan")
    torch.save(payload, path)
    with pytest.raises(ValueError):
        GRUDecoder.load(path, expected_scaler=scaler)


def test_factory_ignores_labels_but_rejects_discontinuous_source_history():
    scaler, training, _ = synthetic()
    first = gru_inputs(training, scaler)
    second = gru_inputs(replace(training, y=np.full_like(training.y, np.nan)), scaler)
    np.testing.assert_array_equal(first.values, second.values)
    ids = training.source_bin_end_indices.copy()
    ids[0, 0] -= 1
    with pytest.raises(ValueError, match="consecutive"):
        gru_inputs(replace(training, source_bin_end_indices=ids), scaler)


def test_mismatched_targets_and_partitions_fail_before_training():
    scaler, training, validation = synthetic()
    features, labels = gru_inputs(training, scaler), targets_from_windows(training)
    val = gru_inputs(validation, scaler)
    wrong = replace(
        labels, index=replace(labels.index, time_s=labels.index.time_s + 0.001)
    )
    with pytest.raises(ValueError, match="mismatch"):
        fit_gru(
            features,
            wrong,
            val,
            targets_from_windows(validation),
            scaler,
            GRUConfig(),
            GRUTrainingConfig(),
        )
    with pytest.raises(ValueError, match="train/validation"):
        fit_gru(
            val,
            targets_from_windows(validation),
            val,
            targets_from_windows(validation),
            scaler,
            GRUConfig(),
            GRUTrainingConfig(),
        )


@pytest.mark.parametrize(
    "change",
    [
        {"learning_rate": 0},
        {"gradient_clip_norm": 0},
        {"adam_beta1": 1.0},
        {"seed": -1},
        {"max_epochs": 0},
    ],
)
def test_impossible_training_configuration_fails(change):
    with pytest.raises(ValueError):
        replace(GRUTrainingConfig(), **change).validate()
