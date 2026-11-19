"""Synthetic count validity, reproducibility, scope, missingness and causality."""

from dataclasses import replace

import numpy as np
import pytest

from neurodecode.robustness import Disruption, perturb_packet, ranked_channels
from neurodecode.streaming import SequentialDecoder
from neurodecode.training import cpu_training_runtime
from tests.test_streaming import fixture, replay


@pytest.mark.parametrize(
    "kind,severity", [("channel_loss", 0.5), ("thinning", 0.5), ("poisson", 1.0)]
)
def test_reproducible_count_valid_limited_channels_times_and_no_input_mutation(
    kind, severity
):
    model, _, _, _, _, packets = fixture("ridge")
    means = model.scaler.means
    condition = Disruption(
        "synthetic", kind, severity, 13, 15, (packets[0].channel_ids[0],), 20261119
    )
    before = [p.counts.copy() for p in packets]
    first = [perturb_packet(p, condition, means) for p in packets]
    second = [perturb_packet(p, condition, means) for p in reversed(packets)][::-1]
    for old, saved, a, b in zip(packets, before, first, second, strict=True):
        np.testing.assert_array_equal(old.counts, saved)
        np.testing.assert_array_equal(a.counts, b.counts)
        assert a.counts.dtype.kind in "iu" and (a.counts >= 0).all()
        assert a.missing is False and a.source_valid == old.source_valid
        assert a.bin_id == old.bin_id and a.identity == old.identity
        np.testing.assert_array_equal(a.counts[1:], old.counts[1:])
        if not 13 <= old.bin_id < 15:
            np.testing.assert_array_equal(a.counts, old.counts)
        elif kind == "channel_loss":
            assert a.counts[0] == 0
        elif kind == "thinning":
            assert a.counts[0] <= old.counts[0]
        else:
            assert a.counts[0] >= old.counts[0]


def test_hash_ranking_uses_identity_not_input_order_or_rate():
    channels = tuple(f"M1 {i:03d}" for i in range(1, 97))
    first = ranked_channels(channels, "synthetic-source", 20261118)
    assert first == ranked_channels(channels[::-1], "synthetic-source", 20261118)
    assert set(first) == set(channels)
    assert first != ranked_channels(channels, "another-source", 20261118)
    assert first != ranked_channels(channels, "synthetic-source", 20261119)
    assert set(first[:10]) < set(first[:24]) < set(first[:48])


@pytest.mark.parametrize("kind", ["thinning", "poisson"])
def test_random_draws_follow_physical_ids_not_array_positions(kind):
    model, _, _, _, _, packets = fixture("ridge")
    p = packets[3]
    condition = Disruption("random", kind, 0.5, 1, 100, seed=42)
    a = perturb_packet(p, condition, model.scaler.means)
    reordered = replace(
        p,
        counts=p.counts[::-1],
        channel_ids=p.channel_ids[::-1],
        unit_indices=p.unit_indices[::-1],
    )
    b = perturb_packet(reordered, condition, model.scaler.means[::-1])
    np.testing.assert_array_equal(a.counts, b.counts[::-1])


@pytest.mark.parametrize("kind,severity", [("thinning", 1.0), ("poisson", 0.0)])
def test_zero_effect_conditions_preserve_raw_counts(kind, severity):
    model, _, _, _, _, packets = fixture("ridge")
    c = Disruption("identity", kind, severity, 1, 100, seed=42)
    for packet in packets:
        np.testing.assert_array_equal(
            perturb_packet(packet, c, model.scaler.means).counts, packet.counts
        )


@pytest.mark.parametrize("kind", ["ridge", "gru", "kalman"])
def test_future_disruptions_cannot_change_earlier_model_predictions(kind):
    model, _, _, _, _, packets = fixture(kind)
    condition = Disruption(
        "future", "channel_loss", 1.0, 15, 17, packets[0].channel_ids
    )
    changed = [perturb_packet(p, condition, model.scaler.means) for p in packets]
    with cpu_training_runtime(7):
        first, second = replay(model, packets), replay(model, changed)
    np.testing.assert_array_equal(
        [r.velocity_mm_s for r in first[:2]], [r.velocity_mm_s for r in second[:2]]
    )
    assert not np.array_equal(
        [r.velocity_mm_s for r in first[2:]], [r.velocity_mm_s for r in second[2:]]
    )


@pytest.mark.parametrize("kind", ["ridge", "gru", "kalman"])
def test_explicit_missing_differs_from_zero_and_resets_history_and_state(kind):
    model, _, _, _, _, packets = fixture(kind)
    last = packets[-1]
    packets += [
        replace(
            last,
            source_row=last.source_row + i,
            bin_id=last.bin_id + i,
            start_s=(last.bin_id + i - 1) * 0.02,
            end_s=(last.bin_id + i) * 0.02,
        )
        for i in range(1, 5)
    ]
    missing = Disruption("missing", "missing", 1.0, 14, 15)
    zero = Disruption("zero", "channel_loss", 1.0, 14, 15, model.scaler.channel_ids)
    absent = [perturb_packet(p, missing, model.scaler.means) for p in packets]
    zeroed = [perturb_packet(p, zero, model.scaler.means) for p in packets]
    assert absent[3].missing and absent[3].counts is None and not absent[3].source_valid
    assert not zeroed[3].missing and zeroed[3].source_valid
    np.testing.assert_array_equal(zeroed[3].counts, np.zeros(2, dtype=np.int64))
    with cpu_training_runtime(7):
        a, z, fresh = (
            replay(model, absent),
            replay(model, zeroed),
            replay(model, packets[4:]),
        )
    assert [r.bin_id for r in a] == [13, 17, 18, 19, 20]
    assert [r.bin_id for r in z] == list(range(13, 21))
    assert a[1].reset
    np.testing.assert_array_equal(a[1].velocity_mm_s, fresh[0].velocity_mm_s)


def test_missing_flag_cannot_hide_a_count_vector_or_a_valid_source():
    model, _, _, _, _, packets = fixture("ridge")
    for invalid in (
        replace(packets[0], missing=True),
        replace(packets[0], missing=True, counts=None),
    ):
        stream = SequentialDecoder(model, 0.0, 0.02)
        with pytest.raises(ValueError, match="Missing packets"):
            stream.step(invalid)


def test_clean_condition_exactly_preserves_original_replay():
    model, _, _, _, _, packets = fixture("ridge")
    clean = Disruption("clean", "clean", 0.0, 1, 100)
    changed = [perturb_packet(p, clean, model.scaler.means) for p in packets]
    assert all(p is q for p, q in zip(packets, changed, strict=True))
    np.testing.assert_array_equal(
        [r.velocity_mm_s for r in replay(model, packets)],
        [r.velocity_mm_s for r in replay(model, changed)],
    )
