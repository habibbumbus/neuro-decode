"""Predeclared, raw-count perturbations keyed by source/bin/channel identity."""

import hashlib
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from neurodecode.streaming import ReplayBin


@dataclass(frozen=True)
class Disruption:
    """One fixed condition; bin-end IDs in [first_bin_id, stop_bin_id)."""

    name: str
    kind: str
    severity: float
    first_bin_id: int
    stop_bin_id: int
    channel_ids: tuple[str, ...] = ()
    seed: int = 0

    def validate(self) -> None:
        if self.kind not in ("clean", "channel_loss", "thinning", "poisson", "missing"):
            raise ValueError("Unknown disruption kind")
        if not self.name or self.stop_bin_id <= self.first_bin_id:
            raise ValueError("Need name and increasing affected bin interval")
        if (
            type(self.seed) is not int
            or self.seed < 0
            or not np.isfinite(self.severity)
        ):
            raise ValueError("Seed/severity must be finite and valid")
        if self.kind == "thinning" and not 0 <= self.severity <= 1:
            raise ValueError("Thinning retention must be in [0,1]")
        if self.kind == "poisson" and self.severity < 0:
            raise ValueError("Poisson multiplier must be nonnegative")
        if len(set(self.channel_ids)) != len(self.channel_ids):
            raise ValueError("Repeated physical channel identity")


def ranked_channels(
    channel_ids: tuple[str, ...], source_sha256: str, seed: int
) -> tuple[str, ...]:
    """Hash rank independent of rates, learned weights and movement quality."""
    if len(set(channel_ids)) != len(channel_ids):
        raise ValueError("Physical channel IDs must be unique")
    return tuple(
        sorted(
            channel_ids,
            key=lambda c: hashlib.sha256(
                f"phase6b-channel-loss-v1|{seed}|{source_sha256}|{c}".encode()
            ).hexdigest(),
        )
    )


def _rng(packet: ReplayBin, channel_id: str, seed: int) -> np.random.Generator:
    source = dict(packet.identity)["source_sha256"]
    key = f"phase6b-count-v1|{seed}|{source}|{packet.partition}|{packet.session}|{packet.bin_id}|{channel_id}"
    entropy = int.from_bytes(hashlib.sha256(key.encode()).digest()[:16], "little")
    return np.random.Generator(np.random.PCG64(entropy))


def perturb_packet(
    packet: ReplayBin,
    condition: Disruption,
    training_means: NDArray[np.float64],
) -> ReplayBin:
    """Perturb one original bin once; all models/windows reuse that result.

    No label, weight, fitted transform update or future packet enters this call.
    Missing bins have no count vector. Zeroed physical channels remain valid.
    """
    condition.validate()
    if not set(condition.channel_ids).issubset(packet.channel_ids):
        raise ValueError("Disruption channel identity is absent from packet")
    if (
        condition.kind == "clean"
        or not condition.first_bin_id <= packet.bin_id < condition.stop_bin_id
    ):
        return packet
    if condition.kind == "missing":
        return replace(packet, counts=None, missing=True, source_valid=False)
    if packet.missing or not packet.source_valid:
        return packet
    if (
        not isinstance(packet.counts, np.ndarray)
        or packet.counts.shape != (len(packet.channel_ids),)
        or packet.counts.dtype.kind not in "iu"
        or np.any(packet.counts < 0)
    ):
        raise ValueError("Need nonnegative integer counts before perturbation")
    if (
        training_means.shape != packet.counts.shape
        or not np.isfinite(training_means).all()
        or np.any(training_means < 0)
    ):
        raise ValueError("Need nonnegative frozen per-channel training means")
    if np.any(packet.counts > np.iinfo(np.int64).max):
        raise ValueError("Raw count overflows int64")
    counts = packet.counts.astype(np.int64, copy=True)
    selected = condition.channel_ids or packet.channel_ids
    for channel, channel_id in enumerate(packet.channel_ids):
        if channel_id not in selected:
            continue
        if condition.kind == "channel_loss":
            counts[channel] = 0
        elif condition.kind == "thinning":
            counts[channel] = _rng(packet, channel_id, condition.seed).binomial(
                int(counts[channel]), condition.severity
            )
        else:
            added = int(
                _rng(packet, channel_id, condition.seed).poisson(
                    condition.severity * training_means[channel]
                )
            )
            value = int(counts[channel]) + added
            if value > np.iinfo(np.int64).max:
                raise ValueError("Perturbed count overflows int64")
            counts[channel] = value
    return replace(packet, counts=counts)


def planned_conditions(
    settings: dict, channel_ids: tuple[str, ...]
) -> list[Disruption]:
    """Exact finite round, resolved before any inference and saved in the manifest."""
    ranking = ranked_channels(
        channel_ids, settings["source_sha256"], settings["channel_seed"]
    )
    if list(ranking[:48]) != settings["channel_rank_prefix"]:
        raise ValueError(
            "Declared physical-channel ranking differs from source identity"
        )
    start, stop = settings["affected_first_bin_id"], settings["affected_stop_bin_id"]
    conditions = [Disruption("clean", "clean", 0.0, start, stop)]
    conditions += [
        Disruption(
            f"channel-loss-{n}",
            "channel_loss",
            n / len(channel_ids),
            start,
            stop,
            ranking[:n],
            settings["channel_seed"],
        )
        for n in settings["channel_counts"]
    ]
    conditions += [
        Disruption(
            f"thinning-{keep:g}",
            "thinning",
            keep,
            start,
            stop,
            (),
            settings["thinning_seed"],
        )
        for keep in settings["thinning_keep_probabilities"]
    ]
    conditions += [
        Disruption(
            f"poisson-{rate:g}",
            "poisson",
            rate,
            start,
            stop,
            (),
            settings["poisson_seed"],
        )
        for rate in settings["poisson_training_mean_multipliers"]
    ]
    conditions += [
        Disruption(
            f"missing-{length}",
            "missing",
            float(length),
            settings["missing_first_bin_id"],
            settings["missing_first_bin_id"] + length,
        )
        for length in settings["missing_lengths_bins"]
    ]
    if len(conditions) != settings["max_conditions"]:
        raise ValueError("Declared condition budget changed")
    for condition in conditions:
        condition.validate()
    return conditions
