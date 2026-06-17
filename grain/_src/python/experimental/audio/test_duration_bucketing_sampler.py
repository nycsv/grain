"""
Tests for DurationBucketingSampler
===================================
Run with:
    pytest test_duration_bucketing_sampler.py -v
"""

import json
import math
import os
import tempfile

import numpy as np
import pytest

from duration_bucketing_sampler import (
    DurationBucketingSampler,
    RecordMetadata,
    _build_buckets,
    _pack_into_batches,
    _shard_indices,
    from_nemo_manifest,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BINS = [2.0, 4.0, 6.0, 8.0, 10.0]

def make_durations(n: int = 100, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    return list(rng.uniform(0.5, 15.0, size=n).astype(float))


def make_sampler(**kwargs) -> DurationBucketingSampler:
    durations = make_durations()
    defaults = dict(
        durations=durations,
        max_duration=30.0,
        bucket_duration_bins=BINS,
        shuffle=True,
        seed=42,
        shard_index=0,
        shard_count=1,
        drop_remainder=False,
        num_epochs=2,
    )
    defaults.update(kwargs)
    return DurationBucketingSampler(**defaults)


# ---------------------------------------------------------------------------
# Unit: _build_buckets
# ---------------------------------------------------------------------------

class TestBuildBuckets:
    def test_all_samples_assigned(self):
        durations = np.array([1.0, 3.0, 5.0, 7.0, 11.0])
        buckets = _build_buckets(durations, [2.0, 4.0, 6.0, 8.0])
        total = sum(len(b) for b in buckets)
        assert total == len(durations)

    def test_correct_bucket_assignment(self):
        # bins=[4.0] → bucket 0: <4, bucket 1: >=4
        durations = np.array([1.0, 3.9, 4.0, 10.0])
        buckets = _build_buckets(durations, [4.0])
        assert set(buckets[0].tolist()) == {0, 1}   # <4
        assert set(buckets[1].tolist()) == {2, 3}   # >=4

    def test_empty_buckets_allowed(self):
        # All samples in one bucket
        durations = np.array([5.0, 6.0, 7.0])
        buckets = _build_buckets(durations, [2.0, 4.0, 100.0])
        assert len(buckets[2]) == 3  # bin [4, 100)
        assert len(buckets[0]) == 0
        assert len(buckets[1]) == 0


# ---------------------------------------------------------------------------
# Unit: _pack_into_batches
# ---------------------------------------------------------------------------

class TestPackIntoBatches:
    def test_respects_max_duration(self):
        durations = np.array([3.0, 3.0, 3.0, 3.0])
        indices = np.array([0, 1, 2, 3])
        batches = _pack_into_batches(indices, durations, max_duration=8.0, drop_remainder=False)
        for batch in batches:
            total = sum(float(durations[i]) for i in batch)
            assert total <= 8.0 + 1e-6

    def test_all_samples_included(self):
        durations = np.array([1.0] * 20)
        indices = np.arange(20)
        batches = _pack_into_batches(indices, durations, max_duration=5.0, drop_remainder=False)
        all_keys = [k for b in batches for k in b]
        assert sorted(all_keys) == list(range(20))

    def test_oversized_sample_emitted_alone(self):
        durations = np.array([100.0, 2.0])
        indices = np.array([0, 1])
        batches = _pack_into_batches(indices, durations, max_duration=10.0, drop_remainder=False)
        # oversized sample must be alone
        assert any(len(b) == 1 and b[0] == 0 for b in batches)

    def test_drop_remainder(self):
        durations = np.array([3.0] * 7)  # 7 samples, max=9 → 3 batches of 3, 3, 1
        indices = np.arange(7)
        batches_keep = _pack_into_batches(indices, durations, 9.0, drop_remainder=False)
        batches_drop = _pack_into_batches(indices, durations, 9.0, drop_remainder=True)
        assert len(batches_keep) > len(batches_drop)


# ---------------------------------------------------------------------------
# Unit: _shard_indices
# ---------------------------------------------------------------------------

class TestShardIndices:
    def test_single_shard_returns_all(self):
        indices = np.arange(100)
        result = _shard_indices(indices, 0, 1, False)
        assert len(result) == 100

    def test_two_shards_no_overlap(self):
        indices = np.arange(100)
        shard0 = set(_shard_indices(indices, 0, 2, True).tolist())
        shard1 = set(_shard_indices(indices, 1, 2, True).tolist())
        assert shard0.isdisjoint(shard1)

    def test_two_shards_union_is_all(self):
        indices = np.arange(100)
        shard0 = set(_shard_indices(indices, 0, 2, False).tolist())
        shard1 = set(_shard_indices(indices, 1, 2, False).tolist())
        assert shard0 | shard1 == set(range(100))


# ---------------------------------------------------------------------------
# Integration: DurationBucketingSampler
# ---------------------------------------------------------------------------

class TestDurationBucketingSampler:
    def test_emits_record_metadata(self):
        sampler = make_sampler(num_epochs=1)
        records = list(iter(sampler))
        assert all(isinstance(r, RecordMetadata) for r in records)

    def test_index_monotonically_increasing(self):
        sampler = make_sampler(num_epochs=1)
        indices = [r.index for r in sampler]
        assert indices == sorted(indices)

    def test_all_unique_indices(self):
        sampler = make_sampler(num_epochs=1)
        indices = [r.index for r in sampler]
        assert len(indices) == len(set(indices))

    def test_all_samples_covered_one_epoch(self):
        durations = make_durations(100)
        sampler = DurationBucketingSampler(
            durations=durations,
            max_duration=30.0,
            bucket_duration_bins=BINS,
            shuffle=False,
            seed=0,
            num_epochs=1,
        )
        record_keys = [r.record_key for r in sampler]
        assert sorted(record_keys) == list(range(100))

    def test_two_epochs_double_samples(self):
        durations = make_durations(50)
        sampler = DurationBucketingSampler(
            durations=durations,
            max_duration=20.0,
            bucket_duration_bins=BINS,
            shuffle=False,
            seed=0,
            num_epochs=2,
        )
        records = list(iter(sampler))
        # 2 epochs → each sample appears exactly twice
        from collections import Counter
        key_counts = Counter(r.record_key for r in records)
        assert all(v == 2 for v in key_counts.values())

    def test_shuffle_produces_different_order(self):
        durations = make_durations(200)
        s1 = DurationBucketingSampler(durations=durations, max_duration=30.0,
                                       bucket_duration_bins=BINS, shuffle=True,
                                       seed=1, num_epochs=1)
        s2 = DurationBucketingSampler(durations=durations, max_duration=30.0,
                                       bucket_duration_bins=BINS, shuffle=True,
                                       seed=2, num_epochs=1)
        keys1 = [r.record_key for r in s1]
        keys2 = [r.record_key for r in s2]
        assert keys1 != keys2, "Different seeds should produce different orders"

    def test_no_shuffle_is_deterministic(self):
        durations = make_durations(50)
        def get_keys(seed):
            s = DurationBucketingSampler(durations=durations, max_duration=20.0,
                                          bucket_duration_bins=BINS, shuffle=False,
                                          seed=seed, num_epochs=1)
            return [r.record_key for r in s]
        assert get_keys(0) == get_keys(99)  # seed doesn't matter without shuffle

    def test_per_record_rng_is_set(self):
        sampler = make_sampler(num_epochs=1)
        records = list(iter(sampler))
        assert all(r.rng is not None for r in records)

    def test_per_record_rng_reproducible(self):
        sampler1 = make_sampler(num_epochs=1)
        sampler2 = make_sampler(num_epochs=1)
        rngs1 = [r.rng.integers(0, 10000) for r in sampler1]
        rngs2 = [r.rng.integers(0, 10000) for r in sampler2]
        assert rngs1 == rngs2

    def test_batch_duration_never_exceeds_max(self):
        """Key correctness test: no batch should exceed max_duration."""
        durations = make_durations(200)
        max_dur = 25.0
        sampler = DurationBucketingSampler(
            durations=durations,
            max_duration=max_dur,
            bucket_duration_bins=BINS,
            shuffle=True,
            seed=42,
            num_epochs=1,
        )
        # Reconstruct batches from the emitted sequence
        dur_array = np.array(durations)
        current_batch_dur = 0.0
        prev_key = None
        batch_durs = []
        batch_dur = 0.0
        prev_idx = -1

        for r in sampler:
            # New batch starts when index jumps (rough heuristic since records
            # within the same batch have consecutive indices)
            if prev_idx != -1 and r.index != prev_idx + 1:
                batch_durs.append(batch_dur)
                batch_dur = 0.0
            batch_dur += float(dur_array[r.record_key])
            prev_idx = r.index
        if batch_dur > 0:
            batch_durs.append(batch_dur)

        # Allow small floating-point tolerance, and allow oversized singles
        oversized = [d for d in batch_durs if d > max_dur + 1e-3]
        assert len(oversized) == 0, f"Batches exceeded max_duration: {oversized}"


# ---------------------------------------------------------------------------
# Integration: Sharding
# ---------------------------------------------------------------------------

class TestSharding:
    def test_two_shards_no_duplicate_keys(self):
        durations = make_durations(200)
        def get_keys(shard_index):
            s = DurationBucketingSampler(
                durations=durations, max_duration=30.0,
                bucket_duration_bins=BINS, shuffle=False,
                seed=0, shard_index=shard_index, shard_count=2,
                num_epochs=1,
            )
            return set(r.record_key for r in s)

        keys0 = get_keys(0)
        keys1 = get_keys(1)
        assert keys0.isdisjoint(keys1), "Shards must not share keys"

    def test_two_shards_cover_all_samples(self):
        durations = make_durations(200)
        def get_keys(shard_index):
            s = DurationBucketingSampler(
                durations=durations, max_duration=30.0,
                bucket_duration_bins=BINS, shuffle=False, seed=0,
                shard_index=shard_index, shard_count=2,
                num_epochs=1, drop_remainder=False,
            )
            return set(r.record_key for r in s)

        keys0 = get_keys(0)
        keys1 = get_keys(1)
        assert keys0 | keys1 == set(range(200))


# ---------------------------------------------------------------------------
# Integration: Checkpointing
# ---------------------------------------------------------------------------

class TestCheckpointing:
    def test_get_set_state_roundtrip(self):
        durations = make_durations(100)
        sampler = DurationBucketingSampler(
            durations=durations, max_duration=20.0,
            bucket_duration_bins=BINS, shuffle=True,
            seed=7, num_epochs=3,
        )
        it = iter(sampler)
        # Consume 50 records
        first_50 = [next(it) for _ in range(50)]

        # Save state
        state = sampler.get_state()

        # Fresh sampler, restore state
        sampler2 = DurationBucketingSampler(
            durations=durations, max_duration=20.0,
            bucket_duration_bins=BINS, shuffle=True,
            seed=7, num_epochs=3,
        )
        sampler2.set_state(state)
        it2 = iter(sampler2)

        # Next records should match
        next_10_original = [next(it) for _ in range(10)]
        next_10_restored  = [next(it2) for _ in range(10)]
        assert [r.record_key for r in next_10_original] == \
               [r.record_key for r in next_10_restored]

    def test_repr_is_stable(self):
        s = make_sampler()
        assert "DurationBucketingSampler" in repr(s)


# ---------------------------------------------------------------------------
# Integration: from_nemo_manifest factory
# ---------------------------------------------------------------------------

class TestFromNemoManifest:
    def test_loads_and_filters(self, tmp_path):
        entries = [
            {"audio_filepath": f"/data/{i}.wav", "duration": float(i) * 0.5 + 1.0,
             "text": f"hello {i}"}
            for i in range(20)
        ]
        manifest_path = tmp_path / "train.json"
        with open(manifest_path, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

        sampler, filtered = from_nemo_manifest(
            str(manifest_path),
            max_duration=20.0,
            bucket_duration_bins=[2.0, 5.0, 8.0],
            shuffle=False,
            seed=0,
            min_duration=2.0,
            max_sample_duration=9.0,
            num_epochs=1,
        )
        # Check filtered entries are within bounds
        for e in filtered:
            assert 2.0 <= e["duration"] <= 9.0

        # Check sampler covers all filtered entries
        record_keys = [r.record_key for r in sampler]
        assert sorted(record_keys) == list(range(len(filtered)))
