"""
DurationBucketingSampler for Google Grain
==========================================
A drop-in replacement for grain.IndexSampler that groups variable-length
audio (or any duration-annotated) samples into batches with minimal padding.

Design goals:
  - Compatible with grain.DataLoader's Sampler protocol
  - Supports multi-host sharding (shard_index / shard_count)
  - Per-epoch re-shuffle (both within buckets and across buckets)
  - Grain-compatible checkpointing via get_state() / set_state()
  - Works with NeMo JSON manifest, Lhotse JSONL, or any dict list with
    a "duration" field

Usage:
    sampler = DurationBucketingSampler(
        durations=[e["duration"] for e in manifest_entries],
        max_duration=300.0,
        bucket_duration_bins=[2, 4, 6, 8, 10, 12],
        shuffle=True,
        seed=42,
        shard_index=0,
        shard_count=1,
        num_epochs=None,   # None = infinite
    )
    loader = grain.DataLoader(
        data_source=my_source,
        sampler=sampler,
        operations=[CollateAudioBatch()],
        worker_count=8,
    )
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
from typing import Iterator, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RecordMetadata (mirrors grain.python.record.RecordMetadata)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class RecordMetadata:
    """Metadata emitted per-record by the sampler.

    index      : monotonically increasing global counter (used for checkpointing)
    record_key : index into the DataSource (__getitem__ key)
    rng        : per-record numpy RNG for stateless augmentation
    """
    index: int
    record_key: Optional[int] = None
    rng: Optional[np.random.Generator] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rng(seed: int, index: int) -> np.random.Generator:
    """Deterministic per-record RNG derived from global seed + record index."""
    return np.random.default_rng([seed, index])


def _build_buckets(
    durations: np.ndarray,
    bucket_duration_bins: Sequence[float],
) -> list[np.ndarray]:
    """Assign each sample index to a bucket.

    bucket_duration_bins = [2, 4, 6] produces 4 buckets:
        bucket 0 : duration < 2
        bucket 1 : 2 <= duration < 4
        bucket 2 : 4 <= duration < 6
        bucket 3 : duration >= 6

    Returns:
        List of numpy arrays, each containing sample indices for that bucket.
    """
    bins = list(bucket_duration_bins)
    num_buckets = len(bins) + 1
    bucket_ids = np.digitize(durations, bins)  # shape: (N,)
    buckets = [
        np.where(bucket_ids == b)[0] for b in range(num_buckets)
    ]
    # Log bucket statistics for diagnostics
    for i, b in enumerate(buckets):
        lo = bins[i - 1] if i > 0 else 0.0
        hi = bins[i] if i < len(bins) else float("inf")
        logger.info(
            "Bucket %d  [%.1fs, %.1fs)  →  %d samples  (%.1f hrs)",
            i, lo, hi, len(b),
            durations[b].sum() / 3600.0 if len(b) > 0 else 0.0,
        )
    return buckets


def _shard_indices(
    indices: np.ndarray,
    shard_index: int,
    shard_count: int,
    drop_remainder: bool,
) -> np.ndarray:
    """Slice indices for this host's shard."""
    if shard_count == 1:
        return indices
    total = len(indices)
    per_shard = total // shard_count
    if not drop_remainder:
        per_shard = math.ceil(total / shard_count)
    start = shard_index * per_shard
    end = min(start + per_shard, total)
    return indices[start:end]


def _pack_into_batches(
    indices: np.ndarray,
    durations: np.ndarray,
    max_duration: float,
    drop_remainder: bool,
) -> list[list[int]]:
    """Greedily pack indices into batches respecting max_duration."""
    batches: list[list[int]] = []
    batch: list[int] = []
    batch_dur = 0.0

    for idx in indices:
        dur = float(durations[idx])
        if dur > max_duration:
            # Single sample exceeds limit — emit alone with a warning
            logger.warning(
                "Sample %d has duration %.2fs > max_duration %.2fs; "
                "emitting as single-sample batch.",
                idx, dur, max_duration,
            )
            if batch:
                batches.append(batch)
                batch, batch_dur = [], 0.0
            batches.append([int(idx)])
            continue

        if batch and batch_dur + dur > max_duration:
            batches.append(batch)
            batch, batch_dur = [], 0.0

        batch.append(int(idx))
        batch_dur += dur

    if batch:
        if drop_remainder and len(batch) < 2:
            pass  # drop partial tail batch
        else:
            batches.append(batch)

    return batches


# ---------------------------------------------------------------------------
# Main sampler
# ---------------------------------------------------------------------------

class DurationBucketingSampler:
    """Grain-compatible sampler that groups samples by audio duration.

    Instead of emitting one RecordMetadata per sample (like IndexSampler),
    this sampler first constructs variable-size batches where all samples
    within a batch have similar duration, then emits them sequentially.

    The DataSource.__getitem__ will be called with individual record_keys,
    and a CollateAudioBatch operation must be used to combine them into
    padded tensors.

    Args:
        durations:
            Sequence of per-sample durations (seconds). Must be aligned
            with DataSource indices (i.e., durations[i] corresponds to
            DataSource[i]).
        max_duration:
            Maximum total audio duration (seconds) per batch.
            Example: 300.0 ≈ 50 samples × 6 sec each.
        bucket_duration_bins:
            Bin edges for bucketing. E.g., [2, 4, 6, 8, 10, 12] creates
            7 buckets. Samples in the same bucket have similar duration,
            minimizing padding within each batch.
        shuffle:
            Whether to shuffle within each bucket and across buckets
            each epoch.
        seed:
            Base random seed. Per-epoch seed = seed + epoch_index.
        shard_index:
            This host's rank (0-indexed). Used for multi-host training.
        shard_count:
            Total number of hosts / shards.
        drop_remainder:
            If True, drop the last (possibly smaller) batch per bucket.
        num_epochs:
            Number of epochs to iterate. None = infinite.
    """

    def __init__(
        self,
        durations: Sequence[float],
        max_duration: float = 300.0,
        bucket_duration_bins: Sequence[float] = (2, 4, 6, 8, 10, 12),
        shuffle: bool = True,
        seed: int = 0,
        shard_index: int = 0,
        shard_count: int = 1,
        drop_remainder: bool = False,
        num_epochs: Optional[int] = None,
    ):
        self._durations = np.array(durations, dtype=np.float32)
        self._max_duration = max_duration
        self._bins = list(bucket_duration_bins)
        self._shuffle = shuffle
        self._seed = seed
        self._shard_index = shard_index
        self._shard_count = shard_count
        self._drop_remainder = drop_remainder
        self._num_epochs = num_epochs

        # Build per-bucket index arrays (pre-computed, re-used every epoch)
        self._raw_buckets: list[np.ndarray] = _build_buckets(
            self._durations, self._bins
        )

        # State for checkpointing
        self._epoch: int = 0
        self._batch_offset: int = 0   # how many batches have been emitted this epoch

        # Pre-compute batches for epoch 0
        self._current_batches: list[list[int]] = self._build_epoch_batches(epoch=0)

        total_samples = sum(len(b) for b in self._current_batches)
        logger.info(
            "DurationBucketingSampler initialized: "
            "%d buckets, %d batches/epoch, %d samples/epoch, "
            "shard %d/%d",
            len(self._raw_buckets),
            len(self._current_batches),
            total_samples,
            shard_index,
            shard_count,
        )

    # ------------------------------------------------------------------
    # Epoch batch construction
    # ------------------------------------------------------------------

    def _build_epoch_batches(self, epoch: int) -> list[list[int]]:
        """Construct all batches for one epoch (deterministic given epoch)."""
        rng = np.random.default_rng(self._seed + epoch)
        all_batches: list[list[int]] = []

        for bucket_indices in self._raw_buckets:
            if len(bucket_indices) == 0:
                continue

            indices = bucket_indices.copy()

            # Shuffle within bucket
            if self._shuffle:
                rng.shuffle(indices)

            # Apply sharding after shuffle so each shard gets a unique subset
            indices = _shard_indices(
                indices,
                self._shard_index,
                self._shard_count,
                self._drop_remainder,
            )

            bucket_batches = _pack_into_batches(
                indices,
                self._durations,
                self._max_duration,
                self._drop_remainder,
            )
            all_batches.extend(bucket_batches)

        # Shuffle across buckets
        if self._shuffle:
            rng.shuffle(all_batches)

        return all_batches

    # ------------------------------------------------------------------
    # Sampler protocol (Iterator of RecordMetadata)
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[RecordMetadata]:
        return self._generate()

    def _generate(self) -> Iterator[RecordMetadata]:
        # global_index must account for records already emitted before this
        # generator started (i.e., records in batches 0..batch_offset-1 of the
        # current epoch, plus all records from previous epochs).
        epoch = self._epoch
        batches_this_epoch = self._current_batches

        # Count how many records were emitted in previous epochs
        # (approximate: use current epoch batch count × avg batch size)
        records_before_this_epoch = sum(
            len(rec) for b_list in
            [self._build_epoch_batches(e) for e in range(epoch)]
            for rec in b_list
        ) if epoch > 0 else 0

        # Count records already emitted this epoch (before batch_offset)
        records_in_epoch_before_offset = sum(
            len(batches_this_epoch[b]) for b in range(self._batch_offset)
        )

        global_index = records_before_this_epoch + records_in_epoch_before_offset

        while self._num_epochs is None or epoch < self._num_epochs:
            batches = batches_this_epoch if epoch == self._epoch else self._build_epoch_batches(epoch)
            start_batch = self._batch_offset if epoch == self._epoch else 0

            for batch_idx in range(start_batch, len(batches)):
                batch = batches[batch_idx]
                self._batch_offset = batch_idx  # track for checkpointing
                for record_key in batch:
                    yield RecordMetadata(
                        index=global_index,
                        record_key=record_key,
                        rng=_make_rng(self._seed, global_index),
                    )
                    global_index += 1

            epoch += 1
            self._epoch = epoch
            self._batch_offset = 0
            self._current_batches = self._build_epoch_batches(epoch)
            batches_this_epoch = self._current_batches

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def get_state(self) -> dict:
        """Return serialisable state for checkpoint/resume.

        Saves batch_offset + 1 so that set_state() resumes from the batch
        immediately AFTER the last fully-emitted batch.
        """
        return {
            "epoch": self._epoch,
            "batch_offset": self._batch_offset + 1,
            "seed": self._seed,
            "shard_index": self._shard_index,
            "shard_count": self._shard_count,
        }

    def set_state(self, state: dict) -> None:
        """Restore from a previously saved state dict."""
        self._epoch = state["epoch"]
        self._batch_offset = state["batch_offset"]
        # Rebuild batches for the restored epoch
        self._current_batches = self._build_epoch_batches(self._epoch)
        logger.info(
            "DurationBucketingSampler restored: epoch=%d, batch_offset=%d",
            self._epoch, self._batch_offset,
        )

    # ------------------------------------------------------------------
    # Convenience: __repr__ for Grain checkpoint string-match check
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"DurationBucketingSampler("
            f"max_duration={self._max_duration}, "
            f"bins={self._bins}, "
            f"shuffle={self._shuffle}, "
            f"seed={self._seed}, "
            f"shard={self._shard_index}/{self._shard_count}, "
            f"num_epochs={self._num_epochs})"
        )


# ---------------------------------------------------------------------------
# Factory: build from NeMo manifest file
# ---------------------------------------------------------------------------

def from_nemo_manifest(
    manifest_path: str,
    max_duration: float = 300.0,
    bucket_duration_bins: Sequence[float] = (2, 4, 6, 8, 10, 12),
    shuffle: bool = True,
    seed: int = 0,
    shard_index: int = 0,
    shard_count: int = 1,
    drop_remainder: bool = False,
    num_epochs: Optional[int] = None,
    min_duration: float = 0.0,
    max_sample_duration: float = float("inf"),
) -> tuple["DurationBucketingSampler", list[dict]]:
    """Build sampler directly from a NeMo JSON manifest.

    Filters samples by [min_duration, max_sample_duration] before building.

    Returns:
        (sampler, filtered_entries)
        filtered_entries is the list of manifest dicts aligned with sampler
        indices — pass it to your DataSource.
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]

    # Filter by duration
    entries = [
        e for e in entries
        if min_duration <= float(e["duration"]) <= max_sample_duration
    ]
    logger.info(
        "Loaded %d entries from %s (after duration filter [%.1f, %.1f])",
        len(entries), manifest_path, min_duration, max_sample_duration,
    )

    durations = [float(e["duration"]) for e in entries]
    sampler = DurationBucketingSampler(
        durations=durations,
        max_duration=max_duration,
        bucket_duration_bins=bucket_duration_bins,
        shuffle=shuffle,
        seed=seed,
        shard_index=shard_index,
        shard_count=shard_count,
        drop_remainder=drop_remainder,
        num_epochs=num_epochs,
    )
    return sampler, entries
