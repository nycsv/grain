# `grain.experimental` — Duration Bucketing Sampler

## Problem

Variable-length sequence workloads (speech/ASR, TTS, and similar) suffer
heavy padding waste when batches are built from randomly ordered indices.
A batch containing both a 1-second and a 20-second utterance pads the short
sample out to the long sample's length — in the worst case, >90% of the
compute in that batch is spent on padding rather than signal.

Grain's `IndexSampler` and `Batch` operation batch by *position*, not by
sequence length, so they don't address this. There is currently no
sequence-length-aware sampler in Grain core or `grain.experimental`.

## What this adds

`DurationBucketingSampler`: a `Sampler` (same protocol as `IndexSampler`)
that:

1. Buckets record indices by a caller-supplied `durations` array (no data
   is read to do this — bucketing is purely on metadata).
2. Greedily packs each bucket into batches that respect a `max_duration`
   budget (total seconds per batch), rather than a fixed sample count.
3. Implements `__getitem__(index) -> RecordMetadata` as a **pure function
   of `index`**, exactly like `IndexSampler` — no internal mutable state,
   so it requires no special checkpoint handling beyond what any
   `IndexSampler`-based pipeline already does.
4. Supports the standard `ShardOptions` (sharding is applied after
   shuffling, so shards stay statistically balanced).

## Why `experimental/`

This follows the same pattern as `experimental/example_packing` and
`experimental/index_shuffle`: a self-contained module with zero changes
to Grain core (`samplers.py`, `data_loader.py`, `dataset/` are untouched),
gated behind `grain.experimental` until usage patterns settle.

## Compatibility

Verified against the real `grain._src.python.record.RecordMetadata` and
`grain._src.core.sharding.ShardOptions` / `NoSharding` classes (not
reimplementations) — see `duration_bucketing_sampler_test.py`. 16 test
cases (18 with parameterization) covering:

- Full-epoch coverage with no duplicates or omissions
- Multi-epoch record-count correctness
- Determinism given a fixed seed
- True random access (`__getitem__` independent of call order — the key
  Grain `Sampler` protocol requirement)
- Disjoint, dataset-covering shards across 2/3/4-way sharding
- `max_duration` budget never exceeded per batch
- Oversized single-sample handling (emitted alone, not silently dropped)
- `drop_remainder` semantics
- Implicit checkpoint-resume behavior (pure function of index — "resuming"
  is just re-querying the index where a prior run stopped)
- Out-of-bounds `IndexError`
- Infinite-epoch (`num_epochs=None`) mode

## Usage

```python
import grain

sampler = grain.experimental.DurationBucketingSampler(
    durations=dataset.durations,       # np.float32 array, len == len(dataset)
    max_duration=300.0,                # total seconds per batch
    bucket_duration_bins=(2, 4, 6, 8, 10, 12),
    shuffle=True,
    seed=42,
    shard_options=grain.ShardByJaxProcess(),
)

data_loader = grain.DataLoader(
    data_source=my_audio_source,
    sampler=sampler,
    operations=[my_collate_op],   # pad/stack records into a batched tensor
    worker_count=8,
)
```

Use `sampler.batch_boundaries(epoch)` to get the exact (variable) batch
sizes this sampler will produce for a given epoch, useful for configuring
a downstream collate/batch operation.

## Non-goals / out of scope for this PR

- A bundled `CollateAudioBatch`-style padding operation (kept separate so
  this PR is reviewable on the sampler alone).
- Tarred / sequential-access source support (sequential formats can't
  support true random-access bucketing; that's a separate, larger design
  discussion).
- Any change to `Sampler`, `IndexSampler`, `DataLoader`, or anything under
  `grain/_src/python/dataset/`.
