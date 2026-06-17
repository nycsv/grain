"""
End-to-end example: DurationBucketingSampler + grain.DataLoader
================================================================
Shows a complete training loop with:
  - NeMo manifest as data source
  - Duration-based dynamic bucketing
  - Multi-worker prefetch
  - Checkpointing

Run:
    python example_asr_pipeline.py
"""

import logging

import numpy as np

# Assume grain is installed: pip install grain
try:
    import grain
    GRAIN_AVAILABLE = True
except ImportError:
    GRAIN_AVAILABLE = False
    print("grain not installed. Showing sampler output only.")

from duration_bucketing_sampler import DurationBucketingSampler, from_nemo_manifest
from asr_data_source import NeMoAudioSource, CollateAudioBatch

logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# Option A: Build from manifest file (recommended for production)
# ---------------------------------------------------------------------------

def example_from_manifest(manifest_path: str):
    sampler, entries = from_nemo_manifest(
        manifest_path=manifest_path,
        max_duration=300.0,        # ~50 × 6sec per batch
        bucket_duration_bins=[2, 4, 6, 8, 10, 12, 15, 20],
        shuffle=True,
        seed=42,
        shard_index=0,             # replace with jax.process_index()
        shard_count=1,             # replace with jax.process_count()
        drop_remainder=True,
        num_epochs=None,           # infinite — control via training steps
        min_duration=0.5,
        max_sample_duration=30.0,
    )

    source = NeMoAudioSource(entries, sample_rate=16_000)

    if not GRAIN_AVAILABLE:
        print("Sampler preview (first 5 records):")
        for i, r in enumerate(sampler):
            print(f"  index={r.index}  record_key={r.record_key}  "
                  f"duration={entries[r.record_key]['duration']:.2f}s")
            if i >= 4:
                break
        return

    loader = grain.DataLoader(
        data_source=source,
        sampler=sampler,
        operations=[CollateAudioBatch(pad_value=0.0)],
        worker_count=8,
        worker_buffer_size=2,
    )

    # Training loop
    for step, batch in enumerate(loader):
        audio   = batch["audio"]           # (B, T_max)  float32
        lengths = batch["audio_lengths"]   # (B,)        int32
        texts   = batch["texts"]           # list[str]

        pad_ratio = 1.0 - lengths.sum() / (audio.shape[0] * audio.shape[1])
        print(
            f"step={step:4d}  "
            f"batch_size={audio.shape[0]}  "
            f"T_max={audio.shape[1]}  "
            f"padding={pad_ratio:.1%}"
        )

        if step >= 9:
            # Checkpoint example
            state = loader.get_state() if hasattr(loader, "get_state") else sampler.get_state()
            print(f"\nCheckpoint state: {state}")
            break


# ---------------------------------------------------------------------------
# Option B: Build from duration array (when manifest already loaded)
# ---------------------------------------------------------------------------

def example_from_durations():
    """Minimal example without a real manifest file."""
    rng = np.random.default_rng(0)
    N = 10_000
    durations = rng.uniform(0.5, 20.0, size=N).tolist()

    sampler = DurationBucketingSampler(
        durations=durations,
        max_duration=60.0,
        bucket_duration_bins=[2, 4, 6, 8, 10, 12, 15],
        shuffle=True,
        seed=42,
        num_epochs=1,
    )

    # Measure padding efficiency
    dur_arr = np.array(durations)
    batch_durs = []
    batch_max_durs = []
    current_batch = []

    prev_batch_start_index = None
    for r in sampler:
        current_batch.append(r.record_key)
        if len(current_batch) >= 2:
            durs = dur_arr[current_batch]
            if durs.sum() >= 59.0:   # batch nearly full
                batch_durs.append(float(durs.sum()))
                batch_max_durs.append(float(durs.max() * len(current_batch)))
                current_batch = []

    if batch_durs:
        efficiency = np.array(batch_durs).sum() / np.array(batch_max_durs).sum()
        print(f"\nPacking efficiency estimate: {efficiency:.1%}")
        print(f"(100% = no padding; lower = more padding waste)")
        print(f"Total batches: {len(batch_durs)}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        example_from_manifest(sys.argv[1])
    else:
        print("Running synthetic example (no manifest file provided)...\n")
        example_from_durations()
