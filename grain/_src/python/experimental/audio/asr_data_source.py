"""
ASR DataSource and Collate Operation for use with DurationBucketingSampler
==========================================================================
Provides:
  - NeMoAudioSource   : RandomAccessDataSource reading audio from NeMo manifest
  - CollateAudioBatch : grain Operation that pads and stacks a list of samples
                        into a batch dict with tensors

Both are designed to work with grain.DataLoader.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import soundfile as sf


# ---------------------------------------------------------------------------
# DataSource
# ---------------------------------------------------------------------------

class NeMoAudioSource:
    """grain RandomAccessDataSource backed by a NeMo JSON manifest.

    Assumes entries is already filtered & aligned with sampler indices
    (use from_nemo_manifest() factory to get the aligned list).

    Each __getitem__ call returns:
        {
            "audio":    np.ndarray  float32, shape (T,)
            "duration": float
            "text":     str
            "idx":      int         (original manifest index)
        }
    """

    def __init__(self, entries: list[dict], sample_rate: int = 16_000):
        self._entries = entries
        self._sample_rate = sample_rate

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        entry = self._entries[idx]
        audio, sr = sf.read(
            entry["audio_filepath"],
            dtype="float32",
            always_2d=False,
        )
        # Resample if needed (simple check — use librosa/resampy for production)
        if sr != self._sample_rate:
            raise ValueError(
                f"Expected sample rate {self._sample_rate}, got {sr} "
                f"for file {entry['audio_filepath']}"
            )
        return {
            "audio":    audio,
            "duration": float(entry["duration"]),
            "text":     entry.get("text", entry.get("transcript", "")),
            "idx":      idx,
        }


# ---------------------------------------------------------------------------
# Collate Operation
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class CollateAudioBatch:
    """grain Operation that pads a list of individual sample dicts into
    a batch dict of numpy arrays.

    Input  : list of dicts  {"audio": (T_i,), "duration": float, "text": str}
    Output : dict {
        "audio":          np.ndarray  float32  (B, T_max)   zero-padded
        "audio_lengths":  np.ndarray  int32    (B,)          unpadded lengths
        "texts":          list[str]            (B,)
        "durations":      np.ndarray  float32  (B,)
    }
    """

    pad_value: float = 0.0

    def __call__(self, samples: list[dict]) -> dict[str, Any]:
        audios    = [s["audio"] for s in samples]
        lengths   = np.array([len(a) for a in audios], dtype=np.int32)
        durations = np.array([s["duration"] for s in samples], dtype=np.float32)
        texts     = [s["text"] for s in samples]

        # Zero-pad to longest in batch
        T_max = int(lengths.max())
        padded = np.full(
            (len(audios), T_max), fill_value=self.pad_value, dtype=np.float32
        )
        for i, a in enumerate(audios):
            padded[i, : len(a)] = a

        return {
            "audio":         padded,
            "audio_lengths": lengths,
            "texts":         texts,
            "durations":     durations,
        }
