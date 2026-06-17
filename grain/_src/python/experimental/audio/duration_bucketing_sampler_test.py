# Copyright 2026 The Grain Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for DurationBucketingSampler."""

from absl.testing import absltest
from absl.testing import parameterized
from grain._src.core import sharding
from grain._src.python.experimental.audio import duration_bucketing_sampler
import numpy as np


DurationBucketingSampler = duration_bucketing_sampler.DurationBucketingSampler


def _make_durations(n: int, seed: int = 0) -> np.ndarray:
  rng = np.random.default_rng(seed)
  return rng.uniform(1.0, 15.0, size=n).astype(np.float32)


class DurationBucketingSamplerTest(parameterized.TestCase):

  def test_rejects_empty_durations(self):
    with self.assertRaises(ValueError):
      DurationBucketingSampler(durations=[])

  def test_rejects_shuffle_without_seed(self):
    with self.assertRaises(ValueError):
      DurationBucketingSampler(durations=[1.0, 2.0], shuffle=True, seed=None)

  def test_no_sample_exceeds_max_duration_in_a_batch(self):
    durations = _make_durations(300, seed=1)
    sampler = DurationBucketingSampler(
        durations=durations,
        max_duration=20.0,
        bucket_duration_bins=(2, 4, 6, 8, 10, 12),
        shuffle=True,
        seed=42,
        num_epochs=1,
    )
    boundaries = sampler.batch_boundaries(epoch=0)
    # Reconstruct batches by walking RecordMetadata in order.
    pos = 0
    for size in boundaries:
      batch_keys = [sampler[pos + j].record_key for j in range(size)]
      total = float(durations[batch_keys].sum())
      self.assertLessEqual(
          total, 20.0,
          msg=f"batch {batch_keys} has total duration {total} > 20.0",
      )
      pos += size

  def test_full_epoch_covers_every_record_exactly_once(self):
    n = 500
    durations = _make_durations(n, seed=2)
    sampler = DurationBucketingSampler(
        durations=durations,
        max_duration=30.0,
        shuffle=True,
        seed=7,
        num_epochs=1,
    )
    seen = [sampler[i].record_key for i in range(len(sampler))]
    self.assertCountEqual(seen, range(n))

  def test_multi_epoch_each_record_seen_exactly_num_epochs_times(self):
    n = 100
    durations = _make_durations(n, seed=3)
    sampler = DurationBucketingSampler(
        durations=durations,
        max_duration=25.0,
        shuffle=True,
        seed=11,
        num_epochs=3,
    )
    counts = {}
    for i in range(len(sampler)):
      key = sampler[i].record_key
      counts[key] = counts.get(key, 0) + 1
    self.assertEqual(set(counts.keys()), set(range(n)))
    self.assertTrue(all(c == 3 for c in counts.values()))

  def test_index_monotonically_increasing_and_unique(self):
    durations = _make_durations(200, seed=4)
    sampler = DurationBucketingSampler(
        durations=durations, max_duration=30.0, shuffle=True, seed=5,
        num_epochs=2,
    )
    indices = [sampler[i].index for i in range(len(sampler))]
    self.assertEqual(indices, sorted(set(indices)))

  def test_deterministic_given_same_seed(self):
    durations = _make_durations(200, seed=5)
    s1 = DurationBucketingSampler(
        durations=durations, max_duration=30.0, shuffle=True, seed=99,
        num_epochs=1,
    )
    s2 = DurationBucketingSampler(
        durations=durations, max_duration=30.0, shuffle=True, seed=99,
        num_epochs=1,
    )
    keys1 = [s1[i].record_key for i in range(len(s1))]
    keys2 = [s2[i].record_key for i in range(len(s2))]
    self.assertEqual(keys1, keys2)

  def test_random_access_matches_sequential_access(self):
    """Critical for Grain's Sampler protocol: __getitem__ must be a pure
    function of `index`, independent of call order."""
    durations = _make_durations(300, seed=6)
    sampler = DurationBucketingSampler(
        durations=durations, max_duration=30.0, shuffle=True, seed=13,
        num_epochs=2,
    )
    sequential = [sampler[i].record_key for i in range(len(sampler))]

    # Re-create and query out of order / repeatedly.
    sampler2 = DurationBucketingSampler(
        durations=durations, max_duration=30.0, shuffle=True, seed=13,
        num_epochs=2,
    )
    out_of_order_indices = list(range(len(sampler2)))[::-1]
    for i in out_of_order_indices:
      self.assertEqual(sampler2[i].record_key, sequential[i])
    # Querying the same index twice must be stable.
    self.assertEqual(sampler2[10].record_key, sampler2[10].record_key)

  @parameterized.named_parameters(
      dict(testcase_name="two_shards", shard_count=2),
      dict(testcase_name="three_shards", shard_count=3),
      dict(testcase_name="four_shards", shard_count=4),
  )
  def test_shards_are_disjoint_and_cover_dataset(self, shard_count):
    n = 503  # deliberately not evenly divisible
    durations = _make_durations(n, seed=7)
    all_keys = set()
    per_shard_keys = []
    for shard_index in range(shard_count):
      sampler = DurationBucketingSampler(
          durations=durations,
          max_duration=30.0,
          shuffle=True,
          seed=21,
          shard_options=sharding.ShardOptions(
              shard_index=shard_index, shard_count=shard_count
          ),
          num_epochs=1,
      )
      keys = {sampler[i].record_key for i in range(len(sampler))}
      per_shard_keys.append(keys)
      all_keys |= keys

    for i in range(shard_count):
      for j in range(i + 1, shard_count):
        self.assertTrue(
            per_shard_keys[i].isdisjoint(per_shard_keys[j]),
            msg=f"shard {i} and shard {j} overlap",
        )
    # drop_remainder defaults to False, so the union must cover everything.
    self.assertEqual(all_keys, set(range(n)))

  def test_oversized_single_sample_emitted_alone_not_dropped(self):
    durations = np.array([1.0, 1.0, 50.0, 1.0], dtype=np.float32)
    sampler = DurationBucketingSampler(
        durations=durations,
        max_duration=10.0,  # smaller than the 50.0s sample
        bucket_duration_bins=(2,),
        shuffle=False,
        num_epochs=1,
    )
    keys = [sampler[i].record_key for i in range(len(sampler))]
    self.assertCountEqual(keys, [0, 1, 2, 3])

  def test_drop_remainder_drops_singleton_tail_batches(self):
    # 5 samples in one bucket, max_duration forces batches of size 2,
    # leaving a singleton tail batch that should be dropped.
    durations = np.array([5.0, 5.0, 5.0, 5.0, 5.0], dtype=np.float32)
    sampler = DurationBucketingSampler(
        durations=durations,
        max_duration=10.0,
        bucket_duration_bins=(),
        shuffle=False,
        drop_remainder=True,
        num_epochs=1,
    )
    keys = [sampler[i].record_key for i in range(len(sampler))]
    self.assertLen(keys, 4)  # the 5th (singleton tail) sample is dropped

  def test_checkpoint_resume_is_implicit_via_pure_index_function(self):
    """DurationBucketingSampler has no mutable get_state/set_state -- it is
    a pure function of `index`, so "resuming" is simply re-querying the
    index where a previous run stopped. This test simulates a preemption
    at an arbitrary index."""
    durations = _make_durations(400, seed=8)
    sampler = DurationBucketingSampler(
        durations=durations, max_duration=30.0, shuffle=True, seed=17,
        num_epochs=2,
    )
    full_run = [sampler[i].record_key for i in range(len(sampler))]

    # Simulate preemption after index 137, then "resume" with a fresh
    # sampler instance (e.g. after a process restart).
    resumed_sampler = DurationBucketingSampler(
        durations=durations, max_duration=30.0, shuffle=True, seed=17,
        num_epochs=2,
    )
    resumed_run = [
        resumed_sampler[i].record_key for i in range(137, len(sampler))
    ]
    self.assertEqual(resumed_run, full_run[137:])

  def test_index_out_of_bounds_raises(self):
    durations = _make_durations(10, seed=9)
    sampler = DurationBucketingSampler(
        durations=durations, max_duration=30.0, num_epochs=1,
    )
    with self.assertRaises(IndexError):
      _ = sampler[len(sampler)]
    with self.assertRaises(IndexError):
      _ = sampler[-1]

  def test_infinite_epochs_when_num_epochs_is_none(self):
    import sys
    durations = _make_durations(10, seed=10)
    sampler = DurationBucketingSampler(durations=durations, num_epochs=None)
    self.assertEqual(len(sampler), sys.maxsize)
    # Spot check a far-future index still resolves without error.
    far_index = sampler._records_per_epoch * 1000 + 1
    _ = sampler[far_index]


if __name__ == "__main__":
  absltest.main()
