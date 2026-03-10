# Grain Transformations Guide

This document explains every dataset transformation available in Grain, with usage examples.

Grain provides two dataset types:
- **`MapDataset`** — random-access (supports `ds[i]`), used with indexed sources
- **`IterDataset`** — streaming (supports `for x in ds`), used with sequential sources

Most transformations are available as chainable methods on both types.

---

## 1. `map`

Applies a function to each element individually.

```python
# Simple function
ds = grain.MapDataset.source([1, 2, 3]).map(lambda x: x * 2)
# [2, 4, 6]

# Using a grain transform class
class Normalize(grain.transforms.Map):
    def map(self, element):
        return element / 255.0

ds = grain.MapDataset.source([128, 255]).map(Normalize())
# [0.502, 1.0]
```

**Variants:**
- **`random_map(transform, seed)`** — the transform receives an additional `np.random.Generator` argument, seeded deterministically per element index.
- **`map_with_index(transform)`** — the transform receives `(index, element)`.

| Property       | Value                       |
|----------------|-----------------------------|
| Dataset types  | MapDataset, IterDataset     |
| Length change   | No (1:1)                   |
| Element change  | Yes (per transform)        |

---

## 2. `batch`

Groups consecutive elements into batches by stacking them along a new leading dimension.

```python
ds = grain.MapDataset.source([1, 2, 3, 4, 5]).batch(batch_size=2)
# [array([1, 2]), array([3, 4]), array([5])]

# Drop the incomplete last batch
ds = grain.MapDataset.source([1, 2, 3, 4, 5]).batch(batch_size=2, drop_remainder=True)
# [array([1, 2]), array([3, 4])]
```

**Parameters:**
- `batch_size` — number of elements per batch.
- `drop_remainder` (default `False`) — if `True`, discards the last batch when it has fewer than `batch_size` elements. If `False`, the last batch is smaller.
- `batch_fn` — custom batching function (defaults to `np.stack`).

| Property       | Value                                                    |
|----------------|----------------------------------------------------------|
| Dataset types  | MapDataset, IterDataset                                  |
| Length change   | Yes — `ceil(N / batch_size)` or `N // batch_size`       |
| Restriction     | Cannot directly follow `filter` or `flatmap` on MapDataset |

---

## 3. `filter`

Keeps only elements for which a predicate returns `True`.

```python
ds = grain.MapDataset.source([1, 2, 3, 4, 5]).filter(lambda x: x % 2 == 0)
# When iterated: [2, 4]

class KeepLong(grain.transforms.Filter):
    def filter(self, element):
        return len(element["text"]) > 10

ds = ds.filter(KeepLong())
```

**Behavior by dataset type:**
- **MapDataset** — returns `None` for filtered-out indices (sparse). Cannot be followed by `batch`.
- **IterDataset** — skips filtered elements entirely.

Grain includes a built-in threshold checker that warns (or raises) if too many elements are being filtered, helping catch logic errors early.

| Property       | Value                          |
|----------------|--------------------------------|
| Dataset types  | MapDataset, IterDataset        |
| Length change   | Unpredictable (sparse)        |
| Sparse          | Yes                           |

---

## 4. `shuffle`

Randomly reorders elements deterministically based on a seed.

```python
# Global shuffle (MapDataset)
ds = grain.MapDataset.source([0, 1, 2, 3, 4]).shuffle(seed=42)

# Window shuffle (IterDataset) — shuffles within fixed-size windows
ds = iter_ds.shuffle(window_size=100, seed=42)
```

**Parameters:**
- `seed` — integer seed (0 to 2^32 - 1). Required.
- `window_size` — (window shuffle only) size of the shuffle window.

**Behavior:**
- **MapDataset** — global shuffle using an index-shuffle algorithm. Different seed per epoch when combined with `repeat(reseed_each_epoch=True)`.
- **IterDataset (window shuffle)** — shuffles elements within a sliding window. Good for streaming data where global shuffle is not feasible.

| Property       | Value                       |
|----------------|-----------------------------|
| Dataset types  | MapDataset, IterDataset     |
| Length change   | No                         |
| Deterministic   | Yes (seed-based)           |

---

## 5. `mix`

Combines multiple datasets by sampling from each according to given proportions.

```python
ds_a = grain.MapDataset.source([1, 2, 3])
ds_b = grain.MapDataset.source([10, 20, 30])

# Equal mixing (default)
ds = grain.MapDataset.mix([ds_a, ds_b])

# Weighted mixing: 2:1 ratio
ds = grain.MapDataset.mix([ds_a, ds_b], proportions=[2, 1])
```

**Parameters:**
- `datasets` — sequence of datasets to mix.
- `proportions` — relative weights (defaults to uniform).

**Behavior:**
- Length is determined by the smallest proportionally-weighted dataset.
- IterDataset stops when any component dataset is exhausted.
- All component datasets should have the same element structure.

**`concatenate`** is a special case that chains datasets sequentially (no interleaving):
```python
ds = grain.MapDataset.mix([ds_a, ds_b], proportions=[1, 1])  # interleaved
# vs sequential concatenation via ConcatenateMapDataset
```

| Property       | Value                       |
|----------------|-----------------------------|
| Dataset types  | MapDataset, IterDataset     |
| Length change   | Depends on proportions      |
| Inputs          | Multiple datasets           |

---

## 6. `flatmap`

Maps each element to multiple elements (one-to-many expansion).

```python
class SplitWords(grain.experimental.FlatMapTransform):
    @property
    def max_fan_out(self):
        return 10  # maximum number of outputs per input

    def flat_map(self, element):
        return element.split()

ds = grain.MapDataset.source(["hello world", "foo"]).flat_map(SplitWords())
# ["hello", "world", "foo", None, ...]
```

**Parameters:**
- `transform` — a `transforms.FlatMap` with `flat_map(element)` method and `max_fan_out` property.

**Behavior:**
- Each input element produces up to `max_fan_out` output elements.
- MapDataset returns `None` for indices beyond actual outputs (sparse).
- IterDataset skips `None` entries.
- `max_fan_out` must be declared upfront; exceeding it raises `ValueError`.

| Property       | Value                                |
|----------------|--------------------------------------|
| Dataset types  | MapDataset, IterDataset              |
| Length change   | Multiplied by `max_fan_out`         |
| Sparse          | Yes (MapDataset)                    |

---

## 7. `cache`

Caches all elements in memory during the first iteration; subsequent iterations read from cache.

```python
ds = expensive_iter_dataset.cache()
```

**Behavior:**
- Elements are cached lazily as they are first consumed.
- After the first full iteration, re-iterating is instant (in-memory).
- Useful when upstream transforms are expensive and the dataset fits in memory.

| Property       | Value                       |
|----------------|-----------------------------|
| Dataset types  | IterDataset only            |
| Length change   | No                         |
| Memory          | Stores all elements in RAM |

---

## 8. `prefetch`

Prefetches elements ahead of time using a thread pool, overlapping data loading with model training.

```python
# Typically called via to_iter_dataset() which adds prefetching automatically
iter_ds = map_ds.to_iter_dataset(
    grain.ReadOptions(num_threads=4, prefetch_buffer_size=500)
)

# Or directly on an IterDataset
ds = iter_ds.prefetch(grain.experimental.ThreadPrefetchIterDataset, buffer_size=100)
```

**Parameters:**
- `num_threads` — number of worker threads.
- `prefetch_buffer_size` — how many elements to buffer ahead.

**Variants:**
- **`PrefetchIterDataset`** — thread-pool based, converts MapDataset to IterDataset.
- **`ThreadPrefetchIterDataset`** — queue-based, wraps an IterDataset.

| Property       | Value                       |
|----------------|-----------------------------|
| Dataset types  | IterDataset only            |
| Length change   | No                         |
| Purpose         | Performance (overlap I/O)  |

---

## 9. `packing`

Packs variable-length sequences into fixed-size bins to minimize padding waste. Critical for efficient training on variable-length data (e.g., NLP).

```python
ds = grain.experimental.FirstFitPackIterDataset(
    parent=iter_ds,
    length_struct={"input_ids": 512, "labels": 512},
    num_packing_bins=1000,
)

# Or best-fit for less wasted space
ds = grain.experimental.BestFitPackIterDataset(
    parent=iter_ds,
    length_struct={"input_ids": 512},
    num_packing_bins=1000,
)
```

**Parameters:**
- `length_struct` — target length per feature.
- `num_packing_bins` — number of bins to pack into.
- `seed` / `shuffle_bins` — optional randomization of output order.
- `meta_features` — features excluded from packing logic.
- `max_sequences_per_bin` — optional cap on sequences per bin.

**Algorithms:**
- **FirstFit** — places each sequence in the first bin that has room.
- **BestFit** — places each sequence in the tightest-fitting bin.

| Property       | Value                       |
|----------------|-----------------------------|
| Dataset types  | IterDataset only            |
| Length change   | Reduced (fewer, denser elements) |
| Use case        | NLP / variable-length sequences |

---

## 10. `interleave`

Interleaves elements from multiple datasets in round-robin fashion with concurrent iterator management.

```python
datasets = [dataset_a, dataset_b, dataset_c]
ds = grain.experimental.InterleaveIterDataset(
    datasets=datasets,
    cycle_length=2,  # process 2 datasets concurrently
)
```

**Parameters:**
- `datasets` — sequence of IterDataset or MapDataset objects.
- `cycle_length` — how many iterators to cycle through concurrently.
- `num_make_iter_threads` — threads for creating iterators asynchronously.
- `iter_buffer_size` — elements to prefetch from each iterator.

**Behavior:**
- Draws one element from each active iterator in rotation.
- When an iterator is exhausted, advances to the next dataset.
- Stops when all datasets have been consumed.

| Property       | Value                       |
|----------------|-----------------------------|
| Dataset types  | IterDataset only            |
| Length change   | Sum of all datasets         |
| Use case        | Combining many data files   |

---

## 11. `repeat`

Repeats the dataset for multiple epochs.

```python
# Repeat 3 times
ds = grain.MapDataset.source([1, 2, 3]).repeat(num_epochs=3)
# [1, 2, 3, 1, 2, 3, 1, 2, 3]

# Infinite repeat
ds = grain.MapDataset.source([1, 2, 3]).repeat(num_epochs=None)

# Reseed random transforms each epoch
ds = grain.MapDataset.source([1, 2, 3]).shuffle(seed=0).repeat(
    num_epochs=None, reseed_each_epoch=True
)
```

**Parameters:**
- `num_epochs` — number of repetitions (`None` = infinite).
- `reseed_each_epoch` (MapDataset only) — if `True`, random transforms (shuffle, random_map) use different seeds per epoch.

| Property       | Value                                  |
|----------------|----------------------------------------|
| Dataset types  | MapDataset, IterDataset                |
| Length change   | Multiplied (`N * epochs` or infinite) |
| Restriction     | Cannot repeat already-infinite datasets |

---

## 12. `slice`

Selects a subset of elements using Python slice semantics.

```python
ds = grain.MapDataset.source([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

ds.slice(slice(0, 5))       # [0, 1, 2, 3, 4]
ds.slice(slice(0, 10, 2))   # [0, 2, 4, 6, 8]
ds[2:7]                      # [2, 3, 4, 5, 6]
```

**Parameters:**
- `sl` — a Python `slice` object (`start`, `stop`, `step`).

| Property       | Value                       |
|----------------|-----------------------------|
| Dataset types  | MapDataset only             |
| Length change   | Yes (reduced or strided)   |
| Supports        | Negative indices, stepping |

---

## 13. `zip`

Combines multiple datasets element-wise into tuples.

```python
ds_x = grain.MapDataset.source([1, 2, 3])
ds_y = grain.MapDataset.source(["a", "b", "c"])

ds = grain.experimental.ZipMapDataset([ds_x, ds_y])
# [(1, "a"), (2, "b"), (3, "c")]

# IterDataset with strict length checking
ds = grain.experimental.ZipIterDataset([iter_a, iter_b], strict=True)
```

**Parameters:**
- `parents` — sequence of datasets to zip.
- `strict` (IterDataset only) — if `True`, raises error if datasets have different lengths.

**Behavior:**
- MapDataset requires all parents to have the same length.
- IterDataset stops at the shortest dataset (unless `strict=True`).

| Property       | Value                       |
|----------------|-----------------------------|
| Dataset types  | MapDataset, IterDataset     |
| Length change   | No (same as inputs)        |
| Requirement     | Same-length inputs (Map)   |

---

## Quick Reference

| Transform   | MapDataset | IterDataset | Chainable Method       | Length Effect         |
|-------------|:----------:|:-----------:|------------------------|-----------------------|
| map         | Y          | Y           | `.map(fn)`             | Same                  |
| batch       | Y          | Y           | `.batch(size)`         | `ceil(N/size)`        |
| filter      | Y          | Y           | `.filter(fn)`          | Reduced (sparse)      |
| shuffle     | Y          | Y           | `.shuffle(seed=)`      | Same                  |
| mix         | Y          | Y           | `MapDataset.mix([..])`  | Weighted combination  |
| flatmap     | Y          | Y           | `.flat_map(fn)`        | `N * max_fan_out`     |
| cache       |            | Y           | `.cache()`             | Same                  |
| prefetch    |            | Y           | `.to_iter_dataset()`   | Same                  |
| packing     |            | Y           | `FirstFitPack...()`    | Reduced               |
| interleave  |            | Y           | `Interleave...()`      | Sum of inputs         |
| repeat      | Y          | Y           | `.repeat(epochs)`      | `N * epochs`          |
| slice       | Y          |             | `.slice(sl)` / `ds[:]` | Reduced               |
| zip         | Y          | Y           | `Zip...Dataset([..])`  | Same                  |
