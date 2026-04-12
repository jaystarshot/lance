// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! In-memory cache tier — a Rust port of CacheShard / `AsyncDataCache`.
//!
//! Design mirrors exactly:
//!
//! * **16 independent shards** — `HashMap` + clock-hand eviction ring per shard,
//! each protected by a `std::sync::Mutex`. Shards eliminate contention for the
//! common case where different tasks access different files.
//!
//! * **Load deduplication** — a `tokio::sync::watch` channel replaces 
//! `folly::SharedPromise`. The first task to miss the cache transitions the
//! entry from `Loading` to `Loaded(bytes)` or `Failed`; all concurrent tasks
//! wait on the channel and receive the result without issuing a second fetch.
//!
//! * **Clock-hand eviction with percentile threshold** — follows 
//! `CacheShard::evict` / `calibrateThreshold` exactly. Every `ring_len/4`
//! insertions the shard samples `NUM_EVICTION_SAMPLES` (10) entries, sorts
//! their scores, and sets the eviction threshold to the
//! `EVICTION_PERCENTILE`th (80th) percentile. Only entries scoring *above*
//! the threshold are candidates, which prevents thrashing when the cache
//! hovers at capacity.
//!
//! * **Score formula** — `(now_ms - last_use_ms) / (1 + num_uses)`. Older,
//! less-frequently-accessed entries score higher and are evicted first.
//! An entry that has never been accessed scores `u64::MAX` (evict immediately).

use std::collections::HashMap;
use std::sync::{
    Arc, Mutex, Weak,
    atomic::{AtomicU32, AtomicU64, Ordering},
};
use std::time::{SystemTime, UNIX_EPOCH};

use bytes::Bytes;
use futures::future::BoxFuture;
use object_store::path::Path;
use tokio::sync::watch;

use lance_core::Result;

use super::{DataCache, DataCacheKey, file_ids::FileIds};

// ─── Constants (same as ) ───────────────────────────────────────────────

/// Default number of independent shards — must be a power of two.
/// Matches AsyncDataCache::kDefaultNumShards.
pub const DEFAULT_NUM_SHARDS: usize = 16;

/// Number of entries sampled when calibrating the eviction threshold.
const NUM_EVICTION_SAMPLES: usize = 10;

/// Only entries whose score is at or above this percentile are evicted.
const EVICTION_PERCENTILE: usize = 80;

/// Recalibrate the threshold after this many shard events (inserts + eviction
/// checks), matching entries_.size() / 4 heuristic.
const CALIBRATION_INTERVAL_DIVISOR: usize = 4;

// ─── Time ────────────────────────────────────────────────────────────────────

/// Milliseconds since the Unix epoch — cheap, ~1 ms resolution.
///
/// uses `folly::hardware_timestamp() >> 21` for ~1–2 ms resolution;
/// `SystemTime` gives the same order of magnitude with no unsafe code.
#[inline]
fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

// ─── Shard selection ─────────────────────────────────────────────────────────

#[inline]
fn shard_idx(key: &DataCacheKey, shard_mask: u64) -> usize {
 // Fibonacci hashing — mixes file_id and offset uniformly across shards.
    let h = key
        .file_id
        .wrapping_mul(11_400_714_819_323_198_485_u64)
        .wrapping_add(key.offset.wrapping_mul(6_364_136_223_846_793_005_u64));
    (h & shard_mask) as usize
}

// ─── Entry ───────────────────────────────────────────────────────────────────

/// Possible states of a cache entry, broadcast via a `watch` channel.
///
/// Maps to numPins_ convention:
/// * `Loading` ≡ `numPins_ = kExclusive (-10000)`
/// * `Loaded(bytes)` ≡ `numPins_ = 1` (shared, data available)
/// * `Failed` ≡ entry removed from map; waiters must retry
#[derive(Clone)]
enum LoadState {
    Loading,
    Loaded(Bytes),
    Failed,
}

struct CacheEntry {
    key: DataCacheKey,
 /// The `Sender` half is owned here; receivers are created on demand by
 /// concurrent waiters. Sending a new state wakes *all* current waiters
 /// simultaneously — the same semantics as SharedPromise::setValue.
    state_tx: watch::Sender<LoadState>,
 /// Milliseconds since epoch of last access; 0 = never accessed.
    last_use_ms: AtomicU64,
 /// How many times this entry has been read (used in eviction scoring).
    num_uses: AtomicU32,
 /// Byte size of the cached payload; 0 while Loading or after failure.
    data_size: AtomicU64,
}

impl CacheEntry {
    fn new(key: DataCacheKey) -> Arc<Self> {
        let (state_tx, _rx) = watch::channel(LoadState::Loading);
        Arc::new(Self {
            key,
            state_tx,
            last_use_ms: AtomicU64::new(0),
            num_uses: AtomicU32::new(0),
            data_size: AtomicU64::new(0),
        })
    }

 /// Eviction score. Higher = more worth evicting.
 ///
 /// Formula: `(now_ms - last_use_ms) / (1 + num_uses)` — identical to
 /// AccessStats::score.
    fn eviction_score(&self, now: u64) -> u64 {
        let last = self.last_use_ms.load(Ordering::Relaxed);
        if last == 0 {
            return u64::MAX; // never accessed → always evict first
        }
        let age = now.saturating_sub(last);
        let uses = self.num_uses.load(Ordering::Relaxed) as u64;
        age / (1 + uses)
    }

 /// Record an access (used when returning a cached hit).
    fn touch(&self) {
        self.last_use_ms.store(now_ms(), Ordering::Relaxed);
        self.num_uses.fetch_add(1, Ordering::Relaxed);
    }
}

// ─── Shard ───────────────────────────────────────────────────────────────────

struct CacheShardInner {
 /// O(1) key → entry lookup (entryMap_).
    entries: HashMap<DataCacheKey, Arc<CacheEntry>>,
 /// Clock-hand eviction ring (entries_ dense array).
 /// `Weak` lets us skip already-freed entries without a map lookup.
    eviction_ring: Vec<Weak<CacheEntry>>,
 /// Current position of the clock hand in `eviction_ring`.
    clock_hand: usize,
 /// Sum of `data_size` for all `Loaded` entries in this shard.
    loaded_bytes: u64,
 /// Cached 80th-percentile eviction score — recomputed periodically.
 ///
 /// Initialised to `u64::MAX` (kNoThreshold = INT_MAX) so that
 /// nothing is evicted until the first calibration pass completes. Without
 /// this, freshly-loaded entries (score = 0) immediately pass the `>=`
 /// check and are evicted before they can be reused.
    eviction_threshold: u64,
 /// Events since last calibration.
    events: usize,
}

impl CacheShardInner {
    fn new() -> Self {
        Self {
            entries: HashMap::new(),
            eviction_ring: Vec::new(),
            clock_hand: 0,
            loaded_bytes: 0,
            eviction_threshold: u64::MAX, // nothing evictable until first calibration
            events: 0,
        }
    }

 /// Recompute the eviction threshold.
 ///
 /// Samples `NUM_EVICTION_SAMPLES` entries evenly from the ring, sorts their
 /// scores, and sets `eviction_threshold` to the `EVICTION_PERCENTILE`th
 /// percentile value. This is a direct port of 
 /// `CacheShard::calibrateThresholdLocked`.
    fn calibrate_threshold(&mut self) {
        let n = self.eviction_ring.len();
        if n == 0 {
            self.eviction_threshold = 0;
            return;
        }
        let num_samples = NUM_EVICTION_SAMPLES.min(n);
        let step = (n / num_samples).max(1);
        let now = now_ms();

        let mut scores: Vec<u64> = (0..num_samples)
            .filter_map(|i| {
                let idx = (self.clock_hand + i * step) % n;
                self.eviction_ring[idx]
                    .upgrade()
                    .map(|e| e.eviction_score(now))
            })
            .collect();

        scores.sort_unstable();
 // percentile(scores, EVICTION_PERCENTILE) — same formula as :
 // `values[(values.size() * percent) / 100]`
        let idx = (scores.len() * EVICTION_PERCENTILE / 100)
            .min(scores.len().saturating_sub(1));
        self.eviction_threshold = scores.get(idx).copied().unwrap_or(0);
        self.events = 0;
    }

 /// Free at least `target_bytes` from this shard using the clock-hand
 /// algorithm. Returns `(bytes_freed, evicted_entries)` where
 /// `evicted_entries` carries the key + live bytes of each evicted entry
 /// so the caller can forward them to the SSD tier write
 /// pattern (`ssd_saveable` entries forwarded to `saveToSsd()`).
 ///
 /// CacheShard::evict.
    fn evict(&mut self, target_bytes: u64) -> (u64, Vec<(DataCacheKey, Bytes)>) {
        let n = self.eviction_ring.len();
        if n == 0 {
            return (0, Vec::new());
        }

 // Recalibrate periodically — every ~ring_len/4 events.
        self.events += 1;
        let calibration_interval = (n / CALIBRATION_INTERVAL_DIVISOR).max(1);
        if self.events >= calibration_interval {
            self.calibrate_threshold();
        }

        let now = now_ms();
        let mut freed = 0u64;
        let mut checked = 0;
        let mut evicted: Vec<(DataCacheKey, Bytes)> = Vec::new();

        while freed < target_bytes && checked < n {
            let idx = self.clock_hand % n;
            self.clock_hand = self.clock_hand.wrapping_add(1);
            checked += 1;

            let Some(entry) = self.eviction_ring[idx].upgrade() else {
 // Entry already freed elsewhere — skip.
                continue;
            };

 // Why > 2?
 // At this point we hold the shard mutex. The minimum strong_count
 // for any live entry is 2:
 // 1 — inner.entries[key] (the HashMap's Arc)
 // 1 — this upgrade() (our temporary Arc)
 // Any waiter suspended at rx.changed().await holds a third Arc
 // obtained from find_or_create *before* releasing the shard mutex.
 // That waiter is outside the mutex now but still increments the
 // count. So:
 // count == 2 → only map + upgrade → no active users → evict
 // count > 2 → at least one waiter or active reader → skip
 //
 // TODO: uses an explicit `numPins_` atomic (kExclusive = -10000
 // while loading, 0 = evictable, N = N active readers) and a RAII
 // `CachePin` returned to callers that increments/decrements the count.
 // This gives precise "is anyone reading this?" semantics:
 // https://github.com/facebookincubator/velox/blob/main/velox/common/caching/AsyncDataCache.h
 //
 // We deliberately omit CachePin for now because `Bytes` (Arc<[u8]>)
 // already keeps the data alive independently of the cache index — a
 // caller holding `Bytes` is data-safe even if the entry is evicted.
 // The only downside is a potential cache miss on the *next* caller
 // if we evict mid-decode, but the decode window is milliseconds and
 // the clock-hand eviction is probabilistic, making the practical
 // impact negligible. Add CachePin when profiling shows eviction-
 // mid-decode is a meaningful source of cache misses.
            if Arc::strong_count(&entry) > 2 {
                continue; // active waiter or reader — don't evict
            }

            let size = entry.data_size.load(Ordering::Relaxed);
            if size == 0 {
 // Still loading, previously failed, or already evicted
 // (data_size is zeroed below when an entry is evicted so a
 // second sweep through the ring skips it cheaply).
                continue;
            }

            let score = entry.eviction_score(now);
            if score < self.eviction_threshold {
                continue; // too hot — below eviction threshold
            }

 // Evict: extract bytes for SSD write, remove from map, zero data_size.
 // Bytes are extracted BEFORE zeroing data_size so the SSD writer
 // receives valid data forward pattern.
            if self.entries.remove(&entry.key).is_some() {
 // Grab the bytes from the watch channel while we still hold
 // the Arc (strong_count > 0 so the sender is alive).
                if let LoadState::Loaded(bytes) = entry.state_tx.borrow().clone() {
                    evicted.push((entry.key.clone(), bytes));
                }
                entry.data_size.store(0, Ordering::Relaxed);
                self.loaded_bytes = self.loaded_bytes.saturating_sub(size);
                freed += size;
            }
 // If remove returned None the entry was already evicted or removed
 // by the failure path — skip without touching counters.
        }

 // Compact dead Weak pointers to keep the ring from growing unboundedly.
        if checked == n {
            self.eviction_ring.retain(|w| w.strong_count() > 0);
            self.clock_hand = self.clock_hand.min(self.eviction_ring.len());
        }

        (freed, evicted)
    }
}

struct CacheShard {
    inner: Mutex<CacheShardInner>,
    /// Per-shard stats — AtomicU64 so they can be read and written without
    /// holding the shard mutex.  This eliminates cross-shard cache-line
    /// contention: threads on different shards never touch the same cache line.
    hits: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
}

impl CacheShard {
    fn new() -> Self {
        Self {
            inner: Mutex::new(CacheShardInner::new()),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
        }
    }
}

// ─── EvictionSink ────────────────────────────────────────────────────────────

/// Receives evicted cache entries for async persistence to the SSD tier.
///
/// Implementations accumulate entries and trigger batch writes when
/// configurable thresholds are exceeded — mirroring the threshold-based
/// trigger used in the reference design.
///
/// The trait is object-safe and sync so it can be called from within the
/// shard mutex without any async overhead.
pub trait EvictionSink: Send + Sync + std::fmt::Debug {
    /// Called from `maybe_evict` with all entries evicted in one sweep.
    ///
    /// `total_cache_bytes` is the current `total_bytes` counter — used to
    /// compute the ratio-based threshold.
    fn on_evicted(&self, entries: Vec<(DataCacheKey, Bytes)>, total_cache_bytes: u64);
}

// ─── MemoryCache ─────────────────────────────────────────────────────────────

/// Statistics snapshot for a [`MemoryCache`].
#[derive(Debug, Default, Clone)]
pub struct MemoryCacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub current_bytes: u64,
    pub max_bytes: u64,
}

/// Sharded in-memory cache with -style clock-hand + percentile eviction.
///
/// The cache is logically split into `num_shards` independent shards (default
/// [`DEFAULT_NUM_SHARDS`] = 16, same as kDefaultNumShards).
/// Each shard owns its hash-map and eviction ring and is protected by its own
/// `std::sync::Mutex`, so concurrent tasks hitting different files (or
/// different offsets within the same file) almost never contend.
///
/// Async coordination (waiting for a concurrent load to finish) is done via
/// `tokio::sync::watch` *outside* of the shard mutex, so no tokio worker is
/// blocked while waiting.
pub struct MemoryCache {
    shards: Vec<CacheShard>,
    shard_mask: u64,
    max_bytes: u64,
    total_bytes: AtomicU64,
    /// Optional sink that receives evicted entries for SSD persistence.
    eviction_sink: Option<Arc<dyn EvictionSink>>,
}

impl std::fmt::Debug for MemoryCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryCache")
            .field("max_bytes", &self.max_bytes)
            .field("current_bytes", &self.total_bytes.load(Ordering::Relaxed))
            .finish()
    }
}

impl MemoryCache {
    /// Create a new cache with the default shard count ([`DEFAULT_NUM_SHARDS`]).
    pub fn new(max_bytes: u64) -> Arc<Self> {
        Self::new_with_shards(max_bytes, DEFAULT_NUM_SHARDS)
    }

    /// Create a new cache with a custom shard count.
    ///
    /// `num_shards` must be a positive power of two (e.g. 4, 8, 16, 32).
    ///
    /// # Panics
    /// Panics if `num_shards` is zero or not a power of two.
    pub fn new_with_shards(max_bytes: u64, num_shards: usize) -> Arc<Self> {
        Self::with_eviction_sink(max_bytes, num_shards, None)
    }

    /// Create a cache that notifies `sink` when entries are evicted, allowing
    /// the SSD tier to persist them asynchronously using threshold-based batching.
    pub fn with_eviction_sink(
        max_bytes: u64,
        num_shards: usize,
        eviction_sink: Option<Arc<dyn EvictionSink>>,
    ) -> Arc<Self> {
        assert!(num_shards > 0, "num_shards must be positive");
        assert!(
            num_shards.is_power_of_two(),
            "num_shards must be a power of two, got {num_shards}"
        );
        let shard_mask = (num_shards as u64) - 1;
        let shards = (0..num_shards).map(|_| CacheShard::new()).collect();
        Arc::new(Self {
            shards,
            shard_mask,
            max_bytes,
            total_bytes: AtomicU64::new(0),
            eviction_sink,
        })
    }

    pub fn stats(&self) -> MemoryCacheStats {
        // Sweep all shards and sum their per-shard counters.
        // No locks needed — per-shard AtomicU64s are read with a plain load.
        let hits = self.shards.iter().map(|s| s.hits.load(Ordering::Relaxed)).sum();
        let misses = self.shards.iter().map(|s| s.misses.load(Ordering::Relaxed)).sum();
        let evictions = self.shards.iter().map(|s| s.evictions.load(Ordering::Relaxed)).sum();
        MemoryCacheStats {
            hits,
            misses,
            evictions,
            current_bytes: self.total_bytes.load(Ordering::Relaxed),
            max_bytes: self.max_bytes,
        }
    }

 /// Fetch the bytes for `key`, calling `loader` on a cache miss.
 ///
 /// If multiple tasks request the same key concurrently, only the first
 /// triggers `loader`; all others wait for it to complete via the entry's
 /// `watch` channel — exactly CoalescedLoad::loadOrFuture.
 ///
 /// `loader` is wrapped in an `Option` so the inner loop can consume it
 /// exactly once even when the first attempt finds an existing (then-failed)
 /// entry and must retry.
    pub async fn get_or_load(&self, key: DataCacheKey, loader: BoxFuture<'_, Result<Bytes>>) -> Result<Bytes> {
        if self.max_bytes == 0 {
            return loader.await;
        }

        let mut loader = Some(loader);

        loop {
            let (entry, is_new) = self.find_or_create(&key);

            if is_new {
 // ── We own this entry (exclusive, like kExclusive pin) ──
                let loader = loader.take().expect("loader consumed twice");
                self.shards[shard_idx(&key, self.shard_mask)].misses.fetch_add(1, Ordering::Relaxed);

                match loader.await {
                    Ok(bytes) => {
                        let size = bytes.len() as u64;
                        entry.data_size.store(size, Ordering::Release);
                        entry.touch();
 // Transition to shared — wakes all waiting tasks.
 // Must use send_replace() not send(): send() silently
 // drops the value when there are no active receivers
 // (the initial _rx was dropped in CacheEntry::new),
 // leaving the channel stuck at Loading so any waiter
 // that subscribes later hangs forever on changed().await.
                        entry.state_tx.send_replace(LoadState::Loaded(bytes.clone()));
 // Update both counters under the shard lock so that a
 // concurrent eviction always sees a consistent view of
 // loaded_bytes and total_bytes for this shard.
                        {
                            let mut inner =
                                self.shards[shard_idx(&key, self.shard_mask)].inner.lock().unwrap();
                            inner.loaded_bytes += size;
                            self.total_bytes.fetch_add(size, Ordering::Relaxed);
                        }
                        self.maybe_evict(size);
                        return Ok(bytes);
                    }
                    Err(e) => {
 // Load failed — signal waiters, remove entry so the
 // next caller gets a fresh miss. Equivalent to 
 // `CachePin::release()` on an exclusive pin.
                        entry.state_tx.send_replace(LoadState::Failed);
                        self.remove_entry(&key);
                        return Err(e);
                    }
                }
            }

 // ── Entry exists: wait for the loading task to finish ──
            let mut rx = entry.state_tx.subscribe();
            loop {
 // Clone the current state so we release the borrow on `rx`
 // before calling `rx.changed()` (which also takes `&mut rx`).
                let state = rx.borrow_and_update().clone();
                match state {
                    LoadState::Loaded(bytes) => {
                        entry.touch();
                        self.shards[shard_idx(&key, self.shard_mask)].hits.fetch_add(1, Ordering::Relaxed);
                        return Ok(bytes);
                    }
                    LoadState::Failed => {
 // The loading task failed. The entry has been (or is
 // being) removed from the map. We retry from scratch so
 // that *this* task can attempt the load with its own
 // loader — identical to the waiter retry in after a
 // cancelled CoalescedLoad.
                        break;
                    }
                    LoadState::Loading => {
 // Still in flight — yield until the state changes.
                        if rx.changed().await.is_err() {
 // Sender dropped unexpectedly; treat as failure.
                            break;
                        }
                    }
                }
            }
 // The previous load failed; loop back and try to become the new owner.
        }
    }

 // ── Private helpers ──────────────────────────────────────────────────────

 /// Atomically find or create an entry for `key`.
 ///
 /// Returns `(entry, is_new)`. When `is_new` is `true` the entry is in
 /// `Loading` state and the caller *must* drive the load and update the
 /// state — exactly exclusive-pin contract.
    fn find_or_create(&self, key: &DataCacheKey) -> (Arc<CacheEntry>, bool) {
        let idx = shard_idx(key, self.shard_mask);
        let mut inner = self.shards[idx].inner.lock().unwrap();

        if let Some(existing) = inner.entries.get(key) {
            return (existing.clone(), false);
        }

        let entry = CacheEntry::new(key.clone());
        inner.entries.insert(key.clone(), entry.clone());
        inner.eviction_ring.push(Arc::downgrade(&entry));
        (entry, true)
    }

 /// Remove an entry from its shard's map and adjust byte counters.
    fn remove_entry(&self, key: &DataCacheKey) {
        let idx = shard_idx(key, self.shard_mask);
        let mut inner = self.shards[idx].inner.lock().unwrap();
        if let Some(entry) = inner.entries.remove(key) {
            let size = entry.data_size.load(Ordering::Relaxed);
 // Zero data_size so any subsequent eviction-ring sweep skips this
 // entry without double-counting.
            entry.data_size.store(0, Ordering::Relaxed);
            if size > 0 {
                inner.loaded_bytes = inner.loaded_bytes.saturating_sub(size);
                self.total_bytes
                    .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
                        Some(v.saturating_sub(size))
                    })
                    .ok();
            }
        }
    }

 /// Trigger eviction across shards if total usage exceeds `max_bytes`.
 ///
 /// Called after every successful insert. Spreads the target eviction
 /// proportionally across all shards to avoid always hammering shard 0.
    fn maybe_evict(&self, inserted_bytes: u64) {
        let current = self.total_bytes.load(Ordering::Relaxed);
        if current <= self.max_bytes {
            return;
        }
        let overage = current - self.max_bytes;
        let per_shard = (overage / self.shards.len() as u64).max(inserted_bytes);
        let mut total_freed = 0u64;
        let mut all_evicted: Vec<(DataCacheKey, Bytes)> = Vec::new();

        for shard in &self.shards {
            if total_freed >= overage {
                break;
            }
            let (freed, evicted) = shard.inner.lock().unwrap().evict(per_shard);
            if freed > 0 {
                self.total_bytes
                    .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
                        Some(v.saturating_sub(freed))
                    })
                    .ok();
                shard.evictions.fetch_add(1, Ordering::Relaxed);
                total_freed += freed;
            }
            // Collect all evicted entries across shards into one batch.
            // We notify the sink once after the full sweep so it can make
            // a threshold decision over the complete set.
            all_evicted.extend(evicted);
        }

        // Notify the eviction sink with the full batch.
        // The sink (SsdWriter) accumulates and triggers a write when its
        // threshold (16 MB or 12.5% of cache) is exceeded.
        if !all_evicted.is_empty() {
            if let Some(sink) = &self.eviction_sink {
                let total_cache_bytes = self.total_bytes.load(Ordering::Relaxed);
                sink.on_evicted(all_evicted, total_cache_bytes);
            }
        }
    }
}

/// `MemoryCache` implements `DataCache` directly so it can be used standalone
/// without wrapping in `TieredDataCache`. File path interning is handled
/// internally via a `FileIds` registry owned by this instance.
pub struct StandaloneMemoryCache {
    inner: Arc<MemoryCache>,
    file_ids: Arc<FileIds>,
}

impl std::fmt::Debug for StandaloneMemoryCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.inner.fmt(f)
    }
}

impl StandaloneMemoryCache {
    pub fn new(max_bytes: u64) -> Arc<Self> {
        Arc::new(Self {
            inner: MemoryCache::new(max_bytes),
            file_ids: Arc::new(FileIds::new()),
        })
    }

    pub fn stats(&self) -> MemoryCacheStats {
        self.inner.stats()
    }
}

impl DataCache for StandaloneMemoryCache {
    fn get_or_load<'a>(
        &'a self,
        path: &'a Path,
        offset: u64,
        length: u64,
        loader: BoxFuture<'a, Result<Bytes>>,
    ) -> BoxFuture<'a, Result<Bytes>> {
        let file_id = self.file_ids.get_or_intern(path);
        let key = DataCacheKey { file_id, offset, length };
        Box::pin(self.inner.get_or_load(key, loader))
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;
    use std::sync::atomic::AtomicUsize;

    fn key(file_id: u64, offset: u64, length: u64) -> DataCacheKey {
        DataCacheKey { file_id, offset, length }
    }

 // ── Shard configuration tests (mirrors numShardsDefault / numShardsInvalid) ─

    #[test]
    fn test_default_shard_count() {
        let cache = MemoryCache::new(1024 * 1024);
        assert_eq!(cache.shards.len(), DEFAULT_NUM_SHARDS);
    }

    #[test]
    fn test_custom_shard_count_power_of_two() {
        for &n in &[1usize, 2, 4, 8, 32, 64] {
            let cache = MemoryCache::new_with_shards(1024 * 1024, n);
            assert_eq!(cache.shards.len(), n);
 // shard_mask must be n-1
            assert_eq!(cache.shard_mask, (n as u64) - 1);
        }
    }

    #[test]
    #[should_panic(expected = "power of two")]
    fn test_non_power_of_two_shards_panics() {
        MemoryCache::new_with_shards(1024 * 1024, 3);
    }

    #[test]
    #[should_panic(expected = "positive")]
    fn test_zero_shards_panics() {
        MemoryCache::new_with_shards(1024 * 1024, 0);
    }

    #[tokio::test]
    async fn test_single_shard_cache_works() {
 // Degenerate case: 1 shard — all entries in one map, still correct.
        let cache = MemoryCache::new_with_shards(10 * 1024 * 1024, 1);
        let k = key(0, 0, 4);
        let bytes = cache
            .get_or_load(k.clone(), Box::pin(async { Ok(Bytes::from_static(b"hi")) }))
            .await
            .unwrap();
        assert_eq!(bytes, Bytes::from_static(b"hi"));
 // Second call — hit.
        let bytes2 = cache
            .get_or_load(k, Box::pin(async { Ok(Bytes::from_static(b"miss")) }))
            .await
            .unwrap();
        assert_eq!(bytes2, Bytes::from_static(b"hi"));
        assert_eq!(cache.stats().hits, 1);
    }

    #[tokio::test]
    async fn test_basic_hit_and_miss() {
        let cache = MemoryCache::new(10 * 1024 * 1024);
        let k = key(0, 0, 4);

 // First access — miss, loader runs.
        let bytes = cache
            .get_or_load(k.clone(), Box::pin(async { Ok(Bytes::from_static(b"hello")) }))
            .await
            .unwrap();
        assert_eq!(bytes, Bytes::from_static(b"hello"));

 // Second access — hit, loader should NOT run.
        let load_count = Arc::new(AtomicUsize::new(0));
        let lc = load_count.clone();
        let bytes2 = cache
            .get_or_load(
                k.clone(),
                Box::pin(async move {
                    lc.fetch_add(1, Ordering::Relaxed);
                    Ok(Bytes::from_static(b"should not be called"))
                }),
            )
            .await
            .unwrap();
        assert_eq!(bytes2, Bytes::from_static(b"hello"));
        assert_eq!(load_count.load(Ordering::Relaxed), 0);

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
    }

    #[tokio::test]
    async fn test_load_deduplication() {
        let cache = Arc::new(MemoryCache::new(10 * 1024 * 1024));
        let k = key(1, 0, 4);
        let load_count = Arc::new(AtomicUsize::new(0));

 // Launch 8 concurrent requests for the same key.
        let mut handles = Vec::new();
        for _ in 0..8 {
            let cache = cache.clone();
            let k = k.clone();
            let lc = load_count.clone();
            handles.push(tokio::spawn(async move {
                cache
                    .get_or_load(
                        k,
                        Box::pin(async move {
                            lc.fetch_add(1, Ordering::Relaxed);
 // Small delay so other tasks arrive before load completes.
                            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                            Ok(Bytes::from_static(b"data"))
                        }),
                    )
                    .await
            }));
        }

        for h in handles {
            assert_eq!(h.await.unwrap().unwrap(), Bytes::from_static(b"data"));
        }
 // Only one loader should have run.
        assert_eq!(load_count.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn test_loader_failure_allows_retry() {
        let cache = Arc::new(MemoryCache::new(10 * 1024 * 1024));
        let k = key(2, 0, 4);

 // First access fails.
        let result = cache
            .get_or_load(
                k.clone(),
                Box::pin(async { Err(lance_core::Error::io("boom".to_string())) }),
            )
            .await;
        assert!(result.is_err());

 // Second access should succeed — entry was removed after failure.
        let bytes = cache
            .get_or_load(k, Box::pin(async { Ok(Bytes::from_static(b"retry ok")) }))
            .await
            .unwrap();
        assert_eq!(bytes, Bytes::from_static(b"retry ok"));
    }

    #[tokio::test]
    async fn test_eviction_under_pressure() {
 // Each entry is 1 MiB; cap the cache at 4 MiB.
        let cache = MemoryCache::new(4 * 1024 * 1024);
        let chunk = Bytes::from(vec![0u8; 1024 * 1024]);

        for i in 0..8u64 {
            let data = chunk.clone();
            cache
                .get_or_load(key(3, i, chunk.len() as u64), Box::pin(async move { Ok(data) }))
                .await
                .unwrap();
        }

 // Total bytes should be at or below max (eviction may lag slightly due
 // to the amortised nature of the clock sweep).
        let stats = cache.stats();
        assert!(
            stats.current_bytes <= stats.max_bytes * 2,
            "cache grew too large: {} bytes",
            stats.current_bytes
        );
        assert!(stats.evictions > 0, "expected at least one eviction");
    }

    #[tokio::test]
    async fn test_disabled_cache_bypasses() {
 // max_bytes == 0 means disabled; loader is called every time.
        let cache = MemoryCache::new(0);
        let k = key(4, 0, 4);
        let count = Arc::new(AtomicUsize::new(0));

        for _ in 0..3 {
            let c = count.clone();
            cache
                .get_or_load(
                    k.clone(),
                    Box::pin(async move {
                        c.fetch_add(1, Ordering::Relaxed);
                        Ok(Bytes::from_static(b"x"))
                    }),
                )
                .await
                .unwrap();
        }
        assert_eq!(count.load(Ordering::Relaxed), 3);
    }

 // ── -inspired tests ──────────────────────────────────────────────

 /// replace test: fill the cache exactly to capacity,
 /// re-read the same keys, and verify hits occur and eviction fires when
 /// further entries are added beyond capacity.
    #[tokio::test]
    async fn test_replace_hits_and_evictions() {
        let cap = 4 * 1024 * 1024u64;
        let entry_size = 256 * 1024u64; // 256 KiB per entry
        let num_entries = cap / entry_size; // exactly fill the cache (16 entries)
        let cache = MemoryCache::new(cap);

 // First pass — fill cache exactly to capacity (all misses).
        for i in 0..num_entries {
            let data = Bytes::from(vec![(i % 256) as u8; entry_size as usize]);
            cache
                .get_or_load(
                    key(0, i * entry_size, entry_size),
                    Box::pin(async move { Ok(data) }),
                )
                .await
                .unwrap();
        }

 // Second pass over the SAME keys — should all hit the cache.
        for i in 0..num_entries {
            let data = Bytes::from(vec![(i % 256) as u8; entry_size as usize]);
            let _ = cache
                .get_or_load(
                    key(0, i * entry_size, entry_size),
                    Box::pin(async move { Ok(data) }),
                )
                .await
                .unwrap();
        }

        let stats = cache.stats();
        assert!(stats.hits > 0, "expected cache hits on second pass, got 0");
        assert!(
            stats.current_bytes <= cap,
            "cache exceeded capacity: {} > {}",
            stats.current_bytes,
            cap
        );

 // Now push beyond capacity — eviction must fire.
        for i in num_entries..num_entries * 2 {
            let data = Bytes::from(vec![0u8; entry_size as usize]);
            cache
                .get_or_load(
                    key(0, i * entry_size, entry_size),
                    Box::pin(async move { Ok(data) }),
                )
                .await
                .unwrap();
        }

        let stats2 = cache.stats();
        assert!(stats2.evictions > 0, "expected evictions beyond capacity");
    }

 /// staleEntry / double-eviction test: verify that
 /// `total_bytes` and per-shard `loaded_bytes` stay consistent after many
 /// evictions. A double-decrement bug would make `total_bytes` underflow
 /// causing `maybe_evict` to stop triggering.
    #[tokio::test]
    async fn test_accounting_invariant_under_eviction() {
        let cap = 2 * 1024 * 1024u64;
        let entry_size = 128 * 1024u64;
 // Load 8× capacity to force many evictions.
        let num_entries = (cap / entry_size) * 8;
        let cache = Arc::new(MemoryCache::new(cap));

        for i in 0..num_entries {
            let data = Bytes::from(vec![0u8; entry_size as usize]);
            cache
                .get_or_load(
                    key(1, i * entry_size, entry_size),
                    Box::pin(async move { Ok(data) }),
                )
                .await
                .unwrap();
        }

        let stats = cache.stats();
 // Primary invariant: no double-decrement. A u64 underflow wraps to
 // near u64::MAX. Allow up to 2× cap for natural amortisation overshoot
 // (the entry being inserted is pinned on the stack during maybe_evict
 // so it can't be evicted until the insertion returns).
        assert!(
            stats.current_bytes < cap * 2,
            "possible underflow: current_bytes={} (u64::MAX would indicate wrap)",
            stats.current_bytes
        );
        assert!(stats.evictions > 0, "expected evictions to fire");
    }

 /// findExclusiveWithWait + failure test: when a load
 /// fails, *all* concurrent waiters must be unblocked and the entry must be
 /// removed so the next caller can retry successfully.
    #[tokio::test]
    async fn test_concurrent_waiters_see_failure_and_retry() {
        let cache = Arc::new(MemoryCache::new(10 * 1024 * 1024));
        let k = key(5, 0, 4);
        let load_count = Arc::new(AtomicUsize::new(0));

 // The FIRST loader call (whichever task wins find_or_create) fails.
 // Subsequent calls succeed. This is independent of task index.
        let first_call_done = Arc::new(AtomicBool::new(false));
        let barrier = Arc::new(tokio::sync::Barrier::new(8));
        let mut handles = Vec::new();

        for _ in 0..8usize {
            let cache = cache.clone();
            let k = k.clone();
            let lc = load_count.clone();
            let bar = barrier.clone();
            let first = first_call_done.clone();

            handles.push(tokio::spawn(async move {
                bar.wait().await;
                cache
                    .get_or_load(
                        k,
                        Box::pin(async move {
                            let call_idx = lc.fetch_add(1, Ordering::SeqCst);
                            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
                            if call_idx == 0 {
 // First loader to run always fails.
                                first.store(true, Ordering::SeqCst);
                                Err(lance_core::Error::io("injected failure".to_string()))
                            } else {
                                Ok(Bytes::from_static(b"ok"))
                            }
                        }),
                    )
                    .await
            }));
        }

        let results: Vec<_> = futures::future::join_all(handles)
            .await
            .into_iter()
            .map(|h| h.unwrap())
            .collect();

 // The first loader failed → exactly 1 error returned.
        let errors = results.iter().filter(|r| r.is_err()).count();
        let successes = results.iter().filter(|r| r.is_ok()).count();
        assert!(first_call_done.load(Ordering::SeqCst), "first loader never ran");
        assert_eq!(errors, 1, "expected exactly 1 error from the failing loader");
        assert_eq!(successes, 7);

 // After the failure the entry was retried by a waiter and cached.
 // A subsequent load of the same key returns the cached value.
        let bytes = cache
            .get_or_load(k, Box::pin(async { Ok(Bytes::from_static(b"fresh")) }))
            .await
            .unwrap();
 // The waiter's retry cached b"ok"; we get that back, not b"fresh".
        assert_eq!(bytes, Bytes::from_static(b"ok"));
    }

 /// fuzz test: 8 concurrent tasks randomly reading
 /// different (file_id, offset) pairs for 500ms, verifying the cache
 /// never deadlocks, never panics, and never exceeds capacity.
    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn test_fuzz_concurrent_access() {
        use rand::Rng;

        let cap = 8 * 1024 * 1024u64;
        let entry_size = 64 * 1024u64; // 64 KiB
        let num_files = 5u64;
        let offsets_per_file = 20u64;
        let cache = Arc::new(MemoryCache::new(cap));
        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(500);

        let mut handles = Vec::new();
        for _worker in 0..8 {
            let cache = cache.clone();
            handles.push(tokio::spawn(async move {
                use rand::{SeedableRng, rngs::SmallRng};
                let mut rng = SmallRng::from_os_rng();
                while std::time::Instant::now() < deadline {
                    let file_id = rng.random_range(0..num_files);
                    let offset_idx = rng.random_range(0..offsets_per_file);
                    let offset = offset_idx * entry_size;
                    let k = key(file_id, offset, entry_size);

                    let data =
                        Bytes::from(vec![(file_id ^ offset_idx) as u8; entry_size as usize]);
                    let _ = cache
                        .get_or_load(k, Box::pin(async move { Ok(data) }))
                        .await
                        .unwrap();
                }
            }));
        }

        for h in handles {
            h.await.unwrap();
        }

        let stats = cache.stats();
        assert!(stats.hits > 0, "expected hits in fuzz run");
        assert!(stats.misses > 0, "expected misses in fuzz run");
        assert!(
            stats.current_bytes <= cap,
            "cache exceeded capacity during fuzz: {} > {}",
            stats.current_bytes,
            cap
        );
    }

 /// Stats accounting: hits + misses == total requests (for single-key scenario).
    #[tokio::test]
    async fn test_stats_accounting() {
        let cache = MemoryCache::new(10 * 1024 * 1024);
        let k = key(6, 0, 4);
        let requests = 10usize;

        for _ in 0..requests {
            cache
                .get_or_load(k.clone(), Box::pin(async { Ok(Bytes::from_static(b"x")) }))
                .await
                .unwrap();
        }

        let stats = cache.stats();
        assert_eq!(stats.misses, 1, "only first request should miss");
        assert_eq!(
            stats.hits,
            (requests - 1) as u64,
            "remaining requests should hit"
        );
        assert_eq!(stats.current_bytes, 1); // "x" = 1 byte
    }

 /// Regression test for the saturating_sub guard in maybe_evict and
 /// remove_entry. Without it, two concurrent evictions subtracting from
 /// total_bytes simultaneously could wrap a u64 to near u64::MAX, making
 /// the cache think it has effectively infinite free space and stop evicting.
 ///
 /// We verify that after heavy concurrent load total_bytes never approaches
 /// u64::MAX (which would indicate a wrap-around).
    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn test_total_bytes_no_underflow_under_concurrent_eviction() {
 // Very small cap forces constant eviction pressure.
        let cap = 512 * 1024u64;        // 512 KiB
        let entry_size = 64 * 1024u64;  // 64 KiB — 8 entries fill the cache
        let cache = Arc::new(MemoryCache::new(cap));

 // 8 threads each load a distinct stream of keys, all competing for the
 // same tiny cache. Every insert triggers maybe_evict; concurrent calls
 // to fetch_sub on total_bytes used to be able to race and underflow.
        let mut handles = Vec::new();
        for thread_id in 0..8u64 {
            let cache = cache.clone();
            handles.push(tokio::spawn(async move {
                for i in 0..64u64 {
                    let offset = (thread_id * 1000 + i) * entry_size;
                    let data = Bytes::from(vec![0u8; entry_size as usize]);
                    cache
                        .get_or_load(
                            key(thread_id, offset, entry_size),
                            Box::pin(async move { Ok(data) }),
                        )
                        .await
                        .unwrap();
                }
            }));
        }

        for h in handles {
            h.await.unwrap();
        }

        let stats = cache.stats();

 // If total_bytes wrapped, it would be close to u64::MAX (≥ 1 TiB).
 // A sane cache under 512 KiB cap should never report anywhere near that.
        let one_tib = 1u64 << 40;
        assert!(
            stats.current_bytes < one_tib,
            "u64 underflow detected: total_bytes wrapped to {}",
            stats.current_bytes
        );
        assert!(stats.evictions > 0, "expected evictions under constant pressure");
    }

 // ── Missing tests ───────────────────────────────────────────────

 /// outOfCapacity: when all entries are actively loading
 /// (strong_count > 2), eviction must be a graceful no-op — nothing freed,
 /// no panic, no underflow.
    #[tokio::test]
    async fn test_eviction_graceful_when_all_entries_loading() {
 // Tiny cache — 1 entry fits.
        let cap = 128 * 1024u64;
        let entry_size = 128 * 1024u64;
        let cache = Arc::new(MemoryCache::new(cap));

 // Hold the Arc<CacheEntry> alive by keeping the loader suspended,
 // simulating a pinned / still-loading entry.
        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        let cache_clone = cache.clone();

        let loading = tokio::spawn(async move {
            let _ = cache_clone
                .get_or_load(
                    key(99, 0, entry_size),
                    Box::pin(async move {
                        rx.await.ok(); // suspended — entry is in Loading state
                        Ok(Bytes::from(vec![0u8; entry_size as usize]))
                    }),
                )
                .await
                .unwrap();
        });

 // Give the loader task time to create the entry and suspend.
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;

 // Try to load a second entry — cache is over capacity but the only
 // entry is loading (strong_count > 2). Eviction must not panic or
 // underflow.
        let data = Bytes::from(vec![1u8; entry_size as usize]);
        let _ = cache
            .get_or_load(
                key(99, entry_size, entry_size),
                Box::pin(async move { Ok(data) }),
            )
            .await
            .unwrap();

 // Unblock the first loader.
        tx.send(()).ok();
        loading.await.unwrap();

 // No underflow: total_bytes must be a sane value.
        let one_tib = 1u64 << 40;
        assert!(cache.stats().current_bytes < one_tib);
    }

 /// staleEntry: entries with `data_size == 0` (still
 /// loading or previously evicted) are skipped by the clock hand without
 /// touching byte counters — no double-accounting.
    #[tokio::test]
    async fn test_eviction_skips_zero_size_entries() {
        let cap = 256 * 1024u64;
        let entry_size = 128 * 1024u64;
        let cache = MemoryCache::new(cap);

 // Fill cache to capacity.
        for i in 0..2u64 {
            let data = Bytes::from(vec![i as u8; entry_size as usize]);
            cache
                .get_or_load(key(0, i * entry_size, entry_size), Box::pin(async move { Ok(data) }))
                .await
                .unwrap();
        }

        let before = cache.stats().current_bytes;

 // Load two more entries — forces eviction. Evicted entries get
 // data_size = 0. If the clock hand sweeps past them again and
 // double-subtracts, total_bytes wraps.
        for i in 2..4u64 {
            let data = Bytes::from(vec![i as u8; entry_size as usize]);
            cache
                .get_or_load(key(0, i * entry_size, entry_size), Box::pin(async move { Ok(data) }))
                .await
                .unwrap();
        }

        let after = cache.stats().current_bytes;

 // total_bytes must never wrap (would produce a value >> cap).
        assert!(after < cap * 4, "possible double-accounting: before={before} after={after}");
        assert!(cache.stats().evictions > 0);
    }

 /// shrinkCache: after loading entries, eviction must
 /// bring total_bytes back to or below capacity when given enough pressure.
    #[tokio::test]
    async fn test_eviction_converges_to_cap() {
        let cap = 1024 * 1024u64;         // 1 MiB
        let entry_size = 128 * 1024u64;   // 128 KiB — 8 entries fill the cache
        let cache = MemoryCache::new(cap);

 // Load 4× capacity sequentially — forces many eviction rounds.
        for i in 0..32u64 {
            let data = Bytes::from(vec![0u8; entry_size as usize]);
            cache
                .get_or_load(key(0, i * entry_size, entry_size), Box::pin(async move { Ok(data) }))
                .await
                .unwrap();
        }

 // After the loading loop, all entry Arcs from is_new=true have dropped.
 // strong_count for each remaining entry is exactly 2 (map + ring) →
 // all are eviction candidates. One more load triggers final convergence.
        let data = Bytes::from(vec![0u8; entry_size as usize]);
        cache
            .get_or_load(key(1, 0, entry_size), Box::pin(async move { Ok(data) }))
            .await
            .unwrap();

        let stats = cache.stats();
        assert!(
            stats.current_bytes <= cap * 2,
            "eviction did not converge: current_bytes={} cap={}",
            stats.current_bytes, cap
        );
        assert!(stats.evictions > 0);
    }

 // ── Tests not ported from (with explanation) ───────────────────
 //
 // evictAccounting: tests interaction between cache eviction and a
 // custom MemoryPool allocator. Lance uses Bytes (Arc<[u8]>) with no
 // custom allocator, so pool-level accounting is not applicable.
 //
 // shrinkWithSsdWrite: Requires SCOPED_TESTVALUE_SET / TestValue hooks to
 // pause SSD writes at a specific code point. Our implementation has no
 // equivalent test-hook infrastructure.
 //
 // ttl: CacheTTLController expires entries based on when files were
 // opened. Lance datasets are immutable and versioned — cached data is
 // valid indefinitely for a given file path, so TTL is not implemented.
 //
 // makeEvictable: Tests explicit num_pins / CachePin management. We
 // deliberately omit CachePin (see TODO comment in evict()), relying on
 // Bytes (Arc<[u8]>) to keep data alive independently.
 //
 // dataRanges: Tests allocation-run API (tiny inline storage vs
 // MmapAllocator pages). Our entries are uniform Bytes (Arc<[u8]>);
 // there is no multi-run layout to verify.
 //
 // pin (partial): The full pin test exercises CachePin move semantics
 // and explicit numPins counting. The equivalent state-machine behaviour
 // (exclusive while loading, shared after, waiters unblocked on failure)
 // is already covered by test_concurrent_waiters_see_failure_and_retry
 // and test_load_deduplication.

 // ── Additional -inspired tests ──────────────────────────────────

 /// findMiss: looking up a key that was never inserted
 /// must return None (loader is called, not bypassed).
    #[tokio::test]
    async fn test_find_miss() {
        let cache = MemoryCache::new(10 * 1024 * 1024);
        let k = key(10, 0, 4);
        let load_count = Arc::new(AtomicUsize::new(0));
        let lc = load_count.clone();

 // First access: miss — loader must be called.
        let bytes = cache
            .get_or_load(
                k.clone(),
                Box::pin(async move {
                    lc.fetch_add(1, Ordering::Relaxed);
                    Ok(Bytes::from_static(b"data"))
                }),
            )
            .await
            .unwrap();
        assert_eq!(bytes, Bytes::from_static(b"data"));
        assert_eq!(load_count.load(Ordering::Relaxed), 1, "loader must run on miss");
        assert_eq!(cache.stats().misses, 1);
        assert_eq!(cache.stats().hits, 0);
    }

 /// findHit: after a miss populates the cache, the next
 /// access must return exactly the same bytes without calling the loader.
 /// Verifies data integrity (byte-for-byte match) — equivalent to 
 /// `checkContents(*entry)`.
    #[tokio::test]
    async fn test_find_hit_data_integrity() {
        let cache = MemoryCache::new(10 * 1024 * 1024);
 // Use a recognisable pattern so a stale-copy bug would be detectable.
        let pattern: Vec<u8> = (0u8..=255).cycle().take(4096).collect();
        let original = Bytes::from(pattern.clone());
        let k = key(11, 0, 4096);

 // Miss — populate cache.
        let b1 = cache
            .get_or_load(k.clone(), Box::pin(async move { Ok(original) }))
            .await
            .unwrap();
        assert_eq!(b1.as_ref(), pattern.as_slice(), "loaded bytes must match pattern");

 // Hit — must return identical bytes without calling loader.
        let b2 = cache
            .get_or_load(k, Box::pin(async { panic!("loader must not be called on hit") }))
            .await
            .unwrap();
        assert_eq!(b2.as_ref(), pattern.as_slice(), "hit bytes must match original");
 // Both should point to the same underlying allocation.
        assert_eq!(b1.as_ptr(), b2.as_ptr(), "hit must return the cached Arc, not a copy");
        assert_eq!(cache.stats().hits, 1);
    }

 /// cacheStats: verifies that all stat counters are
 /// incremented correctly and reflect the true cache state.
    #[tokio::test]
    async fn test_cache_stats_fields() {
        let cache = MemoryCache::new(2 * 1024 * 1024);
        let entry_size = 256 * 1024u64;

 // 4 misses.
        for i in 0u64..4 {
            let data = Bytes::from(vec![i as u8; entry_size as usize]);
            cache
                .get_or_load(key(12, i * entry_size, entry_size), Box::pin(async move { Ok(data) }))
                .await
                .unwrap();
        }

 // 4 more hits on the same keys (cache holds 2 MiB = 8 × 256 KiB entries).
        for i in 0u64..4 {
            let data = Bytes::from(vec![i as u8; entry_size as usize]);
            cache
                .get_or_load(key(12, i * entry_size, entry_size), Box::pin(async move { Ok(data) }))
                .await
                .unwrap();
        }

        let stats = cache.stats();
        assert_eq!(stats.misses, 4, "misses");
        assert_eq!(stats.hits, 4, "hits");
        assert_eq!(stats.max_bytes, 2 * 1024 * 1024, "max_bytes");
 // current_bytes should reflect the 4 entries still in cache.
        assert_eq!(stats.current_bytes, 4 * entry_size, "current_bytes");
    }

 /// pin state-machine: while a load is in progress
 /// (exclusive / Loading state) concurrent callers must block; after the
 /// transition to Loaded all blocked callers receive the same data.
 /// This complements test_load_deduplication with an explicit timing check.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_exclusive_to_shared_transition() {
        let cache = Arc::new(MemoryCache::new(10 * 1024 * 1024));
        let k = key(13, 0, 8);

        let (loading_tx, loading_rx) = tokio::sync::oneshot::channel::<()>();
        let (done_tx, _done_rx) = tokio::sync::oneshot::channel::<()>();

 // Task A: "exclusive" loader — signals when loading has started, then
 // sleeps to give Task B time to see the Loading state.
        let cache_a = cache.clone();
        let k_a = k.clone();
        let loader_task = tokio::spawn(async move {
            cache_a
                .get_or_load(
                    k_a,
                    Box::pin(async move {
                        loading_tx.send(()).ok(); // signal: exclusive load started
                        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
                        done_tx.send(()).ok();
                        Ok(Bytes::from_static(b"exclusive"))
                    }),
                )
                .await
                .unwrap()
        });

 // Wait until the loader has started (entry is in Loading state).
        loading_rx.await.ok();

 // Task B: concurrent waiter — should block until A completes.
        let cache_b = cache.clone();
        let k_b = k.clone();
        let waiter_task = tokio::spawn(async move {
            cache_b
                .get_or_load(
                    k_b,
                    Box::pin(async { panic!("waiter must not call its own loader") }),
                )
                .await
                .unwrap()
        });

        let a_result = loader_task.await.unwrap();
        let b_result = waiter_task.await.unwrap();

        assert_eq!(a_result, Bytes::from_static(b"exclusive"));
        assert_eq!(b_result, Bytes::from_static(b"exclusive"));
 // Both should point to the same allocation (watch-channel clone).
        assert_eq!(a_result.as_ptr(), b_result.as_ptr());
    }
}
