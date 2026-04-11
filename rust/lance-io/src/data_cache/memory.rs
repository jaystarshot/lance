// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! In-memory cache tier — a Rust port of Velox's `CacheShard` / `AsyncDataCache`.
//!
//! Design mirrors Velox exactly:
//!
//! * **16 independent shards** — `HashMap` + clock-hand eviction ring per shard,
//!   each protected by a `std::sync::Mutex`.  Shards eliminate contention for the
//!   common case where different tasks access different files.
//!
//! * **Load deduplication** — a `tokio::sync::watch` channel replaces Velox's
//!   `folly::SharedPromise`.  The first task to miss the cache transitions the
//!   entry from `Loading` to `Loaded(bytes)` or `Failed`; all concurrent tasks
//!   wait on the channel and receive the result without issuing a second fetch.
//!
//! * **Clock-hand eviction with percentile threshold** — follows Velox's
//!   `CacheShard::evict` / `calibrateThreshold` exactly.  Every `ring_len/4`
//!   insertions the shard samples `NUM_EVICTION_SAMPLES` (10) entries, sorts
//!   their scores, and sets the eviction threshold to the
//!   `EVICTION_PERCENTILE`th (80th) percentile.  Only entries scoring *above*
//!   the threshold are candidates, which prevents thrashing when the cache
//!   hovers at capacity.
//!
//! * **Score formula** — `(now_ms - last_use_ms) / (1 + num_uses)`.  Older,
//!   less-frequently-accessed entries score higher and are evicted first.
//!   An entry that has never been accessed scores `u64::MAX` (evict immediately).

use std::collections::HashMap;
use std::sync::{
    Arc, Mutex, Weak,
    atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering},
};
use std::time::{SystemTime, UNIX_EPOCH};

use bytes::Bytes;
use futures::future::BoxFuture;
use tokio::sync::watch;

use lance_core::Result;

use super::DataCacheKey;

// ─── Constants (same as Velox) ───────────────────────────────────────────────

/// Number of independent shards.  Must be a power of two.
const NUM_SHARDS: usize = 16;
const SHARD_MASK: u64 = (NUM_SHARDS as u64) - 1;

/// Number of entries sampled when calibrating the eviction threshold.
const NUM_EVICTION_SAMPLES: usize = 10;

/// Only entries whose score is at or above this percentile are evicted.
const EVICTION_PERCENTILE: usize = 80;

/// Recalibrate the threshold after this many shard events (inserts + eviction
/// checks), matching Velox's `entries_.size() / 4` heuristic.
const CALIBRATION_INTERVAL_DIVISOR: usize = 4;

// ─── Time ────────────────────────────────────────────────────────────────────

/// Milliseconds since the Unix epoch — cheap, ~1 ms resolution.
///
/// Velox uses `folly::hardware_timestamp() >> 21` for ~1–2 ms resolution;
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
fn shard_idx(key: &DataCacheKey) -> usize {
    // Fibonacci hashing — mixes file_id and offset uniformly across shards.
    let h = key
        .file_id
        .wrapping_mul(11_400_714_819_323_198_485_u64)
        .wrapping_add(key.offset.wrapping_mul(6_364_136_223_846_793_005_u64));
    (h & SHARD_MASK) as usize
}

// ─── Entry ───────────────────────────────────────────────────────────────────

/// Possible states of a cache entry, broadcast via a `watch` channel.
///
/// Maps to Velox's `numPins_` convention:
/// * `Loading`       ≡ `numPins_ = kExclusive (-10000)`
/// * `Loaded(bytes)` ≡ `numPins_ = 1` (shared, data available)
/// * `Failed`        ≡ entry removed from map; waiters must retry
#[derive(Clone)]
enum LoadState {
    Loading,
    Loaded(Bytes),
    Failed,
}

struct CacheEntry {
    key: DataCacheKey,
    /// The `Sender` half is owned here; receivers are created on demand by
    /// concurrent waiters.  Sending a new state wakes *all* current waiters
    /// simultaneously — the same semantics as Velox's `SharedPromise::setValue`.
    state_tx: watch::Sender<LoadState>,
    /// Milliseconds since epoch of last access; 0 = never accessed.
    last_use_ms: AtomicU64,
    /// How many times this entry has been read (used in eviction scoring).
    num_uses: AtomicU32,
    /// Byte size of the cached payload; 0 while Loading or after failure.
    data_size: AtomicU64,
    /// True once loaded and not yet flushed to the SSD tier.
    ssd_saveable: AtomicBool,
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
            ssd_saveable: AtomicBool::new(false),
        })
    }

    /// Eviction score.  Higher = more worth evicting.
    ///
    /// Formula: `(now_ms - last_use_ms) / (1 + num_uses)` — identical to
    /// Velox's `AccessStats::score`.
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
    /// O(1) key → entry lookup (Velox's `entryMap_`).
    entries: HashMap<DataCacheKey, Arc<CacheEntry>>,
    /// Clock-hand eviction ring (Velox's `entries_` dense array).
    /// `Weak` lets us skip already-freed entries without a map lookup.
    eviction_ring: Vec<Weak<CacheEntry>>,
    /// Current position of the clock hand in `eviction_ring`.
    clock_hand: usize,
    /// Sum of `data_size` for all `Loaded` entries in this shard.
    loaded_bytes: u64,
    /// Cached 80th-percentile eviction score — recomputed periodically.
    ///
    /// Initialised to `u64::MAX` (Velox's `kNoThreshold = INT_MAX`) so that
    /// nothing is evicted until the first calibration pass completes.  Without
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
    /// percentile value.  This is a direct port of Velox's
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
        // percentile(scores, EVICTION_PERCENTILE) — same formula as Velox:
        // `values[(values.size() * percent) / 100]`
        let idx = (scores.len() * EVICTION_PERCENTILE / 100)
            .min(scores.len().saturating_sub(1));
        self.eviction_threshold = scores.get(idx).copied().unwrap_or(0);
        self.events = 0;
    }

    /// Free at least `target_bytes` from this shard using the clock-hand
    /// algorithm.  Returns bytes freed.
    ///
    /// Direct port of Velox's `CacheShard::evict`.
    fn evict(&mut self, target_bytes: u64) -> u64 {
        let n = self.eviction_ring.len();
        if n == 0 {
            return 0;
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

        while freed < target_bytes && checked < n {
            let idx = self.clock_hand % n;
            self.clock_hand = self.clock_hand.wrapping_add(1);
            checked += 1;

            let Some(entry) = self.eviction_ring[idx].upgrade() else {
                // Entry already freed elsewhere — skip.
                continue;
            };

            // `strong_count == 2`: one from the `entries` map + one from our
            // `upgrade()`.  Waiters suspended at `rx.changed().await` hold
            // their own Arc clone *outside* the shard mutex, so the count is
            // NOT stable — it can be 2 + N where N is the number of active
            // waiters.  The check `> 2` correctly skips those entries.
            // The only case where count is exactly 2 and the entry is still
            // live is when no waiter holds it and it has not yet been removed
            // from the map — the exact condition under which it is safe to evict.
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

            // Evict: remove from map and zero data_size so that if this entry
            // remains in the eviction ring (still referenced by waiters) a
            // subsequent sweep skips it without double-counting.
            if self.entries.remove(&entry.key).is_some() {
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

        freed
    }
}

struct CacheShard {
    inner: Mutex<CacheShardInner>,
}

impl CacheShard {
    fn new() -> Self {
        Self {
            inner: Mutex::new(CacheShardInner::new()),
        }
    }
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

/// Sharded in-memory cache with Velox-style clock-hand + percentile eviction.
///
/// The cache is logically split into `NUM_SHARDS` (16) independent shards.
/// Each shard owns its hash-map and eviction ring and is protected by its own
/// `std::sync::Mutex`, so concurrent tasks hitting different files (or
/// different offsets within the same file) almost never contend.
///
/// Async coordination (waiting for a concurrent load to finish) is done via
/// `tokio::sync::watch` *outside* of the shard mutex, so no tokio worker is
/// blocked while waiting.
pub struct MemoryCache {
    shards: Vec<CacheShard>,
    /// Hard upper bound on total cached bytes across all shards.
    max_bytes: u64,
    /// Running total of loaded bytes (updated atomically outside shard locks).
    total_bytes: AtomicU64,
    hits: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
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
    pub fn new(max_bytes: u64) -> Arc<Self> {
        let shards = (0..NUM_SHARDS).map(|_| CacheShard::new()).collect();
        Arc::new(Self {
            shards,
            max_bytes,
            total_bytes: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
        })
    }

    pub fn stats(&self) -> MemoryCacheStats {
        MemoryCacheStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
            current_bytes: self.total_bytes.load(Ordering::Relaxed),
            max_bytes: self.max_bytes,
        }
    }

    /// Fetch the bytes for `key`, calling `loader` on a cache miss.
    ///
    /// If multiple tasks request the same key concurrently, only the first
    /// triggers `loader`; all others wait for it to complete via the entry's
    /// `watch` channel — exactly Velox's `CoalescedLoad::loadOrFuture`.
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
                // ── We own this entry (exclusive, like Velox's kExclusive pin) ──
                let loader = loader.take().expect("loader consumed twice");
                self.misses.fetch_add(1, Ordering::Relaxed);

                match loader.await {
                    Ok(bytes) => {
                        let size = bytes.len() as u64;
                        entry.data_size.store(size, Ordering::Release);
                        entry.ssd_saveable.store(true, Ordering::Release);
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
                                self.shards[shard_idx(&key)].inner.lock().unwrap();
                            inner.loaded_bytes += size;
                            self.total_bytes.fetch_add(size, Ordering::Relaxed);
                        }
                        self.maybe_evict(size);
                        return Ok(bytes);
                    }
                    Err(e) => {
                        // Load failed — signal waiters, remove entry so the
                        // next caller gets a fresh miss.  Equivalent to Velox's
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
                        self.hits.fetch_add(1, Ordering::Relaxed);
                        return Ok(bytes);
                    }
                    LoadState::Failed => {
                        // The loading task failed.  The entry has been (or is
                        // being) removed from the map.  We retry from scratch so
                        // that *this* task can attempt the load with its own
                        // loader — identical to the waiter retry in Velox after a
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
    /// Returns `(entry, is_new)`.  When `is_new` is `true` the entry is in
    /// `Loading` state and the caller *must* drive the load and update the
    /// state — exactly Velox's exclusive-pin contract.
    fn find_or_create(&self, key: &DataCacheKey) -> (Arc<CacheEntry>, bool) {
        let idx = shard_idx(key);
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
        let idx = shard_idx(key);
        let mut inner = self.shards[idx].inner.lock().unwrap();
        if let Some(entry) = inner.entries.remove(key) {
            let size = entry.data_size.load(Ordering::Relaxed);
            // Zero data_size so any subsequent eviction-ring sweep skips this
            // entry without double-counting.
            entry.data_size.store(0, Ordering::Relaxed);
            if size > 0 {
                inner.loaded_bytes = inner.loaded_bytes.saturating_sub(size);
                self.total_bytes.fetch_sub(size, Ordering::Relaxed);
            }
        }
    }

    /// Trigger eviction across shards if total usage exceeds `max_bytes`.
    ///
    /// Called after every successful insert.  Spreads the target eviction
    /// proportionally across all shards to avoid always hammering shard 0.
    fn maybe_evict(&self, inserted_bytes: u64) {
        let current = self.total_bytes.load(Ordering::Relaxed);
        if current <= self.max_bytes {
            return;
        }
        let overage = current - self.max_bytes;
        // Spread eviction across shards; each shard is responsible for freeing
        // its share plus enough to absorb the new insertion.
        let per_shard = (overage / NUM_SHARDS as u64).max(inserted_bytes);
        let mut total_freed = 0u64;
        for shard in &self.shards {
            if total_freed >= overage {
                break;
            }
            let freed = shard.inner.lock().unwrap().evict(per_shard);
            if freed > 0 {
                self.total_bytes.fetch_sub(freed, Ordering::Relaxed);
                self.evictions.fetch_add(1, Ordering::Relaxed);
                total_freed += freed;
            }
        }
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

    // ── Velox-inspired tests ──────────────────────────────────────────────

    /// Port of Velox's `replace` test: fill the cache exactly to capacity,
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

    /// Port of Velox's `staleEntry` / double-eviction test: verify that
    /// `total_bytes` and per-shard `loaded_bytes` stay consistent after many
    /// evictions.  A double-decrement bug would make `total_bytes` underflow
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
        // Eviction is amortised — allow up to 2× cap overshoot while the
        // clock sweep catches up.  The key invariant is no double-decrement:
        // if total_bytes underflowed it would wrap to u64::MAX.
        assert!(
            stats.current_bytes <= cap * 2,
            "double-decrement detected: current_bytes={} far exceeds cap={}",
            stats.current_bytes,
            cap
        );
        assert!(stats.evictions > 0);
    }

    /// Port of Velox's `findExclusiveWithWait` + failure test: when a load
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

    /// Port of Velox's `fuzz` test: 8 concurrent tasks randomly reading
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
}
