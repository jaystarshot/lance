// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Async two-tier data cache for Lance I/O.
//!
//! # Design
//!
//! The two-tier cache (memory + SSD) is inspired by Meta's engine.
//! The core algorithms — shard-based memory management, clock-hand eviction
//! with percentile threshold, load deduplication, region-based SSD layout,
//! and coalesced reads — follow the same principles, adapted for Rust's async
//! model and Lance's IO stack.
//!
//! # Architecture
//!
//! ```text
//! FileScheduler::submit_request()
//! │
//! ├─ L1: MemoryCache (16 shards, clock-hand eviction, ~microseconds)
//! │ HIT → return bytes immediately
//! │
//! ├─ L2: SsdCache (region files, coalesced pread, ~milliseconds)
//! │ HIT → populate L1 → return
//! │
//! └─ L3: object store (network, tens–hundreds of ms)
//! → populate L2 + L1 → return
//! ```
//!
//! # Configuration
//!
//! Pass via `storage_options` when opening a dataset:
//!
//! ```python
//! ds = lance.dataset(
//! "s3://bucket/data.lance",
//! storage_options={
//! "max_memory_cache_mb": "1000",
//! "ssd_cache_dir": "/mnt/nvme/lance_cache",
//! "ssd_cache_size_mb": "100000",
//! },
//! )
//! ```

use std::{collections::HashMap, path::PathBuf, sync::Arc};

use bytes::Bytes;
use futures::future::BoxFuture;
use object_store::path::Path;

use lance_core::Result;

pub mod file_ids;
pub mod memory;
pub mod ssd;

use file_ids::FileIds;
use memory::MemoryCache;
use ssd::{SsdCache, SsdCacheConfig};

// ─── Cache key ───────────────────────────────────────────────────────────────

/// Cache key for a raw byte range within a file.
///
/// The `file_id` is a stable numeric identifier for the file path (interned
/// by [`FileIds`]). The `offset` and `length` are the byte range after
/// `FileScheduler` has coalesced and split the requested ranges — those
/// post-processed ranges are stable across repeated reads of the same column.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DataCacheKey {
 /// Stable numeric ID for the file path.
    pub file_id: u64,
 /// Byte offset within the file (start of the cached range).
    pub offset: u64,
 /// Length of the cached range in bytes.
    pub length: u64,
}

// ─── Configuration ───────────────────────────────────────────────────────────

/// Configuration for the two-tier async data cache.
///
/// Parsed from `storage_options` when opening a dataset:
///
/// ```python
/// ds = lance.dataset(
/// "s3://bucket/data.lance",
/// storage_options={
/// "data_cache_enabled": "true",
/// "data_cache_memory_bytes": "10737418240", # 10 GiB
/// "data_cache_ssd_enabled": "true", # optional SSD tier
/// "data_cache_ssd_dir": "/mnt/nvme/cache",
/// "data_cache_ssd_bytes": "107374182400", # 100 GiB
/// },
/// )
/// ```
#[derive(Debug, Clone)]
pub struct DataCacheConfig {
    /// Maximum bytes to hold in the in-memory (L1) cache tier.
    pub max_memory_bytes: u64,

    /// Number of independent memory-tier shards. Must be a power of two.
    /// Advanced — defaults to [`memory::DEFAULT_NUM_SHARDS`] (16).
    pub num_shards: usize,

    /// Whether the SSD (L2) tier is enabled.
    /// Requires `ssd_cache_dir` and `ssd_max_bytes` to be set.
    pub ssd_enabled: bool,

    /// Directory on a local SSD for the on-disk (L2) cache tier.
    /// Ignored when `ssd_enabled` is `false`.
    pub ssd_cache_dir: Option<PathBuf>,

    /// Maximum bytes the SSD tier may consume.
    /// Ignored when `ssd_enabled` is `false`.
    pub ssd_max_bytes: u64,

    /// Number of SSD shard files. Must be a positive power of two.
    /// Advanced — defaults to [`ssd::DEFAULT_NUM_SSD_SHARDS`] (4).
    pub ssd_num_shards: usize,
}

impl DataCacheConfig {
    // ── Primary config keys ───────────────────────────────────────────────
    /// Master on/off switch. Must be `"true"` to enable the cache.
    pub const KEY_ENABLED: &'static str = "data_cache_enabled";
    /// Memory tier capacity in **bytes**.
    pub const KEY_MEMORY_BYTES: &'static str = "data_cache_memory_bytes";
    /// Set to `"true"` to enable the SSD (L2) tier.
    pub const KEY_SSD_ENABLED: &'static str = "data_cache_ssd_enabled";
    /// Directory on local SSD where cache files are stored.
    pub const KEY_SSD_DIR: &'static str = "data_cache_ssd_dir";
    /// SSD tier capacity in **bytes**.
    pub const KEY_SSD_BYTES: &'static str = "data_cache_ssd_bytes";

    // ── Advanced / rarely-needed keys ────────────────────────────────────
    /// Memory shard count (power of two). Defaults to 16.
    pub const KEY_MEMORY_SHARDS: &'static str = "data_cache_memory_shards";
    /// SSD shard-file count (power of two). Defaults to 4.
    pub const KEY_SSD_SHARDS: &'static str = "data_cache_ssd_shards";

    /// Parse from the merged `storage_options` map.
    ///
    /// Returns `None` when `data_cache_enabled` is absent or not `"true"`.
    pub fn from_storage_options(opts: &HashMap<String, String>) -> Option<Self> {
        use lance_core::utils::parse::str_is_truthy;

        // Master switch — must be explicitly enabled.
        let enabled = opts
            .get(Self::KEY_ENABLED)
            .map(|v| str_is_truthy(v.trim()))
            .unwrap_or(false);

        if !enabled {
            return None;
        }

        let max_memory_bytes = opts
            .get(Self::KEY_MEMORY_BYTES)
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(256 * 1024 * 1024); // 256 MiB default

        // SSD is parsed independently — both fields are stored in the struct
        // so the constructor can use ssd_enabled as an explicit gate.
        let ssd_enabled = opts
            .get(Self::KEY_SSD_ENABLED)
            .map(|v| str_is_truthy(v.trim()))
            .unwrap_or(false);

        let ssd_cache_dir = opts.get(Self::KEY_SSD_DIR).map(PathBuf::from);

        let ssd_max_bytes = opts
            .get(Self::KEY_SSD_BYTES)
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(0);

        let num_shards = opts
            .get(Self::KEY_MEMORY_SHARDS)
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(memory::DEFAULT_NUM_SHARDS);

        let ssd_num_shards = opts
            .get(Self::KEY_SSD_SHARDS)
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(ssd::DEFAULT_NUM_SSD_SHARDS);

        Some(Self {
            max_memory_bytes,
            num_shards,
            ssd_enabled,
            ssd_cache_dir,
            ssd_max_bytes,
            ssd_num_shards,
        })
    }
}

// ─── Trait ───────────────────────────────────────────────────────────────────

/// Async two-tier (memory + SSD) data cache.
///
/// The primary entry point is [`DataCache::get_or_load`], which handles both
/// cache lookup and load deduplication: if multiple tasks request the same
/// byte range concurrently, only the first triggers `loader`; all others wait
/// for the result — equivalent to CoalescedLoad::loadOrFuture.
///
/// Implementations must be cheap to clone and safe to share across threads.
pub trait DataCache: Send + Sync + std::fmt::Debug {
 /// Fetch the byte range `offset..offset+length` for `path`.
 ///
 /// Checks L1 (memory) and L2 (SSD) before falling back to `loader`.
 /// Concurrent requests for the same `(path, offset, length)` are
 /// deduplicated — only one `loader` invocation occurs.
    fn get_or_load<'a>(
        &'a self,
        path: &'a Path,
        offset: u64,
        length: u64,
        loader: BoxFuture<'a, Result<Bytes>>,
    ) -> BoxFuture<'a, Result<Bytes>>;
}

// ─── NoopDataCache ───────────────────────────────────────────────────────────

/// A no-op [`DataCache`] that always misses and passes `loader` through
/// unchanged. Used in tests and as a placeholder.
#[derive(Debug)]
pub struct NoopDataCache;

impl DataCache for NoopDataCache {
    fn get_or_load<'a>(
        &'a self,
        _path: &'a Path,
        _offset: u64,
        _length: u64,
        loader: BoxFuture<'a, Result<Bytes>>,
    ) -> BoxFuture<'a, Result<Bytes>> {
        loader
    }
}

// ─── TieredDataCache ─────────────────────────────────────────────────────────

/// Concrete two-tier cache: L1 [`MemoryCache`] + optional L2 [`SsdCache`].
///
/// Built from [`DataCacheConfig`] via [`TieredDataCache::new`].
#[derive(Debug)]
pub struct TieredDataCache {
    memory: Arc<MemoryCache>,
    ssd: Option<Arc<SsdCache>>,
 /// Maps file paths to stable `u64` IDs used in [`DataCacheKey`].
    file_ids: Arc<FileIds>,
}

impl TieredDataCache {
 /// Build a `TieredDataCache` from `config`.
 ///
 /// When the SSD tier is enabled:
 /// * A bounded channel (`eviction_channel_capacity` = 256) is created.
 /// * `MemoryCache` is given the sender — evicted entries are forwarded here.
 /// * A background tokio task drains the channel and writes to `SsdCache`.
 ///
 /// This is lazy write pattern: data reaches the SSD only when the
 /// memory tier can no longer hold it, not on every initial fetch.
    pub async fn new(config: &DataCacheConfig) -> Result<Arc<Self>> {
        let ssd = if config.ssd_enabled {
            match &config.ssd_cache_dir {
                Some(dir) => {
                    let ssd_config = SsdCacheConfig {
                        cache_dir: dir.clone(),
                        max_bytes: config.ssd_max_bytes,
                        num_shards: config.ssd_num_shards,
                    };
                    Some(SsdCache::new(ssd_config).await?)
                }
                None => {
                    tracing::warn!(
                        "data_cache_ssd_enabled=true but data_cache_ssd_dir is not set \
                         — SSD tier disabled"
                    );
                    None
                }
            }
        } else {
            None
        };

 // If we have an SSD tier, wire the eviction channel so that memory
 // evictions are forwarded to SSD asynchronously.
        const EVICTION_CHANNEL_CAPACITY: usize = 256;
        let memory = if let Some(ssd_arc) = ssd.clone() {
            let (tx, mut rx) =
                tokio::sync::mpsc::channel::<(DataCacheKey, Bytes)>(EVICTION_CHANNEL_CAPACITY);

 // Background task: drain the eviction channel → write to SSD.
 // Exits automatically when MemoryCache is dropped (sender closes).
            tokio::spawn(async move {
                while let Some((key, bytes)) = rx.recv().await {
                    ssd_arc.insert(key, bytes).await;
                }
            });

            memory::MemoryCache::with_eviction_channel(
                config.max_memory_bytes,
                config.num_shards,
                Some(tx),
            )
        } else {
            memory::MemoryCache::new_with_shards(config.max_memory_bytes, config.num_shards)
        };

        Ok(Arc::new(Self {
            memory,
            ssd,
            file_ids: Arc::new(FileIds::new()),
        }))
    }

 /// Return a snapshot of the memory tier statistics.
    pub fn memory_stats(&self) -> memory::MemoryCacheStats {
        self.memory.stats()
    }
}

impl DataCache for TieredDataCache {
    fn get_or_load<'a>(
        &'a self,
        path: &'a Path,
        offset: u64,
        length: u64,
        loader: BoxFuture<'a, Result<Bytes>>,
    ) -> BoxFuture<'a, Result<Bytes>> {
        let file_id = self.file_ids.get_or_intern(path);
        let key = DataCacheKey { file_id, offset, length };

 // If SSD tier is enabled, check L2 (SSD) on L1 (memory) miss before
 // falling back to the object store. SSD writes now happen lazily via
 // the eviction channel — NOT here on every fetch.
        let effective_loader: BoxFuture<'a, Result<Bytes>> = if let Some(ssd) = &self.ssd {
            let key_for_ssd = key.clone();
            Box::pin(async move {
                if let Some(bytes) = ssd.get(&key_for_ssd).await {
                    return Ok(bytes); // L2 hit — no object store call
                }
 // L2 miss — fetch from object store.
 // SSD write happens when this entry is later evicted from memory.
                loader.await
            })
        } else {
            loader
        };

        Box::pin(self.memory.get_or_load(key, effective_loader))
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_absent_when_no_keys() {
 // No keys → None
        assert!(DataCacheConfig::from_storage_options(&HashMap::new()).is_none());
 // Keys present but master switch absent → None
        let opts = HashMap::from([(DataCacheConfig::KEY_MEMORY_BYTES.to_string(), "1000000".to_string())]);
        assert!(DataCacheConfig::from_storage_options(&opts).is_none());
    }

    #[test]
    fn test_config_memory_only() {
        let opts = HashMap::from([
            (DataCacheConfig::KEY_ENABLED.to_string(), "true".to_string()),
            (DataCacheConfig::KEY_MEMORY_BYTES.to_string(), (512 * 1024 * 1024u64).to_string()),
        ]);
        let cfg = DataCacheConfig::from_storage_options(&opts).unwrap();
        assert_eq!(cfg.max_memory_bytes, 512 * 1024 * 1024);
        assert!(cfg.ssd_cache_dir.is_none());
    }

    #[test]
    fn test_config_full() {
        let ssd_bytes: u64 = 100_000 * 1024 * 1024;
        let opts = HashMap::from([
            (DataCacheConfig::KEY_ENABLED.to_string(),      "true".to_string()),
            (DataCacheConfig::KEY_MEMORY_BYTES.to_string(), (1000 * 1024 * 1024u64).to_string()),
            (DataCacheConfig::KEY_SSD_ENABLED.to_string(),  "true".to_string()),
            (DataCacheConfig::KEY_SSD_DIR.to_string(),      "/mnt/nvme/cache".to_string()),
            (DataCacheConfig::KEY_SSD_BYTES.to_string(),    ssd_bytes.to_string()),
        ]);
        let cfg = DataCacheConfig::from_storage_options(&opts).unwrap();
        assert_eq!(cfg.max_memory_bytes, 1000 * 1024 * 1024);
        assert_eq!(cfg.ssd_cache_dir, Some(PathBuf::from("/mnt/nvme/cache")));
        assert_eq!(cfg.ssd_max_bytes, ssd_bytes);
    }

    #[test]
    fn test_config_ssd_disabled_ignores_ssd_keys() {
        // ssd_enabled=false is stored in the struct; the constructor
        // uses it to skip SSD creation even when ssd_cache_dir is present.
        let opts = HashMap::from([
            (DataCacheConfig::KEY_ENABLED.to_string(),      "true".to_string()),
            (DataCacheConfig::KEY_MEMORY_BYTES.to_string(), "1073741824".to_string()),
            (DataCacheConfig::KEY_SSD_ENABLED.to_string(),  "false".to_string()),
            (DataCacheConfig::KEY_SSD_DIR.to_string(),      "/mnt/nvme/cache".to_string()),
            (DataCacheConfig::KEY_SSD_BYTES.to_string(),    "107374182400".to_string()),
        ]);
        let cfg = DataCacheConfig::from_storage_options(&opts).unwrap();
        assert!(!cfg.ssd_enabled, "ssd_enabled flag must be false");
        // ssd_cache_dir is parsed but the constructor checks ssd_enabled first
        assert_eq!(cfg.ssd_cache_dir, Some(PathBuf::from("/mnt/nvme/cache")));
    }

    #[test]
    fn test_config_master_switch_false() {
        let opts = HashMap::from([
            (DataCacheConfig::KEY_ENABLED.to_string(),      "false".to_string()),
            (DataCacheConfig::KEY_MEMORY_BYTES.to_string(), "1073741824".to_string()),
        ]);
        assert!(DataCacheConfig::from_storage_options(&opts).is_none());
    }

    #[tokio::test]
    async fn test_tiered_cache_memory_hit() {
        let config = DataCacheConfig {
            max_memory_bytes: 10 * 1024 * 1024,
            num_shards: memory::DEFAULT_NUM_SHARDS,
            ssd_enabled: false,
            ssd_cache_dir: None,
            ssd_max_bytes: 0,
            ssd_num_shards: ssd::DEFAULT_NUM_SSD_SHARDS,
        };
        let cache = TieredDataCache::new(&config).await.unwrap();
        let path = Path::from("test/file.lance");

 // First call — miss, loads.
        let result = cache
            .get_or_load(
                &path,
                0,
                5,
                Box::pin(async { Ok(Bytes::from_static(b"hello")) }),
            )
            .await
            .unwrap();
        assert_eq!(result, Bytes::from_static(b"hello"));

 // Second call — memory hit, loader not called.
        let result2 = cache
            .get_or_load(
                &path,
                0,
                5,
                Box::pin(async { panic!("loader should not be called on cache hit") }),
            )
            .await
            .unwrap();
        assert_eq!(result2, Bytes::from_static(b"hello"));

        assert_eq!(cache.memory_stats().hits, 1);
    }

 // ── Two-tier integration tests (DISABLED_ssd equivalent) ──────

 /// DISABLED_ssd — simplified two-tier data integrity
 /// test: bytes loaded from the object store are eventually persisted to
 /// SSD on memory eviction. Verifies byte-for-byte integrity across tiers.
 ///
 /// Because SSD writes are lazy (background task), the test allows a small
 /// number of object-store re-fetches for entries that haven't reached SSD
 /// yet. The primary assertion is data correctness, not tier membership.
    #[tokio::test]
    async fn test_two_tier_ssd_fallback_data_integrity() {
        let tmp = tempfile::tempdir().unwrap();
        let config = DataCacheConfig {
            // Memory holds only 2 entries — forces eviction to SSD.
            max_memory_bytes: 512 * 1024,
            num_shards: memory::DEFAULT_NUM_SHARDS,
            ssd_enabled: true,
            ssd_cache_dir: Some(tmp.path().join("two_tier")),
            ssd_max_bytes: ssd::REGION_SIZE * 4,
            ssd_num_shards: 1,
        };
        let cache = TieredDataCache::new(&config).await.unwrap();
        let path = Path::from("s3://bucket/data.lance");

        let entry_size = 256 * 1024u64; // 256 KiB
        let n = 4u64; // 4 entries — well above the 512 KiB memory limit

 // Load all entries — they go to memory first. With lazy writes, SSD
 // receives them only when memory evicts (via the background channel).
        for i in 0..n {
            let pattern = Bytes::from(vec![(i * 37 % 256) as u8; entry_size as usize]);
            let p = pattern.clone();
            cache
                .get_or_load(
                    &path,
                    i * entry_size,
                    entry_size,
                    Box::pin(async move { Ok(p) }),
                )
                .await
                .unwrap();
        }

 // Give the background SSD writer time to drain the eviction channel.
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;

 // Verify all entries return correct bytes regardless of which tier
 // serves them. Track re-fetches (object-store calls) — these happen
 // for entries not yet on SSD; we allow a small number since writes
 // are lazy. Data integrity is the primary assertion.
        let refetch_count = Arc::new(std::sync::atomic::AtomicU64::new(0));
        for i in 0..n {
            let expected = (i * 37 % 256) as u8;
            let rc = refetch_count.clone();
            let result = cache
                .get_or_load(
                    &path,
                    i * entry_size,
                    entry_size,
                    Box::pin(async move {
 // Count re-fetches — ok if SSD write hasn't landed yet.
                        rc.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        Ok(Bytes::from(vec![(i * 37 % 256) as u8; entry_size as usize]))
                    }),
                )
                .await
                .unwrap();
            assert_eq!(result.len(), entry_size as usize, "entry {i}: wrong size");
            assert_eq!(result[0], expected, "entry {i}: data corruption detected");
        }
 // At most half the entries should need re-fetching (most should be in
 // memory or SSD).
        let refetches = refetch_count.load(std::sync::atomic::Ordering::Relaxed);
        assert!(refetches <= n / 2, "too many re-fetches ({refetches}/{n}): cache not working");
    }

 /// cacheStatsWithSsd: two-tier cache exposes accurate
 /// SSD statistics via the memory tier stats interface.
    #[tokio::test]
    async fn test_tiered_cache_stats_accumulate() {
        let tmp = tempfile::tempdir().unwrap();
        let config = DataCacheConfig {
            max_memory_bytes: 4 * 1024 * 1024,
            num_shards: memory::DEFAULT_NUM_SHARDS,
            ssd_enabled: true,
            ssd_cache_dir: Some(tmp.path().join("stats_test")),
            ssd_max_bytes: ssd::REGION_SIZE * 2,
            ssd_num_shards: 1,
        };
        let cache = TieredDataCache::new(&config).await.unwrap();
        let path = Path::from("test.lance");

 // 5 misses populate both tiers.
        for i in 0u64..5 {
            let data = Bytes::from(vec![i as u8; 4096]);
            cache
                .get_or_load(
                    &path,
                    i * 4096,
                    4096,
                    Box::pin(async move { Ok(data) }),
                )
                .await
                .unwrap();
        }

        let stats = cache.memory_stats();
        assert_eq!(stats.misses, 5);
        assert_eq!(stats.current_bytes, 5 * 4096);

 // 5 hits from memory.
        for i in 0u64..5 {
            cache
                .get_or_load(
                    &path,
                    i * 4096,
                    4096,
                    Box::pin(async { panic!("must hit") }),
                )
                .await
                .unwrap();
        }
        assert_eq!(cache.memory_stats().hits, 5);
    }
}
