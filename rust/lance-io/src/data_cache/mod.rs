// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Async two-tier data cache for Lance I/O.
//!
//! Modelled after Velox's `AsyncDataCache`: a memory (L1) tier backed by an
//! optional SSD (L2) tier.  Raw byte ranges fetched from remote object stores
//! are cached here so repeated reads avoid network round-trips.
//!
//! # Architecture
//!
//! ```text
//! FileScheduler::submit_request()
//!   │
//!   ├─ L1: MemoryCache  (16 shards, clock-hand eviction, ~microseconds)
//!   │     HIT → return bytes immediately
//!   │
//!   ├─ L2: SsdCache  (region files, coalesced pread, ~milliseconds)
//!   │     HIT → populate L1 → return
//!   │
//!   └─ L3: object store  (network, tens–hundreds of ms)
//!          → populate L2 + L1 → return
//! ```
//!
//! # Configuration
//!
//! Pass via `storage_options` when opening a dataset:
//!
//! ```python
//! ds = lance.dataset(
//!     "s3://bucket/data.lance",
//!     storage_options={
//!         "max_memory_cache_mb": "1000",
//!         "ssd_cache_dir":       "/mnt/nvme/lance_cache",
//!         "ssd_cache_size_mb":   "100000",
//!     },
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
/// by [`FileIds`]).  The `offset` and `length` are the byte range after
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
/// Parsed from `storage_options` when opening a dataset.
#[derive(Debug, Clone)]
pub struct DataCacheConfig {
    /// Maximum bytes to hold in the in-memory (L1) cache tier.
    /// Default: 256 MiB when not specified but another cache option is set.
    pub max_memory_bytes: u64,

    /// Directory on a local SSD for the on-disk (L2) cache tier.
    /// When `None`, only the memory tier is active.
    pub ssd_cache_dir: Option<PathBuf>,

    /// Maximum bytes the SSD tier may consume.
    /// Ignored when `ssd_cache_dir` is `None`.
    pub ssd_max_bytes: u64,
}

impl DataCacheConfig {
    pub const KEY_MAX_MEMORY_MB: &'static str = "max_memory_cache_mb";
    pub const KEY_SSD_CACHE_DIR: &'static str = "ssd_cache_dir";
    pub const KEY_SSD_CACHE_SIZE_MB: &'static str = "ssd_cache_size_mb";

    /// Parse from the merged `storage_options` map.
    ///
    /// Returns `None` when none of the recognised keys are present so callers
    /// can cheaply skip cache construction.
    pub fn from_storage_options(opts: &HashMap<String, String>) -> Option<Self> {
        let max_memory_bytes = opts
            .get(Self::KEY_MAX_MEMORY_MB)
            .and_then(|v| v.parse::<u64>().ok())
            .map(|mb| mb * 1024 * 1024);

        let ssd_cache_dir = opts.get(Self::KEY_SSD_CACHE_DIR).map(PathBuf::from);

        let ssd_max_bytes = opts
            .get(Self::KEY_SSD_CACHE_SIZE_MB)
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(0)
            * 1024
            * 1024;

        if max_memory_bytes.is_none() && ssd_cache_dir.is_none() {
            return None;
        }

        Some(Self {
            max_memory_bytes: max_memory_bytes.unwrap_or(256 * 1024 * 1024),
            ssd_cache_dir,
            ssd_max_bytes,
        })
    }
}

// ─── Trait ───────────────────────────────────────────────────────────────────

/// Async two-tier (memory + SSD) data cache.
///
/// The primary entry point is [`DataCache::get_or_load`], which handles both
/// cache lookup and load deduplication: if multiple tasks request the same
/// byte range concurrently, only the first triggers `loader`; all others wait
/// for the result — equivalent to Velox's `CoalescedLoad::loadOrFuture`.
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
/// unchanged.  Used in tests and as a placeholder.
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
    /// Creates the SSD tier and cleans its directory if `ssd_cache_dir` is set.
    pub async fn new(config: &DataCacheConfig) -> Result<Arc<Self>> {
        let memory = MemoryCache::new(config.max_memory_bytes);

        let ssd = if let Some(dir) = &config.ssd_cache_dir {
            let ssd_config = SsdCacheConfig {
                cache_dir: dir.clone(),
                max_bytes: config.ssd_max_bytes,
            };
            Some(SsdCache::new(ssd_config).await?)
        } else {
            None
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

        // If SSD tier is enabled, wrap the remote loader so that on an L1 miss
        // we check L2 before going to the object store.
        let effective_loader: BoxFuture<'a, Result<Bytes>> = if let Some(ssd) = &self.ssd {
            let key_for_ssd = key.clone();
            Box::pin(async move {
                if let Some(bytes) = ssd.get(&key_for_ssd).await {
                    return Ok(bytes);
                }
                let bytes = loader.await?;
                ssd.insert(key_for_ssd, bytes.clone()).await;
                Ok(bytes)
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
        assert!(DataCacheConfig::from_storage_options(&HashMap::new()).is_none());
    }

    #[test]
    fn test_config_memory_only() {
        let opts = HashMap::from([(
            DataCacheConfig::KEY_MAX_MEMORY_MB.to_string(),
            "512".to_string(),
        )]);
        let cfg = DataCacheConfig::from_storage_options(&opts).unwrap();
        assert_eq!(cfg.max_memory_bytes, 512 * 1024 * 1024);
        assert!(cfg.ssd_cache_dir.is_none());
    }

    #[test]
    fn test_config_full() {
        let opts = HashMap::from([
            (
                DataCacheConfig::KEY_MAX_MEMORY_MB.to_string(),
                "1000".to_string(),
            ),
            (
                DataCacheConfig::KEY_SSD_CACHE_DIR.to_string(),
                "/mnt/nvme/cache".to_string(),
            ),
            (
                DataCacheConfig::KEY_SSD_CACHE_SIZE_MB.to_string(),
                "100000".to_string(),
            ),
        ]);
        let cfg = DataCacheConfig::from_storage_options(&opts).unwrap();
        assert_eq!(cfg.max_memory_bytes, 1000 * 1024 * 1024);
        assert_eq!(cfg.ssd_cache_dir, Some(PathBuf::from("/mnt/nvme/cache")));
        assert_eq!(cfg.ssd_max_bytes, 100_000 * 1024 * 1024);
    }

    #[tokio::test]
    async fn test_tiered_cache_memory_hit() {
        let config = DataCacheConfig {
            max_memory_bytes: 10 * 1024 * 1024,
            ssd_cache_dir: None,
            ssd_max_bytes: 0,
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
}
