// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Async two-tier data cache for Lance I/O.
//!
//! Modelled after Velox's AsyncDataCache: a memory (L1) tier backed by an
//! optional SSD (L2) tier.  Raw byte ranges fetched from remote object stores
//! are cached here so that repeated reads avoid network round-trips.
//!
//! The cache is scoped to a [`crate::Session`]-equivalent lifetime and shared
//! across all scanners that open the same dataset.

use std::{collections::HashMap, path::PathBuf};

use bytes::Bytes;

/// Configuration for the two-tier async data cache.
///
/// Parsed from `storage_options` when opening a dataset:
///
/// ```python
/// ds = lance.dataset(
///     "s3://bucket/data.lance",
///     storage_options={
///         "max_memory_cache_mb": "1000",
///         "ssd_cache_dir":       "/mnt/nvme/lance_cache",
///         "ssd_cache_size_mb":   "100000",
///     },
/// )
/// ```
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
    /// Recognized storage-option keys.
    pub const KEY_MAX_MEMORY_MB: &'static str = "max_memory_cache_mb";
    pub const KEY_SSD_CACHE_DIR: &'static str = "ssd_cache_dir";
    pub const KEY_SSD_CACHE_SIZE_MB: &'static str = "ssd_cache_size_mb";

    /// Parse from the merged `storage_options` HashMap.
    ///
    /// Returns `None` when none of the recognised keys are present, so that
    /// callers can cheaply skip cache construction for datasets that don't need
    /// it.
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

        // Only build a config when at least one cache option is present.
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

/// Cache key for a raw byte range within a file.
///
/// The `file_id` is a stable numeric identifier for the file path (interned at
/// `DataCache` construction time).  The `offset` is the byte offset after
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

/// Async two-tier (memory + SSD) data cache.
///
/// Implementations must be cheap to clone (typically an `Arc` wrapper) and
/// safe to share across threads.
///
/// The trait is intentionally minimal at this stage.  Concrete read/write
/// methods will be added alongside the memory and SSD tier implementations.
pub trait DataCache: Send + Sync + std::fmt::Debug {
    /// Look up a cached byte range.  Returns `None` on a cache miss.
    fn get(&self, key: &DataCacheKey) -> Option<Bytes>;

    /// Insert a byte range into the cache.
    fn insert(&self, key: DataCacheKey, data: Bytes);
}

/// A no-op [`DataCache`] used as a placeholder while the real implementation
/// is wired up.  All lookups miss; all inserts are discarded.
#[derive(Debug)]
pub struct NoopDataCache;

impl DataCache for NoopDataCache {
    fn get(&self, _key: &DataCacheKey) -> Option<Bytes> {
        None
    }

    fn insert(&self, _key: DataCacheKey, _data: Bytes) {}
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_absent_when_no_keys() {
        let opts = HashMap::new();
        assert!(DataCacheConfig::from_storage_options(&opts).is_none());
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
        assert_eq!(
            cfg.ssd_cache_dir,
            Some(PathBuf::from("/mnt/nvme/cache"))
        );
        assert_eq!(cfg.ssd_max_bytes, 100_000 * 1024 * 1024);
    }

    #[test]
    fn test_noop_cache() {
        let cache = NoopDataCache;
        let key = DataCacheKey {
            file_id: 1,
            offset: 0,
            length: 4096,
        };
        assert!(cache.get(&key).is_none());
        cache.insert(key, Bytes::from_static(b"hello"));
    }
}
