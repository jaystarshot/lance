// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! SSD cache tier — placeholder module.
//!
//! This module defines the public interface for the SSD tier so that the
//! integration points (memory eviction → SSD write, SSD read on L1 miss) are
//! wired up correctly from day one.  The actual region-based file management,
//! coalesced `pread` I/O, and LRU region eviction will be implemented in a
//! follow-up PR.
//!
//! Design follows Velox's `SsdFile` / `SsdCache`:
//! * Fixed 64 MiB regions packed sequentially in one or more files.
//! * Entry index: `DataCacheKey → SsdRun { region, offset_in_region, size }`.
//! * Region eviction: track per-region read frequency; evict least-read region.
//! * On restart: delete all SSD files and rebuild clean (no checkpoint needed).

use std::path::PathBuf;
use std::sync::Arc;

use bytes::Bytes;

use lance_core::Result;

use super::DataCacheKey;

/// Location of a byte range within an SSD cache file.
///
/// Encodes `(region_index, byte_offset_within_region, payload_size)` compactly.
/// Mirrors Velox's `SsdRun` which packs offset and size into a single `u64`.
#[derive(Debug, Clone, Copy)]
pub struct SsdRun {
    /// Index of the 64 MiB region that contains this entry.
    pub region: u32,
    /// Byte offset of the entry within that region.
    pub offset_in_region: u32,
    /// Payload size in bytes (max 64 MiB per entry, same as Velox's 23-bit cap).
    pub size: u32,
}

/// Size of a single SSD region in bytes (64 MiB — same as Velox).
pub const REGION_SIZE: u64 = 64 * 1024 * 1024;

/// Configuration for the SSD cache tier.
#[derive(Debug, Clone)]
pub struct SsdCacheConfig {
    /// Directory where cache files are stored.
    pub cache_dir: PathBuf,
    /// Maximum total bytes the SSD tier may consume.
    pub max_bytes: u64,
}

/// SSD cache tier.
///
/// All public methods currently return stub results (`None` / no-op) so that
/// the memory tier and integration plumbing can be tested end-to-end before
/// the full SSD implementation lands.
#[derive(Debug)]
pub struct SsdCache {
    #[allow(dead_code)]
    config: SsdCacheConfig,
}

impl SsdCache {
    /// Create a new (empty) SSD cache at `config.cache_dir`.
    ///
    /// On restart the directory is wiped so there is no stale data to recover.
    pub async fn new(config: SsdCacheConfig) -> Result<Arc<Self>> {
        // TODO: create/clean the cache directory.
        Ok(Arc::new(Self { config }))
    }

    /// Look up a byte range in the SSD cache.
    ///
    /// Returns `None` on a miss (current stub behaviour).
    pub async fn get(&self, _key: &DataCacheKey) -> Option<Bytes> {
        // TODO: look up key in entry index, pread from region file.
        None
    }

    /// Write a byte range to the SSD cache asynchronously.
    ///
    /// Currently a no-op; real implementation will pack the entry into the
    /// current writable region and update the entry index.
    pub async fn insert(&self, _key: DataCacheKey, _data: Bytes) {
        // TODO: pack entry into region, update index, trigger region eviction
        // when the SSD limit is reached.
    }
}
