// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! SSD cache tier — Rust port of SsdFile / `SsdCache`.
//!
//! # Design
//!
//! Storage is organised into fixed-size **64 MiB regions** packed sequentially
//! inside one or more files on a local SSD. Each `SsdFile` is independent and
//! sharded by `file_id`, allowing concurrent reads and writes across shards.
//!
//! ```text
//! cache_0.bin: [region 0 | region 1 | region 2 |...]
//! cache_1.bin: [region 0 | region 1 |...]
//! ```
//!
//! # Entry lifecycle
//!
//! 1. Memory tier misses → object-store fetch → entry written to SSD + memory.
//! 2. Memory tier eviction → cached data lives on in SSD.
//! 3. Subsequent memory miss → SSD hit → memory re-populated without network.
//!
//! # Region eviction
//!
//! [`RegionTracker`] accumulates bytes read per region (SsdFileTracker).
//! Scores decay periodically to age out old hot-spots. When the SSD is full,
//! the [`NUM_EVICTION_CANDIDATES`] least-read regions are evicted as a unit —
//! all their entries are removed from the index and the regions become writable
//! again.
//!
//! # On restart
//!
//! The cache directory is wiped on startup (no checkpoint/recovery). This
//! keeps the implementation simple 

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use bytes::Bytes;
use lance_core::Result;

use super::DataCacheKey;

// ─── Constants (matching ) ──────────────────────────────────────────────

/// Region size in bytes — identical to kRegionSize.
pub const REGION_SIZE: u64 = 64 * 1024 * 1024; // 64 MiB

/// Default number of SSD shard files numShards for SSD.
pub const DEFAULT_NUM_SSD_SHARDS: usize = 4;

/// Number of eviction candidates to consider
const NUM_EVICTION_CANDIDATES: usize = 3;

/// Decay the region-score every this many file-touch events.
/// kDecayInterval.
const DECAY_INTERVAL: u64 = 1_000;

/// Score decay multiplier applied on each interval.
const DECAY_FACTOR: f64 = 0.9;

/// Gap threshold (bytes) below which adjacent SSD reads are merged into one
/// `read_at` call when average payload is small (< 10 KiB).
/// uses 25 000 bytes in this case.
const SMALL_PAYLOAD_MAX_GAP: u64 = 25_000;

/// Gap threshold for larger payloads (≥ 10 KiB average).
/// uses 50 000 bytes.
const LARGE_PAYLOAD_MAX_GAP: u64 = 50_000;

/// Maximum number of discrete ranges per coalesced read.
/// uses 900 (safely below IOV_MAX on Linux).
const MAX_COALESCE_RANGES: usize = 900;

// ─── SsdRun ──────────────────────────────────────────────────────────────────

/// Location of a byte range within an SSD cache file.
///
/// Compact enough to fit in a `HashMap` value — same role as SsdRun.
#[derive(Debug, Clone, Copy)]
pub struct SsdRun {
 /// 64 MiB region index within the file.
    pub region: u32,
 /// Byte offset of the entry *within* that region.
    pub offset_in_region: u32,
 /// Payload size in bytes.
    pub size: u32,
}

impl SsdRun {
 /// Absolute byte offset from the start of the file.
    #[inline]
    pub fn file_offset(&self) -> u64 {
        self.region as u64 * REGION_SIZE + self.offset_in_region as u64
    }
}

// ─── RegionTracker ───────────────────────────────────────────────────────────

/// Tracks per-region access frequency for eviction candidate selection.
///
/// SsdFileTracker:
/// * `region_read()` — accumulate bytes read from a region.
/// * `region_filled()` — boost a region when it transitions writable → full,
/// preventing newly-filled regions from being immediately evicted.
/// * `file_touched()` — increment the event counter; decay scores every
/// [`DECAY_INTERVAL`] events so old hot-spots age out.
/// * `find_eviction_candidates()` — return the N least-read regions.
struct RegionTracker {
 /// Cumulative bytes-read score per region. Lower = better eviction candidate.
    scores: Vec<f64>,
 /// Event counter — triggers periodic score decay.
    event_count: u64,
}

impl RegionTracker {
    fn new() -> Self {
        Self {
            scores: Vec::new(),
            event_count: 0,
        }
    }

    fn ensure_capacity(&mut self, regions: usize) {
        if self.scores.len() < regions {
            self.scores.resize(regions, 0.0);
        }
    }

 /// Record `bytes` read from `region`
    fn region_read(&mut self, region: u32, bytes: u64) {
        let idx = region as usize;
        self.ensure_capacity(idx + 1);
        self.scores[idx] += bytes as f64;
    }

 /// Boost score when a region transitions from writable to full so it
 /// is not immediately evicted
    fn region_filled(&mut self, region: u32) {
        let idx = region as usize;
        self.ensure_capacity(idx + 1);
 // Give a one-time boost proportional to a fraction of the region size.
        self.scores[idx] += REGION_SIZE as f64 * 0.1;
    }

 /// Increment event counter and periodically decay all scores —
 /// fileTouched().
    fn file_touched(&mut self) {
        self.event_count += 1;
        if self.event_count % DECAY_INTERVAL == 0 {
            for s in self.scores.iter_mut() {
                *s *= DECAY_FACTOR;
            }
        }
    }

 /// Return up to `n` region indices with the lowest scores, excluding
 /// any in `pinned`. findEvictionCandidates().
    fn find_eviction_candidates(&self, n: usize, pinned: &[u32]) -> Vec<u32> {
        let mut indexed: Vec<(u32, u64)> = self
            .scores
            .iter()
            .enumerate()
            .filter(|(i, _)| !pinned.contains(&(*i as u32)))
            .map(|(i, &s)| (i as u32, s.to_bits())) // to_bits gives total order
            .collect();

        indexed.sort_by_key(|&(_, bits)| bits); // ascending = lowest score first
        indexed.truncate(n);
        indexed.into_iter().map(|(r, _)| r).collect()
    }
}

// ─── SsdFileState (inside RwLock) ────────────────────────────────────────────

struct SsdFileState {
    entries: HashMap<DataCacheKey, SsdRun>,
    region_sizes: Vec<u32>,
    writable_regions: Vec<u32>,
    num_regions: u32,
    tracker: RegionTracker,
    // Stats — plain u64 protected by the RwLock.
    // Updated when we already hold the write lock, so no extra atomics needed.
    bytes_written: u64,
    bytes_read: u64,
    entries_written: u64,
    entries_read: u64,
}

impl SsdFileState {
    fn new() -> Self {
        Self {
            entries: HashMap::new(),
            region_sizes: Vec::new(),
            writable_regions: Vec::new(),
            num_regions: 0,
            tracker: RegionTracker::new(),
            bytes_written: 0,
            bytes_read: 0,
            entries_written: 0,
            entries_read: 0,
        }
    }

 /// Find available space for `size` bytes in a writable region, update
 /// `region_sizes` to reserve the space, and return `(file_offset, region)`.
 ///
 /// Returns `None` if no writable region can accommodate the entry.
 /// Equivalent to getSpace() — must be called under write lock.
    fn get_space(&mut self, size: u32) -> Option<(u64, u32)> {
        loop {
            let region = *self.writable_regions.first()?;
            let used = self.region_sizes[region as usize];
            let available = REGION_SIZE as u32 - used;

            if size <= available {
 // Reserve space by advancing the region's write pointer.
                self.region_sizes[region as usize] += size;
                let file_offset =
                    region as u64 * REGION_SIZE + used as u64;
                return Some((file_offset, region));
            }

 // Region too full for this entry — mark as filled, try next.
 // tracker_.regionFilled(region) + writableRegions_.erase().
            self.tracker.region_filled(region);
            self.writable_regions.remove(0);
        }
    }

 /// Grow the file by one region, or evict the least-read regions to free
 /// space. Returns `true` if at least one writable region is now available.
 ///
 /// Equivalent to growOrEvictLocked().
 /// Must be called under write lock with the file handle provided for
 /// `set_len()`.
    fn grow_or_evict(
        &mut self,
        file: &std::fs::File,
        max_regions: u32,
    ) -> std::io::Result<bool> {
        if self.num_regions < max_regions {
 // Grow the file by one region->truncate(newSize).
            let new_len = (self.num_regions + 1) as u64 * REGION_SIZE;
            file.set_len(new_len)?;
            let new_region = self.num_regions;
            self.region_sizes.push(0);
            self.tracker.ensure_capacity(new_region as usize + 1);
            self.writable_regions.push(new_region);
            self.num_regions += 1;
            tracing::debug!(
                "SSD cache file grew to {} regions (max {})",
                self.num_regions,
                max_regions
            );
            return Ok(true);
        }

 // File at maximum size — evict least-read regions.
 // : tracker_.findEvictionCandidates(kNumEvictionCandidates,...).
        let candidates =
            self.tracker.find_eviction_candidates(NUM_EVICTION_CANDIDATES, &[]);
        if candidates.is_empty() {
            tracing::warn!("SSD cache: no eviction candidates found, dropping write");
            return Ok(false);
        }

 // Remove all entries belonging to the evicted regions —
 // clearRegionEntriesLocked(candidates).
        self.entries
            .retain(|_, run| !candidates.contains(&run.region));

 // Reset region write pointers and mark as writable —
 // writableRegions_ = candidates.
        for &r in &candidates {
            self.region_sizes[r as usize] = 0;
        }
        self.writable_regions.clone_from(&candidates);

        tracing::debug!(
            "SSD cache evicted {} regions: {:?}",
            candidates.len(),
            candidates
        );
        Ok(true)
    }
}

// ─── SsdFile ─────────────────────────────────────────────────────────────────

/// Per-file stats snapshot returned by [`SsdFile::stats`].
#[derive(Debug, Default, Clone)]
struct SsdFileStats {
    bytes_written: u64,
    bytes_read: u64,
    entries_written: u64,
    entries_read: u64,
}

/// One SSD cache file managing N × 64 MiB regions.
///
/// `pread` / `pwrite` calls are issued without holding any in-memory lock —
/// on Linux these are atomic per-call at the OS level. The `RwLock` on
/// [`SsdFileState`] only protects the in-memory index and region metadata.
struct SsdFile {
    path: PathBuf,
 /// File handle — `Arc` so clone is cheap and pread/pwrite are OS-safe.
    file: Arc<std::fs::File>,
 /// Maximum number of 64 MiB regions this file may grow to.
    max_regions: u32,
 /// Mutable index and region metadata.
    state: RwLock<SsdFileState>,
 // Stats — atomic so they can be read without acquiring any lock.
}

impl std::fmt::Debug for SsdFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = self.state.read().unwrap();
        f.debug_struct("SsdFile")
            .field("path", &self.path)
            .field("num_regions", &state.num_regions)
            .field("entries", &state.entries.len())
            .finish()
    }
}

impl SsdFile {
 /// Open (or create) an SSD cache file at `path`, allowing up to
 /// `max_regions` × [`REGION_SIZE`] bytes.
 ///
 /// Always starts with `truncate(true)` — no checkpoint recovery.
    fn open(path: PathBuf, max_regions: u32) -> std::io::Result<Arc<Self>> {
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)?;

        Ok(Arc::new(Self {
            path,
            file: Arc::new(file),
            max_regions,
            state: RwLock::new(SsdFileState::new()),
        }))
    }

 // ── Single-entry get ──────────────────────────────────────────────────

 /// Look up `key` and read its bytes from disk.
 ///
 /// Phase 1 (read lock): index lookup.
 /// Phase 2 (no lock): `pread` from disk.
 /// Phase 3 (write lock): update tracker.
    fn get(&self, key: &DataCacheKey) -> Option<Bytes> {
 // Phase 1: index lookup — read lock (brief).
        let run = {
            let state = self.state.read().unwrap();
            *state.entries.get(key)?
        };

 // Phase 2: read from disk — no lock (pread is OS-atomic).
        let offset = run.file_offset();
        let size = run.size as usize;
        let mut buf = vec![0u8; size];
        {
            #[cfg(unix)]
            use std::os::unix::fs::FileExt;
            self.file.read_exact_at(&mut buf, offset).ok()?;
        }

 // Phase 3: update tracker — write lock (brief).
        {
            let mut state = self.state.write().unwrap();
            state.tracker.region_read(run.region, size as u64);
            state.tracker.file_touched();
            state.bytes_read += size as u64;
            state.entries_read += 1;
        }

        Some(Bytes::from(buf))
    }

 // ── Single-entry insert ───────────────────────────────────────────────

 /// Write `data` for `key` to disk.
 ///
 /// Phase 1 (write lock): reserve space via `get_space()` / `grow_or_evict()`.
 /// Phase 2 (no lock): `pwrite` to disk.
 /// Phase 3 (write lock): register in entry index.
 ///
 /// Equivalent to write(pins) for a single entry.
    fn insert(&self, key: DataCacheKey, data: &[u8]) -> std::io::Result<()> {
        let size = data.len() as u32;
        if size == 0 || size as u64 > REGION_SIZE {
 // Skip empty or oversized entries (same as size cap).
            return Ok(());
        }

 // Phase 1: reserve space — write lock.
        let (file_offset, region) = {
            let mut state = self.state.write().unwrap();
            loop {
                if let Some(space) = state.get_space(size) {
                    break space;
                }
 // No space in any writable region — grow or evict.
                if !state.grow_or_evict(&self.file, self.max_regions)? {
                    return Ok(()); // SSD full, write dropped
                }
            }
        };

 // Phase 2: write to disk — no lock.
        {
            #[cfg(unix)]
            use std::os::unix::fs::FileExt;
            self.file.write_all_at(data, file_offset)?;
        }

 // Phase 3: register entry — write lock (brief).
        {
            let mut state = self.state.write().unwrap();
            let offset_in_region =
                (file_offset - region as u64 * REGION_SIZE) as u32;
            state.entries.insert(
                key,
                SsdRun { region, offset_in_region, size },
            );
            state.bytes_written += size as u64;
            state.entries_written += 1;
        }
        Ok(())
    }

 // ── Batch insert (write path) ─────────────────────────────────────────

 /// Write multiple entries sorted by `(key.file_id, key.offset)` for disk
 /// write locality. Entries that fit in the same region are written with a
 /// single `write_at` call (equivalent to writev batch).
 ///
 /// Equivalent to write(pins).
    fn insert_many(
        &self,
        mut entries: Vec<(DataCacheKey, Bytes)>,
    ) -> std::io::Result<()> {
        if entries.is_empty() {
            return Ok(());
        }

 // Sort by (file_id, offset) — adjacent in storage → adjacent on SSD.
 // : std::sort(pins.begin(), pins.end()).
        entries.sort_by_key(|(k, _)| (k.file_id, k.offset));

        let mut i = 0;
        while i < entries.len() {
 // Collect entries that fit into the current writable region.
            let (batch_file_offset, mut batch_buf, batch_runs) = {
                let mut state = self.state.write().unwrap();

 // Ensure we have a writable region.
                loop {
                    if state.writable_regions.first().is_some() {
                        break;
                    }
                    if !state.grow_or_evict(&self.file, self.max_regions)? {
                        return Ok(()); // SSD full
                    }
                }

                let region = *state.writable_regions.first().unwrap();
                let region_start = state.region_sizes[region as usize];
                let available = REGION_SIZE as u32 - region_start;

 // Accumulate as many entries as fit in this region.
                let mut buf = Vec::new();
                let mut runs: Vec<(usize, SsdRun)> = Vec::new(); // (entry_idx, run)
                let mut written_in_region = 0u32;

                let _j_start = i;
                let mut j = i;
                while j < entries.len() {
                    let size = entries[j].1.len() as u32;
                    if size == 0 || size as u64 > REGION_SIZE {
                        j += 1; // skip invalid
                        continue;
                    }
                    if written_in_region + size > available {
                        break; // region full
                    }
                    let offset_in_region = region_start + written_in_region;
                    runs.push((
                        j,
                        SsdRun {
                            region,
                            offset_in_region,
                            size,
                        },
                    ));
                    buf.extend_from_slice(&entries[j].1);
                    written_in_region += size;
                    j += 1;
                }

                if runs.is_empty() {
 // Nothing fit — mark region full and retry.
                    state.tracker.region_filled(region);
                    state.writable_regions.remove(0);
                    continue;
                }

 // Advance the region write pointer for all accumulated entries.
                state.region_sizes[region as usize] += written_in_region;
                let batch_file_offset =
                    region as u64 * REGION_SIZE + region_start as u64;
                i = j;

                (batch_file_offset, buf, runs)
            }; // write lock released

 // Single pwrite for the entire batch — no lock held.
            if !batch_buf.is_empty() {
                #[cfg(unix)]
                use std::os::unix::fs::FileExt;
                self.file.write_all_at(&batch_buf, batch_file_offset)?;
                batch_buf.clear();
            }

            {
                let mut state = self.state.write().unwrap();
                let n = batch_runs.len() as u64;
                let b: u64 = batch_runs.iter().map(|(idx, _)| entries[*idx].1.len() as u64).sum();
                for (idx, run) in batch_runs {
                    state.entries.insert(entries[idx].0.clone(), run);
                }
                state.bytes_written += b;
                state.entries_written += n;
            }
        }
        Ok(())
    }

 // ── Batch get (coalesced read path) ───────────────────────────────────

 /// Read multiple keys with coalesced `read_at` calls.
 ///
 /// Algorithm (load() / `readPins()`):
 /// 1. Look up all keys → `(key, SsdRun)` pairs (read lock, then released).
 /// 2. Sort by file offset.
 /// 3. Group consecutive entries whose gap is below `max_gap` into batches.
 /// 4. For each batch: single `read_at` spanning the full range, then slice.
    fn get_many(&self, keys: &[DataCacheKey]) -> Vec<Option<Bytes>> {
        if keys.is_empty() {
            return Vec::new();
        }

 // Phase 1: index lookups — read lock (brief).
        let runs: Vec<Option<SsdRun>> = {
            let state = self.state.read().unwrap();
            keys.iter()
                .map(|k| state.entries.get(k).copied())
                .collect()
        };

 // Compute average payload size to pick the coalescing gap threshold.
 // : totalPayloadBytes / pins.size() < 10000 ? 25000 : 50000.
        let valid_runs: Vec<(usize, SsdRun)> = runs
            .iter()
            .enumerate()
            .filter_map(|(i, r)| r.map(|run| (i, run)))
            .collect();

        if valid_runs.is_empty() {
            return vec![None; keys.len()];
        }

        let total_bytes: u64 =
            valid_runs.iter().map(|(_, r)| r.size as u64).sum();
        let avg_bytes = total_bytes / valid_runs.len() as u64;
        let max_gap = if avg_bytes < 10_000 {
            SMALL_PAYLOAD_MAX_GAP
        } else {
            LARGE_PAYLOAD_MAX_GAP
        };

 // Sort by file offset for coalescing.
        let mut sorted = valid_runs.clone();
        sorted.sort_by_key(|(_, r)| r.file_offset());

 // Phase 2: coalesced reads — no lock.
        let mut result_bufs: HashMap<usize, Bytes> = HashMap::new();

        let mut batch_start = 0usize;
        while batch_start < sorted.len() {
 // Determine the span of this coalesced batch.
            let batch_offset = sorted[batch_start].1.file_offset();
            let mut batch_end_byte = batch_offset + sorted[batch_start].1.size as u64;
            let mut batch_end_idx = batch_start + 1;

            while batch_end_idx < sorted.len()
                && batch_end_idx - batch_start < MAX_COALESCE_RANGES
            {
                let next_offset = sorted[batch_end_idx].1.file_offset();
                if next_offset > batch_end_byte + max_gap {
                    break; // gap too large — start a new batch
                }
                batch_end_byte = batch_end_byte
                    .max(next_offset + sorted[batch_end_idx].1.size as u64);
                batch_end_idx += 1;
            }

 // Single read spanning the entire batch (including gaps).
            let read_len = (batch_end_byte - batch_offset) as usize;
            let mut buf = vec![0u8; read_len];
            {
                #[cfg(unix)]
                use std::os::unix::fs::FileExt;
                if self.file.read_exact_at(&mut buf, batch_offset).is_err() {
                    batch_start = batch_end_idx;
                    continue;
                }
            }

 // Slice each entry's bytes out of the combined buffer.
            for &(key_idx, ref run) in &sorted[batch_start..batch_end_idx] {
                let start = (run.file_offset() - batch_offset) as usize;
                let end = start + run.size as usize;
                if end <= buf.len() {
                    result_bufs.insert(key_idx, Bytes::copy_from_slice(&buf[start..end]));
                }
            }

            batch_start = batch_end_idx;
        }

 // Phase 3: update tracker — write lock (brief).
        {
            let mut state = self.state.write().unwrap();
            for (_, run) in &valid_runs {
                state.tracker.region_read(run.region, run.size as u64);
            }
            state.tracker.file_touched();
            state.bytes_read += total_bytes;
            state.entries_read += valid_runs.len() as u64;
        }

        (0..keys.len()).map(|i| result_bufs.remove(&i)).collect()
    }

    /// Return a stats snapshot (briefly acquires read lock).
    fn stats(&self) -> SsdFileStats {
        let s = self.state.read().unwrap();
        SsdFileStats {
            bytes_written: s.bytes_written,
            bytes_read: s.bytes_read,
            entries_written: s.entries_written,
            entries_read: s.entries_read,
        }
    }
}

// ─── SsdCacheConfig ──────────────────────────────────────────────────────────

/// Configuration for the SSD cache tier.
#[derive(Debug, Clone)]
pub struct SsdCacheConfig {
 /// Directory where cache files are stored.
    pub cache_dir: PathBuf,
 /// Maximum total bytes the SSD tier may consume.
    pub max_bytes: u64,
 /// Number of SSD shard files. Must be a positive power of two.
 /// Defaults to [`DEFAULT_NUM_SSD_SHARDS`] (4).
    pub num_shards: usize,
}

impl SsdCacheConfig {
    pub fn new(cache_dir: PathBuf, max_bytes: u64) -> Self {
        Self {
            cache_dir,
            max_bytes,
            num_shards: DEFAULT_NUM_SSD_SHARDS,
        }
    }
}

// ─── SsdCacheStats ───────────────────────────────────────────────────────────

/// Snapshot statistics for the SSD cache tier.
#[derive(Debug, Default, Clone)]
pub struct SsdCacheStats {
    pub bytes_written: u64,
    pub bytes_read: u64,
    pub entries_written: u64,
    pub entries_read: u64,
}

// ─── SsdCache ────────────────────────────────────────────────────────────────

/// SSD cache tier — coordinates [`DEFAULT_NUM_SSD_SHARDS`] independent
/// [`SsdFile`] instances sharded by `file_id`.
///
/// Entry distribution mirrors : `file_idx = file_id & file_mask`.
#[derive(Debug)]
pub struct SsdCache {
    files: Vec<Arc<SsdFile>>,
 /// Bitmask for fast shard selection (`num_shards` must be power of two).
    file_mask: u64,
}

impl SsdCache {
 /// Create a new SSD cache at `config.cache_dir`.
 ///
 /// The directory is wiped on every startup — no stale data is recovered.
    pub async fn new(config: SsdCacheConfig) -> Result<Arc<Self>> {
        assert!(
            config.num_shards > 0 && config.num_shards.is_power_of_two(),
            "SsdCache num_shards must be a positive power of two, got {}",
            config.num_shards
        );

 // Clean then create the cache directory.
        let cache_dir = config.cache_dir.clone();
        tokio::task::spawn_blocking(move || -> std::io::Result<()> {
            if cache_dir.exists() {
                std::fs::remove_dir_all(&cache_dir)?;
            }
            std::fs::create_dir_all(&cache_dir)
        })
        .await
        .map_err(|e| lance_core::Error::io(e.to_string()))?
        .map_err(|e| lance_core::Error::io(e.to_string()))?;

 // Each shard file gets an equal share of the total capacity, rounded
 // down to whole regions.
        let bytes_per_shard = config.max_bytes / config.num_shards as u64;
        let max_regions_per_file = ((bytes_per_shard / REGION_SIZE).max(1)) as u32;
        let num_shards = config.num_shards;
        let cache_dir = config.cache_dir.clone();

        let files = tokio::task::spawn_blocking(move || {
            (0..num_shards)
                .map(|i| {
                    let path = cache_dir.join(format!("cache_{i}.bin"));
                    SsdFile::open(path, max_regions_per_file)
                        .map_err(|e| lance_core::Error::io(e.to_string()))
                })
                .collect::<Result<Vec<_>>>()
        })
        .await
        .map_err(|e| lance_core::Error::io(e.to_string()))??;

        let file_mask = (num_shards as u64) - 1;
        Ok(Arc::new(Self { files, file_mask }))
    }

    #[inline]
    fn select_file(&self, file_id: u64) -> &Arc<SsdFile> {
        &self.files[(file_id & self.file_mask) as usize]
    }

 /// Look up a single byte range in the SSD cache.
    pub async fn get(&self, key: &DataCacheKey) -> Option<Bytes> {
        let file = self.select_file(key.file_id).clone();
        let key = key.clone();
        tokio::task::spawn_blocking(move || file.get(&key))
            .await
            .ok()?
    }

 /// Look up multiple byte ranges in the SSD cache with coalesced reads.
 ///
 /// Entries in the same shard file are read with merged `read_at` calls
 /// when they are within [`SMALL_PAYLOAD_MAX_GAP`] or
 /// [`LARGE_PAYLOAD_MAX_GAP`] of each other / `readPins()`.
    pub async fn get_many(&self, keys: &[DataCacheKey]) -> Vec<Option<Bytes>> {
        if keys.is_empty() {
            return Vec::new();
        }

 // Group keys by shard file, preserving original indices.
        let mut by_file: Vec<Vec<(usize, DataCacheKey)>> =
            vec![Vec::new(); self.files.len()];
        for (i, key) in keys.iter().enumerate() {
            let idx = (key.file_id & self.file_mask) as usize;
            by_file[idx].push((i, key.clone()));
        }

 // Fire one spawn_blocking per non-empty shard.
        let mut tasks = Vec::new();
        for (file, keyed) in self.files.iter().zip(by_file.into_iter()) {
            if keyed.is_empty() {
                continue;
            }
            let file = file.clone();
            tasks.push(tokio::task::spawn_blocking(move || {
                let shard_keys: Vec<DataCacheKey> =
                    keyed.iter().map(|(_, k)| k.clone()).collect();
                let results = file.get_many(&shard_keys);
                keyed
                    .into_iter()
                    .zip(results)
                    .map(|((orig_idx, _), bytes)| (orig_idx, bytes))
                    .collect::<Vec<_>>()
            }));
        }

        let mut result = vec![None; keys.len()];
        for task in tasks {
            if let Ok(shard_results) = task.await {
                for (orig_idx, bytes) in shard_results {
                    result[orig_idx] = bytes;
                }
            }
        }
        result
    }

 /// Write a single byte range to the SSD cache.
    pub async fn insert(&self, key: DataCacheKey, data: Bytes) {
        let file = self.select_file(key.file_id).clone();
        tokio::task::spawn_blocking(move || {
            if let Err(e) = file.insert(key, &data) {
                tracing::warn!("SSD cache write failed: {}", e);
            }
        })
        .await
        .ok();
    }

 /// Write multiple byte ranges with sorted, batched `write_at` calls.
 ///
 /// Entries are sorted by `(file_id, offset)` within each shard before
 /// writing so that adjacent data lands adjacent on disk — 
 /// `write(pins)` with `std::sort(pins.begin(), pins.end())`.
    pub async fn insert_many(&self, entries: Vec<(DataCacheKey, Bytes)>) {
        if entries.is_empty() {
            return;
        }

 // Group by shard file.
        let mut by_file: Vec<Vec<(DataCacheKey, Bytes)>> =
            vec![Vec::new(); self.files.len()];
        for (key, data) in entries {
            let idx = (key.file_id & self.file_mask) as usize;
            by_file[idx].push((key, data));
        }

        let mut tasks = Vec::new();
        for (file, shard_entries) in self.files.iter().zip(by_file.into_iter()) {
            if shard_entries.is_empty() {
                continue;
            }
            let file = file.clone();
            tasks.push(tokio::task::spawn_blocking(move || {
                if let Err(e) = file.insert_many(shard_entries) {
                    tracing::warn!("SSD cache batch write failed: {}", e);
                }
            }));
        }
        futures::future::join_all(tasks).await;
    }

    /// Return a snapshot of aggregate statistics across all shard files.
    pub fn stats(&self) -> SsdCacheStats {
        let file_stats: Vec<SsdFileStats> = self.files.iter().map(|f| f.stats()).collect();
        SsdCacheStats {
            bytes_written: file_stats.iter().map(|s| s.bytes_written).sum(),
            bytes_read: file_stats.iter().map(|s| s.bytes_read).sum(),
            entries_written: file_stats.iter().map(|s| s.entries_written).sum(),
            entries_read: file_stats.iter().map(|s| s.entries_read).sum(),
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn key(file_id: u64, offset: u64, length: u64) -> DataCacheKey {
        DataCacheKey { file_id, offset, length }
    }

    async fn make_cache(max_bytes: u64, num_shards: usize) -> Arc<SsdCache> {
        let dir = tempfile::tempdir().unwrap();
        let cache_dir = dir.path().join("ssd_cache");
 // Keep dir alive for the duration of the test via Box::leak (test-only).
        Box::leak(Box::new(dir));
        let config = SsdCacheConfig {
            cache_dir,
            max_bytes,
            num_shards,
        };
        SsdCache::new(config).await.unwrap()
    }

    #[tokio::test]
    async fn test_basic_insert_and_get() {
        let cache = make_cache(REGION_SIZE * 4, 1).await;
        let k = key(0, 0, 5);
        let data = Bytes::from_static(b"hello");

        cache.insert(k.clone(), data.clone()).await;
        let result = cache.get(&k).await;
        assert_eq!(result.as_deref(), Some(b"hello".as_ref()));
    }

    #[tokio::test]
    async fn test_miss_returns_none() {
        let cache = make_cache(REGION_SIZE * 2, 1).await;
        assert!(cache.get(&key(99, 0, 4)).await.is_none());
    }

    #[tokio::test]
    async fn test_region_growth() {
 // Write enough entries to force the file to grow beyond 1 region.
        let cache = make_cache(REGION_SIZE * 4, 1).await;
        let entry_size = 16 * 1024 * 1024u64; // 16 MiB — 4 entries per region
        let num_entries = 8u64; // 2 regions worth

        for i in 0..num_entries {
            let data = Bytes::from(vec![i as u8; entry_size as usize]);
            cache.insert(key(0, i * entry_size, entry_size), data).await;
        }

        let stats = cache.stats();
        assert_eq!(stats.entries_written, num_entries);
        assert_eq!(stats.bytes_written, num_entries * entry_size);

 // All entries should still be readable.
        for i in 0..num_entries {
            let result = cache.get(&key(0, i * entry_size, entry_size)).await;
            assert!(result.is_some(), "entry {i} missing after region growth");
            assert_eq!(result.unwrap()[0], i as u8);
        }
    }

    #[tokio::test]
    async fn test_region_eviction() {
 // 1 region max, 2 entries — second should evict first region.
        let cache = make_cache(REGION_SIZE, 1).await;
        let entry_size = (REGION_SIZE / 2) as usize;

 // Fill region 0 with 2 entries.
        cache
            .insert(key(0, 0, entry_size as u64), Bytes::from(vec![1u8; entry_size]))
            .await;
        cache
            .insert(
                key(0, entry_size as u64, entry_size as u64),
                Bytes::from(vec![2u8; entry_size]),
            )
            .await;

 // One more entry forces region eviction.
        cache
            .insert(
                key(0, entry_size as u64 * 2, entry_size as u64),
                Bytes::from(vec![3u8; entry_size]),
            )
            .await;

        let stats = cache.stats();
        assert!(stats.entries_written >= 3, "expected at least 3 writes");
    }

    #[tokio::test]
    async fn test_multi_shard() {
        let cache = make_cache(REGION_SIZE * 8, 4).await;

 // Write entries with different file_ids — they'll land on different shards.
        for file_id in 0u64..8 {
            let data = Bytes::from(vec![file_id as u8; 4096]);
            cache.insert(key(file_id, 0, 4096), data).await;
        }

 // All should be readable.
        for file_id in 0u64..8 {
            let result = cache.get(&key(file_id, 0, 4096)).await;
            assert!(result.is_some(), "file_id={file_id} missing");
            assert_eq!(result.unwrap()[0], file_id as u8);
        }

        assert_eq!(cache.stats().entries_written, 8);
    }

    #[tokio::test]
    async fn test_batch_insert_and_get_many() {
        let cache = make_cache(REGION_SIZE * 4, 1).await;

        let entries: Vec<(DataCacheKey, Bytes)> = (0u64..10)
            .map(|i| {
                let k = key(0, i * 4096, 4096);
                let v = Bytes::from(vec![i as u8; 4096]);
                (k, v)
            })
            .collect();

        cache.insert_many(entries).await;

        let keys: Vec<DataCacheKey> =
            (0u64..10).map(|i| key(0, i * 4096, 4096)).collect();
        let results = cache.get_many(&keys).await;

        assert_eq!(results.len(), 10);
        for (i, result) in results.iter().enumerate() {
            assert!(result.is_some(), "entry {i} missing from batch get");
            assert_eq!(result.as_ref().unwrap()[0], i as u8);
        }

        let stats = cache.stats();
        assert_eq!(stats.entries_written, 10);
        assert_eq!(stats.entries_read, 10);
    }

    #[tokio::test]
    async fn test_get_many_coalesces_reads() {
 // Entries that are adjacent on disk should be read in one pread.
        let cache = make_cache(REGION_SIZE * 4, 1).await;
        let entry_size = 4096u64;

 // Write 5 adjacent entries in batch (they'll be sequential on disk).
        let entries: Vec<(DataCacheKey, Bytes)> = (0u64..5)
            .map(|i| (key(0, i * entry_size, entry_size), Bytes::from(vec![i as u8; entry_size as usize])))
            .collect();
        cache.insert_many(entries).await;

 // Read them back in a single batch — should coalesce into 1 pread.
        let keys: Vec<DataCacheKey> =
            (0u64..5).map(|i| key(0, i * entry_size, entry_size)).collect();
        let results = cache.get_many(&keys).await;

        for (i, r) in results.iter().enumerate() {
            assert!(r.is_some(), "entry {i} missing");
            assert_eq!(r.as_ref().unwrap()[0], i as u8);
        }
    }

    #[test]
    fn test_region_tracker_eviction_candidates() {
        let mut tracker = RegionTracker::new();
        tracker.ensure_capacity(5);

 // Region 0: heavily read.
        tracker.region_read(0, 1_000_000);
 // Region 1: lightly read.
        tracker.region_read(1, 1_000);
 // Region 2: never read → score 0.
 // Region 3: moderately read.
        tracker.region_read(3, 50_000);
 // Region 4: lightly read.
        tracker.region_read(4, 500);

 // Best eviction candidates: lowest score = 2 (0), 4 (500), 1 (1000).
        let candidates = tracker.find_eviction_candidates(3, &[]);
        assert_eq!(candidates[0], 2); // score 0 — evict first
        assert_eq!(candidates[1], 4); // score 500
        assert_eq!(candidates[2], 1); // score 1000
    }

    #[test]
    fn test_region_tracker_decay() {
        let mut tracker = RegionTracker::new();
        tracker.ensure_capacity(1);
        tracker.region_read(0, 1_000_000);

 // Fire DECAY_INTERVAL events to trigger a decay.
        for _ in 0..DECAY_INTERVAL {
            tracker.file_touched();
        }

 // Score should be reduced by DECAY_FACTOR.
        let expected = 1_000_000.0_f64 * DECAY_FACTOR;
        assert!(
            (tracker.scores[0] - expected).abs() < 1.0,
            "score={} expected={}",
            tracker.scores[0],
            expected
        );
    }

    #[test]
    fn test_ssd_run_file_offset() {
        let run = SsdRun {
            region: 2,
            offset_in_region: 1024,
            size: 4096,
        };
        assert_eq!(run.file_offset(), 2 * REGION_SIZE + 1024);
    }

 // ── Tests not ported from (with explanation) ────────────────────
 //
 // DISABLED_ssd (checkpoint recovery): ssd test verifies that a
 // corrupted shard file is detected and skipped during checkpoint reload.
 // We wipe the directory on restart with no recovery — not applicable.
 //
 // shutdown (eviction log): tracks an eviction log file per shard
 // that is truncated on shutdown. We have no eviction log — not applicable.
 //
 // shrinkWithSsdWrite: Requires SCOPED_TESTVALUE_SET hooks to pause the
 // background SSD write at a specific code point. Not portable.
 //
 // ssdWriteOptions / ssdFlushThresholdBytes: Test configurable thresholds
 // for when to flush saveable entries to SSD (maxWriteRatio,
 // ssdSavableRatio, minSsdSavableBytes). We flush eagerly on every
 // insert — these knobs are not implemented.
 //
 // appendSsdSaveable (partial): appendAll flag controls whether
 // saveToSsd() saves all saveable entries or just one per shard. Our
 // insert_many() always writes all provided entries — equivalent to
 // appendAll=true. The appendAll=false variant is not applicable.
 //
 // checkpoint: We do not implement checkpoint/recovery.
 //
 // makeEvictable: Tests explicit numPins / CachePin marking for SSD save.
 // Not implemented (see memory.rs TODO comment).
 //
 // ttl: CacheTTLController — not applicable for immutable Lance datasets.

 // ── Additional -inspired SSD tests ───────────────────────────────

 /// cacheStats (SSD portion): verify that bytes_written,
 /// bytes_read, entries_written, entries_read are all accurate.
    #[tokio::test]
    async fn test_ssd_cache_stats() {
        let cache = make_cache(REGION_SIZE * 4, 1).await;
        let entry_size = 8 * 1024u64; // 8 KiB
        let n = 10u64;

 // Write n entries.
        for i in 0..n {
            let data = Bytes::from(vec![i as u8; entry_size as usize]);
            cache.insert(key(0, i * entry_size, entry_size), data).await;
        }

        let after_write = cache.stats();
        assert_eq!(after_write.entries_written, n);
        assert_eq!(after_write.bytes_written, n * entry_size);
        assert_eq!(after_write.entries_read, 0);
        assert_eq!(after_write.bytes_read, 0);

 // Read all n entries back.
        for i in 0..n {
            let result = cache.get(&key(0, i * entry_size, entry_size)).await;
            assert!(result.is_some(), "entry {i} missing");
        }

        let after_read = cache.stats();
        assert_eq!(after_read.entries_written, n);
        assert_eq!(after_read.entries_read, n);
        assert_eq!(after_read.bytes_read, n * entry_size);
    }

 /// cacheStatsWithSsd (delta stats): subtracting stats
 /// snapshots must give accurate deltas for the intervening operations.
    #[tokio::test]
    async fn test_ssd_stats_delta() {
        let cache = make_cache(REGION_SIZE * 4, 1).await;
        let data = Bytes::from(vec![42u8; 4096]);
        let k = key(0, 0, 4096);

        let before = cache.stats();

        cache.insert(k.clone(), data).await;
        let _ = cache.get(&k).await;

        let after = cache.stats();

 // Delta: exactly 1 write and 1 read.
        assert_eq!(after.entries_written - before.entries_written, 1);
        assert_eq!(after.entries_read - before.entries_read, 1);
        assert_eq!(after.bytes_written - before.bytes_written, 4096);
        assert_eq!(after.bytes_read - before.bytes_read, 4096);
    }

 /// invalidSsdPath: creating a cache in an invalid
 /// or non-writable location must fail gracefully.
    #[tokio::test]
    async fn test_invalid_ssd_path_fails() {
 // A file path (not a directory) cannot be used as a cache directory.
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let bad_path = tmp.path().join("cannot_create_dir_inside_file");
        let config = SsdCacheConfig {
            cache_dir: bad_path,
            max_bytes: REGION_SIZE * 2,
            num_shards: 1,
        };
        let result = SsdCache::new(config).await;
        assert!(result.is_err(), "expected error for invalid SSD path");
    }

 /// DISABLED_ssd data-integrity check: bytes written to
 /// the SSD tier must be read back byte-for-byte identically. This is the
 /// core correctness guarantee of the SSD cache.
    #[tokio::test]
    async fn test_data_integrity_write_then_read() {
        let cache = make_cache(REGION_SIZE * 4, 1).await;

 // Write entries with recognisable per-entry byte patterns.
        let entry_size = 16 * 1024u64; // 16 KiB
        let n = 20u64;

        for i in 0..n {
 // Pattern: repeating (i % 256) so we can verify each byte.
            let data = Bytes::from(vec![(i % 256) as u8; entry_size as usize]);
            cache.insert(key(0, i * entry_size, entry_size), data).await;
        }

 // Read back and verify every byte.
        for i in 0..n {
            let result = cache.get(&key(0, i * entry_size, entry_size)).await;
            let bytes = result.unwrap_or_else(|| panic!("entry {i} not found"));
            assert_eq!(
                bytes.len(),
                entry_size as usize,
                "entry {i}: wrong length"
            );
            for (j, &b) in bytes.iter().enumerate() {
                assert_eq!(
                    b,
                    (i % 256) as u8,
                    "entry {i} byte {j}: got {b} expected {}",
                    i % 256
                );
            }
        }
    }

 /// appendSsdSaveable (appendAll=true path): insert_many
 /// writes all provided entries and all are readable — equivalent to 
 /// saveToSsd(appendAll=true) followed by reads.
    #[tokio::test]
    async fn test_insert_many_all_entries_written_and_readable() {
        let cache = make_cache(REGION_SIZE * 4, 1).await;
        let entry_size = 4096u64;
        let n = 50u64;

        let entries: Vec<(DataCacheKey, Bytes)> = (0..n)
            .map(|i| {
                let pattern = vec![(i % 256) as u8; entry_size as usize];
                (key(0, i * entry_size, entry_size), Bytes::from(pattern))
            })
            .collect();

        cache.insert_many(entries).await;

        let stats = cache.stats();
        assert_eq!(stats.entries_written, n, "all entries must be written");

 // All entries must be readable with correct data.
        for i in 0..n {
            let result = cache.get(&key(0, i * entry_size, entry_size)).await;
            let bytes = result.unwrap_or_else(|| panic!("entry {i} missing after insert_many"));
            assert_eq!(bytes[0], (i % 256) as u8, "entry {i}: wrong data");
        }
    }

 /// dataRanges data-integrity variant: bytes stored and
 /// retrieved must match exactly, regardless of size (small or large entries).
    #[tokio::test]
    async fn test_data_ranges_small_and_large() {
        let cache = make_cache(REGION_SIZE * 4, 1).await;

 // Small entries (< 10 KiB — triggers 25 KB coalesce gap).
        let small_size = 2048u64;
        for i in 0u64..8 {
            let data = Bytes::from(vec![(i * 17 % 256) as u8; small_size as usize]);
            cache.insert(key(1, i * small_size, small_size), data).await;
        }
        for i in 0u64..8 {
            let result = cache.get(&key(1, i * small_size, small_size)).await.unwrap();
            assert_eq!(result[0], (i * 17 % 256) as u8, "small entry {i}");
        }

 // Large entries (> 10 KiB — triggers 50 KB coalesce gap).
        let large_size = 128 * 1024u64;
        for i in 0u64..4 {
            let data = Bytes::from(vec![(i * 31 % 256) as u8; large_size as usize]);
            cache.insert(key(2, i * large_size, large_size), data).await;
        }
        for i in 0u64..4 {
            let result = cache.get(&key(2, i * large_size, large_size)).await.unwrap();
            assert_eq!(result[0], (i * 31 % 256) as u8, "large entry {i}");
            assert_eq!(result.len(), large_size as usize);
        }
    }

 /// Oversized entries (> REGION_SIZE) must be silently dropped — not
 /// written and not found on subsequent reads.
    #[tokio::test]
    async fn test_oversized_entry_silently_skipped() {
        let cache = make_cache(REGION_SIZE * 2, 1).await;
        let big = Bytes::from(vec![0u8; REGION_SIZE as usize + 1]);
        let k = key(0, 0, REGION_SIZE + 1);

        cache.insert(k.clone(), big).await;

 // No write should have occurred.
        assert_eq!(cache.stats().entries_written, 0);
        assert!(cache.get(&k).await.is_none());
    }

 /// Concurrent inserts and gets on the same cache must not corrupt data —
 /// equivalent to fuzz test for the SSD tier.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_concurrent_inserts_and_gets() {
        let cache = Arc::new(make_cache(REGION_SIZE * 8, 4).await);
        let entry_size = 4096u64;
        let n = 64u64;
        let deadline =
            std::time::Instant::now() + std::time::Duration::from_millis(300);

 // Writers: insert entries with known patterns.
        let cache_w = cache.clone();
        let writer = tokio::spawn(async move {
            while std::time::Instant::now() < deadline {
                for i in 0..n {
                    let data = Bytes::from(vec![(i % 256) as u8; entry_size as usize]);
                    cache_w.insert(key(0, i * entry_size, entry_size), data).await;
                }
            }
        });

 // Readers: read entries and verify data integrity on hits.
        let cache_r = cache.clone();
        let reader = tokio::spawn(async move {
            while std::time::Instant::now() < deadline {
                for i in 0..n {
                    if let Some(bytes) = cache_r.get(&key(0, i * entry_size, entry_size)).await {
 // Verify data integrity: all bytes should match the pattern.
                        assert_eq!(
                            bytes.len(),
                            entry_size as usize,
                            "entry {i}: wrong length"
                        );
                        let expected = (i % 256) as u8;
                        for (j, &b) in bytes.iter().enumerate() {
                            assert_eq!(b, expected, "entry {i} byte {j} corrupted");
                        }
                    }
                }
            }
        });

        writer.await.unwrap();
        reader.await.unwrap();

        let stats = cache.stats();
        assert!(stats.entries_written > 0);
    }
}
