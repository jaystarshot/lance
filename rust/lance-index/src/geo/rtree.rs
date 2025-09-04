//! Spatial index - now using paged leaf R-tree implementation
//! 
//! This module re-exports the paged leaf R-tree as the default spatial index implementation

// Re-export the paged leaf R-tree implementation as the primary spatial index
pub use crate::geo::paged_leaf_rtree::{
    SpatialDataEntry as SpatialEntry,
    PagedLeafRTreeIndex as SpatialIndex,
    PagedLeafConfig,
};

// Re-export for backward compatibility
pub type RTreeEntry = SpatialEntry;
pub type RTreeIndex = SpatialIndex;

#[cfg(test)]
mod rtree_tests;