//! Spatial indices for GeoArrow geometries

use std::any::Any;
use std::sync::Arc;

use async_trait::async_trait;
use datafusion_common::scalar::ScalarValue;
use datafusion_expr::{BinaryExpr, Expr, Operator};
use deepsize::DeepSizeOf;
use rstar::AABB;
use serde::{Deserialize, Serialize};

use crate::metrics::MetricsCollector;
use crate::scalar::{AnyQuery, IndexStore, SearchResult};
use crate::{Index, IndexParams, IndexType};
use lance_core::{Result};

pub mod builder;
pub mod rtree;

pub const LANCE_GEO_INDEX: &str = "__lance_geo_index";

/// Spatial query types that can be performed against a GeoIndex
#[derive(Debug, Clone, PartialEq)]
pub enum SpatialQuery {
    /// Find all geometries that intersect with the given bounding box
    Intersects(BoundingBox),
    /// Find all geometries that are contained within the given bounding box
    Within(BoundingBox),
    /// Find all geometries that contain the given point
    Contains(Point),
    /// Find all geometries within a given distance of a point
    DWithin(Point, f64),
    /// Find all geometries that touch the given bounding box
    Touches(BoundingBox),
    /// Find all geometries that are disjoint from the given bounding box
    Disjoint(BoundingBox),
}

impl AnyQuery for SpatialQuery {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn format(&self, col: &str) -> String {
        match self {
            Self::Intersects(bbox) => format!("ST_Intersects({}, {})", col, bbox.format()),
            Self::Within(bbox) => format!("ST_Within({}, {})", col, bbox.format()),
            Self::Contains(point) => format!("ST_Contains({}, {})", col, point.format()),
            Self::DWithin(point, distance) => format!("ST_DWithin({}, {}, {})", col, point.format(), distance),
            Self::Touches(bbox) => format!("ST_Touches({}, {})", col, bbox.format()),
            Self::Disjoint(bbox) => format!("ST_Disjoint({}, {})", col, bbox.format()),
        }
    }

    fn to_expr(&self, col: String) -> Expr {
        // For now, return a placeholder expression
        // In a full implementation, this would use spatial functions from DataFusion
        match self {
            Self::Intersects(_) => Expr::BinaryExpr(BinaryExpr {
                left: Box::new(Expr::Column(datafusion_common::Column::new_unqualified(col))),
                op: Operator::Eq,
                right: Box::new(Expr::Literal(ScalarValue::Boolean(Some(true)), None)),
            }),
            _ => Expr::BinaryExpr(BinaryExpr {
                left: Box::new(Expr::Column(datafusion_common::Column::new_unqualified(col))),
                op: Operator::Eq,
                right: Box::new(Expr::Literal(ScalarValue::Boolean(Some(true)), None)),
            }),
        }
    }

    fn dyn_eq(&self, other: &dyn AnyQuery) -> bool {
        match other.as_any().downcast_ref::<Self>() {
            Some(o) => self == o,
            None => false,
        }
    }

    fn needs_recheck(&self) -> bool {
        // Some spatial queries may need rechecking depending on the precision of the index
        false
    }
}

/// A 2D bounding box for spatial queries
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, DeepSizeOf)]
pub struct BoundingBox {
    pub min_x: f64,
    pub min_y: f64,
    pub max_x: f64,
    pub max_y: f64,
}

impl BoundingBox {
    pub fn new(min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Self {
        Self { min_x, min_y, max_x, max_y }
    }

    pub fn format(&self) -> String {
        format!("BBOX({}, {}, {}, {})", self.min_x, self.min_y, self.max_x, self.max_y)
    }

    pub fn to_aabb(&self) -> AABB<[f64; 2]> {
        AABB::from_corners([self.min_x, self.min_y], [self.max_x, self.max_y])
    }
}

/// A 2D point for spatial queries
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, DeepSizeOf)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

impl Point {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub fn format(&self) -> String {
        format!("POINT({} {})", self.x, self.y)
    }

    pub fn to_array(&self) -> [f64; 2] {
        [self.x, self.y]
    }
}

/// Parameters for creating a spatial index
#[derive(Default, Debug, Clone)]
pub struct GeoIndexParams {
    /// Node capacity for the R-tree (default: 32)
    pub node_capacity: Option<usize>,
}

impl GeoIndexParams {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_node_capacity(mut self, capacity: usize) -> Self {
        self.node_capacity = Some(capacity);
        self
    }
}

impl IndexParams for GeoIndexParams {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn index_type(&self) -> IndexType {
        IndexType::Geo
    }

    fn index_name(&self) -> &str {
        LANCE_GEO_INDEX
    }
}

/// Trait for spatial indices that can answer geometric queries
#[async_trait]
pub trait GeoIndex: Send + Sync + std::fmt::Debug + Index + DeepSizeOf {
    /// Search the spatial index for geometries matching the query
    async fn search(
        &self,
        query: &SpatialQuery,
        metrics: &dyn MetricsCollector,
    ) -> Result<SearchResult>;

    /// Returns true if the query can be answered exactly by this index
    fn can_answer_exact(&self, query: &SpatialQuery) -> bool;

    /// Load the spatial index from storage
    async fn load(store: Arc<dyn IndexStore>) -> Result<Arc<Self>>
    where
        Self: Sized;

    /// Get the bounding box that contains all geometries in this index
    fn total_bounds(&self) -> Option<BoundingBox>;

    /// Get statistics about the spatial index
    fn size(&self) -> usize;
}