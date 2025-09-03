#[cfg(test)]
mod tests {
    use crate::geo::{BoundingBox, Point, SpatialQuery};
    use crate::geo::rtree::RTreeEntry;
    
    #[test]
    fn test_bounding_box_creation() {
        let bbox = BoundingBox::new(0.0, 0.0, 10.0, 10.0);
        assert_eq!(bbox.min_x, 0.0);
        assert_eq!(bbox.min_y, 0.0);
        assert_eq!(bbox.max_x, 10.0);
        assert_eq!(bbox.max_y, 10.0);
    }

    #[test]
    fn test_point_creation() {
        let point = Point::new(5.0, 7.0);
        assert_eq!(point.x, 5.0);
        assert_eq!(point.y, 7.0);
    }

    #[test]
    fn test_spatial_query_creation() {
        let bbox = BoundingBox::new(0.0, 0.0, 10.0, 10.0);
        let query = SpatialQuery::Intersects(bbox.clone());
        
        match query {
            SpatialQuery::Intersects(inner_bbox) => {
                assert_eq!(inner_bbox, bbox);
            }
            _ => panic!("Expected Intersects query"),
        }
    }

    #[test]
    fn test_point_format() {
        let point = Point::new(1.5, 2.5);
        assert_eq!(point.format(), "POINT(1.5 2.5)");
    }

    #[test]
    fn test_bbox_format() {
        let bbox = BoundingBox::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(bbox.format(), "BBOX(1, 2, 3, 4)");
    }

    #[test]
    fn test_rtree_entry_serialization() {
        // Test serialization and deserialization of R-tree entries
        let entries = vec![
            RTreeEntry {
                bbox: BoundingBox::new(0.0, 0.0, 1.0, 1.0),
                row_id: 100,
            },
            RTreeEntry {
                bbox: BoundingBox::new(2.0, 2.0, 3.0, 3.0),
                row_id: 200,
            },
            RTreeEntry {
                bbox: BoundingBox::new(5.0, 5.0, 6.0, 6.0),
                row_id: 300,
            },
        ];

        // Test JSON serialization
        let serialized = serde_json::to_vec(&entries).expect("Failed to serialize entries");
        println!("✅ Serialization completed, {} bytes", serialized.len());

        // Test JSON deserialization
        let deserialized: Vec<RTreeEntry> = serde_json::from_slice(&serialized)
            .expect("Failed to deserialize entries");
        println!("✅ Deserialization completed, {} entries", deserialized.len());

        // Verify the data is preserved
        assert_eq!(entries.len(), deserialized.len());
        for (original, loaded) in entries.iter().zip(deserialized.iter()) {
            assert_eq!(original.row_id, loaded.row_id);
            assert_eq!(original.bbox, loaded.bbox);
        }
        
        println!("✅ Serialization test passed - all data preserved");
    }

    #[test]
    fn test_rtree_entry_functionality() {
        // Test basic RTreeEntry creation and properties
        let entry = RTreeEntry {
            bbox: BoundingBox::new(1.0, 2.0, 3.0, 4.0),
            row_id: 42,
        };

        assert_eq!(entry.row_id, 42);
        assert_eq!(entry.bbox.min_x, 1.0);
        assert_eq!(entry.bbox.min_y, 2.0);
        assert_eq!(entry.bbox.max_x, 3.0);
        assert_eq!(entry.bbox.max_y, 4.0);

        // Test that RTreeEntry implements Clone
        let cloned_entry = entry.clone();
        assert_eq!(entry.row_id, cloned_entry.row_id);
        assert_eq!(entry.bbox, cloned_entry.bbox);

        println!("✅ RTreeEntry functionality test passed");
    }
}