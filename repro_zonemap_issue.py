"""
Reproduction script for GitHub issue #5130: Zonemap index reading too many files

This script demonstrates that a zonemap index on a string column is scanning
more fragments than necessary when filtering for a specific value.

Expected behavior: When filtering for a specific test_id, only 1 fragment should be scanned
Actual behavior: Multiple fragments are scanned even though each fragment contains only one unique test_id

Usage:
    # From the lance repository root
    cd python
    # Make sure lance is built and installed
    maturin develop
    # Run the reproduction script
    python3 ../repro_zonemap_issue.py

The script will:
1. Create two datasets with 1000 fragments each (100 rows per fragment)
2. Each fragment contains only ONE unique test_id value
3. Test 1: Create a BITMAP index and query (control test - should scan 1 fragment)
4. Test 2: Create a ZONEMAP index and query (may have bug - scans multiple fragments)
5. Use scan_stats_callback to track exactly how many fragments are scanned
6. Compare results and report if issue #5130 is reproduced
"""

import lance
import pyarrow as pa
import tempfile
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

def create_single_fragment(uri, fragment_id, rows_per_fragment):
    """Create a single fragment with a unique test_id."""
    test_id = f"TEST_ID_{fragment_id:06d}"
    
    # Create struct column with nested data
    metadata_struct = pa.StructArray.from_arrays(
        [
            pa.array([f"device_{fragment_id}"] * rows_per_fragment),
            pa.array([fragment_id * 1000 + j for j in range(rows_per_fragment)]),
            pa.array([fragment_id % 10] * rows_per_fragment)
        ],
        names=["device_id", "timestamp", "status"]
    )
    
    table = pa.table({
        "test_id": [test_id] * rows_per_fragment,
        "value": list(range(rows_per_fragment)),
        "metadata": metadata_struct
    })
    
    # Write the fragment (append mode works for both create and append)
    lance.write_dataset(table, uri, mode="append")
    
    # Log progress
    if (fragment_id + 1) % 500 == 0:
        print(f"  Created fragment {fragment_id + 1}")

def create_dataset_with_fragments(uri, num_fragments=2000, rows_per_fragment=500, num_threads=32):
    """
    Create a dataset with many fragments where each fragment has a unique test_id value.
    Uses parallel threads to speed up creation.
    
    Args:
        uri: Dataset URI
        num_fragments: Number of fragments to create
        rows_per_fragment: Number of rows per fragment
        num_threads: Number of parallel threads to use
    """
    print(f"Creating dataset with {num_fragments} fragments ({rows_per_fragment} rows each) using {num_threads} threads...")
    
    # Create first fragment to initialize the dataset
    test_id = f"TEST_ID_{0:06d}"
    metadata_struct = pa.StructArray.from_arrays(
        [
            pa.array([f"device_0"] * rows_per_fragment),
            pa.array([j for j in range(rows_per_fragment)]),
            pa.array([0] * rows_per_fragment)
        ],
        names=["device_id", "timestamp", "status"]
    )
    table = pa.table({
        "test_id": [test_id] * rows_per_fragment,
        "value": list(range(rows_per_fragment)),
        "metadata": metadata_struct
    })
    lance.write_dataset(table, uri, mode="create")
    print(f"  Created fragment 1")
    
    # Create remaining fragments in parallel
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for i in range(1, num_fragments):
            future = executor.submit(create_single_fragment, uri, i, rows_per_fragment)
            futures.append(future)
        
        # Wait for all fragments to complete
        for future in as_completed(futures):
            future.result()  # Ensure no exceptions
    
    dataset = lance.dataset(uri)
    print(f"Dataset created with {len(dataset.get_fragments())} fragments")
    return dataset

def create_zonemap_index(dataset):
    """Create a zonemap index on the test_id column."""
    print("\nCreating zonemap index on test_id column...")
    dataset.create_scalar_index("test_id", "ZONEMAP")
    print("Zonemap index created")

def create_bitmap_index(dataset):
    """Create a bitmap index on the test_id column."""
    print("\nCreating bitmap index on test_id column...")
    dataset.create_scalar_index("test_id", "BITMAP")
    print("Bitmap index created")

def drop_index(dataset, column_name):
    """Drop the index on the specified column."""
    print(f"\nDropping index on {column_name} column...")
    # Get the index name for the column
    indices = dataset.list_indices()
    for idx in indices:
        # Check different possible key names
        idx_columns = idx.get('columns', idx.get('column', []))
        if isinstance(idx_columns, str):
            idx_columns = [idx_columns]
        
        if column_name in idx_columns or idx_columns == [column_name]:
            idx_name = idx.get('name', idx.get('index_name'))
            dataset.drop_index(idx_name)
            print(f"Dropped index: {idx_name}")
            return
    print(f"No index found for column {column_name}")

def test_filter_query(dataset, target_test_id, index_type):
    """
    Test filtering for a specific test_id and verify fragment scanning behavior.
    
    Args:
        dataset: Lance dataset
        target_test_id: The test_id value to filter for
        index_type: Type of index being tested (for display purposes)
    """
    print(f"\n{'='*60}")
    print(f"Testing with {index_type} index")
    print(f"{'='*60}")
    print(f"Querying for test_id='{target_test_id}'...")
    
    # Track scan statistics
    
    def scan_stats_callback(stats: lance.ScanStatistics):
        scanner._scan_stats = stats
    
    # Create scanner with filter
    filter_expr = f"test_id='{target_test_id}'"
    scanner = dataset.scanner(
        filter=filter_expr,
        scan_stats_callback=scan_stats_callback
    )
    
    # Execute scan
    result = scanner.to_table()
    
    print(f"\nResults:")
    print(f"Scan statistics: {scanner._scan_stats}")
    
    return scanner._scan_stats

def main():
    """Main function to reproduce the issue."""
    # Create temporary directory for dataset
    with tempfile.TemporaryDirectory() as tmpdir:
        uri = os.path.join(tmpdir, "test_dataset")
        
        # Create ONE dataset that we'll use for both tests
        print("="*60)
        print("SETUP: Creating dataset")
        print("="*60)
        dataset = create_dataset_with_fragments(uri)
        
        # Query for the test_id in fragment 500 (middle of the dataset)
        target_test_id = "TEST_ID_000500"
        
        # =====================================================================
        # Test 1: BITMAP Index (Expected to work correctly)
        # =====================================================================
        print("\n" + "="*60)
        print("TEST 1: BITMAP INDEX (Control - Expected to pass)")
        print("="*60)
        
        create_bitmap_index(dataset)
        dataset = lance.dataset(uri)  # Reload to use the index
        bitmap_stats = test_filter_query(dataset, target_test_id, "BITMAP")
        
        # =====================================================================
        # Test 2: ZONEMAP Index (May have the bug)
        # =====================================================================
        print("\n" + "="*60)
        print("TEST 2: ZONEMAP INDEX (Testing for issue #5130)")
        print("="*60)
        
        # Drop the bitmap index first
        drop_index(dataset, "test_id")
        dataset = lance.dataset(uri)  # Reload after dropping index
        
        # Create zonemap index on the SAME dataset
        create_zonemap_index(dataset)
        dataset = lance.dataset(uri)  # Reload to use the new index
        zonemap_stats = test_filter_query(dataset, target_test_id, "ZONEMAP")
        
        # =====================================================================
        # Final Summary
        # =====================================================================
        print("\n" + "="*60)
        print("FINAL SUMMARY")
        print("="*60)
        print(f"Dataset: {uri}")
        print(f"Total fragments: {len(dataset.get_fragments())}")
        print(f"Target test_id: {target_test_id}")
        
        bitmap_frags = bitmap_stats.all_counts["fragments_scanned"]
        zonemap_frags = zonemap_stats.all_counts["fragments_scanned"]
        
        print(f"\nBITMAP Index:")
        print(f"  Fragments scanned: {bitmap_frags}")
        print(f"  Rows scanned: {bitmap_stats.all_counts['rows_scanned']}")
        print(f"  Status: {'✓ PASS' if bitmap_frags == 1 else '✗ FAIL'}")
        
        print(f"\nZONEMAP Index:")
        print(f"  Fragments scanned: {zonemap_frags}")
        print(f"  Rows scanned: {zonemap_stats.all_counts['rows_scanned']}")
        print(f"  Status: {'✓ PASS' if zonemap_frags == 1 else '✗ FAIL'}")
        
        print("\n" + "="*60)
        if bitmap_frags == 1 and zonemap_frags > 1:
            print("⚠️  ISSUE #5130 REPRODUCED!")
            print("="*60)
            print("✓ Bitmap index correctly scans only 1 fragment")
            print(f"✗ Zonemap index incorrectly scans {zonemap_frags} fragments")
            print("\nEach fragment contains only one unique test_id value,")
            print("so the query should only scan 1 fragment for both index types.")
        elif bitmap_frags == 1 and zonemap_frags == 1:
            print("✓ BOTH INDICES WORKING CORRECTLY!")
            print("="*60)
            print("Issue may have been fixed or not reproduced in this environment.")
        else:
            print("⚠️  UNEXPECTED RESULTS")
            print("="*60)
            print("Both indices may have issues or the test setup needs adjustment.")

if __name__ == "__main__":
    main()

