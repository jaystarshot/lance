#!/usr/bin/env python3
"""
Test script to demonstrate multi-bucket logging functionality.
"""

import os
import shutil
import pandas as pd
import lance

def test_multi_bucket_with_logging():
    """Test multi-bucket dataset creation with detailed logging."""
    
    print("🚀 Starting multi-bucket logging test...")
    
    # Use fixed test directory
    base_test_dir = "/Users/jay.narale/work/Uber/test"
    
    # Primary dataset location
    primary_uri = os.path.join(base_test_dir, "primary_bucket")
    
    # Additional data bucket locations  
    bucket2_uri = os.path.join(base_test_dir, "bucket2")
    bucket3_uri = os.path.join(base_test_dir, "bucket3")
    
    # Clear existing directories
    print("🧹 Clearing existing test directories...")
    for directory in [primary_uri, bucket2_uri, bucket3_uri]:
        if os.path.exists(directory):
            shutil.rmtree(directory)
            print(f"🧹 Removed: {directory}")
    
    # Create the bucket directories
    os.makedirs(primary_uri, exist_ok=True)
    os.makedirs(bucket2_uri, exist_ok=True) 
    os.makedirs(bucket3_uri, exist_ok=True)
    
    print(f"📁 Primary URI: {primary_uri}")
    print(f"📁 Bucket 2 URI: {bucket2_uri}")
    print(f"📁 Bucket 3 URI: {bucket3_uri}")
    
    # Create test data that will generate multiple fragments
    data = pd.DataFrame({
        'id': range(500),  # 500 rows
        'value': [f'value_{i}' for i in range(500)],
        'score': [i * 0.1 for i in range(500)]
    })
    
    print(f"📊 Created test data with {len(data)} rows")
    
    try:
        print("\n" + "="*50)
        print("🪣 CREATING MULTI-BUCKET DATASET...")
        print("="*50)
        
        # Create dataset with multi-bucket layout
        # data_bucket_uris registers buckets in manifest, target_bucket_uri specifies where to write
        dataset = lance.write_dataset(
            data, 
            primary_uri,
            mode="create",
            data_bucket_uris=[bucket2_uri, bucket3_uri],  # Register these buckets in manifest
            target_bucket_uri=bucket2_uri,  # Actually write data to bucket2
            max_rows_per_file=100  # Force multiple fragments (5 fragments total)
        )
        
        print("\n" + "="*50)
        print("✅ DATASET CREATION COMPLETE!")
        print("="*50)
        
        print(f"Dataset URI: {dataset.uri}")
        print(f"Dataset version: {dataset.version}")
        print(f"Schema: {dataset.schema}")
        
        # Verify we can read the data back
        result = dataset.to_table().to_pandas()
        print(f"Successfully read back {len(result)} rows")
        
        # Check file distribution across buckets
        print(f"\n📁 File distribution after CREATE:")
        for bucket_name, bucket_uri in [
            ("Primary", primary_uri),
            ("Bucket 2", bucket2_uri), 
            ("Bucket 3", bucket3_uri)
        ]:
            # Check both the bucket root and data subdirectory
            data_dir = os.path.join(bucket_uri, "data")
            if os.path.exists(data_dir):
                files = [f for f in os.listdir(data_dir) if f.endswith('.lance')]
                print(f"  {bucket_name} (data/): {len(files)} files")
                for file in files:
                    print(f"    - {file}")
            elif os.path.exists(bucket_uri):
                files = [f for f in os.listdir(bucket_uri) if f.endswith('.lance')]
                print(f"  {bucket_name}: {len(files)} files")
                for file in files:
                    print(f"    - {file}")
            else:
                print(f"  {bucket_name}: directory not found")
        
        # Test APPEND mode with target_bucket_uri
        print("\n" + "="*50)
        print("🪣 TESTING APPEND MODE...")
        print("="*50)
        
        # Create additional data to append
        append_data = pd.DataFrame({
            'id': range(500, 600),  # 100 more rows
            'value': [f'append_value_{i}' for i in range(500, 600)],
            'score': [i * 0.1 for i in range(500, 600)]
        })
        
        # Append to bucket3 this time
        # Note: In APPEND mode, we don't provide data_bucket_uris (bucket registry already exists)
        dataset = lance.write_dataset(
            append_data,
            dataset,  # Use existing dataset
            mode="append",
            target_bucket_uri=bucket3_uri,  # Write append data to bucket3
            max_rows_per_file=100
        )
        
        print(f"✅ APPEND complete! Total rows: {len(dataset.to_table().to_pandas())}")
        
        # Check file distribution after append
        print(f"\n📁 File distribution after APPEND:")
        for bucket_name, bucket_uri in [
            ("Primary", primary_uri),
            ("Bucket 2", bucket2_uri), 
            ("Bucket 3", bucket3_uri)
        ]:
            data_dir = os.path.join(bucket_uri, "data")
            if os.path.exists(data_dir):
                files = [f for f in os.listdir(data_dir) if f.endswith('.lance')]
                print(f"  {bucket_name} (data/): {len(files)} files")
                for file in files:
                    print(f"    - {file}")
            elif os.path.exists(bucket_uri):
                files = [f for f in os.listdir(bucket_uri) if f.endswith('.lance')]
                print(f"  {bucket_name}: {len(files)} files")
                for file in files:
                    print(f"    - {file}")
            else:
                print(f"  {bucket_name}: directory not found")
        
        # Final verification: Read all data to ensure bucket resolution works correctly
        print("\n" + "="*50)
        print("🔍 FINAL DATA VERIFICATION...")
        print("="*50)
        
        try:
            final_result = dataset.to_table().to_pandas()
            print(f"✅ Successfully read {len(final_result)} total rows from multi-bucket dataset")
            
            # Verify we have the expected data
            original_ids = set(range(500))  # Original CREATE data: 0-499
            append_ids = set(range(500, 600))  # APPEND data: 500-599
            expected_ids = original_ids | append_ids  # Combined: 0-599
            
            actual_ids = set(final_result['id'].tolist())
            
            if actual_ids == expected_ids:
                print("✅ Data integrity verified: All expected rows present")
                print(f"   - Original data (0-499): {len(original_ids & actual_ids)} rows")
                print(f"   - Appended data (500-599): {len(append_ids & actual_ids)} rows")
            else:
                missing_ids = expected_ids - actual_ids
                extra_ids = actual_ids - expected_ids
                print(f"❌ Data integrity issue:")
                if missing_ids:
                    print(f"   - Missing IDs: {sorted(list(missing_ids))[:10]}{'...' if len(missing_ids) > 10 else ''}")
                if extra_ids:
                    print(f"   - Extra IDs: {sorted(list(extra_ids))[:10]}{'...' if len(extra_ids) > 10 else ''}")
                return False
                
            # Show sample data from both buckets
            print(f"\n📊 Sample data verification:")
            print(f"   - First row (from bucket2): ID={final_result.iloc[0]['id']}, value='{final_result.iloc[0]['value']}'")
            print(f"   - Last row (from bucket3): ID={final_result.iloc[-1]['id']}, value='{final_result.iloc[-1]['value']}'")
            
        except Exception as e:
            print(f"❌ Final read verification failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_validation_errors():
    """Test that invalid multi-bucket configurations are properly rejected."""
    
    print("\n" + "="*50)
    print("🚨 TESTING VALIDATION ERRORS...")  
    print("="*50)
    
    base_test_dir = "/Users/jay.narale/work/Uber/test"
    dataset_uri = os.path.join(base_test_dir, "validation_test")
    bucket2_uri = os.path.join(base_test_dir, "bucket2")
    bucket3_uri = os.path.join(base_test_dir, "bucket3")
    
    # Clear existing directory
    if os.path.exists(dataset_uri):
        shutil.rmtree(dataset_uri)
    
    os.makedirs(dataset_uri, exist_ok=True)
    
    # Create test data
    data = pd.DataFrame({
        'id': range(10),
        'value': [f'value_{i}' for i in range(10)]
    })
    
    # Test 1: data_bucket_uris provided but no target_bucket_uri
    print("🧪 Test 1: data_bucket_uris without target_bucket_uri (should fail)")
    try:
        lance.write_dataset(
            data, 
            dataset_uri,
            mode="create",
            data_bucket_uris=[bucket2_uri, bucket3_uri],  # Provided
            # target_bucket_uri missing!                   # Missing
            max_rows_per_file=100
        )
        print("❌ ERROR: Should have failed but didn't!")
        return False
    except Exception as e:
        if "target_bucket_uri must also be specified" in str(e):
            print("✅ Correctly rejected: data_bucket_uris without target_bucket_uri")
        else:
            print(f"❌ Failed with unexpected error: {e}")
            return False
    
    # Test 2: target_bucket_uri not in data_bucket_uris
    print("\n🧪 Test 2: target_bucket_uri not in data_bucket_uris (should fail)")
    try:
        invalid_bucket = os.path.join(base_test_dir, "invalid_bucket")
        lance.write_dataset(
            data, 
            dataset_uri,
            mode="create",
            data_bucket_uris=[bucket2_uri, bucket3_uri],  # bucket2, bucket3
            target_bucket_uri=invalid_bucket,             # invalid_bucket (not in list)
            max_rows_per_file=100
        )
        print("❌ ERROR: Should have failed but didn't!")
        return False
    except Exception as e:
        if "not found in data_bucket_uris" in str(e):
            print("✅ Correctly rejected: target_bucket_uri not in data_bucket_uris")
        else:
            print(f"❌ Failed with unexpected error: {e}")
            return False
    
    # Test 3: target_bucket_uri provided but no data_bucket_uris in CREATE mode
    print("\n🧪 Test 3: target_bucket_uri without data_bucket_uris in CREATE mode (should fail)")
    try:
        lance.write_dataset(
            data, 
            dataset_uri,
            mode="create",
            # data_bucket_uris missing!                    # Missing
            target_bucket_uri=bucket2_uri,               # Provided
            max_rows_per_file=100
        )
        print("❌ ERROR: Should have failed but didn't!")
        return False
    except Exception as e:
        if "data_bucket_uris must also be specified" in str(e):
            print("✅ Correctly rejected: target_bucket_uri without data_bucket_uris in CREATE mode")
        else:
            print(f"❌ Failed with unexpected error: {e}")
            return False
    
    # Test 4: data_bucket_uris provided in APPEND mode (should fail)
    print("\n🧪 Test 4: data_bucket_uris in APPEND mode (should fail)")
    
    # First create a dataset to append to
    try:
        base_dataset = lance.write_dataset(
            data, 
            dataset_uri,
            mode="create",
            data_bucket_uris=[bucket2_uri, bucket3_uri],
            target_bucket_uri=bucket2_uri,
            max_rows_per_file=100
        )
    except Exception as e:
        print(f"❌ Failed to create base dataset for append test: {e}")
        return False
    
    # Now try to append with data_bucket_uris (should fail)
    try:
        lance.write_dataset(
            data, 
            base_dataset,
            mode="append",
            data_bucket_uris=[bucket2_uri, bucket3_uri],  # Should not be provided in APPEND
            target_bucket_uri=bucket3_uri,
            max_rows_per_file=100
        )
        print("❌ ERROR: Should have failed but didn't!")
        return False
    except Exception as e:
        if "should not be provided in Append mode" in str(e):
            print("✅ Correctly rejected: data_bucket_uris in APPEND mode")
        else:
            print(f"❌ Failed with unexpected error: {e}")
            return False
    
    print("✅ All validation tests passed!")
    return True

def test_single_bucket_logging():
    """Test single-bucket mode for comparison."""
    
    print("\n" + "="*50)
    print("🪣 TESTING SINGLE-BUCKET MODE...")  
    print("="*50)
    
    base_test_dir = "/Users/jay.narale/work/Uber/test"
    dataset_uri = os.path.join(base_test_dir, "single_bucket")
    
    # Clear existing directory
    if os.path.exists(dataset_uri):
        shutil.rmtree(dataset_uri)
        print(f"🧹 Removed: {dataset_uri}")
    
    os.makedirs(dataset_uri, exist_ok=True)
    
    # Create smaller test data
    data = pd.DataFrame({
        'id': range(200),
        'value': [f'value_{i}' for i in range(200)]
    })
    
    try:
        dataset = lance.write_dataset(
            data, 
            dataset_uri,
            max_rows_per_file=100  # 2 fragments
        )
        
        print("✅ Single-bucket dataset created successfully!")
        print(f"Records: {len(dataset.to_table().to_pandas())}")
        return True
        
    except Exception as e:
        print(f"❌ Single-bucket error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Multi-Bucket Logging Test")
    print("=" * 60)
    
    success = True
    
    # Test multi-bucket with logging
    success &= test_multi_bucket_with_logging()
    
    # Test validation errors
    success &= test_validation_errors()
    
    # Test single-bucket for comparison
    success &= test_single_bucket_logging()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 All logging tests completed!")
    else:
        print("💥 Some tests failed!")
        exit(1)
