# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""
Tests for multi-bucket dataset functionality.
"""

import tempfile
import shutil
import uuid
from pathlib import Path

import lance
import pandas as pd
import pyarrow as pa
import pytest


class TestMultiBucket:
    """Test multi-bucket dataset functionality with local file system."""

    def setup_method(self):
        """Set up test directories for each test."""
        self.test_dir = tempfile.mkdtemp()
        self.test_id = str(uuid.uuid4())[:8]
        
        # Create primary and additional bucket directories with file:// URIs
        self.primary_uri = Path(self.test_dir).as_uri() + "/primary"
        self.bucket1_uri = Path(self.test_dir).as_uri() + f"/bucket1_{self.test_id}"
        self.bucket2_uri = Path(self.test_dir).as_uri() + f"/bucket2_{self.test_id}"
        self.bucket3_uri = Path(self.test_dir).as_uri() + f"/bucket3_{self.test_id}"
        
        # Create directories (convert back to paths for mkdir)
        for uri in [self.primary_uri, self.bucket1_uri, self.bucket2_uri, self.bucket3_uri]:
            # Convert file:// URI back to path for directory creation
            path = Path(uri.replace("file://", ""))
            path.mkdir(parents=True, exist_ok=True)

    def teardown_method(self):
        """Clean up test directories after each test."""
        if hasattr(self, 'test_dir'):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def create_test_data(self, num_rows=500):
        """Create test data for multi-bucket tests."""
        return pd.DataFrame({
            'id': range(num_rows),
            'value': [f'value_{i}' for i in range(num_rows)],
            'score': [i * 0.1 for i in range(num_rows)]
        })

    def test_multi_bucket_create_and_read(self):
        """Test creating a multi-bucket dataset and reading it back."""
        data = self.create_test_data(500)
        
        # Create dataset with multi-bucket layout
        dataset = lance.write_dataset(
            data, 
            self.primary_uri,
            mode="create",
            data_path_uris=[self.bucket1_uri, self.bucket2_uri, self.bucket3_uri],
            target_path_uri=self.bucket2_uri,  # Write data to bucket2
            max_rows_per_file=100  # Force multiple fragments
        )
        
        assert dataset is not None
        assert dataset.uri == self.primary_uri
        
        # Verify we can read the data back
        result = dataset.to_table().to_pandas()
        assert len(result) == 500
        
        # Verify data integrity
        pd.testing.assert_frame_equal(
            result.sort_values('id').reset_index(drop=True),
            data.sort_values('id').reset_index(drop=True)
        )

    def test_multi_bucket_append_mode(self):
        """Test appending data to a multi-bucket dataset."""
        # Create initial dataset
        initial_data = self.create_test_data(300)
        
        dataset = lance.write_dataset(
            initial_data, 
            self.primary_uri,
            mode="create",
            data_path_uris=[self.bucket1_uri, self.bucket2_uri],
            target_path_uri=self.bucket1_uri,
            max_rows_per_file=100
        )
        
        # Create additional data to append
        append_data = pd.DataFrame({
            'id': range(300, 400),
            'value': [f'append_value_{i}' for i in range(300, 400)],
            'score': [i * 0.1 for i in range(300, 400)]
        })
        
        # Append to different bucket
        updated_dataset = lance.write_dataset(
            append_data,
            dataset,
            mode="append",
            target_path_uri=self.bucket2_uri,  # Write to bucket2
            max_rows_per_file=50
        )
        
        # Verify total data
        result = updated_dataset.to_table().to_pandas()
        assert len(result) == 400
        
        # Verify all data is present
        expected_ids = set(range(400))
        actual_ids = set(result['id'].tolist())
        assert actual_ids == expected_ids

    def test_multi_bucket_overwrite_mode(self):
        """Test overwriting data in a multi-bucket dataset."""
        # Create initial dataset
        initial_data = self.create_test_data(200)
        
        dataset = lance.write_dataset(
            initial_data, 
            self.primary_uri,
            mode="create",
            data_path_uris=[self.bucket1_uri, self.bucket2_uri],
            target_path_uri=self.bucket1_uri,
            max_rows_per_file=100
        )
        
        # Create new data for overwrite
        overwrite_data = pd.DataFrame({
            'id': range(100, 150),
            'value': [f'overwrite_value_{i}' for i in range(100, 150)],
            'score': [i * 0.2 for i in range(100, 150)]  # Different scores
        })
        
        # Overwrite with different target bucket
        updated_dataset = lance.write_dataset(
            overwrite_data,
            dataset,
            mode="overwrite",
            target_path_uri=self.bucket2_uri,
            max_rows_per_file=25
        )
        
        # Verify overwritten data
        result = updated_dataset.to_table().to_pandas()
        assert len(result) == 50
        
        # Verify data content
        expected_ids = set(range(100, 150))
        actual_ids = set(result['id'].tolist())
        assert actual_ids == expected_ids
        
        # Verify scores were updated
        assert all(result['score'] == result['id'] * 0.2)

    def test_multi_bucket_validation_errors(self):
        """Test validation errors for invalid multi-bucket configurations."""
        data = self.create_test_data(100)
        
        # Test CREATE mode: target_path_uri not in data_path_uris
        with pytest.raises(Exception, match="target_path_uri.*not found"):
            lance.write_dataset(
                data,
                self.primary_uri,
                mode="create",
                data_path_uris=[self.bucket1_uri, self.bucket2_uri],
                target_path_uri=self.bucket3_uri  # Not in data_path_uris
            )
        
        # Test CREATE mode: target_path_uri without data_path_uris
        with pytest.raises(Exception, match="data_path_uris must also be specified"):
            lance.write_dataset(
                data,
                self.primary_uri,
                mode="create",
                target_path_uri=self.bucket1_uri  # No data_path_uris
            )

    def test_multi_bucket_fragment_distribution(self):
        """Test that fragments are correctly distributed and readable."""
        data = self.create_test_data(1000)
        
        dataset = lance.write_dataset(
            data, 
            self.primary_uri,
            mode="create",
            data_path_uris=[self.bucket1_uri, self.bucket2_uri],
            target_path_uri=self.bucket1_uri,
            max_rows_per_file=100  # Should create 10 fragments
        )
        
        # Verify fragment count
        fragments = list(dataset.get_fragments())
        assert len(fragments) >= 10  # At least 10 fragments
        
        # Verify all data is readable
        result = dataset.to_table().to_pandas()
        assert len(result) == 1000
        
        # Test scanning with filters
        filtered_result = dataset.scanner(filter="id < 500").to_table().to_pandas()
        assert len(filtered_result) == 500
        assert all(filtered_result['id'] < 500)

    def test_multi_bucket_schema_consistency(self):
        """Test that schema is consistent across multi-bucket operations."""
        # Create dataset with specific schema
        schema = pa.schema([
            pa.field("id", pa.int64()),
            pa.field("name", pa.string()),
            pa.field("score", pa.float64()),
        ])
        
        data = pa.table({
            "id": [1, 2, 3, 4, 5],
            "name": ["a", "b", "c", "d", "e"],
            "score": [1.0, 2.0, 3.0, 4.0, 5.0]
        }, schema=schema)
        
        dataset = lance.write_dataset(
            data,
            self.primary_uri,
            mode="create",
            data_path_uris=[self.bucket1_uri, self.bucket2_uri],
            target_path_uri=self.bucket1_uri
        )
        
        # Verify schema
        assert dataset.schema == schema
        
        # Append with same schema to different bucket
        append_data = pa.table({
            "id": [6, 7, 8],
            "name": ["f", "g", "h"],
            "score": [6.0, 7.0, 8.0]
        }, schema=schema)
        
        updated_dataset = lance.write_dataset(
            append_data,
            dataset,
            mode="append",
            target_path_uri=self.bucket2_uri
        )
        
        # Schema should remain consistent
        assert updated_dataset.schema == schema
        
        # Verify all data
        result = updated_dataset.to_table()
        assert len(result) == 8
        assert result.schema == schema

    def test_multi_bucket_empty_buckets(self):
        """Test behavior with empty bucket configurations."""
        data = self.create_test_data(100)
        
        # Test with single bucket (should work like normal dataset)
        dataset = lance.write_dataset(
            data,
            self.primary_uri,
            mode="create",
            data_path_uris=[self.bucket1_uri],
            target_path_uri=self.bucket1_uri
        )
        
        result = dataset.to_table().to_pandas()
        assert len(result) == 100
        
        # Verify data integrity
        pd.testing.assert_frame_equal(
            result.sort_values('id').reset_index(drop=True),
            data.sort_values('id').reset_index(drop=True)
        )

    def test_multi_bucket_dataset_reopening(self):
        """Test reopening a multi-bucket dataset."""
        data = self.create_test_data(300)
        
        # Create multi-bucket dataset
        dataset = lance.write_dataset(
            data, 
            self.primary_uri,
            mode="create",
            data_path_uris=[self.bucket1_uri, self.bucket2_uri],
            target_path_uri=self.bucket1_uri,
            max_rows_per_file=100
        )
        
        # Close and reopen dataset
        del dataset
        reopened_dataset = lance.dataset(self.primary_uri)
        
        # Verify data is still accessible
        result = reopened_dataset.to_table().to_pandas()
        assert len(result) == 300
        
        # Verify we can still append to different buckets
        append_data = self.create_test_data(50)
        append_data['id'] = range(300, 350)  # Avoid ID conflicts
        
        updated_dataset = lance.write_dataset(
            append_data,
            reopened_dataset,
            mode="append",
            target_path_uri=self.bucket2_uri
        )
        
        final_result = updated_dataset.to_table().to_pandas()
        assert len(final_result) == 350
