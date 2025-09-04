"""
Validation tests to ensure the testing infrastructure is working correctly.
These tests verify that the testing setup, fixtures, and basic functionality work as expected.
"""

import pytest
import os
import sys
import sqlite3
import pandas as pd
from pathlib import Path

# Add the dataset_builder to the path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


@pytest.mark.unit
def test_pytest_configuration():
    """Test that pytest is configured correctly."""
    # This test should pass if pytest is working
    assert True


@pytest.mark.unit
def test_coverage_configuration():
    """Test that coverage configuration is working."""
    # Test that coverage is tracking this file
    assert __file__ is not None
    assert os.path.exists(__file__)


@pytest.mark.unit
def test_pandas_import():
    """Test that pandas is available and working."""
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    assert len(df) == 3
    assert list(df.columns) == ['a', 'b']


@pytest.mark.unit
def test_numpy_import():
    """Test that numpy is available and working."""
    import numpy as np
    arr = np.array([1, 2, 3])
    assert arr.shape == (3,)
    assert arr.sum() == 6


def test_temp_dir_fixture(temp_dir):
    """Test that the temp_dir fixture works correctly."""
    assert temp_dir.exists()
    assert temp_dir.is_dir()
    
    # Create a test file
    test_file = temp_dir / "test.txt"
    test_file.write_text("test content")
    
    assert test_file.exists()
    assert test_file.read_text() == "test content"


def test_sample_dataframe_fixture(sample_dataframe):
    """Test that the sample_dataframe fixture works correctly."""
    assert isinstance(sample_dataframe, pd.DataFrame)
    assert len(sample_dataframe) == 5
    assert 'subject_id' in sample_dataframe.columns
    assert 'hadm_id' in sample_dataframe.columns
    assert 'value' in sample_dataframe.columns
    assert 'timestamp' in sample_dataframe.columns


def test_sample_db_fixture(sample_db_path):
    """Test that the sample database fixture works correctly."""
    assert os.path.exists(sample_db_path)
    
    # Connect to the database and verify data
    conn = sqlite3.connect(sample_db_path)
    cursor = conn.cursor()
    
    # Test patients table
    cursor.execute("SELECT COUNT(*) FROM patients")
    patient_count = cursor.fetchone()[0]
    assert patient_count == 3
    
    # Test admissions table
    cursor.execute("SELECT COUNT(*) FROM admissions")
    admission_count = cursor.fetchone()[0]
    assert admission_count == 3
    
    # Test data integrity
    cursor.execute("""
        SELECT p.subject_id, p.gender, a.hadm_id 
        FROM patients p 
        JOIN admissions a ON p.subject_id = a.subject_id 
        WHERE p.subject_id = 1
    """)
    result = cursor.fetchone()
    assert result is not None
    assert result[0] == 1  # subject_id
    assert result[1] == 'M'  # gender
    assert result[2] == 100  # hadm_id
    
    conn.close()


def test_mock_config_fixture(mock_config):
    """Test that the mock_config fixture works correctly."""
    assert isinstance(mock_config, dict)
    assert 'database_path' in mock_config
    assert 'output_dir' in mock_config
    assert 'batch_size' in mock_config
    assert mock_config['batch_size'] == 100


def test_mock_cursor_fixture(mock_cursor):
    """Test that the mock_cursor fixture works correctly."""
    # Test fetchall
    results = mock_cursor.fetchall()
    assert len(results) == 3
    assert results[0] == (1, 'M', 65, None)
    
    # Test fetchone
    result = mock_cursor.fetchone()
    assert result == (1, 'M', 65, None)


def test_sample_json_data_fixture(sample_json_data):
    """Test that the sample_json_data fixture works correctly."""
    assert isinstance(sample_json_data, dict)
    assert 'patients' in sample_json_data
    assert len(sample_json_data['patients']) == 1
    
    patient = sample_json_data['patients'][0]
    assert patient['subject_id'] == 1
    assert 'admissions' in patient
    assert len(patient['admissions']) == 1


@pytest.mark.integration
def test_integration_marker():
    """Test that integration test markers work."""
    # This is marked as an integration test
    assert True


@pytest.mark.slow
def test_slow_marker():
    """Test that slow test markers work."""
    # This is marked as a slow test
    assert True


@pytest.mark.unit
class TestClassExample:
    """Example test class to verify class-based testing works."""
    
    def test_class_method(self):
        """Test that class-based tests work."""
        assert 1 + 1 == 2
    
    def test_class_method_with_fixture(self, sample_dataframe):
        """Test that fixtures work with class-based tests."""
        assert len(sample_dataframe) > 0


def test_project_structure():
    """Test that the expected project structure exists."""
    project_root = Path(__file__).parent.parent
    
    # Check that the dataset_builder directory exists
    dataset_builder_path = project_root / "dataset_builder"
    assert dataset_builder_path.exists(), "dataset_builder directory should exist"
    
    # Check that some expected Python files exist
    expected_files = [
        "generate_answer.py",
        "preprocess_cohort.py", 
        "preprocess_db.py",
        "preprocess_label.py"
    ]
    
    for file_name in expected_files:
        file_path = dataset_builder_path / file_name
        assert file_path.exists(), f"{file_name} should exist in dataset_builder/"


def test_pyproject_toml_exists():
    """Test that pyproject.toml exists and contains expected content."""
    project_root = Path(__file__).parent.parent
    pyproject_path = project_root / "pyproject.toml"
    
    assert pyproject_path.exists(), "pyproject.toml should exist"
    
    content = pyproject_path.read_text()
    assert "[tool.poetry]" in content
    assert "[tool.pytest.ini_options]" in content
    assert "[tool.coverage.run]" in content