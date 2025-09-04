"""
Shared pytest fixtures for the medical dataset builder test suite.
"""

import os
import tempfile
import pytest
import pandas as pd
import sqlite3
from pathlib import Path
from unittest.mock import Mock, MagicMock


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_db_path(temp_dir):
    """Create a temporary SQLite database for testing."""
    db_path = temp_dir / "test_database.sqlite"
    conn = sqlite3.connect(str(db_path))
    
    # Create sample tables that might be used in the project
    conn.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            subject_id INTEGER PRIMARY KEY,
            gender TEXT,
            anchor_age INTEGER,
            dod DATE
        )
    """)
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS admissions (
            hadm_id INTEGER PRIMARY KEY,
            subject_id INTEGER,
            admittime DATETIME,
            dischtime DATETIME,
            admission_type TEXT,
            FOREIGN KEY (subject_id) REFERENCES patients (subject_id)
        )
    """)
    
    # Insert some sample data
    conn.execute("""
        INSERT INTO patients (subject_id, gender, anchor_age) 
        VALUES (1, 'M', 65), (2, 'F', 45), (3, 'M', 78)
    """)
    
    conn.execute("""
        INSERT INTO admissions (hadm_id, subject_id, admittime, dischtime, admission_type)
        VALUES 
        (100, 1, '2105-01-01 12:00:00', '2105-01-05 10:00:00', 'EMERGENCY'),
        (101, 2, '2105-02-15 09:30:00', '2105-02-20 14:15:00', 'ELECTIVE'),
        (102, 3, '2105-03-10 16:45:00', '2105-03-12 08:20:00', 'URGENT')
    """)
    
    conn.commit()
    conn.close()
    
    yield str(db_path)


@pytest.fixture
def sample_dataframe():
    """Create a sample pandas DataFrame for testing."""
    return pd.DataFrame({
        'subject_id': [1, 2, 3, 4, 5],
        'hadm_id': [100, 101, 102, 103, 104],
        'value': [10.5, 20.1, 15.7, 8.9, 12.3],
        'timestamp': pd.to_datetime([
            '2105-01-01 12:00:00',
            '2105-02-15 09:30:00', 
            '2105-03-10 16:45:00',
            '2105-04-05 14:20:00',
            '2105-05-12 11:10:00'
        ])
    })


@pytest.fixture
def mock_config():
    """Mock configuration dictionary for testing."""
    return {
        'database_path': '/tmp/test.db',
        'output_dir': '/tmp/output',
        'batch_size': 100,
        'time_window': 24,  # hours
        'temperature_range': (35.5, 38.1),
        'heart_rate_range': (60.0, 100.0),
        'bp_systolic_range': (90.0, 120.0),
        'bp_diastolic_range': (60.0, 90.0)
    }


@pytest.fixture
def mock_cursor():
    """Mock database cursor for testing."""
    cursor = Mock()
    cursor.fetchall.return_value = [
        (1, 'M', 65, None),
        (2, 'F', 45, None),
        (3, 'M', 78, '2105-12-15')
    ]
    cursor.fetchone.return_value = (1, 'M', 65, None)
    return cursor


@pytest.fixture
def mock_connection():
    """Mock database connection for testing."""
    connection = Mock()
    connection.cursor.return_value = Mock()
    return connection


@pytest.fixture
def sample_json_data():
    """Sample JSON data structure for testing."""
    return {
        'patients': [
            {
                'subject_id': 1,
                'admissions': [
                    {
                        'hadm_id': 100,
                        'admittime': '2105-01-01 12:00:00',
                        'dischtime': '2105-01-05 10:00:00',
                        'measurements': [
                            {'type': 'temperature', 'value': 36.5, 'time': '2105-01-01 13:00:00'},
                            {'type': 'heart_rate', 'value': 72, 'time': '2105-01-01 13:00:00'}
                        ]
                    }
                ]
            }
        ]
    }


@pytest.fixture(autouse=True)
def cleanup_test_files():
    """Automatically cleanup test files after each test."""
    yield
    # Clean up any temporary files that might have been created during tests
    temp_files = ['/tmp/test_output.json', '/tmp/test_data.csv']
    for file_path in temp_files:
        if os.path.exists(file_path):
            os.remove(file_path)


@pytest.fixture
def capture_logs(caplog):
    """Fixture to capture and return log messages."""
    return caplog


# Custom markers for different test types
pytest_plugins = []