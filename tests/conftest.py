"""Pytest configuration and fixtures."""

import pytest
import tempfile
import shutil
from pathlib import Path

from src.config.settings import Config


@pytest.fixture
def temp_data_dir():
    """Create temporary data directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def test_config(temp_data_dir):
    """Create test configuration."""
    config = Config.default()
    config.data_dir = temp_data_dir
    config.log_dir = temp_data_dir / "logs"
    config.cache_dir = temp_data_dir / "cache"
    config.output_dir = temp_data_dir / "output"
    return config

