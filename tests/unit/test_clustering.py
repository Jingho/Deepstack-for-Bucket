"""Unit tests for clustering module."""

import pytest
import numpy as np
from src.core.clustering.distance import DistanceMatrixCalculator
from src.core.clustering.kmeans import KMeans
from src.core.clustering.kmeans_plus_plus import KMeansPlusPlus
from src.config.settings import Config
from src.config.constants import ClusterInitializationMethod


class TestDistanceMatrixCalculator:
    """Test distance matrix computation."""
    
    def test_compute_euclidean_matrix(self):
        """Test Euclidean distance matrix computation."""
        calculator = DistanceMatrixCalculator()
        centroids = [[0.1], [0.5], [0.9]]
        matrix = calculator.compute_euclidean_matrix(centroids)
        
        assert matrix.shape == (3, 3)
        assert matrix[0, 0] == 0.0  # Diagonal should be zero
        assert matrix[0, 1] == pytest.approx(0.4, abs=1e-6)
        assert matrix[1, 0] == matrix[0, 1]  # Symmetric
    
    def test_compute_emd_matrix(self):
        """Test EMD distance matrix computation."""
        calculator = DistanceMatrixCalculator()
        base_matrix = np.array([[0, 0.1], [0.1, 0]])
        centroids = [[0.8, 0.2], [0.2, 0.8], [0.5, 0.5]]
        matrix = calculator.compute_emd_matrix(base_matrix, centroids)
        
        assert matrix.shape == (3, 3)
        assert matrix[0, 0] == 0.0


class TestKMeans:
    """Test K-means clustering."""
    
    def test_initialization(self):
        """Test K-means initialization."""
        points = [[0.1], [0.2], [0.3], [0.4], [0.5]]
        matrix = np.array([[0, 0.1], [0.1, 0]])
        kmeans = KMeans(
            points,
            k=2,
            emd_matrix=matrix,
            initial_method=ClusterInitializationMethod.RANDOM
        )
        assert kmeans.k == 2
        assert len(kmeans.points_list) == 5

