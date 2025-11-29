"""Distance matrix computation for EMD calculations."""

from typing import List, Optional
import numpy as np
from pyemd import emd

from ...config.settings import Config, get_config
from ...utils.logging import get_logger
from ...utils.cache import DiskCache

logger = get_logger(__name__)


class DistanceMatrixCalculator:
    """
    Compute distance matrices for Earth Mover's Distance calculations.
    
    Provides methods to compute:
    - Euclidean distance matrix for river clusters (1D)
    - EMD distance matrix for turn/flop clusters (multi-dimensional distributions)
    """
    
    def __init__(self, config: Optional[Config] = None, cache: Optional[DiskCache] = None):
        """
        Initialize distance matrix calculator.
        
        Args:
            config: Configuration instance
            cache: Cache instance for storing computed matrices
        """
        self.config = config or get_config()
        self.cache = cache or DiskCache(config=self.config)
        self._matrix_cache: dict = {}
    
    def compute_euclidean_matrix(self, centroids: List[List[float]]) -> np.ndarray:
        """
        Compute Euclidean distance matrix for river cluster centroids.
        
        For 1D centroids (equity values), this computes the absolute difference
        between equity values. This is equivalent to EMD for 1D distributions.
        
        Args:
            centroids: List of river cluster centroids, each is [equity_value]
            
        Returns:
            NxN numpy array containing Euclidean distances between centroids
        """
        n = len(centroids)
        matrix = np.zeros((n, n), dtype=np.float64)
        
        # Vectorized computation for better performance
        centroids_array = np.array([c[0] for c in centroids])
        
        for i in range(n):
            for j in range(i + 1, n):
                distance = abs(centroids_array[i] - centroids_array[j])
                matrix[i, j] = distance
                matrix[j, i] = distance  # Symmetric matrix
        
        return matrix
    
    def compute_emd_matrix(
        self,
        base_matrix: np.ndarray,
        centroids: List[List[float]]
    ) -> np.ndarray:
        """
        Compute EMD distance matrix for turn/flop cluster centroids.
        
        Each centroid is a distribution over the previous round's clusters.
        The distance between two centroids is computed using EMD with the
        provided base distance matrix.
        
        Args:
            base_matrix: Base distance matrix (e.g., river cluster distances)
            centroids: List of cluster centroids (distributions)
            
        Returns:
            NxN numpy array containing EMD distances between centroids
        """
        n = len(centroids)
        matrix = np.zeros((n, n), dtype=np.float64)
        
        # Convert centroids to numpy arrays for efficiency
        centroids_arrays = [np.array(c, dtype=np.float64) for c in centroids]
        base_matrix = np.array(base_matrix, dtype=np.float64)
        
        # Compute EMD distances
        for i in range(n):
            for j in range(i + 1, n):
                distance = abs(emd(centroids_arrays[i], centroids_arrays[j], base_matrix))
                matrix[i, j] = distance
                matrix[j, i] = distance  # Symmetric matrix
        
        return matrix
    
    def clear_cache(self) -> None:
        """Clear the matrix cache."""
        self._matrix_cache.clear()

