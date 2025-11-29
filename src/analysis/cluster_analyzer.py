"""Cluster analysis utilities."""

from typing import List, Tuple, Optional
from pathlib import Path
import numpy as np
from pyemd import emd

from ..config.settings import Config, get_config
from ..config.constants import PokerRound
from ..utils.logging import get_logger
from ..core.clustering.distance import DistanceMatrixCalculator

logger = get_logger(__name__)


class ClusterAnalyzer:
    """
    Analyze cluster centroids and find closest pairs.
    
    Useful for understanding cluster relationships and identifying
    potential merge candidates.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize cluster analyzer.
        
        Args:
            config: Configuration instance
        """
        self.config = config or get_config()
        self.distance_calculator = DistanceMatrixCalculator(self.config)
    
    def load_centroids(self, street: PokerRound, k: Optional[int] = None) -> List[List[float]]:
        """
        Load cluster centroids from file.
        
        Args:
            street: Round name
            k: Cluster count (if None, uses config default)
            
        Returns:
            List of centroids
        """
        if k is None:
            k = self._get_cluster_count(street)
        
        possible_names = [
            f"{street.value}_cluster_k{k}.csv",
            f"{street.value}_cluster.csv"
        ]
        
        for filename in possible_names:
            file_path = self.config.data_dir / filename
            if file_path.exists():
                return self._read_csv(file_path)
        
        raise FileNotFoundError(
            f"Cluster file not found for {street.value}. Tried: {possible_names}"
        )
    
    def _get_cluster_count(self, street: PokerRound) -> int:
        """Get cluster count for street."""
        counts = {
            PokerRound.RIVER: self.config.river_cluster_count,
            PokerRound.TURN: self.config.turn_cluster_count,
            PokerRound.FLOP: self.config.flop_cluster_count
        }
        return counts.get(street, 5)
    
    def _read_csv(self, file_path: Path) -> List[List[float]]:
        """
        Read CSV file with error handling.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            List of data points as lists of floats
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is empty or contains invalid data
            IOError: If file cannot be read
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    line = line.strip()
                    if not line:
                        continue
                    values = [float(x) for x in line.split(',') if x]
                    if values:
                        data.append(values)
                except ValueError as e:
                    logger.warning(
                        f"Skipping invalid line {line_num} in {file_path}: {e}"
                    )
                    continue
        
        if not data:
            raise ValueError(f"File {file_path} contains no valid data")
        
        return data
    
    def compute_distance_matrix(
        self,
        centroids: List[List[float]],
        base_matrix: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute distance matrix between centroids.
        
        Args:
            centroids: List of centroids
            base_matrix: Base distance matrix for EMD (if None, uses Euclidean)
            
        Returns:
            Distance matrix
        """
        n = len(centroids)
        matrix = np.zeros((n, n), dtype=np.float64)
        
        if base_matrix is None:
            # Use Euclidean distance
            for i in range(n):
                for j in range(i + 1, n):
                    if len(centroids[i]) == 1 and len(centroids[j]) == 1:
                        distance = abs(centroids[i][0] - centroids[j][0])
                    else:
                        # For multi-dimensional, use L2 norm
                        distance = np.linalg.norm(
                            np.array(centroids[i]) - np.array(centroids[j])
                        )
                    matrix[i, j] = distance
                    matrix[j, i] = distance
        else:
            # Use EMD distance
            for i in range(n):
                for j in range(i + 1, n):
                    distance = abs(emd(
                        np.array(centroids[i], dtype=np.float64),
                        np.array(centroids[j], dtype=np.float64),
                        base_matrix
                    ))
                    matrix[i, j] = distance
                    matrix[j, i] = distance
        
        return matrix
    
    def find_closest_pairs(
        self,
        centroids: List[List[float]],
        base_matrix: Optional[np.ndarray] = None,
        num_pairs: int = 5
    ) -> List[Tuple[int, int, float]]:
        """
        Find the N closest pairs of centroids.
        
        Args:
            centroids: List of centroids
            base_matrix: Base distance matrix for EMD
            num_pairs: Number of closest pairs to return
            
        Returns:
            List of (index1, index2, distance) tuples, sorted by distance
        """
        distance_matrix = self.compute_distance_matrix(centroids, base_matrix)
        
        # Get upper triangle (excluding diagonal)
        n = len(centroids)
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((i, j, distance_matrix[i, j]))
        
        # Sort by distance
        pairs.sort(key=lambda x: x[2])
        
        return pairs[:num_pairs]
    
    def analyze_clusters(
        self,
        street: PokerRound,
        k: Optional[int] = None
    ) -> List[Tuple[int, int, float]]:
        """
        Analyze clusters for a given street and find closest pairs.
        
        Args:
            street: Round name
            k: Cluster count
            
        Returns:
            List of closest pairs
        """
        centroids = self.load_centroids(street, k)
        
        # Compute base matrix if needed
        base_matrix = None
        if street == PokerRound.TURN:
            river_centroids = self.load_centroids(PokerRound.RIVER)
            base_matrix = self.distance_calculator.compute_euclidean_matrix(river_centroids)
        elif street == PokerRound.FLOP:
            river_centroids = self.load_centroids(PokerRound.RIVER)
            turn_centroids = self.load_centroids(PokerRound.TURN)
            river_matrix = self.distance_calculator.compute_euclidean_matrix(river_centroids)
            base_matrix = self.distance_calculator.compute_emd_matrix(river_matrix, turn_centroids)
        
        closest_pairs = self.find_closest_pairs(centroids, base_matrix)
        
        logger.info(f"Found {len(closest_pairs)} closest pairs for {street.value}")
        return closest_pairs
    
    def save_analysis(
        self,
        closest_pairs: List[Tuple[int, int, float]],
        output_path: str,
        k: int
    ) -> None:
        """
        Save analysis results to file.
        
        Args:
            closest_pairs: List of closest pairs
            output_path: Output file path
            k: Cluster count
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f'-----------k{k:02d}------------\n')
            for idx, (i, j, dist) in enumerate(closest_pairs, 1):
                line = f"{idx}    {dist:.10f}    ({i}, {j})\n"
                f.write(line)
                logger.info(f"Pair {idx}: clusters {i}-{j}, distance {dist:.10f}")

