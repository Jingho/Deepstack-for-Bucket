"""Data visualization utilities for clustering results."""

from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from ..config.settings import Config, get_config
from ..config.constants import PokerRound
from ..utils.logging import get_logger
from ..core.clustering.distance import DistanceMatrixCalculator

logger = get_logger(__name__)


class DataVisualizer:
    """
    Visualize clustering results and data distributions.
    
    Supports visualization of:
    - Raw data distributions
    - Clustered data distributions
    - Comparison between raw and clustered data
    """
    
    def __init__(
        self,
        street: PokerRound,
        config: Optional[Config] = None,
        save_path: Optional[Path] = None
    ):
        """
        Initialize visualizer.
        
        Args:
            street: Round name
            config: Configuration instance
            save_path: Path to save plots
        """
        self.config = config or get_config()
        self.street = street
        self.save_path = save_path or self.config.output_dir / "img"
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        self.distance_calculator = DistanceMatrixCalculator(self.config)
        self.data_points: List[List[float]] = []
        self.centroids: List[List[float]] = []
    
    def load_data(self) -> List[List[float]]:
        """
        Load data points from file.
        
        Returns:
            List of data points
        """
        file_names = {
            PokerRound.RIVER: "river_data.csv",
            PokerRound.TURN: "turn_data.csv",
            PokerRound.FLOP: "flop_data.csv"
        }
        
        filename = file_names.get(self.street)
        if not filename:
            raise ValueError(f"Unknown street: {self.street}")
        
        file_path = self.config.data_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        self.data_points = self._read_csv(file_path)
        logger.info(f"Loaded {len(self.data_points)} data points")
        return self.data_points
    
    def load_centroids(self, k: Optional[int] = None) -> List[List[float]]:
        """
        Load cluster centroids.
        
        Args:
            k: Cluster count (if None, uses config default)
            
        Returns:
            List of centroids
        """
        if k is None:
            k = self._get_cluster_count()
        
        possible_names = [
            f"{self.street.value}_cluster_k{k}.csv",
            f"{self.street.value}_cluster.csv"
        ]
        
        for filename in possible_names:
            file_path = self.config.data_dir / filename
            if file_path.exists():
                self.centroids = self._read_csv(file_path)
                logger.info(f"Loaded {len(self.centroids)} centroids")
                return self.centroids
        
        raise FileNotFoundError(f"Centroid file not found. Tried: {possible_names}")
    
    def _get_cluster_count(self) -> int:
        """Get cluster count for current street."""
        counts = {
            PokerRound.RIVER: self.config.river_cluster_count,
            PokerRound.TURN: self.config.turn_cluster_count,
            PokerRound.FLOP: self.config.flop_cluster_count
        }
        return counts.get(self.street, 5)
    
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
    
    def compute_ehs_values(self, points: List[List[float]]) -> List[float]:
        """
        Compute Expected Hand Strength (EHS) values from points.
        
        For river: direct equity value
        For turn: dot product with river centroids
        For flop: dot product with turn centroids (which are distributions over river)
        
        Args:
            points: Data points
            
        Returns:
            List of EHS values
        """
        if self.street == PokerRound.RIVER:
            return [p[0] for p in points]
        
        elif self.street == PokerRound.TURN:
            river_centroids = self._load_centroids_for_street(PokerRound.RIVER)
            river_values = [c[0] for c in river_centroids]
            return [np.dot(p, river_values) for p in points]
        
        elif self.street == PokerRound.FLOP:
            turn_centroids = self._load_centroids_for_street(PokerRound.TURN)
            river_centroids = self._load_centroids_for_street(PokerRound.RIVER)
            river_values = [c[0] for c in river_centroids]
            turn_values = [np.dot(tc, river_values) for tc in turn_centroids]
            return [np.dot(p, turn_values) for p in points]
        
        return []
    
    def _load_centroids_for_street(self, street: PokerRound) -> List[List[float]]:
        """Load centroids for a specific street."""
        possible_names = [
            f"{street.value}_cluster_k{self._get_cluster_count_for_street(street)}.csv",
            f"{street.value}_cluster.csv"
        ]
        
        for filename in possible_names:
            file_path = self.config.data_dir / filename
            if file_path.exists():
                return self._read_csv(file_path)
        
        raise FileNotFoundError(f"Centroid file not found for {street.value}")
    
    def _get_cluster_count_for_street(self, street: PokerRound) -> int:
        """Get cluster count for street."""
        counts = {
            PokerRound.RIVER: self.config.river_cluster_count,
            PokerRound.TURN: self.config.turn_cluster_count,
            PokerRound.FLOP: self.config.flop_cluster_count
        }
        return counts.get(street, 5)
    
    def plot_raw_data(self, save: bool = True) -> None:
        """
        Plot raw data distribution.
        
        Args:
            save: Whether to save the plot
        """
        if not self.data_points:
            self.load_data()
        
        ehs_values = self.compute_ehs_values(self.data_points)
        
        plt.figure(figsize=(10, 6))
        plt.hist(ehs_values, bins=100, edgecolor='black', alpha=0.7)
        plt.xlim(0, 1)
        plt.title(f'Raw Data Distribution - {self.street.value.upper()}')
        plt.xlabel('Expected Hand Strength (EHS)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        if save:
            output_file = self.save_path / f"{self.street.value}_raw_data_distribution.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {output_file}")
        
        plt.close()
    
    def plot_clustered_data(self, k: Optional[int] = None, save: bool = True) -> None:
        """
        Plot clustered data distribution.
        
        Args:
            k: Cluster count
            save: Whether to save the plot
        """
        if not self.data_points:
            self.load_data()
        if not self.centroids:
            self.load_centroids(k)
        
        # Group data by clusters
        grouped_data = self._group_data_by_clusters()
        ehs_group_data = [self.compute_ehs_values(group) for group in grouped_data]
        raw_ehs = self.compute_ehs_values(self.data_points)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # Plot clustered data
        ax1.hist(ehs_group_data, bins=50, stacked=True, alpha=0.7)
        ax1.set_xlim(0, 1)
        ax1.set_title(f'{self.street.value.upper()} Cluster Result (k={len(self.centroids)})')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        # Plot raw data
        ax2.hist(raw_ehs, bins=50, facecolor='0.5', alpha=0.7, edgecolor='black')
        ax2.set_xlim(0, 1)
        ax2.set_xlabel('Expected Hand Strength (EHS)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Raw Data Distribution')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            output_file = self.save_path / f"{self.street.value}_cluster_result_k{len(self.centroids):02d}.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {output_file}")
        
        plt.close()
    
    def _group_data_by_clusters(self) -> List[List[List[float]]]:
        """Group data points by their nearest cluster."""
        if not self.centroids:
            raise ValueError("Centroids not loaded")
        
        # Compute distance matrix
        if self.street == PokerRound.RIVER:
            matrix = None
        elif self.street == PokerRound.TURN:
            river_centroids = self._load_centroids_for_street(PokerRound.RIVER)
            matrix = self.distance_calculator.compute_euclidean_matrix(river_centroids)
        else:  # FLOP
            river_centroids = self._load_centroids_for_street(PokerRound.RIVER)
            turn_centroids = self._load_centroids_for_street(PokerRound.TURN)
            river_matrix = self.distance_calculator.compute_euclidean_matrix(river_centroids)
            matrix = self.distance_calculator.compute_emd_matrix(river_matrix, turn_centroids)
        
        groups = [[] for _ in range(len(self.centroids))]
        
        from pyemd import emd
        for point in self.data_points:
            distances = []
            for centroid in self.centroids:
                if matrix is None:
                    # River: use Euclidean
                    dist = abs(point[0] - centroid[0])
                else:
                    # Turn/Flop: use EMD
                    dist = emd(
                        np.array(point, dtype=np.float64),
                        np.array(centroid, dtype=np.float64),
                        matrix
                    )
                distances.append(dist)
            
            nearest_cluster = int(np.argmin(distances))
            groups[nearest_cluster].append(point)
        
        return groups

