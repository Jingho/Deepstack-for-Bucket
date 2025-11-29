"""Main clustering class for different poker rounds."""

from typing import List, Optional
import os
import numpy as np
from sklearn.cluster import KMeans as SklearnKMeans

from ...config.settings import Config, get_config
from ...config.constants import PokerRound, ClusterInitializationMethod
from ...utils.logging import get_logger
from .distance import DistanceMatrixCalculator
from .kmeans import KMeans

logger = get_logger(__name__)


class Clusterer:
    """
    Main clustering class that loads data and performs clustering.
    
    Uses different clustering methods depending on the round:
    - River: Standard k-means (1D data) using sklearn
    - Turn/Flop: K-means with EMD distance (distribution data)
    """
    
    def __init__(
        self,
        street: PokerRound,
        k: int,
        config: Optional[Config] = None,
        initial_method: ClusterInitializationMethod = ClusterInitializationMethod.KMEANS_PLUS_PLUS,
        save_results: bool = True
    ):
        """
        Initialize clustering.
        
        Args:
            street: Round name
            k: Number of clusters
            config: Configuration instance
            initial_method: Initialization method
            save_results: Whether to save cluster centroids
        """
        self.config = config or get_config()
        self.street = street
        self.k = k
        self.initial_method = initial_method
        self.save_results = save_results
        
        # Load data and compute distance matrix
        self.points_list = self._load_data()
        self.distance_calculator = DistanceMatrixCalculator(self.config)
        self.matrix = self._compute_distance_matrix()
        self.save_path = self._get_save_path()
    
    def _load_data(self) -> List[List[float]]:
        """
        Load data set for clustering.
        
        Returns:
            List of data points
            
        Raises:
            FileNotFoundError: If data file doesn't exist
        """
        file_names = {
            PokerRound.RIVER: "river_data.csv",
            PokerRound.TURN: "turn_data.csv",
            PokerRound.FLOP: "flop_data.csv"
        }
        
        filename = file_names.get(self.street)
        if not filename:
            raise ValueError(f'Unknown street: {self.street}')
        
        file_path = self.config.data_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(
                f'Data file not found: {file_path}. '
                f'Please generate data first using DataGenerator.'
            )
        
        return self._read_csv(file_path)
    
    def _read_csv(self, file_path: os.PathLike) -> List[List[float]]:
        """
        Read CSV file with data points and error handling.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            List of data points
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is empty or contains invalid data
            IOError: If file cannot be read
        """
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        data_points = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    line = line.strip()
                    if not line:
                        continue
                    
                    values = [float(x) for x in line.split(',') if x]
                    if values:
                        data_points.append(values)
                except ValueError as e:
                    logger.warning(
                        f"Skipping invalid line {line_num} in {file_path}: {e}"
                    )
                    continue
        
        if not data_points:
            raise ValueError(
                f"Data file {file_path} is empty or contains no valid data points"
            )
        
        logger.info(f'Loaded {len(data_points)} data points from {file_path}')
        return data_points
    
    def _compute_distance_matrix(self) -> Optional[np.ndarray]:
        """
        Calculate distance matrix for the current round.
        
        Returns:
            Distance matrix, or None for river round
        """
        if self.street == PokerRound.RIVER:
            return None
        
        # Load previous round's centroids
        if self.street == PokerRound.TURN:
            river_centroids = self._load_centroids(PokerRound.RIVER)
            return self.distance_calculator.compute_euclidean_matrix(river_centroids)
        
        elif self.street == PokerRound.FLOP:
            river_centroids = self._load_centroids(PokerRound.RIVER)
            turn_centroids = self._load_centroids(PokerRound.TURN)
            
            river_matrix = self.distance_calculator.compute_euclidean_matrix(river_centroids)
            return self.distance_calculator.compute_emd_matrix(river_matrix, turn_centroids)
        
        return None
    
    def _load_centroids(self, street: PokerRound) -> List[List[float]]:
        """
        Load cluster centroids from file.
        
        Args:
            street: Round to load centroids for
            
        Returns:
            List of centroids
        """
        # Try to find cluster file (may have k in filename)
        possible_names = [
            f"{street.value}_cluster_k{self.k}.csv",
            f"{street.value}_cluster.csv"
        ]
        
        for filename in possible_names:
            file_path = self.config.data_dir / filename
            if file_path.exists():
                return self._read_csv(file_path)
        
        raise FileNotFoundError(
            f'Cluster file not found for {street.value}. '
            f'Tried: {possible_names}'
        )
    
    def _get_save_path(self) -> os.PathLike:
        """Get path for saving cluster centroids."""
        filename = f"{self.street.value}_cluster_k{self.k}.csv"
        return self.config.data_dir / filename
    
    def cluster(self) -> List[np.ndarray]:
        """
        Perform clustering.
        
        Returns:
            List of cluster centroids
        """
        logger.info(f'Starting clustering for {self.street.value} round with k={self.k}')
        
        if self.street == PokerRound.RIVER:
            centroids = self._cluster_river()
        else:
            centroids = self._cluster_with_emd()
        
        logger.info(f'Clustering completed. Found {len(centroids)} clusters')
        
        if self.save_results:
            self._save_centroids(centroids)
        
        return centroids
    
    def _cluster_river(self) -> List[np.ndarray]:
        """
        Cluster river round data using sklearn (1D data).
        
        Returns:
            List of centroids
        """
        logger.info('Using sklearn KMeans for river round')
        kmeans = SklearnKMeans(n_clusters=self.k, random_state=42, n_init=10)
        kmeans.fit(self.points_list)
        centroids = kmeans.cluster_centers_.flatten()
        centroids = np.sort(centroids)
        return [np.array([c]) for c in centroids]
    
    def _cluster_with_emd(self) -> List[np.ndarray]:
        """
        Cluster data using EMD distance metric.
        
        Returns:
            List of centroids
        """
        if self.matrix is None:
            raise ValueError("Distance matrix is required for EMD clustering")
        
        kmeans = KMeans(
            self.points_list,
            self.k,
            self.matrix,
            self.initial_method,
            self.config
        )
        return kmeans.solve()
    
    def _save_centroids(self, centroids: List[np.ndarray]) -> None:
        """
        Save cluster centroids to file.
        
        Args:
            centroids: List of centroids to save
        """
        with open(self.save_path, 'w', encoding='utf-8') as f:
            if self.street == PokerRound.RIVER:
                # River: single value per centroid
                for centroid in centroids:
                    f.write(str(centroid[0]) + '\n')
            else:
                # Turn/Flop: comma-separated values
                for centroid in centroids:
                    centroid_list = centroid.tolist()
                    line = ','.join(map(str, centroid_list))
                    f.write(line + '\n')
        
        logger.info(f'Centroids saved to {self.save_path}')
