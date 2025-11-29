"""K-means++ initialization algorithm."""

from typing import List
import random
import math
import numpy as np
from pyemd import emd

from ...config.settings import Config, get_config
from ...utils.logging import get_logger

logger = get_logger(__name__)


class KMeansPlusPlus:
    """
    K-means++ initialization for better cluster seeding.
    
    K-means++ improves k-means by choosing initial centroids that are
    far apart, leading to better convergence and clustering results.
    """
    
    def __init__(
        self,
        points_list: List[List[float]],
        k: int,
        emd_matrix: np.ndarray,
        config: Config = None
    ):
        """
        Initialize k-means++ algorithm.
        
        Args:
            points_list: List of data points to cluster
            k: Number of clusters
            emd_matrix: Distance matrix for EMD calculations
            config: Configuration instance
        """
        self.config = config or get_config()
        self.centroid_count = 0
        self.point_count = len(points_list)
        self.cluster_count = k
        self.matrix = emd_matrix
        self.points_list = list(points_list)
        
        logger.info('K-means++ initialization starting')
        self.initialize_first_centroid()
        self.init_centroids_list = self.initialize_other_centroids()
        logger.info(f'K-means++ initialization completed: {len(self.init_centroids_list)} centroids')
    
    def initialize_first_centroid(self) -> None:
        """Pick a random point as the first centroid."""
        self.centroid_list = []
        index = random.randint(0, len(self.points_list) - 1)
        self.centroid_list.append(self.remove_point(index))
        self.centroid_count = 1
    
    def remove_point(self, index: int) -> List[float]:
        """
        Remove point from list and return it.
        
        Args:
            index: Index of point to remove
            
        Returns:
            The removed point
        """
        new_centroid = self.points_list[index]
        del self.points_list[index]
        return new_centroid
    
    def initialize_other_centroids(self) -> List[List[float]]:
        """
        Find the other k-1 centroids using weighted probability selection.
        
        Returns:
            List of k initial centroids
        """
        while not self.is_finished():
            logger.debug(f'Initializing centroid {self.centroid_count + 1}/{self.cluster_count}')
            distances = self.find_smallest_distances()
            chosen_index = self.choose_weighted(distances)
            self.centroid_list.append(self.remove_point(chosen_index))
            self.centroid_count += 1
        return self.centroid_list
    
    def find_smallest_distances(self) -> List[float]:
        """
        Calculate distance from each point to its nearest cluster center.
        
        Returns:
            List of minimum distances for each point
        """
        distance_list = []
        for point in self.points_list:
            distance_list.append(self.find_nearest_centroid(point))
        return distance_list
    
    def find_nearest_centroid(self, point: List[float]) -> float:
        """
        Find centroid nearest to the given point and return its distance.
        
        Args:
            point: Data point
            
        Returns:
            Distance to nearest centroid
        """
        min_distance = math.inf
        for centroid in self.centroid_list:
            distance = self.get_distance(centroid, point)
            if distance < min_distance:
                min_distance = distance
        return min_distance
    
    def choose_weighted(self, distance_list: List[float]) -> int:
        """
        Choose an index based on weighted probability.
        
        Probability is proportional to distance squared.
        
        Args:
            distance_list: List of distances
            
        Returns:
            Chosen index
        """
        distance_squared = [x ** 2 for x in distance_list]
        weighted_list = self.weight_values(distance_squared)
        indices = list(range(len(distance_list)))
        return int(np.random.choice(indices, p=weighted_list))
    
    def weight_values(self, values: List[float]) -> List[float]:
        """
        Normalize values to sum to 1 (create probability distribution).
        
        Args:
            values: List of values to normalize
            
        Returns:
            Normalized values (probabilities)
        """
        total = sum(values)
        if total == 0:
            return [1.0 / len(values)] * len(values)
        return [x / total for x in values]
    
    def get_distance(self, point1: List[float], point2: List[float]) -> float:
        """
        Compute EMD distance between two points.
        
        Args:
            point1: First point (distribution)
            point2: Second point (distribution)
            
        Returns:
            EMD distance
        """
        point_1 = np.array(point1, dtype=np.float64)
        point_2 = np.array(point2, dtype=np.float64)
        return emd(point_1, point_2, self.matrix)
    
    def is_finished(self) -> bool:
        """
        Check if k centroids have been created.
        
        Returns:
            True if finished, False otherwise
        """
        return self.centroid_count == self.cluster_count
    
    def final_centroids(self) -> List[List[float]]:
        """
        Return final centroid values.
        
        Returns:
            List of centroids
        """
        return self.centroid_list

