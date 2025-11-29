"""K-means clustering algorithm with EMD distance metric."""

from typing import List, Optional
import numpy as np
import random
from pyemd import emd

from ...config.settings import Config, get_config
from ...config.constants import ClusterInitializationMethod
from ...utils.logging import get_logger
from .kmeans_plus_plus import KMeansPlusPlus

logger = get_logger(__name__)


class KMeans:
    """
    K-means clustering algorithm with EMD distance metric.
    
    This implementation uses EMD (Earth Mover's Distance) instead of
    Euclidean distance, which is more appropriate for distribution data.
    """
    
    def __init__(
        self,
        points_list: List[List[float]],
        k: int,
        emd_matrix: np.ndarray,
        initial_method: ClusterInitializationMethod = ClusterInitializationMethod.KMEANS_PLUS_PLUS,
        config: Optional[Config] = None
    ):
        """
        Initialize k-means algorithm.
        
        Args:
            points_list: List of data points to cluster
            k: Number of clusters
            emd_matrix: Distance matrix for EMD calculations
            initial_method: Initialization method
            config: Configuration instance
        """
        self.config = config or get_config()
        self.point_count = len(points_list)
        self.k = k
        self.matrix = emd_matrix
        self.points_list = list(points_list)
        self.initial_method = initial_method
    
    def points_best_cluster(
        self,
        centroids: List[List[float]],
        data_point: List[float]
    ) -> int:
        """
        Find the centroid index that the data point is closest to.
        
        Args:
            centroids: List of centroids
            data_point: Data point to assign
            
        Returns:
            Index of closest centroid
        """
        closest_centroid = None
        least_distance = None
        
        for i, centroid in enumerate(centroids):
            distance = emd(
                np.array(data_point, dtype=np.float64),
                np.array(centroid, dtype=np.float64),
                self.matrix
            )
            if least_distance is None or distance < least_distance:
                closest_centroid = i
                least_distance = distance
        
        return closest_centroid
    
    def new_centroid(self, cluster: List[List[float]]) -> np.ndarray:
        """
        Find the new centroid location given a cluster of data points.
        
        For EMD space, we use the arithmetic mean as an approximation.
        
        Args:
            cluster: List of data points in the cluster
            
        Returns:
            New centroid as numpy array
        """
        if len(cluster) == 0:
            return np.zeros(len(self.points_list[0]))
        return np.mean(cluster, axis=0)
    
    def configure_clusters(
        self,
        centroids: List[List[float]],
        data_points: List[List[float]]
    ) -> List[List[List[float]]]:
        """
        Create a new configuration of clusters for the given data points.
        
        Args:
            centroids: List of centroids
            data_points: Set of data points to cluster
            
        Returns:
            List of clusters, where each cluster is a list of data points
        """
        clusters = [[] for _ in range(len(centroids))]
        
        # Assign each point to its closest centroid
        for i in range(self.point_count):
            ideal_cluster = self.points_best_cluster(centroids, data_points[i])
            clusters[ideal_cluster].append(data_points[i])
        
        # Handle empty clusters
        empty_clusters = [i for i, cluster in enumerate(clusters) if len(cluster) == 0]
        
        if empty_clusters:
            cluster_sizes = [len(cluster) for cluster in clusters]
            largest_cluster_idx = int(np.argmax(cluster_sizes))
            largest_cluster = clusters[largest_cluster_idx]
            
            if len(largest_cluster) > len(empty_clusters):
                for empty_idx in empty_clusters:
                    if largest_cluster:
                        clusters[empty_idx].append(largest_cluster.pop())
            else:
                for empty_idx in empty_clusters:
                    if data_points:
                        clusters[empty_idx].append(data_points[0])
        
        return clusters
    
    def get_cluster_rss(
        self,
        cluster: List[List[float]],
        centroid: List[float]
    ) -> float:
        """
        Calculate the cluster's Residual Sum of Squares (RSS).
        
        Args:
            cluster: List of data points in the cluster
            centroid: Centroid of the cluster
            
        Returns:
            RSS value
        """
        sum_rss = 0.0
        centroid_array = np.array(centroid, dtype=np.float64)
        
        for point in cluster:
            point_array = np.array(point, dtype=np.float64)
            distance = emd(point_array, centroid_array, self.matrix)
            sum_rss += distance ** 2
        
        return sum_rss
    
    def solve(self) -> List[np.ndarray]:
        """
        Iteratively cluster data points into k clusters.
        
        Returns:
            List of final centroids
        """
        data_points = self.points_list
        
        # Initialize centroids
        if self.initial_method == ClusterInitializationMethod.KMEANS_PLUS_PLUS:
            kpp = KMeansPlusPlus(self.points_list, self.k, self.matrix, self.config)
            centroids = kpp.init_centroids_list
        elif self.initial_method == ClusterInitializationMethod.RANDOM:
            centroids = random.sample(data_points, self.k)
        else:
            raise ValueError(f'Unknown initial method: {self.initial_method}')
        
        logger.info('K-means clustering starting')
        clusters = self.configure_clusters(centroids, data_points)
        
        # Iterate until convergence
        not_done = True
        last_rss = 0.0
        iteration = 0
        
        # Progress bar for iterations
        from tqdm import tqdm
        pbar = tqdm(
            total=self.config.max_iterations,
            desc="K-means iterations",
            disable=not self.config.show_progress,
            unit="iter"
        )
        
        try:
            while not_done and iteration < self.config.max_iterations:
                iteration += 1
                pbar.update(1)
                
                # Calculate RSS
                clusters_rss = []
                for i in range(len(clusters)):
                    rss = self.get_cluster_rss(clusters[i], centroids[i])
                    clusters_rss.append(rss / self.point_count)
                
                current_rss = sum(clusters_rss)
                logger.debug(f'Iteration {iteration}, RSS: {current_rss:.6f}')
                
                # Check for convergence
                if abs(current_rss - last_rss) < self.config.convergence_threshold:
                    not_done = False
                else:
                    last_rss = current_rss
                
                # Update centroids
                for i in range(len(centroids)):
                    centroids[i] = self.new_centroid(clusters[i])
                
                # Reconfigure clusters
                clusters = self.configure_clusters(centroids, data_points)
            
            if iteration >= self.config.max_iterations:
                logger.warning(f'Reached maximum iterations ({self.config.max_iterations})')
            
            logger.info(f'K-means clustering completed after {iteration} iterations')
            return [np.array(c, dtype=np.float64) for c in centroids]
        finally:
            pbar.close()

