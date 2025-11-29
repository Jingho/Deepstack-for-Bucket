"""Tests for EMD distance calculation with error statistics."""

import pytest
import numpy as np
from pyemd import emd
from scipy.stats import wasserstein_distance

from src.core.clustering.distance import DistanceMatrixCalculator
from src.config.settings import Config


class TestEMDDistanceComparison:
    """
    Compare two EMD distance calculation methods and compute error statistics.
    
    Method 1: pyemd library (exact EMD)
    Method 2: scipy wasserstein_distance (1D EMD approximation)
    """
    
    def test_1d_emd_distribution_comparison(self):
        """
        Compare EMD calculations for 1D distributions.
        
        For 1D distributions, EMD is equivalent to Wasserstein distance,
        which can be computed more efficiently.
        """
        calculator = DistanceMatrixCalculator()
        
        # Create test distributions (1D)
        dist1 = np.array([0.1, 0.3, 0.4, 0.2])
        dist2 = np.array([0.2, 0.2, 0.3, 0.3])
        dist3 = np.array([0.4, 0.1, 0.2, 0.3])
        
        # Normalize distributions
        dist1 = dist1 / dist1.sum()
        dist2 = dist2 / dist2.sum()
        dist3 = dist3 / dist3.sum()
        
        # Create base distance matrix (1D positions)
        n = len(dist1)
        base_matrix_1d = np.zeros((n, n))
        positions = np.arange(n)
        for i in range(n):
            for j in range(n):
                base_matrix_1d[i, j] = abs(positions[i] - positions[j])
        
        # Method 1: pyemd (exact EMD)
        emd_12_pyemd = abs(emd(dist1, dist2, base_matrix_1d))
        emd_13_pyemd = abs(emd(dist1, dist3, base_matrix_1d))
        emd_23_pyemd = abs(emd(dist2, dist3, base_matrix_1d))
        
        # Method 2: scipy wasserstein_distance (1D EMD)
        emd_12_wasserstein = wasserstein_distance(positions, positions, dist1, dist2)
        emd_13_wasserstein = wasserstein_distance(positions, positions, dist1, dist3)
        emd_23_wasserstein = wasserstein_distance(positions, positions, dist2, dist3)
        
        # Compute errors
        errors = [
            abs(emd_12_pyemd - emd_12_wasserstein),
            abs(emd_13_pyemd - emd_13_wasserstein),
            abs(emd_23_pyemd - emd_23_wasserstein)
        ]
        
        mean_error = np.mean(errors)
        max_error = np.max(errors)
        std_error = np.std(errors)
        
        # Assertions
        assert mean_error < 1e-10, f"Mean error too large: {mean_error}"
        assert max_error < 1e-10, f"Max error too large: {max_error}"
        
        print(f"\n1D EMD Comparison Statistics:")
        print(f"  Mean Error: {mean_error:.2e}")
        print(f"  Max Error: {max_error:.2e}")
        print(f"  Std Error: {std_error:.2e}")
    
    def test_euclidean_vs_emd_1d(self):
        """
        Compare Euclidean distance vs EMD for 1D centroids (river clusters).
        
        For 1D equity values, Euclidean distance should equal EMD.
        """
        calculator = DistanceMatrixCalculator()
        
        # Create 1D centroids (equity values)
        centroids = [[0.1], [0.5], [0.9], [0.3], [0.7]]
        
        # Method 1: Euclidean distance matrix
        euclidean_matrix = calculator.compute_euclidean_matrix(centroids)
        
        # Method 2: EMD distance matrix (for 1D, should be equivalent)
        # Create base matrix for EMD (1D positions)
        n = len(centroids)
        base_matrix = np.zeros((n, n))
        values = np.array([c[0] for c in centroids])
        for i in range(n):
            for j in range(n):
                base_matrix[i, j] = abs(values[i] - values[j])
        
        # Create unit distributions for each centroid
        unit_dists = [np.zeros(n) for _ in range(n)]
        for i in range(n):
            unit_dists[i][i] = 1.0
        
        # Compute EMD matrix
        emd_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                emd_dist = abs(emd(unit_dists[i], unit_dists[j], base_matrix))
                emd_matrix[i, j] = emd_dist
                emd_matrix[j, i] = emd_dist
        
        # Compare matrices
        errors = []
        for i in range(n):
            for j in range(n):
                if i != j:
                    error = abs(euclidean_matrix[i, j] - emd_matrix[i, j])
                    errors.append(error)
        
        mean_error = np.mean(errors)
        max_error = np.max(errors)
        
        # For 1D, they should be very close
        assert mean_error < 1e-10, f"Mean error too large: {mean_error}"
        assert max_error < 1e-10, f"Max error too large: {max_error}"
        
        print(f"\nEuclidean vs EMD (1D) Statistics:")
        print(f"  Mean Error: {mean_error:.2e}")
        print(f"  Max Error: {max_error:.2e}")
    
    def test_multidimensional_emd_comparison(self):
        """
        Compare EMD calculations for multi-dimensional distributions.
        
        Tests turn/flop cluster distributions over river clusters.
        """
        calculator = DistanceMatrixCalculator()
        
        # Create river centroids (1D equity values)
        river_centroids = [[0.1], [0.3], [0.5], [0.7], [0.9]]
        river_matrix = calculator.compute_euclidean_matrix(river_centroids)
        
        # Create turn cluster distributions (over river clusters)
        turn_dist1 = np.array([0.2, 0.3, 0.3, 0.15, 0.05])
        turn_dist2 = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
        turn_dist3 = np.array([0.05, 0.15, 0.3, 0.3, 0.2])
        
        # Normalize
        turn_dist1 = turn_dist1 / turn_dist1.sum()
        turn_dist2 = turn_dist2 / turn_dist2.sum()
        turn_dist3 = turn_dist3 / turn_dist3.sum()
        
        turn_centroids = [
            turn_dist1.tolist(),
            turn_dist2.tolist(),
            turn_dist3.tolist()
        ]
        
        # Method 1: Using DistanceMatrixCalculator
        emd_matrix_method1 = calculator.compute_emd_matrix(river_matrix, turn_centroids)
        
        # Method 2: Direct pyemd calculation
        emd_matrix_method2 = np.zeros((3, 3))
        for i in range(3):
            for j in range(i + 1, 3):
                dist = abs(emd(
                    np.array(turn_centroids[i], dtype=np.float64),
                    np.array(turn_centroids[j], dtype=np.float64),
                    river_matrix
                ))
                emd_matrix_method2[i, j] = dist
                emd_matrix_method2[j, i] = dist
        
        # Compare matrices
        errors = []
        for i in range(3):
            for j in range(3):
                if i != j:
                    error = abs(emd_matrix_method1[i, j] - emd_matrix_method2[i, j])
                    errors.append(error)
        
        mean_error = np.mean(errors)
        max_error = np.max(errors)
        std_error = np.std(errors)
        
        # Should be identical (same method)
        assert mean_error < 1e-12, f"Mean error too large: {mean_error}"
        assert max_error < 1e-12, f"Max error too large: {max_error}"
        
        print(f"\nMulti-dimensional EMD Comparison Statistics:")
        print(f"  Mean Error: {mean_error:.2e}")
        print(f"  Max Error: {max_error:.2e}")
        print(f"  Std Error: {std_error:.2e}")
    
    def test_emd_error_statistics_comprehensive(self):
        """
        Comprehensive error statistics for EMD distance calculations.
        
        Tests multiple scenarios and collects detailed statistics.
        """
        calculator = DistanceMatrixCalculator()
        
        # Test scenarios
        scenarios = []
        
        # Scenario 1: Similar distributions
        dist1 = np.array([0.25, 0.25, 0.25, 0.25])
        dist2 = np.array([0.24, 0.26, 0.24, 0.26])
        scenarios.append(("Similar", dist1, dist2))
        
        # Scenario 2: Different distributions
        dist3 = np.array([0.5, 0.3, 0.15, 0.05])
        dist4 = np.array([0.05, 0.15, 0.3, 0.5])
        scenarios.append(("Different", dist3, dist4))
        
        # Scenario 3: Sparse distributions
        dist5 = np.array([0.9, 0.05, 0.03, 0.02])
        dist6 = np.array([0.02, 0.03, 0.05, 0.9])
        scenarios.append(("Sparse", dist5, dist6))
        
        # Create base matrix
        n = 4
        base_matrix = np.zeros((n, n))
        positions = np.arange(n)
        for i in range(n):
            for j in range(n):
                base_matrix[i, j] = abs(positions[i] - positions[j])
        
        all_errors = []
        results = []
        
        for name, d1, d2 in scenarios:
            # Normalize
            d1 = d1 / d1.sum()
            d2 = d2 / d2.sum()
            
            # Method 1: pyemd
            emd_pyemd = abs(emd(d1, d2, base_matrix))
            
            # Method 2: wasserstein_distance
            emd_wasserstein = wasserstein_distance(positions, positions, d1, d2)
            
            error = abs(emd_pyemd - emd_wasserstein)
            all_errors.append(error)
            
            results.append({
                'scenario': name,
                'emd_pyemd': emd_pyemd,
                'emd_wasserstein': emd_wasserstein,
                'error': error,
                'relative_error': error / max(emd_pyemd, 1e-10)
            })
        
        # Compute statistics
        mean_error = np.mean(all_errors)
        max_error = np.max(all_errors)
        std_error = np.std(all_errors)
        median_error = np.median(all_errors)
        
        # Print comprehensive statistics
        print(f"\n{'='*60}")
        print(f"Comprehensive EMD Statistics Analysis")
        print(f"{'='*60}")
        print(f"\nOverall Statistics:")
        print(f"  Mean Error:      {mean_error:.2e}")
        print(f"  Max Error:       {max_error:.2e}")
        print(f"  Median Error:    {median_error:.2e}")
        print(f"  Std Deviation:   {std_error:.2e}")
        
        print(f"\nPer-Scenario Results:")
        for r in results:
            print(f"  {r['scenario']:12s}: "
                  f"EMD={r['emd_pyemd']:.6f}, "
                  f"Error={r['error']:.2e} "
                  f"({r['relative_error']*100:.2f}%)")
        
        # Assertions
        assert mean_error < 1e-10, f"Mean error too large: {mean_error}"
        assert max_error < 1e-10, f"Max error too large: {max_error}"
        
        return {
            'mean_error': mean_error,
            'max_error': max_error,
            'std_error': std_error,
            'median_error': median_error,
            'scenarios': results
        }
    
    def test_emd_matrix_computation_consistency(self):
        """
        Test consistency of EMD matrix computation across multiple calls.
        
        Ensures that the same input produces the same output.
        """
        calculator = DistanceMatrixCalculator()
        
        # Create test data
        river_centroids = [[0.1], [0.3], [0.5], [0.7], [0.9]]
        river_matrix = calculator.compute_euclidean_matrix(river_centroids)
        
        turn_centroids = [
            [0.2, 0.3, 0.3, 0.15, 0.05],
            [0.1, 0.2, 0.4, 0.2, 0.1],
            [0.05, 0.15, 0.3, 0.3, 0.2],
            [0.3, 0.25, 0.2, 0.15, 0.1]
        ]
        
        # Compute multiple times
        matrices = []
        for _ in range(5):
            matrix = calculator.compute_emd_matrix(river_matrix, turn_centroids)
            matrices.append(matrix)
        
        # Compare all matrices
        errors = []
        for i in range(len(matrices)):
            for j in range(i + 1, len(matrices)):
                error = np.abs(matrices[i] - matrices[j]).max()
                errors.append(error)
        
        max_error = np.max(errors)
        mean_error = np.mean(errors)
        
        print(f"\nEMD Matrix Consistency Test:")
        print(f"  Mean Error: {mean_error:.2e}")
        print(f"  Max Error: {max_error:.2e}")
        
        # Should be identical
        assert max_error < 1e-12, f"Matrices not consistent: {max_error}"
