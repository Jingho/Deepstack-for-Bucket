"""Data validation utilities."""

from typing import List, Optional
import numpy as np

from ..utils.logging import get_logger

logger = get_logger(__name__)


class ValidationError(Exception):
    """Raised when validation fails."""
    pass


def validate_distribution(distribution: List[float], tolerance: float = 0.01) -> bool:
    """
    Validate that a list forms a valid probability distribution.
    
    Args:
        distribution: List of probabilities
        tolerance: Tolerance for sum check
        
    Returns:
        True if valid distribution
        
    Raises:
        ValidationError: If validation fails
    """
    if not distribution:
        raise ValidationError("Distribution cannot be empty")
    
    if any(x < 0 for x in distribution):
        raise ValidationError("Distribution values must be non-negative")
    
    total = sum(distribution)
    if abs(total - 1.0) > tolerance:
        raise ValidationError(f"Distribution must sum to 1.0, got {total}")
    
    return True


def validate_cluster_centroid(centroid: List[float], expected_length: Optional[int] = None) -> bool:
    """
    Validate cluster centroid.
    
    Args:
        centroid: Cluster centroid
        expected_length: Expected length of centroid
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If validation fails
    """
    if not centroid:
        raise ValidationError("Centroid cannot be empty")
    
    if expected_length is not None and len(centroid) != expected_length:
        raise ValidationError(
            f"Centroid length mismatch: expected {expected_length}, got {len(centroid)}"
        )
    
    if any(not isinstance(x, (int, float)) for x in centroid):
        raise ValidationError("Centroid must contain numeric values")
    
    return True


def validate_card_string(card_string: str, expected_length: Optional[int] = None) -> bool:
    """
    Validate card string format.
    
    Args:
        card_string: Card string (e.g., "AcKd")
        expected_length: Expected length in characters
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If validation fails
    """
    if len(card_string) % 2 != 0:
        raise ValidationError("Card string length must be even")
    
    if expected_length is not None and len(card_string) != expected_length:
        raise ValidationError(
            f"Card string length mismatch: expected {expected_length}, got {len(card_string)}"
        )
    
    return True

