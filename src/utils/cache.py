"""Caching utilities for expensive computations."""

import hashlib
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Callable

from ..config.settings import Config, get_config
from ..utils.logging import get_logger

logger = get_logger(__name__)


class DiskCache:
    """Disk-based caching for expensive computations."""
    
    def __init__(self, cache_dir: Optional[Path] = None, config: Optional[Config] = None):
        """
        Initialize disk cache.
        
        Args:
            cache_dir: Cache directory. If None, uses config cache_dir.
            config: Configuration instance
        """
        self.config = config or get_config()
        self.cache_dir = cache_dir or self.config.cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_data = pickle.dumps((args, kwargs))
        return hashlib.md5(key_data).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get cached value.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache {key}: {e}")
        return None
    
    def set(self, key: str, value: Any) -> None:
        """
        Cache a value.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        cache_file = self.cache_dir / f"{key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)
        except Exception as e:
            logger.warning(f"Failed to save cache {key}: {e}")
    
    def clear(self) -> None:
        """Clear all cached files."""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
        logger.info("Cache cleared")


def cached_function(cache: Optional[DiskCache] = None):
    """
    Decorator for caching function results.
    
    Args:
        cache: DiskCache instance. If None, creates new one.
    """
    def decorator(func: Callable):
        _cache = cache or DiskCache()
        
        @lru_cache(maxsize=128)
        def memory_cached(*args, **kwargs):
            # Try memory cache first
            return func(*args, **kwargs)
        
        def wrapper(*args, **kwargs):
            # Try disk cache
            cache_key = _cache._get_cache_key(func.__name__, *args, **kwargs)
            cached_value = _cache.get(cache_key)
            
            if cached_value is not None:
                return cached_value
            
            # Compute and cache
            result = memory_cached(*args, **kwargs)
            _cache.set(cache_key, result)
            return result
        
        return wrapper
    return decorator

