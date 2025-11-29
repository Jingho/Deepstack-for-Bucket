"""Performance monitoring and metrics collection."""

import time
from functools import wraps
from typing import Callable, Dict, Any, Optional
from collections import defaultdict

from ..utils.logging import get_logger

logger = get_logger(__name__)


class PerformanceMonitor:
    """Monitor and collect performance metrics."""
    
    def __init__(self):
        """Initialize performance monitor."""
        self.metrics: Dict[str, list] = defaultdict(list)
        self.enabled = True
    
    def time_function(self, func_name: Optional[str] = None):
        """
        Decorator to time function execution.
        
        Args:
            func_name: Name for the function in metrics. If None, uses function name.
        """
        def decorator(func: Callable):
            name = func_name or func.__name__
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not self.enabled:
                    return func(*args, **kwargs)
                
                start_time = time.time()
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                self.metrics[name].append(elapsed)
                logger.debug(f"{name} took {elapsed:.4f}s")
                return result
            
            return wrapper
        return decorator
    
    def get_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics for all collected metrics.
        
        Returns:
            Dictionary mapping function names to statistics
        """
        summary = {}
        for name, times in self.metrics.items():
            if times:
                summary[name] = {
                    'count': len(times),
                    'total': sum(times),
                    'mean': sum(times) / len(times),
                    'min': min(times),
                    'max': max(times)
                }
        return summary
    
    def reset(self) -> None:
        """Reset all collected metrics."""
        self.metrics.clear()


# Global monitor instance
_monitor = PerformanceMonitor()


def get_monitor() -> PerformanceMonitor:
    """Get the global performance monitor."""
    return _monitor

