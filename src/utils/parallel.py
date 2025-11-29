"""Parallel processing utilities."""

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import List, Callable, Any, Optional, Iterator

from ..config.settings import Config, get_config
from ..utils.logging import get_logger

logger = get_logger(__name__)


class ParallelProcessor:
    """Process data in parallel."""
    
    def __init__(
        self,
        num_workers: Optional[int] = None,
        use_processes: Optional[bool] = None,
        config: Optional[Config] = None
    ):
        """
        Initialize parallel processor.
        
        Args:
            num_workers: Number of workers. If None, uses config.
            use_processes: Use processes instead of threads. If None, uses config.
            config: Configuration instance
        """
        self.config = config or get_config()
        self.num_workers = num_workers or self.config.num_workers
        self.use_processes = use_processes if use_processes is not None else self.config.use_multiprocessing
        self.executor_class = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
    
    def process_batch(
        self,
        items: List[Any],
        func: Callable,
        batch_size: Optional[int] = None
    ) -> List[Any]:
        """
        Process items in parallel batches.
        
        Args:
            items: List of items to process
            func: Function to apply to each item
            batch_size: Batch size. If None, uses config batch_size.
            
        Returns:
            List of results
        """
        batch_size = batch_size or self.config.batch_size
        results = []
        
        with self.executor_class(max_workers=self.num_workers) as executor:
            # Submit batches
            futures = []
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                future = executor.submit(self._process_batch, batch, func)
                futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                batch_results = future.result()
                results.extend(batch_results)
        
        return results
    
    @staticmethod
    def _process_batch(batch: List[Any], func: Callable) -> List[Any]:
        """Process a single batch."""
        return [func(item) for item in batch]
    
    def map(self, func: Callable, items: List[Any]) -> Iterator[Any]:
        """
        Map function over items in parallel.
        
        Args:
            func: Function to apply
            items: Items to process
            
        Yields:
            Results as they become available
        """
        with self.executor_class(max_workers=self.num_workers) as executor:
            futures = {executor.submit(func, item): item for item in items}
            
            for future in as_completed(futures):
                yield future.result()

