"""Configuration management with environment variable support."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .constants import (
    DEFAULT_SUIT_COUNT,
    DEFAULT_RANK_COUNT,
    DEFAULT_HOLD_CARD_COUNT,
    DEFAULT_RIVER_CLUSTER_COUNT,
    DEFAULT_TURN_CLUSTER_COUNT,
    DEFAULT_FLOP_CLUSTER_COUNT,
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_WORKERS,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_CONVERGENCE_THRESHOLD,
    DEFAULT_TURN_SAMPLE_COUNT,
    DEFAULT_RIVER_SAMPLE_COUNT,
    DEFAULT_OPPONENT_SAMPLE_COUNT,
    ClusterInitializationMethod,
)


@dataclass
class Config:
    """
    Application configuration with validation.
    
    Supports loading from environment variables and provides sensible defaults.
    """
    
    # Card settings
    suit_count: int = field(default=DEFAULT_SUIT_COUNT)
    rank_count: int = field(default=DEFAULT_RANK_COUNT)
    hold_card_count: int = field(default=DEFAULT_HOLD_CARD_COUNT)
    
    # Cluster settings
    river_cluster_count: int = field(default=DEFAULT_RIVER_CLUSTER_COUNT)
    turn_cluster_count: int = field(default=DEFAULT_TURN_CLUSTER_COUNT)
    flop_cluster_count: int = field(default=DEFAULT_FLOP_CLUSTER_COUNT)
    
    # Clustering algorithm settings
    cluster_initialization_method: ClusterInitializationMethod = field(
        default=ClusterInitializationMethod.KMEANS_PLUS_PLUS
    )
    max_iterations: int = field(default=DEFAULT_MAX_ITERATIONS)
    convergence_threshold: float = field(default=DEFAULT_CONVERGENCE_THRESHOLD)
    
    # Paths
    data_dir: Path = field(default_factory=lambda: Path("data"))
    log_dir: Path = field(default_factory=lambda: Path("logs"))
    cache_dir: Path = field(default_factory=lambda: Path("cache"))
    output_dir: Path = field(default_factory=lambda: Path("output"))
    
    # Performance settings
    batch_size: int = field(default=DEFAULT_BATCH_SIZE)
    num_workers: int = field(default=DEFAULT_NUM_WORKERS)
    use_multiprocessing: bool = field(default=True)
    
    # Sampling settings
    turn_sample_count: int = field(default=DEFAULT_TURN_SAMPLE_COUNT)
    river_sample_count: int = field(default=DEFAULT_RIVER_SAMPLE_COUNT)
    opponent_sample_count: int = field(default=DEFAULT_OPPONENT_SAMPLE_COUNT)
    use_exact_enumeration: bool = field(default=False)
    normalize_distributions: bool = field(default=False)
    
    # Logging settings
    log_level: str = field(default="INFO")
    log_to_file: bool = field(default=True)
    log_to_console: bool = field(default=True)
    log_file_max_bytes: int = field(default=10 * 1024 * 1024)  # 10MB
    log_file_backup_count: int = field(default=5)
    
    # Progress bar settings
    show_progress: bool = field(default=True)
    
    def __post_init__(self) -> None:
        """Validate configuration and create directories."""
        # Validate card settings
        if self.suit_count <= 0:
            raise ValueError("suit_count must be positive")
        if self.rank_count <= 0:
            raise ValueError("rank_count must be positive")
        if self.hold_card_count <= 0:
            raise ValueError("hold_card_count must be positive")
        
        # Validate cluster settings
        if self.river_cluster_count <= 0:
            raise ValueError("river_cluster_count must be positive")
        if self.turn_cluster_count <= 0:
            raise ValueError("turn_cluster_count must be positive")
        if self.flop_cluster_count <= 0:
            raise ValueError("flop_cluster_count must be positive")
        
        # Validate performance settings
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.num_workers <= 0:
            raise ValueError("num_workers must be positive")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        
        # Validate sampling settings
        if self.turn_sample_count <= 0:
            raise ValueError("turn_sample_count must be positive")
        if self.river_sample_count <= 0:
            raise ValueError("river_sample_count must be positive")
        if self.opponent_sample_count <= 0:
            raise ValueError("opponent_sample_count must be positive")
        
        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def card_count(self) -> int:
        """Total number of cards."""
        return self.suit_count * self.rank_count
    
    @classmethod
    def from_env(cls) -> "Config":
        """
        Create configuration from environment variables.
        
        Environment variables:
        - SUIT_COUNT: Number of suits
        - RANK_COUNT: Number of ranks
        - RIVER_CLUSTER_COUNT: Number of river clusters
        - TURN_CLUSTER_COUNT: Number of turn clusters
        - FLOP_CLUSTER_COUNT: Number of flop clusters
        - DATA_DIR: Data directory path
        - LOG_DIR: Log directory path
        - BATCH_SIZE: Batch size for processing
        - NUM_WORKERS: Number of worker processes/threads
        - LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR)
        
        Returns:
            Config instance with values from environment variables
        """
        return cls(
            suit_count=int(os.getenv("SUIT_COUNT", DEFAULT_SUIT_COUNT)),
            rank_count=int(os.getenv("RANK_COUNT", DEFAULT_RANK_COUNT)),
            hold_card_count=int(os.getenv("HOLD_CARD_COUNT", DEFAULT_HOLD_CARD_COUNT)),
            river_cluster_count=int(os.getenv("RIVER_CLUSTER_COUNT", DEFAULT_RIVER_CLUSTER_COUNT)),
            turn_cluster_count=int(os.getenv("TURN_CLUSTER_COUNT", DEFAULT_TURN_CLUSTER_COUNT)),
            flop_cluster_count=int(os.getenv("FLOP_CLUSTER_COUNT", DEFAULT_FLOP_CLUSTER_COUNT)),
            cluster_initialization_method=ClusterInitializationMethod(
                os.getenv("CLUSTER_INIT_METHOD", ClusterInitializationMethod.KMEANS_PLUS_PLUS.value)
            ),
            max_iterations=int(os.getenv("MAX_ITERATIONS", DEFAULT_MAX_ITERATIONS)),
            convergence_threshold=float(os.getenv("CONVERGENCE_THRESHOLD", DEFAULT_CONVERGENCE_THRESHOLD)),
            data_dir=Path(os.getenv("DATA_DIR", "data")),
            log_dir=Path(os.getenv("LOG_DIR", "logs")),
            cache_dir=Path(os.getenv("CACHE_DIR", "cache")),
            output_dir=Path(os.getenv("OUTPUT_DIR", "output")),
            batch_size=int(os.getenv("BATCH_SIZE", DEFAULT_BATCH_SIZE)),
            num_workers=int(os.getenv("NUM_WORKERS", DEFAULT_NUM_WORKERS)),
            use_multiprocessing=os.getenv("USE_MULTIPROCESSING", "true").lower() == "true",
            turn_sample_count=int(os.getenv("TURN_SAMPLE_COUNT", DEFAULT_TURN_SAMPLE_COUNT)),
            river_sample_count=int(os.getenv("RIVER_SAMPLE_COUNT", DEFAULT_RIVER_SAMPLE_COUNT)),
            opponent_sample_count=int(os.getenv("OPPONENT_SAMPLE_COUNT", DEFAULT_OPPONENT_SAMPLE_COUNT)),
            use_exact_enumeration=os.getenv("USE_EXACT_ENUMERATION", "false").lower() == "true",
            normalize_distributions=os.getenv("NORMALIZE_DISTRIBUTIONS", "false").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_to_file=os.getenv("LOG_TO_FILE", "true").lower() == "true",
            log_to_console=os.getenv("LOG_TO_CONSOLE", "true").lower() == "true",
            show_progress=os.getenv("SHOW_PROGRESS", "true").lower() == "true",
            log_file_max_bytes=int(os.getenv("LOG_FILE_MAX_BYTES", str(10 * 1024 * 1024))),
            log_file_backup_count=int(os.getenv("LOG_FILE_BACKUP_COUNT", "5")),
        )
    
    @classmethod
    def default(cls) -> "Config":
        """Create configuration with default values."""
        return cls()


# Global configuration instance (can be overridden)
_config: Optional[Config] = None


def get_config() -> Config:
    """
    Get the global configuration instance.
    
    Returns:
        Current configuration instance
    """
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config


def set_config(config: Config) -> None:
    """
    Set the global configuration instance.
    
    Args:
        config: Configuration instance to use
    """
    global _config
    _config = config

