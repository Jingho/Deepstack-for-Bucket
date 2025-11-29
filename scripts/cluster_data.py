#!/usr/bin/env python3
"""Script to cluster hand representation data."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from src.core.clustering.clusterer import Clusterer
from src.config.settings import Config
from src.config.constants import PokerRound, ClusterInitializationMethod
from src.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Cluster hand representation data")
    parser.add_argument(
        "--street",
        type=str,
        choices=["river", "turn", "flop"],
        required=True,
        help="Poker round"
    )
    parser.add_argument(
        "--k",
        type=int,
        required=True,
        help="Number of clusters"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Data directory"
    )
    parser.add_argument(
        "--initial-method",
        type=str,
        choices=["kmeans++", "random"],
        default="kmeans++",
        help="Initialization method"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save cluster centroids"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    config = Config.from_env()
    config.data_dir = Path(args.data_dir)
    setup_logging(config)
    
    # Create clusterer
    street = PokerRound(args.street)
    init_method = ClusterInitializationMethod(args.initial_method)
    
    clusterer = Clusterer(
        street=street,
        k=args.k,
        config=config,
        initial_method=init_method,
        save_results=not args.no_save
    )
    
    # Perform clustering
    logger.info(f"Starting clustering for {street.value} round with k={args.k}")
    centroids = clusterer.cluster()
    logger.info(f"Clustering completed. Found {len(centroids)} clusters")


if __name__ == "__main__":
    main()

