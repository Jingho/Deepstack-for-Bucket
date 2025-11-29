#!/usr/bin/env python3
"""Script to analyze cluster centroids."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from src.analysis.cluster_analyzer import ClusterAnalyzer
from src.config.settings import Config
from src.config.constants import PokerRound
from src.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Analyze cluster centroids")
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
        help="Cluster count (default: from config)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Data directory"
    )
    parser.add_argument(
        "--num-pairs",
        type=int,
        default=5,
        help="Number of closest pairs to find"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    config = Config.from_env()
    config.data_dir = Path(args.data_dir)
    setup_logging(config)
    
    # Create analyzer
    street = PokerRound(args.street)
    analyzer = ClusterAnalyzer(config)
    
    # Analyze clusters
    logger.info(f"Analyzing clusters for {street.value} round")
    closest_pairs = analyzer.analyze_clusters(street, args.k)
    
    # Save results
    k = args.k or analyzer._get_cluster_count(street)
    output_path = args.output or config.data_dir / f"{street.value}_cluster_analysis_k{k}.txt"
    analyzer.save_analysis(closest_pairs, str(output_path), k)
    
    logger.info("Analysis completed")


if __name__ == "__main__":
    main()

