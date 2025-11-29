#!/usr/bin/env python3
"""Script to map hands to cluster buckets."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from src.core.data.mapper import ClusterMapper
from src.config.settings import Config
from src.config.constants import PokerRound
from src.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Map hands to cluster buckets")
    parser.add_argument(
        "--street",
        type=str,
        choices=["river", "turn", "flop"],
        required=True,
        help="Poker round"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Data directory"
    )
    parser.add_argument(
        "--exact",
        action="store_true",
        help="Use exact enumeration instead of sampling"
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize distributions"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path (default: auto-generated)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    config = Config.from_env()
    config.data_dir = Path(args.data_dir)
    setup_logging(config)
    
    # Create mapper
    street = PokerRound(args.street)
    mapper = ClusterMapper(
        street=street,
        config=config,
        use_exact=args.exact,
        normalize=args.normalize
    )
    
    # Compute mappings
    logger.info(f"Computing cluster mappings for {street.value} round")
    mappings = mapper.compute_all_mappings()
    
    # Save mappings
    mapper.save_mappings(mappings, args.output)
    logger.info("Mapping completed")


if __name__ == "__main__":
    main()

