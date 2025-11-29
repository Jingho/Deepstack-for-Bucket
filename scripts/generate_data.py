#!/usr/bin/env python3
"""Script to generate hand representation data."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from src.core.data.generator import DataGenerator
from src.config.settings import Config
from src.config.constants import PokerRound
from src.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate hand representation data")
    parser.add_argument(
        "--street",
        type=str,
        choices=["river", "turn", "flop"],
        default="river",
        help="Poker round"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Data directory"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    config = Config.from_env()
    config.data_dir = Path(args.data_dir)
    setup_logging(config)
    
    # Create generator
    street = PokerRound(args.street)
    generator = DataGenerator(street=street, config=config)
    
    # Generate data
    logger.info(f"Starting data generation for {street.value} round")
    generator.generate()
    logger.info("Data generation completed")


if __name__ == "__main__":
    main()

