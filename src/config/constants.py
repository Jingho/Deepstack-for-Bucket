"""Constants used throughout the application."""

from enum import Enum
from typing import List


class PokerRound(str, Enum):
    """Poker round enumeration."""
    
    RIVER = "river"
    TURN = "turn"
    FLOP = "flop"
    PREFLOP = "preflop"


class ClusterInitializationMethod(str, Enum):
    """K-means initialization method."""
    
    KMEANS_PLUS_PLUS = "kmeans++"
    RANDOM = "random"


# Card configuration
DEFAULT_SUIT_COUNT = 2
DEFAULT_RANK_COUNT = 6
DEFAULT_HOLD_CARD_COUNT = 2

# Board card counts for each round
BOARD_CARD_COUNTS: List[int] = [0, 3, 4, 5]  # [preflop, flop, turn, river]

# Default cluster counts
DEFAULT_RIVER_CLUSTER_COUNT = 5
DEFAULT_TURN_CLUSTER_COUNT = 5
DEFAULT_FLOP_CLUSTER_COUNT = 80

# Performance defaults
DEFAULT_BATCH_SIZE = 1000
DEFAULT_NUM_WORKERS = 4
DEFAULT_MAX_ITERATIONS = 100
DEFAULT_CONVERGENCE_THRESHOLD = 1e-10

# Sampling defaults
DEFAULT_TURN_SAMPLE_COUNT = 7
DEFAULT_RIVER_SAMPLE_COUNT = 6
DEFAULT_OPPONENT_SAMPLE_COUNT = 100

