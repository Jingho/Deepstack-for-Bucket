"""Map poker hands to cluster buckets."""

from typing import List, Optional, Dict
from pathlib import Path
import random
from itertools import combinations
import numpy as np
from pyemd import emd
from tqdm import tqdm

from ...config.settings import Config, get_config
from ...config.constants import PokerRound
from ...utils.logging import get_logger
from ..poker.card import CardConverter
from ..poker.equity_calculator import EquityCalculator
from ..poker.hand_evaluator import HandEvaluator
from ..clustering.distance import DistanceMatrixCalculator

logger = get_logger(__name__)


class ClusterMapper:
    """
    Map poker hands to cluster buckets.
    
    For each round, computes which cluster bucket a given hand belongs to
    based on the clustering results from previous rounds.
    """
    
    def __init__(
        self,
        street: PokerRound,
        config: Optional[Config] = None,
        use_exact: Optional[bool] = None,
        normalize: bool = False
    ):
        """
        Initialize cluster mapper.
        
        Args:
            street: Round name
            config: Configuration instance
            use_exact: Use exact enumeration instead of sampling
            normalize: Normalize distributions
        """
        self.config = config or get_config()
        self.street = street
        self.use_exact = use_exact if use_exact is not None else self.config.use_exact_enumeration
        self.normalize = normalize or self.config.normalize_distributions
        
        # Initialize components
        self.card_converter = CardConverter(self.config)
        self.hand_evaluator = HandEvaluator(self.card_converter)
        self.equity_calculator = EquityCalculator(
            self.hand_evaluator,
            self.card_converter,
            self.config
        )
        self.distance_calculator = DistanceMatrixCalculator(self.config)
        
        # Load cluster centroids
        self.river_centroids: List[List[float]] = []
        self.turn_centroids: List[List[float]] = []
        self.flop_centroids: List[List[float]] = []
        
        self._load_centroids()
        
        # Generate all cards
        self.all_cards = self.card_converter.generate_all_cards()
    
    def _load_centroids(self) -> None:
        """Load cluster centroids for all relevant rounds."""
        if self.street in [PokerRound.RIVER, PokerRound.TURN, PokerRound.FLOP]:
            self.river_centroids = self._load_centroid_file(PokerRound.RIVER)
        
        if self.street in [PokerRound.TURN, PokerRound.FLOP]:
            self.turn_centroids = self._load_centroid_file(PokerRound.TURN)
        
        if self.street == PokerRound.FLOP:
            self.flop_centroids = self._load_centroid_file(PokerRound.FLOP)
    
    def _load_centroid_file(self, street: PokerRound) -> List[List[float]]:
        """Load centroids from file."""
        possible_names = [
            f"{street.value}_cluster_k{self._get_cluster_count(street)}.csv",
            f"{street.value}_cluster.csv"
        ]
        
        for filename in possible_names:
            file_path = self.config.data_dir / filename
            if file_path.exists():
                return self._read_csv(file_path)
        
        raise FileNotFoundError(
            f"Cluster file not found for {street.value}. Tried: {possible_names}"
        )
    
    def _get_cluster_count(self, street: PokerRound) -> int:
        """Get cluster count for street."""
        counts = {
            PokerRound.RIVER: self.config.river_cluster_count,
            PokerRound.TURN: self.config.turn_cluster_count,
            PokerRound.FLOP: self.config.flop_cluster_count
        }
        return counts.get(street, 5)
    
    def _read_csv(self, file_path: Path) -> List[List[float]]:
        """
        Read CSV file with error handling.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            List of data points as lists of floats
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is empty or contains invalid data
            IOError: If file cannot be read
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    line = line.strip()
                    if not line:
                        continue
                    values = [float(x) for x in line.split(',') if x]
                    if values:
                        data.append(values)
                except ValueError as e:
                    logger.warning(
                        f"Skipping invalid line {line_num} in {file_path}: {e}"
                    )
                    continue
        
        if not data:
            raise ValueError(f"File {file_path} contains no valid data")
        
        return data
    
    def map_hand_to_cluster(
        self,
        hand: List[str],
        board: List[str]
    ) -> int:
        """
        Map a hand and board to cluster index.
        
        Args:
            hand: Player's hand as list of card strings
            board: Board cards as list of card strings
            
        Returns:
            Cluster index (1-indexed)
        """
        if self.street == PokerRound.RIVER:
            return self._map_river(hand, board)
        elif self.street == PokerRound.TURN:
            return self._map_turn(hand, board)
        elif self.street == PokerRound.FLOP:
            return self._map_flop(hand, board)
        else:
            raise ValueError(f"Unknown street: {self.street}")
    
    def _map_river(self, hand: List[str], board: List[str]) -> int:
        """Map river hand to cluster."""
        used_cards = hand + board
        available_cards = [c for c in self.all_cards if c not in used_cards]
        
        hand_str = ''.join(hand)
        board_str = ''.join(board)
        
        equity = self.equity_calculator.compute_equity(
            hand_str,
            board_str,
            available_cards,
            use_exact=self.use_exact
        )
        
        cluster_idx = self._find_nearest_river_cluster(equity)
        return cluster_idx + 1  # Return 1-indexed
    
    def _map_turn(self, hand: List[str], board: List[str]) -> int:
        """Map turn hand to cluster."""
        used_cards = hand + board
        available_cards = [c for c in self.all_cards if c not in used_cards]
        hand_str = ''.join(hand)
        
        # Distribution over river clusters
        river_distribution = [0.0] * len(self.river_centroids)
        
        if self.use_exact:
            for river_card in available_cards:
                river_board = board + [river_card]
                opponent_cards = [c for c in available_cards if c != river_card]
                board_str = ''.join(river_board)
                
                equity = self.equity_calculator.compute_equity(
                    hand_str,
                    board_str,
                    opponent_cards,
                    use_exact=True
                )
                cluster_idx = self._find_nearest_river_cluster(equity)
                river_distribution[cluster_idx] += 1.0
        else:
            for _ in range(self.config.river_sample_count):
                river_card = random.choice(available_cards)
                river_board = board + [river_card]
                opponent_cards = [c for c in available_cards if c != river_card]
                board_str = ''.join(river_board)
                
                equity = self.equity_calculator.compute_equity(
                    hand_str,
                    board_str,
                    opponent_cards,
                    use_exact=False
                )
                cluster_idx = self._find_nearest_river_cluster(equity)
                river_distribution[cluster_idx] += 1.0
        
        # Normalize if requested
        if self.normalize:
            total = sum(river_distribution)
            if total > 0:
                river_distribution = [d / total for d in river_distribution]
        
        # Find nearest turn cluster using EMD
        turn_matrix = self.distance_calculator.compute_euclidean_matrix(self.river_centroids)
        min_distance = float('inf')
        min_index = 0
        
        for i, turn_centroid in enumerate(self.turn_centroids):
            distance = emd(
                np.array(river_distribution, dtype=np.float64),
                np.array(turn_centroid, dtype=np.float64),
                turn_matrix
            )
            if distance < min_distance:
                min_distance = distance
                min_index = i
        
        return min_index + 1  # Return 1-indexed
    
    def _map_flop(self, hand: List[str], board: List[str]) -> int:
        """Map flop hand to cluster."""
        used_cards = hand + board
        available_cards = [c for c in self.all_cards if c not in used_cards]
        hand_str = ''.join(hand)
        
        # Distribution over turn clusters
        turn_distribution = [0.0] * len(self.turn_centroids)
        turn_matrix = self.distance_calculator.compute_euclidean_matrix(self.river_centroids)
        
        if self.use_exact:
            for turn_card in available_cards:
                turn_board = board + [turn_card]
                river_cards = [c for c in available_cards if c != turn_card]
                
                # Distribution over river clusters
                river_distribution = [0.0] * len(self.river_centroids)
                
                for river_card in river_cards:
                    river_board = turn_board + [river_card]
                    opponent_cards = [c for c in river_cards if c != river_card]
                    board_str = ''.join(river_board)
                    
                    equity = self.equity_calculator.compute_equity(
                        hand_str,
                        board_str,
                        opponent_cards,
                        use_exact=True
                    )
                    cluster_idx = self._find_nearest_river_cluster(equity)
                    river_distribution[cluster_idx] += 1.0
                
                # Normalize river distribution before computing EMD
                total = sum(river_distribution)
                if total > 0:
                    river_distribution = [d / total for d in river_distribution]
                
                # Find nearest turn cluster
                min_distance = float('inf')
                min_index = 0
                for i, turn_centroid in enumerate(self.turn_centroids):
                    distance = emd(
                        np.array(river_distribution, dtype=np.float64),
                        np.array(turn_centroid, dtype=np.float64),
                        turn_matrix
                    )
                    if distance < min_distance:
                        min_distance = distance
                        min_index = i
                
                turn_distribution[min_index] += 1.0 / len(available_cards)
        else:
            for _ in range(self.config.turn_sample_count):
                turn_card = random.choice(available_cards)
                turn_board = board + [turn_card]
                river_cards = [c for c in available_cards if c != turn_card]
                
                river_distribution = [0.0] * len(self.river_centroids)
                
                for _ in range(self.config.river_sample_count):
                    river_card = random.choice(river_cards)
                    river_board = turn_board + [river_card]
                    opponent_cards = [c for c in river_cards if c != river_card]
                    board_str = ''.join(river_board)
                    
                    equity = self.equity_calculator.compute_equity(
                        hand_str,
                        board_str,
                        opponent_cards,
                        use_exact=False
                    )
                    cluster_idx = self._find_nearest_river_cluster(equity)
                    river_distribution[cluster_idx] += 1.0
                
                # Normalize river distribution before computing EMD
                total = sum(river_distribution)
                if total > 0:
                    river_distribution = [d / total for d in river_distribution]
                
                min_distance = float('inf')
                min_index = 0
                for i, turn_centroid in enumerate(self.turn_centroids):
                    distance = emd(
                        np.array(river_distribution, dtype=np.float64),
                        np.array(turn_centroid, dtype=np.float64),
                        turn_matrix
                    )
                    if distance < min_distance:
                        min_distance = distance
                        min_index = i
                
                turn_distribution[min_index] += 1.0
        
        # Normalize turn distribution
        if self.normalize:
            total = sum(turn_distribution)
            if total > 0:
                turn_distribution = [d / total for d in turn_distribution]
        
        # Find nearest flop cluster
        flop_matrix = self.distance_calculator.compute_emd_matrix(
            turn_matrix,
            self.turn_centroids
        )
        min_distance = float('inf')
        min_index = 0
        
        for i, flop_centroid in enumerate(self.flop_centroids):
            distance = emd(
                np.array(turn_distribution, dtype=np.float64),
                np.array(flop_centroid, dtype=np.float64),
                flop_matrix
            )
            if distance < min_distance:
                min_distance = distance
                min_index = i
        
        return min_index + 1  # Return 1-indexed
    
    def _find_nearest_river_cluster(self, equity: float) -> int:
        """Find nearest river cluster for equity value."""
        distances = [abs(equity - centroid[0]) for centroid in self.river_centroids]
        return int(np.argmin(distances))
    
    def compute_all_mappings(self) -> Dict[str, int]:
        """
        Compute cluster mappings for all possible hands.
        
        Returns:
            Dictionary mapping card index strings to cluster indices
        """
        from scipy.special import comb
        
        mappings: Dict[str, int] = {}
        
        # Determine state count
        state_counts = {
            PokerRound.RIVER: 7,
            PokerRound.TURN: 6,
            PokerRound.FLOP: 5
        }
        state_count = state_counts.get(self.street, 5)
        
        all_states = list(combinations(self.all_cards, state_count))
        total_states = len(all_states)
        
        logger.info(f"Computing mappings for {total_states} states")
        
        state_iterator = tqdm(
            enumerate(all_states),
            total=total_states,
            desc=f"Mapping {self.street.value}",
            disable=not self.config.show_progress,
            unit="state"
        )
        for state_idx, state in state_iterator:
            
            state_list = list(state)
            all_hands = list(combinations(state_list, self.config.hold_card_count))
            
            for hand in all_hands:
                hand_list = list(hand)
                board = [c for c in state_list if c not in hand_list]
                
                cluster_index = self.map_hand_to_cluster(hand_list, board)
                
                # Convert to index format
                hand_card = ''.join(hand_list)
                board_card = ''.join(board)
                
                hand_card_index = self.card_converter.string_to_board(hand_card)
                board_card_index = self.card_converter.string_to_board(board_card)
                hand_card_index.sort()
                board_card_index.sort()
                
                hand_index_str = ''.join([str(i) for i in hand_card_index])
                board_index_str = ''.join([str(i) for i in board_card_index])
                card_index_string = hand_index_str + board_index_str
                
                mappings[card_index_string] = cluster_index
        
        logger.info(f"Computed {len(mappings)} mappings")
        return mappings
    
    def save_mappings(self, mappings: Dict[str, int], output_path: Optional[str] = None) -> None:
        """
        Save cluster mappings to file.
        
        Args:
            mappings: Dictionary of mappings
            output_path: Output file path. If None, uses default.
        """
        if output_path is None:
            cluster_count = self._get_cluster_count(self.street)
            output_path = self.config.data_dir / f"{self.street.value}_cluster_result_k{cluster_count}.txt"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for card_index, cluster_index in mappings.items():
                f.write(f"{card_index}:{cluster_index}\n")
        
        logger.info(f"Mappings saved to {output_path}")

