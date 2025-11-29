"""Data generation for potential-aware imperfect-recall abstraction."""

from typing import List, Optional
from pathlib import Path
from itertools import combinations
import numpy as np
from tqdm import tqdm

from ...config.settings import Config, get_config
from ...config.constants import PokerRound
from ...utils.logging import get_logger
from ..poker.card import CardConverter
from ..poker.equity_calculator import EquityCalculator
from ..poker.hand_evaluator import HandEvaluator
from ..clustering.distance import DistanceMatrixCalculator

logger = get_logger(__name__)


class DataGenerator:
    """
    Generate hand representation data for clustering.
    
    For each round:
    - River: Compute equity (win rate) directly
    - Turn: Compute distribution over river clusters
    - Flop: Compute distribution over turn clusters (which are distributions over river clusters)
    """
    
    def __init__(
        self,
        street: PokerRound,
        config: Optional[Config] = None
    ):
        """
        Initialize data generator.
        
        Args:
            street: Round name
            config: Configuration instance
        """
        self.config = config or get_config()
        self.street = street
        
        # Initialize components
        self.card_converter = CardConverter(self.config)
        self.hand_evaluator = HandEvaluator(self.card_converter)
        self.equity_calculator = EquityCalculator(
            self.hand_evaluator,
            self.card_converter,
            self.config
        )
        self.distance_calculator = DistanceMatrixCalculator(self.config)
        
        # Initialize round-specific data
        self.river_centroids: Optional[List[List[float]]] = None
        self.turn_centroids: Optional[List[List[float]]] = None
        self.distance_matrix: Optional[np.ndarray] = None
        
        self._initialize_round_data()
        
        # Generate all cards
        self.all_cards = self.card_converter.generate_all_cards()
        if not self.all_cards:
            raise ValueError(f"No cards generated. Check suit_count ({self.config.suit_count}) and rank_count ({self.config.rank_count})")
        
        # Set save path
        self.save_path = self.config.data_dir / f"{street.value}_data.csv"
    
    def _initialize_round_data(self) -> None:
        """Initialize round-specific cluster data."""
        if self.street == PokerRound.RIVER:
            return
        
        # Load river centroids
        river_file = self.config.data_dir / "river_cluster.csv"
        if not river_file.exists():
            # Try with k in filename
            river_file = self.config.data_dir / f"river_cluster_k{self.config.river_cluster_count}.csv"
        
        if river_file.exists():
            self.river_centroids = self._load_csv(river_file)
        else:
            raise FileNotFoundError(
                f"River cluster file not found. Please cluster river round first."
            )
        
        if self.street == PokerRound.FLOP:
            # Load turn centroids and compute distance matrix
            turn_file = self.config.data_dir / "turn_cluster.csv"
            if not turn_file.exists():
                turn_file = self.config.data_dir / f"turn_cluster_k{self.config.turn_cluster_count}.csv"
            
            if turn_file.exists():
                self.turn_centroids = self._load_csv(turn_file)
                
                # Compute distance matrix: river -> turn
                river_matrix = self.distance_calculator.compute_euclidean_matrix(self.river_centroids)
                self.distance_matrix = self.distance_calculator.compute_emd_matrix(
                    river_matrix,
                    self.turn_centroids
                )
            else:
                raise FileNotFoundError(
                    f"Turn cluster file not found. Please cluster turn round first."
                )
    
    def _load_csv(self, file_path: Path) -> List[List[float]]:
        """
        Load CSV file with error handling.
        
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
    
    def generate(self) -> None:
        """
        Generate data for the specified round.
        """
        logger.info(f"Starting data generation for {self.street.value} round")
        
        if self.street == PokerRound.RIVER:
            self._generate_river_data()
        elif self.street == PokerRound.TURN:
            self._generate_turn_data()
        elif self.street == PokerRound.FLOP:
            self._generate_flop_data()
        else:
            raise ValueError(f"Unknown street: {self.street}")
        
        logger.info(f"Data generation completed. Saved to {self.save_path}")
    
    def _generate_river_data(self) -> None:
        """Generate data for river round."""
        buffer = []
        buffer_size = self.config.batch_size
        
        all_states = list(combinations(self.all_cards, 7))
        total_states = len(all_states)
        
        logger.info(f"Generating river data for {total_states} states")
        
        with open(self.save_path, 'wt', encoding='utf-8') as f:
            state_iterator = tqdm(
                enumerate(all_states),
                total=total_states,
                desc="River data",
                disable=not self.config.show_progress,
                unit="state"
            )
            for state_idx, state in state_iterator:
                
                state_list = list(state)
                all_hands = list(combinations(state_list, 2))
                
                for hand in all_hands:
                    hand_list = list(hand)
                    hand_card = hand_list[0] + hand_list[1]
                    public_card = [card for card in state_list if card not in hand_list]
                    public_string = ''.join(public_card)
                    free_cards = [c for c in self.all_cards if c not in state_list]
                    
                    equity = self.equity_calculator.compute_equity(
                        hand_card,
                        public_string,
                        free_cards,
                        use_exact=self.config.use_exact_enumeration
                    )
                    buffer.append(str(equity) + '\n')
                    
                    if len(buffer) >= buffer_size:
                        f.writelines(buffer)
                        buffer = []
            
            if buffer:
                f.writelines(buffer)
        
        logger.info(f"River data generation completed. Saved to {self.save_path}")
    
    def _generate_turn_data(self) -> None:
        """Generate data for turn round."""
        buffer = []
        buffer_size = self.config.batch_size
        
        all_states = list(combinations(self.all_cards, 6))
        total_states = len(all_states)
        
        logger.info(f"Generating turn data for {total_states} states")
        
        with open(self.save_path, 'wt', encoding='utf-8') as f:
            state_iterator = tqdm(
                enumerate(all_states),
                total=total_states,
                desc="Turn data",
                disable=not self.config.show_progress,
                unit="state"
            )
            for state_idx, state in state_iterator:
                    
                    state_list = list(state)
                    all_hands = list(combinations(state_list, 2))
                    
                    for hand in all_hands:
                        hand_list = list(hand)
                        hand_card = hand_list[0] + hand_list[1]
                        public_card_4 = [card for card in state_list if card not in hand_list]
                        free_cards = [c for c in self.all_cards if c not in state_list]
                        
                        # Distribution over river clusters
                        river_distribution = [0.0] * len(self.river_centroids)
                        n_turn_cards = len(free_cards)
                        
                        for river_card in free_cards:
                            public_card_5 = public_card_4 + [river_card]
                            public_string = ''.join(public_card_5)
                            opponent_cards = [c for c in free_cards if c != river_card]
                            
                            equity = self.equity_calculator.compute_equity(
                                hand_card,
                                public_string,
                                opponent_cards,
                                use_exact=self.config.use_exact_enumeration
                            )
                            
                            cluster_idx = self._find_nearest_river_cluster(equity)
                            river_distribution[cluster_idx] += 1.0 / n_turn_cards
                        
                        line = ','.join(map(str, river_distribution)) + '\n'
                        buffer.append(line)
                        
                        if len(buffer) >= buffer_size:
                            f.writelines(buffer)
                            buffer = []
            
            if buffer:
                f.writelines(buffer)
        
        logger.info(f"Turn data generation completed. Saved to {self.save_path}")
    
    def _generate_flop_data(self) -> None:
        """Generate data for flop round."""
        buffer = []
        buffer_size = self.config.batch_size
        
        all_states = list(combinations(self.all_cards, 5))
        total_states = len(all_states)
        
        logger.info(f"Generating flop data for {total_states} states")
        
        with open(self.save_path, 'wt', encoding='utf-8') as f:
            state_iterator = tqdm(
                enumerate(all_states),
                total=total_states,
                desc="Flop data",
                disable=not self.config.show_progress,
                unit="state"
            )
            for state_idx, state in state_iterator:
                    
                    state_list = list(state)
                    all_hands = list(combinations(state_list, 2))
                    
                    for hand in all_hands:
                        hand_list = list(hand)
                        hand_card = hand_list[0] + hand_list[1]
                        public_card_3 = [card for card in state_list if card not in hand_list]
                        free_cards = [c for c in self.all_cards if c not in state_list]
                        
                        # Distribution over turn clusters
                        turn_distribution = [0.0] * len(self.turn_centroids)
                        n_flop_cards = len(free_cards)
                        
                        # Pre-compute river distance matrix once (used for all EMD calculations)
                        river_matrix = None
                        if self.distance_matrix is not None and self.river_centroids:
                            river_matrix = self.distance_calculator.compute_euclidean_matrix(self.river_centroids)
                        
                        for turn_card in free_cards:
                            public_cards_4 = public_card_3 + [turn_card]
                            river_cards = [c for c in free_cards if c != turn_card]
                            
                            # Distribution over river clusters for this turn card
                            river_distribution = [0.0] * len(self.river_centroids)
                            n_river_cards = len(river_cards)
                            
                            for river_card in river_cards:
                                public_card_5 = public_cards_4 + [river_card]
                                public_string = ''.join(public_card_5)
                                opponent_cards = [c for c in river_cards if c != river_card]
                                
                                equity = self.equity_calculator.compute_equity(
                                    hand_card,
                                    public_string,
                                    opponent_cards,
                                    use_exact=self.config.use_exact_enumeration
                                )
                                cluster_idx = self._find_nearest_river_cluster(equity)
                                river_distribution[cluster_idx] += 1.0 / n_river_cards
                        
                        # Find nearest turn cluster using EMD distance
                        if self.distance_matrix is not None and river_matrix is not None:
                            distances = [
                                self._compute_emd_distance(
                                    river_distribution,
                                    turn_centroid,
                                    river_matrix
                                )
                                for turn_centroid in self.turn_centroids
                            ]
                            turn_cluster_idx = int(np.argmin(distances))
                            turn_distribution[turn_cluster_idx] += 1.0 / n_flop_cards
                    
                    line = ','.join(map(str, turn_distribution)) + '\n'
                    buffer.append(line)
                    
                    if len(buffer) >= buffer_size:
                        f.writelines(buffer)
                        buffer = []
            
            if buffer:
                f.writelines(buffer)
        
        logger.info(f"Flop data generation completed. Saved to {self.save_path}")
    
    def _find_nearest_river_cluster(self, equity: float) -> int:
        """Find nearest river cluster for equity value."""
        if self.river_centroids is None:
            raise ValueError("River centroids not loaded")
        distances = [abs(equity - centroid[0]) for centroid in self.river_centroids]
        return int(np.argmin(distances))
    
    def _compute_emd_distance(
        self,
        dist1: List[float],
        dist2: List[float],
        base_matrix: np.ndarray
    ) -> float:
        """Compute EMD distance between two distributions."""
        from pyemd import emd
        return abs(emd(np.array(dist1, dtype=np.float64), np.array(dist2, dtype=np.float64), base_matrix))
