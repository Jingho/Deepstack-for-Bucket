"""Equity (win rate) calculation for poker hands."""

import random
from typing import List, Optional
from itertools import combinations

from .hand_evaluator import HandEvaluator, HandResult
from ...config.settings import Config, get_config
from .card import CardConverter
from ...config.settings import Config, get_config
from ...utils.logging import get_logger

logger = get_logger(__name__)


class EquityCalculator:
    """
    Calculate equity (win rate) for poker hands.
    
    Equity = win_probability + 0.5 * tie_probability
    """
    
    def __init__(
        self,
        hand_evaluator: Optional[HandEvaluator] = None,
        card_converter: Optional[CardConverter] = None,
        config: Optional[Config] = None
    ):
        """
        Initialize equity calculator.
        
        Args:
            hand_evaluator: Hand evaluator instance
            card_converter: Card converter instance
            config: Configuration instance
        """
        self.config = config or get_config()
        self.hand_evaluator = hand_evaluator or HandEvaluator(card_converter)
        self.card_converter = card_converter or CardConverter(self.config)
    
    def compute_equity(
        self,
        hand_string: str,
        board_string: str,
        available_cards: List[str],
        use_exact: Optional[bool] = None
    ) -> float:
        """
        Compute equity for a given hand and board.
        
        Args:
            hand_string: Player's hand as string (e.g., "AcKd")
            board_string: Board cards as string (e.g., "AhKhQh")
            available_cards: List of available cards for opponent
            use_exact: Use exact enumeration instead of sampling. If None, uses config.
            
        Returns:
            Equity value between 0 and 1
        """
        use_exact = use_exact if use_exact is not None else self.config.use_exact_enumeration
        
        if use_exact:
            return self._compute_equity_exact(hand_string, board_string, available_cards)
        else:
            return self._compute_equity_sampled(hand_string, board_string, available_cards)
    
    def _compute_equity_exact(
        self,
        hand_string: str,
        board_string: str,
        available_cards: List[str]
    ) -> float:
        """
        Compute equity using exact enumeration.
        
        Args:
            hand_string: Player's hand
            board_string: Board cards
            available_cards: Available cards for opponent
            
        Returns:
            Equity value
        """
        win_count = 0.0
        opponent_hands = list(combinations(available_cards, self.config.hold_card_count))
        total_hands = len(opponent_hands)
        
        if total_hands == 0:
            logger.warning("No opponent hands available")
            return 0.0
        
        for opponent_cards in opponent_hands:
            opponent_hand = "".join(opponent_cards)
            result = self.hand_evaluator.compare_hands(
                hand_string,
                opponent_hand,
                board_string
            )
            
            if result == HandResult.HAND1_WINS:
                win_count += 1.0
            elif result == HandResult.TIE:
                win_count += 0.5
        
        return win_count / total_hands
    
    def _compute_equity_sampled(
        self,
        hand_string: str,
        board_string: str,
        available_cards: List[str]
    ) -> float:
        """
        Compute equity using Monte Carlo sampling.
        
        Args:
            hand_string: Player's hand
            board_string: Board cards
            available_cards: Available cards for opponent
            
        Returns:
            Equity value
        """
        win_count = 0.0
        sample_count = self.config.opponent_sample_count
        
        if len(available_cards) < self.config.hold_card_count:
            logger.warning("Not enough cards for opponent")
            return 0.0
        
        for _ in range(sample_count):
            opponent_cards = random.sample(available_cards, self.config.hold_card_count)
            opponent_hand = "".join(opponent_cards)
            
            result = self.hand_evaluator.compare_hands(
                hand_string,
                opponent_hand,
                board_string
            )
            
            if result == HandResult.HAND1_WINS:
                win_count += 1.0
            elif result == HandResult.TIE:
                win_count += 0.5
        
        return win_count / sample_count

