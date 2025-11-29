"""Unit tests for poker module."""

import pytest
from src.core.poker.card import CardConverter
from src.core.poker.hand_evaluator import HandEvaluator, HandResult
from src.core.poker.equity_calculator import EquityCalculator
from src.config.settings import Config
from src.config.constants import PokerRound


class TestCardConverter:
    """Test card conversion utilities."""
    
    def test_card_to_string(self):
        """Test card index to string conversion."""
        converter = CardConverter()
        # Test first card (should be "Ac" or similar depending on config)
        card_string = converter.card_to_string(1)
        assert len(card_string) == 2
        assert card_string[0].isalpha()
        assert card_string[1].islower()
    
    def test_string_to_card(self):
        """Test string to card index conversion."""
        converter = CardConverter()
        card_string = converter.card_to_string(1)
        card_index = converter.string_to_card(card_string)
        assert card_index == 1
    
    def test_generate_all_cards(self):
        """Test generation of all cards."""
        converter = CardConverter()
        cards = converter.generate_all_cards()
        config = Config.default()
        expected_count = config.card_count
        assert len(cards) == expected_count
        assert all(len(card) == 2 for card in cards)


class TestHandEvaluator:
    """Test hand evaluation."""
    
    def test_compare_hands(self):
        """Test hand comparison."""
        evaluator = HandEvaluator()
        
        # Test case: AA vs KK on AAK board
        result = evaluator.compare_hands("AcAd", "KcKd", "AhKsQh")
        assert result == HandResult.HAND1_WINS
    
    def test_best_five(self):
        """Test best five card selection."""
        evaluator = HandEvaluator()
        hand = evaluator.card_converter.hand_to_numeric("AcKd")
        community = evaluator.card_converter.hand_to_numeric("AhKhQh")
        best = evaluator._best_five(hand, community)
        assert len(best) == 5


class TestEquityCalculator:
    """Test equity calculation."""
    
    def test_compute_equity_exact(self):
        """Test exact equity computation."""
        config = Config.default()
        calculator = EquityCalculator(config=config)
        
        # Simple test case
        equity = calculator.compute_equity(
            "AcAd",
            "AhKhQh",
            ["2c", "2d", "3c", "3d"],
            use_exact=True
        )
        assert 0.0 <= equity <= 1.0

