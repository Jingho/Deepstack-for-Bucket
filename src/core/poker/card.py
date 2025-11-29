"""Card representation and conversion utilities."""

from typing import Dict, List, Optional
from enum import IntEnum

from ...config.settings import Config, get_config


class Suit(IntEnum):
    """Card suit enumeration."""
    
    CLUBS = 0
    DIAMONDS = 1
    HEARTS = 2
    SPADES = 3


class Rank(IntEnum):
    """Card rank enumeration."""
    
    TWO = 0
    THREE = 1
    FOUR = 2
    FIVE = 3
    SIX = 4
    SEVEN = 5
    EIGHT = 6
    NINE = 7
    TEN = 8
    JACK = 9
    QUEEN = 10
    KING = 11
    ACE = 12


class CardConverter:
    """
    Convert between card representations (string, index, numeric).
    
    Handles conversion between:
    - String format: "Ac", "Kd", etc.
    - Index format: 1, 2, 3, etc. (1-based)
    - Numeric format: [rank, suit] tuples
    """
    
    # Rank and suit mappings
    RANK_TO_STRING: Dict[int, str] = {
        0: "2", 1: "3", 2: "4", 3: "5", 4: "6", 5: "7", 6: "8",
        7: "9", 8: "T", 9: "J", 10: "Q", 11: "K", 12: "A"
    }
    
    SUIT_TO_STRING: Dict[int, str] = {
        0: "c", 1: "d", 2: "h", 3: "s"
    }
    
    STRING_TO_RANK: Dict[str, int] = {
        "2": 0, "3": 1, "4": 2, "5": 3, "6": 4, "7": 5, "8": 6,
        "9": 7, "T": 8, "t": 8, "J": 9, "j": 9, "Q": 10, "q": 10,
        "K": 11, "k": 11, "A": 12, "a": 12
    }
    
    STRING_TO_SUIT: Dict[str, int] = {
        "c": 0, "C": 0, "d": 1, "D": 1, "h": 2, "H": 2, "s": 3, "S": 3
    }
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize card converter.
        
        Args:
            config: Configuration instance. If None, uses global config.
        """
        self.config = config or get_config()
        self._card_to_string_cache: Dict[int, str] = {}
        self._string_to_card_cache: Dict[str, int] = {}
        self._build_caches()
    
    def _build_caches(self) -> None:
        """Build conversion caches for performance."""
        for card_index in range(1, self.config.card_count + 1):
            rank = self.card_to_rank(card_index)
            suit = self.card_to_suit(card_index)
            card_string = self.RANK_TO_STRING[rank] + self.SUIT_TO_STRING[suit]
            self._card_to_string_cache[card_index] = card_string
            self._string_to_card_cache[card_string] = card_index
    
    def card_to_suit(self, card_index: int) -> int:
        """
        Get suit from card index (1-based).
        
        Args:
            card_index: Card index (1 to card_count)
            
        Returns:
            Suit index (0 to suit_count-1)
        """
        return (card_index - 1) % self.config.suit_count
    
    def card_to_rank(self, card_index: int) -> int:
        """
        Get rank from card index (1-based).
        
        Args:
            card_index: Card index (1 to card_count)
            
        Returns:
            Rank index (0 to rank_count-1)
        """
        return (card_index - 1) // self.config.suit_count
    
    def card_to_string(self, card_index: int) -> str:
        """
        Convert card index to string representation.
        
        Args:
            card_index: Card index (1-based)
            
        Returns:
            Card string (e.g., "Ac", "Kd")
        """
        return self._card_to_string_cache.get(card_index, "")
    
    def string_to_card(self, card_string: str) -> int:
        """
        Convert card string to index.
        
        Args:
            card_string: Card string (e.g., "Ac", "Kd")
            
        Returns:
            Card index (1-based), 0 if invalid
        """
        if not card_string or len(card_string) != 2:
            return 0
        return self._string_to_card_cache.get(card_string, 0)
    
    def cards_to_string(self, card_indices: List[int]) -> str:
        """
        Convert list of card indices to string.
        
        Args:
            card_indices: List of card indices
            
        Returns:
            Concatenated card string
        """
        return "".join(self.card_to_string(idx) for idx in card_indices)
    
    def string_to_board(self, card_string: str) -> List[int]:
        """
        Convert card string to list of card indices.
        
        Args:
            card_string: Card string (e.g., "AcKdQh")
            
        Returns:
            List of card indices
        """
        if not card_string:
            return []
        
        board_count = len(card_string) // 2
        card_indices = []
        
        for i in range(board_count):
            card_str = card_string[i * 2:(i * 2 + 2)]
            card_index = self.string_to_card(card_str)
            if card_index > 0:
                card_indices.append(card_index)
        
        return card_indices
    
    def hand_to_numeric(self, hand_string: str) -> List[List[int]]:
        """
        Convert hand string to numeric format.
        
        Args:
            hand_string: Hand string (e.g., "AcKd")
            
        Returns:
            List of [rank, suit] pairs, sorted by rank descending
        """
        result = []
        for i in range(len(hand_string) // 2):
            rank_char = hand_string[i * 2]
            suit_char = hand_string[i * 2 + 1]
            
            rank = self.STRING_TO_RANK.get(rank_char, -1)
            suit = self.STRING_TO_SUIT.get(suit_char, -1)
            
            if rank >= 0 and suit >= 0:
                result.append([rank, suit])
        
        # Sort by rank descending
        result.sort(reverse=True)
        return result
    
    def generate_all_cards(self) -> List[str]:
        """
        Generate all possible cards based on configuration.
        
        Returns:
            List of card strings
        """
        cards = []
        rank_table = list(self.RANK_TO_STRING.values())[:self.config.rank_count]
        suit_table = list(self.SUIT_TO_STRING.values())[:self.config.suit_count]
        
        for rank in rank_table:
            for suit in suit_table:
                cards.append(rank + suit)
        
        return cards

