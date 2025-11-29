"""Texas Hold'em hand evaluation and comparison."""

from typing import List, Tuple, Optional
from enum import IntEnum
from itertools import combinations

from .card import CardConverter


class HandResult(IntEnum):
    """Hand comparison result."""
    
    HAND1_WINS = 0
    HAND2_WINS = 1
    TIE = 2


class HandEvaluator:
    """
    Evaluate and compare Texas Hold'em poker hands.
    
    Supports standard poker hand rankings:
    - Straight Flush
    - Four of a Kind
    - Full House
    - Flush
    - Straight
    - Three of a Kind
    - Two Pair
    - One Pair
    - High Card
    """
    
    def __init__(self, card_converter: Optional[CardConverter] = None):
        """
        Initialize hand evaluator.
        
        Args:
            card_converter: Card converter instance. If None, creates new one.
        """
        self.card_converter = card_converter or CardConverter()
    
    def compare_hands(
        self,
        hand1_string: str,
        hand2_string: str,
        community_string: str
    ) -> HandResult:
        """
        Compare two hands given community cards.
        
        Args:
            hand1_string: First hand as string (e.g., "AcKd")
            hand2_string: Second hand as string (e.g., "QhJs")
            community_string: Community cards as string (e.g., "AhKhQh")
            
        Returns:
            HandResult indicating which hand wins
        """
        hand1_numeric = self.card_converter.hand_to_numeric(hand1_string)
        hand2_numeric = self.card_converter.hand_to_numeric(hand2_string)
        community_numeric = self.card_converter.hand_to_numeric(community_string)
        
        best_hand1 = self._best_five(hand1_numeric, community_numeric)
        best_hand2 = self._best_five(hand2_numeric, community_numeric)
        
        return self._compare_hands(best_hand1, best_hand2)
    
    def _best_five(
        self,
        hand: List[List[int]],
        community: List[List[int]]
    ) -> List[List[int]]:
        """
        Find the best five-card hand from player hand and community cards.
        
        Args:
            hand: Player's hand as [[rank, suit], ...]
            community: Community cards as [[rank, suit], ...]
            
        Returns:
            Best five-card hand
        """
        # Combine all cards
        all_cards = hand + community
        
        # If we have exactly 5 cards, return them sorted
        if len(all_cards) == 5:
            result = self._copy_hand(all_cards)
            result.sort(reverse=True)
            return result
        
        # Try all combinations of 5 cards from all available cards
        current_best = None
        
        for combo in combinations(all_cards, 5):
            test_hand = list(combo)
            test_hand.sort(reverse=True)
            
            if current_best is None:
                current_best = test_hand
            else:
                # Compare with current best
                comparison = self._compare_hands(current_best, test_hand)
                if comparison == HandResult.HAND2_WINS:
                    current_best = test_hand
        
        return current_best if current_best else self._copy_hand(community)
    
    def _compare_hands(
        self,
        hand1: List[List[int]],
        hand2: List[List[int]]
    ) -> HandResult:
        """
        Compare two five-card hands.
        
        Args:
            hand1: First hand as [[rank, suit], ...]
            hand2: Second hand as [[rank, suit], ...]
            
        Returns:
            HandResult indicating which hand wins
        """
        # Check straight flush
        h1_sf = self._check_straight_flush(hand1)
        h2_sf = self._check_straight_flush(hand2)
        if h1_sf and h2_sf:
            return self._compare_straights(hand1, hand2)
        if h1_sf:
            return HandResult.HAND1_WINS
        if h2_sf:
            return HandResult.HAND2_WINS
        
        # Check four of a kind
        h1_4k = self._check_four_of_a_kind(hand1)
        h2_4k = self._check_four_of_a_kind(hand2)
        if h1_4k[0] and h2_4k[0]:
            return self._compare_four_of_a_kind(h1_4k, h2_4k)
        if h1_4k[0]:
            return HandResult.HAND1_WINS
        if h2_4k[0]:
            return HandResult.HAND2_WINS
        
        # Check full house
        h1_fh = self._check_full_house(hand1)
        h2_fh = self._check_full_house(hand2)
        if h1_fh[0] and h2_fh[0]:
            return self._compare_full_house(h1_fh, h2_fh)
        if h1_fh[0]:
            return HandResult.HAND1_WINS
        if h2_fh[0]:
            return HandResult.HAND2_WINS
        
        # Check flush
        h1_flush = self._check_flush(hand1)
        h2_flush = self._check_flush(hand2)
        if h1_flush and h2_flush:
            return self._compare_high_cards(hand1, hand2)
        if h1_flush:
            return HandResult.HAND1_WINS
        if h2_flush:
            return HandResult.HAND2_WINS
        
        # Check straight
        h1_straight = self._check_straight(hand1)
        h2_straight = self._check_straight(hand2)
        if h1_straight and h2_straight:
            return self._compare_straights(hand1, hand2)
        if h1_straight:
            return HandResult.HAND1_WINS
        if h2_straight:
            return HandResult.HAND2_WINS
        
        # Check three of a kind
        h1_3k = self._check_three_of_a_kind(hand1)
        h2_3k = self._check_three_of_a_kind(hand2)
        if h1_3k[0] and h2_3k[0]:
            return self._compare_three_of_a_kind(h1_3k, h2_3k)
        if h1_3k[0]:
            return HandResult.HAND1_WINS
        if h2_3k[0]:
            return HandResult.HAND2_WINS
        
        # Check two pair
        h1_2p = self._check_two_pair(hand1)
        h2_2p = self._check_two_pair(hand2)
        if h1_2p[0] and h2_2p[0]:
            return self._compare_two_pair(h1_2p, h2_2p)
        if h1_2p[0]:
            return HandResult.HAND1_WINS
        if h2_2p[0]:
            return HandResult.HAND2_WINS
        
        # Check one pair
        h1_1p = self._check_one_pair(hand1)
        h2_1p = self._check_one_pair(hand2)
        if h1_1p[0] and h2_1p[0]:
            return self._compare_one_pair(h1_1p, h2_1p)
        if h1_1p[0]:
            return HandResult.HAND1_WINS
        if h2_1p[0]:
            return HandResult.HAND2_WINS
        
        # Compare high cards
        return self._compare_high_cards(hand1, hand2)
    
    def _copy_hand(self, hand: List[List[int]]) -> List[List[int]]:
        """Copy hand (faster than deepcopy)."""
        return [[card[0], card[1]] for card in hand]
    
    def _check_flush(self, hand: List[List[int]]) -> bool:
        """Check if hand is a flush."""
        suits = [card[1] for card in hand]
        return any(suits.count(suit) == 5 for suit in range(4))
    
    def _check_straight(self, hand: List[List[int]]) -> bool:
        """Check if hand is a straight."""
        ranks = sorted([card[0] for card in hand], reverse=True)
        
        # Regular straight
        if ranks == list(range(ranks[0], ranks[0] - 5, -1)):
            return True
        
        # Ace-low straight (A-2-3-4-5)
        if ranks == [12, 3, 2, 1, 0]:
            return True
        
        return False
    
    def _check_straight_flush(self, hand: List[List[int]]) -> bool:
        """Check if hand is a straight flush."""
        return self._check_flush(hand) and self._check_straight(hand)
    
    def _check_four_of_a_kind(self, hand: List[List[int]]) -> Tuple[bool, int, int]:
        """Check for four of a kind. Returns (is_4k, quad_rank, kicker_rank)."""
        ranks = [card[0] for card in hand]
        for rank in range(13):
            if ranks.count(rank) == 4:
                kicker = next(r for r in ranks if r != rank)
                return True, rank, kicker
        return False, 13, 13
    
    def _check_full_house(self, hand: List[List[int]]) -> Tuple[bool, int, int]:
        """Check for full house. Returns (is_fh, trip_rank, pair_rank)."""
        ranks = [card[0] for card in hand]
        for trip_rank in range(13):
            if ranks.count(trip_rank) == 3:
                for pair_rank in range(13):
                    if pair_rank != trip_rank and ranks.count(pair_rank) == 2:
                        return True, trip_rank, pair_rank
        return False, 13, 13
    
    def _check_three_of_a_kind(self, hand: List[List[int]]) -> Tuple[bool, int, List[int]]:
        """Check for three of a kind. Returns (is_3k, trip_rank, [kicker1, kicker2])."""
        ranks = [card[0] for card in hand]
        for trip_rank in range(13):
            if ranks.count(trip_rank) == 3:
                kickers = sorted([r for r in ranks if r != trip_rank], reverse=True)
                return True, trip_rank, kickers
        return False, 13, [13, 13]
    
    def _check_two_pair(self, hand: List[List[int]]) -> Tuple[bool, List[int], int]:
        """Check for two pair. Returns (is_2p, [high_pair, low_pair], kicker)."""
        ranks = [card[0] for card in hand]
        pairs = []
        for rank in range(13):
            if ranks.count(rank) == 2:
                pairs.append(rank)
        
        if len(pairs) == 2:
            pairs.sort(reverse=True)
            kicker = next(r for r in ranks if r not in pairs)
            return True, pairs, kicker
        return False, [13, 13], 13
    
    def _check_one_pair(self, hand: List[List[int]]) -> Tuple[bool, int, List[int]]:
        """Check for one pair. Returns (is_1p, pair_rank, [kicker1, kicker2, kicker3])."""
        ranks = [card[0] for card in hand]
        for pair_rank in range(13):
            if ranks.count(pair_rank) == 2:
                kickers = sorted([r for r in ranks if r != pair_rank], reverse=True)
                return True, pair_rank, kickers
        return False, 13, [13, 13, 13]
    
    def _compare_straights(
        self,
        hand1: List[List[int]],
        hand2: List[List[int]]
    ) -> HandResult:
        """Compare two straight hands."""
        # Compare second card first (for ace-low straights)
        if hand1[1][0] > hand2[1][0]:
            return HandResult.HAND1_WINS
        if hand1[1][0] < hand2[1][0]:
            return HandResult.HAND2_WINS
        # If equal, check first card (ace-low has ace first)
        if hand1[0][0] > hand2[0][0]:
            return HandResult.HAND2_WINS  # Ace-low is lower
        if hand1[0][0] < hand2[0][0]:
            return HandResult.HAND1_WINS
        return HandResult.TIE
    
    def _compare_four_of_a_kind(
        self,
        h1: Tuple[bool, int, int],
        h2: Tuple[bool, int, int]
    ) -> HandResult:
        """Compare two four-of-a-kind hands."""
        if h1[1] > h2[1]:
            return HandResult.HAND1_WINS
        if h1[1] < h2[1]:
            return HandResult.HAND2_WINS
        if h1[2] > h2[2]:
            return HandResult.HAND1_WINS
        if h1[2] < h2[2]:
            return HandResult.HAND2_WINS
        return HandResult.TIE
    
    def _compare_full_house(
        self,
        h1: Tuple[bool, int, int],
        h2: Tuple[bool, int, int]
    ) -> HandResult:
        """Compare two full house hands."""
        if h1[1] > h2[1]:
            return HandResult.HAND1_WINS
        if h1[1] < h2[1]:
            return HandResult.HAND2_WINS
        if h1[2] > h2[2]:
            return HandResult.HAND1_WINS
        if h1[2] < h2[2]:
            return HandResult.HAND2_WINS
        return HandResult.TIE
    
    def _compare_three_of_a_kind(
        self,
        h1: Tuple[bool, int, List[int]],
        h2: Tuple[bool, int, List[int]]
    ) -> HandResult:
        """Compare two three-of-a-kind hands."""
        if h1[1] > h2[1]:
            return HandResult.HAND1_WINS
        if h1[1] < h2[1]:
            return HandResult.HAND2_WINS
        for k1, k2 in zip(h1[2], h2[2]):
            if k1 > k2:
                return HandResult.HAND1_WINS
            if k1 < k2:
                return HandResult.HAND2_WINS
        return HandResult.TIE
    
    def _compare_two_pair(
        self,
        h1: Tuple[bool, List[int], int],
        h2: Tuple[bool, List[int], int]
    ) -> HandResult:
        """Compare two two-pair hands."""
        for p1, p2 in zip(h1[1], h2[1]):
            if p1 > p2:
                return HandResult.HAND1_WINS
            if p1 < p2:
                return HandResult.HAND2_WINS
        if h1[2] > h2[2]:
            return HandResult.HAND1_WINS
        if h1[2] < h2[2]:
            return HandResult.HAND2_WINS
        return HandResult.TIE
    
    def _compare_one_pair(
        self,
        h1: Tuple[bool, int, List[int]],
        h2: Tuple[bool, int, List[int]]
    ) -> HandResult:
        """Compare two one-pair hands."""
        if h1[1] > h2[1]:
            return HandResult.HAND1_WINS
        if h1[1] < h2[1]:
            return HandResult.HAND2_WINS
        for k1, k2 in zip(h1[2], h2[2]):
            if k1 > k2:
                return HandResult.HAND1_WINS
            if k1 < k2:
                return HandResult.HAND2_WINS
        return HandResult.TIE
    
    def _compare_high_cards(
        self,
        hand1: List[List[int]],
        hand2: List[List[int]]
    ) -> HandResult:
        """Compare high cards."""
        ranks1 = [card[0] for card in hand1]
        ranks2 = [card[0] for card in hand2]
        
        for r1, r2 in zip(ranks1, ranks2):
            if r1 > r2:
                return HandResult.HAND1_WINS
            if r1 < r2:
                return HandResult.HAND2_WINS
        
        return HandResult.TIE

