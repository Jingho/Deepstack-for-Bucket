# -*- coding:utf-8 -*-

from __future__ import division
import settings

'''
Conversion hand represented between strings and index
@CARD_TO_STRING
'''

class CARD_TO_STRING():

    def __init__(self):
        self.suit_table = ['h', 's', 'd', 'c']   # All possible cards suit
        self.rank_table = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
        self.card_to_string_table = self.card_to_string_dict()
        self.string_to_card_table = self.string_to_card_dict()

    def card_to_suit(self, card):
        return (card - 1) % settings.suit_count + 1

    def card_to_rank(self, card):
        return (card - 1) // settings.suit_count + 1

    def card_to_string_dict(self):
        card_to_string_dict = dict()
        for card in range(1,settings.card_count + 1):
            rank_name = self.rank_table[self.card_to_rank(card)-1]
            suit_name = self.suit_table[self.card_to_suit(card)-1]
            card_to_string_dict[card] = rank_name + suit_name
        return card_to_string_dict

    '''
    Create a dictionary that converts strings into cards
    @string_to_card_dict
    '''
    def string_to_card_dict(self):
        string_to_card_dict = dict()
        for card in range(1,settings.card_count + 1):
            string_to_card_dict[self.card_to_string_table[card]] = card
        return string_to_card_dict

    '''
    Convert the card into a corresponding suit
    @card_to_string
    '''

    def card_to_string(self, card):
        return self.card_to_string_table[card]

    '''
    Convert the string to the index of the corresponding card
    @string_to_card
    '''

    def string_to_card(self, card_string):
        return self.string_to_card_table[card_string]

    '''
    Convert the card to the corresponding string representation
    @cards_to_string
    '''

    def cards_to_string(self, cards):
        out = " "
        if len(cards) == 0:
            return " "
        for i in range(len(cards)):
            out = out + self.card_to_string(cards[i])
        return out

    '''
    Convert the string to the card combination
    @string_to_board
    '''

    def string_to_board(self, card_string):
        out = []
        if len(card_string) != 0:
            board_count = len(card_string) // 2
            card_count = 0
            while len(out) < board_count:
                tem_card_string = card_string[card_count:(card_count+2)]
                out.append(self.string_to_card(tem_card_string))
                card_count = card_count + 2
        else:
            out = []
        return out
