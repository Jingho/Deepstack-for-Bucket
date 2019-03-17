# -*- coding:utf-8 -*-

from __future__ import division
from judging import judging
import random
import numpy as np
import os
import argparse
import copy
from card_to_string_conversion import CARD_TO_STRING
from calc_matrix import Cmatrix
from pyemd import emd
from itertools import combinations,product

STREET = 'flop'
FILE_PATH = 'data/'

'''
Traverse all possible hands and generate a hand representation
@param street the round name
@param file_path the relative path for storing data
@class Gen_data
'''

class Gen_data(Cmatrix):
    def __init__(self,
                street=None,
                file_path='data/',
                ):

        super().__init__(street, file_path)
        self.savename = self.get_savename()
        assert street in ['river', 'turn', 'flop'], 'The parameter street is error'

        if street == 'river':
            pass
        elif street == 'turn':
            self.centroids_5 = self.load_data('river_cluster.csv')
        else:
            self.centroids_5 = self.load_data('river_cluster.csv')
            self.centroids_4 = self.load_data('turn_cluster.csv')
            self.matrix = self.get_Euclidean_Matrix(self.centroids_5)

        card_to_string = CARD_TO_STRING()
        card = card_to_string.rank_table[:6]
        flower = card_to_string.suit_table[:2]
        self.all_cards = [[i+j] for i in card for j in flower]

    '''Cluster center point save file name'''
    def get_savename(self):
        savenames = {'river':self.file_path + 'river_data.csv',
                    'turn':  self.file_path + 'turn_data.csv',
                    'flop':  self.file_path + 'flop_data.csv'}
        return savenames.get(self.street)

    '''
    Traverse all possible opponents' hands and calculate our winning percentage in the current state
    @param free_cards possible hand cards pool
    @param hand player's current hand cards
    @param public Current 5 public cards
    return int win rate
    '''

    def win_rate_compute(self, free_cards, hand, public):
        win_rate = [0 for _ in range(3)]
        hand = ''.join(hand)
        public = [p[0] for p in public]
        public = ''.join(public)
        all_opponent = list(combinations(free_cards,2))
        all_opponent = list(map(list,all_opponent))
        n = len(all_opponent)
        for opponent in all_opponent:
            opponent = opponent[0][0] + opponent[1][0]
            win_rate[judging(hand, opponent, public)] += 1./n
        win_rate = win_rate[0] + win_rate[2] / 2.0
        return win_rate

    '''Generate data in river round'''
    def data_generator_5(self):
        f = open(self.savename,'wt')
        all_state = list(combinations(self.all_cards,7))
        all_state = list(map(list,all_state))
        state_count = 0
        state_num = len(all_state)
        for state in all_state:
            print('state {0} / {1}:'.format(state_count,state_num))
            state_count += 1
            all_hand = list(combinations(state,2))
            all_hand = list(map(list,all_hand))
            hand_count = 0
            for hand in all_hand:
                print("--- {0} th hand".format(hand_count))
                hand_count += 1
                hand_card = hand[0] + hand[1]
                public_card = [card for card in state
                                if card not in hand]
                free_cards = [card for card in self.all_cards
                                if card not in state]
                win_rate = self.win_rate_compute(free_cards, hand_card, public_card)
                to_str = str(win_rate)
                f.write(to_str + '\n')
        f.close()

    '''Generate data in turn round'''
    def data_generator_4(self):

        f = open(self.savename,'wt')
        all_state = list(combinations(self.all_cards,6))
        all_state = list(map(list,all_state))
        state_count = 0
        for state in all_state:
            state_num = len(all_state)
            print('state {0} / {1}:'.format(state_count,state_num))
            state_count += 1
            all_hand = list(combinations(state,2))
            all_hand = list(map(list,all_hand))
            hand_count = 0
            for hand in all_hand:
                print("--- {0} th hand".format(hand_count))
                hand_count += 1
                hand_card = hand[0] + hand[1]
                public_card_4 = [card for card in state
                                    if card not in hand]
                free_cards = [card for card in self.all_cards
                                if card not in state]
                n_turn_count = len(free_cards)
                cha = [0 for _ in range(len(self.centroids_5))]
                for public_card_1 in free_cards:
                    public_card = public_card_4 + [public_card_1]
                    all_opponent_card = copy.deepcopy(free_cards)
                    all_opponent_card.remove(public_card_1)
                    win_rate = self.win_rate_compute(all_opponent_card, hand_card, public_card)
                    index = np.argmin(list(map(lambda x: abs(win_rate - x[0]),self.centroids_5)))
                    cha[index] += 1. / n_turn_count
                cha = list(map(str,cha))
                to_str = ','.join(cha)
                f.write(to_str + '\n')
        f.close()

    '''Generate data in flop round'''
    def data_generator_3(self):
        f = open(self.savename,'wt')
        all_state = list(combinations(self.all_cards,5))
        all_state = list(map(list,all_state))
        state_count = 0
        state_num = len(all_state)
        for state in all_state:
            print('state {0} / {1}:'.format(state_count,state_num))
            state_count += 1
            all_hand = list(combinations(state,2))
            all_hand = list(map(list,all_hand))
            hand_count = 0
            for hand in all_hand:
                print("--- {0} th hand".format(hand_count))
                hand_count += 1
                hand_card = hand[0] + hand[1]
                public_card_3 = [card for card in state
                                    if card not in hand]
                free_cards = [card for card in self.all_cards
                                if card not in state]

                n_flod_count = len(free_cards)
                cha_2 = [0] * len(self.centroids_4)
                for public_card_turn in free_cards:
                    public_cards_4 = public_card_3 + [public_card_turn]
                    free_cards_2 = copy.deepcopy(free_cards)
                    free_cards_2.remove(public_card_turn)

                    n_turn_count = len(free_cards_2)
                    cha = [0 for _ in range(len(self.centroids_5))]
                    for public_card_river in free_cards_2:
                        public_card_5 = public_cards_4 + [public_card_river]
                        all_opponent_card = copy.deepcopy(free_cards_2)
                        all_opponent_card.remove(public_card_river)
                        win_rate = self.win_rate_compute(all_opponent_card,hand_card,public_card_5)
                        index = np.argmin(list(map(lambda x: abs(win_rate - x[0]),self.centroids_5)))
                        cha[index] += 1. / n_turn_count

                    distance_list = list(map(lambda x: emd(np.array(cha),np.array(x),self.matrix),self.centroids_4))
                    min_distance_index = np.argmin(distance_list)
                    cha_2[min_distance_index] += 1 / n_flod_count
                to_str = ','.join(list(map(str,cha_2)))
                # print(to_str)
                f.write(to_str + '\n')
        f.close()

    '''Main function for generating data'''
    def generate_data(self):
        if self.street == 'river':
            self.data_generator_5()
        if self.street == 'turn':
            self.data_generator_4()
        if self.street == 'flop':
            self.data_generator_3()

def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--street", type=str, default='river')
    parser.add_argument("--file_path", type=str, default='data/')

    args, _ = parser.parse_known_args()
    return args

if __name__ == '__main__':
    params = vars(get_params())
    data = Gen_data(street=params['street'], file_path=params['file_path'])
    data.generate_data()
