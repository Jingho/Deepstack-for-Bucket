# -*- coding:utf-8 -*-

from judging import judging
import random
import numpy as np
from pyemd import emd
import copy
from card_to_string_conversion import CARD_TO_STRING
import settings
from scipy.special import comb, perm
from itertools import combinations,permutations
import math

STREET = 'turn'

'''
Traverse all the hands and save their bucket index to a txt file
This program is mainly a compromise solution for docking with lua code.
@class CLUSTER_RESULT
'''

class CLUSTER_RESULT():

    def __init__(self, street = None, file_path='data/', turn_sample_count = 7, river_sample_count = 6, opponent_sample_count = 100,
                 comb_flag = False, normalize_flag = False):

        self.card_to_string = CARD_TO_STRING()
        self.cards = self.get_all_cards()
        self.street = street
        self.file_path = file_path
        self.sample_state_count = None
        self.cluster_result_file = None
        self.card_to_cluster_dict = dict()
        self.turn_sample_count = turn_sample_count
        self.river_sample_count = river_sample_count
        self.opponent_sample_count = opponent_sample_count
        self.comb_flag = comb_flag
        self.normalize_flag = normalize_flag

        if self.street in ['river', 'turn', 'flop']:
            self.centroids_river = self.read_cluster_centroids("river")
        if self.street in ['turn', 'flop']:
            self.centroids_turn = self.read_cluster_centroids("turn")
        if self.street == 'flop':
            self.centroids_flop = self.read_cluster_centroids("flop")

    # 读取各圈的聚类中心(列表的形式)
    def read_cluster_centroids(self, street_type):
        centroids = []
        file_name = ""
        if street_type == "river":
            file_name = "river_cluster.csv"
            self.cluster_state_count = 1
        elif street_type == "turn":
            file_name = "turn_cluster.csv"
            self.cluster_state_count = settings.river_cluster_count
        elif street_type == "flop":
            file_name = "flop_cluster.csv"
            self.cluster_state_count = settings.turn_cluster_count
        else:
            print("The file name is None")
        file_name = self.file_path + file_name

        with open(file_name) as file:
            for line in file:
                string_line = line.split(",")
                centroid = []
                for i in range(self.cluster_state_count):
                    line_ = float(string_line[i])
                    centroid.append(line_)
                centroids.append(centroid)
        return centroids

    # 生成所有牌的组合
    def get_all_cards(self):
        cards = []
        card = self.card_to_string.rank_table[:settings.rank_count]
        flower = self.card_to_string.suit_table[:settings.suit_count]
        # 生成所有牌
        for i in card:
            for j in flower:
                cards.append(i + j)
        return cards

    def string_to_list(self,card):
        assert len(card) % 2 == 0, "the number of string card is wrong"
        card_count = len(card) // 2
        card_list = []
        iter_count = 0
        card_string_start = 0
        while iter_count < card_count:
            card_list.append(card[card_string_start:(card_string_start+2)])
            card_string_start +=2
            iter_count += 1
        assert len(card_list) == card_count, 'the number of card is wrong'
        return card_list

    # 计算河牌圈牌的类别，输入手牌与公共牌(列表的形式)，输出河牌圈所属的类别
    def _computer_river_card_category(self, hand, board):
        out_win_rate = 0
        used_card_list = hand + board
        available_card = copy.deepcopy(self.cards)
        hand_card = self.list_to_string(hand)
        board_card = self.list_to_string(board)

        for card in used_card_list:
            available_card.remove(card)

        for i in range(self.opponent_sample_count):
            opponent_card = random.sample(available_card,2)
            opponent = opponent_card[0] + opponent_card[1]
            if judging(hand_card, opponent, board_card) == 0:
                out_win_rate += 1
        out_win_rate = out_win_rate / self.opponent_sample_count
        min_distance_index = 0
        min_distance = 10000
        for i in range(len(self.centroids_river)):
            distance = math.pow(self.centroids_river[i][0]-out_win_rate, 2)
            # distance = emd(np.array(win_rate), np.array(self.centroids_river[i]), matrix)
            if distance < min_distance:
                # print(i)
                min_distance_index = i
                min_distance = distance
        return min_distance_index + 1

    def _computer_river_card_category_comb(self, hand, board, state):
        out_win_rate = 0
        available_card = copy.deepcopy(self.cards)

        for card in state:
            available_card.remove(card)

        opponent_cards = list(combinations(available_card,2))
        for card in opponent_cards:
            opponent_card = list(card)
            opponent = opponent_card[0] + opponent_card[1]
            if judging(hand, opponent, board) == 0:
                out_win_rate += 1
        out_win_rate = out_win_rate / len(opponent_cards)
        min_distance_index = 0
        min_distance = 10000
        for i in range(len(self.centroids_river)):
            distance = math.pow(self.centroids_river[i][0]-out_win_rate, 2)
            # distance = emd(np.array(win_rate), np.array(self.centroids_river[i]), matrix)
            if distance < min_distance:
                # print(i)
                min_distance_index = i
                min_distance = distance
        return min_distance_index + 1

    def computer_distance_matrix(self, street):

        if street == 'turn':
            matrix = np.zeros([settings.river_cluster_count, settings.river_cluster_count])
            for i in range(settings.river_cluster_count):
                for j in range(settings.river_cluster_count):
                    matrix[i][j] = np.abs(self.centroids_river[i][0]-self.centroids_river[j][0])
        elif street == 'flop':
            matrix = np.zeros([settings.turn_cluster_count, settings.turn_cluster_count])
            matrix_turn = np.zeros([settings.river_cluster_count, settings.river_cluster_count])
            for i in range(settings.river_cluster_count):
                for j in range(settings.river_cluster_count):
                    matrix_turn[i][j] = np.abs(self.centroids_river[i][0]-self.centroids_river[j][0])
            for i in range(settings.turn_cluster_count):
                for j in range(settings.turn_cluster_count):
                    matrix[i][j] = emd(np.array(self.centroids_turn[i]),np.array(self.centroids_turn[j]), matrix_turn)
        else:
            pass
            matrix = np.array([[0,1/3.0,2/3.0],[1/3.0,0,1/3.0],[2/3.0,1/3.0,0]])
        return matrix

    # 计算转牌圈牌的类别，输入手牌与公共牌(列表的形式)，输出转牌圈所属的类别
    def _computer_turn_card_category(self, hand, board):

        used_card_list = hand + board
        available_card = copy.deepcopy(self.cards)
        hand_card = self.list_to_string(hand)
        for card in used_card_list:
            available_card.remove(card)

        cha = [float(0)] * len(self.centroids_river)
        # matrix_river = np.array([[0, 1 / 3.0, 2 / 3.0], [1 / 3.0, 0, 1 / 3.0], [2 / 3.0, 1 / 3.0, 0]])
        # matrix_river = self.computer_distance_matrix('river')
        for i in range(self.river_sample_count):
            river_board = copy.deepcopy(board)
            river_card = random.sample(available_card, 1)
            river_board.append(river_card[0])
            opponent_available_card = copy.deepcopy(available_card)
            opponent_available_card.remove(river_card[0])
            board_card = self.list_to_string(river_board)
            assert len(board_card) // 2 == 5, "the number of the river board is not 5"
            win_rate = 0
            for _ in range(self.opponent_sample_count):
                opponent_card = random.sample(opponent_available_card, 2)
                opponent = opponent_card[0] + opponent_card[1]
                # win_rate[judging(hand_card, opponent, board_card)] += 1 / self.opponent_sample_count
                if judging(hand_card, opponent, board_card) == 0:
                    win_rate += 1
            win_rate = win_rate / self.opponent_sample_count

            river_min_distance_index = 0
            river_min_distance = 10000
            for j in range(len(self.centroids_river)):
                # distance = emd(np.array(win_rate), np.array(self.centroids_river[j]), matrix_river)
                distance = math.pow(win_rate-self.centroids_river[j][0], 2)
                if distance < river_min_distance:
                    # print(i)
                    river_min_distance_index = j
                    river_min_distance = distance
            cha[river_min_distance_index] += 1
            # print(cha)
        if self.normalize_flag:
            sum_cha = sum(cha)
            for i in range(len(cha)):
                cha[i] = cha[i] / sum_cha

        matrix = self.computer_distance_matrix('turn')
        min_distance_index = 0
        min_distance = 10000
        for i in range(len(self.centroids_turn)):
            distance = emd(np.array(cha), np.array(self.centroids_turn[i]), matrix)
            if distance < min_distance:
                # print(i)
                min_distance_index = i
                min_distance = distance
        return min_distance_index + 1

    def _computer_turn_card_category_comb(self, hand, board, state):

        available_card = copy.deepcopy(self.cards)
        for card in state:
            available_card.remove(card)

        cha = [float(0)] * len(self.centroids_river)
        for river_card in available_card:
            river_board = board + river_card
            opponent_available_card = copy.deepcopy(available_card)
            opponent_available_card.remove(river_card)
            assert len(river_board) // 2 == 5, "the number of the river board is not 5"
            win_rate = 0

            opponent_cards = list(combinations(opponent_available_card,2))
            for card in opponent_cards:
                opponent_card = list(card)
                opponent = opponent_card[0] + opponent_card[1]
                if judging(hand, opponent, river_board) == 0:
                    win_rate += 1
                if judging(hand, opponent, river_board) == 2:
                    win_rate += 1 / 2
            win_rate = win_rate / len(opponent_cards)

            river_min_distance_index = 0
            river_min_distance = 10000
            for j in range(len(self.centroids_river)):
                distance = math.pow(win_rate - self.centroids_river[j][0], 2)
                if distance < river_min_distance:
                    river_min_distance_index = j
                    river_min_distance = distance
            cha[river_min_distance_index] += 1
        if self.normalize_flag:
            sum_cha = sum(cha)
            for i in range(len(cha)):
                cha[i] = cha[i] / sum_cha

        matrix = self.computer_distance_matrix('turn')
        min_distance_index = 0
        min_distance = 10000
        for i in range(len(self.centroids_turn)):
            distance = emd(np.array(cha), np.array(self.centroids_turn[i]), matrix)
            if distance < min_distance:
                # print(i)
                min_distance_index = i
                min_distance = distance
        return min_distance_index + 1

    # 计算翻牌圈牌的类别，输入手牌与公共牌(列表的形式)，输出翻牌圈所属的类别
    def _computer_flop_card_category(self, hand, board):

        used_card_list = hand + board
        available_card = copy.deepcopy(self.cards)
        hand_card = self.list_to_string(hand)

        for card in used_card_list:
            available_card.remove(card)
        cha_2 = [float(0)] * len(self.centroids_turn)
        matrix_turn = self.computer_distance_matrix('turn')
        for i in range(self.turn_sample_count):
            turn_board = copy.deepcopy(board)
            turn_card = random.sample(available_card,1)
            turn_board.append(turn_card[0])
            river_available_card = copy.deepcopy(available_card)
            river_available_card.remove(turn_card[0])
            assert len(river_available_card) == 6, "The number of river available card is not 6"
            assert len(turn_board) == 4, "The number of turn board is not 4"
            cha = [float(0)] * len(self.centroids_river)
            # matrix_river = np.array([[0, 1 / 3.0, 2 / 3.0], [1 / 3.0, 0, 1 / 3.0], [2 / 3.0, 1 / 3.0, 0]])
            for j in range(self.river_sample_count):
                river_board = copy.deepcopy(turn_board)
                river_card = random.sample(river_available_card, 1)
                river_board.append(river_card[0])
                opponent_available_card = copy.deepcopy(river_available_card)
                opponent_available_card.remove(river_card[0])
                board_card = self.list_to_string(river_board)
                assert len(opponent_available_card) == 5, "The number of opponent available card is not 5"
                assert len(board_card) // 2 == 5, "the number of the river board is not 5"
                win_rate = 0
                for _ in range(self.opponent_sample_count):
                    opponent_card = random.sample(opponent_available_card, 2)
                    opponent = opponent_card[0] + opponent_card[1]
                    if judging(hand_card, opponent, board_card) == 0:
                        win_rate += 1
                win_rate = win_rate / self.opponent_sample_count
                river_min_distance_index = 0
                river_min_distance = 10000
                for index in range(len(self.centroids_river)):
                    # distance = emd(np.array(win_rate), np.array(self.centroids_river[index]), matrix_river)
                    distance = math.pow(win_rate - self.centroids_river[index][0], 2)
                    if distance < river_min_distance:
                        river_min_distance_index = index
                        river_min_distance = distance
                cha[river_min_distance_index] += 1

            turn_min_distance_index = 0
            turn_min_distance = 10000
            for turn_index in range(len(self.centroids_turn)):
                distance = emd(np.array(cha), np.array(self.centroids_turn[turn_index]), matrix_turn)
                if distance < turn_min_distance:
                    # print(i)
                    turn_min_distance_index = turn_index
                    turn_min_distance = distance
            cha_2[turn_min_distance_index] += 1

        matrix = self.computer_distance_matrix('flop')
        min_distance_index = 0
        min_distance = 10000
        for i in range(len(self.centroids_flop)):
            distance = emd(np.array(cha_2), np.array(self.centroids_flop[i]), matrix)
            if distance < min_distance:
                # print(i)
                min_distance_index = i
                min_distance = distance
        return min_distance_index + 1

    def _computer_flop_card_category_comb(self, hand, board, state):

        available_card = copy.deepcopy(self.cards)
        for card in state:
            available_card.remove(card)
        cha_2 = [float(0)] * len(self.centroids_turn)
        matrix_turn = self.computer_distance_matrix('turn')

        for turn_card in available_card:
            turn_board = board + turn_card
            river_available_card = copy.deepcopy(available_card)
            river_available_card.remove(turn_card)
            assert len(river_available_card) == 6, "The number of river available card is not 6"
            assert len(turn_board) // 2 == 4, "The number of turn board is not 4"
            cha = [float(0)] * len(self.centroids_river)

            for river_card in river_available_card:
                river_board = turn_board + river_card
                opponent_available_card = copy.deepcopy(river_available_card)
                opponent_available_card.remove(river_card)
                assert len(opponent_available_card) == 5, "The number of opponent available card is not 5"
                assert len(river_board) // 2 == 5, "the number of the river board is not 5"
                win_rate = 0

                opponent_cards = list(combinations(opponent_available_card, 2))
                for card in opponent_cards:
                    opponent_card = list(card)
                    opponent = opponent_card[0] + opponent_card[1]
                    if judging(hand, opponent, river_board) == 0:
                        win_rate += 1
                    if judging(hand, opponent, river_board) == 2:
                        win_rate += 1 / 2
                win_rate = win_rate / len(opponent_cards)
                river_min_distance_index = 0
                river_min_distance = 10000
                for index in range(len(self.centroids_river)):
                    # distance = emd(np.array(win_rate), np.array(self.centroids_river[index]), matrix_river)
                    distance = math.pow(win_rate - self.centroids_river[index][0], 2)
                    if distance < river_min_distance:
                        river_min_distance_index = index
                        river_min_distance = distance
                cha[river_min_distance_index] += 1
            if self.normalize_flag:
                sum_cha = sum(cha)
                for i in range(len(cha)):
                    cha[i] = cha[i] / sum_cha

            turn_min_distance_index = 0
            turn_min_distance = 10000
            for turn_index in range(len(self.centroids_turn)):
                distance = emd(np.array(cha), np.array(self.centroids_turn[turn_index]), matrix_turn)
                if distance < turn_min_distance:
                    # print(i)
                    turn_min_distance_index = turn_index
                    turn_min_distance = distance
            cha_2[turn_min_distance_index] += 1
        if self.normalize_flag:
            sum_cha2 = sum(cha_2)
            for i in range(len(cha_2)):
                cha_2[i] = cha_2[i] / sum_cha2
        matrix = self.computer_distance_matrix('flop')
        min_distance_index = 0
        min_distance = 10000
        for i in range(len(self.centroids_flop)):
            distance = emd(np.array(cha_2), np.array(self.centroids_flop[i]), matrix)
            if distance < min_distance:
                # print(i)
                min_distance_index = i
                min_distance = distance
        return min_distance_index + 1

    def list_to_string(self,list1):
        string1 = ''
        for i in range(len(list1)):
            string1 = string1 + str(list1[i])
        return string1

    def computer_cluster_result(self):

        if self.street == "river":
            file_name = self.file_path + "river_cluster_result_k" + str(settings.river_cluster_count) + ".txt"
            self.cluster_result_file = open(file_name, "w")
            self.sample_state_count = 7
        elif self.street == "turn":
            file_name = self.file_path + "turn_cluster_result_k" + str(settings.turn_cluster_count) + ".txt"
            self.cluster_result_file = open(file_name, "w")
            self.sample_state_count = 6
        elif self.street == "flop":
            file_name = self.file_path + "flop_cluster_result_k" + str(settings.flop_cluster_count) + ".txt"
            self.cluster_result_file = open(file_name, "w")
            self.sample_state_count = 5
        else:
            pass

        all_state = list(combinations(self.cards, self.sample_state_count))
        state_count = 0
        for state in all_state:
            state = list(state)
            print('state {0}:'.format(state_count), self.list_to_string(state))
            possible_hand_comb = list(combinations(state, settings.hold_card_count))
            hand_count = 0
            for hand in possible_hand_comb:
                hand = list(hand)
                print("--- {0} th hand".format(hand_count), self.list_to_string(hand))
                hand_card = hand[0] + hand[1]
                board_card = ""
                board = copy.deepcopy(state)
                for card in hand:
                    board.remove(card)
                for i in range(len(board)):
                    board_card = board_card + board[i]
                # board_card = board[0] + board[1] + board[2] + board[3] + board[4]

                if self.street == "river":
                    if self.comb_flag:
                        cluster_index = self._computer_river_card_category_comb(hand_card,board_card,state)
                    else:
                        cluster_index = self._computer_river_card_category(hand,board)
                elif self.street == "turn":
                    if self.comb_flag:
                        cluster_index = self._computer_turn_card_category_comb(hand_card,board_card,state)
                    else:
                        cluster_index = self._computer_turn_card_category(hand,board)
                elif self.street == "flop":
                    if self.comb_flag:
                        cluster_index = self._computer_flop_card_category_comb(hand_card,board_card,state)
                    else:
                        cluster_index = self._computer_flop_card_category(hand,board)
                else:
                    cluster_index = 0

                assert cluster_index > 0, "The cluster index is wrong"

                # print("--- {0} cluster".format(cluster_index))
                hand_card_index = self.card_to_string.string_to_board(hand_card)
                board_card_index = self.card_to_string.string_to_board(board_card)
                hand_card_index.sort()
                board_card_index.sort()
                hand_card_index_string = self.list_to_string(hand_card_index)
                board_card_index_string = self.list_to_string(board_card_index)
                card_index_string = hand_card_index_string + board_card_index_string
                self.cluster_result_file.write(card_index_string + ":" + str(cluster_index) + "\n")
                self.card_to_cluster_dict[card_index_string] = cluster_index
                hand_count += 1
            state_count += 1

        assert len(self.card_to_cluster_dict) == comb(12,self.sample_state_count) * comb(self.sample_state_count, \
                    settings.hold_card_count), "the number of possible card combination is wrong"
        print("the result len: {0}".format(len(self.card_to_cluster_dict)))
        print(str(comb(12,self.sample_state_count)) + "*" + str(comb(self.sample_state_count,settings.hold_card_count)))
        self.cluster_result_file.close()

if __name__ == "__main__":

    cluster_result = CLUSTER_RESULT(street=STREET, turn_sample_count=20,river_sample_count=10, opponent_sample_count=20,
                                    comb_flag=True, normalize_flag=True)
    cluster_result.computer_cluster_result()
