# -*- coding:utf-8 -*-

import numpy as np
from pyemd import emd
import copy

'''
Finding the 5 smallest EMD clusters in the cluster center point and save them in the txt file.
'''

street = 'flop'
river_centroids_path = 'data/centroids_5.csv'
turn_centroids_path = 'data/centroids_4.csv'
txt_save_path = "data/flop_cluster_anlysis.txt"
k_list = list(range(80, 81, 5))
data_name_list = ['data/centroids_3_k{0:0>2}.csv'.format(k) for k in k_list]

def distance_matrix(matrix_,centroids_matrix):
    matrix_, centroids_matrix = np.array(matrix_), np.array(centroids_matrix)
    matrix = np.zeros(shape=(len(centroids_matrix),len(centroids_matrix)))
    for i,mi in enumerate(centroids_matrix):
        for j,mj in enumerate(centroids_matrix):
            if i == j:
                matrix[i][j] = 0
            else:
                matrix[i][j] = abs(emd(np.array(mi), np.array(mj),matrix_))
    return matrix

def get_EHSmatrix(centroids):

    matrix = np.zeros((len(centroids),len(centroids)))
    for i,ic in enumerate(centroids):
        for j,jc in enumerate(centroids):
            matrix[i,j] = abs(ic[0] - jc[0])
    return matrix

def get_EMDmatrix(street):
    if street == 'turn':
        river_centroids = load_data(river_centroids_path)
        return get_EHSmatrix(river_centroids)
    if street == 'flop':
        river_centroids = load_data(river_centroids_path)
        turn_centroids = load_data(turn_centroids_path)
        river_centroids_matrix = get_EHSmatrix(river_centroids)
        turn_centroids_matrix = distance_matrix(river_centroids_matrix, turn_centroids)
        return turn_centroids_matrix

def load_data(filename):
    data_path = filename
    datas = []
    with open(data_path) as file:
        for line in file:
            data = []
            string_line = line.split("\n")[0].split(",")
            string_line.pop() if string_line[-1] == '' else None
            # print(string_line)
            data = [float(strline) for _,strline in enumerate(string_line)]
            datas.append(data)
    return datas

def coord(num,n):
    cx = int(num / n)
    cy = int(num % n)
    return cx, cy

def find_bottom_5(mat):
    mat = np.array(mat)
    x, y = np.shape(mat)
    list_mat = np.reshape(mat, (x * y))
    min_mat = np.argsort(list_mat)
    min_mat = np.reshape(min_mat, (x, y))
    min_num_list = [min_mat[1][0]]
    loop_list = np.reshape(min_mat[1:], [(x-1)*y])
    for i in loop_list[1:]:
        if len(min_num_list) < 10:
            min_num_list.append(i)
    min_coord_list = list(map(lambda a: coord(a, x), min_num_list))
    return min_coord_list

def transform(coord, data, mat):
    num = 1
    lists = []
    for i, j in coord:
        cent1, cent2 = data[i], data[j]
        dist = round(mat[i][j], 17)
        # lists.append([num, dist, coord, [cent1, cent2]])
        lists.append([num, dist, coord[num-1]])
        num += 1
    return lists

def remove_duplicate_element(list):
    list = copy.deepcopy(list)
    element_1 = list[0]
    for i in list:
        if all([i in list, (i[1], i[0]) in list]):
            list.remove(i)
    return list

if __name__ == '__main__':
    street = 'flop'
    f = open(txt_save_path, "w", encoding='utf-8')
    for n, data_name in enumerate(data_name_list):
        data = load_data(data_name)
        matrix_ = get_EMDmatrix(street)
        dist_mat = distance_matrix(matrix_, data)
        coord_list = find_bottom_5(dist_mat)
        coord_list = remove_duplicate_element(coord_list)[:5]
        print(coord_list)
        save_data = transform(coord_list, data, dist_mat)
        f.write('-----------k{0:0>2}------------\n'.format(k_list[n]))
        for i in save_data:
            save_list = list(map(str, i))
            save_list[1] = '{:.10}'.format(save_list[1])
            tostr = '    '.join(save_list)
            print(tostr)
            f.write(tostr + '\n')
        f.write('\n')
    f.close()
