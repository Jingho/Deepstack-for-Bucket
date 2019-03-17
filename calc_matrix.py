# -*- coding:utf-8 -*-

import os
import numpy as np
from pyemd import emd

'''
Load the data from @{Generate_data.py}
Computes the matrix for computing Earth Mover's Distance in generating
clustering data
@param street the round name
@param file_path the relative path for storing data
@Cmatrix
'''

class Cmatrix(object):
    def __init__(self,
                street=None,
                file_path='data/',
                ):

        assert street in ['flop','turn','river'], \
        'The street is None'

        self.street = street
        self.file_path = file_path

        if not os.path.exists(self.file_path[:-1]):
            os.makedirs(self.file_path[:-1])
            print(self.file_path,'created sucessfully')

    '''
    Load the data from @{Generate_data.py}
    @param filename hand representation file name
    return an NxM list containing N vectors of hand representation generated with @{Generate_data.py}
    @load_data
    '''

    def load_data(self, filename):
        data_path = self.file_path + filename
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

    '''
    Computes the matrix for computing Earth Mover's Distance in flop round
    @param matrix_ the cluster center point distance matrix of turn round
    @param mat_centroids the cluster center point of turn round
    return an NxN np.array containing EMD distance of the center point of the turn round
    @get_NextEMD_Matrix
    '''

    def get_NextEMD_Matrix(self,matrix_,mat_centroids):
        matrix = np.zeros(shape=(len(mat_centroids),len(mat_centroids)))
        for i, mi in enumerate(mat_centroids):
            for j, mj in enumerate(mat_centroids):
                if i == j:
                    matrix[i][j] = 0
                else:
                    matrix[i][j] = abs(emd(np.array(mi), np.array(mj),matrix_))
        return matrix

    '''
    Computes the matrix for computing Earth Mover's Distance in turn round
    @param mat_centroids the cluster center point of river round
    return an NxN np.array containing Euclidean distance of the center point of the river round
    @get_Euclidean_Matrix
    '''

    def get_Euclidean_Matrix(self, mat_centroids):
        n = len(mat_centroids)
        matrix = np.zeros((n, n))
        for i, ic in enumerate(mat_centroids):
            for j, jc in enumerate(mat_centroids):
                matrix[i, j] = abs(ic[0] - jc[0])
        return matrix


if __name__ == '__main__':
    print('test the class Cmatrix:')
    m = Cmatrix(street='river')
    print(len(m.load_data('data_5.csv')))
