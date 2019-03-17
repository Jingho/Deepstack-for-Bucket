# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from pyemd import emd
import time
import ast
import argparse

'''
The program mainly evaluates the clustering effect by visualizing the data distribution before and after clustering.
Note:
The program can visualize three rounds of data, but the turn and flop round data is a distribution, and may result in less than prefetching due to forced restore to EHS.

@param street the round name
@param file_path the relative path of loading data
@param file_path the relative path of saving data
@param mode plot the data distribution before and after clustering separately
       through @{data} and @{results} and plot data distribution before and after clustering under different cluster numbers
@param ifsave whether to save the cluster center point
@class PlotData
'''

class PlotData(object):
    def __init__(self,
                street=None,
                file_path='data/',
                save_path='img/',
                mode='data',
                ifsave=True,
                ):
        assert street in ['flop','turn','river'], \
            'The street is None'
        assert mode in ['data', 'results', 'test'], \
            'The method is None'

        self.street = street
        self.file_path = file_path
        self.save_path = save_path
        self.mode = mode
        self.ifsave = ifsave
        self.matrix = self.get_matrix()

        self.points_list = self.get_data_set()
        self.EHS_list = self.EHS_data_set(self.points_list)
        self.n_data = len(self.points_list)
        self.main()

    def main(self):
        if self.mode == 'data':
            self.plot_data()
        elif self.mode == 'results':
            self.centroids = self.get_centroids(self.street)
            self.k = len(self.centroids)
            self.group_data = self.get_group_data()
            self.ehs_group_data = list(map(self.EHS_data_set,self.group_data))
            self.plot_result()
        else:
            from Cluster_data import Clustering

            for i in range(mink, maxk+1):
                kmean = Clustering(street=self.street, file_path=self.file_path, k=i, ifsave=False)
                self.centroids = kmean.main()
                self.k = len(self.centroids)
                self.group_data = self.get_group_data()
                self.ehs_group_data = list(map(self.EHS_data_set,self.group_data))
                self.plot_result()

    def get_data_set(self):
        file_name = {"river": "river_data.csv",
                    "turn": "turn_data.csv",
                    "flop": "flop_data.csv"}
        data = self.load_data(file_name.get(self.street))
        return data

    def EHS_data_set(self,points_list):
        if self.street == 'river':
            ehslist = list(map(lambda x: x[0], points_list))
        elif self.street == 'turn':
            centroids = self.get_centroids('river')
            centroids = np.reshape(centroids, len(centroids))
            ehslist = list(map(lambda x: np.dot(x, centroids), points_list))
        elif self.street == 'flop':
            centroids = self.get_centroids('turn')
            centroids = np.reshape((1, len(centroids)))
            ehslist = list(map(lambda x: np.dot(x, centroids), points_list))
        return ehslist

    def get_centroids(self, street):
        name = {"river": "river_cluster.csv",
                "turn": "turn_cluster.csv",
                "flop": "flop_cluster.csv"}
        path = name.get(street)
        centroids = self.load_data(path)
        return centroids

    def get_matrix(self):
        if self.street == "river":
            return

        elif self.street == "turn":
            river_cluster_name = 'river_cluster.csv'
            river_cluster = self.load_data(river_cluster_name)
            matrix = np.zeros((len(river_cluster),len(river_cluster)))
            for i,ic in enumerate(river_cluster):
                for j,jc in enumerate(river_cluster):
                    matrix[i,j] = abs(ic[0] - jc[0])
            return matrix

        elif self.street == "flop":
            turn_cluster_name = 'turn_cluster.csv'
            river_cluster_name = 'river_cluster.csv'
            river_cluster = self.load_data(river_cluster_name)
            turn_cluster = self.load_data(turn_cluster_name)
            matrix_ = np.zeros((len(river_cluster),len(river_cluster)))
            for i,ic in enumerate(river_cluster):
                for j,jc in enumerate(river_cluster):
                    matrix_[i,j] = abs(ic[0] - jc[0])
            matrix = self.distance_matrix(matrix_,turn_cluster)
            return matrix

    def get_p2p_dist(self, point_1, point_2):
        point_1, point_2 = np.array(point_1), np.array(point_2)
        if self.street == 'river':
            return abs(point_1 - point_2)
        else:
            return emd(point_1, point_2, self.matrix)

    def get_group_data(self):
        ''' Classify data by cluster centroids'''
        start_time = time.time()
        Cgroups = [[] for _ in range(self.k)]
        for data in self.points_list:
            d_list = [self.get_p2p_dist(data, x) for x in self.centroids]
            min_distance_index = np.argmin(list(d_list))
            Cgroups[min_distance_index].append(data)
        during_time = time.time() - start_time
        print('Grouping data time consuming {0:.3f} s'.format(during_time))
        for i,di in enumerate(Cgroups):
            print('cluster {} :'.format(i + 1), len(di))
        return Cgroups

    def load_data(self,filename):
        file_path = self.file_path + filename
        datas = []
        with open(file_path) as file:
            for line in file:
                data = []
                string_line = line.split("\n")[0].split(",")
                string_line.pop() if string_line[-1] == '' else None
                # print(string_line)
                data = [float(strline) for _,strline in enumerate(string_line)]
                datas.append(data)
        return datas

    def plot_data(self):
        plt.ion()
        plt.hist(self.EHS_list,100)
        plt.xlim(0, 1)
        plt.title('Raw data')
        plt.xlabel('EHS')
        plt.ylabel('Frequency')
        if self.ifsave:
            plt.savefig('img/{0}/{0} raw data distribution.jpg'.format(self.street))
        plt.close(1)

    def plot_result(self):
        plt.figure(1)
        plt.ion()
        plt.subplot(211)
        plt.xlim(0, 1)
        plt.hist(self.ehs_group_data, 50, stacked=True)
        plt.title('{0} cluster result(k={1})'.format(self.street, self.k))
        plt.ylabel('result')

        plt.subplot(212)
        plt.hist(self.EHS_list, 50, facecolor='0.5')
        plt.xlim(0, 1)
        plt.xlabel('EHS')
        plt.ylabel('data')
        if self.ifsave:
            plt.savefig('img/{0}/{0} result distribution(k={1:0>2}).jpg'.format(self.street, self.k))
        plt.pause(1)

''' Get parameters from command line '''
def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--street", type=str, default='river')
    parser.add_argument("--mode", type=str, default='test')
    parser.add_argument("--ifsave", type=ast.literal_eval, default=True)

    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':
    mink, maxk = [3,30]
    params = vars(get_params())
    p = PlotData(street=params['street'], mode=params['mode'], ifsave=params['ifsave'])
