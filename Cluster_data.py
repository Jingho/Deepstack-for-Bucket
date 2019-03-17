# -*- coding:utf-8 -*-

import imp
import numpy as np
import random
import math
import time
import os
import ast
import argparse
from pyemd import emd
from calc_matrix import Cmatrix
from sklearn.cluster import KMeans

STREET = 'turn'
FILE_PATH = 'data/'
INITIAL_METHOD = 'kmean++'
IFSAVE = True
CLUSTER_K = 5

"""
Initialize the cluster center point using kmeans++
@param points_list list the cluster point
@param k int the number of clusters
@param EMDmatrix np.array the cluster center point distance matrix of the last round
return k initial seeds
@K_Means_Plus_Plus
"""
class K_Means_Plus_Plus(object):

    def __init__(self, points_list, k, EMDmatrix):
        self.centroid_count = 0
        self.point_count = len(points_list)
        self.cluster_count = k
        self.matrix = EMDmatrix
        self.points_list = list(points_list)
        print('---K-means++ Cluster point initialization start---')
        self.initialize_first_centroid()
        self.init_centroids_list = self.initialize_other_centroids()
        print('---K-means++ Cluster point initialization ends---\n')

    """
    Picks a random point to serve as the first centroid
    @initialize_first_centroid
    """

    def initialize_first_centroid(self):
        self.centroid_list = []
        index = random.randint(0, len(self.points_list)-1)

        self.centroid_list.append(self.remove_point(index))
        self.centroid_count = 1

    """
    Removes point associated with given index so it cannot be picked as a future centroid.
    Returns list containing coordinates of newly removed centroid
    @remove_point
    """
    def remove_point(self, index):
        new_centroid = self.points_list[index]
        del self.points_list[index]

        return new_centroid

    """
    Finds the other k-1 centroids from the remaining lists of points
    @initialize_other_centroids
    """

    def initialize_other_centroids(self):
        while not self.is_finished():
            print('centroid count: {}'.format(self.centroid_count))
            distances = self.find_smallest_distances()
            chosen_index = self.choose_weighted(distances)
            self.centroid_list.append(self.remove_point(chosen_index))
            self.centroid_count += 1
        return self.centroid_list

    """
    Calculates distance from each point to its nearest cluster center. Then chooses new
    center based on the weighted probability of these distances
    @find_smallest_distances
    """
    def find_smallest_distances(self):
        distance_list = []

        for point in self.points_list:
            distance_list.append(self.find_nearest_centroid(point))

        return distance_list

    """
    Finds centroid nearest to the given point, and returns its distance
    @find_nearest_centroid
    """
    def find_nearest_centroid(self, point):
        min_distance = math.inf

        for values in self.centroid_list:
            distance = self.get_distance(values, point)
            if distance < min_distance:
                min_distance = distance

        return min_distance

    """
    Chooses an index based on weighted probability
    @choose_weighted
    """
    def choose_weighted(self, distance_list):
        distance_list = [x**2 for x in distance_list]
        weighted_list = self.weight_values(distance_list)
        indices = [i for i in range(len(distance_list))]
        return np.random.choice(indices, p = weighted_list)

    """
    Weights values from [0,1]
    @weight_values
    """
    def weight_values(self, list):
        sum = np.sum(list)
        return [x/sum for x in list]

    """
    Computes N-d euclidean distance between two points represented as lists:
    (x1, x2, ..., xn) and (y1, y2, ..., yn)
    @get_distance
    """
    def get_distance(self, point1, point2):
        point_1 = np.array(point1, dtype=np.float64)
        point_2 = np.array(point2, dtype=np.float64)

        distance = emd(point_1, point_2, self.matrix)

        return distance

    """
    Checks to see if final condition has been satisfied (when K centroids have been created)
    @is_finished
    """

    def is_finished(self):
        outcome = False
        if self.centroid_count == self.cluster_count:
            outcome = True
        return outcome

    """
    Returns final centroid values
    @final_centroids
    """

    def final_centroids(self):
        return self.centroid_list

"""
Solves the problem of clustering a set of random data points into k clusters.

The process is iterative and visually shown how the clusters convergence on
the optimal solution.
@param points_list list the cluster point
@param k int the number of clusters
@param EMDmatrix np.array the cluster center point distance matrix of the last round
@param initialMethod string initialize cluster center point method
return Cluster center point
@Kmeans
"""

class Kmeans(object):

    def __init__(self, points_list, k, EMDmatrix,
                initialMethod='kmean++'):
        assert initialMethod in ['kmean++','random'], 'initialMethod input error'
        self.point_count = len(points_list)
        self.k = k
        self.matrix = EMDmatrix
        self.points_list = list(points_list)
        self.initialMethod = initialMethod

    """Takes the dataPoint and find the centroid index that it is closest too.
    @param centroids The list of centroids
    @param dataPoint The dataPoint that is going to be determined which centroid it is closest too
    @points_best_cluster
    """
    def points_best_cluster(self, centroids, dataPoint):
        closestCentroid = None
        leastDistance = None

        for i in range(len(centroids)):
            distance = emd(np.array(dataPoint),np.array(centroids[i]),self.matrix)
            #print(distance)
            if (leastDistance == None or distance < leastDistance ):
                closestCentroid = i
                leastDistance = distance

        return closestCentroid

    """
    Finds the new centroid location given the cluster of data points. The
    mean of all the data points is the new location of the centroid.

    @param cluster A single cluster of data points, used to find the new centroid
    @new_centroid
    """
    def new_centroid(self, cluster):
        return np.mean(cluster, axis=0)

    """
    Creates a new configuration of clusters for the given set of dataPoints
    and centroids.
    @param centroids The list of centroids
    @param dataPoints The set of random data points to be clustered
    return The set of new cluster configurations around the centroids
    @configure_clusters
    """

    def configure_clusters(self, centroids, dataPoints):
        clusters = []
        for i in range(len(centroids)):
            cluster = []
            clusters.append(cluster)

        # For all the dataPoints, place them in initial clusters
        for i in range(self.point_count):
            idealCluster = self.points_best_cluster(centroids, dataPoints[i])
            clusters[idealCluster].append(dataPoints[i])
        #NOTE:it is dangerous
        max = 0
        max_index = 0
        blank = []
        for i in range(len(clusters)):
            if len(clusters[i]) > max:
                max = len(clusters[i])
                max_index = i
            if len(clusters[i]) == 0:
                blank.append(i)
        for i in range(len(blank)):
            clusters[blank[i]].append(clusters[max_index].pop())

        return clusters

    """
    Calculates the cluster's Residual Sum of Squares (RSS)
    @param cluster The list of data points of one cluster
    @param centroid The centroid point of the corresponding cluster
    @get_cluster_RSS
    """
    def get_cluster_RSS(self, cluster, centroid):
        sumRSS = 0

        for i in range(len(cluster)):
            sumRSS += pow(abs(emd(np.array(cluster[i]), np.array(centroid),self.matrix)), 2)

        return sumRSS

    """
    Iteratively clusters the dataPoints into the most appropriate cluster
    based on the centroid's distance. Each centroid's position is updated to
    the new mean of the cluster on each iteration. When the RSS doesn't change
    anymore then the best cluster configuration is found.

    @param dataPoints The set of random data points to be clustered
    @param k The number of clusters
    @solve
    """

    def solve(self):
        # Create the initial centroids and clusters
        dataPoints = self.points_list
        k = self.k
        l = len(dataPoints[0])
        if self.initialMethod == 'kmean++':
            kpp = K_Means_Plus_Plus(self.points_list,self.k,self.matrix)
            centroids = kpp.init_centroids_list
        elif self.initialMethod == 'random':
            centroids = random.sample(dataPoints, self.k)
        print('------K-means Clustering start------')
        clusters = self.configure_clusters(centroids, dataPoints)

        # Loop till algorithm is done
        allRSS = []
        notDone = True
        lastRSS = 0
        while (notDone):
          # Find Residual Sum of Squares of the clusters
            clustersRSS = []
            for i in range(len(clusters)):
                clustersRSS.append(self.get_cluster_RSS(clusters[i], centroids[i]) / self.point_count)
            currentRSS = sum(clustersRSS)
            allRSS.append(currentRSS)
            print("RSS", currentRSS)

          # See if the kmean algorithm has converged
            if (currentRSS == lastRSS):
                notDone = False
            else:
                lastRSS = currentRSS

            # Update each of the centroids to the new mean location
            for i in range(len(centroids)):
                centroids[i] = self.new_centroid(clusters[i])

            # Reconfigure the clusters to the new centroids
            clusters = self.configure_clusters(centroids, dataPoints)
        print('------K-means Clustering ends------')
        return centroids

    def main(self):
        if self.street == 'river':
            print('clusters by sklearn:')
            cluster = KMeans(n_clusters=self.k)
            results = cluster.fit(self.points_list)
            centroids = cluster.cluster_centers_.flatten()
            centroids = np.sort(centroids).tolist()
        else:
            centroids = self.solve()
        return centroids

"""
Load data and calculate cluster center points by class @{Kmeans}

@param street the round name
@param file_path the relative path of data
@param k int the number of clusters
@param initialMethod string initialize cluster center point method
@param ifsave whether to save the cluster center point
@Clustering
"""

class Clustering(Cmatrix):
    def __init__(self,
                street=None,
                file_path='data/',
                k=None,
                initialMethod='kmean++',
                ifsave=True,
                ):
        super().__init__(street, file_path)
        assert street in ['flop','turn','river'], \
            'The street is None'
        assert isinstance(k, int), 'The k is not a valid parameter'
        self.street = street
        self.file_path = file_path
        self.initialMethod = initialMethod
        self.k = k
        self.ifsave = ifsave
        self.points_list = self.get_data_set()
        self.matrix = self.get_EHSmatrix()
        self.savename = self.get_savename()

    '''Cluster center point save file name'''
    def get_savename(self):
        savename = {"river": "river_cluster_k{}.csv".format(self.k),
                    "turn": "turn_cluster_k{}.csv".format(self.k),
                    "flop": "flop_cluster_k{}.csv".format(self.k)}
        return self.file_path + savename.get(self.street)

    '''load data'''
    def get_data_set(self):
        file_name = {"river": "river_data.csv",
                    "turn": "turn_data.csv",
                    "flop": "flop_data.csv"}
        return self.load_data(file_name.get(self.street))

    '''Calculate the last round of cluster center point distance matrix'''
    def get_EHSmatrix(self):
        if self.street == "river":
            return

        elif self.street == "turn":
            river_cluster_name = 'river_cluster.csv'
            river_cluster = self.load_data(river_cluster_name)
            return self.get_Euclidean_Matrix(river_cluster)


        elif self.street == "flop":
            turn_cluster_name = 'turn_cluster.csv'
            river_cluster_name = 'river_cluster.csv'
            assert os.path.exists(self.file_path+river_cluster_name), river_cluster_name + ' is not exists'
            assert os.path.exists(self.file_path+turn_cluster_name), turn_cluster_name + 'is not exists'
            river_cluster = self.load_data(river_cluster_name)
            turn_cluster = self.load_data(turn_cluster_name)
            matrix_ = self.get_Euclidean_Matrix(river_cluster)
            return self.get_NextEMD_Matrix(matrix_, turn_cluster)

    '''Calculate the center point of the river round cluster'''
    def skmeans(self):
        print('clusters by sklearn:')
        cluster = KMeans(n_clusters=self.k)
        results = cluster.fit(self.points_list)
        centroids = cluster.cluster_centers_.flatten()
        centroids = np.sort(centroids)
        return centroids

    '''Save cluster center point'''
    def savecentroids(self, centroids_list):
        if self.street == 'river':
            with open(self.savename,'w') as file:
                for i in centroids_list:
                    temp_string = str(i.tolist())
                    file.write(temp_string+'\n')
        else:
            with open(self.savename,'w') as file:
                for i in centroids_list:
                    temp_string = str(i.tolist()[0])
                    for j in range(1,len(i)):
                        temp_string = temp_string + ',' + str(i.tolist()[j])
                    file.write(temp_string+'\n')
        print(self.street,'centroids saved in',self.savename)

    '''Main program'''
    def main(self):
        print(self.street, 'clustering start')
        if self.street == 'river':
            centroids_list = self.skmeans()
        else:
            kmeans = Kmeans(self.points_list, self.k, self.matrix, self.initialMethod)
            centroids_list = kmeans.solve()
        print('final_centroids:')
        for cent in centroids_list:print(cent.tolist())
        if self.ifsave:
            self.savecentroids(centroids_list)
        return centroids_list

''' Get parameters from command line '''
def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--street", type=str, default='river')
    parser.add_argument("--file_path", type=str, default='data/')
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--initialMethod", type=str, default='kmean++')
    parser.add_argument("--ifsave", type=ast.literal_eval, default=True)

    args, _ = parser.parse_known_args()
    return args

if __name__ == '__main__':
    params = vars(get_params())
    test = Clustering(street=params['street'], file_path=params['file_path'], k=params['k'], initialMethod=params['initialMethod'], ifsave=params['ifsave'])
    centroid_list = test.main()
