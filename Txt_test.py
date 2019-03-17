# -*- coding:utf-8 -*-

import settings

'''
Calculate the number of data contained in each bucket in @Cluster_result.py
'''

test = "flop"

if test == "river":
    file = open("river_cluster_result_10.txt", "r")
    cluster_count = settings.river_cluster_count
elif test == "turn":
    file = open("turn_cluster_result40.txt", "r")
    cluster_count = settings.turn_cluster_count
else:
    file = open("flop_cluster_result80.txt", "r")
    cluster_count = settings.flop_cluster_count


card_cluster_id = []
line = file.readline()
line = line[:-1]
id = line.split(":")[1]
card_cluster_id.append(int(id))
while line:
    line = file.readline()
    line = line[:-1]
    if line == "":
        break
    id = line.split(":")[1]
    card_cluster_id.append(int(id))
file.close()

print(len(card_cluster_id))
for i in range(1,cluster_count+1):
    print("---- {0} th---:".format(i),card_cluster_id.count(i))
