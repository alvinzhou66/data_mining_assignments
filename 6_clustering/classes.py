from math import sqrt
import csv
import json
import os
import random
import sys
import time
def toPoint(x):
    result = Point(x)
    return result
def euclidDis(p1, p2):
    ans = 0
    for i in range(len(p1)):
        ans += (p1[i]-p2[i])**2
    return sqrt(ans)
def rddSUM(x):
    rddsum = [0 for i in range(x[1][0].dim)]
    rddsumq = [0 for i in range(x[1][0].dim)]
    points = set()
    for p in x[1]:
        for i in range(len(rddsum)):
            rddsum[i] += p.vector[i]
            rddsumq[i] += p.vector[i]**2
        points.add(p.index)
    return (x[0], rddsum, rddsumq, points)
def KMeans(input_data, k, current_cluster_id, RS):
    data = list()
    for item in input_data:
        data.append(Point(item))   
    randomCenter = random.sample(data, 1)
    clusters = {0+current_cluster_id: Cluster(0+current_cluster_id, randomCenter[0])}
    selectedPoint = [randomCenter[0].vector]

    for i in range(1, k):
        newCenter = None
        farthest_distance = 0
        for point in data:
            distance = point.farthestPoint(selectedPoint)
            if distance > farthest_distance:
                farthest_distance = distance
                newCenter = point
        selectedPoint.append(newCenter.vector)
        clusters[i+current_cluster_id] = Cluster(i+current_cluster_id, newCenter)

    cluster_centers = {}
    for index, c in clusters.items():
        cluster_centers[index] = c.center

    if_changed = 1
    loops = 0
    while if_changed != 0:
        if_changed = 0
        for point in data:
            point.updateCluster(cluster_centers)
            if point.changed == 1:
                clusters[point.cluster].addPoints(point)
                if_changed = 1
                if point.preCluster != -1:
                    clusters[point.preCluster].deletePoints(point.index)
        loops += 1
        # if loops > 24:
        #     if_changed = 0
    cluster_centers = {}
    for index, c in clusters.items():
        cluster_centers[index] = c.center
    if RS == 1:
        outlier = {}
        for index, c in clusters.items():
            outlier.update(c.createRS(1.5))
            clusters[index] = c
        if outlier:
            clusters[-1] = Cluster(-1, outlier.pop(list(outlier.keys())[0]))
            for point in outlier.values():
                clusters[-1].addPoints(point)
    return clusters
class Point:
    def __init__(self, data):
        self.index = data[0]
        self.vector = data[1]
        self.dim = len(data[1])
        self.cluster = -1
        self.preCluster = -1
        self.changed = 0
    def farthestPoint(self, point_list):
        result = 9999999999999999999
        for p in point_list:
            if p == self.vector:
                return -1
            distance = euclidDis(p, self.vector)
            if distance < result:
                result = distance
        return result
    def updateCluster(self, cluster_centers):
        distance = 9999999999999999999
        newCluster = 0
        for index, center in cluster_centers.items():
            temp = euclidDis(center, self.vector)
            if temp < distance:
                distance = temp
                newCluster = index
        if self.cluster == -1 or self.cluster != newCluster:
            self.preCluster = self.cluster 
            self.cluster = newCluster
            self.changed = 1
        else:
            self.changed = 0
    def calculateMAH(self, center, std):
        distance = 0
        for i in range(self.dim):
            minus = self.vector[i] - center[i]
            if std[i] != 0:
                distance += (minus/std[i])**2
            else:
                distance += minus**2
        return sqrt(distance)
    def selectDS(self, ds, coef):
        threshold = coef*sqrt(self.dim)
        distance = 9999999999
        for index in ds.keys():
            s = ds[index]
            mah_distance = self.calculateMAH(s.center, s.std())
            if mah_distance < distance and mah_distance <= threshold:
                self.cluster = index
        return (self.cluster, self)

class Cluster:
    def __init__(self, index, center):
        self.index = index
        self.dim = center.dim
        self.points = {center.index: center}
        self.points_count = 1
        self.updateCenter()
    def updateCenter(self):
        for point in self.points.values():
            sum_vector = map(lambda x, y: x+y, [0 for z in range(self.dim)], point.vector)
        self.center = []
        for num in sum_vector:
            self.center.append(num/self.points_count)
    def addPoints(self, point):
        self.points[point.index] = point
        self.points_count = len(self.points)
        self.updateCenter()
    def deletePoints(self, point_index):
        del self.points[point_index]
        self.points_count = len(self.points)
        self.updateCenter()
    def createRS(self, a):
        result = {}
        total_distance = 0
        for point in self.points.values():
            total_distance += euclidDis(point.vector, self.center)
        r = total_distance/self.points_count
        for p in self.points.items():
            if euclidDis(p[1].vector, self.center) > r*a:
                 result[p[0]] = p[1]
                 self.deletePoints(p[0])
        return result
    def getInput(self):
        result = list()
        for index, point in self.points.items():
            result.append([index] + point.vector)
        return result

class CreateSet:
    def __init__(self, cluster: Cluster):
        self.index = cluster.index
        self.dim = cluster.dim
        self.points = set(cluster.points.keys())
        self.N = cluster.points_count
        self.SUM = [0 for i in range(self.dim)]
        self.SUMQ = [0 for i in range(self.dim)]
        self.calculateSUM(cluster)
        self.center = [s/self.N for s in self.SUM]
    def std(self):
        std = []
        for i in range(self.dim):
            std.append(sqrt(abs(self.SUMQ[i]/self.N) - (self.SUM[i]/self.N)**2))
        return std
    def calculateSUM(self, cluster: Cluster):
        for point in cluster.points.values():
            for i in range(self.dim):
                self.SUM[i] += point.vector[i]
                self.SUMQ[i] += point.vector[i]**2
    def updateSUM(self, new_points, new_sum, new_sumq):
        self.N += len(new_points)
        for i in range(self.dim):
            self.SUM[i] += new_sum[i]
            self.SUMQ[i] += new_sumq[i]
        self.std()
        self.points = self.points.union(new_points)
        self.center = [s/self.N for s in self.SUM]   
        return self
    def addFromCS(self, CS):
        return self.updateSUM(CS.points, CS.SUM, CS.SUMQ)
    def calculateMAH(self, center, std):
        distance = 0
        for i in range(self.dim):
            minus = self.center[i] - center[i]
            if std[i] != 0:
                distance += (minus/std[i])**2
            else:
                distance += minus**2
        return sqrt(distance)
    def combine(self, CS):      
        threshold = sqrt(self.dim)*2
        distance = 999999999999
        cs_index = -1
        for index, cs in CS.items():
            if index != self.index:
                mah_distance = self.calculateMAH(cs.center, cs.std())
                if mah_distance < distance and mah_distance <= threshold:
                    distance = mah_distance
                    cs_index = index
        return (self.index, cs_index, distance)
