# -*- coding: utf-8 -*-

"""
This is a pure Python implementation of the K-Means Clustering algorithmn.

This script specifically avoids using numpy or other more obscure libraries. It
is meant to be *clear* not fast.
__author__ : Valentin Nyzam
"""

import math
import random
# from collections import Counter
from model.wmd import word_mover_distance
from copy import deepcopy
import threading
import queue
from globals import THREAD


def getDistance(a, b, wvmodel):
    """getDistance
    compute distance between two Points
    :param a: list of word
    :param b: list of word
    """
    return word_mover_distance(a, b, wvmodel)


class Point(object):
    '''
    A point in n dimensional space
    '''
    def __init__(self, concepts):
        '''
        concepts - A list of concept
        '''

        self.concepts = concepts
        # self.n = len(coords)

    def __repr__(self):
        return str(self.concepts)


class Cluster(object):
    '''
    A set of points and their centroid
    '''

    def __init__(self, points, wvmodel):
        '''
        points - A list of point objects
        '''

        if len(points) == 0:
            raise Exception("ERROR: empty cluster")

        # The points that belong to this cluster
        self.points = points
        self.model = wvmodel
        # The dimensionality of the points in this cluster
        # self.n = points[0].n

        # Assert that all points are of the same dimensionality
        # for p in points:
        # if p.n != self.n:
        # raise Exception("ERROR: inconsistent dimensions")

        # Set up the initial centroid (this is usually based off one point)
        self.centroid = self.calculateCentroid()

    def __repr__(self):
        '''
        String representation of this object
        '''
        return str(self.points)

    def update(self, points):
        '''
        Returns the distance between the previous centroid and the new after
        recalculating and storing the new centroid.

        Note: Initially we expect centroids to shift around a lot and then
        gradually settle down.
        '''
        old_centroid = self.centroid
        self.points = points
        self.centroid = self.calculateCentroid()
        shift = getDistance(old_centroid, self.centroid, self.model)
        return shift

    def calculateCentroid(self):
        '''
        Finds a virtual center point for a group of n-dimensional points
        '''
        # numPoints = len(self.points)
        # Get a list of all coordinates in this cluster
        concepts = set([p for p in self.points])
        # Counter(self.points)
        # Reformat that so all x's are together, all y'z etc.
        # unzipped = zip(*coords)
        # Calculate the mean for each dimension
        # centroid_concept = concepts.most_common(20)

        # [math.fsum(dList)/numPoints for dList in unzipped]

        # return Point(centroid_coords)
        return concepts


def kmeans(points, k, cutoff, wvmodel):
    points = [set(point) for point in points]
    # Pick out k random points to use as our initial centroids
    initial = random.sample(points, k)

    # Create k clusters using those centroids
    # Note: Cluster takes lists, so we wrap each point in a list here.
    clusters = [Cluster(set(p), wvmodel) for p in initial]

    # Loop through the dataset until the clusters stabilize
    loopCounter = 0
    while True:
        # Create a list of lists to hold the points in each cluster
        lists = [[] for _ in clusters]

        # Start counting loops
        loopCounter += 1
        # For every point in the dataset ...

        # Make multithread
        q = queue.Queue()
        threads = []
        lock = threading.Lock()
        for i in range(THREAD):
            _clusters = deepcopy(clusters)
            t = threading.Thread(target=_evaluate_point,
                                 args=(q, lock, lists, _clusters,
                                       wvmodel, ))
            t.start()
            threads.append(t)
        for p in points:
            q.put(p)

        # block until all tasks are done
        q.join()
        # stop workers
        for i in range(THREAD):
            q.put(None)
        for t in threads:
            t.join()

        for p in points:
            q.put(p)
        # Set our biggest_shift to zero for this iteration
        biggest_shift = 0.0

        # For each cluster ...
        for i in range(len(clusters)):
            # Calculate how far the centroid moved in this iteration
            shift = clusters[i].update(lists[i])
            # Keep track of the largest move from all cluster centroid updates
            biggest_shift = max(biggest_shift, shift)

        print(clusters)

        # If the centroids have stopped moving much, say we're done!
        if biggest_shift < cutoff:
            print("Converged after %s iterations" % loopCounter)
            break
    return clusters


def _evaluate_point(q, lock, lists, clusters, wvmodel):
    # Get the distance between that point and the centroid of the first
    # cluster.
    while True:
        p = q.get()
        smallest_distance = getDistance(p, clusters[0].centroid, wvmodel)

        # Set the cluster this point belongs to
        clusterIndex = 0

        # For the remainder of the clusters ...
        for i in range(len(clusters) - 1):
            # calculate the distance of that point to each other cluster's
            # centroid.
            distance = getDistance(p, clusters[i+1].centroid, wvmodel)
            # If it's closer to that cluster's centroid update what we
            # think the smallest distance is
            if distance < smallest_distance:
                smallest_distance = distance
                clusterIndex = i+1
        # After finding the cluster the smallest distance away
        # set the point to belong to that cluster
        with lock:
            lists[clusterIndex].append(p)


def euclidean_distance(a, b):
    '''
    Euclidean distance between two n-dimensional points.
    https://en.wikipedia.org/wiki/Euclidean_distance#n_dimensions
    Note: This can be very slow and does not scale well
    '''
    if a.n != b.n:
        raise Exception("ERROR: non comparable points")

    accumulatedDifference = 0.0
    for i in range(a.n):
        squareDifference = pow((a.coords[i]-b.coords[i]), 2)
        accumulatedDifference += squareDifference
    distance = math.sqrt(accumulatedDifference)

    return distance


def makeRandomPoint(n, lower, upper):
    '''
    Returns a Point object with n dimensions and values between lower and
    upper in each of those dimensions
    '''
    p = Point([random.uniform(lower, upper) for _ in range(n)])
    return p
