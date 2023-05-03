# -*- coding: utf-8 -*-
"""
This file contains all the functions needed in the assignment.
Feel free to use them but, please, give credits whenever
it helps in your amazing works!! :) The functions are implementations from the
following papers:

@article{sebastian2021codkf,
  title={All-in-one: Certifiable Optimal Distributed Kalman Filter under Unknown Correlations},
  author={Sebastián, Eduardo and Montijano, Eduardo and Sagues, Carlos},
  journal={arXiv preprint arXiv:2105.15061},
  year={2021}
}

@article{montijano2012chebyshev,
  title={Chebyshev polynomials in distributed consensus applications},
  author={Montijano, Eduardo and Montijano, Juan Ignacio and Sagues, Carlos},
  journal={IEEE Transactions on Signal Processing},
  volume={61},
  number={3},
  pages={693--706},
  year={2012},
  publisher={IEEE}
}

Current Version: 7th of June of 2021

Eduardo Sebastián Rodríguez, PhD Candidate / sites.google.com/unizar.es/eduardosebastianrodriguez
Perception-Oriented Control (POC) Team / https://sites.google.com/unizar.es/poc-team
Robotics, Perception and Real Time Group (RoPeRT) / robots.unizar.es
Department of Computer Science and Systems Engineering / diis.unizar.es
University of Zaragoza / unizar.es
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.optimize
from scipy.optimize import linear_sum_assignment


def randomPositionsProximity(N: int,
                             side: float,
                             dmin: float,
                             dmax: float):
    """
    This function generates a random proximity graph. A proper selection
    of parameters allows to a fast convergence towards a random graph;
    otherwise, the method may keep runing ad eternum

    Arguments:
        N {int}        -- Number of nodes
        side {float}   -- Side of the square area for generating the graph (square area)
        dmin {float}   -- Minimum distance between nodes
        dmax {float}   -- Maximum distance two nodes communicate

    Returns:
        positions {list}     -- Position of the nodes in 2D
        Degree    {np.array} -- Degree matrix
        Adjency   {np.array} -- Adjency matrix
        Laplacian {np.array} -- Laplacian of the graph

    """

    # Initialise positions and Laplacian
    positions = [np.zeros(2) for i in range(N)]
    matrix = np.zeros([N, N])

    # First, place the nodes in the 2D square area
    for i in range(N):
        isNewNodeAllowable = False
        while not isNewNodeAllowable:
            # Throw the coin and place a new node in a random position
            x_position = side * (np.random.uniform(0, 1) - 0.5)
            y_position = side * (np.random.uniform(0, 1) - 0.5)

            # Check if new node is allowable
            j = 0
            isNewNodeAtMinDistance = True
            isNewNodeAtMaxDistance = False
            while (isNewNodeAtMinDistance) and (j <= i - 1):
                distance = np.linalg.norm(positions[j] - np.array([x_position, y_position]))
                if distance <= dmin:
                    isNewNodeAtMinDistance = False
                if distance <= dmax:
                    isNewNodeAtMaxDistance = True
                if isNewNodeAtMinDistance:
                    # Try again with the next node
                    j += 1
            if (isNewNodeAtMinDistance and isNewNodeAtMaxDistance) or (i == 0):
                isNewNodeAllowable = True
        # Add new node
        positions[i][0] = x_position
        positions[i][1] = y_position

    # Given the random placed nodes, build the Laplacian
    # Remember that the Laplacian is equal to the Degree matrix minus
    # the adjency matrix:
    #                       L = D-A
    for i in range(N):
        for j in range(i + 1, N):
            distance = np.linalg.norm(positions[j] - positions[i])
            if distance <= dmax:
                matrix[i, j] = -1
                matrix[j, i] = -1

    Degree = np.diag(np.sum(-matrix, 1))
    Adjency = -matrix
    Laplacian = Degree - Adjency

    return positions, Degree, Adjency, Laplacian


def get_cmap(n: int):
    """
    This function generates an equally space array of RGB colours
    """
    return plt.cm.get_cmap('hsv', n + 1)


def angle_diff(a, b):
    a1 = a % (2 * math.pi)
    b1 = b % (2 * math.pi)
    d1 = a1 - b1
    d2 = 2 * math.pi - np.abs(d1)
    if d1 > 0:
        d2 = -d2
    if np.abs(d1) < np.abs(d2):
        error = np.abs(d1)
        if b1 > a1:
            sense = 1
        else:
            sense = -1
    else:
        error = np.abs(d2)
        if b1 > a1:
            sense = -1
        else:
            sense = 1
    error = error * sense;
    return error, sense


def motion(x: np.array,
           A: np.array,
           Q: np.array):
    """
    This function simulates the motion of a 2D particle

    Arguments:
        x {np.array} -- Current position of the particle
        A {np.array} -- Motion model
        Q {np.array} -- Covariance of the stochastic gaussian noise affecting the motion

    Returns:
        x {np.array} -- Next position

    """

    return A @ x + np.random.multivariate_normal(np.zeros(x.size), Q).reshape([x.size, 1])
