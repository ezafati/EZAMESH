from __future__ import division
import math, sys
from math import sqrt
import random
import itertools
import matplotlib.pyplot as plt
from utils import *


class vector(object):
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y


class point(object):
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def prescale(self, x, y, d):
        self.x = (self.x - x) / d
        self.y = (self.y - y) / d
        return self

    def inv_prescale(self, x, y, d):
        self.x = d * self.x + x
        self.y = d * self.y + y
        return self


class triangle(object):
    def __init__(self, x=[], y=[], z=[]):
        self.points = x
        self.childs = y
        self.adjacent = z


class segment(object):
    def __init__(self, x=0, y=0, dens=0):
        self.extrA = x
        self.extrB = y
        self.dens = dens


class mesh(object):
    def __init__(self, x=[], y=0, slabel=[], nnodes=0, polist=[], llist={}, size=[], ltri=[], terminate=0):
        self.boundary = x
        self.nboseg = y
        self.seglab = slabel
        self.nnodes = nnodes
        self.point_list = polist
        self.label_list = llist
        self.triangle_list = ltri
        self.asize = size
        self.terminate = terminate

    def add_bound_seg(self, fields, n_line):
        try:
            self.seglab.append(fields[0])
            NA = self.label_list[fields[3]]
            NB = self.label_list[fields[4]]
            l = float(fields[5])
            A = self.point_list[NA - 1]
            B = self.point_list[NB - 1]
            dist = pow(A.x - B.x, 2) + pow(A.y - B.y, 2)
            dist = sqrt(dist)
            NN = dist / l
            k = 1
            ifpart = math.modf(NN)
            if ifpart[1] == 0 or (ifpart[1] == 1 and ifpart[0] <= 0.5):
                self.nboseg += 1
                self.boundary.append({NA - 1, NB - 1})
            else:
                if ifpart[0] <= 0.5:
                    NN = ifpart[1]
                else:
                    NN = ifpart[1] + 1
                while 0 < k < NN:
                    self.nnodes += 1
                    C = point()
                    C.x = (1 - (k / NN)) * A.x + (k / NN) * B.x
                    C.y = (1 - (k / NN)) * A.y + (k / NN) * B.y
                    self.point_list.append(C)
                    if k == 1:
                        self.nboseg += 1
                        self.boundary.append({NA - 1, self.nnodes - 1})
                    if k == NN - 1:
                        self.nboseg += 1
                        self.boundary.append({self.nnodes - 1, NB - 1})
                        if k != 1:
                            self.nboseg += 1
                            self.boundary.append({(self.nnodes - 1) - 1, self.nnodes - 1})
                    if (k != 1) and (k != NN - 1):
                        self.nboseg += 1
                        self.boundary.append({(self.nnodes - 1) - 1, self.nnodes - 1})
                    k += 1
        except Exception as err:
            print('exit for the error in line ', n_line, ': ', err)
            sys.exit()

    def plot_bound(self):
        fig = plt.figure()  # create figure object
        ax = fig.add_subplot(1, 1, 1)  # create an axes object
        for l in self.boundary:
            l = list(l)
            coord = [[self.point_list[p].x for p in l], [self.point_list[p].y for p in l]]
            ax.plot(coord[0], coord[1], 'k-*')
        plt.show()

    def add_point(self, fields):
        if len(fields) == 2:
            self.nnodes += 1
            self.point_list.append(point(fields[0], fields[1]))
        else:
            self.nnodes += 1
            self.point_list.append(point(float(fields[3]), float(fields[4])))
            self.label_list[fields[0]] = self.nnodes
