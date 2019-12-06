from __future__ import division

import matplotlib.pyplot as plt
import numpy as np

from utils import *


def check_in_disc(pt, lpt):
    A = lpt[0]
    B = lpt[1]
    C = lpt[2]
    a = np.array(([B.x - A.x, C.x - A.x], [B.y - A.y, C.y - A.y]))
    array1 = [A.x - pt.x, A.y - pt.y, pow(A.x - pt.x, 2) + pow(A.y - pt.y, 2)]
    array2 = [B.x - pt.x, B.y - pt.y, pow(B.x - pt.x, 2) + pow(B.y - pt.y, 2)]
    array3 = [C.x - pt.x, C.y - pt.y, pow(C.x - pt.x, 2) + pow(C.y - pt.y, 2)]
    b = np.array((array1, array2, array3))
    if np.linalg.slogdet(a)[0] * np.linalg.slogdet(b)[0] > 0:
        return 1
    else:
        return 0


def remove_triangle(tr1, tr2):
    l = 0
    for adj in tr1.adjacent:
        if set(adj.points) == set(tr2.points):
            tr1.adjacent.remove(l)
            break
        l += 1


class TriangleTree:
    def __init__(self, root=None):
        self.root = root

    def search_triangle(self, pt, plist):
        for child in self.root.childs:
            check = self.check_in_triangle(child, pt, plist)
            if check:
                return check
            else:
                continue

    def check_in_triangle(self, tr, pt, plist):
        if tr.point_in(pt, plist):
            if not tr.childs:
                return tr
            else:
                for child in tr.childs:
                    self.check_in_triangle(child, pt, plist)
        else:
            pass

    def insert_point(self, p, plist):
        pt = plist[p]
        tr = self.search_triangle(pt, plist)
        t1 = Triangle([p, tr.points[0], tr.points[1]])
        t2 = Triangle([p, tr.points[2], tr.points[0]])
        t3 = Triangle([p, tr.points[1], tr.points[2]])
        tr.childs.append(t1)
        tr.childs.append(t2)
        tr.childs.append(t3)
        t1.adjacent = t1.adjacent + [t2, t3]
        t2.adjacent = t2.adjacent + [t1, t3]
        t3.adjacent = t3.adjacent + [t2, t1]
        for adj in tr.adjacent:
            remove_triangle(adj, tr)
            for child in tr.childs:
                if len(set(child.points).intersection(set(adj.points))) > 1:
                    child.adjacent.append(adj)
                    adj.adjacent.append(child)
                    break
        list_tmp = tr.adjacent
        while list_tmp:
            tr_tmp = list_tmp.pop(0)
            if check_in_disc(pt, [plist[m] for m in tr_tmp.points]):
                pass


class Vector(object):
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y


class Point(object):
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def prescale(self, x, y, d):
        self.x = (self.x - x) / d
        self.y = (self.y - y) / d
        return self

    def postscale(self, x, y, d):
        self.x = d * self.x + x
        self.y = d * self.y + y
        return self


class Triangle(object):
    def __init__(self, x=[], y=[], z=[]):
        self.points = x
        self.childs = y
        self.adjacent = z


class Segment(object):
    def __init__(self, x=0, y=0, dens=0):
        self.extrA = x
        self.extrB = y
        self.dens = dens


class Mesh(object):
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
                    C = Point()
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
            self.point_list.append(Point(fields[0], fields[1]))
        else:
            self.nnodes += 1
            self.point_list.append(Point(float(fields[3]), float(fields[4])))
            self.label_list[fields[0]] = self.nnodes
