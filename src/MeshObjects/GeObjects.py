from __future__ import division

import matplotlib.pyplot as plt
import numpy as np

from utils import *


def check_intersection(plist, seg1, seg2):
    seg1 = list(seg1)
    seg2 = list(seg2)
    A = plist[seg1[0]]
    B = plist[seg1[1]]
    C = plist[seg2[0]]
    D = plist[seg2[1]]
    det = (A.x - B.x) * (D.y - C.y) - (D.x - C.x) * (A.y - B.y)
    h1 = (B.y - C.y) * (D.x - C.x) - (B.x - C.x) * (D.y - C.y)
    h2 = (C.x - B.x) * (A.y - B.y) - (A.x - B.x) * (C.y - B.y)
    if abs(det) < 1e-15:
        return 0
    t1 = h1 / det
    t2 = h2 / det
    if (0 < t1 < 1) and (0 < t2 < 1):
        return 1
    else:
        return 0


def bound_condition(tr, bound, plist):
    inter_pt = bound.intersection(set(tr.points))
    seg = set(tr.points).difference(inter_pt)
    if len(inter_pt) == 1 and check_intersection(plist, bound, seg):
        return 1
    else:
        return 0

    pass


def point_in(tr, pt, plist):
    eps = 1e-10
    A = plist[tr.points[0]]
    B = plist[tr.points[1]]
    C = plist[tr.points[2]]
    u1 = Vector(A.y - B.y, B.x - A.x)
    u2 = Vector(A.y - C.y, C.x - A.x)
    u3 = Vector(C.y - B.y, B.x - C.x)
    if C.x * u1.x + C.y * u1.y > A.x * u1.x + A.y * u1.y:
        u1.x = -1 * u1.x
        u1.y = -1 * u1.y
    if B.x * u2.x + B.y * u2.y > A.x * u2.x + A.y * u2.y:
        u2.x = -1 * u2.x
        u2.y = -1 * u2.y
    if A.x * u3.x + A.y * u3.y > B.x * u3.x + B.y * u3.y:
        u3.x = -1 * u3.x
        u3.y = -1 * u3.y
    # print "vectors",  u1.x, u1.y, u2.x, u2.y,  u3.x, u3.y
    if (pt.x * u1.x + pt.y * u1.y <= A.x * u1.x + A.y * u1.y + eps) and (
            pt.x * u2.x + pt.y * u2.y <= A.x * u2.x + A.y * u2.y + eps) and (
            pt.x * u3.x + pt.y * u3.y <= B.x * u3.x + B.y * u3.y + eps):
        return 1
    else:
        return 0


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
    for adj in tr1.adjacent:
        if set(adj.points) == set(tr2.points):
            tr1.adjacent.remove(tr2)
            break


def swap_tr(tr1, tr2):
    inter = list(set(tr1.points).intersection(set(tr2.points)))
    diff = list(set(tr1.points).symmetric_difference(set(tr2.points)))
    tr1.points = tr1.points + diff
    tr2.points = tr2.points + diff
    tr1.points.append(inter[0])
    tr2.points.append(inter[1])
    del tr1.points[0:3]
    del tr2.points[0:3]
    for adj1 in tr1.adjacent:
        if len(set(adj1.points).intersection(set(tr1.points))) < 2:
            tr2.adjacent.append(adj1)
            tr1.adjacent.remove(adj1)
            adj1.adjacent.remove(tr1)
            adj1.adjacent.append(tr2)
            # print n1, l, triangle_list[l].adjacent
            break
    for adj2 in tr2.adjacent:
        if len(set(adj2.points).intersection(set(tr2.points))) < 2:
            tr1.adjacent.append(adj2)
            tr2.adjacent.remove(adj2)
            adj2.adjacent.remove(tr2)
            adj2.adjacent.append(tr1)
            # print n2, l, triangle_list[l].adjacent
            break


def plot_triangle(tr, plist, ax):
    if not tr.childs:
        vertx = tr.points
        vertx.append(tr.points[0])
        coordx = [plist[p].x for p in vertx]
        coordy = [plist[p].y for p in vertx]
        ax.plot(coordx, coordy, 'k-*')
    else:
        for child in tr.childs:
            plot_triangle(child, plist, ax)


class TriangleTree:
    def __init__(self, root=None):
        self.root = root

    def plot_mesh(self, plist):
        fig = plt.figure()  # create figure object
        ax = fig.add_subplot(1, 1, 1)  # create an axes object
        plt.gca().set_aspect('equal', adjustable='box')
        for child in self.root.childs:
            plot_triangle(child, plist, ax)
        plt.show()

    def search_triangle(self, fcond, *args):
        for child in self.root.childs:
            check = self.check_in_triangle(child, fcond, *args)
            if check:
                return check
            else:
                continue

    def check_in_triangle(self, tr, fcond, *args):
        if tr.childs:
            for child in tr.childs:
                check = self.check_in_triangle(child, fcond, *args)
                if check:
                    return check
                else:
                    continue
        elif fcond(tr, *args):
            return tr
        else:
            pass

    def enforce_boundary(self, boundary, plist):
        for bound in boundary:
            tr1 = self.search_triangle(bound_condition, bound, plist)
            if tr1:
                pt = set(tr1.points).intersection(bound)
                tr2 = tr1
                while bound != set(tr1.points).intersection(set(tr2.points)):
                    seg = set(tr1.points).difference(pt)
                    tr2 = [tr for tr in tr1.adjacent if len(seg.intersection(set(tr.points))) > 1][0]

        pass

    def insert_point(self, p, plist):
        pt = plist[p]
        tr = self.search_triangle(point_in, pt, plist)
        t1 = Triangle([p, tr.points[0], tr.points[1]], [], [])
        t2 = Triangle([p, tr.points[2], tr.points[0]], [], [])
        t3 = Triangle([p, tr.points[1], tr.points[2]], [], [])
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
        list_tmp = set(tr.adjacent)
        while list_tmp:
            tr_tmp = list_tmp.pop()
            if check_in_disc(pt, [plist[m] for m in tr_tmp.points]):
                list_tmp = list_tmp.union(set(tr_tmp.adjacent))
                tr_to_swap = [m for m in tr_tmp.adjacent if p in m.points][0]
                swap_tr(tr_tmp, tr_to_swap)


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
