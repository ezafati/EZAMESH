from __future__ import division

import itertools

import matplotlib.pyplot as plt
import numpy as np

from utils import *


def check_intersection(seg1, seg2):
    A, B = [p for p in seg1]
    C, D = [p for p in seg2]
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


def out_domain(tr, n):
    list_tmp = [p for p in tr.points if p >= n]
    return bool(list_tmp)


def bound_condition(tr, bound, plist):
    inter_pt = bound.intersection(set(tr.points))
    seg = set(tr.points).difference(inter_pt)
    seg_coord = [plist[p] for p in seg]
    bound_coord = [plist[p] for p in bound]
    return len(inter_pt) == 1 and check_intersection(bound_coord, seg_coord)


def point_in(tr, pt, plist):
    eps = 1e-10
    A, B, C = [plist[p] for p in tr.points]
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
    return (pt.x * u1.x + pt.y * u1.y <= A.x * u1.x + A.y * u1.y + eps) and (
            pt.x * u2.x + pt.y * u2.y <= A.x * u2.x + A.y * u2.y + eps) and (
                   pt.x * u3.x + pt.y * u3.y <= B.x * u3.x + B.y * u3.y + eps)


def check_in_disc(pt, lpt):
    A, B, C = lpt
    a = np.array(([B.x - A.x, C.x - A.x], [B.y - A.y, C.y - A.y]))
    array1 = [A.x - pt.x, A.y - pt.y, pow(A.x - pt.x, 2) + pow(A.y - pt.y, 2)]
    array2 = [B.x - pt.x, B.y - pt.y, pow(B.x - pt.x, 2) + pow(B.y - pt.y, 2)]
    array3 = [C.x - pt.x, C.y - pt.y, pow(C.x - pt.x, 2) + pow(C.y - pt.y, 2)]
    b = np.array((array1, array2, array3))
    return np.linalg.slogdet(a)[0] * np.linalg.slogdet(b)[0] > 0


def remove_triangle(tr1, tr2):
    for adj in tr1.adjacent:
        if adj.points == tr2.points:
            tr1.adjacent.remove(tr2)
            break


def swap_tr(tr1, tr2):
    inter = set(tr1.points).intersection(set(tr2.points))
    diff = set(tr1.points).symmetric_difference(set(tr2.points))
    tr1.points = tr1.points + [pt for pt in diff]
    tr2.points = tr2.points + [pt for pt in diff]
    tr1.points.append(inter.pop())
    tr2.points.append(inter.pop())
    del tr1.points[0:3]
    del tr2.points[0:3]
    for adj1 in tr1.adjacent:
        if len(set(adj1.points).intersection(set(tr1.points))) < 2:
            tr2.adjacent.add(adj1)
            tr1.adjacent.remove(adj1)
            adj1.adjacent.remove(tr1)
            adj1.adjacent.add(tr2)
            break
    for adj2 in tr2.adjacent:
        if len(set(adj2.points).intersection(set(tr2.points))) < 2:
            tr1.adjacent.add(adj2)
            tr2.adjacent.remove(adj2)
            adj2.adjacent.remove(tr2)
            adj2.adjacent.add(tr1)
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
            if isinstance(child, Triangle):
                plot_triangle(child, plist, ax)


def insert_point(p, plist, tr):
    pt = plist[p]
    list_tr = map(lambda l: Triangle([p, l[0], l[1]]), itertools.combinations(tr.points, 2))
    list_tr = list(list_tr)
    tr.childs.extend(list_tr)
    for tri in list_tr:
        tri.adjacent, = [set(l) for l in itertools.combinations(list_tr, 2) if tri not in l]
    for adj in tr.adjacent:
        remove_triangle(adj, tr)
        for child in tr.childs:
            if len(set(child.points).intersection(set(adj.points))) > 1:
                child.adjacent.add(adj)
                adj.adjacent.add(child)
                break
    list_tmp = tr.adjacent
    while list_tmp:
        tr_tmp = list_tmp.pop()
        if check_in_disc(pt, [plist[m] for m in tr_tmp.points]):
            list_tmp = list_tmp.union(tr_tmp.adjacent)
            tr_to_swap, = [m for m in tr_tmp.adjacent if p in m.points]
            swap_tr(tr_tmp, tr_to_swap)


class TriangleTree:
    def __init__(self, root=None):
        self.root = root

    @classmethod
    def triangle_tree_refinement(cls, tree):
        tree_tmp = cls(Triangle())
        for child in tree.root.childs:
            tree_tmp.add_child(child)
        return tree_tmp

    def add_child(self, tr):
        if tr.childs:
            for child in tr.childs:
                if isinstance(child, Triangle):
                    self.add_child(child)
        else:
            self.root.childs.append(tr)

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

    def get_initial_constrained_mesh(self, boundary, plist, n):
        for p in range(n):
            pt = plist[p]
            tr = self.search_triangle(point_in, pt, plist)
            insert_point(p, plist, tr)
        self.enforce_boundary(boundary, plist)
        tr = self.search_triangle(out_domain, n)
        tr.childs.append(-1)
        list_tmp = tr.adjacent.copy()
        while list_tmp:
            tr_tmp = list_tmp.pop()
            for adj in tr_tmp.adjacent:
                seg = set(tr_tmp.points).intersection(set(adj.points))
                if -1 in adj.childs and seg not in boundary:
                    tr_tmp.childs.append(-1)
                    list_tmp = list_tmp.union({m for m in tr_tmp.adjacent if -1 not in m.childs})
                    break
        self.eliminate_extra_triangles(self.root)

    def eliminate_extra_triangles(self, tr):
        if tr.childs:
            for child in tr.childs:
                if isinstance(child, Triangle):
                    self.eliminate_extra_triangles(child)
                else:
                    for adj in tr.adjacent:
                        adj.adjacent.remove(tr)

    def enforce_boundary(self, boundary, plist):
        for bound in boundary:
            bound_coord = [plist[p] for p in bound]
            tr1 = self.search_triangle(bound_condition, bound, plist)
            if tr1:
                pt = set(tr1.points).intersection(bound)
                tr2 = tr1
                while bound != set(tr1.points).intersection(set(tr2.points)):
                    seg = set(tr2.points).difference(pt)
                    seg_coord = [plist[p] for p in seg]
                    if check_intersection(bound_coord, seg_coord):
                        tr1, tr2 = tr2, tr1
                    else:
                        sys.exit('FATAL ERROR!')
                    tr2, = [tr for tr in tr1.adjacent if len(seg.intersection(set(tr.points))) > 1]
                    swap_tr(tr1, tr2)


class Vector(object):
    def __init__(self, x=None, y=None):
        self.x = x
        self.y = y


class Point(object):
    def __init__(self, x=None, y=None):
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
    def __init__(self, x=None, y=None, z=None):
        if x is None:
            x = []
        if y is None:
            y = []
        if z is None:
            z = set()
        self.points = x
        self.childs = y
        self.adjacent = z


class Segment(object):
    def __init__(self, x=0, y=0, dens=0):
        self.extrA = x
        self.extrB = y
        self.dens = dens


class Mesh(object):
    def __init__(self, x=None, y=0, slabel=None, nnodes=0, polist=None, llist=None, size=None, ltri=None, terminate=0):
        if ltri is None:
            ltri = []
        if size is None:
            size = []
        if llist is None:
            llist = {}
        if polist is None:
            polist = []
        if slabel is None:
            slabel = []
        if x is None:
            x = []
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
