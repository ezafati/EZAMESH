""" Geometry and mesh Objects
Copyright (c) 2019-2020, E Zafati
 All rights reserved"""
from __future__ import division

import itertools

from collections import deque
from functools import reduce
from typing import Type, Dict, List, Set, Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
from math import sqrt, acos, modf, sin, cos, copysign
import sys
import logging

import scipy.integrate
import scipy.optimize

from systemutils import AlreadyExistError, NotApproValueError, UnknownElementError


def prim_spline(A: 'Point', B: 'Point', C: 'Point') -> Callable[[float], float]:
    v1 = Vector(C.x - A.x, C.y - A.y)
    v2 = Vector(B.x - C.x, B.y - C.y)
    nv1 = v1.x * v1.x + v1.y * v1.y
    nv2 = v2.x * v2.x + v2.y * v2.y
    dtv1v2 = v1.x * v2.x + v1.y * v2.y
    return lambda x: 2 * sqrt((1 - x) ** 2 * nv1 + x ** 2 * nv2 + 2 * x * (1 - x) * dtv1v2)


def len_spline(t: float, a: 'Point', b: 'Point', c: 'Point') -> float:
    func = np.vectorize(prim_spline(a, b, c))
    result, *_ = scipy.integrate.fixed_quad(lambda x: func(x), 0, t, n=3)
    return result


def get_center(seg: List['Point'], radius: float) -> 'Point':
    A, B = seg
    H = Point(1 / 2 * (A.x + B.x), 1 / 2 * (A.y + B.y))
    dist2 = (A.x - B.x) ** 2 + (A.y - B.y) ** 2
    alpha = abs(radius ** 2 - dist2 / 4)
    center = Point()
    try:
        ratio = (A.y - B.y) / (A.x - B.x)
        beta = 1 + ratio ** 2
        if B.x > A.x:
            center.y = H.y + sqrt(alpha / beta)
        else:
            center.y = H.y - sqrt(alpha / beta)
        center.x = H.x - (center.y - H.y) * ratio
    except ZeroDivisionError:
        ratio = (A.x - B.x) / (A.y - B.y)
        beta = 1 + ratio ** 2
        if B.y > A.y:
            center.x = H.x - sqrt(alpha / beta)
        else:
            center.x = H.x + sqrt(alpha / beta)
        center.y = H.y - (center.x - H.x) * ratio
    return center


def check_intersection(seg1: Iterable['Point'], seg2: Iterable['Point']) -> bool:
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


def out_domain(tr: 'Triangle', n: int) -> bool:
    list_tmp = [p for p in tr.points if p >= n]
    return bool(list_tmp)


def bound_condition(tr: 'Triangle', bound: Set[int], plist: 'List[Point]') -> bool:
    inter_pt = bound.intersection(set(tr.points))
    seg = set(tr.points).difference(inter_pt)
    seg_coord = [plist[p] for p in seg]
    bound_coord = [plist[p] for p in bound]
    return len(inter_pt) == 1 and check_intersection(bound_coord, seg_coord)


def point_in(tr: 'Triangle', pt: 'Point', plist: List['Point']) -> bool:
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


def check_in_disc(pt: 'Point', lpt: Iterable['Point']) -> bool:
    A, B, C = lpt
    a = np.array(([B.x - A.x, C.x - A.x], [B.y - A.y, C.y - A.y]))
    array1 = [A.x - pt.x, A.y - pt.y, pow(A.x - pt.x, 2) + pow(A.y - pt.y, 2)]
    array2 = [B.x - pt.x, B.y - pt.y, pow(B.x - pt.x, 2) + pow(B.y - pt.y, 2)]
    array3 = [C.x - pt.x, C.y - pt.y, pow(C.x - pt.x, 2) + pow(C.y - pt.y, 2)]
    b = np.array((array1, array2, array3))
    return np.linalg.slogdet(a)[0] * np.linalg.slogdet(b)[0] > 0


def remove_triangle(tr1: 'Triangle', tr2: 'Triangle'):
    for adj in tr1.adjacent:
        if adj.points == tr2.points:
            tr1.adjacent.remove(tr2)
            break


def swap_tr(tr1: 'Triangle', tr2: 'Triangle'):
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


def plot_triangle(tr: 'Triangle', plist: 'List[Point]', ax: 'AxesSubplot'):
    if not tr.childs:
        vertx = tr.points
        vertx.append(tr.points[0])
        coordx = [plist[p].x for p in vertx]
        coordy = [plist[p].y for p in vertx]
        ax.plot(coordx, coordy, 'k-')
    else:
        for child in tr.childs:
            if isinstance(child, Triangle):
                plot_triangle(child, plist, ax)


def insert_point(p: int, plist: List[int], tr: 'Triangle'):
    pt = plist[p]
    list_tr = map(lambda l: Triangle([p, l[0], l[1]]), itertools.combinations(tr.points, 2))
    list_tr = list(list_tr)
    for tri in list_tr:
        tri.parent = tr.parent
        tri.adjacent, = [set(l) for l in itertools.combinations(list_tr, 2) if tri not in l]
    for adj in tr.adjacent:
        remove_triangle(adj, tr)
        for child in list_tr:
            if len(set(child.points).intersection(set(adj.points))) > 1:
                child.adjacent.add(adj)
                adj.adjacent.add(child)
                break
    dq = deque(tr.adjacent)
    tr.parent.childs.extend(list_tr)
    tr.parent.childs.remove(tr)
    while dq:
        tr_tmp = dq.pop()
        if check_in_disc(pt, [plist[m] for m in tr_tmp.points]):
            for tri in tr_tmp.adjacent:
                if tri not in dq:
                    dq.append(tri)
            tr_to_swap, = [m for m in tr_tmp.adjacent if p in m.points]
            swap_tr(tr_tmp, tr_to_swap)


class TriangleTree:
    def __init__(self, root=None):
        self.root = root
        self.terminate = False

    @classmethod
    def _triangle_tree_refinement(cls, tree):
        tree_tmp = cls(Triangle())
        for child in tree.root.childs:
            tree_tmp._add_child(child)
        return tree_tmp

    def _add_child(self, tr):
        if tr.childs:
            for child in tr.childs:
                if isinstance(child, Triangle):
                    self._add_child(child)
        else:
            self.root.childs.append(tr)
            tr.parent = self.root

    def plot_mesh(self, plist):
        fig = plt.figure()  # create figure object
        ax = fig.add_subplot(1, 1, 1)  # create an axes object
        plt.gca().set_aspect('equal', adjustable='box')
        for child in self.root.childs:
            plot_triangle(child, plist, ax)
        plt.show()

    def _search_triangle(self, fcond, *args):
        for child in self.root.childs:
            check = self._check_in_triangle(child, fcond, *args)
            if check:
                return check
            else:
                continue

    def _check_in_triangle(self, tr, fcond, *args):
        if tr.childs:
            for child in tr.childs:
                check = self._check_in_triangle(child, fcond, *args)
                if check:
                    return check
                else:
                    continue
        elif fcond(tr, *args):
            return tr
        else:
            pass

    def _get_initial_constrained_mesh(self, boundary, plist, n, process):
        for p in range(n):
            pt = plist[p]
            tr = self._search_triangle(point_in, pt, plist)
            insert_point(p, plist, tr)
        self._enforce_boundary(boundary, plist)
        tr = self._search_triangle(out_domain, n)
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
        self._eliminate_extra_triangles(self.root)

    def _eliminate_extra_triangles(self, tr):
        if tr.childs:
            for child in tr.childs:
                if isinstance(child, Triangle):
                    self._eliminate_extra_triangles(child)
                else:
                    for adj in tr.adjacent:
                        adj.adjacent.remove(tr)

    def _enforce_boundary(self, boundary, plist):
        for bound in boundary:
            bound_coord = [plist[p] for p in bound]
            tr1 = self._search_triangle(bound_condition, bound, plist)
            if tr1:
                pt = set(tr1.points).intersection(bound)
                tr2 = tr1
                while bound != set(tr1.points).intersection(set(tr2.points)):
                    seg = set(tr2.points).difference(pt)
                    seg_coord = [plist[p] for p in seg]
                    if check_intersection(bound_coord, seg_coord):
                        tr1, tr2 = tr2, tr1
                    else:
                        logging.error('EXIT WITH ERROR IN BOUNDARY')
                        raise Exception('FATAL ERROR!')
                    tr2, = [tr for tr in tr1.adjacent if len(seg.intersection(set(tr.points))) > 1]
                    swap_tr(tr1, tr2)


class FileParser(object):
    def __init__(self):
        self.points = dict()
        self.segments = dict()
        self.parts = dict()

    def make_mesh(self, fields: List[str], n_line: int):
        label = fields[3]
        try:
            assert label in self.parts
        except UnknownElementError:
            raise UnknownElementError(f'Element {label} provided in line {n_line} is not defined') from None
        self.parts[label].create = True

    def make_part(self, fields: List[str], n_line: int):
        label = fields[0]
        try:
            assert label not in self.parts
        except Exception:
            raise AlreadyExistError(f'The label {label} is provided more than once') from None
        part = Part()
        part.name = label
        part.listboundary = fields[3:]
        self.parts[label] = part

    def make_point(self, fields: List[str], n_line: int):
        label = fields[0]
        try:
            assert label not in self.points
        except Exception:
            raise AlreadyExistError(f'The label {label} is provided more than once') from None
        try:
            pt = Point(float(fields[3]), float(fields[4]))
        except ValueError:
            raise ValueError(f'the entries in line {n_line} should be flaots ') from None
        self.points[label] = pt

    def make_boundary(self, fields: List[str], n_line: int):
        label = fields[0]
        try:
            assert label not in self.segments
        except:
            raise AlreadyExistError(f'The label {label} is provided more than once') from None
        bound = BoundElement()
        try:
            bound.type = fields[2]
            try:
                assert fields[3] in self.points and fields[4] in self.points
                bound.extr = (fields[3], fields[4])
            except SyntaxError:
                raise SyntaxError(f'At least one point is not defined  in line {n_line}')
            bound.sizes = (float(fields[5]), float(fields[6]))
        except (IndexError, ValueError) as e:
            print(e)
        try:
            bound.extrargs = fields[7:]
        except IndexError:
            pass
        self.segments[label] = bound


class Part(object):
    def __init__(self):
        self.name = None
        self.listboundary = None
        self.create = False

    def create_mesh(self, parserfile: Type[FileParser]):
        mesh = Mesh()
        ptlist = reduce(lambda p, q: p | q, (set(parserfile.segments[label].extr) for label in self.listboundary))
        mesh.add_point(ptlist, parserfile.points)
        for label in self.listboundary:
            seg = parserfile.segments[label]
            eval(f'mesh.add_{seg.type}'.lower() + f'(seg)')
        return mesh


class Vector(object):
    def __init__(self, x=None, y=None):
        self.x = x
        self.y = y


class BoundElement(object):
    def __init__(self):
        self.type = None
        self.extr = None
        self.sizes = None
        self.extrargs = None


class Point(object):
    __slots__ = ('x', 'y', 'size')

    def __init__(self, x=None, y=None, size=None):
        self.x = x
        self.y = y
        self.size = size

    def prescale(self, x, y, d):
        self.x = (self.x - x) / d
        self.y = (self.y - y) / d
        return self

    def postscale(self, xmin, ymin, dmax):
        self.x = dmax * self.x + xmin
        self.y = dmax * self.y + ymin
        return self

    def __add__(self, other):
        if isinstance(other, Point):
            return Point(self.x + other.x, self.y + other.y)
        else:
            raise TypeError('The two instances should be the type Point')

    def __sub__(self, other):
        if isinstance(other, Point):
            return Point(self.x - other.x, self.y - other.y)
        else:
            raise TypeError('The two instances should be the type Point')

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Point(other * self.x, other * self.y)
        else:
            raise TypeError('One of the  instances should be of the type int or float')

    def __rmul__(self, other):
        return self * other


class Triangle(object):
    __slots__ = ('points', 'childs', 'adjacent', 'parent', 'visited')

    def __init__(self, x=None, y=None, z=None, par=None, vis=False):
        if x is None:
            x = []
        if y is None:
            y = []
        if z is None:
            z = set()
        self.points = x
        self.childs = y
        self.adjacent = z
        self.parent = par
        self.visited = vis


class Mesh(object):
    def __init__(self):
        """constructor with initial attribute values:
        self.boundary = [] (all the boundray elements)
        self.nbnodes = 0   (total of the boundray points)
        self.listpoint = []  (mesh point list)
        self.pointlabel = dict()  (dict mapping label to a point object)
        self.triangle_list = []  (list contains tuples NA,NB,NC)
        self.meshstrategy = 'default' (mesh strategy default==second Chew algorithm)"""
        self.boundary = []
        self.nbnodes = 0
        self.listpoint = list()
        self.pointlabel = dict()
        self.triangle_list = []
        self.meshstrategy = 'default'
        self.save = True
        self.savefile = 'test.txt'

    def add_arc(self, seg: Type[BoundElement]):
        """add_spline(obj, seg) add a discretized
        circle arc boundary to the mesh"""
        NA, NB = [self.pointlabel[p] for p in seg.extr]
        l1, l2 = [dens for dens in seg.sizes]
        A, B = [self.listpoint[p - 1] for p in (NA, NB)]
        point_list = [A, B]
        try:
            radius = float(seg.extrargs[0])
        except (IndexError, ValueError) as e:
            raise Exception(f'Maybe the radius is not provided or the value cannot be converted to float'
                            f'for arc {seg.extr}') from None
        center = get_center(point_list, radius)
        scalar_product = (A.x - center.x) * (B.x - center.x) + (A.y - center.y) * (B.y - center.y)
        theta = acos((scalar_product / radius ** 2))
        slen = radius * theta
        if l1 > l2:
            l1, l2 = l2, l1
            A, B = B, A
            NA, NB = NB, NA
            theta = -1 * theta
        ratio = slen / (l1 + (l2 - l1) / 2)
        if ratio < 1:
            raise ValueError(
                f"The densities specified for the arc {seg.extr} are too large for the boundary length {slen}") \
                from None
        estep = modf(ratio)
        if estep[0] > 0.5:  # add correction to l1 and l2 for better subdivision
            corr = 1 / 2 * (l1 + l2 - 2 * slen / (estep[1] + 1))
            l1 -= corr
            l2 -= corr
            tsteps = estep[1] + 1
        else:
            corr = 1 / 2 * (2 * slen / estep[1] - l1 - l2)
            l1 += corr
            l2 += corr
            tsteps = estep[1]
        if tsteps == 1:  # no points to add
            self.boundary.append({NA - 1, NB - 1})
            return 1
        step = (l2 - l1) / estep[1]
        count = 1
        while count < tsteps:
            self.nbnodes += 1
            C = Point()
            theta_M = (count * l1 + count * (count - 1) * step / 2) / radius * copysign(1, theta)
            C.x = center.x + cos(theta_M) * (A.x - center.x) - sin(theta_M) * (A.y - center.y)
            C.y = center.y + sin(theta_M) * (A.x - center.x) + cos(theta_M) * (A.y - center.y)
            C.size = l1 + (count - 1) * step
            self.listpoint.append(C)
            if count == tsteps - 1:
                self.boundary.append({self.nbnodes - 1, NB - 1})
                if count == 1:
                    self.boundary.append({NA - 1, self.nbnodes - 1})
                else:
                    self.boundary.append({(self.nbnodes - 1) - 1, self.nbnodes - 1})
            elif count == 1:
                self.boundary.append({NA - 1, self.nbnodes - 1})
            else:
                self.boundary.append({(self.nbnodes - 1) - 1, self.nbnodes - 1})
            count += 1

    def add_line(self, seg: Type[BoundElement]):
        """add_spline(obj, seg) add a discretized
        line or a segment boundary to the mesh"""
        NA, NB = [self.pointlabel[p] for p in seg.extr]
        l1, l2 = [dens for dens in seg.sizes]
        if l1 > l2:
            NA, NB = NB, NA
            l1, l2 = l2, l1
        A, B = [self.listpoint[p - 1] for p in (NA, NB)]
        A.size, B.size = l1, l2
        slen = sqrt(pow(A.x - B.x, 2) + pow(A.y - B.y, 2))
        ratio = slen / (l1 + (l2 - l1) / 2)
        if ratio < 1:
            raise ValueError(
                f"The densities specified for the line {seg.extr} are too large for the boundary length {slen}") \
                from None
        estep = modf(ratio)
        if estep[0] > 0.5:  # add correction to l1 and l2 for get a better subdivision
            corr = 1 / 2 * (l1 + l2 - 2 * slen / (estep[1] + 1))
            l1 -= corr
            l2 -= corr
            tsteps = estep[1] + 1
        else:
            corr = 1 / 2 * (2 * slen / estep[1] - l1 - l2)
            l1 += corr
            l2 += corr
            tsteps = estep[1]
        if tsteps == 1:  # no points to add
            self.boundary.append({NA - 1, NB - 1})
            return 1
        step = (l2 - l1) / estep[1]
        count = 1
        while count < tsteps:
            self.nbnodes += 1
            C = Point()
            C.x = A.x + (B.x - A.x) / slen * (count * l1 + count * (count - 1) * step / 2)
            C.y = A.y + (B.y - A.y) / slen * (count * l1 + count * (count - 1) * step / 2)
            C.size = l1 + (count - 1) * step
            self.listpoint.append(C)
            if count == tsteps - 1:
                self.boundary.append({self.nbnodes - 1, NB - 1})
                if count == 1:
                    self.boundary.append({NA - 1, self.nbnodes - 1})
                else:
                    self.boundary.append({(self.nbnodes - 1) - 1, self.nbnodes - 1})
            elif count == 1:
                self.boundary.append({NA - 1, self.nbnodes - 1})
            else:
                self.boundary.append({(self.nbnodes - 1) - 1, self.nbnodes - 1})
            count += 1

    def add_spline(self, seg: Type[BoundElement]):
        """add_spline(obj, seg) add a discretized
        spline boundary to the mesh"""
        NA, NB = [self.pointlabel[p] for p in seg.extr]
        l1, l2 = [dens for dens in seg.sizes]
        cpt = Point()
        try:
            cpt.x, cpt.y = [float(p) for p in seg.extrargs[0:2]]
        except (IndexError, ValueError) as e:
            print(e)
        A, B = [self.listpoint[p - 1] for p in (NA, NB)]
        A.size, B.size = l1, l2
        if l2 < l1:
            NA, NB = NB, NA
            A, B = B, A
            l1, l2 = l2, l1
        slen = len_spline(1, A, B, cpt)
        fprim = prim_spline(A, B, cpt)
        ratio = slen / (l1 + (l2 - l1) / 2)
        if ratio < 1:
            raise ValueError(
                f"The densities specified ifor the spline {seg.extr} are too large for the boundary length {slen}") \
                from None
        estep = modf(ratio)
        if estep[0] > 0.5:  # add correction to l1 and l2 for better subdivision
            corr = 1 / 2 * (l1 + l2 - 2 * slen / (estep[1] + 1))
            l1 -= corr
            l2 -= corr
            tsteps = estep[1] + 1
        else:
            corr = 1 / 2 * (2 * slen / estep[1] - l1 - l2)
            l1 += corr
            l2 += corr
            tsteps = estep[1]
        if tsteps == 1:  # no points to add
            self.boundary.append({NA - 1, NB - 1})
            return 1
        step = (l2 - l1) / estep[1]
        count = 1
        while count < tsteps:
            self.nbnodes += 1
            C = Point()
            target = (count * l1 + count * (count - 1) * step / 2)
            rt = scipy.optimize.newton(lambda x: len_spline(x, A, B, cpt) - target, 0, fprime=fprim)
            p11 = Point((1 - rt) * A.x + rt * cpt.x, (1 - rt) * A.y + rt * cpt.y)
            p21 = Point((1 - rt) * cpt.x + rt * B.x, (1 - rt) * cpt.y + rt * B.y)
            C.x = (1 - rt) * p11.x + rt * p21.x
            C.y = (1 - rt) * p11.y + rt * p21.y
            C.size = l1 + (count - 1) * step
            self.listpoint.append(C)
            if count == tsteps - 1:
                self.boundary.append({self.nbnodes - 1, NB - 1})
                if count == 1:
                    self.boundary.append({NA - 1, self.nbnodes - 1})
                else:
                    self.boundary.append({(self.nbnodes - 1) - 1, self.nbnodes - 1})
            elif count == 1:
                self.boundary.append({NA - 1, self.nbnodes - 1})
            else:
                self.boundary.append({(self.nbnodes - 1) - 1, self.nbnodes - 1})
            count += 1

    def add_point(self, label_pts: List[str], list_pts: Dict[str, Type[FileParser]]):
        for label in label_pts:
            self.nbnodes += 1
            self.listpoint.append(list_pts[label])
            self.pointlabel[label] = self.nbnodes
