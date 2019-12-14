from globalvar import *
from functools import reduce
import itertools


def add_point():
    pass


def insert_midpoint(p,  tr, seg):
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
    for adj in tr.adjacent:
        try:
            tr_to_swap, = [p for p in adj.adjacent if seg.issubset(set(p.points))]
            swap_tr(adj, tr_to_swap)
        except ValueError:
            pass


def chew_add_point(tree, plist, nl):
    tr = tree.search_triangle(is_poor_quality, plist)
    pt = circumcircle_center(tr, plist)
    if not point_in_adjacent(tr, pt, plist):
        seg, tr1 = find_segment(tr, pt, plist)
        pm = Point()
        plist.append(pm)
        seg = set(seg)
        p1, *_ = [plist[p] for p in seg]
        pm.x = 1 / 2 * reduce(lambda l, m: plist[l].x + plist[m].x, seg)
        pm.y = 1 / 2 * reduce(lambda l, m: plist[l].y + plist[m].y, seg)
        radius = length_segment(pm, p1)
        list_tmp = collect_points(tr, seg, radius, pm, plist, nl)
        insert_midpoint(len(plist), tr1, seg)
        if list_tmp:
            for p in list_tmp:
                pass
    else:
        adj = point_in_adjacent(tr, pt, plist)
        insert_point(pt, plist, adj)


def point_in_adjacent(tr, pt, plist):
    for adj in tr.adjacent:
        if point_in(adj, pt, plist):
            return adj


def collect_points(tr1, seg, r, pm, plist, n):
    tr2, = filter(lambda tr: len(seg.intersection(set(tr.points))) > 1, tuple(tr1.adjacent))
    ps = [set(tr.points).difference(seg).pop() for tr in (tr1, tr2)]
    list_points = [p for p in ps if length_segment(plist[p], pm) < r and p >= n]
    adj = tr1.adjacent.union(tr2.adjacent)
    while adj:
        tr = adj.pop()
        p = set(tr.points).difference(seg)
        if length_segment(plist[p], pm) < r and p >= n:
            list_points.append(p)
            adj = adj.union(tr.adjacent)
    return list_points


def find_segment(tr, pt, plist):
    mass_cent = Point()
    mass_cent.x = 1 / 3 * reduce(lambda l, m: plist[l].x + plist[m].x, tr.points)
    mass_cent.y = 1 / 3 * reduce(lambda l, m: plist[l].y + plist[m].y, tr.points)
    segt = [mass_cent, pt]
    for adj in tr.adjacent:
        try:
            seg_tmp, = filter(lambda seg: check_intersection((seg[0], seg[1]), segt),
                              itertools.combinations(adj.points, 2))
            if len(set(tr.points).intersection(set(seg_tmp))) > 1:
                pass
            else:
                return seg_tmp, adj
        except ValueError:
            pass


def length_segment(a, b):
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2))


def circumcircle_center(tr, plist):
    A, B, C = [plist[p] for p in tr.points]
    f = 1 / 2 * (A.x * A.x - B.x * B.y)
    g = 1 / 2 * (B.x * B.y - C.x * C.Y)
    det = (A.x - B.x) * (B.y - C.y) - (B.x - C.x) * (A.y - B.y)
    center = Point()
    try:
        center.x = ((B.y - C.y) * f - (A.y - B.y) * g) / det
        center.y = ((A.x - B.x) * g - (B.x - C.x) * f) / det
        return center
    except ZeroDivisionError:
        return 0


def circumcircle_radius(tr, plist):
    A, B, C = [plist[p] for p in tr.points]
    l1 = length_segment(A, B)
    l2 = length_segment(B, C)
    l3 = length_segment(A, C)
    num = l1 * l2 * l3
    dom = sqrt((l1 + l2 + l3) * (l2 + l3 - l1) * (l3 + l1 - l2) * (l1 + l2 - l3))
    try:
        radius = num / dom
        return radius
    except ZeroDivisionError:
        print('Very skinny triangle found with parallel segments ')
        return None


def is_poor_quality(tr, plist):
    cst = sqrt(2)
    list_tmp = [plist[p] for p in tr.points]
    lmin = min([length_segment(p, q) for p, q in itertools.combinations(list_tmp, 2)])
    radius = circumcircle_radius(tr, plist)
    try:
        test_ratio = radius / lmin
        pt = circumcircle_center(tr, plist)
        booli = find_segment(tr, pt, plist) or point_in_adjacent(tr, pt, plist)
        return test_ratio >= cst and booli
    except TypeError:
        return 1
