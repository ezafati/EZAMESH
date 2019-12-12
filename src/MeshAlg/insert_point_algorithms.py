from globalvar import *


def add_point():
    pass


def chew_add_point(tree, plist):
    tr = tree.search_triangle(is_poor_quality, plist)
    pass


def collect_points(tr1, seg, r, pm, plist, n):
    tr2 = [tr for tr in tr1.adjacent if len(seg.intersection(set(tr1.points))) > 1][0]
    ps = [set(tr.points).difference(seg) for tr in (tr1, tr2)]
    list_points = [p for p in ps if length_segment(plist[p], pm) < r and p >= n]
    adj = tr1.adjacent.union(tr2.adjacent)
    while adj:
        tr = adj.pop()
        p = set(tr.points).difference(seg)
        if length_segment(plist[p], pm) < r and p >= n:
            list_points.append(p)
            adj = adj.union(tr.adjacent)


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
    A, B, C = [plist[p] for p in tr.points]
    l1 = length_segment(A, B)
    l2 = length_segment(B, C)
    l3 = length_segment(A, C)
    lmin = min(l1, l2, l3)
    radius = circumcircle_radius(tr, plist)
    try:
        test_ratio = radius / lmin
        if test_ratio >= cst:
            return 1
        else:
            return 0
    except TypeError:
        return 1
