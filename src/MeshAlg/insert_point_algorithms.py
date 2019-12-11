from globalvar import *


def add_point():
    pass


def chew_add_point(tree, plist):
    tr = tree.search_triangle(is_poor_quality, plist)
    pass


def length_segment(a, b):
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2))


def circumcircle_radius(tr, plist):
    A, B, C = [plist[p] for p in tr.points]
    l1 = length_segment(A, B)
    l2 = length_segment(B, C)
    l3 = length_segment(A, C)
    num = l1 * l2 * l3
    dom = sqrt((l1 + l2 + l3) * (l2 + l3 - l1) * (l3 + l1 - l2) * (l1 + l2 - l3))
    try:
        radius = num / dom
    except ZeroDivisionError as e:
        print('Fatal error while computing the radius of the circumcirle: ', e)
        sys.exit()
    return radius


def is_poor_quality(tr, plist):
    cst = sqrt(2)
    A, B, C = [plist[p] for p in tr.points]
    l1 = length_segment(A, B)
    l2 = length_segment(B, C)
    l3 = length_segment(A, C)
    lmin = min(l1, l2, l3)
    radius = circumcircle_radius(tr, plist)
    test_ratio = radius / lmin
    if test_ratio >= cst:
        return 1
    else:
        return 0
