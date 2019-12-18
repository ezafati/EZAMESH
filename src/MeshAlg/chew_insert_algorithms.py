from globalvar import *
from functools import reduce
import itertools


def insert_midpoint(p, tr, seg):
    """function that insert the midpoint in the mesh"""
    list_tr = map(lambda l: Triangle([p, l[0], l[1]]), itertools.combinations(tr.points, 2))
    list_tr = list(list_tr)
    tr.childs.extend(list_tr)
    for tri in list_tr:
        tri.parent = tr
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


def seek_triangle(tr, seg, plist, list_check):
    """ seek for the triangle used to link midpoint and
    the deleted point"""
    for adj in tr.adjacent:
        if adj not in list_check:
            list_check.append(adj)
            if bound_condition(adj, seg, plist) or seg.issubset(set(adj.points)):
                return seg
            else:
                seek_triangle(adj, seg, plist, list_check)


def enforce_segment(tr, index, p, plist):
    """Enforce the segment linking the midpoint and a deleted point seen from
    the first one"""
    seg = {p, index}
    tr1 = False
    for child in tr.childs:
        if bound_condition(child, seg, plist) or seg.issubset(set(child.points)):
            tr1 = child
            break
    if not tr1:
        list_check = list()
        for child in tr.childs:
            check = seek_triangle(child, seg, plist, list_check)
            if check:
                tr1 = check
    try:
        pt = set(tr1.points).intersection(seg)
        if len(pt) == 1:
            tr2 = tr1
            seg_coord = [plist[p] for p in seg]
            while seg != set(tr1.points).intersection(set(tr2.points)):
                seg2 = set(tr2.points).difference(pt)
                seg2_coord = [plist[p] for p in seg2]
                if check_intersection(seg_coord, seg2_coord):
                    tr1, tr2 = tr2, tr1
                else:
                    sys.exit('FATAL ERROR!')
                tr2, = [tri for tri in tr1.adjacent if len(seg2.intersection(set(tri.points))) > 1]
                swap_tr(tr1, tr2)
        else:
            tr2, = [tri for tri in tr1.adjacent if len(seg.intersection(set(tri.points))) > 1]
        return tr1, tr2
    except AttributeError:
        sys.exit('FATAL ERROR ! Maybe the triangle has an empty child list')


def replace_vertex(p, index, tr, list_tr):
    """replace each deleted vertex returned by
    collect_point method by the added midpoint"""
    for adj in tr.adjacent:
        if p in adj.points:
            list_tr.add(adj)
            adj.points.remove(p)
            adj.points.append(index)
            replace_vertex(p, index, adj, list_tr)


def chew_add_point(tree, plist, nl):
    """Chew method to add a new point in the domain"""
    l_tr = [0, 0]
    tree.search_triangle(is_well_shaped, plist, l_tr)
    tr = l_tr[1]
    if not tr:
        tree.search_triangle(is_well_sized, plist, l_tr, nl)
        tr = l_tr[1]
    if tr:
        pt = circumcircle_center(tr, plist)
        if not point_in_adjacent(tr, pt, plist):
            seg, tr1 = find_segment(tr, pt, plist)
            pm = Point()
            plist.append(pm)
            seg = set(seg)
            p1, *_ = [plist[p] for p in seg]
            pm.x = 1 / 2 * reduce(lambda l, m: l + m, [plist[q].x for q in seg])
            pm.y = 1 / 2 * reduce(lambda l, m: l + m, [plist[q].y for q in seg])
            radius = length_segment(pm, p1)
            list_tmp = collect_points(tr, seg, radius, pm, plist, nl)
            index = len(plist) - 1
            insert_midpoint(index, tr1, seg)
            if list_tmp:
                for p in list_tmp:
                    list_new_tris = set()
                    list_tri_elim = enforce_segment(tr1, index, p, plist)
                    for tri in list_tri_elim:
                        replace_vertex(p, index, tri, list_new_tris)
                    for tri in list_tri_elim:
                        tri.parent.childs.remove(tri)
                        for adj in tri.adjacent:
                            adj.adjacent.remove(tri)
                    adj_triangles = [pl for pl in itertools.combinations(list_new_tris, 2) if
                                     len(set(pl[0].points).intersection(pl[1].points)) > 1]
                    for trk, trj in adj_triangles:
                        if trk not in trj.adjacent:
                            trk.adjacent.add(trj)
                            trj.adjacent.add(trk)
        else:
            plist.append(pt)
            adj = point_in_adjacent(tr, pt, plist)
            insert_point(len(plist) - 1, plist, adj)
    else:
        tree.terminate = True
        print('NO MORE SKINNY TRIANGLE IS FOUND')


def point_in_adjacent(tr, pt, plist):
    """check if the triangle tr or  one of its adjacents
    contains the circumcenter pt"""
    if point_in(tr, pt, plist):
        return tr
    for adj in tr.adjacent:
        if point_in(adj, pt, plist):
            return adj


def collect_points(tr1, seg, r, pm, plist, n):
    """Collect the set of points seen from
    the circumcenter added as the midpoint"""
    tr2, = filter(lambda tr: len(seg.intersection(set(tr.points))) > 1, tuple(tr1.adjacent))
    ps = [set(tr.points).difference(seg).pop() for tr in (tr1, tr2)]
    list_points = set([p for p in ps if length_segment(plist[p], pm) < r and p >= n])
    adj = tr1.adjacent.union(tr2.adjacent).difference({tr1, tr2})
    while adj:
        tr = adj.pop()
        p = set(tr.points).difference(seg)
        for pt in p:
            if length_segment(plist[pt], pm) < r and pt >= n and pt not in list_points:
                list_points.add(pt)
                adj = adj.union(tr.adjacent)
    return list_points


def find_segment(tr, pt, plist):
    """Find the segment separating the circumcenter and
    the failed triangle (poorly sized or poorly shaped)"""
    mass_cent = Point()
    mass_cent.x = 1 / 3 * reduce(lambda l, m: l + m, [plist[q].x for q in tr.points])
    mass_cent.y = 1 / 3 * reduce(lambda l, m: l + m, [plist[q].x for q in tr.points])
    segt = [mass_cent, pt]
    for adj in tr.adjacent:
        try:
            seg_tmp, = filter(lambda seg: check_intersection((plist[seg[0]], plist[seg[1]]), segt),
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
    f = 1 / 2 * (A.x * A.x + A.y * A.y - B.x * B.x - B.y * B.y)
    g = 1 / 2 * (B.x * B.x + B.y * B.y - C.x * C.x - C.y * C.y)
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


def is_well_shaped(tr, plist, r_tr):
    """This function check if the current triangle is well shaped with
    respect to the previous tested one r_tr[1] """
    cst = 1 * sqrt(2)
    list_tmp = [plist[p] for p in tr.points]
    lmin = min([length_segment(p, q) for p, q in itertools.combinations(list_tmp, 2)])
    radius = circumcircle_radius(tr, plist)
    try:
        test_ratio = radius / lmin
        pt = circumcircle_center(tr, plist)
        booli = find_segment(tr, pt, plist) or point_in_adjacent(tr, pt, plist)
        if test_ratio >= cst and booli and radius > r_tr[0]:
            r_tr[0], r_tr[1] = radius, tr
    except TypeError:
        pass


def is_well_sized(tr, plist, ratio_tr, nl):
    """This function check if the current triangle is well shaped with
    respect to the previous tested one ratio_tr[1] """
    radius = circumcircle_radius(tr, plist)
    pt = circumcircle_center(tr, plist)
    h = size_function(pt, plist, nl)
    try:
        booli = find_segment(tr, pt, plist) or point_in_adjacent(tr, pt, plist)
        ratio = radius / h
        if radius > h and booli and ratio > ratio_tr[0]:
            ratio_tr[0], ratio_tr[1] = ratio, tr
    except TypeError:
        pass


def size_function(pt, plist, nl):
    """evaluate the size function at the point pt"""
    g = 0.07
    size = min([plist[l].size + g * length_segment(pt, plist[l]) for l in range(nl)])
    return size
