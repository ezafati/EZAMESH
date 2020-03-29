from typing import Tuple, Dict
from MeshObjects.GeObjects import Point


def sort_vtk_file(gmesh):
    size = 3  # to generalize after
    f = open("./sort/mesh.vtk", "w")
    f.write("# vtk DataFile Version 4.2\n")
    f.write("mesh details \n")
    f.write("ASCII \n")
    f.write("\n")
    f.write("DATASET UNSTRUCTURED_GRID \n")
    f.write("POINTS {:d} {type}\n".format(len(gmesh.point_list), type="float"))
    for i in range(len(gmesh.point_list)):
        f.write("{:6.4f} {:6.4f} {:6.4f}\n".format(gmesh.point_list[i].x, gmesh.point_list[i].y, 0))
    f.write("\n")
    f.write("CELLS {:d} {:d} \n".format(len(gmesh.triangle_list), (size + 1) * len(gmesh.triangle_list)))
    for i in range(len(gmesh.triangle_list)):
        f.write("{:d} {:d} {:d} {:d} \n".format(size, gmesh.triangle_list[i][0], gmesh.triangle_list[i][1],
                                                gmesh.triangle_list[i][2]))
    f.write("\n")
    f.write("CELL_TYPES {:d}\n".format(len(gmesh.triangle_list)))
    for i in range(len(gmesh.triangle_list)):
        f.write("{type} \n".format(type=5))
    f.close()


def save_ezamesh_file(partmesh, tree_refinement):
    plist = partmesh.listpoint
    with open(partmesh.savefile, 'a+') as f:
        f.write(f'PART_NAME {partmesh.label}\n')
        f.write(f'ELEMENT_TYPE TRI3\n')
        f.write(f'NAMED_POINTS \n')
        f.write(f'TOTAL: {len(partmesh.pointlabel)}\n')
        for kpt in partmesh.pointlabel:
            pt = partmesh.pointlabel[kpt]
            f.write(f'{kpt}  {pt - 1}\n')
        f.write(f'NAMED_BOUNDARIES \n')
        f.write(f'TOTAL: {len(partmesh.mapboundpts)}\n')
        for kbound in partmesh.mapboundpts:
            ptlist = [str(p) for p in partmesh.mapboundpts[kbound]]
            f.write(f'{kbound}\n')
            sep = ','
            f.write(f'{sep.join(ptlist)}\n')
        f.write(f'POINT_LIST\n')
        f.write(f'TOTAL: {len(plist)}\n')
        for l in range(len(plist)):
            f.write('{ct:d} {px:3f} {py:3f} \n'.format(ct=l, px=plist[l].x, py=plist[l].y))
        f.write(f'TOPOLOGY\n')
        f.write(f'TOTAL: {len(tree_refinement.root.childs)}\n')
        count = 1
        for tr in tree_refinement.root.childs:
            f.write('{ct:d} {p1:d} {p2:d} {p3:d} \n'.format(ct=count, p1=tr.points[0], p2=tr.points[1],
                                                            p3=tr.points[2]))
            count += 1
    create_tri6_mesh(partmesh=partmesh, tree_refinement=tree_refinement)
    with open(partmesh.savefile, 'a+') as f:
        f.write(f'PART_NAME {partmesh.label}\n')
        f.write(f'ELEMENT_TYPE TRI6\n')
        f.write(f'NAMED_POINTS \n')
        f.write(f'TOTAL: {len(partmesh.pointlabel)}\n')
        for kpt in partmesh.pointlabel:
            pt = partmesh.pointlabel[kpt]
            f.write(f'{kpt}  {pt - 1}\n')
        f.write(f'NAMED_BOUNDARIES \n')
        f.write(f'TOTAL: {len(partmesh.mapboundpts)}\n')
        for kbound in partmesh.mapboundpts:
            ptlist = [str(p) for p in partmesh.mapboundpts[kbound]]
            f.write(f'{kbound}\n')
            sep = ','
            f.write(f'{sep.join(ptlist)}\n')
        f.write(f'POINT_LIST\n')
        f.write(f'TOTAL: {len(plist)}\n')
        for l in range(len(plist)):
            f.write('{ct:d} {px:3f} {py:3f} \n'.format(ct=l, px=plist[l].x, py=plist[l].y))
        f.write(f'TOPOLOGY\n')
        f.write(f'TOTAL: {len(tree_refinement.root.childs)}\n')
        count = 1
        for tr in tree_refinement.root.childs:
            f.write(
                '{ct:d} {p1:d} {p2:d} {p3:d} {p4:d} {p5:d} {p6:d}\n'.format(ct=count, p1=tr.points[0], p2=tr.points[1],
                                                                            p3=tr.points[2], p4=tr.points[3],
                                                                            p5=tr.points[4],
                                                                            p6=tr.points[5]))
            count += 1


def create_tri6_mesh(partmesh, tree_refinement):
    plist = partmesh.listpoint
    init_tr = tree_refinement.root.childs[0]
    insert_mid_point(tr=init_tr, plist=plist, partmesh=partmesh)


def check_seg_in_boundary(seg: Tuple, listbound: Dict):
    for key in listbound:
        if all([seg[0] in listbound[key], seg[1] in listbound[key]]):
            return key


def insert_mid_point(tr, plist, partmesh):
    size = len(tr.points)
    seg_list = []
    for l in range(size - 1):
        seg_list.append((tr.points[l], tr.points[l + 1]))
    seg_list.append((tr.points[0], tr.points[size - 1]))
    for seg in seg_list:
        key = check_seg_in_boundary(seg=seg, listbound=partmesh.mapboundpts)
        if key:
            A = plist[seg[0]]
            B = plist[seg[1]]
            midp = Point(1 / 2 * (A.x + B.x), 1 / 2 * (A.y + B.y))
            plist.append(midp)
            partmesh.mapboundpts[key].append(len(plist) - 1)
            insert_mid_point_triangle(tr=tr, seg=set(seg), npt=len(plist) - 1)
        else:
            adj = [p for p in tr.adjacent if len(set(p.points) & set(seg)) > 1][0]
            if not adj.visited:
                A = plist[seg[0]]
                B = plist[seg[1]]
                midp = Point(1 / 2 * (A.x + B.x), 1 / 2 * (A.y + B.y))
                plist.append(midp)
                insert_mid_point_triangle(tr=tr, seg=set(seg), npt=len(plist) - 1)
                insert_mid_point_triangle(tr=adj, seg=set(seg), npt=len(plist) - 1)
    tr.visited = True
    for adj in tr.adjacent:
        if not adj.visited:
            insert_mid_point(tr=adj, plist=plist, partmesh=partmesh)


def insert_mid_point_triangle(tr, seg, npt):
    pts = tr.points
    for p in range(len(pts) - 1):
        if seg == {pts[p], pts[p + 1]}:
            tr.points.insert(p + 1, npt)
            return 1
    tr.points.append(npt)
