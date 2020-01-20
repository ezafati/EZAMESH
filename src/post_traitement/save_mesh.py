import module_var


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


def save_ezamesh_file():
    plist = module_var.partmesh.listpoint
    with open(module_var.partmesh.savefile, 'w') as f:
        f.write(f'PART NAME: {module_var.partmesh.label}\n')
        f.write(f'NAMED POINTS: \n')
        for kpt in module_var.partmesh.pointlabel:
            pt = module_var.partmesh.pointlabel[kpt]
            f.write(f'{kpt}  {pt-1}\n')
        f.write(f'NAMED BOUNDARIES: \n')
        for kbound in module_var.partmesh.mapboundpts:
            ptlist = [str(p) for p in module_var.partmesh.mapboundpts[kbound]]
            f.write(f'{kbound}\n')
            sep = ','
            f.write(f'{sep.join(ptlist)}\n')
        f.write(f'POINT LIST\n')
        f.write(f'TOTAL: {len(plist)}\n')
        for l in range(len(plist)):
            f.write('{ct:d} {px:3f} {py:3f} \n'.format(ct=l, px=plist[l].x, py=plist[l].y))
        f.write(f'TOPOLOGY\n')
        f.write(f'TOTAL: {len(module_var.tree_refinement.root.childs)}\n')
        count = 1
        for tr in module_var.tree_refinement.root.childs:
            f.write('{ct:d} {p1:d} {p2:d} {p3:d} \n'.format(ct=count, p1=tr.points[0], p2=tr.points[1],
                                                            p3=tr.points[2]))
            count += 1
