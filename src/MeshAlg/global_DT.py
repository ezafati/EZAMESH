import matplotlib
import importlib

import module_var
from MeshObjects.GeObjects import *
from module_var import dispatcher


matplotlib.use("TkAgg")


def dt_global(vmesh, process):
    _module = importlib.import_module(f"MeshAlg.{dispatcher[vmesh.meshstrategy][0]}")
    refinement_method = _module.__dict__[dispatcher[vmesh.meshstrategy][1]]
    plist = vmesh.listpoint
    boundary = vmesh.boundary
    nl = len(plist)
    logging.info(f'DOMAIN WITH {nl} POINTS IN BOUNDARY')
    xmax = max([plist[p].x for p in range(nl)])
    xmin = min([plist[p].x for p in range(nl)])
    ymax = max([plist[p].y for p in range(nl)])
    ymin = min([plist[p].y for p in range(nl)])
    dmax = max((xmax - xmin), (ymax - ymin))
    plist = [plist[p].prescale(xmin, ymin, dmax) for p in range(nl)]
    plist.append(Point(-0.5, -0.5))
    plist.append(Point(1.5, -0.5))
    plist.append(Point(1.5, 1.5))
    plist.append(Point(-0.5, 1.5))
    T1 = Triangle([nl, nl + 1, nl + 2])
    T2 = Triangle([nl + 2, nl + 3, nl])
    T1.adjacent.add(T2)
    T2.adjacent.add(T1)
    # initialize Tree
    Tree = TriangleTree(Triangle())
    Tree.root.childs = Tree.root.childs + [T1, T2]
    for tr in Tree.root.childs:
        tr.parent = Tree.root
    Tree._get_initial_constrained_mesh(boundary, plist, nl, process)
    del plist[nl:]
    module_var.tree_refinement = TriangleTree()._triangle_tree_refinement(Tree)
    plist = [plist[p].postscale(xmin, ymin, dmax) for p in range(nl)]
    vmesh.listpoint = plist
    del Tree
    count = 0
    while not module_var.tree_refinement.terminate:
        if count % 10 == 0:
            #logging.info(f'Memory infos: {process.memory_info()}')
            #logging.info(f'CPU used percentage: {process.cpu_percent()}')
            pass
        refinement_method(module_var.tree_refinement, plist, nl)
        count += 1
    logging.info(f'MESH GENERATED WITH {len(plist)} POINTS')
    module_var.tree_refinement.plot_mesh(plist)
