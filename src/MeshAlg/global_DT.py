import matplotlib
import importlib
from MeshObjects.GeObjects import Point, Triangle, TriangleTree
from module_var import dispatcher
import logging

matplotlib.use("TkAgg")


def dt_global(vmesh):
    _module = importlib.import_module(f"MeshAlg.{dispatcher[vmesh.meshstrategy][0]}")
    refinement_method = _module.__dict__[dispatcher[vmesh.meshstrategy][1]]
    plist = vmesh.point_list
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
    Tree.get_initial_constrained_mesh(boundary, plist, nl)
    del plist[nl:]
    TreeRefinement = TriangleTree().triangle_tree_refinement(Tree)
    plist = [plist[p].postscale(xmin, ymin, dmax) for p in range(nl)]
    del Tree
    while not TreeRefinement.terminate:
        refinement_method(TreeRefinement, plist, nl)
    logging.info(f'MESH GENERATED WITH {len(plist)} POINTS')
    logging.info('######################## END PROCESS: SUCCESS ##################')
    TreeRefinement.plot_mesh(plist)
