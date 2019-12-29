import time

import matplotlib
import importlib

import module_var
from MeshObjects.GeObjects import *
from module_var import dispatcher
from multiprocessing import Process, JoinableQueue, Value

from MeshAlg.chew_insert_algorithm import worker


def dt_global(vmesh, process):
    _module = importlib.import_module(f"MeshAlg.{dispatcher[vmesh.meshstrategy].module_name}")
    refinement_method = _module.__dict__[dispatcher[vmesh.meshstrategy].mesh_func]
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
    if __name__ == 'MeshAlg.global_DT':
        ratio = Value('d', 0.0, lock=False)
        num = Value('i', len(module_var.tree_refinement.root.childs), lock=False)
        nbprocess = 2
        while not module_var.tree_refinement.terminate:
            task_queue = JoinableQueue()
            for _ in range(nbprocess):
                Process(target=worker, args=(task_queue,  ratio, num)).start()
            if count % 10 == 0:
                # logging.info(f'Memory infos: {process.memory_info()}')
                # logging.info(f'CPU used percentage: {process.cpu_percent()}')
                pass
            refinement_method(module_var.tree_refinement, plist, nl, task_queue, num)
            count += 1
            for _ in range(nbprocess):
                task_queue.put('STOP')
            num.value = len(module_var.tree_refinement.root.childs)
            ratio.value = 0.0
        logging.info(f'MESH GENERATED WITH {len(plist)} POINTS')
        module_var.tree_refinement.plot_mesh(plist)
