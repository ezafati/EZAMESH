import importlib
from multiprocessing import JoinableQueue, Value

import module_var
from MeshObjects.GeObjects import *
from systemutils import launch_processes, worker
from module_var import dispatcher


def run_tri_mesh(vmesh: 'Mesh', process: 'Process'):
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
        params, cache_value = list(), list()
        for param in dispatcher[vmesh.meshstrategy].init_params:
            params.append(Value(param['type'], param['val'], lock=False))
            cache_value.append(param['val'])
        while not module_var.tree_refinement.terminate:
            task_queue = JoinableQueue()
            with launch_processes(task_queue, worker, *params):
                if count % 10 == 0:
                    # logging.info(f'Memory infos: {process.memory_info()}')
                    # logging.info(f'CPU used percentage: {process.cpu_percent()}')
                    pass
                refinement_method(module_var.tree_refinement, plist, nl, task_queue, *params)
                count += 1
            for index in range(len(params)):
                params[index].value = cache_value[index]
        logging.info(f'MESH GENERATED WITH {len(plist)} POINTS')
        module_var.tree_refinement.plot_mesh(plist)
