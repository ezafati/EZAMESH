import itertools
import module_var
from MeshObjects.GeObjects import Point
from scipy.optimize import minimize, least_squares
import numpy as np


def optmize_mesh(func, *args):
    nnodes = len(module_var.gmesh.listpoint)
    xinit = np.ndarray(shape=(2 * nnodes,))
    plist = module_var.gmesh.listpoint
    for l in range(nnodes):
        xinit[2 * l] = plist[l].x
        xinit[2 * l + 1] = plist[l].y
    const = {'type': 'eq', 'fun': constr_func}
    opt = {'disp': True, 'maxiter': 100}
    result = minimize(energy_based_func, xinit, args=(func, plist, *args), method='SLSQP', constraints=const,
                      options=opt)
    return result


def optmize_mesh_lsq(func, *args):
    nnodes = len(module_var.gmesh.listpoint)
    xinit = np.ndarray(shape=(2 * nnodes,))
    plist = module_var.gmesh.listpoint
    for l in range(nnodes):
        xinit[2 * l] = plist[l].x
        xinit[2 * l + 1] = plist[l].y
    result = least_squares(energy_based_func_lsq, xinit, args=(func, plist, *args), method='dogbox', verbose=2, xtol=1e-04, jac='3-point')
    return result


def energy_based_func_lsq(xlist, func, plist, *args):
    ntr = len(module_var.tree_refinement.root.childs)
    nnodes = len(module_var.gmesh.listpoint)
    nedges = ntr + nnodes - 1
    shape = nedges - module_var.gmesh.nbnodes
    vect = np.ndarray(shape=(2 * shape,))
    count = 0
    for tr in module_var.tree_refinement.root.childs:
        for seg in itertools.combinations(tr.points, 2):
            if set(seg) not in module_var.gmesh.boundary:
                p1, p2 = [Point(xlist[2 * l], xlist[2 * l + 1]) for l in seg]
                pt = p1 - p2
                pm = (p1 + p2) * 0.5
                vect[count] = pt.x ** 2 + pt.y ** 2 - func(pm, plist, *args) ** 2
                count += 1
    return vect


def energy_based_func(xlist, func, plist, *args):
    energy = 0
    for tr in module_var.tree_refinement.root.childs:
        for seg in itertools.combinations(tr.points, 2):
            p1, p2 = [Point(xlist[2 * l], xlist[2 * l + 1]) for l in seg]
            pt = p1 - p2
            pm = (p1 + p2) * 0.5
            if set(seg) in module_var.gmesh.boundary:
                energy += 2 * (pt.x ** 2 + pt.y ** 2 - func(pm, plist, *args) ** 2) ** 2
            else:
                energy += (pt.x ** 2 + pt.y ** 2 - func(pm, plist, *args) ** 2) ** 2
    return energy


def constr_func(xlist):
    nbnodes = module_var.gmesh.nbnodes
    plist = module_var.gmesh.listpoint
    constr_sum = 0
    for l in range(nbnodes):
        constr_sum += (xlist[2 * l] - plist[l].x) ** 2 + (xlist[2 * l + 1] - plist[l].y) ** 2
    return constr_sum
