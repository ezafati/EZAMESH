import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from MeshObjects.GeObjects import *
from MeshAlg.insert_point_algorithms import add_point

matplotlib.use("TkAgg")
import tkinter as tk


def dt_initial(vmesh):
    plist = vmesh.point_list
    boundary = vmesh.boundary
    nl = len(plist)
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
    TreeRefinement = TriangleTree().triangle_tree_refinement(Tree)
    TreeRefinement.plot_mesh(plist)
    Tree = None