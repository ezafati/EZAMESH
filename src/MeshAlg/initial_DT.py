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
    Nl = len(plist)
    xmax = max([plist[p].x for p in range(Nl)])
    xmin = min([plist[p].x for p in range(Nl)])
    ymax = max([plist[p].y for p in range(Nl)])
    ymin = min([plist[p].y for p in range(Nl)])
    dmax = max((xmax - xmin), (ymax - ymin))
    plist = [plist[p].prescale(xmin, ymin, dmax) for p in range(Nl)]
    plist.append(Point(-0.5, -0.5))
    plist.append(Point(1.5, -0.5))
    plist.append(Point(1.5, 1.5))
    plist.append(Point(-0.5, 1.5))
    T1 = Triangle([Nl, Nl + 1, Nl + 2], [], [])
    T2 = Triangle([Nl + 2, Nl + 3, Nl], [], [])
    T1.adjacent.append(T2)
    T2.adjacent.append(T1)
    # initialize Tree
    Tree = TriangleTree(Triangle())
    Tree.root.childs = Tree.root.childs + [T1, T2]
    for p in range(Nl):
        Tree.insert_point(p, plist)
    Tree.plot_mesh(plist)
