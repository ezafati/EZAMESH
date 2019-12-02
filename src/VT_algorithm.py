#import pdb; pdb.set_trace() 
from class_DT import *
from DT_algorithm import *
import math, sys
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt 

def VT_algo(mesh, vmesh):
    vmesh.v_add_grid(mesh,"rand")
    #sys.exit()
    plist = mesh.point_list[:]
    boundary = mesh.boundary[:]
    Nl = len(plist)
    xmax = max([ plist[p].x for p in range(Nl)])
    xmin = min([ plist[p].x for p in range(Nl)])
    ymax = max([ plist[p].y for p in range(Nl)])
    ymin = min([ plist[p].y for p in range(Nl)])
    dmax = max ((xmax-xmin),(ymax-ymin))
    #print plist[0].x, plist[0].y
    #print xmin, ymin, dmax
    plist = [ plist[p].prescale(xmin,ymin,dmax) for p in range(Nl)]
    #print plist[0].x,plist[0].y
    #sys.exit() 
    plist.append(point(-0.5,-0.5))
    plist.append(point(1.5,-0.5))
    plist.append(point(1.5,1.5))
    plist.append(point(-0.5,1.5))
    Ntot= Nl+4
    triangle_list=[]
    # Creation des super triangles
    triangle_list.append(triangle([Nl, Nl+1, Nl+2],[],[1]))
    triangle_list.append(triangle([Nl+2, Nl+3, Nl],[], [0]))
    Nt =2
    for p in range(Nl):
        print "point", p, "is inserted"
        Nel = find_triangle(plist[p], plist,  triangle_list)
        k = insert_point(p, plist,  Nel, Nt,  triangle_list)
        Nt = Nt +k
    triangle_list_final = {}
    for p in range(len(triangle_list)):
        if  triangle_list[p].childs == []:
            triangle_list_final[p] = triangle_list[p]
    boundary_enforc(plist, triangle_list, triangle_list_final, boundary)
    #print plist[0].x, plist[0].y
    #print xmin, xmax, dmax
    plist = [ plist[p].inv_prescale(xmin,ymin,dmax) for p in range(Ntot)]
    triangle_vertex = {}
    l = 0
    # add vertices for each point 
    for p in range(Nl):
        vmesh.cell_list.append([])
        for k in triangle_list_final.keys():
            if p in triangle_list_final[k].points:
               break
        if k not in triangle_vertex.keys():
           cent = cent_coord(triangle_list_final[k].points, plist)
           vmesh.vertex_list.append(cent)
           triangle_vertex[k] = l
           l += 1
        vmesh.cell_list[p].append(triangle_vertex[k])
        test = 1
        while test:
              for m in triangle_list_final[k].adjacent:
                  test = 0 
                  if p in triangle_list_final[m].points and (m not in triangle_vertex.keys() or triangle_vertex[m] not in vmesh.cell_list[p]):
                     if m not in triangle_vertex.keys():
                        cent = cent_coord(triangle_list_final[m].points, plist)
                        vmesh.vertex_list.append(cent)
                        triangle_vertex[m] = l
                        l += 1
                     vmesh.cell_list[p].append(triangle_vertex[m])
                     test = 1
                     break 
              k = m
    # end of the loop " add vertices" 
    plot_VT(vmesh.vertex_list, vmesh.cell_list, plist, triangle_list_final) 

    triangle_list_elim (triangle_list_final, boundary, Nl, Ntot)
    for k in triangle_list_final.keys():
          if  -1 not in triangle_list_final[k].childs:
               mesh.triangle_list.append(triangle_list_final[k].points)
    """Renumerotation code: list points and related numerotations are updated """
    set_point_tmp = {}
    mesh.point_list = []
    cell_list_tmp = vmesh.cell_list[:]
    vmesh.cell_list = []
    n = 0
    for el in mesh.triangle_list:
        for k in range(len(el)):
            if el[k] not in set_point_tmp.keys():
               mesh.point_list.append(plist[el[k]])
               vmesh.cell_list.append(cell_list_tmp[el[k]])
               set_point_tmp[el[k]] = n
               n += 1
            el[k] = set_point_tmp[el[k]]
    #plot_VT(vmesh.vertex_list, vmesh.cell_list, plist, triangle_list_final)
    """to use correctly the 2 following  commads the previous renumerotation code should be commented""" 
    #plot_mesh(plist, triangle_list_final)
    #plot_DT(plist, triangle_list)

def cent_coord(tr_points, plist): 
    A =  plist[tr_points[0]]
    B =  plist[tr_points[1]]
    C =  plist[tr_points[2]]
    D = 2*(A.x*(B.y-C.y)+B.x*(C.y-A.y)+C.x*(A.y-B.y))
    Ux = ((A.x**2+A.y**2)*(B.y-C.y)+(B.x**2+B.y**2)*(C.y-A.y)+(C.x**2+C.y**2)*(A.y-B.y))/D
    Uy = ((A.x**2+A.y**2)*(C.x-B.x)+(B.x**2+B.y**2)*(A.x-C.x)+(C.x**2+C.y**2)*(B.x-A.x))/D
    return point(Ux,Uy)
 
def plot_VT(vertex_list, cell_list, plist, triangle_list_final):
    fig = plt.figure() # create figure object
    ax = fig.add_subplot(1,1,1) # create an axes object
    plt.gca().set_aspect('equal', adjustable='box')
    k=0
    while (k <= len(cell_list)-1):
          coordx =[vertex_list[p].x for p in cell_list[k]]+[vertex_list[cell_list[k][0]].x]
          coordy =[vertex_list[p].y for p in cell_list[k]]+[vertex_list[cell_list[k][0]].y]
          ax.plot(coordx, coordy, 'k-')
          k += 1
    for k in triangle_list_final.keys():
          coordx =[plist[p].x for p in triangle_list_final[k].points]+[plist[triangle_list_final[k].points[0]].x]
          coordy =[plist[p].y for p in triangle_list_final[k].points]+[plist[triangle_list_final[k].points[0]].y]
          ax.plot(coordx, coordy, 'r-')

    plt.show()

