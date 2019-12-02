from MeshObjects.GeObjects import *
import math, sys
import matplotlib.pyplot as plt
import numpy as np

def DT_algo(mesh):
    #mesh.add_grid("rand")
    plist = mesh.point_list[:]
    boundary = mesh.boundary[:]
    Nl = len(plist)
    xmax = max([ plist[p].x for p in range(Nl)])
    xmin = min([ plist[p].x for p in range(Nl)])
    ymax = max([ plist[p].y for p in range(Nl)])
    ymin = min([ plist[p].y for p in range(Nl)])
    dmax = max ((xmax-xmin),(ymax-ymin))
    #print xmin, ymin, dmax
    plist = [ plist[p].prescale(xmin,ymin,dmax) for p in range(Nl)]
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
    triangle_list_elim (triangle_list_final, boundary, Nl, Ntot)
    #plot_DT(plist, triangle_list)
    for k in triangle_list_final.keys():
          if  -1 not in triangle_list_final[k].childs:
               mesh.triangle_list.append(triangle_list_final[k].points)
    plist = [ plist[p].inv_prescale(xmin,ymin,dmax) for p in range(Nl)]
    """Renumerotation code: list points and related numerotations are updated """
    set_point_tmp = {}
    mesh.point_list = []
    n = 0
    for el in mesh.triangle_list:
        for k in range(len(el)):
            if el[k] not in set_point_tmp.keys():
               mesh.point_list.append(plist[el[k]])
               set_point_tmp[el[k]] = n
               n += 1
            el[k] = set_point_tmp[el[k]]
    """to use correctly the 2 following  commads the previous renumerotation code should be commented""" 
    
    plot_mesh(mesh.point_list, mesh.triangle_list)


def triangle_list_elim(triangle_list_final, boundary, Nl, Ntot):
    for p in triangle_list_final.keys():
        if   len(set(range(Nl, Ntot)).intersection(set(triangle_list_final[p].points))) >=1:
             break
    triangle_list_final[p].childs.append(-1)
    adjacent = triangle_list_final[p].adjacent[:]
    while adjacent != []:
          Ntmp= adjacent[0]
          del  adjacent[0]
          for l in triangle_list_final[Ntmp].adjacent:
              seg_inter = set(triangle_list_final[Ntmp].points).intersection(set(triangle_list_final[l].points))
              if  (-1 in triangle_list_final[l].childs) and (seg_inter not in boundary):
                  triangle_list_final[Ntmp].childs.append(-1)
                  list_tmp = [m for m in triangle_list_final[Ntmp].adjacent if -1 not in triangle_list_final[m].childs]
                  adjacent = list(set(adjacent).union(set(list_tmp)))
                  break


def boundary_enforc(plist, triangle_list, triangle_list_final,  boundary): 
    for l in boundary:
        #print "boundary", l
        for n1 in triangle_list_final.keys():
               points_set= l.intersection(set(triangle_list[n1].points))
               if len(points_set) >=1 and (len(points_set) == 2  or check_intersection(plist, l, set(triangle_list[n1].points).difference(points_set))):
                     break
 
        if len(points_set) ==1:
           print "the following boundary element is enforced", l 
           n2 = n1
           while  l != set(triangle_list[n1].points).intersection(set(triangle_list[n2].points)):
                  #print triangle_list[n1].points, triangle_list[n2].points
                  inter_seg = set(triangle_list[n1].points).difference(points_set)
                  if check_intersection(plist, l, set(triangle_list[n1].points).difference(points_set)):
                        n2 =  [m for m in triangle_list[n1].adjacent if len(inter_seg.intersection(set(triangle_list[m].points)))>1][0]
                        #print n1, n2, triangle_list[n1].adjacent
                  elif check_intersection(plist, l, set(triangle_list[n2].points).difference(points_set)): 
                        n1 = n2
                        inter_seg = set(triangle_list[n1].points).difference(points_set)
                        n2 =  [m for m in triangle_list[n1].adjacent if len(inter_seg.intersection(set(triangle_list[m].points)))>1][0]
                  else:
                        sys.exit("fatal error: maybe the chosen  boundary element is union of other boundary elements")
                  #print "before swap", triangle_list[n1].points, triangle_list[n2].points
                  swapping_triangles(triangle_list, n1, n2)

def check_intersection(plist, seg1 , seg2 ):
    #print seg1, seg2
    seg1 = list(seg1)
    seg2 = list(seg2)
    A = plist[seg1[0]]
    B = plist[seg1[1]]
    C = plist[seg2[0]]
    D = plist[seg2[1]]
    det =  (A.x-B.x)*(D.y-C.y)-(D.x-C.x)*(A.y-B.y)
    h1  =  (B.y-C.y)*(D.x-C.x)-(B.x-C.x)*(D.y-C.y)
    h2  =  (C.x-B.x)*(A.y-B.y)- (A.x-B.x)*(C.y-B.y)
    #print "determinant est", det
    if abs(det) < 1e-15:
       return 0
       #sys.exit("fatal error: determinant close to zero: the error should be reported")
    t1 = h1/det
    t2 = h2/det 
    #print h1, h2, t1, t2
    if (0 < t1 < 1) and  (0 < t2 < 1):
       return 1
    else:
       return 0



def insert_point(p, plist, Nel, Nt,  triangle_list):
    list_del=[]
    # initialization of the first triangles in the triangle with index Nel
    triangle_list.append(triangle([p, triangle_list[Nel].points[0], triangle_list[Nel].points[1]], [], [Nt+1, Nt+2]))
    triangle_list.append(triangle([p, triangle_list[Nel].points[0], triangle_list[Nel].points[2]],[], [Nt, Nt+2]))
    triangle_list.append(triangle([p, triangle_list[Nel].points[1], triangle_list[Nel].points[2]], [], [Nt, Nt+1]))
    print "triangle with reference ", Nel, " generate 3 triangles", [triangle_list[l].points for l in range(Nt,Nt+3)]
    k = 3
    list_del.append(Nel)
    for l in triangle_list[Nel].adjacent: 
        #print Nel, l
        triangle_list[l].adjacent.remove(Nel)
        for t in range(Nt, Nt+k): 
            if len(set(triangle_list[l].points).intersection(set(triangle_list[t].points))) > 1:
               triangle_list[t].adjacent.append(l)
               triangle_list[l].adjacent.append(t)
               break
    list_adja =  triangle_list[Nel].adjacent[:]
    print "list of adjacent triangle before loop while", list_adja
    triangle_list[Nel].adjacent = []
    while list_adja != []:
          print "list adjacent triangle after loop while", list_adja
          Ntmp = list_adja[0]
          del list_adja[0]
          print "one triangle is deleted"
          if check_in_circle(plist[p],[plist[l] for l in triangle_list[Ntmp].points]):
             list_del.append(Ntmp)
             print "reference of the triangle deleted with satisfied condition",  Ntmp
             k += 1
             triangle_list.append(triangle(triangle_list[Ntmp].points, triangle_list[Ntmp].childs, triangle_list[Ntmp].adjacent))
             list_adja = list(set(list_adja).union(set([m for m in triangle_list[Ntmp].adjacent if p not in triangle_list[m].points])))
             print len([m for m in triangle_list[Ntmp].adjacent if p not in triangle_list[m].points]), "new triangles added to list adj"
             Np =  [m  for m in triangle_list[Ntmp].adjacent if p in triangle_list[m].points][0]
             triangle_list[Ntmp].adjacent = []
             #print "adjac Nt+k-1", triangle_list[Nt+k-1].adjacent
             for l in triangle_list[Nt+k-1].adjacent:
                 #print "Nt+k-1", Nt+k-1
                 triangle_list[l].adjacent.remove(Ntmp)
                 triangle_list[l].adjacent.append(Nt+k-1)
             print "triangles before swapping", [triangle_list[l].points for l in [Np, Nt+k-1]]
             swapping_triangles(triangle_list, Nt+k-1, Np)
             print "triangles after swapped", triangle_list[Nt+k-1].points, triangle_list[Np].points
    for n in list_del:
        #print "element elimine", n
        triangle_list[n].childs=range(Nt, Nt+k)
    #print "childs", range(Nt, Nt+k)
    return k 


def swapping_triangles(triangle_list, n1, n2):
    inter = list(set(triangle_list[n1].points).intersection(set(triangle_list[n2].points)))
    diff   = list(set(triangle_list[n1].points).symmetric_difference(set(triangle_list[n2].points)))
    triangle_list[n1].points = triangle_list[n1].points + diff
    triangle_list[n2].points = triangle_list[n2].points + diff
    triangle_list[n1].points.append(inter[0])
    triangle_list[n2].points.append(inter[1])
    del triangle_list[n1].points[0:3]
    del triangle_list[n2].points[0:3]
    for l in  triangle_list[n1].adjacent:
        if  len(set(triangle_list[l].points).intersection(set(triangle_list[n1].points))) < 2:
             triangle_list[n2].adjacent.append(l)
             triangle_list[n1].adjacent.remove(l)
             triangle_list[l].adjacent.remove(n1)
             triangle_list[l].adjacent.append(n2)
             #print n1, l, triangle_list[l].adjacent
             break
    for l in  triangle_list[n2].adjacent:
        if  len(set(triangle_list[l].points).intersection(set(triangle_list[n2].points))) < 2:
             triangle_list[n1].adjacent.append(l)
             triangle_list[n2].adjacent.remove(l)
             triangle_list[l].adjacent.remove(n2)
             triangle_list[l].adjacent.append(n1)
             #print n2, l, triangle_list[l].adjacent
             break
    #print triangle_list[n1].points, triangle_list[n1].adjacent, triangle_list[n2].points, triangle_list[n2].adjacent    
    

def check_in_circle(p, lp): 
    A = lp[0]
    B = lp[1]
    C = lp[2]
    a = np.array(([B.x-A.x, C.x-A.x],[B.y-A.y, C.y-A.y]))
    array1= [A.x-p.x, A.y-p.y, pow(A.x-p.x,2)+pow(A.y-p.y,2)]
    array2= [B.x-p.x, B.y-p.y, pow(B.x-p.x,2)+pow(B.y-p.y,2)]
    array3= [C.x-p.x, C.y-p.y, pow(C.x-p.x,2)+pow(C.y-p.y,2)]
    b = np.array((array1,array2,array3))
    if  np.linalg.slogdet(a)[0]*np.linalg.slogdet(b)[0]>0:
        return 1
    else: 
        return 0


def find_triangle(p, plist, triangle_list):
    #find the parent 
    Nel = 1
    #print p.x, p.y
    #print [[plist[triangle_list[0].points[l]].x, plist[triangle_list[0].points[l]].y]  for l in range(3)]
    if  check_in_triangle(p, [plist[l] for l in triangle_list[0].points]):
       Nel = 0
    childs = triangle_list[Nel].childs
    #print "Nel est", Nel
    #print "childs", childs
    while childs != []:
          test = 1
          k = 0
          while test and (k <= len(childs)-1):
               #print [[plist[triangle_list[childs[k]].points[l]].x, plist[triangle_list[childs[k]].points[l]].y]  for l in range(3)] 
               if check_in_triangle(p, [plist[l] for l in triangle_list[childs[k]].points]):
                  #print [[plist[triangle_list[childs[k]].points[l]].x, plist[triangle_list[childs[k]].points[l]].y]  for l in range(3)] 
                  test = 0
                  Nel = childs[k]
                  print "triangle found with reference and points:", Nel, triangle_list[Nel].points
                  childs = triangle_list[Nel].childs
               k += 1
          if test == 1:
               print childs
               print " point not found",  p.x, p.y
               plot_point_DT(p, plist, [triangle_list[m] for m in childs])
               sys.exit("error: mesh fails to be created: the error should be reported")
    return Nel
    
def check_in_triangle(p, trian):
    eps = 1e-10
    A = trian[0]
    B = trian[1]
    C = trian[2]
    #print A.x, A.y, B.x, B.y, C.x, C.y
    u1 =  vector(A.y-B.y,B.x-A.x)
    u2 =  vector(A.y-C.y,C.x-A.x)
    u3 =  vector(C.y-B.y,B.x-C.x)
    if (C.x*u1.x+C.y*u1.y > A.x*u1.x+A.y*u1.y):
       u1.x = -1*u1.x
       u1.y = -1*u1.y
    if (B.x*u2.x+B.y*u2.y > A.x*u2.x+A.y*u2.y):
       u2.x = -1*u2.x
       u2.y = -1*u2.y
    if (A.x*u3.x+A.y*u3.y > B.x*u3.x+B.y*u3.y):
       u3.x = -1*u3.x
       u3.y = -1*u3.y
    #print "vectors",  u1.x, u1.y, u2.x, u2.y,  u3.x, u3.y
    if  (p.x*u1.x+p.y*u1.y <= A.x*u1.x+A.y*u1.y + eps) and (p.x*u2.x+p.y*u2.y <= A.x*u2.x+A.y*u2.y +eps) and (p.x*u3.x+p.y*u3.y <= B.x*u3.x+B.y*u3.y+eps):
         return 1
    else:
         return 0



def plot_mesh(plist, triangle_list):
    fig = plt.figure() # create figure object
    ax = fig.add_subplot(1,1,1) # create an axes object
    plt.gca().set_aspect('equal', adjustable='box')
    for triangle in triangle_list:
        triangle = triangle + [triangle[0]]
        coordx =[plist[p].x for p in triangle ]
        coordy =[plist[p].y for p in triangle ]
        ax.plot(coordx, coordy, 'k-*')
    plt.show()

