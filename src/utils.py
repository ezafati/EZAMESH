from __future__ import division
import math, sys
from math import sqrt
from scipy.optimize import minimize

def  objective(p, *args):
     v, size = args
     sum1 =0
     for i in range(len(v)): 
         dist = sqrt((p[0]-v[i].x)**2+(p[1]-v[i].y)**2)
         sum1 += (dist-size)**2
     return sum1
"""def  constraint1(p, *args):
     v = args
     for""" 
def  dist_mini(vx,size, ax, ay, cent):
     bx = (cent.x-0.5*ax, cent.x+0.5*ax)
     by = (cent.y-0.5*ay, cent.y+0.5*ay)
     bnds =(bx,by)
     p0 = [cent.x, cent.y]
     sol =  minimize(objective, p0, args = (vx, size),  method = 'SLSQP', bounds =bnds)
     return  sol.x
