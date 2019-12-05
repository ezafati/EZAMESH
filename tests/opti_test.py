import sys 

sys.path.insert(0,'/home/ezafati/mesh_project/src/')

from class_DT import point
from utils import *


x=[]

x.append(point(0,0))
x.append(point(1,0))
x.append(point(0,1))
size = 1
ax =  1
ay =  1

cent = point(1,1)
#cent1 = [1,3]
#yy= objective(cent1, x, size)

#print yy
dist_mini(x, size, ax, ay, cent)

