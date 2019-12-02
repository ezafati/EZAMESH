import sys 

sys.path.insert(0,'/home/ezafati/mesh_project/src/')

from class_DT import point
from DT_algorithm import *

A = point (0, 0)

B= point(20, 0)

C = point(20, 30)


D= point(0, 25)

E = point(1,9)

F= point(0,0)

list_point =[A, B, C, D ]

DT_algo(list_point, [{0,1}, {1,2},{2,3}, {3,0}])

