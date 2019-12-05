import sys 

sys.path.insert(0,'/home/ezafati/mesh_project/src/')

from class_DT import point
from DT_algorithm import *

A = point (10, 11)

B= point(24, 50)

C = point(12, 18)


D= point(20, 30)

E = point(1,9)

F= point(0,0)

list_point =[A, B, C, D, E, F]

DT_algo(list_point)

