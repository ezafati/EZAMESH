
import sys 

sys.path.insert(0,'/home/ezafati/mesh_project/src/')

from class_DT import *
from DT_algorithm import *

triangle_list = []

triangle_list.append(triangle([1, 2 ,3], [], [1 ,3 ,4]))
triangle_list.append(triangle([1, 2, 5], [], [0, 5, 6]))
triangle_list.append( triangle([4, 2, 5], [], []))
triangle_list.append( triangle([1, 3, 7], [], [0]))
triangle_list.append(triangle([10, 2, 3], [], [0]))
triangle_list.append(triangle([20, 2, 5], [], [1]))
triangle_list.append(triangle([1 ,5 ,30], [], [1]))

swapping_triangles(triangle_list, 0, 1)


p =point(0,0)

lp = [point(1 ,2), point(1, 3), point(0, 2)]

test = check_in_circle(p, lp)

print test

