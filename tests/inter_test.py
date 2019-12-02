import sys 

sys.path.insert(0,'/home/ezafati/mesh_project/src/')

from class_DT import *
from DT_algorithm import *

plist = []


plist.append(point(1,1))

plist.append(point(0,0))

plist.append(point(0, 3))

plist.append(point(0.5, 0))

plist.append(point(-1,0))

test= check_intersection(plist, {0,1}, {3,2})

print test
