from MeshObjects.GeObjects import *
from MeshAlg.insert_point_algorithms import chew_add_point


gmesh = Mesh()
# vmesh = voronoi_mesh()
dispatcher = {'chew': chew_add_point}