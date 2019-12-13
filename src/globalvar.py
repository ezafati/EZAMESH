from MeshObjects.GeObjects import *
from MeshAlg.chew_insert_algorithms import chew_add_point


gmesh = Mesh()
# vmesh = voronoi_mesh()
dispatcher = {'chew': chew_add_point}