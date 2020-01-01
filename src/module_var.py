from MeshObjects.GeObjects import Mesh, ParseMeshFile
from collections import namedtuple

gmesh = Mesh()
parsefile = ParseMeshFile()
tree_refinement = None

method_properties = namedtuple('method_properties', ('module_name', 'mesh_func', 'init_params'))

ini_val = [{'type': 'd', 'val': 0.0},
           {'type': 'i', 'val': int(1e9)}]

dispatcher = {'default': method_properties('chew_insert_algorithm', 'chew_add_point', ini_val)}


