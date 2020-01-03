from MeshObjects.GeObjects import Mesh, FileParser
from collections import namedtuple

partmesh = Mesh()
globalmesh = list()
parsefile = FileParser()
tree_refinement = None

method_properties = namedtuple('method_properties', ('module_name', 'mesh_func', 'init_params'))

ini_val = [{'type': 'd', 'val': 0.0},
           {'type': 'i', 'val': int(1e9)}]

dispatcher = {'default': method_properties('chew_insert_algorithm', 'chew_add_point', ini_val)}


