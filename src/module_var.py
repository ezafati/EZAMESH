from MeshObjects.GeObjects import Mesh
from collections import namedtuple

gmesh = Mesh()
tree_refinement = None

method_properties = namedtuple('method_properties', ('module_name', 'mesh_func', 'proc_prop'))

dispatcher = {'default': method_properties('chew_insert_algorithm', 'chew_add_point', ({'type': 'd', 'init': 0.0},
                                                                                       {'type': 'i', 'init': len(
                                                                                           tree_refinement.root.childs)}))}
