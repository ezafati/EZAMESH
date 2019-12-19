import module_var
import os.path
import os
import logging
from MeshAlg.global_DT import dt_global


"""root_path = os.path.join('../', os.getcwd())
FILE_LOG_PATH = os.path.join(root_path, '/log/mesh.log')
logging.basicConfig(filename=FILE_LOG_PATH, level=logging.INFO)"""


def read_file(meshfile):
    try:
        with open(meshfile, 'r') as f:
            if f.readline() == "":
                sys.exit("error: mesh file is empty")
            else:
                f.seek(0, 0)
                line = next(f)
                n_line = 1
                while line != "":
                    if line.strip():
                        fields = line.split()
                        switch_case(fields, n_line)
                    line = next(f)
                    n_line += 1
    except (FileNotFoundError, PermissionError) as e:
        print("Error File: ", os.path.abspath(meshfile), "Not Found  or Permission Error", e)
    except StopIteration as e:
        print("############### END READ MESH FILE WITH SUCCESS   ###########################")
        print("############### BEGIN PROCESS ###########################")
        dt_global(module_var.gmesh)


def switch_case(fields, n_line):
    switcher = {
        'POINT': lambda fields: module_var.gmesh.add_point(fields),
        'LINE': lambda fields: module_var.gmesh.add_line(fields, n_line),
        'ARC': lambda fields: module_var.gmesh.add_arc(fields, n_line),
        'PART': lambda fields: module_var.gmesh.close_check(fields, n_line)
    }
    if switcher.get(fields[2], 'INVALID') == 'INVALID':
        print('error: line ', n_line, 'see details below')
        sys.exit("error: the object is unknown ... ")
    switcher.get(fields[2])(fields)
