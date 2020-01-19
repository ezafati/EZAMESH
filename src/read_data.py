import module_var
import os.path
import os

try:
    import psutil
except ImportError:
    pass

import logging
from MeshAlg.global_DT import run_tri_mesh


def read_file(meshfile, process):
    try:
        with open(meshfile, 'r') as f:
            if f.readline() == "":
                logging.error(f'FILE {os.path.abspath(meshfile)} EMPTY')
            else:
                f.seek(0, 0)
                line = next(f)
                n_line = 1
                while line != "":
                    if line.strip():
                        fields = line.split()
                        switch_case(fields, n_line, meshfile)
                    line = next(f)
                    n_line += 1
    except (FileNotFoundError, PermissionError) as e:
        logging.error(f" IN FILE {os.path.abspath(meshfile)}:  {e}")
        raise Exception()
    except StopIteration as e:
        logging.info("############### END READ MESH FILE WITH SUCCESS   ###########################")
        for part in module_var.parsefile.parts.values():
            module_var.partmesh = part.create_mesh(module_var.parsefile)
            run_tri_mesh(module_var.partmesh, process)


def switch_case(fields, n_line, meshfile):
    switcher = {
        'POINT': lambda fields: module_var.parsefile.make_point(fields, n_line),
        'LINE': lambda fields: module_var.parsefile.make_boundary(fields, n_line),
        'ARC': lambda fields: module_var.parsefile.make_boundary(fields, n_line),
        'SPLINE': lambda fields: module_var.parsefile.make_boundary(fields, n_line),
        'PART': lambda fields: module_var.parsefile.make_part(fields, n_line),
        'MAKEMESH': lambda fields: module_var.parsefile.make_mesh(fields, n_line)
    }
    try:
        assert switcher.get(fields[2], 'INVALID') != 'INVALID'
        switcher.get(fields[2])(fields)
    except AssertionError:
        raise SyntaxError(f'IN LINE {n_line} IN FILE {os.path.abspath(meshfile)}: THE OPTION ({fields[2]}) IS NOT '
                          f'EXPECTED ')
