import module_var
import os.path
import os

try:
    import psutil
except ImportError:
    pass

import sys

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
        sys.exit()
    except StopIteration as e:
        logging.info("############### END READ MESH FILE WITH SUCCESS   ###########################")
        run_tri_mesh(module_var.gmesh, process)


def switch_case(fields, n_line, meshfile):
    switcher = {
        'POINT': lambda fields: module_var.gmesh.add_point(fields),
        'LINE': lambda fields: module_var.gmesh.add_line(fields, n_line),
        'ARC': lambda fields: module_var.gmesh.add_arc(fields, n_line),
        'SPLINE': lambda fields: module_var.gmesh.add_spline(fields, n_line),
        'PART': lambda fields: module_var.gmesh.close_check(fields, n_line)
    }
    if switcher.get(fields[2], 'INVALID') == 'INVALID':
        logging.error(
            f'IN LINE {n_line} IN FILE {os.path.abspath(meshfile)}: THE OPTION ({fields[2]}) IS NOT EXPECTED ')
    switcher.get(fields[2])(fields)


def switch_demo(fields, n_line, meshfile):
    switcher = {
        'POINT': lambda fields: module_var.parsefile.add_point(fields),
        'LINE': lambda fields: module_var.parsefile.add_bound(fields, n_line),
        'ARC': lambda fields: module_var.parsefile.add_bound(fields, n_line),
        'SPLINE': lambda fields: module_var.parsefile.add_bound(fields, n_line),
        'PART': lambda fields: module_var.parsefile.add_part(fields, n_line)
    }
    try:
        assert switcher.get(fields[2], 'INVALID') != 'INVALID'
        switcher.get(fields[2])(fields)
    except Exception:
        raise SyntaxError(f'IN LINE {n_line} IN FILE {os.path.abspath(meshfile)}: THE OPTION ({fields[2]}) IS NOT '
                          f'EXPECTED ')
