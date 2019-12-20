import module_var
import os.path
import os
import sys
import logging
from MeshAlg.global_DT import dt_global

FILE_LOG_PATH = 'mesh.log'
try:
    logging.basicConfig(filename=FILE_LOG_PATH, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
except FileNotFoundError:
    sys.exit(f'FILE {FILE_LOG_PATH} NOT FOUND')


def read_file(meshfile):
    try:
        with open(meshfile, 'r') as f:
            if f.readline() == "":
                logging.error(f'FILE {os.path.abspath(meshfile)} EMPTY')
                sys.exit(f'SEE LOG FILE {os.path.abspath(FILE_LOG_PATH)}')
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
        logging.info("############### BEGIN PROCESS ###########################")
        dt_global(module_var.gmesh)


def switch_case(fields, n_line, meshfile):
    switcher = {
        'POINT': lambda fields: module_var.gmesh.add_point(fields),
        'LINE': lambda fields: module_var.gmesh.add_line(fields, n_line),
        'ARC': lambda fields: module_var.gmesh.add_arc(fields, n_line),
        'SPLINE': lambda fields: module_var.gmesh.add_spline(fields, n_line),
        'PART': lambda fields: module_var.gmesh.close_check(fields, n_line)
    }
    if switcher.get(fields[2], 'INVALID') == 'INVALID':
        logging.error(f'IN LINE {n_line} IN FILE {os.path.abspath(meshfile)}: THE OPTION ({fields[2]}) IS NOT EXPECTED ')
        sys.exit(f'SEE LOG FILE {os.path.abspath(FILE_LOG_PATH)}')
    switcher.get(fields[2])(fields)
