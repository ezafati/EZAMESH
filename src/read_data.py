import globalvar
import os.path
from MeshAlg.initial_DT import dt_initial
from MeshObjects.GeObjects import *


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
        dt_initial(globalvar.gmesh)


def switch_case(fields, n_line):
    switcher = {
        'POINT': lambda fields: globalvar.gmesh.add_point(fields),
        'LINE': lambda fields: globalvar.gmesh.add_line(fields, n_line)
    }
    if switcher.get(fields[2], 'INVALID') == 'INVALID':
        print('error: line ', n_line, 'see details below')
        sys.exit("error: the object is unknown ... ")
    switcher.get(fields[2])(fields)
