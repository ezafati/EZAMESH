import globalvar
import os.path
from MeshAlg.initial_DT import dt_initial
from MeshObjects.GeObjects import *


def read_file(meshfile):
    try:
        f = open(meshfile, 'r')
        if f.readline() == "":
            sys.exit("error: mesh file is empty")
        else:
            f.seek(0, 0)
            line = f.readline()
            n_line = 1
            while line != "":
                if line.strip():
                    fields = line.split()
                    switcher_demo(fields, n_line)
                line = f.readline()
                n_line += 1
            f.close()
            dt_initial(globalvar.gmesh)
    except (FileNotFoundError, PermissionError) as e:
        print("Error File: ", os.path.abspath(meshfile), "Not Found  or Permission Error")


def switcher_demo(fields, n_line):
    switcher = {
        'POINT': lambda fields: globalvar.gmesh.add_point(fields),
        'LINE': lambda fields: globalvar.gmesh.add_bound_seg(fields, n_line)
    }
    if switcher.get(fields[2], 'INVALID') == 'INVALID':
        print('error: line ', n_line, 'see details below')
        sys.exit("error: the object is unknown ... ")
    switcher.get(fields[2])(fields)
