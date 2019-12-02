import os.path, sys, math
import config
from MeshAlg.DT_prim import DT_algo
from MeshObjects.GeObjects import *
from math import sqrt

def read_file(file):
    try:
      f=open(file, 'r')
      if (f.readline() == ""):
           sys.exit("error: mesh file is empty")
      else:
           f.seek(0,0)
           line=f.readline()
           n_line = 1
           while (line != ""):
                 if  line.strip():
                     fields=line.split()
                     switcher_demo(fields,n_line) 
                 line=f.readline()
                 n_line += 1 
           f.close()
           DT_algo(config.gmesh)
    except (FileNotFoundError, PermissionError) as e:
      print("unexpected error: Maybe the file", file, " does not exist or the permissions should be modified")

def switcher_demo(fields, n_line):
    switcher = {
      'P':  lambda fields: config.gmesh.add_point(fields),
      'D':  lambda fields: config.gmesh.add_bound_seg(fields, n_line),
      'AS': lambda fields: config.gmesh.asize.append(float(fields[3]))
    }
    func =  switcher.get(fields[2], 'INVALID')
    if  func == 'INVALID':
        print 'error: line ',  n_line, 'see details below'
        sys.exit("error: the object is unknown ... ")
    switcher.get(fields[2])(fields)





