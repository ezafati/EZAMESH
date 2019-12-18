#!/usr/bin/env python  
import sys 

sys.path.append('/home/ezafati/mesh_project/src/')

from read_data import *
 
#PP= point(0,1) 

read_file("maillage5.txt")

#read_file("vt_mesh.txt")
#print "nnodes = ", config.nnodes

#switcher_demo(['NAME', '=', 'P', '1', '1.52'], 1)
#witcher_demo(['NAME_1', '=', 'P', '1', '1'], 1)


#print config.nnodes, config.label_list, config.point_list[2].x
