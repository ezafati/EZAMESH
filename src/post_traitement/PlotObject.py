
class post_traitement(object):
   def sort_vtk_file(self, gmesh):
       size =  3 # to generalize after
       f = open ("./sort/mesh.vtk","w")
       f.write("# vtk DataFile Version 4.2\n") 
       f.write("mesh details \n")
       f.write("ASCII \n")
       f.write("\n")
       f.write("DATASET UNSTRUCTURED_GRID \n")
       f.write("POINTS {:d} {type}\n".format(len(gmesh.point_list), type = "float"))
       for i in range(len(gmesh.point_list)):
           f.write("{:6.4f} {:6.4f} {:6.4f}\n".format( gmesh.point_list[i].x, gmesh.point_list[i].y, 0))
       f.write("\n")
       f.write("CELLS {:d} {:d} \n".format(len(gmesh.triangle_list), (size+1)*len(gmesh.triangle_list)))
       for i in range(len(gmesh.triangle_list)):
           f.write("{:d} {:d} {:d} {:d} \n".format(size, gmesh.triangle_list[i][0], gmesh.triangle_list[i][1], gmesh.triangle_list[i][2] ))
       f.write("\n")
       f.write("CELL_TYPES {:d}\n".format(len(gmesh.triangle_list)))
       for i in range(len(gmesh.triangle_list)):
           f.write("{type} \n".format(type = 5))
       f.close()
