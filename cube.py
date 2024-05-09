import gmsh
import sys

# Initialize gmsh:
gmsh.initialize()

""" This line sets the 'characteristic length' (lc) for the mesh elements. This parameter controls the size of the mesh 
elements in the resulting mesh. """
lc = 0.01  # 1e-2  # Define the size of the mesh elements

# Define the points of the cube
""" These lines define the eight vertices (points) of a cube in 3D space. Each point is
created using the add_point function provided by the gmsh module. The (x, y, z) coordinates of each point are
specified, along with the characteristic length lc for mesh generation."""
point1 = gmsh.model.geo.add_point(0, 0, 0, lc)
point2 = gmsh.model.geo.add_point(1, 0, 0, lc)
point3 = gmsh.model.geo.add_point(1, 1, 0, lc)
point4 = gmsh.model.geo.add_point(0, 1, 0, lc)
point5 = gmsh.model.geo.add_point(0, 1, 1, lc)
point6 = gmsh.model.geo.add_point(0, 0, 1, lc)
point7 = gmsh.model.geo.add_point(1, 0, 1, lc)
point8 = gmsh.model.geo.add_point(1, 1, 1, lc)

# Edge of cube: Define the edges of the cube
""" These lines define the edges of the cube by connecting the previously defined vertices (points). Each edge is 
created using the add_line function provided by the gmsh module. """
line1 = gmsh.model.geo.add_line(point1, point2)
line2 = gmsh.model.geo.add_line(point2, point3)
line3 = gmsh.model.geo.add_line(point3, point4)
line4 = gmsh.model.geo.add_line(point4, point1)
line5 = gmsh.model.geo.add_line(point5, point6)
line6 = gmsh.model.geo.add_line(point6, point7)
line7 = gmsh.model.geo.add_line(point7, point8)
line8 = gmsh.model.geo.add_line(point8, point5)
line9 = gmsh.model.geo.add_line(point4, point5)
line10 = gmsh.model.geo.add_line(point6, point1)
line11 = gmsh.model.geo.add_line(point7, point2)
line12 = gmsh.model.geo.add_line(point3, point8)

# Define the faces of the cube
"""These lines define the faces of the cube by creating closed loops of curves (edges). Each face is defined by 
specifying the edges that form its boundary. The - sign before some edges indicates that the edge is traversed in the 
opposite direction. """
face1 = gmsh.model.geo.add_curve_loop([line1, line2, line3, line4])
face2 = gmsh.model.geo.add_curve_loop([line5, line6, line7, line8])
face3 = gmsh.model.geo.add_curve_loop([line9, line5, line10, -line4])
face4 = gmsh.model.geo.add_curve_loop([line9, -line8, -line12, line3])
face5 = gmsh.model.geo.add_curve_loop([line6, line11, -line1, -line10])
face6 = gmsh.model.geo.add_curve_loop([line11, line2, line12, -line7])

# Define the surfaces of the cube
"""These lines create the surfaces of the cube by specifying the curve loops (faces) that bound each surface. Each 
surface is defined using the add_plane_surface function provided by the gmsh module. """
gmsh.model.geo.add_plane_surface([face1])
gmsh.model.geo.add_plane_surface([face2])
gmsh.model.geo.add_plane_surface([face3])
gmsh.model.geo.add_plane_surface([face4])
gmsh.model.geo.add_plane_surface([face5])
gmsh.model.geo.add_plane_surface([face6])

# Create the relevant Gmsh data structures from Gmsh model.
"""This line synchronizes the geometry defined in the Gmsh model, creating the relevant data structures required for 
mesh generation based on the geometric entities (points, lines, curves, surfaces, etc.). """
gmsh.model.geo.synchronize()

# Generate mesh:
"""This line generates the mesh based on the geometry defined in the Gmsh model. The type and density of the mesh 
elements are determined by the characteristic length lc specified earlier. """
gmsh.model.mesh.generate()

# Write mesh data:
gmsh.write("cubeMESH.msh")

# Creates  graphical user interface
"""This part of the code creates a graphical user interface (GUI) for visualizing the mesh if the string 'close' is 
not found in the command line arguments. """
if 'close' not in sys.argv:
    gmsh.fltk.run()

# Finalize the gmsh library
gmsh.finalize()

