# Import modules:
import gmsh
import sys

# Initialize gmsh:
gmsh.initialize()

# cube points:
lc = 0.01  # 1e-2
point1 = gmsh.model.geo.add_point(0, 0, 0, lc)
point2 = gmsh.model.geo.add_point(1, 0, 0, lc)
point3 = gmsh.model.geo.add_point(1, 1, 0, lc)
point4 = gmsh.model.geo.add_point(0, 1, 0, lc)
point5 = gmsh.model.geo.add_point(0, 1, 1, lc)
point6 = gmsh.model.geo.add_point(0, 0, 1, lc)
point7 = gmsh.model.geo.add_point(1, 0, 1, lc)
point8 = gmsh.model.geo.add_point(1, 1, 1, lc)

# Edge of cube:
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

# faces of cube:
face1 = gmsh.model.geo.add_curve_loop([line1, line2, line3, line4])
face2 = gmsh.model.geo.add_curve_loop([line5, line6, line7, line8])
face3 = gmsh.model.geo.add_curve_loop([line9, line5, line10, -line4])
face4 = gmsh.model.geo.add_curve_loop([line9, -line8, -line12, line3])
face5 = gmsh.model.geo.add_curve_loop([line6, line11, -line1, -line10])
face6 = gmsh.model.geo.add_curve_loop([line11, line2, line12, -line7])

# surfaces of cube:
gmsh.model.geo.add_plane_surface([face1])
gmsh.model.geo.add_plane_surface([face2])
gmsh.model.geo.add_plane_surface([face3])
gmsh.model.geo.add_plane_surface([face4])
gmsh.model.geo.add_plane_surface([face5])
gmsh.model.geo.add_plane_surface([face6])

# Points of bigger pentagon:
point9 = gmsh.model.geo.add_point(0.3, 0.3, -2, lc)
point10 = gmsh.model.geo.add_point(0.7, 0.3, -2, lc)
point11 = gmsh.model.geo.add_point(0.7, 0.5, -2, lc)
point12 = gmsh.model.geo.add_point(0.5, 0.7, -2, lc)
point13 = gmsh.model.geo.add_point(0.3, 0.5, -2, lc)

# Points of smaller pentagon:
point14 = gmsh.model.geo.add_point(0.4, 0.4, 2, lc)
point15 = gmsh.model.geo.add_point(0.6, 0.4, 2, lc)
point16 = gmsh.model.geo.add_point(0.6, 0.5, 2, lc)
point17 = gmsh.model.geo.add_point(0.5, 0.6, 2, lc)
point18 = gmsh.model.geo.add_point(0.4, 0.5, 2, lc)

# lines of bigger pentagon:
line13 = gmsh.model.geo.add_line(point9, point10)
line14 = gmsh.model.geo.add_line(point10, point11)
line15 = gmsh.model.geo.add_line(point11, point12)
line16 = gmsh.model.geo.add_line(point12, point13)
line17 = gmsh.model.geo.add_line(point13, point9)

# lines of smaller pentagon:
line18 = gmsh.model.geo.add_line(point14, point15)
line19 = gmsh.model.geo.add_line(point15, point16)
line20 = gmsh.model.geo.add_line(point16, point17)
line21 = gmsh.model.geo.add_line(point17, point18)
line22 = gmsh.model.geo.add_line(point18, point14)

# face of bigger pentagon.
face7 = gmsh.model.geo.add_curve_loop([line13, line14, line15, line16, line17])

# face of smaller pentagon.
face8 = gmsh.model.geo.add_curve_loop([line18, line19, line20, line21, line22])

# connection of cube faces with pentagon
# and bigger pentagon with smaller.
gmsh.model.geo.add_plane_surface([face1, face7])
gmsh.model.geo.add_plane_surface([face2, face8])
gmsh.model.geo.add_plane_surface([face7, face8])

# Create the relevant Gmsh data structures
# from Gmsh model.
gmsh.model.geo.synchronize()

# Generate mesh:
gmsh.model.mesh.generate()

# Write mesh data:
gmsh.write("pentagonMESH.msh")

# Creates graphical user interface
if 'close' not in sys.argv:
	gmsh.fltk.run()

# It finalize the Gmsh API
gmsh.finalize()
