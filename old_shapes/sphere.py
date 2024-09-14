import gmsh
import sys

# Initialize gmsh
gmsh.initialize()

# Define the points of the cube
lc = 0.1
point1 = gmsh.model.geo.addPoint(0, 0, 0, lc)
point2 = gmsh.model.geo.addPoint(1, 0, 0, lc)
point3 = gmsh.model.geo.addPoint(1, 1, 0, lc)
point4 = gmsh.model.geo.addPoint(0, 1, 0, lc)
point5 = gmsh.model.geo.addPoint(0, 0, 1, lc)
point6 = gmsh.model.geo.addPoint(1, 0, 1, lc)
point7 = gmsh.model.geo.addPoint(1, 1, 1, lc)
point8 = gmsh.model.geo.addPoint(0, 1, 1, lc)

# Define the lines of the cube
line1 = gmsh.model.geo.addLine(point1, point2)
line2 = gmsh.model.geo.addLine(point2, point3)
line3 = gmsh.model.geo.addLine(point3, point4)
line4 = gmsh.model.geo.addLine(point4, point1)
line5 = gmsh.model.geo.addLine(point5, point6)
line6 = gmsh.model.geo.addLine(point6, point7)
line7 = gmsh.model.geo.addLine(point7, point8)
line8 = gmsh.model.geo.addLine(point8, point5)
line9 = gmsh.model.geo.addLine(point1, point6)
line10 = gmsh.model.geo.addLine(point2, point7)
line11 = gmsh.model.geo.addLine(point3, point8)
line12 = gmsh.model.geo.addLine(point4, point5)

# Define the surfaces of the cube
curve1 = gmsh.model.geo.addCurveLoop([line1, line2, line3, line4])
curve2 = gmsh.model.geo.addCurveLoop([line5, line6, line7, line8])
curve3 = gmsh.model.geo.addCurveLoop([-line9, -line10, -line11, -line12])
curve4 = gmsh.model.geo.addCurveLoop([line9, -line10, line3, -line12])
curve5 = gmsh.model.geo.addCurveLoop([line6, line10, -line1, -line5])
curve6 = gmsh.model.geo.addCurveLoop([line7, -line11, line12, -line8])

# Define the surfaces of the cube
surface1 = gmsh.model.geo.addPlaneSurface([curve1])
surface2 = gmsh.model.geo.addPlaneSurface([curve2])
surface3 = gmsh.model.geo.addPlaneSurface([curve3])
surface4 = gmsh.model.geo.addPlaneSurface([curve4])
surface5 = gmsh.model.geo.addPlaneSurface([curve5])
surface6 = gmsh.model.geo.addPlaneSurface([curve6])

# Define the sphere
sphere1 = gmsh.model.geo.addSphere(0.5, 0.5, 0.5, 0.25)

# Define the boolean operation
gmsh.model.geo.addVolume([surface1, surface2, surface3, surface4, surface5, surface6, sphere1])

# Synchronize the geometry
gmsh.model.geo.synchronize()

# Generate the mesh
gmsh.model.mesh.generate(3)

# Write the mesh data to a file
gmsh.write("3d_mesh.msh")

# If 'close' not in sys.argv, create graphical user interface
if 'close' not in sys.argv:
    gmsh.fltk.run()

# Finalize gmsh
gmsh.finalize()
