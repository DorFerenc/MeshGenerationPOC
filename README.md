#Mesh Generation POC

###Mesh Generation: Utilizing the preprocessed point cloud data, the system employs the 3D Delaunay tetrahedralization algorithm to form a solid mesh representation of the captured scene. This algorithm connects neighboring points in the point cloud to create a network of triangles and other geometric shapes, effectively enclosing the space captured by the point cloud.

###Requirements 
* pip install gmsh
* pip install numpy

###Creating a mesh process: 
1) Get the points,
2) Add minimum lines between important points(try looking for basic shapes like squares)
3) Add curves between lines 
4) Create a surface
5) Synchronize all the surfaces
6) Generate model
7) Save the model as a ‘.msh’ file

