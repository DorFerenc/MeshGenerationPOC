# Mesh Generation POC

This project demonstrates the process of generating a 3D mesh from point cloud data using the Delaunay triangulation algorithm. It includes functionality for loading point cloud data, generating a mesh, applying various refinement techniques, and saving the resulting mesh.

## Features

- Load point cloud data from CSV files
- Generate 3D meshes using Delaunay triangulation
- Apply mesh refinement techniques including Laplacian and bilateral smoothing
- Visualize point clouds and generated meshes
- Save generated meshes in various formats (.ply, .vtp, .stl, .vtk)

## Requirements

To run this project, you need to have Python installed along with the following packages:

```
pip install numpy
pip install pyvista
pip install gmsh
```

## Usage

### Point Cloud to Mesh

The main functionality is provided by the `PointCloudToMesh` class. Here's a basic usage example:

```python
from PointCloudToMesh import PointCloudToMesh

# Create an instance
pc_to_mesh = PointCloudToMesh()

# Load point cloud data
pc_to_mesh.load_point_cloud_from_csv("path/to/your/pointcloud.csv")

# Generate mesh
pc_to_mesh.generate_mesh()

# Refine mesh
pc_to_mesh.refine_mesh()

# Save the mesh (supported formats: .ply, .vtp, .stl, .vtk)
pc_to_mesh.save_mesh("output_mesh.msh")

# Visualize results
pc_to_mesh.visualize_mesh()
```

### Mesh Generation Process

1. Load point cloud data
2. Apply Delaunay triangulation to create initial mesh
3. Remove degenerate triangles
4. Apply Laplacian smoothing
5. Apply bilateral smoothing
6. Save the resulting mesh in one of the supported formats (.ply, .vtp, .stl, .vtk)
8. (Optional) Visualize the resulting mesh

## Testing

To run the test suite:

```
python -m unittest test_PointCloudToMesh.py
```

## Additional Information

- The project uses PyVista for mesh operations and visualization.
- Logging is implemented to track the mesh generation and refinement process.
- Error handling is in place to manage common issues in the mesh generation pipeline.
- Generated meshes can be saved in .ply, .vtp, .stl, or .vtk formats for further use or analysis.

## Future Improvements

- Implement more advanced mesh refinement techniques
- Add support for different point cloud file formats
- Optimize performance for large point cloud datasets
- Support additional mesh file formats for saving

For more detailed information about the implementation, please refer to the source code and inline documentation.