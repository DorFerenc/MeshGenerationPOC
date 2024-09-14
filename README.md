# Mesh Generation POC

This project demonstrates the process of generating a 3D mesh from point cloud data using the Delaunay triangulation algorithm. It includes functionality for generating colored point clouds, loading point cloud data, generating meshes with optimal alpha values, applying texture mapping, and visualizing the results.

## Features

- Generate colored point clouds for various shapes (sphere, cube, torus)
- Load point cloud data from CSV files
- Generate 3D meshes using Delaunay triangulation with automatic optimal alpha calculation
- Apply texture mapping to the generated meshes
- Visualize point clouds, generated meshes, and textured meshes
- Save generated meshes in various formats (.ply, .vtp, .stl, .vtk)

## Requirements

To run this project, you need to have Python installed along with the following packages:

```
pip install numpy
pip install pandas
pip install pyvista
pip install scipy
```

## Project Structure

- `GenColoredPointCloud.py`: Generates colored point clouds for different shapes
- `PointCloudToMesh.py`: Contains the `PointCloudToMesh` class for mesh generation
- `TextureMapper.py`: Contains the `TextureMapper` class for texture mapping
- `app2.py`: Main application script that orchestrates the entire process
- `test_PointCloudToMesh.py`: Unit tests for the `PointCloudToMesh` class

## Process

1. **Point Cloud Generation**:
   - The `GenColoredPointCloud.py` script generates colored point clouds for sphere, cube, and torus shapes.
   - Each point in the cloud has x, y, z coordinates and r, g, b color values.
   - The generated point clouds are saved as CSV files.

2. **Point Cloud Loading**:
   - The `app2.py` script loads the point cloud data from a CSV file.
   - If the loaded data doesn't include colors, it generates colors based on the points' positions.

3. **Mesh Generation**:
   - The `PointCloudToMesh` class handles the mesh generation process.
   - It calculates an optimal alpha value for Delaunay triangulation based on the point cloud characteristics.
   - The mesh is generated using the calculated alpha value, which helps in creating a mesh that best represents the original shape.

4. **Texture Mapping**:
   - The `TextureMapper` class applies texture to the generated mesh.
   - It maps colors from the point cloud to the mesh vertices.
   - A spherical UV mapping is applied to the mesh.
   - The texture is smoothed to improve visual quality.

5. **Visualization**:
   - The process includes visualization steps for:
     - The original point cloud
     - The generated mesh
     - The textured mesh
   - Visualization is done using PyVista, providing interactive 3D views of each step.

6. **Mesh Saving**:
   - The final textured mesh can be saved in various formats (.ply, .vtp, .stl, .vtk) for further use or analysis.

## Usage

To use this project:

1. Run `GenColoredPointCloud.py` to generate point cloud data:
   ```
   python GenColoredPointCloud.py
   ```

2. Run `app2.py` to process the point cloud, generate the mesh, and apply texturing:
   ```
   python app2.py
   ```

3. Follow the on-screen prompts to visualize each step of the process.

## Testing

To run the test suite:

```
python -m unittest test_PointCloudToMesh.py
```

## Future Improvements

- Implement more advanced mesh refinement techniques
- Add support for additional point cloud file formats
- Optimize performance for large point cloud datasets
- Enhance texture mapping for complex geometries
- Implement automatic shape detection for optimal alpha calculation

For more detailed information about the implementation, please refer to the source code and inline documentation.