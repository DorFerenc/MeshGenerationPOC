import numpy as np
import pyvista as pv
import logging
import os
from scipy.spatial import cKDTree

class PointCloudToMesh:
    """
    A class to convert point cloud data to a 3D mesh.

    This class provides functionality to load point cloud data,
    generate a 3D mesh using Delaunay triangulation, and apply various smoothing
    and refinement techniques to improve the mesh quality.

    Attributes:
        point_cloud (np.ndarray): The loaded point cloud data.
        mesh (pv.PolyData): The generated 3D mesh.
        logger (logging.Logger): Logger for the class.
    """

    def __init__(self):
        """Initialize the PointCloudToMesh object."""
        self.point_cloud = None
        self.mesh = None
        self.logger = logging.getLogger(self.__class__.__name__)

    def set_point_cloud(self, points):
        """
        Set the point cloud data.

        Args:
            points (np.ndarray): Array of 3D point coordinates.
        """
        if len(points) == 0:
            raise ValueError("Point cloud cannot be empty")
        if points.shape[1] != 3:
            raise ValueError("Point cloud must have 3 dimensions (x, y, z)")
        self.point_cloud = points
        self.logger.info(f"Point cloud set with {len(points)} points")

    def calculate_optimal_alpha(self, percentile=95):
        """
        Calculate an optimal alpha value for mesh generation based on point cloud characteristics.

        This method uses the distance to the nearest neighbor for each point to estimate
        an appropriate alpha value. It aims to create a mesh that captures the shape of the
        point cloud without creating too many artifacts or holes.

        Args:
            percentile (int): Percentile of nearest neighbor distances to use for alpha calculation.
                              Default is 95, which works well for most point clouds.

        Returns:
            float: The calculated optimal alpha value.
        """
        if self.point_cloud is None or len(self.point_cloud) < 2:
            raise ValueError("Point cloud not set or has insufficient points")

        self.logger.info("Calculating optimal alpha value...")

        # Build a KD-tree for efficient nearest neighbor search
        tree = cKDTree(self.point_cloud)

        # Find the distance to the nearest neighbor for each point
        distances, _ = tree.query(self.point_cloud, k=2)  # k=2 because the nearest neighbor of a point is itself
        nearest_neighbor_distances = distances[:, 1]  # Take the second-nearest neighbor (first is the point itself)

        # Calculate the alpha value based on the specified percentile of nearest neighbor distances
        alpha = np.percentile(nearest_neighbor_distances, percentile)

        # Apply a scaling factor to fine-tune the alpha value
        # This factor can be adjusted based on empirical results
        if self.is_cube_like():
            scaling_factor = 25.2
        else:
            scaling_factor = 2.2
        # alpha *= scaling_factor
        # scaling_factor = 2.0
        alpha *= scaling_factor

        self.logger.info(f"Calculated optimal alpha: {alpha:.6f}")
        return alpha

    # Overwrite to cube like stuff
    def is_cube_like(self):
        # Check if the point cloud resembles a cube
        min_coords = np.min(self.point_cloud, axis=0)
        max_coords = np.max(self.point_cloud, axis=0)
        dimensions = max_coords - min_coords
        aspect_ratios = dimensions / np.max(dimensions)
        return np.all(aspect_ratios > 0.8)  # Consider it cube-like if all dimensions are similar

    def generate_mesh(self, alpha=None):
        """
        Generate a 3D mesh from the loaded point cloud data using Delaunay triangulation.

        Args:
            alpha (float, optional): The alpha value for the Delaunay triangulation algorithm.
                                     If None, calculates the optimal alpha value.

        Raises:
            ValueError: If no point cloud data has been loaded.
        """
        if self.point_cloud is None:
            self.logger.error("No point cloud data loaded")
            raise ValueError("No point cloud data loaded. Use set_point_cloud() first.")
        if len(self.point_cloud) < 4:
            raise ValueError("At least 4 points are required to generate a 3D mesh")

        if alpha is None:
            alpha = self.calculate_optimal_alpha()

        self.logger.info(f"Generating mesh with alpha={alpha}")
        try:
            poly_data = pv.PolyData(self.point_cloud)
            self.mesh = poly_data.delaunay_3d(alpha=alpha)
            self.mesh = self.mesh.extract_surface()

            # Remove degenerate triangles
            self.mesh.clean(tolerance=1e-6)

            n_cells = self.mesh.n_cells
            self.logger.info(f"Mesh generated with {self.mesh.n_points} points and {n_cells} cells")

            # Compute and log mesh quality
            self.log_mesh_quality()
            return self.mesh
        except Exception as e:
            self.logger.error(f"Error generating mesh: {str(e)}")
            raise

    def log_mesh_quality(self):
        """
        Compute and log the quality metrics of the generated mesh.

        This method calculates various quality metrics for the mesh cells, including
        minimum, maximum, and average quality values. These metrics help assess the
        overall quality of the generated mesh and can be used to identify potential
        issues or areas for improvement in the mesh generation process.

        The quality metric used is based on the ratio of the inscribed sphere radius
        to the circumscribed sphere radius for each cell, scaled to [0, 1].

        If the mesh has no cells, a warning is logged instead.

        Note:
            This method assumes that the mesh has already been generated and stored
            in the `self.mesh` attribute.

        Raises:
            AttributeError: If `self.mesh` is None or doesn't have the required methods.
        """
        quality = self.mesh.compute_cell_quality()
        quality_array = quality['CellQuality']
        if len(quality_array) > 0:
            min_quality = np.min(quality_array)
            max_quality = np.max(quality_array)
            avg_quality = np.mean(quality_array)
            self.logger.info(
                f"Mesh quality - Min: {min_quality:.4f}, Max: {max_quality:.4f}, Avg: {avg_quality:.4f}")
        else:
            self.logger.warning("Unable to compute mesh quality. No cells in the mesh.")

    def visualize_mesh(self):
        """
        Visualize the generated mesh.

        Raises:
            ValueError: If no mesh has been generated.
        """
        if self.mesh is None:
            raise ValueError("No mesh generated. Use generate_mesh() first.")

        p = pv.Plotter()
        p.add_mesh(self.mesh, color='orange')
        p.show()

    def save_mesh(self, filename):
        """
        Save the generated mesh to a file.

        Args:
            filename (str): The name of the file to save the mesh to.
                            Supported formats: .ply, .vtp, .stl, .vtk

        Raises:
            ValueError: If no mesh has been generated or if the file extension is not supported.
        """
        if self.mesh is None:
            self.logger.error("No mesh generated to save")
            raise ValueError("No mesh generated. Use generate_mesh() first.")

        supported_extensions = ['.ply', '.vtp', '.stl', '.vtk']
        file_extension = os.path.splitext(filename)[1].lower()

        if file_extension not in supported_extensions:
            self.logger.error(f"Unsupported file extension. Supported formats: {', '.join(supported_extensions)}")
            raise ValueError(f"Unsupported file extension. Supported formats: {', '.join(supported_extensions)}")

        try:
            self.mesh.save(filename)
            self.logger.info(f"Mesh saved successfully to {filename}")
        except Exception as e:
            self.logger.error(f"Error saving mesh: {str(e)}")
            raise


##########################################
# OPTIONAL FOR UPGRADE
###########################################
class MeshRefiner:
    @staticmethod
    def refine_mesh(mesh):
        # Implement mesh refinement techniques here
        # For example:
        # refined_mesh = mesh.smooth(n_iter=100, relaxation_factor=0.1)
        # return refined_mesh
        return mesh  # Placeholder, replace with actual refinement logic
