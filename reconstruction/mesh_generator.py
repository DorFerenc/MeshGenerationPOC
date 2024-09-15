import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree
from .utils.error_handling import MeshGenerationError
import logging

logger = logging.getLogger(__name__)

class MeshGenerator:
    """
    Generates a 3D mesh from point cloud data.

    This class uses Delaunay triangulation to create a mesh from input point cloud data.
    It includes functionality for optimal alpha calculation and mesh quality assessment.
    """

    def __init__(self):
        """Initialize the MeshGenerator."""
        self.mesh = None

    def generate_mesh(self, point_cloud):
        """
        Generate a 3D mesh from the given point cloud data.

        Args:
            point_cloud (numpy.ndarray): The input point cloud data.

        Returns:
            pyvista.PolyData: The generated mesh.

        Raises:
            MeshGenerationError: If mesh generation fails.
        """
        try:
            if len(point_cloud) < 4:
                raise ValueError("At least 4 points are required to generate a 3D mesh")

            if self.are_points_collinear(point_cloud):
                raise ValueError("The points are collinear and cannot form a 3D mesh")

            alpha = self.calculate_optimal_alpha(point_cloud)
            poly_data = pv.PolyData(point_cloud)
            self.mesh = poly_data.delaunay_3d(alpha=alpha)
            self.mesh = self.mesh.extract_surface()

            if self.mesh.n_cells == 0:
                raise ValueError("Failed to generate a valid mesh")

            # Remove degenerate triangles
            self.mesh.clean(tolerance=1e-6)

            self.log_mesh_quality()
            return self.mesh, alpha
        except Exception as e:
            logger.error(f"Mesh generation failed: {str(e)}")
            raise MeshGenerationError(f"Failed to generate mesh: {str(e)}")

    def calculate_optimal_alpha(self, point_cloud, percentile=95):
        """
        Calculate an optimal alpha value for mesh generation.

        Args:
            point_cloud (numpy.ndarray): The input point cloud data.
            percentile (int): Percentile of nearest neighbor distances to use for alpha calculation.

        Returns:
            float: The calculated optimal alpha value.
        """
        tree = cKDTree(point_cloud)
        distances, _ = tree.query(point_cloud, k=2)
        nearest_neighbor_distances = distances[:, 1]
        alpha = np.percentile(nearest_neighbor_distances, percentile)

        scaling_factor = 25.2 if self.is_cube_like(point_cloud) else 2.2
        return alpha * scaling_factor

    def is_cube_like(self, point_cloud):
        """
        Determine if the point cloud resembles a cube.

        Args:
            point_cloud (numpy.ndarray): The input point cloud data.

        Returns:
            bool: True if the point cloud is cube-like, False otherwise.
        """
        # Normalize the point cloud
        normalized_points = (point_cloud - np.min(point_cloud, axis=0)) / (
                    np.max(point_cloud, axis=0) - np.min(point_cloud, axis=0))

        # Check aspect ratio
        dimensions = np.max(normalized_points, axis=0) - np.min(normalized_points, axis=0)
        aspect_ratios = dimensions / np.max(dimensions)
        if not np.all(aspect_ratios > 0.7):
            return False

        # Check distribution along each axis
        for axis in range(3):
            hist, _ = np.histogram(normalized_points[:, axis], bins=10)
            if np.std(hist) / np.mean(hist) > 0.5:  # High variation indicates non-uniform distribution
                return False

        return True

    def log_mesh_quality(self):
        """
        Compute and log the quality metrics of the generated mesh.

        This method calculates various quality metrics for the mesh cells and logs them.
        """
        if self.mesh is None:
            logger.warning("No mesh to analyze quality")
            return

        quality = self.mesh.compute_cell_quality()
        quality_array = quality['CellQuality']
        if len(quality_array) > 0:
            min_quality = np.min(quality_array)
            max_quality = np.max(quality_array)
            avg_quality = np.mean(quality_array)
            logger.info(f"Mesh quality - Min: {min_quality:.4f}, Max: {max_quality:.4f}, Avg: {avg_quality:.4f}")
        else:
            logger.warning("Unable to compute mesh quality. No cells in the mesh.")

    def are_points_collinear(self, points):
        if len(points) < 3:
            return True

        # Calculate vectors between consecutive points
        vectors = np.diff(points, axis=0)

        # Normalize the vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        normalized_vectors = vectors / norms

        # Check if all vectors are parallel
        dot_products = np.abs(np.dot(normalized_vectors[0], normalized_vectors[1:].T))
        return np.all(np.isclose(dot_products, 1.0, atol=1e-6))
