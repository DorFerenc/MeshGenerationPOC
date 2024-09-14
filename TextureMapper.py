import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

class TextureMapper:
    """
    A class for mapping textures onto 3D meshes.

    This class provides functionality to map colors from a point cloud onto a mesh,
    apply UV mapping, and smooth the resulting texture.

    Attributes:
        mesh (pyvista.PolyData): The mesh to be textured.
        point_cloud (numpy.ndarray): The point cloud data.
        colors (numpy.ndarray): The color data corresponding to the point cloud.
    """

    def __init__(self):
        self.mesh = None
        self.point_cloud = None
        self.colors = None

    def load_mesh(self, mesh):
        """
        Load a mesh for texture mapping.

        Args:
            mesh (pyvista.PolyData): The mesh to be textured.
        """
        self.mesh = mesh

    def load_point_cloud_with_colors(self, points, colors):
        """
        Load a point cloud with corresponding color data.

        Args:
            points (numpy.ndarray): The point cloud data.
            colors (numpy.ndarray): The color data corresponding to the point cloud.
        """
        self.point_cloud = points
        self.colors = colors

    def map_colors_to_mesh(self):
        """
        Map colors from the point cloud to the mesh vertices.

        This method uses a KD-tree for efficient nearest neighbor search to map
        colors from the point cloud to the mesh vertices.

        Raises:
            ValueError: If the mesh or point cloud with colors hasn't been loaded.
        """
        if self.mesh is None or self.point_cloud is None or self.colors is None:
            raise ValueError("Mesh and point cloud with colors must be loaded before mapping.")

        tree = cKDTree(self.point_cloud)
        distances, indices = tree.query(self.mesh.points)
        vertex_colors = self.colors[indices]
        self.mesh.point_data["RGB"] = vertex_colors

    def spherical_coordinates(self, xyz):
        """
        Convert Cartesian coordinates to spherical coordinates.

        Args:
            xyz (numpy.ndarray): Array of Cartesian coordinates.

        Returns:
            numpy.ndarray: Array of spherical coordinates (theta, phi).
        """
        ptsnew = np.zeros(xyz.shape)
        xy = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
        ptsnew[:, 0] = np.arctan2(np.sqrt(xy), xyz[:, 2])  # theta
        ptsnew[:, 1] = np.arctan2(xyz[:, 1], xyz[:, 0])  # phi
        return ptsnew

    def apply_uv_mapping(self):
        """
        Apply a simple spherical UV mapping to the mesh.

        Raises:
            ValueError: If the mesh hasn't been loaded.
        """
        if self.mesh is None:
            raise ValueError("Mesh must be loaded before applying UV mapping.")

        # Use our custom spherical_coordinates function
        uvw = self.spherical_coordinates(self.mesh.points)
        self.mesh.point_data["UV"] = uvw[:, :2] / np.array([np.pi, 2 * np.pi])

    def smooth_texture(self, iterations=10):
        """
        Smooth the texture on the mesh using Laplacian smoothing.

        Args:
            iterations (int): Number of smoothing iterations to perform.

        Raises:
            ValueError: If the mesh hasn't been loaded.
        """
        if self.mesh is None:
            raise ValueError("Mesh must be loaded before smoothing texture.")

        for _ in range(iterations):
            smooth = self.mesh.smooth(n_iter=1, relaxation_factor=0.1, feature_smoothing=False,
                                      boundary_smoothing=True, edge_angle=100, feature_angle=100)
            self.mesh.point_data["RGB"] = smooth.point_data["RGB"]

    def get_textured_mesh(self):
        """
        Get the textured mesh.

        Returns:
            pyvista.PolyData: The textured mesh.

        Raises:
            ValueError: If the mesh hasn't been loaded.
        """
        if self.mesh is None:
            raise ValueError("Mesh must be loaded before getting textured mesh.")
        return self.mesh