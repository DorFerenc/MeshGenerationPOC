import numpy as np
from scipy.spatial import cKDTree
from scipy import ndimage


class TextureMapper:
    """
    A class for mapping textures onto 3D meshes.

    This class provides functionality to map colors from a point cloud onto a mesh,
    apply UV mapping, smooth textures, and generate texture images.

    Attributes:
        mesh (pyvista.PolyData): The mesh to be textured.
        point_cloud (numpy.ndarray): The point cloud data.
        colors (numpy.ndarray): The color data corresponding to the point cloud.
        texture_resolution (int): The resolution of the generated texture image.
    """

    def __init__(self, texture_resolution=1024):
        """
        Initialize the TextureMapper with the specified texture resolution.

        This class handles the process of applying textures to 3D meshes, including
        color mapping, UV mapping, and texture smoothing.

        Args:
            texture_resolution (int, optional): The resolution of the generated texture image.
                                                Higher values provide more detail but require more memory.
                                                Defaults to 1024x1024.

        Attributes:
            mesh (pyvista.PolyData): The mesh to be textured.
            point_cloud (numpy.ndarray): The point cloud data used for color mapping.
            colors (numpy.ndarray): The color data corresponding to the point cloud.
            texture_resolution (int): The resolution of the generated texture image.
        """
        self.mesh = None
        self.point_cloud = None
        self.colors = None
        self.texture_resolution = texture_resolution

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

    def apply_texture(self):
        """
        Apply the complete texturing process to the loaded mesh.

        This method performs the following steps in order:
        1. Maps colors from the point cloud to the mesh vertices.
        2. Applies smart UV mapping based on the mesh geometry.
        3. Smooths the texture to improve visual quality.

        Note:
            Before calling this method, ensure that a mesh has been loaded using `load_mesh()`
            and point cloud data with colors has been loaded using `load_point_cloud_with_colors()`.

        Raises:
            ValueError: If the mesh or point cloud with colors hasn't been loaded.
        """
        self.map_colors_to_mesh()
        self.smooth_texture()
        # Apply UV mapping (if needed for OBJ export)
        self.apply_smart_uv_mapping()

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

        # Ensure colors are in the range [0, 1]
        if vertex_colors.max() > 1.0:
            vertex_colors = vertex_colors / 255.0

        self.mesh.point_data["RGB"] = vertex_colors

    def apply_smart_uv_mapping(self):
        """
        Apply an appropriate UV mapping method based on the mesh's shape.

        This method analyzes the mesh's dimensions and chooses between planar,
        cylindrical, or spherical UV mapping.

        Raises:
            ValueError: If the mesh hasn't been loaded.
        """
        if self.mesh is None:
            raise ValueError("Mesh must be loaded before applying UV mapping.")

        bounds = self.mesh.bounds
        dimensions = np.array([bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]])
        aspect_ratio = dimensions / np.max(dimensions)

        if aspect_ratio[2] < 0.5:  # Flat object
            self.apply_planar_uv_mapping()
        elif aspect_ratio[0] > 0.8 and aspect_ratio[1] > 0.8:  # Roughly cubic or spherical
            self.apply_spherical_uv_mapping()
        else:  # Elongated object
            self.apply_cylindrical_uv_mapping()

    def apply_planar_uv_mapping(self):
        """
        Apply planar UV mapping to the mesh.

        This method projects the mesh onto a plane and uses the x and y coordinates
        as UV coordinates.
        """
        bounds = self.mesh.bounds
        min_point = np.array([bounds[0], bounds[2], bounds[4]])
        max_point = np.array([bounds[1], bounds[3], bounds[5]])

        normalized_points = (self.mesh.points - min_point) / (max_point - min_point)

        u = normalized_points[:, 0]
        v = normalized_points[:, 1]

        self.mesh.point_data["UV"] = np.column_stack((u, v))

    def apply_cylindrical_uv_mapping(self):
        """
        Apply cylindrical UV mapping to the mesh.

        This method uses cylindrical coordinates to create UV mapping, suitable for
        elongated objects.
        """
        center = self.mesh.center
        bounds = self.mesh.bounds
        height = bounds[5] - bounds[4]

        normalized_points = self.mesh.points - center

        r = np.sqrt(normalized_points[:, 0] ** 2 + normalized_points[:, 1] ** 2)
        theta = np.arctan2(normalized_points[:, 1], normalized_points[:, 0])
        z = normalized_points[:, 2]

        u = (theta + np.pi) / (2 * np.pi)
        v = (z - bounds[4]) / height

        self.mesh.point_data["UV"] = np.column_stack((u, v))

    def apply_spherical_uv_mapping(self):
        """
        Apply spherical UV mapping to the mesh.

        This method uses spherical coordinates to create UV mapping, suitable for
        roughly spherical or cubic objects.
        """
        center = self.mesh.center
        max_radius = np.max(np.linalg.norm(self.mesh.points - center, axis=1))
        normalized_points = (self.mesh.points - center) / max_radius

        x, y, z = normalized_points.T
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        theta = np.arccos(np.clip(z / r, -1, 1))
        phi = np.arctan2(y, x)

        u = (phi + np.pi) / (2 * np.pi)
        v = theta / np.pi

        self.mesh.point_data["UV"] = np.column_stack((u, v))

    def smooth_texture(self, iterations=5):
        """
        Smooth the texture on the mesh.

        This method applies Laplacian smoothing to the color data on the mesh.

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

    def generate_texture_image(self):
        """
        Generate a texture image from the mesh's color and UV data.

        This method creates a 2D texture image based on the mesh's color data and
        UV coordinates. It uses a distance transform to fill in empty areas of the texture.

        Returns:
            numpy.ndarray: The generated texture image.

        Raises:
            ValueError: If the mesh doesn't have RGB and UV data.
        """
        if 'RGB' not in self.mesh.point_data or 'UV' not in self.mesh.point_data:
            raise ValueError("Mesh must have RGB and UV data before generating texture image.")

        colors = self.mesh.point_data['RGB']
        uv_coords = self.mesh.point_data['UV']

        texture_image = np.zeros((self.texture_resolution, self.texture_resolution, 3), dtype=np.float32)

        # Create a mask to track filled pixels
        mask = np.zeros((self.texture_resolution, self.texture_resolution), dtype=bool)

        for i, uv in enumerate(uv_coords):
            x = int(uv[0] * (self.texture_resolution - 1))
            y = int((1 - uv[1]) * (self.texture_resolution - 1))
            texture_image[y, x] = colors[i]
            mask[y, x] = True

        # Use a distance transform to fill empty areas
        distance = ndimage.distance_transform_edt(~mask)
        indices = ndimage.distance_transform_edt(~mask, return_distances=False, return_indices=True)

        for channel in range(3):
            channel_image = texture_image[:, :, channel]
            texture_image[:, :, channel] = channel_image[tuple(indices)]

        return texture_image

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
