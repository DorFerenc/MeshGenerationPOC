import numpy as np
from scipy.spatial import cKDTree
from scipy import ndimage
from .utils.error_handling import TexturingError

class TextureMapper:
    """
    Applies textures to 3D meshes.

    This class provides functionality to map colors from a point cloud onto a mesh,
    apply UV mapping, smooth textures, and generate texture images.
    """

    def __init__(self, texture_resolution=1024):
        """
        Initialize the TextureMapper.

        Args:
            texture_resolution (int): The resolution of the generated texture image.
        """
        self.mesh = None
        self.point_cloud = None
        self.colors = None
        self.texture_resolution = texture_resolution

    def apply_texture(self, mesh, point_cloud):
        """
        Apply the complete texturing process to the given mesh.

        Args:
            mesh (pyvista.PolyData): The mesh to be textured.
            point_cloud (numpy.ndarray): The point cloud data with color information.

        Returns:
            pyvista.PolyData: The textured mesh.

        Raises:
            TexturingError: If the texturing process fails.
        """
        try:
            self.mesh = mesh
            self.point_cloud = point_cloud[:, :3]
            self.colors = point_cloud[:, 3:] if point_cloud.shape[1] > 3 else None

            self.map_colors_to_mesh()
            self.apply_smart_uv_mapping()
            self.smooth_texture()

            return self.mesh
        except Exception as e:
            raise TexturingError(f"Failed to apply texture: {str(e)}")

    def map_colors_to_mesh(self):
        """
        Map colors from the point cloud to the mesh vertices.

        Raises:
            TexturingError: If color mapping fails.
        """
        if self.colors is None:
            raise TexturingError("No color data available for mapping")

        tree = cKDTree(self.point_cloud)
        distances, indices = tree.query(self.mesh.points)
        vertex_colors = self.colors[indices]

        if vertex_colors.max() > 1.0:
            vertex_colors = vertex_colors / 255.0

        self.mesh.point_data["RGB"] = vertex_colors

    def apply_smart_uv_mapping(self):
        """
        Apply an appropriate UV mapping method based on the mesh's shape.

        This method analyzes the mesh's dimensions and chooses between planar,
        cylindrical, or spherical UV mapping.
        """
        bounds = self.mesh.bounds
        dimensions = np.array([bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]])
        aspect_ratio = dimensions / np.max(dimensions)

        if aspect_ratio[2] < 0.5:
            self.apply_planar_uv_mapping()
        elif aspect_ratio[0] > 0.8 and aspect_ratio[1] > 0.8:
            self.apply_spherical_uv_mapping()
        else:
            self.apply_cylindrical_uv_mapping()

    def apply_planar_uv_mapping(self):
        """
        Apply planar UV mapping to the mesh.

        This method projects the mesh onto a plane and uses the x and y coordinates
        as UV coordinates. It adds a small epsilon to avoid division by zero when
        the mesh has points with identical coordinates.
        """
        bounds = self.mesh.bounds
        min_point = np.array([bounds[0], bounds[2], bounds[4]])
        max_point = np.array([bounds[1], bounds[3], bounds[5]])

        # Add small epsilon to avoid division by zero
        epsilon = 1e-6
        normalized_points = (self.mesh.points - min_point) / (np.maximum(max_point - min_point, epsilon))

        u = normalized_points[:, 0]
        v = normalized_points[:, 1]

        self.mesh.point_data["UV"] = np.column_stack((u, v))

    def apply_cylindrical_uv_mapping(self):
        """Apply cylindrical UV mapping to the mesh."""
        center = self.mesh.center
        bounds = self.mesh.bounds
        height = bounds[5] - bounds[4]
        normalized_points = self.mesh.points - center
        r = np.sqrt(normalized_points[:, 0] ** 2 + normalized_points[:, 1] ** 2)
        theta = np.arctan2(normalized_points[:, 1], normalized_points[:, 0])
        u = (theta + np.pi) / (2 * np.pi)
        v = (normalized_points[:, 2] - bounds[4]) / height
        self.mesh.point_data["UV"] = np.column_stack((u, v))

    def apply_spherical_uv_mapping(self):
        """Apply spherical UV mapping to the mesh."""
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

        This method applies a custom smoothing to the color data on the mesh.

        Args:
            iterations (int): Number of smoothing iterations to perform. Default is 5.
        """
        for _ in range(iterations):
            smooth_colors = np.zeros_like(self.mesh.point_data['RGB'])
            n_points = self.mesh.n_points

            # Get the faces of the mesh
            faces = self.mesh.faces

            # Create a list to store neighboring points for each point
            neighbors = [set() for _ in range(n_points)]

            # Populate the neighbors list
            i = 0
            while i < len(faces):
                n_vertices = faces[i]
                for j in range(1, n_vertices + 1):
                    for k in range(1, n_vertices + 1):
                        if j != k:
                            neighbors[faces[i + j]].add(faces[i + k])
                i += n_vertices + 1

            # For each point, average the colors of its neighbors
            for i in range(n_points):
                if neighbors[i]:
                    neighbor_colors = self.mesh.point_data['RGB'][list(neighbors[i])]
                    smooth_colors[i] = np.mean(neighbor_colors, axis=0)
                else:
                    smooth_colors[i] = self.mesh.point_data['RGB'][i]

            self.mesh.point_data['RGB'] = smooth_colors

    def generate_texture_image(self):
        """
        Generate a texture image from the mesh's color and UV data.

        Returns:
            numpy.ndarray: The generated texture image.

        Raises:
            TexturingError: If texture image generation fails.
        """
        if 'RGB' not in self.mesh.point_data or 'UV' not in self.mesh.point_data:
            raise TexturingError("Mesh must have RGB and UV data before generating texture image.")

        colors = self.mesh.point_data['RGB']
        uv_coords = self.mesh.point_data['UV']

        texture_image = np.zeros((self.texture_resolution, self.texture_resolution, 3), dtype=np.float32)
        mask = np.zeros((self.texture_resolution, self.texture_resolution), dtype=bool)

        for i, uv in enumerate(uv_coords):
            x = int(uv[0] * (self.texture_resolution - 1))
            y = int((1 - uv[1]) * (self.texture_resolution - 1))
            texture_image[y, x] = colors[i]
            mask[y, x] = True

        distance = ndimage.distance_transform_edt(~mask)
        indices = ndimage.distance_transform_edt(~mask, return_distances=False, return_indices=True)

        for channel in range(3):
            channel_image = texture_image[:, :, channel]
            texture_image[:, :, channel] = channel_image[tuple(indices)]

        return texture_image
