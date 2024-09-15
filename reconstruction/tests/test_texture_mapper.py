import unittest
import numpy as np
import pyvista as pv
from reconstruction.texture_mapper import TextureMapper
from reconstruction.utils import TexturingError

class TestTextureMapper(unittest.TestCase):
    def setUp(self):
        self.texture_mapper = TextureMapper()
        self.mesh = pv.Cube()
        self.point_cloud = np.array([
            [0, 0, 0, 255, 0, 0],
            [1, 0, 0, 0, 255, 0],
            [0, 1, 0, 0, 0, 255],
            [1, 1, 0, 255, 255, 0],
            [0, 0, 1, 255, 0, 255],
            [1, 0, 1, 0, 255, 255],
            [0, 1, 1, 128, 128, 128],
            [1, 1, 1, 255, 255, 255]
        ])

    def test_apply_texture(self):
        textured_mesh = self.texture_mapper.apply_texture(self.mesh, self.point_cloud)
        self.assertIn('RGB', textured_mesh.point_data)
        self.assertIn('UV', textured_mesh.point_data)

    def test_map_colors_to_mesh(self):
        self.texture_mapper.mesh = self.mesh
        self.texture_mapper.point_cloud = self.point_cloud[:, :3]
        self.texture_mapper.colors = self.point_cloud[:, 3:]
        self.texture_mapper.map_colors_to_mesh()
        self.assertIn('RGB', self.texture_mapper.mesh.point_data)

    def test_apply_smart_uv_mapping(self):
        self.texture_mapper.mesh = self.mesh
        self.texture_mapper.apply_smart_uv_mapping()
        self.assertIn('UV', self.texture_mapper.mesh.point_data)

    def test_generate_texture_image(self):
        self.texture_mapper.mesh = self.mesh
        self.texture_mapper.point_cloud = self.point_cloud[:, :3]
        self.texture_mapper.colors = self.point_cloud[:, 3:] / 255.0
        self.texture_mapper.map_colors_to_mesh()
        self.texture_mapper.apply_smart_uv_mapping()
        texture_image = self.texture_mapper.generate_texture_image()
        self.assertEqual(texture_image.shape, (1024, 1024, 3))

    def test_apply_texture_without_colors(self):
        point_cloud_no_color = self.point_cloud[:, :3]
        with self.assertRaises(TexturingError):
            self.texture_mapper.apply_texture(self.mesh, point_cloud_no_color)

    def test_apply_planar_uv_mapping(self):
        self.texture_mapper.mesh = pv.Plane()
        self.texture_mapper.apply_planar_uv_mapping()
        uv_coords = self.texture_mapper.mesh.point_data['UV']
        self.assertTrue(np.all(uv_coords >= 0) and np.all(uv_coords <= 1))

    def test_apply_cylindrical_uv_mapping(self):
        self.texture_mapper.mesh = pv.Cylinder()
        self.texture_mapper.apply_cylindrical_uv_mapping()
        uv_coords = self.texture_mapper.mesh.point_data['UV']
        self.assertTrue(np.all(uv_coords >= 0) and np.all(uv_coords <= 1))

    def test_apply_spherical_uv_mapping(self):
        self.texture_mapper.mesh = pv.Sphere()
        self.texture_mapper.apply_spherical_uv_mapping()
        uv_coords = self.texture_mapper.mesh.point_data['UV']
        self.assertTrue(np.all(uv_coords >= 0) and np.all(uv_coords <= 1))

    def test_smooth_texture(self):
        """
        Test the smooth_texture method of the TextureMapper class.

        This test creates a sphere mesh with a striped texture and applies
        smoothing. It then checks if the smoothing operation has changed the
        colors by comparing the colors before and after smoothing.
        """
        # Create a sphere mesh
        sphere = pv.Sphere(theta_resolution=32, phi_resolution=32)

        # Create a striped texture
        n_points = sphere.n_points
        stripes = np.zeros((n_points, 3))
        for i in range(n_points):
            theta = np.arctan2(sphere.points[i, 1], sphere.points[i, 0])
            stripes[i] = [1, 1, 1] if int(theta * 5 / np.pi) % 2 == 0 else [0, 0, 0]

        self.texture_mapper.mesh = sphere
        self.texture_mapper.mesh.point_data['RGB'] = stripes

        # Store original colors
        original_colors = self.texture_mapper.mesh.point_data['RGB'].copy()

        # Apply smoothing
        self.texture_mapper.smooth_texture(iterations=20)

        # Get smoothed colors
        smoothed_colors = self.texture_mapper.mesh.point_data['RGB']

        # Check if colors have changed
        diff = np.abs(original_colors - smoothed_colors).sum()
        self.assertGreater(diff, 0, "Smoothing had no effect on the colors")

        # Check if the number of unique colors has increased (indicating smoothing)
        original_unique_colors = np.unique(original_colors, axis=0).shape[0]
        smoothed_unique_colors = np.unique(smoothed_colors, axis=0).shape[0]
        self.assertGreater(smoothed_unique_colors, original_unique_colors,
                           "Smoothing should increase the number of unique colors")


if __name__ == '__main__':
    unittest.main()
