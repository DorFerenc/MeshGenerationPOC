import unittest
import numpy as np
import pyvista as pv
from reconstruction.mesh_generator import MeshGenerator
from reconstruction.utils import MeshGenerationError
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestMeshGenerator(unittest.TestCase):
    def setUp(self):
        self.mesh_generator = MeshGenerator()
        self.sphere_points = self.generate_sphere_points()
        self.cube_points = self.generate_cube_points()

    def generate_sphere_points(self, n=1000):
        phi = np.random.uniform(0, np.pi, n)
        theta = np.random.uniform(0, 2 * np.pi, n)
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        return np.column_stack((x, y, z))

    def generate_cube_points(self, n=1000):
        points = np.random.rand(n, 3) * 2 - 1
        return points

    def test_generate_mesh_sphere(self):
        mesh, alpha = self.mesh_generator.generate_mesh(self.sphere_points)
        self.assertIsInstance(mesh, pv.PolyData)
        self.assertGreater(mesh.n_cells, 0)
        self.assertGreater(mesh.n_points, 0)
        self.assertIsInstance(alpha, float)

    def test_generate_mesh_cube(self):
        mesh, alpha = self.mesh_generator.generate_mesh(self.cube_points)
        self.assertIsInstance(mesh, pv.PolyData)
        self.assertGreater(mesh.n_cells, 0)
        self.assertGreater(mesh.n_points, 0)
        self.assertIsInstance(alpha, float)

    def test_calculate_optimal_alpha_sphere(self):
        alpha = self.mesh_generator.calculate_optimal_alpha(self.sphere_points)
        self.assertGreater(alpha, 0)

    def test_calculate_optimal_alpha_cube(self):
        alpha = self.mesh_generator.calculate_optimal_alpha(self.cube_points)
        self.assertGreater(alpha, 0)

    def test_is_cube_like(self):
        logger.info(f"Testing is_cube_like for cube points")
        logger.info(f"Cube points shape: {self.cube_points.shape}")
        self.assertTrue(self.mesh_generator.is_cube_like(self.cube_points))

        logger.info(f"Testing is_cube_like for sphere points")
        logger.info(f"Sphere points shape: {self.sphere_points.shape}")
        self.assertFalse(self.mesh_generator.is_cube_like(self.sphere_points))

    def test_log_mesh_quality(self):
        mesh, _ = self.mesh_generator.generate_mesh(self.sphere_points)
        self.mesh_generator.mesh = mesh
        self.mesh_generator.log_mesh_quality()
        # This test mainly checks if the method runs without errors
        # You might want to capture and check the logged output in a more advanced test

    def test_generate_mesh_with_few_points(self):
        with self.assertRaises(MeshGenerationError):
            self.mesh_generator.generate_mesh(np.array([[0, 0, 0], [1, 1, 1]]))

    def test_generate_mesh_with_collinear_points(self):
        collinear_points = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])
        logger.info(f"Testing with collinear points: {collinear_points}")
        with self.assertRaises(MeshGenerationError):
            self.mesh_generator.generate_mesh(collinear_points)

    def test_mesh_consistency(self):
        mesh1, _ = self.mesh_generator.generate_mesh(self.sphere_points)
        mesh2, _ = self.mesh_generator.generate_mesh(self.sphere_points)
        self.assertAlmostEqual(mesh1.n_cells, mesh2.n_cells, delta=mesh1.n_cells * 0.1)


if __name__ == '__main__':
    unittest.main()
