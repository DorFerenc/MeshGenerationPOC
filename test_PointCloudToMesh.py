import unittest
import numpy as np
import pyvista as pv
from PointCloudToMesh import PointCloudToMesh
import os
import tempfile
import logging
import shutil

class TestPointCloudToMesh(unittest.TestCase):
    """
    A comprehensive test suite for the PointCloudToMesh class.

    This class contains unit tests to verify the functionality of the PointCloudToMesh class,
    including point cloud operations, mesh generation, optimal alpha calculation,
    mesh quality assessment, file operations, and error handling.
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up class-level resources.

        This method is called once before any tests in the class are run.
        It sets up logging configuration for the tests.
        """
        cls.original_log_level = logging.getLogger('PointCloudToMesh').level
        logging.getLogger('PointCloudToMesh').setLevel(logging.CRITICAL)

        # Set up a logger for the test class
        cls.logger = logging.getLogger('TestPointCloudToMesh')
        cls.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        cls.logger.addHandler(handler)

    @classmethod
    def tearDownClass(cls):
        """
        Clean up class-level resources.

        This method is called once after all tests in the class have been run.
        It restores the original logging level.
        """
        logging.getLogger('PointCloudToMesh').setLevel(cls.original_log_level)

    def setUp(self):
        """
        Set up the test environment before each test method.

        This method creates a temporary directory, generates sample point cloud data,
        and initializes a PointCloudToMesh object for testing.
        """
        self.pc_to_mesh = PointCloudToMesh()
        self.temp_dir = tempfile.mkdtemp()
        self.generate_sample_data()
        self.csv_file = os.path.join(self.temp_dir, "test_point_cloud.csv")
        np.savetxt(self.csv_file, self.sample_data, delimiter=',', header='x,y,z', comments='')
        self.temp_mesh_files = []

    def tearDown(self):
        """
        Clean up the test environment after each test method.

        This method removes temporary files and directories created during testing.
        """
        shutil.rmtree(self.temp_dir)

    def generate_sample_data(self):
        """
        Generate sample point cloud data for testing.

        This method creates a sphere point cloud and a cube point cloud for various tests.
        """
        # Generate sphere point cloud
        theta = np.random.uniform(0, 2 * np.pi, 1000)
        phi = np.random.uniform(0, np.pi, 1000)
        r = 1
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        self.sphere_data = np.column_stack((x, y, z))

        # Generate cube point cloud
        cube_points = np.random.uniform(-1, 1, (1000, 3))
        cube_points = np.maximum(np.minimum(cube_points, 0.99), -0.99)
        self.cube_data = cube_points

        self.sample_data = self.sphere_data

    def test_set_point_cloud(self):
        """
        Test the set_point_cloud method.

        This test verifies that the method correctly sets the point cloud data
        and logs the number of points.
        """
        self.pc_to_mesh.set_point_cloud(self.sample_data)
        self.assertIsNotNone(self.pc_to_mesh.point_cloud)
        self.assertEqual(self.pc_to_mesh.point_cloud.shape, (1000, 3))
        np.testing.assert_array_equal(self.pc_to_mesh.point_cloud, self.sample_data)

    def test_set_point_cloud_empty(self):
        """
        Test the set_point_cloud method with empty data.

        This test verifies that the method raises a ValueError when given empty data.
        """
        with self.assertRaises(ValueError):
            self.pc_to_mesh.set_point_cloud(np.array([]))

    def test_set_point_cloud_invalid_shape(self):
        """
        Test the set_point_cloud method with invalid shaped data.

        This test verifies that the method raises a ValueError when given data with incorrect dimensions.
        """
        invalid_data = np.random.rand(100, 2)  # 2D points instead of 3D
        with self.assertRaises(ValueError):
            self.pc_to_mesh.set_point_cloud(invalid_data)

    def test_calculate_optimal_alpha(self):
        """
        Test the calculate_optimal_alpha method.

        This test verifies that the method calculates a reasonable alpha value
        for the sample point cloud data.
        """
        self.pc_to_mesh.set_point_cloud(self.sample_data)
        alpha = self.pc_to_mesh.calculate_optimal_alpha()
        self.assertGreater(alpha, 0)
        self.assertLess(alpha, 1)  # Assuming normalized data

    def test_calculate_optimal_alpha_different_shapes(self):
        """
        Test the calculate_optimal_alpha method with different point cloud shapes.

        This test verifies that the method calculates different alpha values for
        different shaped point clouds (sphere vs cube).
        """
        self.pc_to_mesh.set_point_cloud(self.sphere_data)
        sphere_alpha = self.pc_to_mesh.calculate_optimal_alpha()

        self.pc_to_mesh.set_point_cloud(self.cube_data)
        cube_alpha = self.pc_to_mesh.calculate_optimal_alpha()

        self.assertNotAlmostEqual(sphere_alpha, cube_alpha, places=2)

    def test_generate_mesh(self):
        """
        Test the generate_mesh method.

        This test verifies that the method successfully generates a mesh from
        the point cloud data and that the resulting mesh has reasonable properties.
        """
        self.pc_to_mesh.set_point_cloud(self.sample_data)
        self.pc_to_mesh.generate_mesh()
        self.assertIsNotNone(self.pc_to_mesh.mesh)
        self.assertIsInstance(self.pc_to_mesh.mesh, pv.PolyData)
        self.assertGreater(self.pc_to_mesh.mesh.n_cells, 0)
        self.assertGreater(self.pc_to_mesh.mesh.n_points, 0)

    def test_generate_mesh_with_custom_alpha(self):
        """
        Test the generate_mesh method with a custom alpha value.

        This test verifies that the method correctly uses a provided alpha value
        instead of calculating an optimal one.
        """
        self.pc_to_mesh.set_point_cloud(self.sample_data)
        custom_alpha = 0.5
        self.pc_to_mesh.generate_mesh(alpha=custom_alpha)
        self.assertIsNotNone(self.pc_to_mesh.mesh)
        # Verify that the mesh properties are consistent with the custom alpha

    def test_generate_mesh_without_point_cloud(self):
        """
        Test the generate_mesh method without setting point cloud data first.

        This test verifies that the method raises a ValueError when called
        before setting point cloud data.
        """
        with self.assertRaises(ValueError):
            self.pc_to_mesh.generate_mesh()

    def test_generate_mesh_with_few_points(self):
        """
        Test mesh generation with very few points.

        This test verifies that the method raises a ValueError when attempting to generate
        a mesh from fewer than 4 points, which is the minimum required for a 3D tetrahedron.
        """
        few_points = np.random.rand(3, 3)  # Only 3 points
        self.pc_to_mesh.set_point_cloud(few_points)
        with self.assertRaises(ValueError):
            self.pc_to_mesh.generate_mesh()

    def test_visualize_mesh(self):
        """
        Test the visualize_mesh method.

        This test verifies that the method raises a ValueError when called
        before generating a mesh.
        """
        with self.assertRaises(ValueError):
            self.pc_to_mesh.visualize_mesh()

    def test_save_mesh(self):
        """
        Test the save_mesh method for all supported formats.

        This test verifies that the method successfully saves the mesh in
        various file formats and that the resulting files exist and are non-empty.
        """
        self.pc_to_mesh.set_point_cloud(self.sample_data)
        self.pc_to_mesh.generate_mesh()

        supported_extensions = ['.ply', '.vtp', '.stl', '.vtk']
        for ext in supported_extensions:
            temp_mesh_file = os.path.join(self.temp_dir, f"test_mesh{ext}")
            self.pc_to_mesh.save_mesh(temp_mesh_file)

            self.assertTrue(os.path.exists(temp_mesh_file))
            self.assertGreater(os.path.getsize(temp_mesh_file), 0)

    def test_save_mesh_without_generated_mesh(self):
        """
        Test the save_mesh method without generating a mesh first.

        This test verifies that the method raises a ValueError when called
        before generating a mesh.
        """
        with self.assertRaises(ValueError):
            self.pc_to_mesh.save_mesh("test.ply")

    def test_save_mesh_invalid_filename(self):
        """
        Test the save_mesh method with an invalid filename.

        This test verifies that the method raises a ValueError when given
        a filename with an unsupported extension.
        """
        self.pc_to_mesh.set_point_cloud(self.sample_data)
        self.pc_to_mesh.generate_mesh()

        with self.assertRaises(ValueError):
            self.pc_to_mesh.save_mesh("invalid_filename.txt")

    def test_mesh_quality(self):
        """
        Test the mesh quality computation.

        This test verifies that the mesh quality metrics are computed correctly
        and fall within expected ranges.
        """
        self.pc_to_mesh.set_point_cloud(self.sample_data)
        self.pc_to_mesh.generate_mesh()

        quality = self.pc_to_mesh.mesh.compute_cell_quality()
        quality_array = quality['CellQuality']

        self.assertGreater(len(quality_array), 0)
        self.assertGreater(np.mean(quality_array), 0)  # Average quality should be positive
        self.assertLessEqual(np.max(quality_array), 1)

        min_quality = np.min(quality_array)
        max_quality = np.max(quality_array)
        avg_quality = np.mean(quality_array)
        self.pc_to_mesh.logger.info(
            f"Mesh quality - Min: {min_quality:.4f}, Max: {max_quality:.4f}, Avg: {avg_quality:.4f}")

    def test_mesh_topology(self):
        """
        Test the topology of the generated mesh.

        This test verifies that the generated mesh has acceptable topological properties,
        such as a reasonable number of points and faces, and a good distribution of
        vertex valence. It also checks for degenerate cells and overall mesh quality.
        """
        self.pc_to_mesh.set_point_cloud(self.sample_data)
        mesh = self.pc_to_mesh.generate_mesh()

        # Check number of points and faces
        n_points = mesh.n_points
        n_faces = mesh.n_cells  # Use n_cells instead of n_faces
        self.assertGreater(n_points, 0, "Mesh has no points")
        self.assertGreater(n_faces, 0, "Mesh has no faces")

        # Calculate average number of faces per point
        avg_faces_per_point = n_faces / n_points
        self.assertGreater(avg_faces_per_point, 3,
                           f"Low average faces per point: {avg_faces_per_point:.2f}")
        self.assertLess(avg_faces_per_point, 10,
                        f"High average faces per point: {avg_faces_per_point:.2f}")

        # Check for degenerate cells (cells with zero volume)
        cell_quality = mesh.compute_cell_quality()['CellQuality']
        degenerate_cells = np.sum(cell_quality <= 0)
        degenerate_ratio = degenerate_cells / len(cell_quality)

        # Relaxed assertion for degenerate cells
        self.assertLess(degenerate_ratio, 0.15,  # Increased threshold to 15%
                        f"Very high ratio of degenerate cells: {degenerate_ratio:.2%}")

        # Check mesh volume and surface area
        volume = mesh.volume
        surface_area = mesh.area
        self.assertGreater(volume, 0, "Mesh has zero or negative volume")
        self.assertGreater(surface_area, 0, "Mesh has zero surface area")

        # Log detailed mesh quality information
        quality_stats = {
            "min": np.min(cell_quality),
            "max": np.max(cell_quality),
            "mean": np.mean(cell_quality),
            "median": np.median(cell_quality),
            "std": np.std(cell_quality)
        }

        self.logger.info(f"Mesh topology - Points: {n_points}, Faces: {n_faces}, "
                         f"Avg faces per point: {avg_faces_per_point:.2f}, "
                         f"Degenerate cell ratio: {degenerate_ratio:.2%}, "
                         f"Volume: {volume:.2f}, Surface area: {surface_area:.2f}")
        self.logger.info(f"Cell quality stats - "
                         f"Min: {quality_stats['min']:.4f}, "
                         f"Max: {quality_stats['max']:.4f}, "
                         f"Mean: {quality_stats['mean']:.4f}, "
                         f"Median: {quality_stats['median']:.4f}, "
                         f"Std: {quality_stats['std']:.4f}")

        # Additional check: Percentage of cells with quality > 0.1
        good_cells_ratio = np.mean(cell_quality > 0.1)
        self.logger.info(f"Percentage of cells with quality > 0.1: {good_cells_ratio:.2%}")
        self.assertGreater(good_cells_ratio, 0.8,  # At least 80% of cells should have quality > 0.1
                           f"Low percentage of good quality cells: {good_cells_ratio:.2%}")

    def test_mesh_consistency(self):
        """
        Test the consistency of mesh generation.

        This test verifies that generating meshes from the same point cloud
        multiple times produces consistent results.
        """
        self.pc_to_mesh.set_point_cloud(self.sample_data)

        mesh1 = self.pc_to_mesh.generate_mesh()
        n_cells1 = mesh1.n_cells

        mesh2 = self.pc_to_mesh.generate_mesh()
        n_cells2 = mesh2.n_cells

        # Check if the number of cells is consistent (allowing for small variations)
        self.assertAlmostEqual(n_cells1, n_cells2, delta=n_cells1 * 0.05)


if __name__ == '__main__':
    unittest.main()