import unittest
import numpy as np
import pyvista as pv
from PointCloudToMesh import PointCloudToMesh
import os
import tempfile
import logging


class TestPointCloudToMesh(unittest.TestCase):
    """
    A comprehensive test suite for the PointCloudToMesh class.

    This class contains unit tests to verify the functionality of the PointCloudToMesh class,
    including point cloud operations, mesh generation, optimal alpha calculation,
    mesh quality assessment, and file operations.
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up class-level resources.

        This method is called once before any tests in the class are run.
        It sets up logging configuration for the tests.
        """
        # Store the original logging level
        cls.original_log_level = logging.getLogger('PointCloudToMesh').level
        # Set the logging level to CRITICAL to suppress most messages during tests
        logging.getLogger('PointCloudToMesh').setLevel(logging.CRITICAL)

    @classmethod
    def tearDownClass(cls):
        """
        Clean up class-level resources.

        This method is called once after all tests in the class have been run.
        It restores the original logging level.
        """
        # Restore the original logging level
        logging.getLogger('PointCloudToMesh').setLevel(cls.original_log_level)

    def setUp(self):
        """
        Set up the test environment before each test method.

        This method creates a temporary directory, generates sample point cloud data,
        and initializes a PointCloudToMesh object for testing.
        """
        self.pc_to_mesh = PointCloudToMesh()

        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()

        # Generate sample point cloud data (sphere)
        theta = np.random.uniform(0, 2 * np.pi, 1000)
        phi = np.random.uniform(0, np.pi, 1000)
        r = 1
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        self.sample_data = np.column_stack((x, y, z))

        # Save sample data to a temporary CSV file
        self.csv_file = os.path.join(self.temp_dir, "test_point_cloud.csv")
        np.savetxt(self.csv_file, self.sample_data, delimiter=',', header='x,y,z', comments='')

        self.temp_mesh_files = []

    def tearDown(self):
        """
        Clean up the test environment after each test method.

        This method removes temporary files and directories created during testing.
        """
        os.remove(self.csv_file)
        for temp_file in self.temp_mesh_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        os.rmdir(self.temp_dir)

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
        # Add more specific assertions about the mesh properties if possible

    def test_generate_mesh_without_point_cloud(self):
        """
        Test the generate_mesh method without setting point cloud data first.

        This test verifies that the method raises a ValueError when called
        before setting point cloud data.
        """
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
            self.temp_mesh_files.append(temp_mesh_file)
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
        and fall within expected ranges. We allow for some negative quality values
        but ensure that the average quality is positive and the maximum quality
        does not exceed 1.
        """
        self.pc_to_mesh.set_point_cloud(self.sample_data)
        self.pc_to_mesh.generate_mesh()

        quality = self.pc_to_mesh.mesh.compute_cell_quality()
        quality_array = quality['CellQuality']

        self.assertGreater(len(quality_array), 0)
        self.assertGreater(np.mean(quality_array), 0)  # Average quality should be positive
        self.assertLessEqual(np.max(quality_array), 1)

        # Log the quality statistics for information
        min_quality = np.min(quality_array)
        max_quality = np.max(quality_array)
        avg_quality = np.mean(quality_array)
        self.pc_to_mesh.logger.info(
            f"Mesh quality - Min: {min_quality:.4f}, Max: {max_quality:.4f}, Avg: {avg_quality:.4f}")


if __name__ == '__main__':
    unittest.main()