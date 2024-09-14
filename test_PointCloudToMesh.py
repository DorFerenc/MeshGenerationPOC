import unittest
import numpy as np
import pyvista as pv
from PointCloudToMesh import PointCloudToMesh
import os
import tempfile


class TestPointCloudToMesh(unittest.TestCase):
    """
    A test suite for the PointCloudToMesh class.

    This class contains unit tests to verify the functionality of the PointCloudToMesh class,
    including data loading, mesh generation, and mesh refinement operations.
    """

    def setUp(self):
        """
        Set up the test environment before each test method.

        This method creates a temporary CSV file with sample point cloud data and
        initializes a PointCloudToMesh object for testing.
        """
        self.pc_to_mesh = PointCloudToMesh()

        # Create a temporary CSV file with sample point cloud data
        self.temp_dir = tempfile.mkdtemp()
        self.csv_file = os.path.join(self.temp_dir, "test_point_cloud.csv")
        self.sample_data = np.random.rand(100, 3)
        np.savetxt(self.csv_file, self.sample_data, delimiter=',')

    def tearDown(self):
        """
        Clean up the test environment after each test method.

        This method removes the temporary CSV file and directory created during setUp.
        """
        os.remove(self.csv_file)
        os.rmdir(self.temp_dir)

    def test_load_point_cloud_from_csv(self):
        """
        Test the load_point_cloud_from_csv method.

        This test verifies that the method correctly loads point cloud data from a CSV file.
        """
        self.pc_to_mesh.load_point_cloud_from_csv(self.csv_file)
        self.assertIsNotNone(self.pc_to_mesh.point_cloud)
        self.assertEqual(self.pc_to_mesh.point_cloud.shape, (100, 3))
        np.testing.assert_array_equal(self.pc_to_mesh.point_cloud, self.sample_data)

    def test_load_point_cloud_from_csv_file_not_found(self):
        """
        Test the load_point_cloud_from_csv method with a non-existent file.

        This test verifies that the method raises a FileNotFoundError when given an invalid file path.
        """
        with self.assertRaises(FileNotFoundError):
            self.pc_to_mesh.load_point_cloud_from_csv("non_existent_file.csv")

    def test_generate_mesh(self):
        """
        Test the generate_mesh method.

        This test verifies that the method successfully generates a mesh from the loaded point cloud data.
        """
        self.pc_to_mesh.load_point_cloud_from_csv(self.csv_file)
        self.pc_to_mesh.generate_mesh()
        self.assertIsNotNone(self.pc_to_mesh.mesh)
        self.assertIsInstance(self.pc_to_mesh.mesh, pv.PolyData)

    def test_generate_mesh_without_point_cloud(self):
        """
        Test the generate_mesh method without loading point cloud data first.

        This test verifies that the method raises a ValueError when called before loading point cloud data.
        """
        with self.assertRaises(ValueError):
            self.pc_to_mesh.generate_mesh()

    # def test_apply_laplacian_smoothing(self):
    #     """
    #     Test the apply_laplacian_smoothing method.
    #
    #     This test verifies that the method successfully applies Laplacian smoothing to the generated mesh.
    #     """
    #     self.pc_to_mesh.load_point_cloud_from_csv(self.csv_file)
    #     self.pc_to_mesh.generate_mesh()
    #     original_points = self.pc_to_mesh.mesh.points.copy()
    #     self.pc_to_mesh.apply_laplacian_smoothing()
    #     self.assertFalse(np.array_equal(original_points, self.pc_to_mesh.mesh.points))

    def test_apply_bilateral_smoothing(self):
        """
        Test the apply_bilateral_smoothing method.

        This test verifies that the method successfully applies bilateral smoothing to the generated mesh.
        """
        self.pc_to_mesh.load_point_cloud_from_csv(self.csv_file)
        self.pc_to_mesh.generate_mesh()
        original_points = self.pc_to_mesh.mesh.points.copy()
        self.pc_to_mesh.apply_bilateral_smoothing()
        self.assertFalse(np.array_equal(original_points, self.pc_to_mesh.mesh.points))

    def test_refine_mesh(self):
        """
        Test the refine_mesh method.

        This test verifies that the method successfully refines the generated mesh by applying both
        Laplacian and bilateral smoothing.
        """
        self.pc_to_mesh.load_point_cloud_from_csv(self.csv_file)
        self.pc_to_mesh.generate_mesh()
        original_points = self.pc_to_mesh.mesh.points.copy()
        self.pc_to_mesh.refine_mesh()
        self.assertFalse(np.array_equal(original_points, self.pc_to_mesh.mesh.points))


if __name__ == '__main__':
    unittest.main()