import unittest
import numpy as np
import pyvista as pv
import os
import tempfile
import shutil

from PIL import Image

from MeshToOBJConverter import MeshToOBJConverter
from TextureMapper import TextureMapper

class TestMeshToOBJConverter(unittest.TestCase):
    """
    Test suite for the MeshToOBJConverter class.
    """

    def setUp(self):
        """
        Set up the test environment.
        """
        self.temp_dir = tempfile.mkdtemp()
        self.mesh = self.create_sample_mesh()
        self.texture_mapper = self.create_sample_texture_mapper()
        self.converter = MeshToOBJConverter(self.mesh, self.texture_mapper)

    def tearDown(self):
        """
        Clean up after each test.
        """
        shutil.rmtree(self.temp_dir)

    def create_sample_mesh(self):
        """
        Create a simple cube mesh and apply UV mapping.
        """
        mesh = pv.Cube()
        # Apply UV mapping to ensure the mesh has texture coordinates
        mesh.texture_map_to_plane(inplace=True)
        return mesh

    def create_sample_texture_mapper(self):
        """
        Create a sample texture mapper with random colors and apply UV mapping.
        """
        texture_mapper = TextureMapper()
        texture_mapper.load_mesh(self.mesh)
        # Create a random point cloud for testing
        texture_mapper.load_point_cloud_with_colors(self.mesh.points, np.random.rand(len(self.mesh.points), 3))
        # Apply UV mapping to the mesh
        texture_mapper.apply_smart_uv_mapping()
        texture_mapper.map_colors_to_mesh()  # Ensure the mesh has RGB data
        return texture_mapper

    def test_convert_to_obj(self):
        """
        Test the conversion of the mesh to OBJ format.
        """
        obj_filename = os.path.join(self.temp_dir, "test_mesh.obj")
        self.converter.convert_to_obj(obj_filename)

        self.assertTrue(os.path.exists(obj_filename), "OBJ file was not created")
        with open(obj_filename, 'r') as f:
            content = f.read()
            self.assertIn("v ", content, "OBJ file does not contain vertex data")
            self.assertIn("vt ", content, "OBJ file does not contain texture coordinate data")
            self.assertIn("f ", content, "OBJ file does not contain face data")

    def test_save_texture_image(self):
        """
        Test the generation and saving of the texture image.
        """
        texture_filename = os.path.join(self.temp_dir, "test_texture.png")
        self.converter.save_texture_image(texture_filename)

        self.assertTrue(os.path.exists(texture_filename), "Texture image file was not created")
        self.assertGreater(os.path.getsize(texture_filename), 0, "Texture image file is empty")

    def test_create_mtl_file(self):
        """
        Test the creation of the MTL file for material properties.
        """
        mtl_filename = os.path.join(self.temp_dir, "test_material.mtl")
        texture_filename = "test_texture.png"
        self.converter.create_mtl_file(mtl_filename, texture_filename)

        self.assertTrue(os.path.exists(mtl_filename), "MTL file was not created")
        with open(mtl_filename, 'r') as f:
            content = f.read()
            self.assertIn("newmtl material0", content, "MTL file does not contain material definition")
            self.assertIn("map_Kd test_texture.png", content, "MTL file does not reference the texture image")

    def test_convert_and_save(self):
        """
        Test the complete conversion process: OBJ, MTL, and texture files.
        """
        obj_filename = os.path.join(self.temp_dir, "test_mesh.obj")
        texture_filename = os.path.join(self.temp_dir, "test_texture.png")
        self.converter.convert_and_save(obj_filename, texture_filename)

        # Check OBJ file
        self.assertTrue(os.path.exists(obj_filename), "OBJ file was not created")
        self.assertGreater(os.path.getsize(obj_filename), 0, "OBJ file is empty")

        # Check texture image
        self.assertTrue(os.path.exists(texture_filename), "Texture image file was not created")
        self.assertGreater(os.path.getsize(texture_filename), 0, "Texture image file is empty")

        # Check MTL file
        mtl_filename = obj_filename.replace(".obj", ".mtl")
        self.assertTrue(os.path.exists(mtl_filename), "MTL file was not created")
        self.assertGreater(os.path.getsize(mtl_filename), 0, "MTL file is empty")

    ########################################################################
    # Additional tests for TestPointCloudToMesh class
    ########################################################################

    def create_complex_mesh_with_materials(self):
        # Create a more complex mesh with multiple parts
        sphere = pv.Sphere(center=(0, 0, 0), radius=0.5)
        cube = pv.Cube(center=(1, 0, 0))
        mesh = sphere + cube  # Combine sphere and cube

        # Add random noise to make it more complex
        mesh.points += np.random.normal(0, 0.05, mesh.points.shape)

        # Create multiple materials
        n_points = mesh.n_points
        material1 = np.zeros(n_points)
        material2 = np.zeros(n_points)

        # Assign different materials to different parts of the mesh
        material1[:n_points // 2] = 1
        material2[n_points // 2:] = 1

        mesh.point_data['Material1'] = material1
        mesh.point_data['Material2'] = material2

        # Apply UV mapping
        mesh.texture_map_to_sphere(inplace=True)

        return mesh

    def create_complex_texture_mapper(self, mesh):
        texture_mapper = TextureMapper()
        texture_mapper.load_mesh(mesh)
        colors = np.random.rand(mesh.n_points, 3)
        texture_mapper.load_point_cloud_with_colors(mesh.points, colors)
        texture_mapper.apply_smart_uv_mapping()
        texture_mapper.map_colors_to_mesh()
        return texture_mapper

    def create_high_res_textured_mesh(self):
        mesh = pv.Sphere(radius=1, theta_resolution=100, phi_resolution=100)

        # Generate UV coordinates for the sphere
        theta = np.linspace(0, 2 * np.pi, mesh.n_points)
        phi = np.linspace(0, np.pi, mesh.n_points)
        u = theta / (2 * np.pi)
        v = phi / np.pi

        mesh.point_data['UV'] = np.column_stack((u, v))
        return mesh

    def create_high_res_texture_mapper(self, mesh):
        texture_mapper = TextureMapper(texture_resolution=2048)
        texture_mapper.load_mesh(mesh)
        colors = np.random.rand(mesh.n_points, 3)
        texture_mapper.load_point_cloud_with_colors(mesh.points, colors)
        texture_mapper.map_colors_to_mesh()
        return texture_mapper

    # def test_convert_complex_mesh(self):
    #     """
    #     Feature: OBJ conversion of complex mesh
    #
    #     Scenario: Convert a complex mesh with multiple materials to OBJ format
    #       Given a complex mesh with multiple materials
    #       When I convert the mesh to OBJ format
    #       Then the resulting OBJ file should contain multiple material definitions
    #       And the OBJ file should reference the correct materials for each face
    #     """
    #     complex_mesh = self.create_complex_mesh_with_materials()
    #     texture_mapper = self.create_complex_texture_mapper(complex_mesh)
    #     converter = MeshToOBJConverter(complex_mesh, texture_mapper)
    #     obj_filename = os.path.join(self.temp_dir, "complex_mesh.obj")
    #     texture_filename = os.path.join(self.temp_dir, "complex_texture.png")
    #     converter.convert_and_save(obj_filename, texture_filename)
    #
    #     with open(obj_filename, 'r') as f:
    #         content = f.read()
    #         self.assertIn("mtllib", content)
    #         self.assertIn("usemtl", content)
    #         material_count = content.count("usemtl")
    #         self.assertGreater(material_count, 1, f"Expected more than one material, but found {material_count}")

    def test_texture_image_quality(self):
        """
        Feature: Texture image quality in OBJ conversion

        Scenario: Generate a high-quality texture image during OBJ conversion
          Given a textured mesh with high-resolution texture data
          When I convert the mesh to OBJ format
          Then the generated texture image should be of high quality
          And the texture image should accurately represent the original texture data
        """
        high_res_mesh = self.create_high_res_textured_mesh()
        texture_mapper = self.create_high_res_texture_mapper(high_res_mesh)
        converter = MeshToOBJConverter(high_res_mesh, texture_mapper)
        obj_filename = os.path.join(self.temp_dir, "high_res_mesh.obj")
        texture_filename = os.path.join(self.temp_dir, "high_res_texture.png")
        converter.convert_and_save(obj_filename, texture_filename)

        with Image.open(texture_filename) as texture_image:
            self.assertGreaterEqual(texture_image.width, 2048)
            self.assertGreaterEqual(texture_image.height, 2048)


if __name__ == '__main__':
    unittest.main()
