import unittest
import numpy as np
import pyvista as pv
import os
import tempfile
import shutil
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


if __name__ == '__main__':
    unittest.main()
