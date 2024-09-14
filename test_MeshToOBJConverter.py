import unittest
import numpy as np
import pyvista as pv
import os
import tempfile
import shutil
from MeshToOBJConverter import MeshToOBJConverter
from TextureMapper import TextureMapper
from PIL import Image


class TestMeshToOBJConverter(unittest.TestCase):
    """
    A comprehensive test suite for the MeshToOBJConverter class.

    This class contains unit tests to verify the functionality of the MeshToOBJConverter class,
    including OBJ file generation, texture image creation, MTL file creation, and various edge cases.
    The tests ensure that the converter correctly handles different mesh types, texture mappings,
    and file operations.
    """

    def setUp(self):
        """
        Set up the test environment before each test method.

        This method creates a temporary directory, generates sample meshes and textures,
        and initializes a MeshToOBJConverter object for testing. It also sets up mock
        objects and sample data to be used across multiple tests.
        """
        self.temp_dir = tempfile.mkdtemp()
        self.cube_mesh = self.create_sample_cube_mesh()
        self.sphere_mesh = self.create_sample_sphere_mesh()
        self.texture_mapper = self.create_sample_texture_mapper(self.cube_mesh)
        self.converter = MeshToOBJConverter(self.cube_mesh, self.texture_mapper)

        # Debug information
        print("\nSetup Debug Info:")
        print(f"Cube Mesh - Points: {self.cube_mesh.n_points}, Faces: {self.cube_mesh.n_cells}")
        print(f"Cube Mesh Data: {self.cube_mesh.point_data.keys()}")
        print(f"Sphere Mesh - Points: {self.sphere_mesh.n_points}, Faces: {self.sphere_mesh.n_cells}")
        print(f"Sphere Mesh Data: {self.sphere_mesh.point_data.keys()}")

    def tearDown(self):
        """
        Clean up the test environment after each test method.

        This method removes temporary files and directories created during testing,
        ensuring a clean slate for the next test.
        """
        shutil.rmtree(self.temp_dir)

    def create_sample_cube_mesh(self):
        """
        Create a sample cube mesh for testing.

        Returns:
            pyvista.PolyData: A simple cube mesh with texture coordinates and RGB data.
        """
        mesh = pv.Cube()
        mesh.texture_map_to_plane(inplace=True)
        mesh.point_data["RGB"] = np.random.rand(mesh.n_points, 3)
        return mesh

    def create_sample_sphere_mesh(self):
        """
        Create a sample sphere mesh for testing.

        Returns:
            pyvista.PolyData: A simple sphere mesh with texture coordinates and RGB data.
        """
        mesh = pv.Sphere()
        mesh.texture_map_to_sphere(inplace=True)
        mesh.point_data["RGB"] = np.random.rand(mesh.n_points, 3)
        return mesh

    def create_sample_texture_mapper(self, mesh):
        """
        Create a sample TextureMapper for testing.

        Args:
            mesh (pyvista.PolyData): The mesh to associate with the TextureMapper.

        Returns:
            TextureMapper: A TextureMapper instance with sample data.
        """
        texture_mapper = TextureMapper()
        texture_mapper.mesh = mesh
        texture_mapper.point_cloud = mesh.points
        texture_mapper.colors = mesh.point_data["RGB"]
        return texture_mapper

    def test_mesh_setup(self):
        """
        Test to ensure that the mesh is set up correctly with UV coordinates and RGB data.
        """
        print("\nMesh Setup Debug Info:")
        print(f"Cube Mesh Data: {self.cube_mesh.point_data.keys()}")
        print(f"Texture Mapper Mesh Data: {self.texture_mapper.mesh.point_data.keys()}")

        self.assertIn('Texture Coordinates', self.cube_mesh.point_data.keys(), "UV coordinates not found in mesh")
        self.assertIn('RGB', self.cube_mesh.point_data.keys(), "RGB data not found in mesh")
        self.assertEqual(self.cube_mesh.n_points, len(self.texture_mapper.colors),
                         "Mismatch in number of points and colors")

    def test_convert_to_obj(self):
        """
        Test the convert_to_obj method.

        This test verifies that the method correctly converts a mesh to OBJ format,
        including proper formatting of vertices, texture coordinates, and faces.
        It also checks that the MTL file is referenced correctly.
        """
        obj_filename = os.path.join(self.temp_dir, "test.obj")
        try:
            self.converter.convert_to_obj(obj_filename)
        except ValueError as e:
            self.fail(f"convert_to_obj raised ValueError: {str(e)}")

        self.assertTrue(os.path.exists(obj_filename), "OBJ file was not created")

        with open(obj_filename, 'r') as f:
            content = f.read()

        self.assertIn("mtllib", content, "MTL file is not referenced in the OBJ file")
        self.assertIn("v ", content, "Vertices are missing in the OBJ file")
        self.assertIn("vt ", content, "Texture coordinates are missing in the OBJ file")
        self.assertIn("f ", content, "Faces are missing in the OBJ file")

    def test_save_texture_image(self):
        """
        Test the save_texture_image method.

        This test ensures that the texture image is correctly generated and saved,
        verifying the dimensions and color mode of the resulting image.
        """
        texture_filename = os.path.join(self.temp_dir, "texture.png")
        try:
            self.converter.save_texture_image(texture_filename)
        except ValueError as e:
            self.fail(f"save_texture_image raised ValueError: {str(e)}")

        self.assertTrue(os.path.exists(texture_filename), "Texture image was not created")

        with Image.open(texture_filename) as img:
            self.assertEqual(img.mode, "RGB", "Texture image is not in RGB mode")
            self.assertEqual(img.size, (self.texture_mapper.texture_resolution,
                                        self.texture_mapper.texture_resolution),
                             "Texture image has incorrect dimensions")

    def test_create_mtl_file(self):
        """
        Test the create_mtl_file method.

        This test verifies that the MTL file is correctly created with the proper
        material properties and texture map reference.
        """
        mtl_filename = os.path.join(self.temp_dir, "test.mtl")
        texture_filename = "texture.png"
        self.converter.create_mtl_file(mtl_filename, texture_filename)

        self.assertTrue(os.path.exists(mtl_filename), "MTL file was not created")

        with open(mtl_filename, 'r') as f:
            content = f.read()

        self.assertIn("newmtl", content, "Material definition is missing in the MTL file")
        self.assertIn(f"map_Kd {os.path.basename(texture_filename)}", content,
                      "Texture map is not correctly referenced in the MTL file")

    def test_convert_and_save(self):
        """
        Test the convert_and_save method.

        This test ensures that the entire conversion process works correctly,
        including OBJ file creation, texture image saving, and MTL file generation.
        It verifies that all necessary files are created and properly linked.
        """
        obj_filename = os.path.join(self.temp_dir, "complete_test.obj")
        texture_filename = os.path.join(self.temp_dir, "complete_test_texture.png")

        self.converter.convert_and_save(obj_filename, texture_filename)

        self.assertTrue(os.path.exists(obj_filename), "OBJ file was not created")
        self.assertTrue(os.path.exists(texture_filename), "Texture image was not created")
        self.assertTrue(os.path.exists(obj_filename[:-4] + ".mtl"), "MTL file was not created")

        with open(obj_filename, 'r') as f:
            obj_content = f.read()
        self.assertIn(f"mtllib {os.path.basename(obj_filename)[:-4]}.mtl", obj_content,
                      "MTL file is not correctly referenced in the OBJ file")

    def test_convert_mesh_without_uv(self):
        """
        Test converting a mesh without UV coordinates.

        This test verifies that the converter raises a ValueError when attempting
        to convert a mesh that does not have texture coordinates.
        """
        mesh_without_uv = pv.Cube()  # Create a cube without UV mapping
        converter = MeshToOBJConverter(mesh_without_uv, self.texture_mapper)

        with self.assertRaises(ValueError):
            converter.convert_to_obj(os.path.join(self.temp_dir, "no_uv.obj"))

    def test_convert_complex_mesh(self):
        """
        Test converting a more complex mesh.

        This test ensures that the converter can handle a more complex mesh (sphere)
        with a higher number of vertices and faces, verifying that all elements are
        correctly written to the OBJ file.
        """
        complex_converter = MeshToOBJConverter(self.sphere_mesh, self.create_sample_texture_mapper(self.sphere_mesh))
        obj_filename = os.path.join(self.temp_dir, "complex.obj")
        complex_converter.convert_to_obj(obj_filename)

        with open(obj_filename, 'r') as f:
            content = f.readlines()

        vertex_count = sum(1 for line in content if line.startswith('v '))
        face_count = sum(1 for line in content if line.startswith('f '))

        self.assertGreater(vertex_count, 100, "Complex mesh has too few vertices")
        self.assertGreater(face_count, 100, "Complex mesh has too few faces")

    def test_texture_image_content(self):
        """
        Test the content of the generated texture image.

        This test verifies that the generated texture image contains non-uniform color data,
        ensuring that the texture mapping process is creating meaningful textures.
        """
        texture_filename = os.path.join(self.temp_dir, "texture_content.png")
        self.converter.save_texture_image(texture_filename)

        with Image.open(texture_filename) as img:
            pixel_data = np.array(img)

        self.assertFalse(np.all(pixel_data == pixel_data[0, 0]),
                         "Texture image appears to be uniform, expected varied color data")

    def test_obj_file_formatting(self):
        """
        Test the formatting of the generated OBJ file.

        This test checks the detailed formatting of the OBJ file, ensuring that
        vertices, texture coordinates, and faces are correctly formatted and
        that face indices are 1-based as per OBJ specification.
        """
        obj_filename = os.path.join(self.temp_dir, "formatting_test.obj")
        self.converter.convert_to_obj(obj_filename)

        with open(obj_filename, 'r') as f:
            content = f.readlines()

        vertex_line = next(line for line in content if line.startswith('v '))
        texcoord_line = next(line for line in content if line.startswith('vt '))
        face_line = next(line for line in content if line.startswith('f '))

        self.assertRegex(vertex_line, r'^v -?\d+\.\d+ -?\d+\.\d+ -?\d+\.\d+$',
                         "Vertex line is not correctly formatted")
        self.assertRegex(texcoord_line, r'^vt \d+\.\d+ \d+\.\d+$',
                         "Texture coordinate line is not correctly formatted")
        self.assertRegex(face_line, r'^f \d+/\d+ \d+/\d+ \d+/\d+$',
                         "Face line is not correctly formatted")

        # Check that face indices are 1-based
        face_indices = [int(idx.split('/')[0]) for idx in face_line.split()[1:]]
        self.assertTrue(all(idx > 0 for idx in face_indices),
                        "Face indices are not 1-based as required by OBJ specification")


if __name__ == '__main__':
    unittest.main()