import unittest
import numpy as np
import pyvista as pv
from PIL import Image
from reconstruction.obj_converter import OBJConverter
from reconstruction.utils import OBJConversionError
import tempfile
import shutil
import os

class TestOBJConverter(unittest.TestCase):
    def setUp(self):
        self.obj_converter = OBJConverter()
        self.create_test_mesh()
        self.output_folder = tempfile.mkdtemp()  # Create a temporary directory for saving files

    def tearDown(self):
        """Cleanup temporary directories after tests."""
        shutil.rmtree(self.output_folder)  # Remove the temporary directory

    def create_test_mesh(self):
        self.mesh = pv.Cube()
        self.mesh.point_data['RGB'] = np.random.rand(self.mesh.n_points, 3)
        self.mesh.point_data['UV'] = np.random.rand(self.mesh.n_points, 2)

    def test_convert_and_save_files(self):
        """Test if the OBJ, MTL, and PNG files are saved properly."""
        result = self.obj_converter.convert(self.mesh, self.output_folder)

        # Verify that files are saved in the output directory
        obj_file = os.path.join(self.output_folder, "model.obj")
        mtl_file = os.path.join(self.output_folder, "material.mtl")
        png_file = os.path.join(self.output_folder, "texture.png")

        self.assertTrue(os.path.exists(obj_file), "OBJ file was not saved")
        self.assertTrue(os.path.exists(mtl_file), "MTL file was not saved")
        self.assertTrue(os.path.exists(png_file), "PNG texture file was not saved")

        # Verify the content of the OBJ and MTL files
        with open(obj_file, 'r') as obj_f:
            obj_content = obj_f.read()
            self.assertIn('v ', obj_content, "OBJ content is missing vertex data")
            self.assertIn('vt ', obj_content, "OBJ content is missing texture coordinate data")

        with open(mtl_file, 'r') as mtl_f:
            mtl_content = mtl_f.read()
            self.assertIn('newmtl', mtl_content, "MTL content is incorrect")

        # Verify that the PNG file can be opened as an image
        image = Image.open(png_file)
        self.assertEqual(image.size, (1024, 1024), "Texture image has incorrect size")
        self.assertEqual(image.mode, 'RGB', "Texture image mode is incorrect")

    def test_convert_without_uv_or_rgb(self):
        """Test that OBJ conversion fails if UV or RGB data is missing."""
        mesh_without_uv = pv.Cube()
        mesh_without_uv.point_data['RGB'] = np.random.rand(mesh_without_uv.n_points, 3)
        with self.assertRaises(OBJConversionError):
            self.obj_converter.convert(mesh_without_uv, self.output_folder)

        mesh_without_rgb = pv.Cube()
        mesh_without_rgb.point_data['UV'] = np.random.rand(mesh_without_rgb.n_points, 2)
        with self.assertRaises(OBJConversionError):
            self.obj_converter.convert(mesh_without_rgb, self.output_folder)

    def test_convert(self):
        result = self.obj_converter.convert(self.mesh, self.output_folder)
        self.assertIn('obj_content', result)
        self.assertIn('mtl_content', result)
        self.assertIn('texture_image', result)
        self.assertIsInstance(result['texture_image'], Image.Image)

    def test_generate_obj_content(self):
        self.obj_converter.mesh = self.mesh
        obj_content = self.obj_converter.generate_obj_content()
        self.assertIn('v ', obj_content)
        self.assertIn('vt ', obj_content)
        self.assertIn('f ', obj_content)
        self.assertIn('mtllib', obj_content)
        self.assertIn('usemtl', obj_content)

    def test_generate_mtl_content(self):
        mtl_content = self.obj_converter.generate_mtl_content()
        self.assertIn('newmtl', mtl_content)
        self.assertIn('Ka', mtl_content)
        self.assertIn('Kd', mtl_content)
        self.assertIn('Ks', mtl_content)
        self.assertIn('map_Kd', mtl_content)

    def test_generate_texture_image(self):
        self.obj_converter.mesh = self.mesh
        texture_image = self.obj_converter.generate_texture_image()
        self.assertIsInstance(texture_image, Image.Image)
        self.assertEqual(texture_image.size, (1024, 1024))
        self.assertEqual(texture_image.mode, 'RGB')

    def test_convert_without_uv(self):
        mesh_without_uv = pv.Cube()
        mesh_without_uv.point_data['RGB'] = np.random.rand(mesh_without_uv.n_points, 3)
        with self.assertRaises(OBJConversionError):
            self.obj_converter.convert(mesh_without_uv, self.output_folder)

    def test_convert_without_rgb(self):
        mesh_without_rgb = pv.Cube()
        mesh_without_rgb.point_data['UV'] = np.random.rand(mesh_without_rgb.n_points, 2)
        with self.assertRaises(OBJConversionError):
            self.obj_converter.convert(mesh_without_rgb, self.output_folder)

    def test_obj_content_validity(self):
        self.obj_converter.mesh = self.mesh
        obj_content = self.obj_converter.generate_obj_content()
        lines = obj_content.split('\n')
        v_count = sum(1 for line in lines if line.startswith('v '))
        vt_count = sum(1 for line in lines if line.startswith('vt '))
        f_count = sum(1 for line in lines if line.startswith('f '))
        self.assertEqual(v_count, self.mesh.n_points)
        self.assertEqual(vt_count, self.mesh.n_points)
        self.assertGreater(f_count, 0)

    def test_texture_image_content(self):
        self.obj_converter.mesh = self.mesh
        texture_image = self.obj_converter.generate_texture_image()
        image_array = np.array(texture_image)
        self.assertGreater(np.std(image_array), 0)  # Ensure the image isn't blank


if __name__ == '__main__':
    unittest.main()
