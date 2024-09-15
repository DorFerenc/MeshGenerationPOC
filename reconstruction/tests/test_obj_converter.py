import unittest
import numpy as np
import pyvista as pv
from PIL import Image
from reconstruction.obj_converter import OBJConverter
from reconstruction.utils import OBJConversionError

class TestOBJConverter(unittest.TestCase):
    def setUp(self):
        self.obj_converter = OBJConverter()
        self.create_test_mesh()

    def create_test_mesh(self):
        self.mesh = pv.Cube()
        self.mesh.point_data['RGB'] = np.random.rand(self.mesh.n_points, 3)
        self.mesh.point_data['UV'] = np.random.rand(self.mesh.n_points, 2)

    def test_convert(self):
        result = self.obj_converter.convert(self.mesh)
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
            self.obj_converter.convert(mesh_without_uv)

    def test_convert_without_rgb(self):
        mesh_without_rgb = pv.Cube()
        mesh_without_rgb.point_data['UV'] = np.random.rand(mesh_without_rgb.n_points, 2)
        with self.assertRaises(OBJConversionError):
            self.obj_converter.convert(mesh_without_rgb)

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
