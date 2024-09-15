import unittest
import numpy as np
from unittest.mock import Mock, patch
from reconstruction.reconstruction_service import ReconstructionService
from reconstruction.utils import ReconstructionError, InputProcessingError, MeshGenerationError, TexturingError, OBJConversionError

class TestReconstructionService(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.reconstruction_service = ReconstructionService()
        self.test_input_data = np.random.rand(1000, 3)  # Random 3D point cloud

    @patch('reconstruction.point_cloud_processor.PointCloudProcessor.process')
    @patch('reconstruction.mesh_generator.MeshGenerator.generate_mesh')
    @patch('reconstruction.texture_mapper.TextureMapper.apply_texture')
    @patch('reconstruction.obj_converter.OBJConverter.convert')
    async def test_reconstruct_success(self, mock_convert, mock_apply_texture, mock_generate_mesh, mock_process):
        # Set up mock return values
        mock_process.return_value = self.test_input_data
        mock_generate_mesh.return_value = Mock()  # Mock mesh object
        mock_apply_texture.return_value = Mock()  # Mock textured mesh object
        mock_convert.return_value = {'obj_content': 'mock_obj', 'mtl_content': 'mock_mtl', 'texture_image': Mock()}

        # Run the reconstruction process
        result = await self.reconstruction_service.reconstruct(self.test_input_data)

        # Assertions
        self.assertIn('obj_content', result)
        self.assertIn('mtl_content', result)
        self.assertIn('texture_image', result)

        # Verify that all steps were called
        mock_process.assert_called_once()
        mock_generate_mesh.assert_called_once()
        mock_apply_texture.assert_called_once()
        mock_convert.assert_called_once()

    @patch('reconstruction.point_cloud_processor.PointCloudProcessor.process')
    async def test_reconstruct_input_processing_error(self, mock_process):
        mock_process.side_effect = InputProcessingError("Invalid input data")

        with self.assertRaises(ReconstructionError):
            await self.reconstruction_service.reconstruct(self.test_input_data)

    @patch('reconstruction.point_cloud_processor.PointCloudProcessor.process')
    @patch('reconstruction.mesh_generator.MeshGenerator.generate_mesh')
    async def test_reconstruct_mesh_generation_error(self, mock_generate_mesh, mock_process):
        mock_process.return_value = self.test_input_data
        mock_generate_mesh.side_effect = MeshGenerationError("Mesh generation failed")

        with self.assertRaises(ReconstructionError):
            await self.reconstruction_service.reconstruct(self.test_input_data)

    @patch('reconstruction.point_cloud_processor.PointCloudProcessor.process')
    @patch('reconstruction.mesh_generator.MeshGenerator.generate_mesh')
    @patch('reconstruction.texture_mapper.TextureMapper.apply_texture')
    async def test_reconstruct_texturing_error(self, mock_apply_texture, mock_generate_mesh, mock_process):
        mock_process.return_value = self.test_input_data
        mock_generate_mesh.return_value = Mock()
        mock_apply_texture.side_effect = TexturingError("Texturing failed")

        with self.assertRaises(ReconstructionError):
            await self.reconstruction_service.reconstruct(self.test_input_data)

    @patch('reconstruction.point_cloud_processor.PointCloudProcessor.process')
    @patch('reconstruction.mesh_generator.MeshGenerator.generate_mesh')
    @patch('reconstruction.texture_mapper.TextureMapper.apply_texture')
    @patch('reconstruction.obj_converter.OBJConverter.convert')
    async def test_reconstruct_obj_conversion_error(self, mock_convert, mock_apply_texture, mock_generate_mesh, mock_process):
        mock_process.return_value = self.test_input_data
        mock_generate_mesh.return_value = Mock()
        mock_apply_texture.return_value = Mock()
        mock_convert.side_effect = OBJConversionError("OBJ conversion failed")

        with self.assertRaises(ReconstructionError):
            await self.reconstruction_service.reconstruct(self.test_input_data)

    def test_get_progress(self):
        # Simulate some progress
        self.reconstruction_service.progress_reporter.update(50, "Halfway there")
        progress = self.reconstruction_service.get_progress()

        self.assertEqual(progress['progress'], 50)
        self.assertEqual(progress['status'], "Halfway there")

    @patch('reconstruction.point_cloud_processor.PointCloudProcessor.process')
    @patch('reconstruction.mesh_generator.MeshGenerator.generate_mesh')
    @patch('reconstruction.texture_mapper.TextureMapper.apply_texture')
    @patch('reconstruction.obj_converter.OBJConverter.convert')
    async def test_reconstruct_progress_reporting(self, mock_convert, mock_apply_texture, mock_generate_mesh, mock_process):
        mock_process.return_value = self.test_input_data
        mock_generate_mesh.return_value = Mock()
        mock_apply_texture.return_value = Mock()
        mock_convert.return_value = {'obj_content': 'mock_obj', 'mtl_content': 'mock_mtl', 'texture_image': Mock()}

        await self.reconstruction_service.reconstruct(self.test_input_data)

        # Check if progress was updated correctly
        final_progress = self.reconstruction_service.get_progress()
        self.assertEqual(final_progress['progress'], 100)
        self.assertEqual(final_progress['status'], "Conversion to OBJ complete")

    async def test_run_async(self):
        def sync_func():
            return "test result"

        result = await self.reconstruction_service.run_async(sync_func)
        self.assertEqual(result, "test result")


if __name__ == '__main__':
    unittest.main()
