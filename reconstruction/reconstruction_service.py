import asyncio
from .point_cloud_processor import PointCloudProcessor
from .mesh_generator import MeshGenerator
from .texture_mapper import TextureMapper
from .obj_converter import OBJConverter
from .utils.error_handling import ReconstructionError
from .utils.progress_reporter import ProgressReporter


class ReconstructionService:
    """
    Orchestrates the entire 3D reconstruction process.

    This class integrates all components of the reconstruction pipeline,
    managing the flow from point cloud processing to OBJ conversion.
    It handles error management and progress reporting throughout the process.
    """

    def __init__(self):
        """Initialize the ReconstructionService with its component classes."""
        self.point_cloud_processor = PointCloudProcessor()
        self.mesh_generator = MeshGenerator()
        self.texture_mapper = TextureMapper()
        self.obj_converter = OBJConverter()
        self.progress_reporter = ProgressReporter()

    async def reconstruct(self, input_data):
        """
        Perform the complete reconstruction process asynchronously.

        Args:
            input_data: The input data for reconstruction (e.g., point cloud data or file path).

        Returns:
            dict: The final OBJ data including the obj content, mtl content, and texture image.

        Raises:
            ReconstructionError: If any step of the reconstruction process fails.
        """
        try:
            self.progress_reporter.update(0, "Starting reconstruction")

            point_cloud = await self.run_async(self.point_cloud_processor.process, input_data)
            self.progress_reporter.update(20, "Point cloud processed")

            mesh = await self.run_async(self.mesh_generator.generate_mesh, point_cloud)
            self.progress_reporter.update(50, "Mesh generated")

            textured_mesh = await self.run_async(self.texture_mapper.apply_texture, mesh, point_cloud)
            self.progress_reporter.update(80, "Texture applied")

            obj_data = await self.run_async(self.obj_converter.convert, textured_mesh)
            self.progress_reporter.update(100, "Conversion to OBJ complete")

            return obj_data
        except Exception as e:
            raise ReconstructionError(f"Reconstruction failed: {str(e)}")

    async def run_async(self, func, *args):
        """
        Run a function asynchronously.

        Args:
            func: The function to run asynchronously.
            *args: Arguments to pass to the function.

        Returns:
            The result of the function.
        """
        return await asyncio.to_thread(func, *args)

    def get_progress(self):
        """
        Get the current progress of the reconstruction process.

        Returns:
            dict: A dictionary containing the progress percentage and status message.
        """
        return self.progress_reporter.get_progress()
