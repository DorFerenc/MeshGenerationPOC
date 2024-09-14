import numpy as np
import pyvista as pv
import logging
import os

class PointCloudToMesh:
    """
    A class to convert point cloud data to a 3D mesh.

    This class provides functionality to load point cloud data,
    generate a 3D mesh using Delaunay triangulation, and apply various smoothing
    and refinement techniques to improve the mesh quality.

    Attributes:
        point_cloud (np.ndarray): The loaded point cloud data.
        mesh (pv.PolyData): The generated 3D mesh.
        logger (logging.Logger): Logger for the class.
    """

    def __init__(self):
        """Initialize the PointCloudToMesh object."""
        self.point_cloud = None
        self.mesh = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def set_point_cloud(self, points):
        """
        Set the point cloud data.

        Args:
            points (np.ndarray): Array of 3D point coordinates.
        """
        self.point_cloud = points
        self.logger.info(f"Point cloud set with {len(points)} points")

    def generate_mesh(self, alpha=0.002):
        """
        Generate a 3D mesh from the loaded point cloud data using Delaunay triangulation.

        Args:
            alpha (float): The alpha value for the Delaunay triangulation algorithm.
                           Lower values create a denser mesh. Default is 0.002.

        Raises:
            ValueError: If no point cloud data has been loaded.
        """
        if self.point_cloud is None:
            self.logger.error("No point cloud data loaded")
            raise ValueError("No point cloud data loaded. Use set_point_cloud() first.")

        self.logger.info(f"Generating mesh with alpha={alpha}")
        try:
            poly_data = pv.PolyData(self.point_cloud)
            self.mesh = poly_data.delaunay_3d(alpha=alpha)
            self.mesh = self.mesh.extract_surface()

            # Remove degenerate triangles
            self.mesh.clean(tolerance=1e-6)

            n_cells = self.mesh.n_cells
            self.logger.info(f"Mesh generated with {self.mesh.n_points} points and {n_cells} cells")

            # Compute and log mesh quality
            quality = self.mesh.compute_cell_quality()
            quality_array = quality['CellQuality']
            if len(quality_array) > 0:
                min_quality = np.min(quality_array)
                max_quality = np.max(quality_array)
                avg_quality = np.mean(quality_array)
                self.logger.info(
                    f"Mesh quality - Min: {min_quality:.4f}, Max: {max_quality:.4f}, Avg: {avg_quality:.4f}")
            else:
                self.logger.warning("Unable to compute mesh quality. No cells in the mesh.")

        except Exception as e:
            self.logger.error(f"Error generating mesh: {str(e)}")
            raise

    # def generate_mesh(self, alpha: float = 0.002) -> None:  # BEST
    #     """
    #     Generate a 3D mesh from the loaded point cloud data using Delaunay triangulation.
    #
    #     Args:
    #         alpha (float): The alpha value for the Delaunay triangulation algorithm.
    #                        Lower values create a denser mesh. Default is 0.002.
    #
    #     Raises:
    #         ValueError: If no point cloud data has been loaded.
    #     """
    #     if self.point_cloud is None:
    #         self.logger.error("No point cloud data loaded")
    #         raise ValueError("No point cloud data loaded. Use load_point_cloud_from_csv() first.")
    #
    #     self.logger.info(f"Generating mesh with alpha={alpha}")
    #     try:
    #         poly_data = pv.PolyData(self.point_cloud)
    #         self.mesh = poly_data.delaunay_3d(alpha=alpha)
    #         self.mesh = self.mesh.extract_surface()
    #
    #         # Remove degenerate triangles
    #         self.mesh.clean(tolerance=1e-6)
    #
    #         n_cells = self.mesh.n_cells
    #         self.logger.info(f"Mesh generated with {self.mesh.n_points} points and {n_cells} cells")
    #
    #         # Compute and log mesh quality
    #         quality = self.mesh.compute_cell_quality()
    #         quality_array = quality['CellQuality']
    #         if len(quality_array) > 0:
    #             min_quality = np.min(quality_array)
    #             max_quality = np.max(quality_array)
    #             avg_quality = np.mean(quality_array)
    #             self.logger.info(
    #                 f"Mesh quality - Min: {min_quality:.4f}, Max: {max_quality:.4f}, Avg: {avg_quality:.4f}")
    #         else:
    #             self.logger.warning("Unable to compute mesh quality. No cells in the mesh.")
    #
    #     except Exception as e:
    #         self.logger.error(f"Error generating mesh: {str(e)}")
    #         raise

    def visualize_mesh(self):
        """
        Visualize the generated mesh.

        Raises:
            ValueError: If no mesh has been generated.
        """
        if self.mesh is None:
            raise ValueError("No mesh generated. Use generate_mesh() first.")

        p = pv.Plotter()
        p.add_mesh(self.mesh, color='orange')
        p.show()

    def save_mesh(self, filename):
        """
        Save the generated mesh to a file.

        Args:
            filename (str): The name of the file to save the mesh to.
                            Supported formats: .ply, .vtp, .stl, .vtk

        Raises:
            ValueError: If no mesh has been generated or if the file extension is not supported.
        """
        if self.mesh is None:
            self.logger.error("No mesh generated to save")
            raise ValueError("No mesh generated. Use generate_mesh() first.")

        supported_extensions = ['.ply', '.vtp', '.stl', '.vtk']
        file_extension = os.path.splitext(filename)[1].lower()

        if file_extension not in supported_extensions:
            self.logger.error(f"Unsupported file extension. Supported formats: {', '.join(supported_extensions)}")
            raise ValueError(f"Unsupported file extension. Supported formats: {', '.join(supported_extensions)}")

        try:
            self.mesh.save(filename)
            self.logger.info(f"Mesh saved successfully to {filename}")
        except Exception as e:
            self.logger.error(f"Error saving mesh: {str(e)}")
            raise

# import numpy as np
# import pyvista as pv
# import multiprocessing
# from typing import Optional
# import logging
# import os
#
#
# class PointCloudToMesh:
#     """
#     A class to convert point cloud data to a 3D mesh.
#
#     This class provides functionality to load point cloud data from a CSV file,
#     generate a 3D mesh using Delaunay triangulation, and apply various smoothing
#     and refinement techniques to improve the mesh quality.
#
#     Attributes:
#         point_cloud (np.ndarray): The loaded point cloud data.
#         mesh (pv.PolyData): The generated 3D mesh.
#         logger (logging.Logger): Logger for the class.
#     """
#
#     def __init__(self):
#         """Initialize the PointCloudToMesh object."""
#         self.point_cloud: Optional[np.ndarray] = None
#         self.mesh: Optional[pv.PolyData] = None
#         self.logger = logging.getLogger(self.__class__.__name__)
#         self.logger.setLevel(logging.INFO)
#         handler = logging.StreamHandler()
#         formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#         handler.setFormatter(formatter)
#         self.logger.addHandler(handler)
#
#     def load_point_cloud_from_csv(self, csv_file: str) -> None:
#         """
#         Load point cloud data from a CSV file.
#
#         Args:
#             csv_file (str): Path to the CSV file containing point cloud data.
#
#         Raises:
#             FileNotFoundError: If the specified CSV file does not exist.
#             ValueError: If the CSV file is empty or contains invalid data.
#         """
#         self.logger.info(f"Loading point cloud from {csv_file}")
#         try:
#             self.point_cloud = np.loadtxt(csv_file, delimiter=',')
#             if self.point_cloud.size == 0:
#                 raise ValueError("The CSV file is empty.")
#             self.logger.info(f"Successfully loaded {self.point_cloud.shape[0]} points")
#         except FileNotFoundError:
#             self.logger.error(f"The file {csv_file} was not found")
#             raise
#         except ValueError as e:
#             self.logger.error(f"Error loading CSV file: {str(e)}")
#             raise
#
#     # def generate_mesh(self, alpha: float = 0.002) -> None:
#     #     """
#     #     Generate a 3D mesh from the loaded point cloud data using Delaunay triangulation.
#     #
#     #     Args:
#     #         alpha (float): The alpha value for the Delaunay triangulation algorithm.
#     #                        Lower values create a denser mesh. Default is 0.002.
#     #
#     #     Raises:
#     #         ValueError: If no point cloud data has been loaded.
#     #     """
#     #     if self.point_cloud is None:
#     #         self.logger.error("No point cloud data loaded")
#     #         raise ValueError("No point cloud data loaded. Use load_point_cloud_from_csv() first.")
#     #
#     #     self.logger.info(f"Generating mesh with alpha={alpha}")
#     #     try:
#     #         poly_data = pv.PolyData(self.point_cloud)
#     #         self.mesh = poly_data.delaunay_3d(alpha=alpha)
#     #         self.mesh = self.mesh.extract_surface()
#     #         self.logger.info(f"Mesh generated with {self.mesh.n_points} points and {self.mesh.n_faces} faces")
#     #     except Exception as e:
#     #         self.logger.error(f"Error generating mesh: {str(e)}")
#     #         raise
#
#     def generate_mesh(self, alpha: float = 0.002) -> None:  # BEST
#         """
#         Generate a 3D mesh from the loaded point cloud data using Delaunay triangulation.
#
#         Args:
#             alpha (float): The alpha value for the Delaunay triangulation algorithm.
#                            Lower values create a denser mesh. Default is 0.002.
#
#         Raises:
#             ValueError: If no point cloud data has been loaded.
#         """
#         if self.point_cloud is None:
#             self.logger.error("No point cloud data loaded")
#             raise ValueError("No point cloud data loaded. Use load_point_cloud_from_csv() first.")
#
#         self.logger.info(f"Generating mesh with alpha={alpha}")
#         try:
#             poly_data = pv.PolyData(self.point_cloud)
#             self.mesh = poly_data.delaunay_3d(alpha=alpha)
#             self.mesh = self.mesh.extract_surface()
#
#             # Remove degenerate triangles
#             self.mesh.clean(tolerance=1e-6)
#
#             n_cells = self.mesh.n_cells
#             self.logger.info(f"Mesh generated with {self.mesh.n_points} points and {n_cells} cells")
#
#             # Compute and log mesh quality
#             quality = self.mesh.compute_cell_quality()
#             quality_array = quality['CellQuality']
#             if len(quality_array) > 0:
#                 min_quality = np.min(quality_array)
#                 max_quality = np.max(quality_array)
#                 avg_quality = np.mean(quality_array)
#                 self.logger.info(
#                     f"Mesh quality - Min: {min_quality:.4f}, Max: {max_quality:.4f}, Avg: {avg_quality:.4f}")
#             else:
#                 self.logger.warning("Unable to compute mesh quality. No cells in the mesh.")
#
#         except Exception as e:
#             self.logger.error(f"Error generating mesh: {str(e)}")
#             raise
#
#     def apply_laplacian_smoothing(self, iterations: int = 20) -> None:
#         """
#         Apply Laplacian smoothing to the mesh to remove local artifacts.
#
#         Args:
#             iterations (int): Number of smoothing iterations. Default is 20.
#
#         Raises:
#             ValueError: If no mesh has been generated.
#         """
#         if self.mesh is None:
#             self.logger.error("No mesh generated")
#             raise ValueError("No mesh generated. Use generate_mesh() first.")
#
#         self.logger.info(f"Applying Laplacian smoothing with {iterations} iterations")
#         try:
#             self.mesh = self.mesh.smooth(n_iter=iterations)
#             self.logger.info("Laplacian smoothing completed")
#         except Exception as e:
#             self.logger.error(f"Error during Laplacian smoothing: {str(e)}")
#             raise
#
#     def apply_bilateral_smoothing(self, iterations: int = 10, edge_preserving_value: float = 0.1,
#                                   feature_angle: float = 45.0) -> None:
#         """
#         Apply bilateral smoothing to the mesh to enhance visual quality while preserving sharp features.
#
#         Args:
#             iterations (int): Number of smoothing iterations. Default is 10.
#             edge_preserving_value (float): Smoothing parameter. Lower values preserve more edges. Default is 0.1.
#             feature_angle (float): Angle to preserve sharp features. Default is 45.0 degrees.
#
#         Raises:
#             ValueError: If no mesh has been generated.
#         """
#         if self.mesh is None:
#             self.logger.error("No mesh generated")
#             raise ValueError("No mesh generated. Use generate_mesh() first.")
#
#         self.logger.info(f"Applying bilateral smoothing with {iterations} iterations")
#         try:
#             self.mesh = self.mesh.smooth_taubin(n_iter=iterations, pass_band=edge_preserving_value,
#                                                 feature_angle=feature_angle)
#             self.logger.info("Bilateral smoothing completed")
#         except Exception as e:
#             self.logger.error(f"Error during bilateral smoothing: {str(e)}")
#             raise
#
#     def refine_mesh(self) -> None:
#         """
#         Refine the mesh by applying Laplacian and bilateral smoothing.
#
#         This method applies both Laplacian and bilateral smoothing to improve the overall mesh quality.
#
#         Raises:
#             ValueError: If no mesh has been generated.
#         """
#         if self.mesh is None:
#             self.logger.error("No mesh generated")
#             raise ValueError("No mesh generated. Use generate_mesh() first.")
#
#         self.logger.info("Starting mesh refinement")
#         try:
#             self.apply_laplacian_smoothing()
#             self.apply_bilateral_smoothing()
#             self.logger.info("Mesh refinement completed")
#         except Exception as e:
#             self.logger.error(f"Error during mesh refinement: {str(e)}")
#             raise
#
#     def visualize_point_cloud(self) -> None:
#         """
#         Visualize the loaded point cloud.
#
#         Raises:
#             ValueError: If no point cloud data has been loaded.
#         """
#         if self.point_cloud is None:
#             raise ValueError("No point cloud data loaded. Use load_point_cloud_from_csv() first.")
#
#         point_cloud = pv.PolyData(self.point_cloud)
#         point_cloud.plot(eye_dome_lighting=True)
#
#     def visualize_mesh(self) -> None:
#         """
#         Visualize the generated mesh.
#
#         Raises:
#             ValueError: If no mesh has been generated.
#         """
#         if self.mesh is None:
#             raise ValueError("No mesh generated. Use generate_mesh() first.")
#
#         self.mesh.plot(color='orange')
#
#     def visualize_mesh_with_point_cloud(self) -> None:
#         """
#         Visualize the loaded point cloud and the generated mesh together.
#
#         Raises:
#             ValueError: If no point cloud data has been loaded.
#         """
#         if self.point_cloud is None:
#             raise ValueError("No point cloud data loaded. Use load_point_cloud_from_csv() first.")
#
#         plotter = pv.Plotter()
#
#         point_cloud = pv.PolyData(self.point_cloud)
#         plotter.add_mesh(point_cloud, color='white', point_size=5, render_points_as_spheres=True, name='Point Cloud')
#
#         if self.mesh is not None:
#             plotter.add_mesh(self.mesh, color='orange', opacity=0.5, name='Mesh')
#
#         plotter.show()
#
#     def save_mesh(self, filename: str) -> None:
#         """
#         Save the generated mesh to a file.
#
#         Args:
#             filename (str): The name of the file to save the mesh to.
#                             Supported formats: .ply, .vtp, .stl, .vtk
#
#         Raises:
#             ValueError: If no mesh has been generated or if the file extension is not supported.
#         """
#         if self.mesh is None:
#             self.logger.error("No mesh generated to save")
#             raise ValueError("No mesh generated. Use generate_mesh() first.")
#
#         supported_extensions = ['.ply', '.vtp', '.stl', '.vtk']
#         file_extension = os.path.splitext(filename)[1].lower()
#
#         if file_extension not in supported_extensions:
#             self.logger.error(f"Unsupported file extension. Supported formats: {', '.join(supported_extensions)}")
#             raise ValueError(f"Unsupported file extension. Supported formats: {', '.join(supported_extensions)}")
#
#         try:
#             self.mesh.save(filename)
#             self.logger.info(f"Mesh saved successfully to {filename}")
#         except Exception as e:
#             self.logger.error(f"Error saving mesh: {str(e)}")
#             raise
#
#     #########################################################################
#     # currently, not in use as default value of 0.0002 works best
#     #########################################################################
#     def find_optimal_alpha(self, start_alpha: float = 0.001, max_alpha: float = 0.1, max_iterations: int = 10) -> float:
#         """
#         Find an optimal alpha value for mesh generation.
#
#         This method iteratively tries different alpha values to minimize degenerate triangles
#         while maintaining mesh detail.
#
#         Args:
#             start_alpha (float): The initial alpha value to try. Default is 0.001.
#             max_alpha (float): The maximum alpha value to try. Default is 0.1.
#             max_iterations (int): Maximum number of iterations. Default is 10.
#
#         Returns:
#             float: The optimal alpha value found.
#
#         Raises:
#             ValueError: If no point cloud data has been loaded.
#         """
#         if self.point_cloud is None:
#             self.logger.error("No point cloud data loaded")
#             raise ValueError("No point cloud data loaded. Use load_point_cloud_from_csv() first.")
#
#         self.logger.info("Finding optimal alpha value")
#
#         best_alpha = start_alpha
#         min_degenerate = float('inf')
#
#         for _ in range(max_iterations):
#             try:
#                 poly_data = pv.PolyData(self.point_cloud)
#                 mesh = poly_data.delaunay_3d(alpha=best_alpha)
#                 mesh = mesh.extract_surface()
#
#                 # Count degenerate triangles
#                 quality = mesh.compute_cell_quality()
#                 degenerate_count = np.sum(quality['CellQuality'] < 1e-6)
#
#                 if degenerate_count < min_degenerate:
#                     min_degenerate = degenerate_count
#                     if min_degenerate == 0:
#                         break
#                 else:
#                     # If we're not improving, increase alpha more aggressively
#                     best_alpha *= 2
#
#                 best_alpha *= 1.5
#                 if best_alpha > max_alpha:
#                     break
#
#             except Exception as e:
#                 self.logger.warning(f"Error during alpha optimization: {str(e)}")
#                 best_alpha *= 1.5
#
#         self.logger.info(f"Optimal alpha found: {best_alpha:.6f} with {min_degenerate} degenerate triangles")
#         return best_alpha
#
#
# def visualize_in_process(func):
#     """
#     Run a visualization function in a separate process.
#
#     This decorator allows for non-blocking visualization by running the visualization
#     function in a separate process.
#
#     Args:
#         func (callable): The visualization function to run in a separate process.
#
#     Returns:
#         multiprocessing.Process: The process running the visualization function.
#     """
#     proc = multiprocessing.Process(target=func)
#     proc.start()
#     return proc
#
#
# # Example usage
# if __name__ == "__main__":
#     pc_to_mesh = PointCloudToMesh()
#     pc_to_mesh.load_point_cloud_from_csv("point_cloud.csv")
#     ############# new to get best alpha currently not in use
#     # optimal_alpha = pc_to_mesh.find_optimal_alpha()
#     # pc_to_mesh.generate_mesh(alpha=optimal_alpha)
#     ############# up to here
#     proc_point_cloud = visualize_in_process(pc_to_mesh.visualize_point_cloud)
#
#     pc_to_mesh.generate_mesh()
#     proc_mesh = visualize_in_process(pc_to_mesh.visualize_mesh)
#
#     pc_to_mesh.refine_mesh()
#     proc_refined_mesh = visualize_in_process(pc_to_mesh.visualize_mesh)
#
#     proc_point_cloud.join()
#     proc_mesh.join()
#     proc_refined_mesh.join()
