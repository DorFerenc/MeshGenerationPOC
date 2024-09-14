import numpy as np
import pyvista as pv
import multiprocessing
from typing import Optional


class PointCloudToMesh:
    """
    A class to convert point cloud data to a 3D mesh.

    This class provides functionality to load point cloud data from a CSV file,
    generate a 3D mesh using Delaunay triangulation, and apply various smoothing
    and refinement techniques to improve the mesh quality.

    Attributes:
        point_cloud (np.ndarray): The loaded point cloud data.
        mesh (pv.PolyData): The generated 3D mesh.
    """

    def __init__(self):
        """Initialize the PointCloudToMesh object."""
        self.point_cloud: Optional[np.ndarray] = None
        self.mesh: Optional[pv.PolyData] = None

    def load_point_cloud_from_csv(self, csv_file: str) -> None:
        """
        Load point cloud data from a CSV file.

        Args:
            csv_file (str): Path to the CSV file containing point cloud data.

        Raises:
            FileNotFoundError: If the specified CSV file does not exist.
            ValueError: If the CSV file is empty or contains invalid data.
        """
        try:
            self.point_cloud = np.loadtxt(csv_file, delimiter=',')
            if self.point_cloud.size == 0:
                raise ValueError("The CSV file is empty.")
        except FileNotFoundError:
            raise FileNotFoundError(f"The file {csv_file} was not found.")
        except ValueError as e:
            raise ValueError(f"Error loading CSV file: {str(e)}")

    def generate_mesh(self, alpha: float = 0.002) -> None:
        """
        Generate a 3D mesh from the loaded point cloud data using Delaunay triangulation.

        Args:
            alpha (float): The alpha value for the Delaunay triangulation algorithm.
                           Lower values create a denser mesh. Default is 0.002.

        Raises:
            ValueError: If no point cloud data has been loaded.
        """
        if self.point_cloud is None:
            raise ValueError("No point cloud data loaded. Use load_point_cloud_from_csv() first.")

        poly_data = pv.PolyData(self.point_cloud)
        self.mesh = poly_data.delaunay_3d(alpha=alpha)
        self.mesh = self.mesh.extract_surface()

    def apply_laplacian_smoothing(self, iterations: int = 20) -> None:
        """
        Apply Laplacian smoothing to the mesh to remove local artifacts.

        Args:
            iterations (int): Number of smoothing iterations. Default is 20.

        Raises:
            ValueError: If no mesh has been generated.
        """
        if self.mesh is None:
            raise ValueError("No mesh generated. Use generate_mesh() first.")

        self.mesh = self.mesh.smooth(n_iter=iterations)

    def apply_bilateral_smoothing(self, iterations: int = 10, edge_preserving_value: float = 0.1,
                                  feature_angle: float = 45.0) -> None:
        """
        Apply bilateral smoothing to the mesh to enhance visual quality while preserving sharp features.

        Args:
            iterations (int): Number of smoothing iterations. Default is 10.
            edge_preserving_value (float): Smoothing parameter. Lower values preserve more edges. Default is 0.1.
            feature_angle (float): Angle to preserve sharp features. Default is 45.0 degrees.

        Raises:
            ValueError: If no mesh has been generated.
        """
        if self.mesh is None:
            raise ValueError("No mesh generated. Use generate_mesh() first.")

        self.mesh = self.mesh.smooth_taubin(n_iter=iterations, pass_band=edge_preserving_value,
                                            feature_angle=feature_angle)

    def refine_mesh(self) -> None:
        """
        Refine the mesh by applying Laplacian and bilateral smoothing.

        This method applies both Laplacian and bilateral smoothing to improve the overall mesh quality.

        Raises:
            ValueError: If no mesh has been generated.
        """
        if self.mesh is None:
            raise ValueError("No mesh generated. Use generate_mesh() first.")

        self.apply_laplacian_smoothing()
        self.apply_bilateral_smoothing()

    def visualize_point_cloud(self) -> None:
        """
        Visualize the loaded point cloud.

        Raises:
            ValueError: If no point cloud data has been loaded.
        """
        if self.point_cloud is None:
            raise ValueError("No point cloud data loaded. Use load_point_cloud_from_csv() first.")

        point_cloud = pv.PolyData(self.point_cloud)
        point_cloud.plot(eye_dome_lighting=True)

    def visualize_mesh(self) -> None:
        """
        Visualize the generated mesh.

        Raises:
            ValueError: If no mesh has been generated.
        """
        if self.mesh is None:
            raise ValueError("No mesh generated. Use generate_mesh() first.")

        self.mesh.plot(color='orange')

    def visualize_mesh_with_point_cloud(self) -> None:
        """
        Visualize the loaded point cloud and the generated mesh together.

        Raises:
            ValueError: If no point cloud data has been loaded.
        """
        if self.point_cloud is None:
            raise ValueError("No point cloud data loaded. Use load_point_cloud_from_csv() first.")

        plotter = pv.Plotter()

        point_cloud = pv.PolyData(self.point_cloud)
        plotter.add_mesh(point_cloud, color='white', point_size=5, render_points_as_spheres=True, name='Point Cloud')

        if self.mesh is not None:
            plotter.add_mesh(self.mesh, color='orange', opacity=0.5, name='Mesh')

        plotter.show()


def visualize_in_process(func):
    """
    Run a visualization function in a separate process.

    This decorator allows for non-blocking visualization by running the visualization
    function in a separate process.

    Args:
        func (callable): The visualization function to run in a separate process.

    Returns:
        multiprocessing.Process: The process running the visualization function.
    """
    proc = multiprocessing.Process(target=func)
    proc.start()
    return proc


# Example usage
if __name__ == "__main__":
    pc_to_mesh = PointCloudToMesh()
    pc_to_mesh.load_point_cloud_from_csv("point_cloud.csv")
    proc_point_cloud = visualize_in_process(pc_to_mesh.visualize_point_cloud)

    pc_to_mesh.generate_mesh()
    proc_mesh = visualize_in_process(pc_to_mesh.visualize_mesh)

    pc_to_mesh.refine_mesh()
    proc_refined_mesh = visualize_in_process(pc_to_mesh.visualize_mesh)

    proc_point_cloud.join()
    proc_mesh.join()
    proc_refined_mesh.join()
