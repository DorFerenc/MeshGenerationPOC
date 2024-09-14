import numpy as np
import pyvista as pv
import multiprocessing



class PointCloudToMesh:
    def __init__(self):
        self.point_cloud = None
        self.mesh = None

    def load_point_cloud_from_csv(self, csv_file):
        """
        Load point cloud data from a CSV file.

        Args:
        - csv_file (str): Path to the CSV file containing point cloud data.
        """
        # Load CSV file
        self.point_cloud = np.loadtxt(csv_file, delimiter=',')

    def generate_mesh(self):
        """
        Generate a 3D mesh from the loaded point cloud data.
        """
        if self.point_cloud is None:
            print("Error: No point cloud data loaded.")
            return

        # Convert point cloud to PyVista PolyData
        poly_data = pv.PolyData(self.point_cloud)

        # Perform surface reconstruction
        # self.mesh = poly_data.delaunay_3d(alpha=0.02)
        self.mesh = poly_data.delaunay_3d(alpha=0.002)
        self.mesh = self.mesh.extract_surface()

    def apply_laplacian_smoothing(self, iterations=20):
        """
        Apply Laplacian smoothing to the mesh to remove local artifacts.

        Args:
        - iterations (int): Number of smoothing iterations.
        """
        if self.mesh is None:
            print("Error: No mesh generated.")
            return

        self.mesh = self.mesh.smooth(n_iter=iterations)

    def apply_bilateral_smoothing(self, iterations=10, edge_preserving_value=0.1, feature_angle=45.0):
        """
        Apply bilateral smoothing to the mesh to enhance visual quality.

        Args:
        - iterations (int): Number of smoothing iterations.
        - edge_preserving_value (float): Smoothing parameter.
        - feature_angle (float): Angle to preserve sharp features.
        """
        if self.mesh is None:
            print("Error: No mesh generated.")
            return

        self.mesh = self.mesh.smooth_taubin(n_iter=iterations, feature_angle=feature_angle)

    def refine_mesh(self):
        """
        Refine the mesh by applying Laplacian and bilateral smoothing.
        """
        self.apply_laplacian_smoothing()
        self.apply_bilateral_smoothing()


    def visualize_point_cloud(self):
        """
        Visualize the loaded point cloud.
        """
        if self.point_cloud is None:
            print("Error: No point cloud data loaded.")
            return

        point_cloud = pv.PolyData(self.point_cloud)
        point_cloud.plot(eye_dome_lighting=True)

    def visualize_mesh(self):
        """
        Visualize the generated mesh.
        """
        if self.mesh is None:
            print("Error: No mesh generated.")
            return

        self.mesh.plot(color='orange')

    def visualize_mesh_with_point_cloud(self):
        """
        Visualize the loaded point cloud and the generated mesh together.
        """
        if self.point_cloud is None:
            print("Error: No point cloud data loaded.")
            return

        plotter = pv.Plotter()

        # Add point cloud
        point_cloud = pv.PolyData(self.point_cloud)
        plotter.add_mesh(point_cloud, color='white', point_size=5, render_points_as_spheres=True, name='Point Cloud')

        # Add mesh if it exists
        if self.mesh is not None:
            plotter.add_mesh(self.mesh, color='orange', name='Mesh')

        plotter.show()


def visualize_in_process(func):
    """
    Run a visualization function in a separate process.
    """
    proc = multiprocessing.Process(target=func)
    proc.start()
    return proc


# Example usage:
if __name__ == "__main__":
    # Instantiate the PointCloudToMesh class
    pc_to_mesh = PointCloudToMesh()

    # Load point cloud data from CSV file and visualize it.
    pc_to_mesh.load_point_cloud_from_csv("point_cloud.csv")
    proc_point_cloud = visualize_in_process(pc_to_mesh.visualize_point_cloud)

    # Generate mesh from the loaded point cloud data
    pc_to_mesh.generate_mesh()
    proc_mesh = visualize_in_process(pc_to_mesh.visualize_mesh)

    # Refine the generated mesh and visualize it
    pc_to_mesh.refine_mesh()
    proc_refined_mesh = visualize_in_process(pc_to_mesh.visualize_mesh)

    # Wait for the visualizations to complete before proceeding
    proc_point_cloud.join()
    proc_mesh.join()
    proc_refined_mesh.join()
