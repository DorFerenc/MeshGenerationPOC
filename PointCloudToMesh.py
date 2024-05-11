import numpy as np
import pyvista as pv


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
        self.mesh = poly_data.delaunay_3d()

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


# Example usage:
if __name__ == "__main__":
    # Instantiate the PointCloudToMesh class
    pc_to_mesh = PointCloudToMesh()

    # Load point cloud data from CSV file
    pc_to_mesh.load_point_cloud_from_csv("point_cloud.csv")

    # Generate mesh from the loaded point cloud data
    pc_to_mesh.generate_mesh()

    # Visualize the point cloud
    pc_to_mesh.visualize_point_cloud()

    # Visualize the generated mesh
    pc_to_mesh.visualize_mesh()
