import numpy as np
import pyvista as pv
from PointCloudToMesh import PointCloudToMesh, MeshRefiner
from TextureMapper import TextureMapper
from MeshToOBJConverter import MeshToOBJConverter
import logging
import os
import pandas as pd
import traceback
import multiprocessing


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MeshGenerator:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.pc_to_mesh = PointCloudToMesh()
        self.texture_mapper = TextureMapper()
        self.mesh_refiner = MeshRefiner()

    def generate_mesh(self, point_cloud):
        self.pc_to_mesh.set_point_cloud(point_cloud)
        return self.pc_to_mesh.generate_mesh()

    def refine_mesh(self, mesh):
        return self.mesh_refiner.refine_mesh(mesh)

    def apply_texture(self, mesh, point_cloud, colors):
        self.texture_mapper.load_mesh(mesh)
        self.texture_mapper.load_point_cloud_with_colors(point_cloud, colors)
        self.texture_mapper.apply_texture()
        return self.texture_mapper.get_textured_mesh()

    def save_as_obj(self, textured_mesh, obj_filename, texture_filename):
        try:
            converter = MeshToOBJConverter(textured_mesh, self.texture_mapper)
            converter.convert_and_save(obj_filename, texture_filename)
        except Exception as e:
            self.logger.error(f"Error saving as OBJ: {str(e)}")
            raise


def load_point_cloud_from_csv(filename):
    """
    Load point cloud data from a CSV file.

    Args:
        filename (str): Path to the CSV file containing point cloud data.

    Returns:
        tuple: (points, colors) where points is a numpy array of 3D coordinates
               and colors is a numpy array of RGB values (or None if not present).
    """
    try:
        # Read the CSV file
        df = pd.read_csv(filename)

        # Check if the file has a header
        if not all(col in df.columns for col in ['x', 'y', 'z']):
            df = pd.read_csv(filename, header=None)
            df.columns = ['x', 'y', 'z'] + [f'col_{i}' for i in range(3, len(df.columns))]

        # Extract points
        points = df[['x', 'y', 'z']].values

        # Extract colors if present
        if all(col in df.columns for col in ['r', 'g', 'b']):
            colors = df[['r', 'g', 'b']].values
        elif len(df.columns) > 3:
            colors = df.iloc[:, 3:6].values
        else:
            colors = None

        logger.info(f"Loaded {len(points)} points from {filename}")
        return points, colors
    except Exception as e:
        logger.error(f"Error loading point cloud from CSV: {str(e)}")
        raise


def generate_colors(points, method='random'):
    """
    Generate colors for points if original data doesn't include color information.

    Args:
        points (numpy.ndarray): Array of 3D point coordinates.
        method (str): Method to generate colors. Options: 'random', 'height', 'distance'

    Returns:
        numpy.ndarray: Array of RGB color values for each point.
    """
    if method == 'random':
        return np.random.rand(len(points), 3)
    elif method == 'height':
        # Color based on height (z-coordinate)
        z_values = points[:, 2]
        colors = np.zeros((len(points), 3))
        colors[:, 0] = (z_values - z_values.min()) / (z_values.max() - z_values.min())  # Red channel
        colors[:, 2] = 1 - colors[:, 0]  # Blue channel
        return colors
    elif method == 'distance':
        # Color based on distance from center
        center = np.mean(points, axis=0)
        distances = np.linalg.norm(points - center, axis=1)
        colors = np.zeros((len(points), 3))
        colors[:, 0] = (distances - distances.min()) / (distances.max() - distances.min())  # Red channel
        colors[:, 2] = 1 - colors[:, 0]  # Blue channel
        return colors
    else:
        raise ValueError(f"Unknown color generation method: {method}")


def visualize_point_cloud(points, colors=None):
    """
    Visualize the point cloud.

    Args:
        points (numpy.ndarray): Array of 3D point coordinates.
        colors (numpy.ndarray, optional): Array of RGB color values for each point.
    """
    plotter = pv.Plotter()
    point_cloud = pv.PolyData(points)
    if colors is not None:
        point_cloud["colors"] = colors
    plotter.add_mesh(point_cloud, render_points_as_spheres=True, point_size=5, rgb=True if colors is not None else False)
    plotter.show()

def visualize_mesh(mesh):
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color='orange', show_edges=True)
    plotter.show()

def visualize_textured_mesh(mesh):
    plotter = pv.Plotter()
    if 'RGB' in mesh.point_data:
        plotter.add_mesh(mesh, scalars='RGB', rgb=True)
    else:
        plotter.add_mesh(mesh, color='white')
    plotter.show()

def main():
    # Load point cloud from CSV
    # csv_file = "colored_sphere_point_cloud.csv"  # Replace with your CSV file name
    # csv_file = "colored_cube_point_cloud.csv"  # Replace with your CSV file name
    # csv_file = "colored_torus_point_cloud.csv.csv"  # Replace with your CSV file name
    csv_file = "point_cloud.csv"
    logger.info(f"Loading point cloud from {csv_file}...")
    points, colors = load_point_cloud_from_csv(csv_file)

    # Visualize point cloud
    p = multiprocessing.Process(target=visualize_point_cloud, args=(points, colors))
    p.start()

    # Generate colors if not present in the original data
    if colors is None:
        logger.info("No color data found. Generating colors based on height...")
        colors = generate_colors(points, method='height')

    # Create MeshGenerator instance
    mesh_generator = MeshGenerator()

    try:
        # Generate textured mesh
        logger.info("Generating textured mesh...")
        mesh = mesh_generator.generate_mesh(points)

        # Visualize generated mesh
        p = multiprocessing.Process(target=visualize_mesh, args=(mesh,))
        p.start()

        # Refine mesh
        logger.info("Refining mesh...")
        refined_mesh = mesh_generator.mesh_refiner.refine_mesh(mesh)

        # Visualize refined mesh
        p = multiprocessing.Process(target=visualize_mesh, args=(refined_mesh,))
        p.start()

        # Apply texture
        logger.info("Applying texture...")
        textured_mesh = mesh_generator.apply_texture(refined_mesh, points, colors)

        # Visualize textured mesh
        p = multiprocessing.Process(target=visualize_textured_mesh, args=(textured_mesh,))
        p.start()

        # Save as OBJ
        obj_filename = "output_mesh.obj"
        texture_filename = "output_texture.png"
        logger.info("Converting textured mesh to OBJ format...")
        mesh_generator.save_as_obj(textured_mesh, obj_filename, texture_filename)

        logger.info("Process completed successfully.")
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")


if __name__ == "__main__":
    main()
