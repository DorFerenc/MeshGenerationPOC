import numpy as np
import pyvista as pv
from PointCloudToMesh import PointCloudToMesh
from TextureMapper import TextureMapper
import logging
import os
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
    cloud = pv.PolyData(points)
    if colors is not None:
        cloud.point_data["RGB"] = colors

    p = pv.Plotter()
    p.add_mesh(cloud, render_points_as_spheres=True, point_size=5, rgb=True if colors is not None else False)
    p.show()


def main():
    # Load point cloud from CSV
    # csv_file = "colored_sphere_point_cloud.csv"  # Replace with your CSV file name
    csv_file = "colored_cube_point_cloud.csv"  # Replace with your CSV file name
    # csv_file = "colored_torus_point_cloud.csv.csv"  # Replace with your CSV file name
    # csv_file = "point_cloud.csv"  # Replace with your CSV file name
    logger.info(f"Loading point cloud from {csv_file}...")
    points, colors = load_point_cloud_from_csv(csv_file)

    # Generate colors if not present in the original data
    if colors is None:
        logger.info("No color data found. Generating colors based on height...")
        colors = generate_colors(points, method='height')  # You can change the method if desired

    # Step 1: Visualize point cloud
    logger.info("Visualizing point cloud...")
    visualize_point_cloud(points, colors)

    # Create PointCloudToMesh object
    pc_to_mesh = PointCloudToMesh()
    pc_to_mesh.set_point_cloud(points)

    # Step 2: Generate and visualize mesh
    logger.info("Generating mesh...")
    try:
        # Calculate optimal alpha and generate mesh
        optimal_alpha = pc_to_mesh.calculate_optimal_alpha()
        pc_to_mesh.generate_mesh(alpha=optimal_alpha)
        logger.info("Mesh generated successfully.")
    except Exception as e:
        logger.error(f"Error generating mesh: {str(e)}")
        return

    logger.info("Visualizing generated mesh...")
    pc_to_mesh.visualize_mesh()

    # Step 3: Apply texture mapping and visualize textured mesh
    logger.info("Applying texture mapping...")
    texture_mapper = TextureMapper()
    texture_mapper.load_mesh(pc_to_mesh.mesh)
    texture_mapper.load_point_cloud_with_colors(points, colors)
    texture_mapper.map_colors_to_mesh()
    texture_mapper.apply_uv_mapping()
    texture_mapper.smooth_texture()
    textured_mesh = texture_mapper.get_textured_mesh()

    logger.info("Visualizing textured mesh...")
    p = pv.Plotter()
    p.add_mesh(textured_mesh, rgb=True)
    p.show()

    # Optionally, save the mesh
    try:
        output_file = "output_mesh.ply"
        pc_to_mesh.save_mesh(output_file)
        logger.info(f"Mesh saved as '{output_file}'")
    except Exception as e:
        logger.error(f"Error saving mesh: {str(e)}")


if __name__ == "__main__":
    main()
