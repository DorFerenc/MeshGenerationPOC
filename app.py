import numpy as np
import pyvista as pv
from PointCloudToMesh import PointCloudToMesh
from TextureMapper import TextureMapper
import logging
import os  # Add this import

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_cube_point_cloud(num_points_per_face=1000):
    """
    Generate a point cloud representing a cube.

    Args:
        num_points_per_face (int): Number of points to generate per face of the cube.

    Returns:
        tuple: (points, colors) where points is a numpy array of 3D coordinates
               and colors is a numpy array of RGB values.
    """
    points = []
    colors = []

    for i in range(6):  # 6 faces of a cube
        if i < 2:  # Front and back faces (fix z)
            x = np.random.rand(num_points_per_face)
            y = np.random.rand(num_points_per_face)
            z = np.full(num_points_per_face, i)
            face_points = np.column_stack((x, y, z))
        elif i < 4:  # Left and right faces (fix x)
            y = np.random.rand(num_points_per_face)
            z = np.random.rand(num_points_per_face)
            x = np.full(num_points_per_face, i - 2)
            face_points = np.column_stack((x, y, z))
        else:  # Top and bottom faces (fix y)
            x = np.random.rand(num_points_per_face)
            z = np.random.rand(num_points_per_face)
            y = np.full(num_points_per_face, i - 4)
            face_points = np.column_stack((x, y, z))

        points.append(face_points)

        # Generate a random color for each face
        face_color = np.random.rand(3)
        colors.append(np.tile(face_color, (num_points_per_face, 1)))

    points = np.vstack(points)
    colors = np.vstack(colors)

    return points, colors


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
    p.add_mesh(cloud, render_points_as_spheres=True, point_size=5, rgb=True)
    p.show()


def main():
    # Generate cube point cloud
    logger.info("Generating cube point cloud...")
    points, colors = generate_cube_point_cloud()

    # Step 1: Visualize point cloud
    logger.info("Visualizing point cloud...")
    visualize_point_cloud(points, colors)

    # Create PointCloudToMesh object
    pc_to_mesh = PointCloudToMesh()
    pc_to_mesh.set_point_cloud(points)

    # Step 2: Generate and visualize mesh
    logger.info("Generating mesh...")
    try:
        pc_to_mesh.generate_mesh(alpha=0.1)  # Increased alpha for a coarser mesh
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

    # Optionally, save the textured mesh
    try:
        pc_to_mesh.save_mesh("textured_cube_mesh.ply")
        logger.info("Mesh saved as 'textured_cube_mesh.ply'")
    except Exception as e:
        logger.error(f"Error saving mesh: {str(e)}")


if __name__ == "__main__":
    main()