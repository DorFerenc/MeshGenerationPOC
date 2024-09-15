import numpy as np
import pyvista as pv
import multiprocessing
import pandas as pd
import os
import logging
from reconstruction.point_cloud_processor import PointCloudProcessor
from reconstruction.mesh_generator import MeshGenerator
from reconstruction.texture_mapper import TextureMapper

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def visualize_point_cloud(points, colors=None):
    logger.info("Visualizing point cloud")
    plotter = pv.Plotter()
    point_cloud = pv.PolyData(points)
    if colors is not None:
        point_cloud["colors"] = colors
    plotter.add_mesh(point_cloud, render_points_as_spheres=True, point_size=5, rgb=True if colors is not None else False)
    plotter.show()

def visualize_mesh(mesh):
    logger.info("Visualizing mesh")
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color='orange', show_edges=True)
    plotter.show()

def visualize_textured_mesh(mesh):
    logger.info("Visualizing textured mesh")
    plotter = pv.Plotter()
    if 'RGB' in mesh.point_data:
        plotter.add_mesh(mesh, scalars='RGB', rgb=True)
    else:
        plotter.add_mesh(mesh, color='white')
    plotter.show()

def generate_sample_point_cloud(num_points=1000):
    logger.info(f"Generating sample point cloud with {num_points} points")
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    phi = np.random.uniform(0, np.pi, num_points)
    r = np.random.uniform(0.8, 1, num_points)  # Slight variation in radius

    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)

    points = np.column_stack((x, y, z))

    # Generate colors based on position
    colors = np.column_stack((
        (x + 1) / 2,  # Red channel
        (y + 1) / 2,  # Green channel
        (z + 1) / 2  # Blue channel
    ))

    return np.column_stack((points, colors))


def load_point_cloud_from_csv(file_path):
    logger.info(f"Loading point cloud from CSV: {file_path}")
    try:
        df = pd.read_csv(file_path)
        if not all(col in df.columns for col in ['x', 'y', 'z']):
            df = pd.read_csv(file_path, header=None)
            df.columns = ['x', 'y', 'z'] + [f'col_{i}' for i in range(3, len(df.columns))]

        points = df[['x', 'y', 'z']].values
        if all(col in df.columns for col in ['r', 'g', 'b']):
            colors = df[['r', 'g', 'b']].values
        elif len(df.columns) > 3:
            colors = df.iloc[:, 3:6].values
        else:
            colors = None

        logger.info(f"Loaded point cloud with {len(points)} points")
        logger.info(f"Point cloud shape: {points.shape}")
        logger.info(f"Color data available: {'Yes' if colors is not None else 'No'}")

        if colors is not None:
            return np.column_stack((points, colors))
        else:
            return points
    except Exception as e:
        logger.error(f"Error loading point cloud from CSV: {str(e)}")
        raise


def generate_colors(points):
    logger.info("Generating colors based on point positions")
    # Normalize x, y, z to [0, 1] range for RGB values
    min_vals = np.min(points, axis=0)
    max_vals = np.max(points, axis=0)
    normalized_points = (points - min_vals) / (max_vals - min_vals)
    return normalized_points


def run_visualization_pipeline(point_cloud_with_colors):
    logger.info("Starting visualization pipeline")
    point_cloud_processor = PointCloudProcessor()
    mesh_generator = MeshGenerator()
    texture_mapper = TextureMapper()

    points = point_cloud_with_colors[:, :3]
    colors = point_cloud_with_colors[:, 3:] if point_cloud_with_colors.shape[1] > 3 else None

    if colors is None:
        logger.warning("No color data found. Generating colors based on point positions.")
        colors = generate_colors(points)
        point_cloud_with_colors = np.column_stack((points, colors))

    # 1. Visualize point cloud
    logger.info("Visualizing initial point cloud")
    p = multiprocessing.Process(target=visualize_point_cloud, args=(points, colors))
    p.start()

    # Process point cloud
    logger.info("Processing point cloud")
    processed_point_cloud = point_cloud_processor.process(point_cloud_with_colors)

    # 2. Generate and visualize mesh
    logger.info("Generating mesh")
    mesh, alpha = mesh_generator.generate_mesh(processed_point_cloud[:, :3])
    logger.info(f"Mesh generated with alpha value: {alpha:.6f}")
    logger.info(f"Mesh generated with {mesh.n_points} points and {mesh.n_cells} cells")
    p2 = multiprocessing.Process(target=visualize_mesh, args=(mesh,))
    p2.start()

    # 3. Apply texture and visualize
    logger.info("Applying texture")
    try:
        # Adjust this line based on the actual signature of apply_texture
        textured_mesh = texture_mapper.apply_texture(mesh, processed_point_cloud)
        logger.info("Texture applied successfully")
        p3 = multiprocessing.Process(target=visualize_textured_mesh, args=(textured_mesh,))
        p3.start()

    except Exception as e:
        logger.error(f"Error applying texture: {str(e)}")
        raise

    # textured_mesh = texture_mapper.apply_texture(mesh, processed_point_cloud[:, :3],
    #                                              processed_point_cloud[:, 3:] if processed_point_cloud.shape[
    #                                                                                  1] > 3 else None)

    # Wait for all processes to finish
    logger.info("Waiting for visualization processes to complete")
    p.join()
    p2.join()
    if 'p3' in locals():
        p3.join()
    # for process in [p, p2, p3]:
    #     process.join()

    logger.info("Visualization pipeline completed")


if __name__ == '__main__':
    options = [
        "Generated point cloud",
        "colored_sphere_point_cloud.csv",
        "colored_cube_point_cloud.csv",
        "point_cloud.csv"
    ]

    print("Choose a point cloud source:")
    for i, option in enumerate(options):
        print(f"{i + 1}. {option}")

    choice = int(input("Enter your choice (1-4): "))

    if choice == 1:
        point_cloud_with_colors = generate_sample_point_cloud()
    else:
        csv_file = options[choice - 1]
        if not os.path.exists(csv_file):
            logger.error(f"Error: File {csv_file} not found. Make sure it's in the same directory as this script.")
            exit(1)
        point_cloud_with_colors = load_point_cloud_from_csv(csv_file)

    run_visualization_pipeline(point_cloud_with_colors)

# import numpy as np
# import pyvista as pv
# import multiprocessing
# import pandas as pd
# import os
# from reconstruction.point_cloud_processor import PointCloudProcessor
# from reconstruction.mesh_generator import MeshGenerator
# from reconstruction.texture_mapper import TextureMapper
#
#
# def visualize_point_cloud(points, colors=None):
#     plotter = pv.Plotter()
#     point_cloud = pv.PolyData(points)
#     if colors is not None:
#         point_cloud["colors"] = colors
#     plotter.add_mesh(point_cloud, render_points_as_spheres=True, point_size=5,
#                      rgb=True if colors is not None else False)
#     plotter.show()
#
#
# def visualize_mesh(mesh):
#     plotter = pv.Plotter()
#     plotter.add_mesh(mesh, color='orange', show_edges=True)
#     plotter.show()
#
#
# def visualize_textured_mesh(mesh):
#     plotter = pv.Plotter()
#     if 'RGB' in mesh.point_data:
#         plotter.add_mesh(mesh, scalars='RGB', rgb=True)
#     else:
#         plotter.add_mesh(mesh, color='white')
#     plotter.show()
#
#
# def generate_sample_point_cloud(num_points=1000):
#     theta = np.random.uniform(0, 2 * np.pi, num_points)
#     phi = np.random.uniform(0, np.pi, num_points)
#     r = np.random.uniform(0.8, 1, num_points)  # Slight variation in radius
#
#     x = r * np.sin(phi) * np.cos(theta)
#     y = r * np.sin(phi) * np.sin(theta)
#     z = r * np.cos(phi)
#
#     points = np.column_stack((x, y, z))
#
#     # Generate colors based on position
#     colors = np.column_stack((
#         (x + 1) / 2,  # Red channel
#         (y + 1) / 2,  # Green channel
#         (z + 1) / 2  # Blue channel
#     ))
#
#     return np.column_stack((points, colors))
#
#
# def load_point_cloud_from_csv(file_path):
#     try:
#         df = pd.read_csv(file_path)
#         if not all(col in df.columns for col in ['x', 'y', 'z']):
#             df = pd.read_csv(file_path, header=None)
#             df.columns = ['x', 'y', 'z'] + [f'col_{i}' for i in range(3, len(df.columns))]
#
#         points = df[['x', 'y', 'z']].values
#         if all(col in df.columns for col in ['r', 'g', 'b']):
#             colors = df[['r', 'g', 'b']].values
#         elif len(df.columns) > 3:
#             colors = df.iloc[:, 3:6].values
#         else:
#             colors = None
#
#         if colors is not None:
#             return np.column_stack((points, colors))
#         else:
#             return points
#     except Exception as e:
#         raise ValueError(f"Error loading point cloud from CSV: {str(e)}")
#
#
# def run_visualization_pipeline(point_cloud_with_colors):
#     point_cloud_processor = PointCloudProcessor()
#     mesh_generator = MeshGenerator()
#     texture_mapper = TextureMapper()
#
#     points = point_cloud_with_colors[:, :3]
#     colors = point_cloud_with_colors[:, 3:] if point_cloud_with_colors.shape[1] > 3 else None
#
#     # 1. Visualize point cloud
#     p = multiprocessing.Process(target=visualize_point_cloud, args=(points, colors))
#     p.start()
#
#     # Process point cloud
#     processed_point_cloud = point_cloud_processor.process(point_cloud_with_colors)
#
#     # 2. Generate and visualize mesh
#     mesh = mesh_generator.generate_mesh(processed_point_cloud[:, :3])
#     p2 = multiprocessing.Process(target=visualize_mesh, args=(mesh,))
#     p2.start()
#
#     # 3. Refine and visualize mesh
#     refined_mesh = mesh_generator.refine_mesh(mesh)
#     p3 = multiprocessing.Process(target=visualize_mesh, args=(refined_mesh,))
#     p3.start()
#
#     # 4. Apply texture and visualize
#     textured_mesh = texture_mapper.apply_texture(refined_mesh, processed_point_cloud[:, :3],
#                                                  processed_point_cloud[:, 3:] if processed_point_cloud.shape[
#                                                                                      1] > 3 else None)
#     p4 = multiprocessing.Process(target=visualize_textured_mesh, args=(textured_mesh,))
#     p4.start()
#
#     # Wait for all processes to finish
#     for process in [p, p2, p3, p4]:
#         process.join()
#
#
# if __name__ == '__main__':
#     options = [
#         "Generated point cloud",
#         "colored_sphere_point_cloud.csv",
#         "colored_cube_point_cloud.csv",
#         "point_cloud.csv"
#     ]
#
#     print("Choose a point cloud source:")
#     for i, option in enumerate(options):
#         print(f"{i + 1}. {option}")
#
#     choice = int(input("Enter your choice (1-4): "))
#
#     if choice == 1:
#         point_cloud_with_colors = generate_sample_point_cloud()
#     else:
#         csv_file = options[choice - 1]
#         if not os.path.exists(csv_file):
#             print(f"Error: File {csv_file} not found. Make sure it's in the same directory as this script.")
#             exit(1)
#         point_cloud_with_colors = load_point_cloud_from_csv(csv_file)
#
#     run_visualization_pipeline(point_cloud_with_colors)