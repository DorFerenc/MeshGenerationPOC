import os
import uuid
import numpy as np
import pyvista as pv
import logging
import pandas as pd
import multiprocessing
from reconstruction.point_cloud_processor import PointCloudProcessor
from reconstruction.mesh_generator import MeshGenerator
from reconstruction.texture_mapper import TextureMapper
from reconstruction.obj_converter import OBJConverter
import tempfile
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def visualize_point_cloud(points, colors=None, output_folder=None):
    logger.info("Visualizing point cloud")
    plotter = pv.Plotter()
    point_cloud = pv.PolyData(points)
    if colors is not None:
        point_cloud["colors"] = colors
    plotter.add_mesh(point_cloud, render_points_as_spheres=True, point_size=5, rgb=True if colors is not None else False)
    plotter.show()


def visualize_mesh(mesh, output_folder=None):
    logger.info("Visualizing mesh")
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color='orange', show_edges=True)
    plotter.show()


def visualize_textured_mesh(mesh, output_folder=None):
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
    r = np.random.uniform(0.8, 1, num_points)

    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)

    points = np.column_stack((x, y, z))

    # Generate colors based on position
    colors = np.column_stack((
        (x + 1) / 2,  # Red channel
        (y + 1) / 2,  # Green channel
        (z + 1) / 2   # Blue channel
    ))

    return np.column_stack((points, colors))


def load_point_cloud_from_csv(file_path):
    """Load point cloud data from a CSV file."""
    logger.info(f"Loading point cloud from CSV: {file_path}")
    df = pd.read_csv(file_path)

    if not all(col in df.columns for col in ['x', 'y', 'z']):
        raise ValueError("CSV file must contain 'x', 'y', and 'z' columns.")

    points = df[['x', 'y', 'z']].values
    if all(col in df.columns for col in ['r', 'g', 'b']):
        colors = df[['r', 'g', 'b']].values
        return np.column_stack((points, colors))

    return points


def run_visualization_pipeline(point_cloud_with_colors, output_folder, model_name=None):
    logger.info("Starting visualization pipeline")
    point_cloud_processor = PointCloudProcessor()
    mesh_generator = MeshGenerator()
    texture_mapper = TextureMapper()
    obj_converter = OBJConverter()

    # Generate a unique model name if not provided
    model_name = model_name or f"model_{uuid.uuid4().hex[:8]}"
    model_folder = os.path.join(output_folder, model_name)
    os.makedirs(model_folder, exist_ok=True)

    points = point_cloud_with_colors[:, :3]
    colors = point_cloud_with_colors[:, 3:] if point_cloud_with_colors.shape[1] > 3 else None

    # Visualize point cloud
    logger.info("Visualizing initial point cloud")
    p = multiprocessing.Process(target=visualize_point_cloud, args=(points, colors))
    p.start()

    # Process point cloud
    logger.info("Processing point cloud")
    processed_point_cloud = point_cloud_processor.process(point_cloud_with_colors)

    # Generate and visualize mesh
    logger.info("Generating mesh")
    mesh, alpha = mesh_generator.generate_mesh(processed_point_cloud[:, :3])
    logger.info(f"Mesh generated with alpha value: {alpha:.6f}")
    logger.info(f"Mesh generated with {mesh.n_points} points and {mesh.n_cells} cells")
    p2 = multiprocessing.Process(target=visualize_mesh, args=(mesh,))
    p2.start()

    # Apply texture and visualize
    logger.info("Applying texture")
    try:
        textured_mesh = texture_mapper.apply_texture(mesh, processed_point_cloud)
        logger.info("Texture applied successfully")
        p3 = multiprocessing.Process(target=visualize_textured_mesh, args=(textured_mesh,))
        p3.start()

        # Wait for all processes to finish before saving the object
        logger.info("Waiting for visualization processes to complete")
        p.join()
        p2.join()
        p3.join()

        # Save the textured mesh as OBJ after all processes have finished
        logger.info("Saving textured mesh as OBJ")
        obj_converter.convert(textured_mesh, model_folder)
        logger.info(f"OBJ, MTL, and texture files saved in {model_folder}")

    except Exception as e:
        logger.error(f"Error applying texture: {str(e)}")
        raise

    logger.info(f"Visualization pipeline completed. Files saved in {model_folder}")


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

    # If i use this then also uncomment at the end to delete this at the end
    # output_folder = tempfile.mkdtemp()
    # print(f"Saving results to {output_folder}")

    # Change the output to a persistent folder instead of a temp folder
    output_folder = os.path.join(os.getcwd(), "output")
    os.makedirs(output_folder, exist_ok=True)
    print(f"Saving results to {output_folder}")

    if choice == 1:
        point_cloud_with_colors = generate_sample_point_cloud()
    else:
        csv_file = options[choice - 1]
        if not os.path.exists(csv_file):
            logger.error(f"Error: File {csv_file} not found. Make sure it's in the same directory as this script.")
            exit(1)
        point_cloud_with_colors = load_point_cloud_from_csv(csv_file)

    run_visualization_pipeline(point_cloud_with_colors, output_folder)

    # Uncomment if you want to remove the temporary output folder after testing
    # shutil.rmtree(output_folder)
