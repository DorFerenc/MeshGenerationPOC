import numpy as np


def generate_sphere_point_cloud(num_points=10000, radius=1.0):
    """
    Generate a colored point cloud representing a sphere.

    Args:
        num_points (int): Number of points to generate.
        radius (float): Radius of the sphere.

    Returns:
        numpy.ndarray: Array of points with their coordinates and colors.
    """
    # Generate random points on a sphere
    phi = np.random.uniform(0, 2 * np.pi, num_points)
    cos_theta = np.random.uniform(-1, 1, num_points)
    theta = np.arccos(cos_theta)

    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * cos_theta

    # Generate colors based on position
    r = (x + radius) / (2 * radius)
    g = (y + radius) / (2 * radius)
    b = (z + radius) / (2 * radius)

    # Combine coordinates and colors
    points = np.column_stack((x, y, z, r, g, b))
    return points

def save_point_cloud_to_csv(points, filename):
    """
    Save the point cloud to a CSV file with a header.

    Args:
        points (numpy.ndarray): Array of points with their coordinates and colors.
        filename (str): Name of the CSV file to save.
    """
    header = "x,y,z,r,g,b"
    np.savetxt(filename, points, delimiter=',', header=header, comments='')
    print(f"Point cloud saved to {filename}")


if __name__ == "__main__":
    # Generate a colored point cloud
    point_cloud = generate_sphere_point_cloud()

    # Save the point cloud to a CSV file
    save_point_cloud_to_csv(point_cloud, "colored_sphere_point_cloud.csv")
