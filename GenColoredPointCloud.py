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


def generate_cube_point_cloud(num_points=10000, size=1.0):
    """
    Generate a colored point cloud representing a cube.

    Args:
        num_points (int): Number of points to generate.
        size (float): Size of the cube.

    Returns:
        numpy.ndarray: Array of points with their coordinates and colors.
    """
    points = []
    for _ in range(num_points):
        # Randomly choose a face of the cube
        face = np.random.randint(0, 6)
        if face < 2:  # Front or back face
            x = np.random.uniform(-size / 2, size / 2)
            y = np.random.uniform(-size / 2, size / 2)
            z = size / 2 if face == 0 else -size / 2
        elif face < 4:  # Left or right face
            y = np.random.uniform(-size / 2, size / 2)
            z = np.random.uniform(-size / 2, size / 2)
            x = size / 2 if face == 2 else -size / 2
        else:  # Top or bottom face
            x = np.random.uniform(-size / 2, size / 2)
            z = np.random.uniform(-size / 2, size / 2)
            y = size / 2 if face == 4 else -size / 2

        # Generate colors based on position
        r = (x + size / 2) / size
        g = (y + size / 2) / size
        b = (z + size / 2) / size

        points.append([x, y, z, r, g, b])

    return np.array(points)


def generate_torus_point_cloud(num_points=10000, major_radius=1.0, minor_radius=0.3):
    """
    Generate a colored point cloud representing a torus.

    Args:
        num_points (int): Number of points to generate.
        major_radius (float): Distance from the center of the tube to the center of the torus.
        minor_radius (float): Radius of the tube.

    Returns:
        numpy.ndarray: Array of points with their coordinates and colors.
    """
    u = np.random.uniform(0, 2 * np.pi, num_points)
    v = np.random.uniform(0, 2 * np.pi, num_points)

    x = (major_radius + minor_radius * np.cos(v)) * np.cos(u)
    y = (major_radius + minor_radius * np.cos(v)) * np.sin(u)
    z = minor_radius * np.sin(v)

    # Generate colors based on position
    r = (np.cos(u) + 1) / 2
    g = (np.sin(u) + 1) / 2
    b = (np.sin(v) + 1) / 2

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
    # Generate colored point clouds
    sphere_cloud = generate_sphere_point_cloud()
    cube_cloud = generate_cube_point_cloud()
    torus_cloud = generate_torus_point_cloud()

    # Save the point clouds to CSV files
    save_point_cloud_to_csv(sphere_cloud, "colored_sphere_point_cloud.csv")
    save_point_cloud_to_csv(cube_cloud, "colored_cube_point_cloud.csv")
    save_point_cloud_to_csv(torus_cloud, "colored_torus_point_cloud.csv")