from PointCloudToMesh import PointCloudToMesh

def main():
    pc_to_mesh = PointCloudToMesh()

    # Step 1: Load and visualize point cloud
    pc_to_mesh.load_point_cloud_from_csv("colored_sphere_point_cloud.csv")
    pc_to_mesh.visualize_point_cloud()

    # Step 2: Generate and visualize mesh
    pc_to_mesh.generate_mesh()
    pc_to_mesh.visualize_mesh()

    # Apply mesh refinement
    pc_to_mesh.refine_mesh()

    # Step 3: Apply texture mapping and visualize textured mesh
    pc_to_mesh.apply_texture_mapping()
    pc_to_mesh.visualize_textured_mesh()

    # Optionally, save the textured mesh
    pc_to_mesh.save_mesh("textured_sphere_mesh.ply")


if __name__ == "__main__":
    main()
