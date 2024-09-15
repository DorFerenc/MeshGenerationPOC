import numpy as np
from PIL import Image
import os
from .utils.error_handling import OBJConversionError

class OBJConverter:
    """
    Converts textured meshes to OBJ format.

    This class provides functionality to convert a textured mesh to OBJ format,
    including generating the associated MTL file and texture image.
    """

    def __init__(self):
        """Initialize the OBJConverter."""
        self.mesh = None
        self.texture_mapper = None

    # def convert(self, textured_mesh):
    #     """
    #     Convert the textured mesh to OBJ format.
    #
    #     Args:
    #         textured_mesh (pyvista.PolyData): The textured mesh to be converted.
    #
    #     Returns:
    #         dict: A dictionary containing the OBJ content, MTL content, and texture image.
    #
    #     Raises:
    #         OBJConversionError: If the conversion process fails.
    #     """
    #     try:
    #         self.mesh = textured_mesh
    #         obj_content = self.generate_obj_content()
    #         mtl_content = self.generate_mtl_content()
    #         texture_image = self.generate_texture_image()
    #
    #         return {
    #             "obj_content": obj_content,
    #             "mtl_content": mtl_content,
    #             "texture_image": texture_image
    #         }
    #     except Exception as e:
    #         raise OBJConversionError(f"Failed to convert mesh to OBJ: {str(e)}")
    #
    def convert(self, textured_mesh, output_folder):
        """
        Convert the textured mesh to OBJ format and save the results to disk.

        Args:
            textured_mesh (pyvista.PolyData): The textured mesh to be converted.
            output_folder (str): The folder where the output files will be saved.

        Returns:
            dict: A dictionary containing the OBJ content, MTL content, and texture image.

        Raises:
            OBJConversionError: If the conversion process fails.
        """
        try:
            self.mesh = textured_mesh
            obj_content = self.generate_obj_content()
            mtl_content = self.generate_mtl_content()
            texture_image = self.generate_texture_image()

            # Save files to the specified output folder
            self.save_obj_file(os.path.join(output_folder, "model.obj"), obj_content)
            self.save_mtl_file(os.path.join(output_folder, "material.mtl"), mtl_content)
            self.save_texture_image(os.path.join(output_folder, "texture.png"), texture_image)

            return {
                "obj_content": obj_content,
                "mtl_content": mtl_content,
                "texture_image": texture_image
            }
        except Exception as e:
            raise OBJConversionError(f"Failed to convert mesh to OBJ: {str(e)}")

    def save_obj_file(self, filepath, content):
        """Save the OBJ content to a file."""
        with open(filepath, 'w') as file:
            file.write(content)

    def save_mtl_file(self, filepath, content):
        """Save the MTL content to a file."""
        with open(filepath, 'w') as file:
            file.write(content)

    def save_texture_image(self, filepath, image):
        """Save the texture image to a file."""
        image.save(filepath)

    def generate_obj_content(self):
        """
        Generate the content for the OBJ file.

        Returns:
            str: The content of the OBJ file.

        Raises:
            OBJConversionError: If OBJ content generation fails.
        """
        if 'UV' not in self.mesh.point_data:
            raise OBJConversionError("Mesh does not have texture coordinates.")

        vertices = self.mesh.points
        faces = self.mesh.faces
        texture_coords = self.mesh.point_data['UV']

        obj_lines = ["# OBJ file", "mtllib material.mtl", "o TexturedMesh", ""]

        for v in vertices:
            obj_lines.append(f"v {v[0]} {v[1]} {v[2]}")

        obj_lines.append("")

        for vt in texture_coords:
            obj_lines.append(f"vt {vt[0]} {1 - vt[1]}")

        obj_lines.append("")
        obj_lines.append("usemtl material0")

        face_index = 0
        while face_index < len(faces):
            n_vertices = faces[face_index]
            face_vertex_indices = faces[face_index + 1:face_index + 1 + n_vertices]
            face_str = " ".join([f"{vi + 1}/{vi + 1}" for vi in face_vertex_indices])
            obj_lines.append(f"f {face_str}")
            face_index += n_vertices + 1

        return "\n".join(obj_lines)

    def generate_mtl_content(self):
        """
        Generate the content for the MTL file.

        Returns:
            str: The content of the MTL file.
        """
        mtl_lines = [
            "# MTL file",
            "newmtl material0",
            "Ka 1.000 1.000 1.000",
            "Kd 1.000 1.000 1.000",
            "Ks 0.000 0.000 0.000",
            "d 1.0",
            "illum 2",
            "map_Kd texture.png"
        ]
        return "\n".join(mtl_lines)

    def generate_texture_image(self):
        """
        Generate the texture image for the mesh.

        Returns:
            PIL.Image.Image: The generated texture image.

        Raises:
            OBJConversionError: If texture image generation fails.
        """
        if 'RGB' not in self.mesh.point_data or 'UV' not in self.mesh.point_data:
            raise OBJConversionError("Mesh must have RGB and UV data for texture image generation.")

        colors = self.mesh.point_data['RGB']
        uv_coords = self.mesh.point_data['UV']

        texture_resolution = 1024  # You can adjust this or make it a parameter
        texture_image = np.zeros((texture_resolution, texture_resolution, 3), dtype=np.uint8)

        for i, uv in enumerate(uv_coords):
            x = int(uv[0] * (texture_resolution - 1))
            y = int((1 - uv[1]) * (texture_resolution - 1))
            texture_image[y, x] = (colors[i] * 255).astype(np.uint8)

        return Image.fromarray(texture_image)
