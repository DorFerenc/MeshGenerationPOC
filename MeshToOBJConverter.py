import numpy as np
from PIL import Image
import os
import logging


class MeshToOBJConverter:
    """
    A class for converting textured meshes to OBJ format.

    This class provides functionality to convert a textured mesh to OBJ format,
    including generating the associated MTL file and texture image.

    Attributes:
        mesh (pyvista.PolyData): The textured mesh to be converted.
        texture_mapper (TextureMapper): The TextureMapper object used for texture generation.
        logger (logging.Logger): Logger for the class.
    """

    def __init__(self, mesh, texture_mapper):
        """
        Initialize the MeshToOBJConverter.

        Args:
            mesh (pyvista.PolyData): The textured mesh to be converted.
            texture_mapper (TextureMapper): The TextureMapper object used for texture generation.
        """
        self.mesh = mesh
        self.texture_mapper = texture_mapper
        self.logger = logging.getLogger(self.__class__.__name__)

    def convert_to_obj(self, output_filename):
        """
        Convert the mesh to OBJ format and save it.

        This method writes the mesh geometry (vertices, texture coordinates, and faces)
        to an OBJ file.

        Args:
            output_filename (str): The name of the output OBJ file.

        Raises:
            ValueError: If the mesh does not have texture coordinates.
        """
        if 'UV' not in self.mesh.point_data:
            raise ValueError("Mesh does not have texture coordinates. Apply UV mapping before converting to OBJ.")

        vertices = self.mesh.points
        faces = self.mesh.faces

        with open(output_filename, 'w') as f:
            f.write("# OBJ file\n")

            mtl_filename = output_filename.rsplit('.', 1)[0] + '.mtl'
            f.write(f"mtllib {os.path.basename(mtl_filename)}\n")

            # Write vertices
            for v in vertices:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")

            # Write texture coordinates
            texture_coords = self.mesh.point_data['UV']
            for vt in texture_coords:
                f.write(f"vt {vt[0]} {1 - vt[1]}\n")  # Flip V coordinate

            f.write("g TexturedMesh\n")
            f.write("usemtl material0\n")

            # Write faces
            face_index = 0
            while face_index < len(faces):
                n_vertices = faces[face_index]  # First element is the number of vertices in the face
                face_vertex_indices = faces[face_index + 1:face_index + 1 + n_vertices]
                face_str = " ".join([f"{vi + 1}/{vi + 1}" for vi in face_vertex_indices])
                f.write(f"f {face_str}\n")
                face_index += n_vertices + 1

        self.logger.info(f"OBJ file saved as {output_filename}")

    def save_texture_image(self, texture_filename):
        """
        Generate and save the texture image.

        This method uses the TextureMapper to generate a texture image and saves it
        to a file.

        Args:
            texture_filename (str): The name of the output texture image file.

        Raises:
            Exception: If there's an error during texture image generation or saving.
        """
        try:
            texture_image = self.texture_mapper.generate_texture_image()

            # Convert to 8-bit color
            texture_image_8bit = (texture_image * 255).astype(np.uint8)

            image = Image.fromarray(texture_image_8bit)
            image.save(texture_filename)
            self.logger.info(f"Texture image saved as {texture_filename}")
        except Exception as e:
            self.logger.error(f"Error saving texture image: {str(e)}")
            raise

    def create_mtl_file(self, obj_filename, texture_filename):
        """
        Create the Material Template Library (MTL) file for the OBJ model.

        The MTL file defines the material properties of the 3D model, including
        color, texture, and lighting characteristics. This method creates a basic
        MTL file that references the texture image generated for the model.

        Args:
           obj_filename (str): The filename of the OBJ file. The MTL filename is derived from this.
           texture_filename (str): The filename of the texture image to be referenced in the MTL file.

        Note:
           The MTL file is saved with the same name as the OBJ file but with a .mtl extension.
           It defines a single material named 'material0' with default properties and links
           to the specified texture file.

        Raises:
           IOError: If there's an error writing the MTL file.
        """
        mtl_filename = obj_filename.rsplit('.', 1)[0] + '.mtl'
        with open(mtl_filename, 'w') as f:
            f.write("# MTL file\n")
            f.write("newmtl material0\n")
            f.write("Ka 1.000 1.000 1.000\n")  # Ambient color
            f.write("Kd 1.000 1.000 1.000\n")  # Diffuse color
            f.write("Ks 0.000 0.000 0.000\n")  # Specular color
            f.write("d 1.0\n")  # Opacity
            f.write("illum 2\n")  # Illumination model
            f.write(f"map_Kd {os.path.basename(texture_filename)}\n")

        self.logger.info(f"MTL file saved as {mtl_filename}")

    def convert_and_save(self, obj_filename, texture_filename):
        """
        Convert the mesh to OBJ format and save all associated files.


        This method orchestrates the entire conversion process, including:
        1. Converting the mesh to OBJ format and saving it
        2. Generating and saving the texture image
        3. Creating and saving the Material Template Library (MTL) file

        The method ensures that all necessary files (OBJ, MTL, and texture image)
        are created and properly referenced to represent a complete textured 3D model.

        Args:
            obj_filename (str): The name of the output OBJ file. This should include
                                the full path if you want to save it in a specific directory.
            texture_filename (str): The name of the output texture image file. This should
                                    include the full path if you want to save it in a specific directory.

        Raises:
            Exception: If there's an error during any part of the conversion process.
                       This could include file I/O errors, texture generation errors,
                       or any other exceptions raised by the component methods.
        """
        try:
            self.convert_to_obj(obj_filename)
            self.save_texture_image(texture_filename)
            self.create_mtl_file(obj_filename, texture_filename)
            self.logger.info(f"OBJ file saved as {obj_filename}")
            self.logger.info(f"Texture image saved as {texture_filename}")
        except Exception as e:
            self.logger.error(f"Error in convert_and_save: {str(e)}")
            raise
