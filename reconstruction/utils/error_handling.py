class ReconstructionError(Exception):
    """Base exception class for all reconstruction-related errors."""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class InputProcessingError(ReconstructionError):
    """Exception raised for errors in the input processing stage."""
    pass

class MeshGenerationError(ReconstructionError):
    """Exception raised for errors in the mesh generation stage."""
    pass

class TexturingError(ReconstructionError):
    """Exception raised for errors in the texturing stage."""
    pass

class OBJConversionError(ReconstructionError):
    """Exception raised for errors in the OBJ conversion stage."""
    pass

def handle_reconstruction_error(error):
    """
    Handle reconstruction errors and return appropriate error messages.

    Args:
        error (ReconstructionError): The error that occurred during reconstruction.

    Returns:
        dict: A dictionary containing the error type and message.
    """
    error_type = type(error).__name__
    error_message = str(error)

    error_response = {
        "error_type": error_type,
        "error_message": error_message
    }

    # Log the error here if needed
    print(f"Reconstruction Error: {error_type} - {error_message}")

    return error_response
