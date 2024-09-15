"""
Test suite for the 3D Reconstruction Module.
"""

import os
import sys

# Add the parent directory to the Python path to allow importing the reconstruction module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# You can add any common test utilities or fixtures here if needed
def setup_test_data():
    """
    Set up common test data for reconstruction tests.
    """
    # Example implementation:
    import numpy as np
    return np.random.rand(1000, 3)  # Random 3D point cloud

# If you have any cleanup functions, you can define them here as well
def teardown_test_data():
    """
    Clean up any resources created for tests.
    """
    pass
