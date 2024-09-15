import unittest
import numpy as np
import tempfile
import os
from reconstruction.point_cloud_processor import PointCloudProcessor
from reconstruction.utils import InputProcessingError


class TestPointCloudProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = PointCloudProcessor()
        self.test_data = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])

    def test_process_numpy_array(self):
        result = self.processor.process(self.test_data)
        np.testing.assert_array_equal(result, self.test_data)

    def test_process_list(self):
        test_list = self.test_data.tolist()
        result = self.processor.process(test_list)
        np.testing.assert_array_equal(result, self.test_data)

    def test_process_dict(self):
        test_dict = {'points': self.test_data.tolist()}
        result = self.processor.process(test_dict)
        np.testing.assert_array_equal(result, self.test_data)

    def test_process_file(self):
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            np.savetxt(temp_file, self.test_data, delimiter=',')

        try:
            result = self.processor.process(temp_file.name)
            np.testing.assert_array_almost_equal(result, self.test_data, decimal=5)
        finally:
            os.unlink(temp_file.name)

    def test_invalid_input_type(self):
        with self.assertRaises(InputProcessingError):
            self.processor.process("invalid input")

    def test_invalid_array_shape(self):
        invalid_data = np.array([[1, 2], [3, 4]])
        with self.assertRaises(InputProcessingError):
            self.processor.process(invalid_data)

    def test_empty_input(self):
        with self.assertRaises(InputProcessingError):
            self.processor.process(np.array([]))

    def test_process_with_color_data(self):
        color_data = np.array([
            [0, 0, 0, 255, 0, 0],
            [1, 0, 0, 0, 255, 0],
            [0, 1, 0, 0, 0, 255],
            [0, 0, 1, 255, 255, 255]
        ])
        result = self.processor.process(color_data)
        np.testing.assert_array_equal(result, color_data)


if __name__ == '__main__':
    unittest.main()
