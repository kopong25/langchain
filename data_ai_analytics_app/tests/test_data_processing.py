import unittest
import pandas as pd
from data_processing.data_processing import DataProcessor

class TestDataProcessor(unittest.TestCase):
    def setUp(self):
        # Sample data for testing
        self.sample_data = pd.DataFrame({
            'A': [1, 2, 2, None],
            'B': ['x', 'y', 'y', 'z'],
            'target': [0, 1, 1, 0]
        })
        self.processor = DataProcessor('')
        self.processor.data = self.sample_data

    def test_clean_data(self):
        self.processor.clean_data()
        # Check if duplicates are removed
        self.assertEqual(len(self.processor.data), 3)
        # Check if missing values are handled
        self.assertFalse(self.processor.data.isnull().values.any())

    def test_transform_data(self):
        self.processor.transform_data()
        # Check if column names are lowercase
        self.assertTrue(all(col.islower() for col in self.processor.data.columns))

if __name__ == '__main__':
    unittest.main()

