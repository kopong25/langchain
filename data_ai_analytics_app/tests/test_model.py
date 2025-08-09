import unittest
import pandas as pd
from machine_learning.model import MachineLearningModel

class TestMachineLearningModel(unittest.TestCase):
    def setUp(self):
        # Sample data for testing
        self.sample_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [5, 4, 3, 2, 1],
            'target': [1, 0, 1, 0, 1]
        })
        self.model = MachineLearningModel(self.sample_data)
        self.model.prepare_data()

    def test_train_model(self):
        self.model.train_model()
        # Check if the model has been trained
        self.assertIsNotNone(self.model.model.coef_)

    def test_evaluate_model(self):
        self.model.train_model()
        mse = self.model.evaluate_model()
        # Check if the mean squared error is a float
        self.assertIsInstance(mse, float)

    def test_predict(self):
        self.model.train_model()
        new_data = pd.DataFrame({
            'feature1': [6, 7],
            'feature2': [0, -1]
        })
        predictions = self.model.predict(new_data)
        # Check if predictions are made
        self.assertEqual(len(predictions), len(new_data))

if __name__ == '__main__':
    unittest.main()

