import pandas as pd

class DataProcessor:
    def __init__(self, data_source):
        self.data_source = data_source
        self.data = None

    def load_data(self):
        """Load data from the source."""
        self.data = pd.read_csv(self.data_source)

    def clean_data(self):
        """Clean the data by removing duplicates and handling missing values."""
        self.data.drop_duplicates(inplace=True)
        self.data.fillna(method='ffill', inplace=True)

    def transform_data(self):
        """Transform the data for analysis."""
        # Example transformation: convert all column names to lowercase
        self.data.columns = [col.lower() for col in self.data.columns]

