from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

class MachineLearningModel:
    def __init__(self, data):
        self.data = data
        self.model = LinearRegression()
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

    def prepare_data(self):
        """Prepare data for training and testing."""
        X = self.data.drop('target', axis=1)
        y = self.data['target']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def train_model(self):
        """Train the machine learning model."""
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        """Evaluate the model's performance."""
        predictions = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, predictions)
        return mse

    def predict(self, new_data):
        """Make predictions with the trained model."""
        return self.model.predict(new_data)

