import matplotlib.pyplot as plt

class DataVisualizer:
    def __init__(self, data):
        self.data = data

    def plot_histogram(self, column):
        """Plot a histogram of a specified column."""
        plt.figure(figsize=(10, 6))
        plt.hist(self.data[column], bins=30, alpha=0.7, color='blue')
        plt.title(f'Histogram of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()

    def plot_scatter(self, x_column, y_column):
        """Plot a scatter plot of two specified columns."""
        plt.figure(figsize=(10, 6))
        plt.scatter(self.data[x_column], self.data[y_column], alpha=0.7, color='red')
        plt.title(f'Scatter Plot of {x_column} vs {y_column}')
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.grid(True)
        plt.show()

