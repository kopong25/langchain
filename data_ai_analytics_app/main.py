from data_processing.data_processing import DataProcessor
from machine_learning.model import MachineLearningModel
from visualization.visualization import DataVisualizer

def main():
    # Initialize data processor and load data
    data_processor = DataProcessor('data.csv')
    data_processor.load_data()
    data_processor.clean_data()
    data_processor.transform_data()

    # Initialize machine learning model and prepare data
    ml_model = MachineLearningModel(data_processor.data)
    ml_model.prepare_data()
    ml_model.train_model()
    mse = ml_model.evaluate_model()
    print(f'Model Mean Squared Error: {mse}')

    # Initialize data visualizer and plot data
    visualizer = DataVisualizer(data_processor.data)
    visualizer.plot_histogram('feature_column')
    visualizer.plot_scatter('feature_column1', 'feature_column2')

if __name__ == "__main__":
    main()

