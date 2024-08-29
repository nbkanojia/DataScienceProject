
# DataScienceProject

## Developing Data Science Techniques for Technical Analysis in Stock Market Prediction

## Overview
This project focuses on predicting stock prices using machine learning to help investors make better decisions, specifically using Microsoft stock data. The stock market is influenced by many factors like economic data, company earnings, and global events, making accurate predictions essential for good investment strategies. Traditional methods, like technical analysis, use tools such as Weighted Moving Average (WMA), Volume Price Trend (VPT), and Exponential Moving Average (EMA) of VPT to find trends in past price and volume data. With the rise of machine learning, these predictions have become more accurate. This project develops and tests models like Logistic Regression, Gradient Boosting, and LSTM networks, using these technical indicators to predict when to buy, sell, or hold Microsoft stock, aiming to maximize profits compared to traditional methods.

## Objectives
The main goal of this project is to increase profits by using machine learning techniques. To achieve this, the project focuses on developing and refining models like Logistic Regression, Gradient Boosting, and LSTM networks to predict buy, sell, or hold signals for Microsoft stock. It also compares these models' effectiveness against traditional methods, such as WMA, VPT, and EMA of VPT. Additionally, the project aims to identify the strengths and weaknesses of these machine learning models in stock market prediction by evaluating their Precision, Recall, F1 Score, and overall profitability.

## Directory Structure
The project directory is organized as follows:

```
-root
|-- data
|   |-- AAPL_2010-2020_daily_data.csv
|   |-- TSLA_2010-2020_daily_data.csv
|   |-- BABA_2010-2020_daily_data.csv
|-- code
|   |-- yfinancetool.py
|   |-- stockmarketML.ipynb
|   |-- model_helper.py
|   |-- techinal_indicato.py
|-- README.md
```

- **.csv files:** Named according to the stock ticker symbol and the date range, e.g., AAPL_2010-2020_daily_data.csv.
- **yfinancetool.py:** Script for downloading data from Yahoo Finance using the yfinance library.
- **stockmarketML.ipynb:** Main notebook containing logic for pre-processing, training models, and displaying output.
- **model_helper.py:** Contains machine learning code for training and testing different models, and generating evaluation metrics.
- **techinal_indicato.py:** Logic for pre-processing and calculation of technical indicators.
- **README.md:** Provides information about the project, including how to use the code and data, and other relevant details.

## Dependencies and Installation
This project depends on Python 3.12.4 and the following modules. You can install all dependencies using the following pip commands:

```bash
pip install tensorflow[and-cuda]
pip install pandas
pip install yfinance
pip install scikit-learn  
pip install plotly
pip install imblearn
pip install seaborn
pip install keras-tuner
pip install --upgrade nbformat
pip install scikit-learn tensorflow
pip install keras
pip install scikeras
pip install scikeras[tensorflow]
pip install scikeras tensorflow
```

## Data Source and Collection
The data is collected from Yahoo Finance using the yfinance Python library. It is publicly available and intended for research and educational purposes. The yfinance library is distributed under the Apache Software License. Users should refer to Yahoo's terms of use for details on their rights to use the actual data downloaded.

## Metadata
The dataset includes columns for the date, opening price, closing price, highest price, lowest price, adjusted closing price, and trading volume.

## How to Use

1. **Set Up the Environment:**
   - Install Python 3.12.4 or higher.
   - Install the required dependencies using the provided pip commands.

2. **Prepare the Data:**
   - Download necessary stock data using the `yfinancetool.py` script or use the provided CSV files in the `data` directory.

3. **Train a Model:**
   - Use one of the helper classes (e.g., `LogisticRegressionHelper`, `GradientBoostClassifierHelper`, `LongShortTermMemoryMLHelper`) to train a model on the stock data.
   - Example for Logistic Regression:
     ```python
     from code.model_helper import LogisticRegressionHelper
     lr_helper = LogisticRegressionHelper()
     features = ["WMA_pct_change", "VPT_EMA_Signal_Line_diff", "adj_close_pct_change"]
     target = "WMA_VPT_Signal"
     model, scaler = lr_helper.train_logistic_regression_model(training_data, features, target)
     ```

4. **Predict Signals:**
   - After training, predict buy, sell, or hold signals:
     ```python
     predicted_signals = lr_helper.predict_signals(model, scaler, test_data, features)
     ```

5. **Visualize Results:**
   - Use the `PlotHelper` class to create visualizations of the results:
     ```python
     from code.plots import PlotHelper
     plot_helper = PlotHelper()
     plot_helper.plot_profit_trend("Profit Trend", dates, profits)
     ```

6. **Evaluate Model Performance:**
   - Evaluate the performance of your model by plotting confusion matrices or comparing profits:
     ```python
     plot_helper.plot_confusion_matrix(y_test, y_pred)
     ```

7. **Save and Load Models:**
   - Save your trained models for future use, or load them as needed:
     ```python
     lr_helper.save_model(model, scaler, "logistic_regression_model.pkl")
     model, scaler = lr_helper.load_model("logistic_regression_model.pkl")
     ```

## Classes and Methods

This project includes several classes designed to support machine learning model development, evaluation, and visualization for stock market prediction. Below is a description of each class and its methods.

### 1. LogisticRegressionHelper Class
This class provides helper methods for training and predicting using a Logistic Regression model.

- **Methods:**
  - `train_logistic_regression_model(self, df_train, features, target)`: Trains a Logistic Regression model on the provided dataset and features. Returns the trained model and the scaler used for standardization.
  - `predict_signals(self, model, scaler, df, features)`: Predicts trading signals (buy, sell, neutral) using a trained Logistic Regression model. Returns a pandas Series of predicted signals.
  - `predict_signals_from_saved_model(self, df, features)`: Loads a saved Logistic Regression model and scaler, then predicts trading signals. Returns a pandas Series of predicted signals.

### 2. GradientBoostClassifierHelper Class
This class provides helper methods for training and predicting using a Gradient Boosting Classifier model.

- **Methods:**
  - `train_gradient_classifier_model(self, df_train, features, target)`: Trains a Gradient Boosting Classifier using GridSearchCV for hyperparameter tuning. Returns the best model and the scaler used for standardization.
  - `predict_signals(self, model, scaler, df, features)`: Predicts trading signals using a trained Gradient Boosting model. Returns a pandas Series of predicted signals.
  - `predict_signals_from_saved_model(self, df, features)`: Loads a saved Gradient Boosting model and scaler, then predicts trading signals. Returns a pandas Series of predicted signals.

### 3. LongShortTermMemoryMLHelper Class
This class provides helper methods for working with Long Short-Term Memory (LSTM) neural networks.

- **Methods:**
  - `create_train_dataset(self, data, labels, time_step=1)`: Creates a training dataset for LSTM with the specified time steps. Returns numpy arrays for features (X) and labels (y).
  - `create_test_dataset(self, data, time_step=1)`: Creates a test dataset for LSTM with the specified time steps. Returns a numpy array of features.
  - `train_lstm_model(self, df_train, features, target)`: Trains an LSTM model on the provided dataset. Returns the trained LSTM model and the scaler used for standardization.
  - `predict_signals(self, model, scaler, df, features)`: Predicts trading signals using a trained LSTM model. Returns a pandas Series of predicted signals.
  - `predict_signals_from_saved_model(self, df, features)`: Loads a saved LSTM model and scaler, then predicts trading signals. Returns a pandas Series of predicted signals.

### 4. PlotHelper Class
This class provides methods for various types of visualizations related to stock market data.

- **Methods:**
  - `plot_class_distribution(self, stock_df)`: Plots the distribution of classes (buy, sell, neutral) as a pie chart.
  - `plot_confusion_matrix(self, y_test, y_pred)`: Plots the confusion matrix for model predictions.
  - `plot_correlation_matrix(self, df)`: Plots the correlation matrix as a heatmap.
  - `plot_profit_trend(self, title, x, y)`: Plots the profit trend over time.
  - `plot_signals(self, title, df)`: Plots trading signals as a barplot.
  - `plot_profit_compare(self, title, profit_curr, profit_pct, hue_model)`: Plots a comparison of profits between different models.
  - `plot_feature_selection_score(self, columns, scores)`: Plots the F-scores for feature selection.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
