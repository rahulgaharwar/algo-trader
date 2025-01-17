
## Overview
This project demonstrates the development of an algorithmic trading strategy using historical stock market data. It includes tasks such as data preprocessing, feature engineering, predictive modeling, trading strategy development, backtesting, and optimization.

## File Structure

1. **`task_1.ipynb`**
   - **Objective**: Data preprocessing and feature engineering.
   - **Key Steps**:
     - Handled missing values and outliers.
     - Created technical indicators: Moving Averages (MA20, MA50), Relative Strength Index (RSI), Moving Average Convergence Divergence (MACD), and Signal Line.
     - Calculated daily returns.
   - **Output**:
     - Enhanced `stock_data.csv` with additional features for use in modeling.

2. **`task_2.ipynb`**
   - **Objective**: Predictive model development.
   - **Key Steps**:
     - Built a Random Forest model to predict the next day's price movement (up or down).
     - Split the data into training and testing sets.
     - Evaluated the model using accuracy, precision, recall, F1-score, confusion matrix, and ROC-AUC curve.
   - **Output**:
     - Trained model saved as `trained_model.pkl`.

3. **`task_3.ipynb`**
   - **Objective**: Develop and backtest a trading strategy.
   - **Key Steps**:
     - Implemented a trading strategy based on model predictions.
     - Simulated buy/sell orders with a transaction cost of 0.1%.
     - Backtested the strategy and compared it with a buy-and-hold approach.
     - Calculated performance metrics: cumulative returns, maximum drawdown, and Sharpe ratio.
   - **Output**:
     - Backtest results saved as `strategy_report.csv`.

4. **`task_4.ipynb`**
   - **Objective**: Optimization and refinement.
   - **Key Steps**:
     - Performed parameter optimization for the Random Forest model using Grid Search.
     - Refined the trading strategy to include risk management (stop-loss and take-profit mechanisms).
     - Evaluated the optimized strategy's performance.
   - **Output**:
     - Optimized model saved as `optimized_model.pkl`.

5. **`stock_data.csv`**
   - **Description**: Historical stock market data used for modeling and backtesting.
   - **Contents**:
     - Columns: Date, Close, High, Low, Open, Volume, MA20, MA50, RSI, MACD, Signal Line, Returns.

6. **`trained_model.pkl`**
   - **Description**: Trained Random Forest model for predicting price movements.

7. **`strategy_report.csv`**
   - **Description**: Backtest results of the trading strategy, including performance metrics.

8. **`optimized_model.pkl`**
   - **Description**: Optimized Random Forest model after parameter tuning.

## Instructions

1. **Task 1: Preprocessing and Feature Engineering**
   - Open `task_1.ipynb` in Google Colab.
   - Run all cells to preprocess the data and engineer features.

2. **Task 2: Predictive Modeling**
   - Open `task_2.ipynb` in Google Colab.
   - Train the Random Forest model.
   - Save the trained model as `trained_model.pkl`.

3. **Task 3: Backtesting**
   - Open `task_3.ipynb` in Google Colab.
   - Load the trained model and stock data.
   - Simulate and backtest the trading strategy.
   - Save the results as `strategy_report.csv`.

4. **Task 4: Optimization**
   - Open `task_4.ipynb` in Google Colab.
   - Perform hyperparameter optimization and refine the trading strategy.
   - Save the optimized model as `optimized_model.pkl`.

## Deliverables
- **Models**: `trained_model.pkl`, `optimized_model.pkl`.
- **Data**: `stock_data.csv`.
- **Reports**: `strategy_report.csv`.

## Insights
- Feature engineering improves model performance by providing additional signals.
- Risk management (e.g., stop-loss, take-profit) significantly impacts strategy profitability and reduces drawdown.
- Parameter optimization refines predictive accuracy and enhances trading strategy results.

## Tools Used
- **Google Colab**: For running Jupyter notebooks.
- **Pandas**: Data manipulation and preprocessing.
- **Scikit-learn**: Machine learning model development.
- **Matplotlib**: Data visualization.
- **Pickle**: Model saving and loading.

For further improvements, consider incorporating additional technical indicators, experimenting with other machine learning models, or adding multi-asset support for diversification.

