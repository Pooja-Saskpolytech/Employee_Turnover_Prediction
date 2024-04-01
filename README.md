
# Employee Turnover Prediction

This project implements a web application for predicting employee turnover using a machine learning model. The application allows users to upload a CSV file containing employee data, visualize the importance of different features, and obtain predictions on whether an employee is likely to leave or stay.

## Features

- Upload CSV file: Users can upload a CSV file containing employee data for prediction.
- Input Features: Users can input various features such as department, promotion status, satisfaction level, etc., to obtain predictions.
- Predictions: The model predicts whether an employee is likely to leave or stay based on the input features.
- Actionable Recommendations: Based on the prediction and feature importance, actionable recommendations are provided to mitigate turnover risk.
- Visualizations: Feature importance scores are visualized using bar charts.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your_username/employee-turnover-prediction.git
   ```

2. Navigate to the project directory:

   ```bash
   cd employee-turnover-prediction
   ```


## Usage

1. Start the web application:

   ```bash
   streamlit run app.py
   ```

2. Access the application in your web browser at [http://localhost:8501](http://localhost:8501).

3. Upload a CSV file containing employee data or input individual feature values.

4. Obtain predictions and actionable recommendations based on the input.

## Project Structure

The project structure is organized as follows:

- **app.py**: Main Streamlit application script containing the user interface and logic.
- **model.py**: Script containing functions for model training, prediction, and preprocessing.
- **data.csv**: Example CSV file containing employee data (can be replaced with your own dataset).
- **xgboost_model.pkl**: Pretrained XGBoost machine learning model.
- **feature_names.pkl**: Pickle file containing the names of features used by the model.
- **requirements.txt**: File containing Python dependencies required for the project.
- **README.md**: Project documentation.

## Requirements

- Python 3.6+
- Streamlit
- Pandas
- Scikit-learn
- XGBoost
- Matplotlib

