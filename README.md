# 🏡 Real Estate Investment Advisor

An interactive real estate analytics and machine learning application built with Python and Streamlit. The project combines exploratory data analysis, interactive visualizations, property investment classification, and five-year property price prediction.

## 📌 Project Overview

The Real Estate Investment Advisor helps users explore property data, understand property characteristics, analyze locality-level patterns, and generate machine learning predictions using saved models.

The application provides a unified interface for data exploration and prediction, while MLflow supports experiment tracking and model management in the model-development workflow.

## ✨ Key Features

### 📊 Exploratory Data Analysis (EDA)
- Dataset overview and preview
- Descriptive statistics for numerical features
- Missing-value analysis
- Duplicate record detection
- Numerical distributions using histograms
- Outlier exploration using box plots
- Correlation heatmaps
- Interactive scatter plots and scatter matrices
- Categorical analysis using bar and pie charts
- Average metric comparisons across categories
- Year-based trend analysis when suitable data is available
- Locality-level property age analysis

### 🤖 Machine Learning Predictions
- **Classification:** Predicts the investment-suitability category represented by `Good_Investment`
- **Regression:** Estimates the five-year property price target represented by `Future_Price_5Y`
- Select from available saved classification and regression models
- Enter property details through interactive input forms
- View prediction results
- View class probabilities when supported by the selected classifier
- Inspect the feature vector passed to the model

### 🗄️ Data and Model Management
- SQLite property database integration
- CSV upload for EDA when a supported database is unavailable
- Loading saved models using Joblib or Pickle
- Loading model feature lists, scalers, and metadata
- MLflow experiment tracking in the model-development workflow

## 🧰 Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application and data science logic |
| Streamlit | Interactive web application |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Plotly Express | Interactive visualizations |
| SQLite | Property data storage |
| Scikit-learn | Machine learning and preprocessing |
| XGBoost | Regression model development |
| Joblib / Pickle | Loading saved model artifacts |
| JSON | Application metadata |
| MLflow | Experiment tracking and model registry |

## 📁 Project Structure

The following is the expected structure for the application components shown in the code. Your notebooks and additional project files can remain alongside these files.

```text
Real Estate Investment Advisor/
│
├── app.py
├── real_estate.db
├── mlflow.db
│
├── saved_models/
│   ├── app_metadata.json
│   │
│   ├── classification/
│   │   ├── feature_columns.pkl
│   │   ├── numeric_features.pkl
│   │   ├── scaler.pkl
│   │   ├── logistic_regression.pkl
│   │   ├── gradient_boosting.pkl
│   │   ├── decision_tree.pkl
│   │   ├── random_forest.pkl
│   │   └── xgboost.pkl
│   │
│   └── regression/
│       ├── feature_columns.pkl
│       ├── numeric_features.pkl
│       ├── scaler.pkl
│       ├── linear_regression.pkl
│       ├── decision_tree.pkl
│       ├── gradient_boosting.pkl
│       ├── random_forest.pkl
│       └── xgboost.pkl
│
├── real_estate.ipynb
├── classification_model.ipynb
├── regression_model.ipynb
├── mlflow.ipynb
├── project_report.md
├── requirements.txt
└── README.md
```


## ⚙️ Installation and Setup

### 1. Install Dependencies

If the project contains a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

Otherwise, install the main application dependencies:

```bash
pip install streamlit pandas numpy plotly scikit-learn xgboost joblib mlflow
```

### 2. Check Project Files

Before starting the application, confirm that:

- `app.py` is in the project folder.
- A supported property database or a CSV dataset is available.
- `saved_models/app_metadata.json` exists.
- The relevant saved models and feature artifacts are present in the expected folders.

### 3. Run the Application

```bash
python -m streamlit run app.py
```

Streamlit will provide a local URL, typically:

```text
http://localhost:8501
```

Open the displayed URL in your browser.

## 🖥️ Application Sections

### 1. Introduction
Provides an overview of the project, dataset row count, number of saved models, and project file status.

### 2. EDA Visualization
Allows users to select a database table or upload a CSV file and explore the data through statistics, quality checks, and interactive charts.

### 3. Prediction
Allows users to choose classification or regression, select an available saved model, enter property details, and view the resulting prediction.

## 🧪 Machine Learning Workflow

The model-development workflow follows these general steps:

1. Load and inspect the real estate dataset.
2. Perform data cleaning and exploratory analysis.
3. Prepare input features and target variables.
4. Encode categorical features and scale numerical features where required.
5. Train classification and regression models.
6. Evaluate models using suitable metrics.
7. Save trained models and preprocessing artifacts.
8. Track experiments and manage model versions with MLflow.
9. Load compatible model artifacts in the Streamlit application.
10. Generate predictions from user-provided property details.

## 📈 Evaluation

Classification models can be evaluated using metrics such as:

- Accuracy
- Precision
- Recall
- F1-score
- ROC_AUC

Regression models can be evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Squared Error (MSE)
- R-squared (\(R^2\))

Actual evaluation values should be reported from the project's saved metrics or test results rather than assumed.

## ⚠️ Limitations

- Prediction quality depends on the dataset, trained model, and input feature preparation.
- The targets `Good_Investment` and `Future_Price_5Y` were generated using dataset-based formulas. Strong evaluation scores may therefore reflect those formulas and do not establish real-world predictive performance.
- Future-price estimates are not guaranteed market prices.
- Model predictions require compatible feature columns and the correct preprocessing artifacts.
- The application is intended for educational and analytical use, not as a substitute for professional financial or real estate advice.

## 🔮 Future Enhancements

- Validate predictions against real property transaction data.
- Incorporate historical market prices and location-specific trends.
- Add rental yield and property expense analysis.
- Improve handling of missing and previously unseen categorical values.
- Add more location-based comparisons and investment filters.
- Monitor model performance using new market data.

## 🎯 Learning Outcomes

This project demonstrates practical experience with:

- Python-based data analysis
- Exploratory Data Analysis
- Interactive data visualization
- Classification and regression
- Feature engineering and preprocessing
- Model evaluation
- MLflow experiment tracking
- SQLite database integration
- Streamlit application development

## 📚 References

- [Python Documentation](https://docs.python.org/3/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [NumPy Documentation](https://numpy.org/doc/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Python Documentation](https://plotly.com/python/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [MLflow Documentation](https://mlflow.org/docs/latest/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)