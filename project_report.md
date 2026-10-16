# Real Estate Investment Advisor
## Project Documentation

### 1. Project Overview

The Real Estate Investment Advisor is a data science and machine learning project developed to explore real estate data and support preliminary property investment analysis.

The project combines Exploratory Data Analysis (EDA), interactive visualizations, machine learning predictions, SQLite database access, and MLflow-based model development and tracking.

The Streamlit application provides three main sections: Introduction, EDA Visualization, and Prediction. Users can explore property data, investigate patterns and relationships, and obtain model predictions using the available trained models.

The project is intended for analytical and educational purposes. Its predictions should not be treated as guaranteed investment outcomes.

### 2. Project Objectives

The main objectives are:

- Explore and analyze property-related data.
- Understand numerical and categorical property characteristics.
- Identify missing values, duplicate records, and potential outliers.
- Visualize property distributions, correlations, and category-level patterns.
- Analyze property characteristics across localities.
- Develop classification models for investment suitability.
- Develop regression models for five-year property price estimation.
- Organize and track machine learning experiments using MLflow.
- Build an interactive application using Streamlit.

### 3. Methodology

#### 3.1 Data Loading and Database Management

The application searches for a supported property database file, such as `real_estate.db`, `realestate.db`, `real_estate.sqlite`, or `housing.db`.

SQLite is used to retrieve available table names and load the selected table into a Pandas DataFrame. The application opens the database in read-only mode for these operations.

If no supported database is available, the EDA section allows users to upload a CSV file for analysis.

The `ID` column is excluded from analysis where it serves only as a record identifier and does not provide meaningful analytical information.

#### 3.2 Data Preparation

The dataset is inspected using its dimensions, data types, descriptive statistics, and data-quality summaries.

The application checks:

- Total rows and columns.
- Duplicate records.
- Missing values.
- Numerical and categorical columns.
- Numerical distributions and potential outliers.

For model prediction, the application uses saved feature-column information, metadata, and preprocessing artifacts when available. These artifacts help construct model inputs in the format expected by the trained model.

#### 3.3 Exploratory Data Analysis (EDA)

The EDA section provides interactive tools for examining the property dataset.

The analysis includes:

- **Dataset overview:** Displays the number of rows, columns, duplicate records, and missing cells.
- **Dataset preview:** Shows sample records and descriptive statistics for numerical columns.
- **Missing-value analysis:** Identifies columns containing missing values.
- **Numerical distributions:** Uses histograms to examine numerical feature distributions.
- **Outlier analysis:** Uses box plots to inspect the spread and potential outliers of numerical features.
- **Correlation analysis:** Uses a heatmap to investigate relationships between numerical variables.
- **Scatter plots:** Examines relationships between selected numerical features, with optional category-based coloring.
- **Scatter matrix:** Compares multiple numerical features simultaneously.
- **Categorical analysis:** Displays category counts and proportions.
- **Average metric by category:** Compares a selected numerical metric across categories.
- **Year-based trends:** When a suitable numerical year column is available, displays changes in a selected metric over time.

For performance, the application uses a sample of up to 5,000 rows for selected visualizations and a smaller sample for the scatter matrix.

The project also includes locality-level analysis, such as calculating the median age of properties in each locality. Median values help summarize typical property age while reducing the influence of extreme observations.

#### 3.4 Classification Models

The classification component predicts whether a property belongs to the investment-suitability category represented by the `Good_Investment` target.

The model development workflow includes classification algorithms such as:

- Logistic Regression Classifier
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier
- XGBoost Classifier

The Prediction section loads the available model files from `saved_models/Classification/`. The user selects a model and enters property details through an interactive form.

The application constructs a feature row using property attributes, categorical features, locality-frequency information, amenity indicators, and other available metadata. It then applies the saved numerical scaling process where required and sends the prepared features to the selected model.

For models that support probability estimates, the application can display predicted class probabilities.

#### 3.5 Regression Models

The regression component estimates the `Future_Price_5Y` target, representing the dataset's five-year future property price.

The model development workflow includes algorithms such as:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

The Prediction section loads available regression model files from `saved_models/Regression/`.

After the user enters property details, the application constructs the required feature row, applies the available preprocessing artifacts, and generates a numerical prediction.

The interface displays the predicted price after five years in lakhs, the current property price, and the estimated percentage difference between the two values.

These outputs represent model estimates based on the training data and should not be interpreted as guaranteed future market prices.

#### 3.6 MLflow Experiment Tracking

MLflow is used in the broader machine learning workflow to track experiments, record model metrics and parameters, and manage registered model versions.

The project maintains separate classification and regression model-development workflows. MLflow tracking storage is separate from the SQLite property database used by the application.

The supplied Streamlit app loads saved model artifacts and associated metadata from the project directory. It does not directly display the MLflow interface.

#### 3.7 Streamlit Application Development

The application is developed using Streamlit and provides three main navigation sections.

**Introduction**
- Presents the project's purpose and key capabilities.
- Displays dataset row count and the number of saved models.
- Shows the detected property database, selected table, and metadata status.

**EDA Visualization**
- Allows selection of a database table.
- Supports CSV upload when a property database is unavailable.
- Provides interactive charts, descriptive statistics, and data-quality summaries.

**Prediction**
- Allows selection between classification and regression.
- Loads available model files and preprocessing artifacts.
- Collects property details through interactive form controls.
- Displays prediction results and, when supported, classification probabilities.
- Provides an expandable view of the feature vector sent to the model.

### 4. Key Findings

The project provides a framework for examining property data and exploring machine learning predictions.

1. **Property characteristics:** Features such as size, current price, number of bedrooms, construction year, floor information, nearby facilities, and locality can be investigated to understand differences between properties.

2. **Locality-level patterns:** Median property age by locality helps compare the typical age of properties across different areas.

3. **Data quality:** Missing-value summaries, duplicate counts, descriptive statistics, and box plots help identify potential data-quality issues.

4. **Feature relationships:** Correlation heatmaps and scatter plots help investigate relationships between numerical attributes. Correlation does not establish causation.

5. **Investment classification:** Classification models can produce predictions for the `Good_Investment` target when the required model and compatible features are available.

6. **Future-price estimation:** Regression models can estimate the `Future_Price_5Y` target using the patterns learned from the training dataset.

7. **Model management:** MLflow supports organized experiment tracking and model version management as part of the broader model-development workflow.

8. **Interactive analysis:** The Streamlit interface combines data exploration, visual analysis, and prediction functionality in a single application.

**Interpretation limitation:** The project's targets, including `Good_Investment` and `Future_Price_5Y`, were generated using dataset-based formulas. Therefore, high model evaluation scores may partly reflect those formulas rather than the ability to predict real-world investment performance. Validation against reliable real-world property data is necessary before relying on the predictions for actual investment decisions.

### 5. Use Cases

#### 5.1 Property Data Exploration

Users can inspect property records, review descriptive statistics, and explore distributions of numerical and categorical attributes.

#### 5.2 Locality Comparison

Locality-level summaries, including median property age, can help identify differences between areas and guide further investigation.

#### 5.3 Preliminary Investment Screening

The classification model demonstrates how property attributes can be used to predict an investment-suitability category defined by the dataset.

The result can support preliminary screening but should not replace detailed financial analysis.

#### 5.4 Future Property Price Estimation

The regression model demonstrates how property characteristics can be used to estimate the dataset's five-year price target.

The estimate requires independent validation before it can be considered reliable for real investment planning.

#### 5.5 Data Quality Assessment

Missing-value summaries, duplicate checks, distributions, and box plots help users understand the condition of the dataset before further analysis.

#### 5.6 Machine Learning Demonstration

The project demonstrates an end-to-end workflow involving data loading, EDA, feature preparation, model prediction, and experiment tracking.

#### 5.7 Interactive Project Presentation

The Streamlit application provides a convenient interface for demonstrating real estate analytics and model predictions to reviewers, learners, and other stakeholders.

### 6. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Streamlit | Interactive web application |
| Pandas | Data loading and manipulation |
| NumPy | Numerical operations |
| Plotly Express | Interactive data visualizations |
| SQLite | Property database storage |
| Scikit-learn | Machine learning algorithms and preprocessing |
| XGBoost | Regression model development |
| Joblib and Pickle | Loading saved models and preprocessing artifacts |
| MLflow | Experiment tracking and model registry |
| JSON | Application metadata storage |

Matplotlib and Seaborn may also be used in the project's separate EDA notebooks where applicable.

### 7. Limitations and Future Scope

The project can be extended in the following ways:

- Validate model predictions using real transaction prices and subsequent market outcomes.
- Incorporate location-specific market trends and historical price data.
- Evaluate model performance on independent datasets.
- Save and reuse complete preprocessing pipelines to improve consistency between training and prediction.
- Improve handling of missing values and previously unseen categorical values.
- Add rental-yield calculations and estimates of maintenance costs, taxes, and transaction expenses.
- Include additional property and neighborhood indicators where reliable data is available.
- Monitor prediction performance as the underlying property market changes.

The current application is a project demonstration and should not be considered a source of guaranteed investment returns or professional financial advice.

### 8. Conclusion

The Real Estate Investment Advisor demonstrates how data science and machine learning techniques can be applied to real estate analytics.

The project combines property data loading, exploratory data analysis, interactive visualization, classification, regression, saved model artifacts, and MLflow-based experiment management. Its Streamlit interface makes these capabilities accessible through a unified application.

The project provides a foundation for property exploration and preliminary investment analysis. Further testing and validation using reliable real-world market data would be required before its predictions could be used for practical investment decisions.

### 9. References

- Python: https://docs.python.org/3/
- Pandas: https://pandas.pydata.org/docs/
- NumPy: https://numpy.org/doc/
- Streamlit: https://docs.streamlit.io/
- Plotly: https://plotly.com/python/
- Scikit-learn: https://scikit-learn.org/stable/
- MLflow: https://mlflow.org/docs/latest/
- XGBoost: https://xgboost.readthedocs.io/