This project demonstrates the application of machine learning and deep learning techniques to predict house prices using a dataset of housing features. The goal is to build and evaluate models that can accurately estimate house prices based on various attributes.
Overview:
The project includes the following steps:
1.Data Loading and Exploration
2.Feature Engineering and Preprocessing
3.Model Building and Evaluation
4.Residuals Analysis

Data LoadingThe dataset is loaded from Google Drive using Google Colab:from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
df = pd.read_csv('/content/drive/MyDrive/HousePricePrediction (2).csv')

Exploratory Data Analysis (EDA)Distribution of Sale Prices:
i am visualising the distribution of house sale prices using a histogram with a Kernel Density Estimate (KDE) to understand the price distribution:
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.histplot(df['SalePrice'], bins=30, kde=True)
plt.title('Distribution of Sale Prices')
plt.xlabel('SalePrice')
plt.ylabel('Frequency')
plt.show()

Feature Correlations:
A heatmap is created to visualize the correlations between numeric features in the dataset:
df_numeric = df.select_dtypes(include=['number'])
plt.figure(figsize=(12, 8))
sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlations')
plt.show()

Feature Engineering:
A new feature, HouseAge, is added based on the difference between the year of remodeling and the year the house was built:
df['HouseAge'] = df['YearRemodAdd'] - df['YearBuilt']

Handling Missing Values:
Missing values are handled using forward fill. This is a basic imputation strategy:
df.fillna(method='ffill', inplace=True)

Data Preprocessing:
Separating Features and Target:
i am defining numeric and categorical features and prepare the feature matrix X and target vector y:
numeric_features = ['LotArea', 'OverallCond', 'TotalBsmtSF', 'HouseAge']
categorical_features = ['MSZoning', 'LotConfig', 'BldgType', 'Exterior1st']

X = df.drop(columns=['Id', 'SalePrice', 'YearBuilt', 'YearRemodAdd'])
y = df['SalePrice']

Preprocessing Pipeline
I will set up a pipeline to standardize numeric features and one-hot encode categorical features:
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ])

Model Building and Evaluation:
Random Forest Regressor
i will use a Random Forest Regressor and optimize hyperparameters with RandomizedSearchCV:
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [10, 20],
}

rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(random_state=42))
])

rf_grid = RandomizedSearchCV(rf_pipeline, param_distributions=param_grid, n_iter=5, cv=3, n_jobs=-1, verbose=1, random_state=42)
rf_grid.fit(X_train, y_train)

rf_best = rf_grid.best_estimator_
rf_pred = rf_best.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_pred)
print(f"Optimized Random Forest MSE: {rf_mse}")

Deep Learning Model:
A Neural Network model is built using TensorFlow with layers for Dense, BatchNormalization, and Dropout:
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

tf_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_processed.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1)
])

tf_model.compile(optimizer='adam', loss='mse')

history = tf_model.fit(X_train_processed, y_train, epochs=50, validation_split=0.2, verbose=1)
tf_pred = tf_model.predict(X_test_processed)
tf_mse = mean_squared_error(y_test, tf_pred)
print(f"TensorFlow Neural Network MSE: {tf_mse}")

Residuals Analysis:
Residual plots are created to compare the performance of the Random Forest and TensorFlow models:
plt.figure(figsize=(6, 6))
plt.scatter(y_test, rf_pred - y_test, label='Random Forest Residuals')
plt.scatter(y_test, tf_pred.flatten() - y_test, label='TensorFlow Residuals', alpha=0.6)
plt.hlines(0, min(y_test), max(y_test), colors='red')
plt.title('Residuals of Predictions')
plt.xlabel('True Values')
plt.ylabel('Residuals')
plt.legend()
plt.show()

Conclusion:
This project demonstrates the use of both traditional machine learning and deep learning techniques for predicting house prices. By comparing model performance through residuals analysis, i can assess the effectiveness of each approach.

Requirements:
1.Python 
2.TensorFlow
3.scikit-learn
4.pandas
5.numpy
6.seaborn
7.matplotlib

Running the Code
Loading and Preprocessing Data: Update the dataset path and preprocess the data.
Training the Models: Fit and evaluate Random Forest and Neural Network models.
Analysing the Results: Reviewing the residuals to compare model performance.
