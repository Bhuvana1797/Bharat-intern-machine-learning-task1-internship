from google.colab import drive
drive.mount('/content/drive')
import pandas as pd
df=pd.read_csv('/content/drive/MyDrive/HousePricePrediction (2).csv')
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
# Loading the dataset
df = pd.read_csv('/content/drive/MyDrive/HousePricePrediction (2).csv')
# Exploratory Data Analysis (EDA)
plt.figure(figsize=(10, 6))
sns.histplot(df['SalePrice'], bins=30, kde=True)
plt.title('Distribution of Sale Prices')
plt.xlabel('SalePrice')
plt.ylabel('Frequency')
plt.show()
# Handle non-numeric columns before calculating correlations
df_numeric = df.select_dtypes(include=['number'])  # Select only numeric columns
plt.figure(figsize=(12, 8))
sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlations')
plt.show()
# Feature Engineering: Adding new features based on domain knowledge
df['HouseAge'] = df['YearRemodAdd'] - df['YearBuilt']

# Handling missing values (if any) - basic imputation strategy
# Here, I'm using simple forward fill or mean imputation as an initial approach
df.fillna(method='ffill', inplace=True)

# Separating features and target variable
numeric_features = ['LotArea', 'OverallCond', 'TotalBsmtSF', 'HouseAge']
categorical_features = ['MSZoning', 'LotConfig', 'BldgType', 'Exterior1st']

# Preprocessing pipeline: Standardization and OneHotEncoding
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

# Define feature matrix X and target vector y
X = df.drop(columns=['Id', 'SalePrice', 'YearBuilt', 'YearRemodAdd'])
y = df['SalePrice']
# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Simplify param grid
param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [10, 20],
}

# Updated Preprocessor with 'handle_unknown' set to 'ignore'
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ])

# Pipeline for Random Forest Regressor
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(random_state=42))
])

# Randomized search
rf_grid = RandomizedSearchCV(rf_pipeline, param_distributions=param_grid, n_iter=5, cv=3, n_jobs=-1, verbose=1, random_state=42)
rf_grid.fit(X_train, y_train)

# Best model evaluation
rf_best = rf_grid.best_estimator_
rf_pred = rf_best.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_pred)
print(f"Optimized Random Forest MSE: {rf_mse}")
# Preprocess the training and testing data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Deep Learning Model using TensorFlow (Sequential API)
tf_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_processed.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1)
])

# Compile model
tf_model.compile(optimizer='adam', loss='mse')

# Model training with validation split
history = tf_model.fit(X_train_processed, y_train, epochs=50, validation_split=0.2, verbose=1)  # Use preprocessed data

# Evaluate TensorFlow model
tf_pred = tf_model.predict(X_test_processed)  # Use preprocessed data
tf_mse = mean_squared_error(y_test, tf_pred)
print(f"TensorFlow Neural Network MSE: {tf_mse}")
# Residuals Analysis: Comparing model performance
plt.figure(figsize=(6, 6))
plt.scatter(y_test, rf_pred - y_test, label='Random Forest Residuals')
plt.scatter(y_test, tf_pred.flatten() - y_test, label='TensorFlow Residuals', alpha=0.6) # Flatten tf_pred to 1D
plt.hlines(0, min(y_test), max(y_test), colors='red')
plt.title('Residuals of Predictions')
plt.xlabel('True Values')
plt.ylabel('Residuals')
plt.legend()
plt.show()


