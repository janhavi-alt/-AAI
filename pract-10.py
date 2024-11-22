#pip install pandas numpy scikit-learn xgboost lightgbm matplotlib
#if above cmd isnt working use - pip install --index-url https://pypi.org/simple xgboost lightgbm matplotlib   

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt

# Load a sample dataset (For example, the Titanic dataset)
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
data = pd.read_csv(url)

# Basic data preprocessing (handling missing values, encoding categorical features)
data = data.drop(columns=['Name', 'Ticket', 'Cabin'])  # Drop irrelevant columns
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})  # Encode 'Sex' feature
data = pd.get_dummies(data, drop_first=True)  # One-hot encoding for 'Embarked'

# Define the target variable and features
X = data.drop(columns='Survived')
y = data['Survived']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Preprocessing pipeline (handling missing values, scaling features)
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Handle missing values
    ('scaler', StandardScaler())  # Feature scaling
])

# Apply the same transformation for categorical features
categorical_features = X.select_dtypes(include=['object']).columns
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # Handle missing categorical values
    ('encoder', 'passthrough')  # Use 'passthrough' to handle categorical features in pd.get_dummies outside the pipeline
])

# ColumnTransformer to apply different transformations to numeric and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Model selection pipeline with different classifiers
models = [
    ('Logistic Regression', LogisticRegression(max_iter=1000)),
    ('Random Forest', RandomForestClassifier()),
    ('SVM', SVC()),
    ('XGBoost', xgb.XGBClassifier()),
    ('LightGBM', lgb.LGBMClassifier())
]

# Loop through each model and perform cross-validation
best_score = 0
best_model = None

for model_name, model in models:
    print(f"Training {model_name}...")
    # Create a pipeline for each model
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    
    # Evaluate the model using cross-validation
    cv_score = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
    mean_score = np.mean(cv_score)
    print(f"{model_name} Mean Cross-Validation Score: {mean_score:.4f}")
    
    if mean_score > best_score:
        best_score = mean_score
        best_model = model_name

# Output the best model
print(f"\nThe best model is: {best_model} with a mean cross-validation score of {best_score:.4f}")

# Find the index of the best model in the models list
best_model_index = next(i for i, (name, _) in enumerate(models) if name == best_model)

# Final model training and evaluation on the test set
best_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', models[best_model_index][1])])
best_pipeline.fit(X_train, y_train)
test_score = best_pipeline.score(X_test, y_test)
print(f"Test set score for the best model: {test_score:.4f}")

# Plotting feature importances for the best model (if available)
if hasattr(best_pipeline.named_steps['classifier'], 'feature_importances_'):
    importances = best_pipeline.named_steps['classifier'].feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.title(f"Feature Importances for {best_model}")
    plt.bar(range(X_train.shape[1]), importances[indices], align='center')
    plt.xticks(range(X_train.shape[1]), X_train.columns[indices], rotation=90)
    plt.show()
