import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from scipy.stats import uniform
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error
import numpy as np

# Load the cleaned datasets
train_clean = pd.read_csv('playground-series-s4e9/clean_train.csv')
test_clean = pd.read_csv('playground-series-s4e9/clean_test.csv')

# Prepare the data
X = train_clean.drop(columns=['price', 'id'])  # Features
y = train_clean['price']  # Target

# Split 10% of the training data for hyperparameter tuning
X_tune, _, y_tune, _ = train_test_split(X, y, test_size=0.8, random_state=42)

# Define the XGBoost model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', seed=42)

# Define the parameter grid for RandomizedSearchCV
param_dist = {
    'max_depth': 5,
    'eta': 0.2,  # Uniform distribution for eta between 0.1 and 0.9
    'subsample': 0.95, # Uniform distribution for subsample between 0.1 and 1.0
    'colsample_bytree': 0.85,  # Uniform distribution for colsample_bytree between 0.1 and 1.0
    'n_estimators': 1500,
    'gamma': 0.225,
    'alpha': 0.225,  # Uniform distribution for alpha between 0 and 1
    'lambda': 0.225  # Uniform distribution for lambda between 0 and 1
}

# Define RMSE scoring
rmse_scorer = make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)), greater_is_better=False)

# Perform hyperparameter tuning using RandomizedSearchCV on 10% of the data
random_search = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_dist, scoring=rmse_scorer, cv=5, n_iter=250, verbose=4, n_jobs=-1, random_state=42)
random_search.fit(X_tune, y_tune)

# Get the best parameters
best_params = random_search.best_params_
print(f'Best Parameters: {best_params}')

# Train the model with the best hyperparameters on the entire dataset
best_model = xgb.XGBRegressor(objective='reg:squarederror', seed=42, **best_params)
best_model.fit(X, y)

# Evaluate the model on the validation set
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
y_pred_val = best_model.predict(X_val)

# Calculate RMSE on the validation set
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
print(f'Validation RMSE: {rmse_val}')

# Prepare the test data
X_test = test_clean.drop(columns=['id'])

# Make predictions on the test set
y_pred_test = best_model.predict(X_test)

importances = best_model.feature_importances_
features = X_train.columns

# Create a DataFrame for feature importances
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Print the feature importances table
print("Feature Importances - XGBoost")
print(importance_df)
# Prepare the submission file
submission = pd.DataFrame({
    'id': test_clean['id'],
    'price': y_pred_test
})

# Save the submission file
submission.to_csv('submission.csv', index=False)

# current best = 6709
