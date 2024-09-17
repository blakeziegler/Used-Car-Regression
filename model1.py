import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import numpy as np

# Load the cleaned datasets
train_clean = pd.read_csv('playground-series-s4e9/clean_train.csv')
test_clean = pd.read_csv('playground-series-s4e9/clean_test.csv')

# Prepare the data
X = train_clean.drop(columns=['price', 'id'])  # Features
y = train_clean['price']  # Target

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the XGBoost model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', seed=42)

# Define the parameter search space for Bayesian optimization
search_space = {
    'max_depth': Integer(3, 10),
    'eta': Real(0.01, 0.3, 'uniform'),
    'subsample': Real(0.6, 1.0, 'uniform'),
    'colsample_bytree': Real(0.6, 1.0, 'uniform'),
    'n_estimators': Integer(200, 2000),
    'gamma': Real(0, 1, 'uniform'),
    'alpha': Real(0, 1, 'uniform'),
    'lambda': Real(0, 1, 'uniform')
}

# Perform Bayesian optimization using BayesSearchCV
opt = BayesSearchCV(xgb_model, search_space, n_iter=250, scoring='neg_root_mean_squared_error', cv=5, verbose=1, n_jobs=-1, random_state=42)
opt.fit(X_train, y_train)

# Get the best parameters
best_params = opt.best_params_
print(f'Best Parameters: {best_params}')

# Train the model with the best hyperparameters on the entire dataset
best_model = xgb.XGBRegressor(objective='reg:squarederror', seed=42, **best_params)
best_model.fit(X_train, y_train)

# Evaluate the model on the validation set
y_pred_val = best_model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
print(f'Validation RMSE: {rmse}')

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

# Define base models for stacking
lgb_model = LGBMRegressor()
cat_model = CatBoostRegressor(learning_rate=0.1, depth=6, iterations=1500, verbose=0)

# Define stacking model
stacking_model = StackingRegressor(
    estimators=[('xgb', xgb.XGBRegressor(objective='reg:squarederror', seed=42, **best_params)), 
                ('lgb', lgb_model), 
                ('cat', cat_model)],
    final_estimator=Ridge()
)
# Fit the stacking model
stacking_model.fit(X_train, y_train)

# Evaluate the stacking model
y_pred_val_stack = stacking_model.predict(X_val)
rmse_stack = np.sqrt(mean_squared_error(y_val, y_pred_val_stack))
print(f'Validation RMSE (Stacking): {rmse_stack}')

# Prepare the test data
X_test = test_clean.drop(columns=['id'])

# Make predictions on the test set using the stacking model
y_pred_test = stacking_model.predict(X_test)


# Assuming lgb_model is your trained LightGBM model
importances = lgb_model.feature_importances_
features = X_train.columns

# Create a DataFrame for feature importances
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Print the feature importances table
print("Feature Importances - LightGBM")
print(importance_df)


base_models = stacking_model.named_estimators_

# Iterate over base models and print feature importances
for name, model in base_models:
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        print(f"\nFeature Importances - {name}")
        print(importance_df)



# Prepare the submission file
submission = pd.DataFrame({
    'id': test_clean['id'],
    'price': y_pred_test
})

# Save the submission file
submission.to_csv('submission.csv', index=False)
