import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from scipy.stats import uniform
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error
import numpy as np
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

# Split 10% of the training data for hyperparameter tuning
X_tune, _, y_tune, _ = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the XGBoost model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', seed=42)
# Define the parameter grid for RandomizedSearchCV
param_dist = {
    'max_depth': [4, 5, 6, 7],
    'n_estimators': [750, 1000, 1200],
    'eta': Real(0.001, 0.01),
    'alpha': Real(0.01, 1),
    'reg_lambda': Real(0, 15),
    'colsample_bytree': Real(0.01, 1),
    'min_child_weight': Real(0.3, 0.7),
	'subsample': Real(0.90, 1)
}

rmse_scorer = make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)), greater_is_better=False)
random_search = BayesSearchCV(
    estimator=xgb_model, 
    search_spaces=param_dist,
    scoring=rmse_scorer,
    cv=3,
    n_iter=250,
    verbose=5,
    n_jobs=6,
    random_state=42)
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



lgb_model = LGBMRegressor()
cat_model = CatBoostRegressor(learning_rate=0.1, depth=6, iterations=1000, verbose=0)

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
submission = pd.DataFrame({
    'id': test_clean['id'],
    'price': y_pred_test
})

# Save the submission file
submission.to_csv('submission.csv', index=False)

