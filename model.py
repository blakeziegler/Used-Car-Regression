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

# load data
train_clean = pd.read_csv('data_cleaning/clean_train.csv')
test_clean = pd.read_csv('data_cleaning/clean_test.csv')

# set target and features
X = train_clean.drop(columns=['price', 'id'])
y = train_clean['price']

# split data
X_tune, _, y_tune, _ = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize  XGBoost model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', seed=42)

# set parameters and parameter ranges
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

# set search algo
random_search = BayesSearchCV(
    estimator=xgb_model, 
    search_spaces=param_dist,
    scoring='neg_root_mean_squared_error',
    cv=3,
    n_iter=200,
    verbose=5,
    n_jobs=-1,
    random_state=42)
random_search.fit(X_tune, y_tune)


# find best params
best_params = random_search.best_params_
print(f'Best Parameters: {best_params}')

# train the model with the best params
best_model = xgb.XGBRegressor(objective='reg:squarederror', seed=42, **best_params)
best_model.fit(X, y)

# Evaluate the model on the validation set
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
y_pred_val = best_model.predict(X_val)
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
print(f'Validation RMSE: {rmse_val}')


lgb_model = LGBMRegressor()
cat_model = CatBoostRegressor(learning_rate=0.1, depth=6, iterations=1000, verbose=0)

# stack model structure
stacking_model = StackingRegressor(
    estimators=[('xgb', xgb.XGBRegressor(objective='reg:squarederror', seed=42, **best_params)), 
                ('lgb', lgb_model), 
                ('cat', cat_model)],
    final_estimator=Ridge()
)

# Fit the stacking model
stacking_model.fit(X_train, y_train)

# Evaluate stacking model
y_pred_val_stack = stacking_model.predict(X_val)
rmse_stack = np.sqrt(mean_squared_error(y_val, y_pred_val_stack))
print(f'RMSE (Stacking): {rmse_stack}')

# Prep the test data
X_test = test_clean.drop(columns=['id'])

# Make predictions using stacking model
y_pred_test = stacking_model.predict(X_test)
submission = pd.DataFrame({
    'id': test_clean['id'],
    'price': y_pred_test
})

# save for submission
submission.to_csv('submission.csv', index=False)

