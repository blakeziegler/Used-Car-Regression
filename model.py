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

# Split the training data for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


best_model = xgb.XGBRegressor(
    n_estimators=2500,
    eta=0.003,
    max_depth=5,
    min_child_weight=0.004,
    subsample=0.99,
    colsample_bytree=0.45,
    reg_lambda=12.5,
    alpha=0.52,
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42
)
rmse_scorer = make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)), greater_is_better=False)
# Train the model with the best hyperparameters on the entire dataset
best_model.fit(X, y)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
y_pred_val = best_model.predict(X_val)

# Evaluate the model on the validation set
al = best_model.predict(X_val)

# Calculate RMSE on the validation set
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
print(f'Validation RMSE: {rmse_val}')

# Prepare the test data
X_test = test_clean.drop(columns=['id'])

# Make predictions on the test set
y_pred_test = best_model.predict(X_test)
# Prepare the submission file
submission = pd.DataFrame({
    'id': test_clean['id'],
    'price': y_pred_test
})

# Save the submission file
submission.to_csv('submission3.csv', index=False)


