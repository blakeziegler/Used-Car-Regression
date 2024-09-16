import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# Load the cleaned datasets
train_clean = pd.read_csv('train_clean.csv')
test_clean = pd.read_csv('test_clean.csv')

# Prepare the data
X = train_clean.drop(columns=['Response', 'id'])  # Features
y = train_clean['Response']  # Target

# Split the training data for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the XGBoost model with the adjusted hyperparameters
best_params = {
    'alpha': 0.475,  # Ensuring alpha is within [0, 1]
    'colsample_bytree': 0.88,
    'eta': 0.28,
    'gamma': 0.221,
    'lambda': 0.254,  # Ensuring lambda is within [0, 1]
    'max_depth': 7,
    'n_estimators': 2500,
    'subsample': 0.99,
    'tree_method': 'hist',
    'grow_policy': 'depthwise'
}

# Train the model with the best hyperparameters on the entire dataset
best_model = xgb.XGBClassifier(objective='binary:logistic', seed=42, **best_params)
best_model.fit(X_train, y_train)

# Evaluate the model on the validation set
y_pred_val = best_model.predict_proba(X_val)[:, 1]
y_pred_val_bin = (y_pred_val > 0.5).astype(int)

accuracy = accuracy_score(y_val, y_pred_val_bin)
auc = roc_auc_score(y_val, y_pred_val)

print(f'Validation Accuracy: {accuracy}')
print(f'Validation AUC: {auc}')

# Prepare the test data
X_test = test_clean.drop(columns=['id'])

# Make predictions on the test set
y_pred_test = best_model.predict_proba(X_test)[:, 1]

# Prepare the submission file
submission = pd.DataFrame({
    'id': test_clean['id'],
    'Response': y_pred_test
})

# Save the submission file
submission.to_csv('submission2.csv', index=False)

#submission.csv
# Validation Accuracy: 0.8805989673875252
# Validation AUC: 0.8802304726348824