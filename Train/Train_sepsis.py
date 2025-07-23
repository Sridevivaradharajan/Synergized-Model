# Preprocessing steps followed from a Kaggle notebook on Sepsis prediction
# Not included to respect original author’s licensing
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import joblib

# Load sample preprocessed data
df_train_impute = pd.read_csv("../sample_data/sepsis_preprocessed.csv")

# Define features and label
X = df_train_impute.drop('SepsisLabel', axis=1)
y = df_train_impute['SepsisLabel']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Clean column names (convert to strings, remove 'Unnamed')
X_train.columns = X_train.columns.astype(str)
X_test.columns = X_test.columns.astype(str)
X_train = X_train.loc[:, ~X_train.columns.str.contains('^Unnamed')]
X_test = X_test.loc[:, ~X_test.columns.str.contains('^Unnamed')]

# Your original XGBoost model setup
model = XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=10, random_state=0)
model.fit(X_train, y_train)

# Evaluate
xgb_predictions = model.predict(X_test)
print(classification_report(y_test, xgb_predictions))

# Save trained model (optional)
joblib.dump(model, '../models/sepsis_model.pkl')
