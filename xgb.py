import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# Load your dataset
df = pd.read_csv("ecommerce_fraud_data.csv")

# Replace with your actual target column name
target_col = 'IsFraudulent'

X = df.drop(columns=[target_col])
y = df[target_col]

# Encode categorical columns
label_encoders = {}
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

# Evaluate
y_pred = xgb_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ XGBoost Accuracy: {acc * 100:.2f}%")

# Save model and encoders
pickle.dump(xgb_model, open("xgb_model.pkl", "wb"))
pickle.dump(label_encoders, open("label_encoders.pkl", "wb"))

print("🎯 xgb_model.pkl and label_encoders.pkl saved successfully.")
