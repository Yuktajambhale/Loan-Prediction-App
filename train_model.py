import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------
# 🔹 1. Load Dataset
# -------------------------------
df = pd.read_csv("C:/Users/Admin/Downloads/project_loan/clean_loan_data.csv")  # or use 'train.csv' if using raw

# -------------------------------
# 🔹 2. Define Features and Target
# -------------------------------
# -------------------------------
# 🔹 2. Define Features and Target
# -------------------------------
# Drop ID and target column
X = df.drop(["Loan_ID", "Loan_Status"], axis=1)
y = df["Loan_Status"]


# -------------------------------
# 🔹 3. Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 🔹 4. Scale the Features
# -------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------
# 🔹 5. Train Logistic Regression Model
# -------------------------------
model = LogisticRegression(max_iter=2000, class_weight="balanced")
model.fit(X_train_scaled, y_train)

# -------------------------------
# 🔹 6. Evaluate
# -------------------------------
y_pred = model.predict(X_test_scaled)

print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📄 Classification Report:\n", classification_report(y_test, y_pred))

# -------------------------------
# 🔹 7. Save Model and Scaler
# -------------------------------
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/loan_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("\n✅ Model and scaler saved in 'model/' folder.")
