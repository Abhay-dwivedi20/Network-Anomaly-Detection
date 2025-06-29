import os
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from utils.preprocessing import load_and_preprocess
from utils.evaluation import evaluate_model

# Load data
X, y, scaler = load_and_preprocess("data/network-data.csv")

# 🔐 Ensure labels exist
if y is None:
    raise ValueError("❌ Error: The dataset must contain a 'Label' column to train the model.")

# 🧠 Convert string labels to binary:
# Normal (Benign) → 1, Anomalies (everything else) → -1
y = y.apply(lambda label: 1 if str(label).strip().lower() == "benign" else -1)

# 🧪 Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🚀 Train model
model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
model.fit(X_train)

# 📊 Predict and evaluate
y_pred = model.predict(X_test)
evaluate_model(y_test, y_pred)

# 💾 Save model and scaler
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/isolation_forest.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("✅ Model and scaler saved successfully.")
