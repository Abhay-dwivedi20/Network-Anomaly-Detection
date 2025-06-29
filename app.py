from flask import Flask, render_template, request
import pandas as pd
import joblib
from utils.preprocessing import load_and_preprocess
import os

app = Flask(__name__)

MODEL_PATH = "models/isolation_forest.pkl"
SCALER_PATH = "models/scaler.pkl"

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.endswith('.csv'):
            filepath = os.path.join("data", file.filename)
            file.save(filepath)

            # Preprocess uploaded data
            X, _, _ = load_and_preprocess(filepath)

            model = joblib.load(MODEL_PATH)
            predictions = model.predict(X)
            labels = ['Anomaly' if p == -1 else 'Normal' for p in predictions]

            df = pd.DataFrame({'Prediction': labels})

            return render_template('result.html', tables=[df.to_html(classes='data')], titles=df.columns.values)

        return "❌ Invalid file format. Please upload a CSV."
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
