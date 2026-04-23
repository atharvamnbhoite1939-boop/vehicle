import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, jsonify
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

app = Flask(__name__)

# ── Train or load model ──────────────────────────────────────────────────────
MODEL_PATH = "model.pkl"

def train_model():
    """Train model from CSV; returns fitted model + accuracy."""
    df = pd.read_csv("vehicle_maintenance_data.csv")

    features = ['Mileage', 'Reported_Issues', 'Vehicle_Age',
                 'Engine_Size', 'Odometer_Reading',
                 'Service_History', 'Accident_History']
    target = 'Need_Maintenance'

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_class = [1 if i > 0.5 else 0 for i in y_pred]
    acc = accuracy_score(y_test, y_pred_class)
    cm  = confusion_matrix(y_test, y_pred_class).tolist()

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    return model, round(acc * 100, 2), cm

def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    model, _, _ = train_model()
    return model

# Train on startup
model, MODEL_ACCURACY, CONFUSION_MATRIX = train_model()

# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html",
                           accuracy=MODEL_ACCURACY,
                           confusion_matrix=CONFUSION_MATRIX)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.form
        input_data = np.array([[
            float(data["mileage"]),
            int(data["reported_issues"]),
            int(data["vehicle_age"]),
            float(data["engine_size"]),
            float(data["odometer_reading"]),
            int(data["service_history"]),
            int(data["accident_history"]),
        ]])
        prediction = model.predict(input_data)[0]
        result = 1 if prediction > 0.5 else 0
        confidence = round(abs(prediction) * 100, 1)
        label = "Needs Maintenance ⚠️" if result == 1 else "Good Condition ✅"
        return render_template("index.html",
                               result=label,
                               confidence=min(confidence, 100),
                               accuracy=MODEL_ACCURACY,
                               confusion_matrix=CONFUSION_MATRIX,
                               form_data=data)
    except Exception as e:
        return render_template("index.html",
                               error=str(e),
                               accuracy=MODEL_ACCURACY,
                               confusion_matrix=CONFUSION_MATRIX)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API endpoint for programmatic access."""
    body = request.get_json(force=True)
    input_data = np.array([[
        float(body["mileage"]),
        int(body["reported_issues"]),
        int(body["vehicle_age"]),
        float(body["engine_size"]),
        float(body["odometer_reading"]),
        int(body["service_history"]),
        int(body["accident_history"]),
    ]])
    pred = model.predict(input_data)[0]
    result = int(pred > 0.5)
    return jsonify({
        "prediction": result,
        "label": "Needs Maintenance" if result == 1 else "Good Condition",
        "raw_score": round(float(pred), 4)
    })

@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_accuracy": MODEL_ACCURACY})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
