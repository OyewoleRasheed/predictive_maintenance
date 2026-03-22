import pandas as pd
from flask import Flask, render_template, request, send_file, jsonify
import joblib
import os
import io

app = Flask(__name__)

model = joblib.load('machine_failure_model.pkl')

# ---- Input validation ranges (based on AI4I 2020 dataset) ----
VALID_RANGES = {
    'air_temp':     (290.0, 305.0),
    'process_temp': (305.0, 315.0),
    'rot_speed':    (1168.0, 2886.0),
    'torque':       (3.8, 76.6),
    'tool_wear':    (0.0, 253.0),
}

FEATURE_COLUMNS = [
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]',
]

COLUMN_MAP = {
    'air_temp':     'Air temperature [K]',
    'process_temp': 'Process temperature [K]',
    'rot_speed':    'Rotational speed [rpm]',
    'torque':       'Torque [Nm]',
    'tool_wear':    'Tool wear [min]',
}


def validate_inputs(values: dict) -> list[str]:
    """Return a list of validation error messages (empty = all good)."""
    errors = []
    for field, (low, high) in VALID_RANGES.items():
        val = values.get(field)
        if val is None:
            errors.append(f"'{field}' is required.")
        elif not (low <= val <= high):
            errors.append(
                f"'{field}' value {val} is out of expected range [{low}, {high}]."
            )
    return errors


def classify_risk(probability: float) -> str:
    if probability < 0.3:
        return "Low Risk 🟢"
    elif probability < 0.6:
        return "Medium Risk 🟠"
    return "High Risk 🔴"


# ---------------- HOME ----------------
@app.route('/')
def home():
    return render_template(
        "index.html",
        probability=None,
        prediction=None,
        risk=None,
        error=None,
    )


# ---------------- SINGLE PREDICTION (Form) ----------------
@app.route('/predict_form', methods=['POST'])
def predict_form():
    try:
        values = {
            'air_temp':     float(request.form['air_temp']),
            'process_temp': float(request.form['process_temp']),
            'rot_speed':    float(request.form['rot_speed']),
            'torque':       float(request.form['torque']),
            'tool_wear':    float(request.form['tool_wear']),
        }
    except (ValueError, KeyError) as e:
        return render_template(
            "index.html",
            prediction=None,
            probability=None,
            risk=None,
            error=f"Invalid input: {e}",
        )

    errors = validate_inputs(values)
    if errors:
        return render_template(
            "index.html",
            prediction=None,
            probability=None,
            risk=None,
            error=" | ".join(errors),
        )

    features = pd.DataFrame([{COLUMN_MAP[k]: v for k, v in values.items()}])
    prediction = model.predict(features)[0]
    probability = float(model.predict_proba(features)[0][1])

    result = "Failure Likely ⚠️" if prediction == 1 else "Machine Healthy ✅"

    return render_template(
        "index.html",
        prediction=result,
        probability=round(probability, 2),
        risk=classify_risk(probability),
        error=None,
    )


# ---------------- BATCH PREDICTION ----------------
@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    file = request.files.get('file')

    if not file or file.filename == '':
        return "No file uploaded.", 400

    extension = os.path.splitext(file.filename)[1].lower()

    if extension == '.csv':
        try:
            df = pd.read_csv(file, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(file, encoding='latin-1')
    elif extension == '.xlsx':
        df = pd.read_excel(file)
    else:
        return "Unsupported file type. Please upload a .csv or .xlsx file.", 400

    # Accept both short column names (from template) and full names
    df = df.rename(columns=COLUMN_MAP)

    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        return (
            f"File is missing required columns: {missing}. "
            "Download the template to see the expected format."
        ), 400

    features = df[FEATURE_COLUMNS]
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]

    df['Prediction'] = ['Failure Likely' if p == 1 else 'Machine Healthy'
                        for p in predictions]
    df['Failure_Probability'] = probabilities.round(4)
    df['Risk_Level'] = [
        'Low' if p < 0.3 else 'Medium' if p < 0.6 else 'High'
        for p in probabilities
    ]

    output = io.BytesIO()
    df.to_csv(output, index=False)
    output.seek(0)

    return send_file(
        output,
        as_attachment=True,
        download_name="batch_predictions.csv",
        mimetype="text/csv",
    )


# ---------------- DOWNLOAD TEMPLATE ----------------
@app.route('/download_template')
def download_template():
    df = pd.DataFrame(
        [[300.0, 310.0, 1500.0, 40.0, 5.0]],
        columns=list(COLUMN_MAP.keys()),  # short names users fill in
    )

    output = io.BytesIO()
    df.to_excel(output, index=False, engine='openpyxl')
    output.seek(0)

    return send_file(
        output,
        as_attachment=True,
        download_name="machine_template.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# ---------------- API — SINGLE PREDICTION ----------------
@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    try:
        values = {
            'air_temp':     float(data['air_temp']),
            'process_temp': float(data['process_temp']),
            'rot_speed':    float(data['rot_speed']),
            'torque':       float(data['torque']),
            'tool_wear':    float(data['tool_wear']),
        }
    except KeyError as e:
        return jsonify({"error": f"Missing required field: {e}"}), 400
    except (TypeError, ValueError) as e:
        return jsonify({"error": f"Invalid value: {e}"}), 400

    errors = validate_inputs(values)
    if errors:
        return jsonify({"error": errors}), 422

    features = pd.DataFrame([{COLUMN_MAP[k]: v for k, v in values.items()}])
    prediction = int(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0][1])

    risk_map = {0: "Low", 1: "Medium", 2: "High"}
    risk_index = 0 if probability < 0.3 else 1 if probability < 0.6 else 2

    return jsonify({
        "prediction": "Failure Likely" if prediction == 1 else "Machine Healthy",
        "failure_probability": round(probability, 4),
        "risk_level": risk_map[risk_index],
    })


# ---------------- API — HEALTH CHECK ----------------
@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "model": "machine_failure_model.pkl"})


if __name__ == '__main__':
    app.run(debug=True)