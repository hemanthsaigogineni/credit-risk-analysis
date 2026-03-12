from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib
import logging
import os

# ============================================================
# Flask REST API for Credit Risk Assessment
# Author: Hemanth Sai Gogineni
# Role: Data Scientist @ Mizuho Bank
# ============================================================

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load pre-trained model and scaler
MODEL_PATH = os.getenv('MODEL_PATH', 'best_credit_risk_model.h5')
SCALER_PATH = os.getenv('SCALER_PATH', 'scaler.pkl')

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    logger.info("Model and scaler loaded successfully.")
except Exception as e:
    logger.warning(f"Could not load model/scaler: {e}. Using dummy for demo.")
    model = None
    scaler = None


def preprocess_input(data: dict) -> np.ndarray:
    """
    Preprocess incoming JSON request into model-ready feature array.
    Expected fields: age, income, loan_amount, credit_score,
                     employment_years, debt_existing, num_late_payments
    """
    features = [
        data.get('age', 35),
        data.get('income', 60000),
        data.get('loan_amount', 15000),
        data.get('credit_score', 650),
        data.get('employment_years', 5),
        data.get('debt_existing', 5000),
        data.get('num_late_payments', 0),
    ]
    income = data.get('income', 60000)
    loan = data.get('loan_amount', 15000)
    features.append(loan / (income + 1))  # debt_to_income

    X = np.array(features).reshape(1, -1)
    return X


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None}), 200


@app.route('/predict', methods=['POST'])
def predict_credit_risk():
    """
    Predict credit risk for a single applicant.

    Request body (JSON):
    {
        "age": 35,
        "income": 75000,
        "loan_amount": 20000,
        "credit_score": 680,
        "employment_years": 8,
        "debt_existing": 3000,
        "num_late_payments": 1
    }

    Returns:
    {
        "risk_score": 0.23,
        "risk_label": "Low Risk",
        "default_probability": 0.23
    }
    """
    try:
        data = request.get_json(force=True)
        logger.info(f"Received prediction request: {data}")

        if not data:
            return jsonify({'error': 'No input data provided'}), 400

        X = preprocess_input(data)

        if scaler:
            X = scaler.transform(X)

        # Reshape for LSTM: (1, time_steps, features) - simulate sequence
        X_seq = np.tile(X, (1, 10, 1))  # repeat input across 10 time steps

        if model:
            risk_score = float(model.predict(X_seq)[0][0])
        else:
            # Demo mode: use heuristic scoring
            credit_score = data.get('credit_score', 650)
            late_payments = data.get('num_late_payments', 0)
            risk_score = max(0.0, min(1.0, (850 - credit_score) / 850 + late_payments * 0.05))

        risk_label = (
            'High Risk' if risk_score > 0.7
            else 'Medium Risk' if risk_score > 0.4
            else 'Low Risk'
        )

        response = {
            'risk_score': round(risk_score, 4),
            'default_probability': round(risk_score, 4),
            'risk_label': risk_label,
            'applicant_data': data
        }
        logger.info(f"Prediction result: {response}")
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Batch credit risk prediction for multiple applicants.

    Request body (JSON):
    {
        "applicants": [
            {"age": 35, "income": 75000, ...},
            {"age": 42, "income": 50000, ...}
        ]
    }
    """
    try:
        data = request.get_json(force=True)
        applicants = data.get('applicants', [])

        if not applicants:
            return jsonify({'error': 'No applicants provided'}), 400

        results = []
        for applicant in applicants:
            X = preprocess_input(applicant)
            if scaler:
                X = scaler.transform(X)
            X_seq = np.tile(X, (1, 10, 1))

            if model:
                risk_score = float(model.predict(X_seq)[0][0])
            else:
                credit_score = applicant.get('credit_score', 650)
                late_payments = applicant.get('num_late_payments', 0)
                risk_score = max(0.0, min(1.0, (850 - credit_score) / 850 + late_payments * 0.05))

            risk_label = (
                'High Risk' if risk_score > 0.7
                else 'Medium Risk' if risk_score > 0.4
                else 'Low Risk'
            )
            results.append({
                'applicant': applicant,
                'risk_score': round(risk_score, 4),
                'risk_label': risk_label
            })

        return jsonify({'predictions': results, 'count': len(results)}), 200

    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
    logger.info(f"Credit Risk API running on port {port}")
