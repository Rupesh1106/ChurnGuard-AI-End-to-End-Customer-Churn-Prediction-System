"""
Prediction Module
=================
Loads a trained model and makes churn predictions
for new customer data.
"""

import pandas as pd
import numpy as np
import joblib
import shap


class ChurnPredictor:
    """Churn prediction with SHAP explainability."""

    def __init__(self, model_path="models/best_model.pkl"):
        """Load the trained model."""
        self.model = joblib.load(model_path)
        self.explainer = None
        print(f"✅ Model loaded from {model_path}")

    def predict(self, data):
        """Predict churn for given customer data."""
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        elif isinstance(data, pd.Series):
            data = data.to_frame().T

        prediction = self.model.predict(data)
        probability = self.model.predict_proba(data)

        result = {
            "prediction": int(prediction[0]),
            "label": "Churn" if prediction[0] == 1 else "No Churn",
            "probability": float(probability[0][1]),
            "confidence": float(max(probability[0])),
        }
        return result

    def predict_batch(self, data):
        """Predict churn for a batch of customers."""
        predictions = self.model.predict(data)
        probabilities = self.model.predict_proba(data)[:, 1]

        results_df = data.copy()
        results_df["Churn_Prediction"] = predictions
        results_df["Churn_Probability"] = probabilities
        results_df["Risk_Level"] = pd.cut(
            probabilities,
            bins=[0, 0.3, 0.6, 1.0],
            labels=["Low", "Medium", "High"],
        )
        return results_df

    def explain_prediction(self, data, feature_names=None):
        """Generate SHAP explanation for a prediction."""
        if isinstance(data, dict):
            data = pd.DataFrame([data])

        if self.explainer is None:
            self.explainer = shap.TreeExplainer(self.model)

        shap_values = self.explainer.shap_values(data)

        if feature_names is None:
            feature_names = data.columns.tolist()

        # Get feature importance for this prediction
        feature_impact = sorted(
            zip(feature_names, shap_values[0]),
            key=lambda x: abs(x[1]),
            reverse=True,
        )

        explanation = {
            "shap_values": shap_values,
            "base_value": self.explainer.expected_value,
            "top_features": feature_impact[:5],
        }
        return explanation


if __name__ == "__main__":
    predictor = ChurnPredictor("models/best_model.pkl")

    sample_customer = {
        "tenure": 12,
        "MonthlyCharges": 70.5,
        "TotalCharges": 846.0,
        "Contract": 0,
        "PaymentMethod": 2,
        "InternetService": 1,
        "TechSupport": 0,
        "OnlineSecurity": 0,
    }

    result = predictor.predict(sample_customer)
    print(f"\n🔮 Prediction: {result['label']}")
    print(f"   Probability: {result['probability']:.2%}")
    print(f"   Confidence:  {result['confidence']:.2%}")
