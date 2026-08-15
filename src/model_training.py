"""
Model Training Module
=====================
Trains multiple ML models, evaluates performance,
and selects the best model for churn prediction.
"""

import pandas as pd
import numpy as np
import yaml
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix,
)
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings("ignore")


class ChurnModelTrainer:
    """Multi-model training and evaluation pipeline."""

    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self, filepath):
        """Load processed data and split into train/test sets."""
        print("📂 Loading processed data...")
        df = pd.read_csv(filepath)
        target = self.config["data"]["target_column"]

        X = df.drop(target, axis=1)
        y = df[target]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.config["data"]["test_size"],
            random_state=self.config["data"]["random_state"],
            stratify=y,
        )
        print(f"   Train: {self.X_train.shape}, Test: {self.X_test.shape}")

    def _initialize_models(self):
        """Initialize models with configured hyperparameters."""
        cfg = self.config["models"]

        self.models = {
            "XGBoost": XGBClassifier(
                n_estimators=cfg["xgboost"]["n_estimators"],
                max_depth=cfg["xgboost"]["max_depth"],
                learning_rate=cfg["xgboost"]["learning_rate"],
                subsample=cfg["xgboost"]["subsample"],
                colsample_bytree=cfg["xgboost"]["colsample_bytree"],
                eval_metric=cfg["xgboost"]["eval_metric"],
                random_state=self.config["data"]["random_state"],
                use_label_encoder=False,
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=cfg["random_forest"]["n_estimators"],
                max_depth=cfg["random_forest"]["max_depth"],
                min_samples_split=cfg["random_forest"]["min_samples_split"],
                min_samples_leaf=cfg["random_forest"]["min_samples_leaf"],
                random_state=self.config["data"]["random_state"],
            ),
            "Logistic Regression": LogisticRegression(
                C=cfg["logistic_regression"]["C"],
                max_iter=cfg["logistic_regression"]["max_iter"],
                solver=cfg["logistic_regression"]["solver"],
                random_state=self.config["data"]["random_state"],
            ),
        }

    def _evaluate_model(self, model, name):
        """Evaluate a single model and return metrics."""
        y_pred = model.predict(self.X_test)
        y_prob = model.predict_proba(self.X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(self.y_test, y_pred),
            "precision": precision_score(self.y_test, y_pred),
            "recall": recall_score(self.y_test, y_pred),
            "f1": f1_score(self.y_test, y_pred),
            "roc_auc": roc_auc_score(self.y_test, y_prob),
        }

        cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring="roc_auc")
        metrics["cv_roc_auc_mean"] = cv_scores.mean()
        metrics["cv_roc_auc_std"] = cv_scores.std()

        return metrics

    def train_and_evaluate(self):
        """Train all models and evaluate performance."""
        print("\n" + "=" * 55)
        print("🤖 ChurnGuard AI — Model Training & Evaluation")
        print("=" * 55)

        self._initialize_models()

        for name, model in self.models.items():
            print(f"\n🔄 Training {name}...")
            model.fit(self.X_train, self.y_train)
            metrics = self._evaluate_model(model, name)
            self.results[name] = metrics

            print(f"   Accuracy:  {metrics['accuracy']:.4f}")
            print(f"   Precision: {metrics['precision']:.4f}")
            print(f"   Recall:    {metrics['recall']:.4f}")
            print(f"   F1-Score:  {metrics['f1']:.4f}")
            print(f"   ROC-AUC:   {metrics['roc_auc']:.4f}")
            print(f"   CV AUC:    {metrics['cv_roc_auc_mean']:.4f} ± {metrics['cv_roc_auc_std']:.4f}")

        # Select best model based on ROC-AUC
        self.best_model_name = max(self.results, key=lambda k: self.results[k]["roc_auc"])
        self.best_model = self.models[self.best_model_name]

        print(f"\n🏆 Best Model: {self.best_model_name}")
        print(f"   ROC-AUC: {self.results[self.best_model_name]['roc_auc']:.4f}")

        return self.results

    def save_best_model(self, filepath="models/best_model.pkl"):
        """Save the best model to disk."""
        if self.best_model is None:
            raise ValueError("No model trained yet. Call train_and_evaluate() first.")

        joblib.dump(self.best_model, filepath)
        print(f"\n💾 Best model saved to {filepath}")

    def get_results_df(self):
        """Return results as a formatted DataFrame."""
        return pd.DataFrame(self.results).T.round(4)


if __name__ == "__main__":
    trainer = ChurnModelTrainer()
    trainer.load_data("data/processed/cleaned_data.csv")
    results = trainer.train_and_evaluate()
    trainer.save_best_model("models/best_model.pkl")

    print("\n📊 Results Summary:")
    print(trainer.get_results_df())
