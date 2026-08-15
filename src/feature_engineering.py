"""
Feature Engineering Module
==========================
Creates advanced features from the preprocessed data
to improve model performance.
"""

import pandas as pd
import numpy as np


class FeatureEngineer:
    """Feature engineering pipeline for churn prediction."""

    def __init__(self):
        self.feature_names = []

    def create_tenure_groups(self, df):
        """Create tenure-based customer segments."""
        df = df.copy()
        if "tenure" in df.columns:
            df["tenure_group"] = pd.cut(
                df["tenure"],
                bins=[0, 12, 24, 48, 60, 72],
                labels=["0-1yr", "1-2yr", "2-4yr", "4-5yr", "5-6yr"],
            )
        return df

    def create_charge_features(self, df):
        """Create features based on charges."""
        df = df.copy()
        if "MonthlyCharges" in df.columns and "tenure" in df.columns:
            df["AvgChargesPerMonth"] = np.where(
                df["tenure"] > 0,
                df["TotalCharges"] / df["tenure"],
                df["MonthlyCharges"],
            )

        if "MonthlyCharges" in df.columns and "TotalCharges" in df.columns:
            df["ChargeRatio"] = np.where(
                df["TotalCharges"] > 0,
                df["MonthlyCharges"] / df["TotalCharges"],
                0,
            )
        return df

    def create_service_count(self, df):
        """Count the number of services each customer has subscribed to."""
        df = df.copy()
        service_cols = [
            "PhoneService", "MultipleLines", "InternetService",
            "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies",
        ]
        existing_cols = [col for col in service_cols if col in df.columns]
        if existing_cols:
            df["TotalServices"] = df[existing_cols].apply(
                lambda x: (x != 0).sum() if x.dtype != "object" else (x == "Yes").sum(),
                axis=1,
            )
        return df

    def create_contract_value(self, df):
        """Create a contract value metric."""
        df = df.copy()
        if "MonthlyCharges" in df.columns and "Contract" in df.columns:
            contract_multiplier = {0: 1, 1: 12, 2: 24}  # Encoded values
            df["ContractValue"] = df.apply(
                lambda row: row["MonthlyCharges"] * contract_multiplier.get(row["Contract"], 1),
                axis=1,
            )
        return df

    def transform(self, df):
        """Apply all feature engineering steps."""
        print("\n⚙️ Feature Engineering Pipeline")
        print("-" * 40)

        initial_cols = len(df.columns)

        df = self.create_tenure_groups(df)
        df = self.create_charge_features(df)
        df = self.create_service_count(df)
        df = self.create_contract_value(df)

        new_cols = len(df.columns) - initial_cols
        print(f"   ✅ Created {new_cols} new features")
        print(f"   📊 Total features: {len(df.columns)}")

        self.feature_names = df.columns.tolist()
        return df


if __name__ == "__main__":
    df = pd.read_csv("data/processed/cleaned_data.csv")
    engineer = FeatureEngineer()
    df_engineered = engineer.transform(df)
    df_engineered.to_csv("data/processed/engineered_data.csv", index=False)
    print("\n💾 Saved engineered data.")
