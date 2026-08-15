"""
Data Preprocessing Module
=========================
Handles data loading, cleaning, encoding, and splitting
for the churn prediction pipeline.
"""

import pandas as pd
import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings

warnings.filterwarnings("ignore")


class DataPreprocessor:
    """End-to-end data preprocessing pipeline for churn prediction."""

    def __init__(self, config_path="config/config.yaml"):
        """Initialize preprocessor with configuration."""
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.target_col = self.config["data"]["target_column"]

    def load_data(self, filepath):
        """Load raw data from CSV file."""
        print(f"📂 Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        return df

    def clean_data(self, df):
        """Clean the dataset — handle missing values, duplicates, and type corrections."""
        print("🧹 Cleaning data...")
        df = df.copy()

        # Drop customer ID if present
        if "customerID" in df.columns:
            df.drop("customerID", axis=1, inplace=True)

        # Convert TotalCharges to numeric
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

        # Handle missing values
        missing_before = df.isnull().sum().sum()
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col].fillna(df[col].median(), inplace=True)
        for col in df.select_dtypes(include=["object"]).columns:
            df[col].fillna(df[col].mode()[0], inplace=True)
        missing_after = df.isnull().sum().sum()
        print(f"   Missing values: {missing_before} → {missing_after}")

        # Drop duplicates
        dupes = df.duplicated().sum()
        df.drop_duplicates(inplace=True)
        print(f"   Duplicates removed: {dupes}")

        return df

    def encode_features(self, df):
        """Encode categorical features using Label Encoding."""
        print("🔢 Encoding categorical features...")
        df = df.copy()
        categorical_cols = df.select_dtypes(include=["object"]).columns

        for col in categorical_cols:
            if col != self.target_col:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le

        # Encode target variable if it's categorical
        if self.target_col in df.columns and df[self.target_col].dtype == "object":
            le = LabelEncoder()
            df[self.target_col] = le.fit_transform(df[self.target_col])
            self.label_encoders[self.target_col] = le

        print(f"   Encoded {len(categorical_cols)} categorical columns")
        return df

    def scale_features(self, df, target_col=None):
        """Scale numerical features using StandardScaler."""
        print("📏 Scaling numerical features...")
        target = target_col or self.target_col
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if target in numerical_cols:
            numerical_cols.remove(target)

        df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
        print(f"   Scaled {len(numerical_cols)} numerical columns")
        return df

    def split_data(self, df):
        """Split data into training and test sets."""
        print("✂️ Splitting data...")
        X = df.drop(self.target_col, axis=1)
        y = df[self.target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config["data"]["test_size"],
            random_state=self.config["data"]["random_state"],
            stratify=y,
        )
        print(f"   Train: {X_train.shape}, Test: {X_test.shape}")
        return X_train, X_test, y_train, y_test

    def fit_transform(self, filepath):
        """Run the full preprocessing pipeline."""
        print("\n" + "=" * 50)
        print("🚀 ChurnGuard AI — Data Preprocessing Pipeline")
        print("=" * 50 + "\n")

        df = self.load_data(filepath)
        df = self.clean_data(df)
        df = self.encode_features(df)
        df = self.scale_features(df)

        print("\n✅ Preprocessing complete!")
        print(f"   Final shape: {df.shape}\n")
        return df


if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    df = preprocessor.fit_transform("data/raw/customer_data.csv")
    df.to_csv("data/processed/cleaned_data.csv", index=False)
    print("💾 Saved processed data to data/processed/cleaned_data.csv")
