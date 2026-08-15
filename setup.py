"""Setup script for ChurnGuard AI."""

from setuptools import setup, find_packages

setup(
    name="churnguard-ai",
    version="1.0.0",
    author="Rupesh",
    description="End-to-End Customer Churn Prediction System",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Rupesh1106/ChurnGuard-AI-End-to-End-Customer-Churn-Prediction-System",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "shap>=0.42.0",
        "streamlit>=1.28.0",
        "plotly>=5.15.0",
        "joblib>=1.3.0",
        "pyyaml>=6.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
