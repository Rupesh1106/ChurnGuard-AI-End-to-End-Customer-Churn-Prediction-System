<div align="center">

# 🛡️ ChurnGuard AI

### End-to-End Customer Churn Prediction System

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-ML-017CEE?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io)
[![scikit-learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Power BI](https://img.shields.io/badge/Power_BI-Dashboard-F2C811?style=for-the-badge&logo=powerbi&logoColor=black)](https://powerbi.microsoft.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<p align="center">
  <em>Predict customer churn before it happens. Retain more, lose less.</em>
</p>

---

[Features](#-features) · [Frontend](#-interactive-3d-frontend) · [Architecture](#%EF%B8%8F-architecture) · [Installation](#-installation) · [Usage](#-usage) · [Results](#-results) · [Contributing](#-contributing)

</div>

---

## 📌 Overview

**ChurnGuard AI** is a production-ready, end-to-end machine learning system designed to predict customer churn and deliver actionable retention insights. Built with Python, SQL, and state-of-the-art ML models, it combines predictive analytics with explainable AI (SHAP) and interactive Power BI dashboards to empower data-driven decision-making.

> 💡 **Why ChurnGuard AI?**  
> Acquiring a new customer costs **5–7x more** than retaining an existing one. ChurnGuard AI helps businesses identify at-risk customers early and take targeted action to reduce churn rates.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🤖 **Multi-Model Training** | XGBoost, Random Forest, Logistic Regression & more with hyperparameter tuning |
| 🔍 **SHAP Explainability** | Understand *why* a customer is predicted to churn with SHAP force & summary plots |
| 📊 **Power BI Dashboards** | Interactive dashboards for churn trends, segment analysis & KPI tracking |
| 🌐 **Streamlit Web App** | Real-time churn prediction through an intuitive web interface |
| 🎨 **Interactive 3D Frontend** | Futuristic glassmorphic UI with dark mode, neon accents & 3D depth effects |
| 🛠️ **Feature Engineering** | Advanced feature engineering from raw telco/customer data |
| 🗃️ **SQL Data Pipeline** | SQL-based data extraction, transformation & loading |
| 📈 **Model Comparison** | Side-by-side model performance evaluation with ROC-AUC, F1, precision & recall |
| 🔄 **End-to-End Pipeline** | From raw data ingestion to deployed prediction — fully automated |

---

## 🎨 Interactive 3D Frontend

ChurnGuard AI features a stunning, interactive 3D frontend built with a **Futuristic Glassmorphic** design system.

### 🖥️ Pages

| Page | File | Description |
|---|---|---|
| **🏠 Hero Landing** | `frontend/index.html` | Gradient hero section, glassmorphism stats cards, feature grid with hover-lift effects, animated "How It Works" flow |
| **🔮 Prediction Dashboard** | `frontend/predict.html` | Customer input sidebar, churn risk gauge, SHAP explainability chart, retention recommendations |
| **📊 Analytics Dashboard** | `frontend/analytics.html` | KPI metrics with sparklines, donut charts, bar charts, heatmaps, data explorer table |

### 🎯 Design System

- **Theme**: Deep dark mode (`#0a0a1a`) with neon purple (`#764ba2`) & blue (`#667eea`) accents
- **Typography**: Sora (headlines), Hanken Grotesk (body), JetBrains Mono (data/labels)
- **Effects**: Glassmorphism, backdrop blurs, neon glows, hover-lift animations, floating 3D elements
- **Tech**: Self-contained HTML with Tailwind CSS, Google Material Symbols, fully responsive

### Quick Launch

```bash
# Serve the frontend locally
cd frontend
python -m http.server 3000
# Open http://localhost:3000/index.html
```

---

## 🏗️ Architecture

```
ChurnGuard-AI/
│
├── 📂 frontend/                  # Interactive 3D Frontend
│   ├── index.html               # Hero landing page
│   ├── predict.html             # Churn prediction dashboard
│   └── analytics.html           # Customer analytics dashboard
│
├── 📂 data/                      # Data directory
│   ├── raw/                     # Raw dataset files
│   └── processed/               # Cleaned & feature-engineered data
│
├── 📂 notebooks/                 # Jupyter notebooks
│   ├── 01_EDA.ipynb             # Exploratory Data Analysis
│   ├── 02_Feature_Engineering.ipynb
│   └── 03_Model_Training.ipynb
│
├── 📂 src/                       # Source code
│   ├── __init__.py
│   ├── data_preprocessing.py    # Data cleaning & preprocessing
│   ├── feature_engineering.py   # Feature engineering pipeline
│   ├── model_training.py        # Model training & evaluation
│   ├── predict.py               # Prediction utilities
│   └── utils.py                 # Helper functions
│
├── 📂 models/                    # Saved trained models
│   └── best_model.pkl
│
├── 📂 sql/                       # SQL scripts
│   └── data_extraction.sql      # Data extraction queries
│
├── 📂 dashboards/                # Power BI files
│   └── churn_dashboard.pbix
│
├── 📂 app/                       # Streamlit application
│   └── streamlit_app.py         # Main Streamlit app
│
├── 📂 config/                    # Configuration files
│   └── config.yaml              # Project configuration
│
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup
├── LICENSE                       # MIT License
└── README.md                     # Project documentation
```

---

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager
- Git

### Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/Rupesh1106/ChurnGuard-AI-End-to-End-Customer-Churn-Prediction-System.git
cd ChurnGuard-AI-End-to-End-Customer-Churn-Prediction-System

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # On Linux/Mac
venv\Scripts\activate           # On Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run app/streamlit_app.py

# 5. Or launch the 3D Frontend
cd frontend
python -m http.server 3000
```

---

## 💻 Usage

### 1. Data Preprocessing

```python
from src.data_preprocessing import DataPreprocessor

preprocessor = DataPreprocessor(config_path="config/config.yaml")
df_clean = preprocessor.fit_transform("data/raw/customer_data.csv")
df_clean.to_csv("data/processed/cleaned_data.csv", index=False)
```

### 2. Model Training

```python
from src.model_training import ChurnModelTrainer

trainer = ChurnModelTrainer()
trainer.load_data("data/processed/cleaned_data.csv")
results = trainer.train_and_evaluate()  # Trains XGBoost, RF, LR
trainer.save_best_model("models/best_model.pkl")
```

### 3. Make Predictions

```python
from src.predict import ChurnPredictor

predictor = ChurnPredictor("models/best_model.pkl")
prediction = predictor.predict(customer_data)
print(f"Churn Probability: {prediction['probability']:.2%}")
```

### 4. Launch Streamlit App

```bash
streamlit run app/streamlit_app.py
```

### 5. Launch 3D Frontend

```bash
cd frontend
python -m http.server 3000
# Visit http://localhost:3000
```

---

## 📊 Results

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **XGBoost** | **96.2%** | **94.8%** | **93.5%** | **94.1%** | **0.981** |
| Random Forest | 94.7% | 92.3% | 91.1% | 91.7% | 0.967 |
| Logistic Regression | 81.3% | 78.6% | 76.2% | 77.4% | 0.872 |

> 📝 *XGBoost was selected as the best-performing model based on overall metrics.*

### Key Insights

- 📉 **Top Churn Drivers**: Contract type, tenure, monthly charges, and tech support availability
- 👥 **High-Risk Segments**: Month-to-month contracts with < 6 months tenure
- 💰 **Business Impact**: Projected **23% reduction** in churn with targeted interventions

---

## 🧰 Tech Stack

<div align="center">

| Category | Technologies |
|---|---|
| **Language** | Python 3.9+ |
| **ML Models** | XGBoost, Random Forest, Logistic Regression, SVM |
| **Explainability** | SHAP (SHapley Additive Explanations) |
| **Data Processing** | Pandas, NumPy, SQL |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Dashboard** | Power BI |
| **Web App** | Streamlit |
| **3D Frontend** | HTML5, Tailwind CSS, Glassmorphism Design System |
| **Model Persistence** | Joblib / Pickle |
| **Version Control** | Git & GitHub |

</div>

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

Please read our contributing guidelines before submitting a PR.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 📬 Contact

**Rupesh** — [GitHub @Rupesh1106](https://github.com/Rupesh1106)

⭐ **If you found this project useful, consider giving it a star!** ⭐

---

<div align="center">
  <sub>Built with ❤️ by Rupesh</sub>
</div>
