### 🛡️ Real-Time UPI Fraud Detection using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-ML%20Model-red?style=for-the-badge)](https://xgboost.readthedocs.io)
[![AUC](https://img.shields.io/badge/AUC%20Score-0.98+-brightgreen?style=for-the-badge)](/)
[![Dataset](https://img.shields.io/badge/Dataset-140K%20Samples-orange?style=for-the-badge)](/)
[![Hackathon](https://img.shields.io/badge/Built%20For-Hackathon-purple?style=for-the-badge)](/)

<br/>
a high-performance Machine Learning solution designed to identify fraudulent Virtual Payment Addresses (VPAs). By analyzing structural patterns, typosquatting techniques, and linguistic features, this model prevents "Man-in-the-middle" and "Impersonation" scams in the UPI ecosystem.


---



## 🚀 Getting Started



**1️⃣ Clone the repo**
```bash
git clone https://github.com/FarhanFarooqui122/fraud-upi_id-detection-ml.git
cd fraud-upi_id-detection-ml
```

**2️⃣ Install dependencies**
```bash
pip install -r requirements.txt
```

**3️⃣ Train the model** *(run once — takes ~15 mins)*
```bash
python train.py
```

**4️⃣ Launch the app**
```bash
python app.py
```

---
## 🌟 Key Features

* 🏗️ Synthetic Data Engine: Generates 140k+ realistic samples using Faker (en_IN) to simulate complex Indian fraud patterns.

* 🎯 Typosquatting Guard: Uses SequenceMatcher to catch "look-alike" handles (e.g., oksb1 vs oksbi).

* 🔬 Multi-Factor Analysis: Extracts structural, character, and linguistic features including Shannon Entropy and digit ratios.

* 🔍 Keyword Scanner: Identifies 150+ high-risk triggers like KYC, Urgency, Refund, and Govt Schemes.

* 🏆 Optimized XGBoost: Tuned for maximum AUC (0.99+) to minimize false positives in real-time.


## 📁 Project Structure
```bash
fraud-upi_id-detection/
├── 🧠 train.py                →  Data generation + model training
├── 🖥️ app.py                  →  CLI prediction app  
├── 📦 upi_fraud_model.pkl     →  Pre-trained XGBoost model
├── 📋 feature_columns.pkl     →  Saved feature names
└── 📄 requirements.txt        →  All dependencies
````


## 🛠️ Tech Stack





| Layer | Technology |
|---|---|
| Language | Python 3.8+ |
| ML Model | XGBoost |
| Balancing | SMOTE |
| Data Gen | Faker (Indian locale) |
| Processing | Pandas, NumPy |
| Serialization | Joblib |



---





## 👨‍💻 Built By

**Farhan Farooqui**
*Hackathon Project — Fraud UPI & Digital Payments Detection*

<br/>


