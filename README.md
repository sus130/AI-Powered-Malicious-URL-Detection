**Overview**

Phishing and malware URLs are becoming increasingly sophisticated, making manual or rule-based detection ineffective.
This project presents a Machine Learning–based URL Classification System that categorizes URLs as Safe, Suspicious, or Malicious. The system supports:

Real-time URL prediction (Streamlit)

Batch prediction via CSV (FastAPI)

Automated MLOps using MLflow

Model retraining and experiment tracking

**Objectives**

Build a machine learning model to classify URLs as Safe / Suspicious / Malicious

Provide real-time prediction using Streamlit

Enable bulk URL prediction using FastAPI

Implement MLOps pipelines for local model retraining

Track experiments and metrics using MLflow (local)

**Scope**

Real-time and batch URL classification

Entirely local (datasets, models, logs)

Suitable for individuals, organizations, ISPs

Extendable for firewalls, email security, corporate filters

**Dataset Details**

11,055 URLs (kaggle dataset)

30 structural, content, and domain-based features

Balanced: 6,157 legitimate & 4,898 phishing

**Model Training**

Models trained:

Logistic Regression

Decision Tree

Random Forest

Gradient Boosting

XGBoost
Evaluation metrics: Accuracy, precision, recall, F1
Best models: Random Forest & Decision Tree

**Tools & Technologies**

Python

Pandas, NumPy, Scikit-learn

Matplotlib, Seaborn

Streamlit (frontend)

FastAPI (backend)

MLflow (experiment tracking)

MongoDB Atlas (logging)

Amazon S3 / local storage

Google Colab for training

**system architecture** 

<img width="1932" height="777" alt="image" src="https://github.com/user-attachments/assets/102d48e1-825f-4bff-8fe0-a8ceb0ab79a6" />

**FRONTEND **

<img width="865" height="475" alt="image" src="https://github.com/user-attachments/assets/bfbe5b9e-fdc6-49cc-9d7d-490406bead30" />
<img width="890" height="475" alt="image" src="https://github.com/user-attachments/assets/89605157-6552-454b-8df7-fd8149d22795" />
