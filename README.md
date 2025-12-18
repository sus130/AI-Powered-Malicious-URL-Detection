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

**BACKEND **

<img width="885" height="497" alt="image" src="https://github.com/user-attachments/assets/c7ab95d2-eeb4-4085-a95d-2bda4291de28" />
<img width="971" height="544" alt="image" src="https://github.com/user-attachments/assets/fff1a3ea-2de0-4a44-87f1-5884af0f44c3" />

**MLFLOW**

<img width="1080" height="522" alt="image" src="https://github.com/user-attachments/assets/1d6d7fc4-848b-447f-9d0d-56929e496a65" />

**MONGODB ATLAS**

<img width="1911" height="927" alt="image" src="https://github.com/user-attachments/assets/eb26108d-8da9-4f7d-ae19-caf5cef76f83" />

**AMAZON S3**

<img width="882" height="576" alt="image" src="https://github.com/user-attachments/assets/37aa1bf4-1fb5-40da-a96f-d385daf8093d" />
<img width="890" height="428" alt="image" src="https://github.com/user-attachments/assets/083d028e-1e33-429d-8122-74b530a36946" />









