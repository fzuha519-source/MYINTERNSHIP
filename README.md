# MYINTERNSHIP
AI Internship Project — InternSpark

## Overview
Churn prediction system built using Machine Learning,
deployed as a REST API with explainability analysis.

## Goals Completed
- ✅ Goal 1 — Supervised Classification (Logistic Regression vs Random Forest)
- ✅ Goal 3 — FastAPI Deployment + Docker
- ✅ Goal 4 — SHAP Explainability & Bias Analysis

## Dataset
Telco Customer Churn Dataset (Kaggle)
7,043 customers | 20 features | Binary classification

## Results
| Model | Accuracy | F1 Score | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 81.55% | 62.43% | 0.86 |
| Random Forest | 79.56% | 55.00% | 0.84 |

**Winner: Logistic Regression** ✅

## Key Findings
- Contract type is the #1 churn driver
- Low tenure customers churn most
- Senior citizens show 8% accuracy bias
- API predicts churn in real time with probability score

## How to Run the API
```bash
cd goal3_deployment
pip install fastapi uvicorn scikit-learn numpy
uvicorn app:app --reload
```
Open http://127.0.0.1:8000/docs

## Tools Used
Python | Pandas | Scikit-learn | SHAP | FastAPI | Docker
