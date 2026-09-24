# Cost-Sensitive Loan Approval Pipeline

> **Business-driven machine learning** for loan decisions — optimizing real financial impact instead of accuracy.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/Framework-CRISP--DM-orange.svg)](#methodology)

---

## Overview

FinTech Innovations partners with traditional banks to accelerate credit assessment. Manual review by loan officers creates bottlenecks, inconsistent standards, and slow turnaround.

This project builds an end-to-end **cost-sensitive classification pipeline** on 20,000 historical loan applications. Instead of maximizing accuracy, the model aligns decision boundaries with the true business costs of errors:

| Error Type | Description | Cost |
|------------|-------------|------|
| **False Positive** | Approving a loan that defaults | **$50,000** |
| **False Negative** | Rejecting a creditworthy applicant | **$8,000** |

**Cost Ratio** = $50k / $8k = **6.25×**  
Approving a bad loan is more than six times as expensive as rejecting a good one. Standard accuracy metrics treat every error equally and are therefore inadequate.

---

## Key Objectives

1. Minimize total portfolio cost:  
   `Cost = 50,000 × FP + 8,000 × FN`
2. Optimize for precision-focused metrics (F-β with β = 0.5)
3. Deliver at least a **30% reduction** in financial error cost versus an unweighted baseline
4. Produce an interpretable, production-ready decision engine for loan officers

---

## Methodology (CRISP-DM)

| Phase | Focus |
|-------|-------|
| **1. Business Understanding** | Cost matrix, stakeholder needs, success criteria |
| **2. Data Understanding & EDA** | Quality audit, missing-value analysis, diagnostic visualizations |
| **3. Data Preparation** | Currency parsing, imputation, feature engineering, preprocessing pipelines |
| **4. Modeling** | Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost + hyperparameter search |
| **5. Evaluation** | Custom cost metric, ROC-AUC, PR-AUC, threshold optimization, feature importance |
| **6. Deployment Recommendations** | Threshold policy, monitoring, and operational rollout guidance |

---

## Tech Stack

- **Python 3.9+**
- **pandas**, **numpy**, **scipy**
- **scikit-learn** (pipelines, preprocessing, models, metrics)
- **XGBoost**
- **matplotlib**, **seaborn** (EDA & reporting)
- **Jupyter Notebook**

---

## Project Structure

```text
Cost_Sensitive_Loan_Approval/
├── data/                          # Raw & processed datasets (git-ignored)
│   ├── financial_loan_data.csv
│   └── Loan.csv
├── financial_loan_risk.ipynb      # Full CRISP-DM pipeline & analysis
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
