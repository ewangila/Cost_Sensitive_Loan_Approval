# Cost-Sensitive Loan Approval Pipeline

**Business-driven machine learning for credit decisions — optimizing financial impact rather than classification accuracy alone.**

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-111111?style=flat&logo=xgboost&logoColor=white)
![Framework](https://img.shields.io/badge/Framework-CRISP--DM-orange?style=flat)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## Project Overview

Loan approval decisions involve balancing financial risk against the opportunity to extend credit to eligible applicants. Traditional classification approaches often optimize metrics such as accuracy without accounting for the unequal financial consequences of incorrect decisions.

This project develops an end-to-end **cost-sensitive machine learning pipeline** for loan approval assessment, using 20,000 historical loan applications.

Framed around a financial services business scenario, FinTech Innovations seeks to improve credit assessment by reducing manual review bottlenecks, promoting consistent decision criteria, and minimizing the financial impact of classification errors.

Rather than treating all misclassifications equally, the pipeline incorporates an asymmetric business cost matrix to align model evaluation and decision thresholds with the financial consequences of lending decisions.

### The Business Problem

Two classification errors have different financial consequences:

| Error Type | Description | Assumed Financial Cost |
|---|---|---:|
| **False Positive (FP)** | Approving an applicant who subsequently defaults. | **$50,000** |
| **False Negative (FN)** | Rejecting an applicant who would have repaid the loan. | **$8,000** |

### Financial Risk Asymmetry

The assumed cost ratio is:

\[
\text{Cost Ratio} = \frac{50,000}{8,000} = 6.25
\]

An incorrect approval is therefore modeled as 6.25 times more expensive than rejecting a creditworthy applicant.

This asymmetry makes standard accuracy optimization insufficient as the sole decision criterion.

**The central objective is to minimize expected financial error cost while maintaining an appropriate balance between credit risk and access to credit.**

*Note: The financial cost estimates represent project assumptions and should be validated against actual portfolio loss data and business stakeholder requirements before operational use.*

---

## Key Objectives

The project focuses on five business and technical objectives:

1. **Minimize Financial Error Cost**

   Optimize classification decisions using the asymmetric business cost function:

   \[
   \text{Total Cost} = (50,000 \times FP) + (8,000 \times FN)
   \]

2. **Optimize Decision Thresholds**

   Move beyond the default classification threshold to identify decision thresholds that minimize estimated financial cost.

3. **Prioritize Precision Where Appropriate**

   Use the F-beta score with \(\beta = 0.5\) as a supplementary metric, placing greater emphasis on precision than recall.

4. **Benchmark Multiple Classification Algorithms**

   Compare linear, tree-based, ensemble, and gradient-boosting approaches.

5. **Establish Measurable Business Value**

   Target at least a **30% reduction in financial error cost** relative to an unweighted baseline.

   This is a project success target, not an assumed or reported achievement.

---

## Methodology: CRISP-DM

The project follows the **Cross-Industry Standard Process for Data Mining (CRISP-DM)** framework to connect business requirements with analytical implementation and deployment recommendations.

| Phase | Activities |
|---|---|
| **1. Business Understanding** | Define the lending problem, stakeholder requirements, asymmetric cost matrix, and success criteria. |
| **2. Data Understanding** | Audit data quality, examine feature distributions, investigate missing values, and perform exploratory data analysis. |
| **3. Data Preparation** | Parse financial variables, handle missing values, engineer features, and construct preprocessing pipelines. |
| **4. Modeling** | Train and compare Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, and XGBoost models. |
| **5. Evaluation** | Assess predictive performance, calculate business costs, optimize decision thresholds, and analyze feature importance. |
| **6. Deployment Recommendations** | Define decision policies, human oversight requirements, monitoring strategies, and operational rollout recommendations. |

---

## Modeling Approach

The pipeline evaluates several classification algorithms to investigate their ability to identify creditworthy applicants and manage default risk.

### Models

| Algorithm | Purpose |
|---|---|
| **Logistic Regression** | Establishes an interpretable linear classification baseline. |
| **Decision Tree** | Captures nonlinear relationships through rule-based classification. |
| **Random Forest** | Uses an ensemble of decision trees to improve predictive robustness. |
| **Gradient Boosting** | Builds sequential tree-based models to improve predictive performance. |
| **XGBoost** | Applies optimized gradient boosting for structured classification problems. |

Hyperparameter search is incorporated into the modeling workflow to explore suitable model configurations.

### Cost-Sensitive Decision Making

A conventional classifier typically assigns a default decision threshold of 0.5.

However, this threshold does not necessarily minimize financial loss when false positives and false negatives have different costs.

The pipeline investigates alternative thresholds based on the business cost function.

For each candidate threshold:

1. Generate predicted class probabilities.
2. Convert probabilities into binary classification decisions.
3. Calculate false positives and false negatives.
4. Apply the financial cost matrix.
5. Compare total estimated costs across thresholds.
6. Select a threshold using validation data.

The intended decision objective is:

\[
t^* = \arg\min_t \left[50,000 \times FP(t) + 8,000 \times FN(t)\right]
\]

Where:

- \(t^*\) represents the selected decision threshold.
- \(FP(t)\) represents false positives at threshold \(t\).
- \(FN(t)\) represents false negatives at threshold \(t\).

The final threshold should be evaluated on an independent test set to obtain an unbiased estimate of performance.

---

## Evaluation Framework

The evaluation framework combines traditional classification metrics with financial performance indicators.

| Metric | Purpose | Importance |
|---|---|---|
| **Business Cost** | Quantifies the financial impact of incorrect predictions. | Primary optimization objective. |
| **F-beta Score (\(\beta=0.5\))** | Places greater emphasis on precision than recall. | Supplementary classification metric. |
| **Precision** | Measures the proportion of predicted positive cases that are actually positive. | Important for limiting costly false approvals, depending on target encoding. |
| **Recall** | Measures the proportion of actual positive cases correctly identified. | Helps assess missed cases. |
| **ROC-AUC** | Measures discrimination across classification thresholds. | Threshold-independent model comparison. |
| **PR-AUC** | Evaluates the precision-recall trade-off. | Particularly useful when classes are imbalanced. |
| **Confusion Matrix** | Shows true positives, true negatives, false positives, and false negatives. | Supports error analysis. |
| **Cost Reduction** | Measures improvement over the unweighted baseline. | Business success indicator. |

### Primary Business Success Metric

The target is to reduce financial error cost by at least 30% relative to the baseline.

\[
\text{Cost Reduction (\%)} =
\frac{C_{\text{baseline}}-C_{\text{optimized}}}
{C_{\text{baseline}}}
\times 100
\]

Where:

- \(C_{\text{baseline}}\) is the baseline model's estimated financial cost.
- \(C_{\text{optimized}}\) is the cost of the optimized decision policy.

The comparison should use the same evaluation dataset and cost assumptions.

**Business cost takes priority over accuracy when assessing whether the optimized decision policy delivers financial value.**


---

## Project Structure

```text
Cost_Sensitive_Loan_Approval/
│
├── data/
│   ├── financial_loan_data.csv
│   └── Loan.csv
│
├── financial_loan_risk.ipynb
├── pipeline.py
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

### File Descriptions

| File or Directory | Description |
|---|---|
| `data/` | Contains the loan application datasets. |
| `financial_loan_data.csv` | Historical financial loan application data. |
| `Loan.csv` | Additional loan-related dataset used in the project. |
| `financial_loan_risk.ipynb` | Main notebook containing the CRISP-DM workflow, exploration, modeling, and evaluation. |
| `pipeline.py` | Python script for executing the consolidated modeling workflow. |
| `requirements.txt` | Lists the project's Python dependencies. |
| `.gitignore` | Specifies files and directories Git should ignore. |
| `LICENSE` | Defines the project's licensing terms. |
| `README.md` | Project documentation. |

---

## Installation and Setup



### 1. Clone the Repository

```bash
git clone https://github.com/ewangila/Cost_Sensitive_Loan_Approval.git
```

Navigate to the project directory:

```bash
cd Cost_Sensitive_Loan_Approval
```

### 2. Create a Virtual Environment

Creating an isolated environment helps prevent dependency conflicts.

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

Upgrade pip and install the required packages:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Prepare the Data

Ensure the required CSV files are located inside the `data/` directory:

```text
data/
├── financial_loan_data.csv
└── Loan.csv
```

Check that the dataset paths referenced in the notebook and Python script match the actual file locations.

### 5. Run the Project

**Option A: Jupyter Notebook**

Start Jupyter:

```bash
jupyter notebook
```

Open:

```text
financial_loan_risk.ipynb
```

Run the notebook cells in sequence.

**Option B: Python Script**

Execute the consolidated pipeline:

```bash
python pipeline.py
```

The available execution steps and generated outputs depend on the implementation in `pipeline.py`.

---

## Responsible Lending and Model Governance

Credit decisions can have significant financial and personal consequences. Therefore, predictive performance alone is insufficient for determining whether a lending model is suitable for operational use.

Important considerations include:

- **Model interpretability:** Provide explanations of relevant model features and decision factors.
- **Fairness and bias:** Assess whether model errors or approval patterns disproportionately affect protected or vulnerable groups.
- **Human oversight:** Treat model predictions as decision-support outputs rather than automatically issuing final lending decisions.
- **Data privacy:** Protect sensitive applicant information and comply with applicable data protection requirements.
- **Model monitoring:** Monitor changes in applicant characteristics, prediction distributions, default rates, and financial outcomes.
- **Threshold governance:** Document the assumptions and business rationale behind the selected decision threshold.
- **Cost validation:** Reassess the financial cost estimates as portfolio conditions and lending policies change.

The pipeline is an analytical prototype and should undergo appropriate validation, governance, security, and regulatory review before any production lending application.

---

## Assumptions and Limitations

### Financial Cost Assumptions

The $50,000 false-positive cost and $8,000 false-negative cost are project assumptions. Actual lending losses and opportunity costs can vary substantially across applicants, loan products, and portfolio conditions.

### Historical Data

Historical application records may contain patterns that do not represent future applicant populations or changing economic conditions.

### Classification Trade-offs

Optimizing financial cost does not automatically guarantee fair or appropriate credit decisions. A threshold that minimizes modeled cost may create undesirable approval or rejection patterns.

### Model Generalization

Performance on historical test data does not guarantee future performance. Independent validation and ongoing monitoring are necessary.

### Probability Calibration

Threshold optimization assumes that predicted probabilities are sufficiently meaningful for the intended decision process. Probability calibration should be assessed before relying heavily on probability-based decisions.

---

## Potential Future Enhancements

Potential extensions to the project include:

- **Explainable AI:** Integrate SHAP or permutation importance to investigate model predictions and feature contributions.
- **Probability Calibration:** Compare calibration methods to improve the reliability of predicted probabilities.
- **Fairness Auditing:** Evaluate model performance across relevant applicant subgroups.
- **Model Monitoring:** Implement drift detection and performance monitoring for changing credit portfolios.
- **Interactive Decision Dashboard:** Build a Streamlit application for exploring predictions, thresholds, confusion matrices, and financial costs.
- **Scenario Analysis:** Investigate how changing false-positive and false-negative costs affects the optimal decision policy.
- **Model Experiment Tracking:** Record model configurations, thresholds, and evaluation results for reproducibility.

These are proposed enhancements rather than claims about functionality already implemented.

---

## 📄 License

This project is released under the **MIT License**. See the [`LICENSE`](LICENSE) file for the full terms.

---

## Author

**Eugin Wangila**

Location: Nairobi, Kenya

---

## Project Summary

The Cost-Sensitive Loan Approval Pipeline demonstrates how machine learning can be aligned with real business objectives by incorporating financial consequences into model evaluation and decision-making.

By combining **CRISP-DM, predictive modeling, asymmetric cost optimization, and threshold analysis**, the project explores an important distinction between building an accurate classifier and developing a financially informed decision-support system.

The central principle is simple:

**A model should not just predict well. It should help organizations make better decisions.**
````
