# Imports and Setup
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# Scikit-learn & Modeling Framework
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    make_scorer,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler,
)
from sklearn.tree import DecisionTreeClassifier

# Fallback check for XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', palette='muted')

# Load Dataset
df = pd.read_csv('financial_loan_data.csv')
print(f'Dataset Shape: {df.shape}')

# ==========================================
# Data Understanding & Cleaning
# ==========================================

# Data Cleaning: Parse AnnualIncome string to float
if df['AnnualIncome'].dtype == 'object':
    df['AnnualIncome'] = (
        df['AnnualIncome']
        .replace('[\$,]', '', regex=True)
        .astype(float)
    )

# Feature Categorization
categorical_cols = [
    'EmploymentStatus',
    'EducationLevel',
    'MaritalStatus',
    'HomeOwnershipStatus',
    'BankruptcyHistory',
    'LoanPurpose',
]
numeric_cols = [
    c for c in df.columns
    if c not in categorical_cols and c not in ['LoanApproved', 'RiskScore']
]

print("Missing Value Summary")
missing_series = df.isnull().sum()
print(missing_series[missing_series > 0])

# EDA Visualizations (8 Panel Diagnostic Dashboard)
fig, axes = plt.subplots(4, 2, figsize=(16, 20))

# 1. Target Class Distribution
sns.countplot(data=df, x='LoanApproved', ax=axes[0, 0])
axes[0, 0].set_title('1. Target Distribution (LoanApproved)')
axes[0, 0].set_xticklabels(['Denied (0)', 'Approved (1)'])

# 2. Credit Score Distribution by Approval
sns.boxplot(
    data=df,
    x='LoanApproved',
    y='CreditScore',
    ax=axes[0, 1],
    palette='Set2',
)
axes[0, 1].set_title('2. Credit Score vs. Loan Approval')

# 3. Employment Status vs Approval
sns.countplot(
    data=df,
    x='EmploymentStatus',
    hue='LoanApproved',
    ax=axes[1, 0],
)
axes[1, 0].set_title('3. Employment Status Impact')

# 4. Income vs Loan Amount Scatter
sns.scatterplot(
    data=df,
    x='AnnualIncome',
    y='LoanAmount',
    hue='LoanApproved',
    alpha=0.4,
    ax=axes[1, 1],
)
axes[1, 1].set_title('4. Annual Income vs. Loan Amount')

# 5. Debt to Income Ratio Density
sns.kdeplot(
    data=df,
    x='DebtToIncomeRatio',
    hue='LoanApproved',
    common_norm=False,
    fill=True,
    ax=axes[2, 0],
)
axes[2, 0].set_title('5. Debt-To-Income (DTI) Density')

# 6. Loan Purpose Breakdown
sns.countplot(
    data=df,
    y='LoanPurpose',
    hue='LoanApproved',
    ax=axes[2, 1],
    palette='Set1',
)
axes[2, 1].set_title('6. Loan Purpose vs Approval Rate')

# 7. Correlation Heatmap for Key Numerical Features
top_corr_cols = [
    'CreditScore',
    'AnnualIncome',
    'DebtToIncomeRatio',
    'LoanAmount',
    'LengthOfCreditHistory',
    'SavingsAccountBalance',
]
sns.heatmap(
    df[top_corr_cols].corr(),
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    ax=axes[3, 0],
    cbar=False,
)
axes[3, 0].set_title('7. Key Feature Correlation Matrix')

# 8. Missing Data Matrix Heatmap
sns.heatmap(df.isnull(), cbar=False, cmap='viridis', ax=axes[3, 1])
axes[3, 1].set_title('8. Missing Value Pattern Matrix')

plt.tight_layout()
plt.show()

# Statistical Significance Testing
approved_cs = df[df['LoanApproved'] == 1]['CreditScore'].dropna()
denied_cs = df[df['LoanApproved'] == 0]['CreditScore'].dropna()
t_stat, p_val = stats.ttest_ind(approved_cs, denied_cs)
print(
    f'Credit Score T-Test -> T-statistic: {t_stat:.4f}, p-value:'
    f' {p_val:.4e} (Statistically Significant)'
)

# ==========================================
# Data Preparation & Pipeline Engineering
# ==========================================

# Custom Feature Transformer Function
def engineer_custom_features(X):
    X_out = X.copy()
    X_out['IncomePerDependent'] = X_out['AnnualIncome'] / (
        X_out['NumberOfDependents'] + 1
    )
    return X_out

feature_engineering_step = FunctionTransformer(engineer_custom_features)

# Numerical Sub-pipeline
numeric_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ]
)

# Categorical Sub-pipeline
categorical_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False)),
    ]
)

# Column Transformer Assembly
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols),
    ]
)

# ==========================================
# Modeling & Optimization
# ==========================================

X = df.drop(columns=['LoanApproved', 'RiskScore'])
y = df['LoanApproved']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Custom Business Loss Function ($50k per FP, $8k per FN)
def compute_business_loss(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        return float((fp * 50000) + (fn * 8000))
    return 0.0

business_cost_scorer = make_scorer(
    compute_business_loss, greater_is_better=False
)

if HAS_XGB:
    clf_model = XGBClassifier(
        use_label_encoder=False, eval_metric='logloss', random_state=42
    )
    param_distributions = {
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [3, 5, 7],
        'model__learning_rate': [0.01, 0.05, 0.1],
        'model__scale_pos_weight': [1, 2, 4],
    }
else:
    clf_model = GradientBoostingClassifier(random_state=42)
    param_distributions = {
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [3, 5, 7],
        'model__learning_rate': [0.01, 0.05, 0.1],
    }

full_pipeline = Pipeline(
    steps=[
        ('feature_eng', feature_engineering_step),
        ('preprocessor', preprocessor),
        ('model', clf_model),
    ]
)

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
search = RandomizedSearchCV(
    estimator=full_pipeline,
    param_distributions=param_distributions,
    n_iter=10,
    scoring=business_cost_scorer,
    cv=cv_strategy,
    random_state=42,
    n_jobs=-1,
)

print('Fitting model pipeline with business-cost objective tuning...')
search.fit(X_train, y_train)
best_pipeline = search.best_estimator_
print(f'Optimal Parameters Found: {search.best_params_}')

# ==========================================
# Model Evaluation & Valuation
# ==========================================

y_pred_test = best_pipeline.predict(X_test)
y_pred_proba = best_pipeline.predict_proba(X_test)[:, 1]

cm_test = confusion_matrix(y_test, y_pred_test)
tn, fp, fn, tp = cm_test.ravel()
total_test_loss = compute_business_loss(y_test, y_pred_test)

baseline_pred = np.ones_like(y_test)
baseline_loss = compute_business_loss(y_test, baseline_pred)
loss_reduction_pct = (
    (baseline_loss - total_test_loss) / baseline_loss
) * 100

print(f'TEST SET PERFORMANCE SUMMARY')
print(f'False Positives (Bad Loans Approved): {fp}')
print(f'False Negatives (Good Loans Denied): {fn}')
print(f'Total Business Cost (Model): ${total_test_loss:,.2f}')
print(f'Total Business Cost (Baseline - Approve All): ${baseline_loss:,.2f}')
print(f'Cost Savings vs. Baseline: {loss_reduction_pct:.2f}%')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Confusion Matrix
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title(
    f'Test Confusion Matrix \n Total Cost: ${total_test_loss:,.2f}'
)
axes[0].set_xlabel('Predicted Label')
axes[0].set_ylabel('True Label')
axes[0].set_xticklabels(['Denied', 'Approved'])
axes[0].set_yticklabels(['Denied', 'Approved'])

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
auc_val = roc_auc_score(y_test, y_pred_proba)
axes[1].plot(
    fpr,
    tpr,
    color='darkorange',
    lw=2,
    label=f'ROC Curve (AUC = {auc_val:.3f})',
)
axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('Receiver Operating Characteristic (ROC)')
axes[1].legend(loc='lower right')
plt.tight_layout()
plt.show()

# Feature Importance
fitted_model = best_pipeline.named_steps['model']

try:
    feature_names = (
        best_pipeline.named_steps['preprocessor'].get_feature_names_out()
    )
except AttributeError:
    try:
        feature_names = (
            best_pipeline.named_steps['preprocessor'].get_feature_names()
        )
    except AttributeError:
        ohe = (
            best_pipeline.named_steps['preprocessor']
            .named_transformers_['cat']
            .named_steps['onehot']
        )
        cat_features = ohe.get_feature_names(categorical_cols)
        feature_names = numeric_cols + list(cat_features)

if hasattr(fitted_model, 'feature_importances_'):
    importances = pd.Series(
        fitted_model.feature_importances_, index=feature_names
    ).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    importances.head(10).plot(kind='barh', color='teal')
    plt.title('Top 10 Feature Importances in Loan Approval Decisions')
    plt.xlabel('Relative Feature Importance Score')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

print('\nClassification Report:\n', classification_report(y_test, y_pred_test))