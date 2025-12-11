# Customer Churn Analysis Machine Learning

## 📊 Project Overview
This project performs a comprehensive machine learning analysis to predict customer churn in a telecommunications company using the Telco Customer Churn dataset. The analysis follows a structured learning approach with clear day-by-day sections, starting from basic data exploration through advanced feature engineering and preprocessing. The project aims to identify key factors influencing customer churn and build a predictive model to help reduce customer attrition.

**Key Objective:** Predict which customers are likely to leave the service so that targeted retention strategies can be implemented.

## 📁 Project Structure
```
Customer Churn Analysis Machine Learning/
├── customer_churn_anaylsis.ipynb      # Main Jupyter notebook with complete analysis
├── Telco-Customer-Churn.csv           # Dataset containing customer information
└── README.md                          # This file
```

## 📈 Dataset
**Source:** Telco Customer Churn dataset  
**Size:** 7,043 customers (cleaned from 7,042 original records)  
**Features:** 20+ customer attributes  
**Target:** Churn (Yes/No) - indicates whether customer left the company

### Key Attributes:
- **Tenure:** Number of months customer has been with the company
- **Monthly Charges:** Monthly billing amount
- **Total Charges:** Cumulative charges over customer lifetime
- **Services:** Internet, phone, streaming, security, backup, etc.
- **Contract Type:** Month-to-month, one year, two year
- **Payment Method:** Electronic check, mailed check, bank transfer, credit card
- **Demographics:** Senior citizen status, dependents, partner information

### Data Quality:
- **Initial Records:** 7,042 customers
- **Final Records:** 7,032 customers (after cleaning)
- **Records Removed:** 10 rows with missing `TotalCharges` values
- **Missing Data:** Handled by converting to NaN and dropping affected rows

## 🔧 Technologies & Libraries
- **Python 3.x** - Primary programming language
- **pandas** - Data manipulation, cleaning, and analysis
- **numpy** - Numerical computations and array operations
- **matplotlib** - Static data visualization and plotting
- **scikit-learn** - Machine learning utilities:
  - `train_test_split` - Split data into training and testing sets
  - `ColumnTransformer` - Transform features with different pipelines
  - `OneHotEncoder` - Encode categorical variables
- **imblearn (imbalanced-learn)** - Handle class imbalance:
  - `SMOTE` - Synthetic Minority Over-sampling Technique
- **shap** - Explain model predictions and identify important features
- **Jupyter Notebook / VS Code** - Interactive development environment

## 📋 Analysis Pipeline

### Day 2: Data Import & Exploration
**Objective:** Load and explore the dataset to understand its structure and characteristics
- Import essential libraries (pandas, numpy, matplotlib, sklearn, imblearn, shap)
- Load the Telco customer churn dataset from CSV file
- Check dataset shape: (7,042 rows × 21 columns)
- Explore data types and value distributions
- Analyze class distribution using `value_counts()`
- Example: Check `InternetService` distribution

**Key Code:**
```python
dataset = pd.read_csv("Telco-Customer-Churn.csv")
print(f"Dataset shape: {dataset.shape}")
dataset['InternetService'].value_counts(normalize=True) * 100
```

### Day 3: Data Cleaning
**Objective:** Identify and handle data quality issues, missing values, and inconsistencies
- Detect erroneous data by calculating statistical means:
  - `tenure` mean: ~32 months
  - `MonthlyCharges` mean: ~65
- Identify blank/space entries in `TotalCharges` column (10 occurrences found)
- Convert invalid string entries to NaN using `pd.to_numeric()`
- Remove rows with missing `TotalCharges` values
- Final clean dataset: 7,032 rows (10 rows removed)

**Key Code:**
```python
dataset['TotalCharges'] = pd.to_numeric(dataset['TotalCharges'], errors='coerce')
dataset = dataset.dropna(subset=['TotalCharges']).copy()
```

### Day 4: Data Preprocessing (Part 1)
**Objective:** Prepare data for machine learning by separating features and target, handling encoding

#### Step 1: Remove Unnecessary Columns
- Drop `customerID` (unique identifier, not predictive)
- Separate target variable `Churn` from features
- Features (X): 19 columns remaining
- Target (y): Single binary column

#### Step 2: Prepare Target Variable
- Map `Churn` from categorical (Yes/No) to numerical (1/0)
- Yes → 1 (churned)
- No → 0 (retained)

#### Step 3: Train-Test Split
- Split ratio: 80% training, 20% testing
- Use `stratify=y` to maintain class distribution
- Random state: 42 (for reproducibility)
- Training set: ~5,600 samples
- Testing set: ~1,400 samples
- Class distribution preserved in both sets (~27% churn, ~73% retention)

**Key Code:**
```python
X = dataset.drop(columns=["customerID", "Churn"])
y = dataset['Churn'].map({'No': 0, 'Yes': 1})
X_trainset, X_testset, y_trainset, y_testset = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

### Day 5: Feature Engineering (Parts 3-4)
**Objective:** Encode categorical variables and prepare final feature matrix for modeling

#### Step 1: Identify Numerical & Categorical Columns
- **Numerical Columns:** tenure, MonthlyCharges, TotalCharges (3 columns)
- **Categorical Columns:** Gender, SeniorCitizen, Partner, Dependents, InternetService, PhoneService, etc. (16 columns)

#### Step 2: Apply One-Hot Encoding
- Use `ColumnTransformer` for flexible preprocessing pipeline
- **Categorical variables:** Apply `OneHotEncoder`
  - Handle unknown categories with 'ignore'
  - Dense output (no sparse matrices)
- **Numerical variables:** Pass through unchanged

#### Step 3: Create Feature Matrix
- Transform entire feature set X
- Generate feature names from encoded categorical columns
- Combine encoded categorical names with numerical column names
- Create final DataFrame with 40+ features

**Example Output:**
```
Numerical columns: ['tenure', 'MonthlyCharges', 'TotalCharges']
Categorical columns: ['gender', 'SeniorCitizen', 'Partner', ...]
Encoded features: ['gender_Female', 'gender_Male', 'SeniorCitizen_0', ...]
Total features after encoding: 40+
```

**Key Code:**
```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()

pre_processed = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
    ("num", "passthrough", numeric_cols)
])

X_preprocessed = pre_processed.fit_transform(X)
tr_df = pd.DataFrame(X_preprocessed, columns=all_iv_names)
```

---

## 📊 Data Transformation Summary

| Stage | Rows | Columns | Description |
|-------|------|---------|-------------|
| Raw Data | 7,042 | 21 | Original dataset |
| After Cleaning | 7,032 | 21 | 10 rows removed (missing TotalCharges) |
| Features (X) | 7,032 | 19 | After dropping customerID and Churn |
| Training Set | 5,625 | 19 | 80% stratified split |
| Test Set | 1,407 | 19 | 20% stratified split |
| After Encoding | 5,625 | 40+ | One-hot encoded categorical variables |

## 🚀 How to Use

### Prerequisites
Install required Python packages:
```bash
pip install pandas numpy matplotlib scikit-learn imbalanced-learn shap jupyter
```

Or install from a requirements file (if available):
```bash
pip install -r requirements.txt
```

### System Requirements
- Python 3.7+
- 4GB RAM minimum
- 500MB disk space for dataset and outputs

### Running the Analysis
1. **Clone or download the repository**
   ```bash
   git clone https://github.com/yourusername/customer-churn-analysis.git
   cd customer-churn-analysis
   ```

2. **Open the Jupyter Notebook**
   ```bash
   jupyter notebook customer_churn_anaylsis.ipynb
   # OR in VS Code
   code customer_churn_anaylsis.ipynb
   ```

3. **Run cells sequentially from top to bottom**
   - Each day section builds upon the previous one
   - Do not skip cells or change the execution order
   - Watch output for data quality checks and validation

4. **Monitor progress**
   - Day 2: Check dataset loads successfully (7,042 rows × 21 columns)
   - Day 3: Verify 10 rows removed with missing values (7,032 remaining)
   - Day 4: Confirm 80/20 train-test split with stratification
   - Day 5: Validate 40+ features after one-hot encoding

### Expected Outputs
- **Data statistics:** Shape, types, means, distributions
- **Data quality reports:** Missing values, blank entries
- **Processed matrices:** Feature arrays ready for ML models
- **Column mappings:** Encoded feature names and indices
- **(Future) Model metrics:** Accuracy, precision, recall, F1-score
- **(Future) Feature importance:** SHAP values and rankings

### Troubleshooting
**Error: File not found (CSV)**
- Ensure `Telco-Customer-Churn.csv` is in the same directory as notebook

**Error: Module not found**
- Run: `pip install pandas numpy matplotlib scikit-learn imbalanced-learn shap`

**Error: Memory issues**
- Dataset is small (~7MB), shouldn't cause issues on modern machines
- If persists, try closing other applications

## 📊 Key Findings & Insights

### Data Quality
- **Original dataset:** 7,042 customers
- **Data cleaning impact:** 10 rows removed (0.14% of data)
- **Root cause:** Missing/invalid values in `TotalCharges` column
- **Cleaned dataset:** 7,032 customers ready for analysis

### Dataset Characteristics
- **Class distribution:** ~27% churned, ~73% retained (imbalanced)
- **Tenure range:** 0-72 months (median ~9 months suggests high early churn)
- **Monthly charges:** Average ~$65/month
- **Total charges:** Average ~$2,283 per customer lifetime
- **Services diversity:** Customers use combination of internet, phone, streaming, security services

### Feature Insights
- **Categorical features:** Gender, contract type, internet service, payment method, etc.
- **Numerical features:** Tenure, monthly charges, total charges
- **Encoded features:** 40+ features after one-hot encoding categorical variables
- **Feature engineering:** Critical for capturing non-linear relationships

### Data Imbalance
- **Churn class:** ~27% (minority class)
- **Retention class:** ~73% (majority class)
- **Solution planned:** SMOTE (Synthetic Minority Over-sampling) to balance classes
- **Benefit:** Prevents model bias toward predicting retention

## 🎯 Next Steps & Future Development

### Phase 2: Model Selection & Training
- [ ] Apply SMOTE to training set for class balance
- [ ] Test multiple algorithms:
  - Logistic Regression (baseline)
  - Random Forest
  - Gradient Boosting (XGBoost, LightGBM)
  - Neural Networks
- [ ] Perform hyperparameter tuning
- [ ] Use cross-validation for robust evaluation

### Phase 3: Model Evaluation
- [ ] Evaluate on test set
- [ ] Calculate metrics:
  - Accuracy, Precision, Recall, F1-Score
  - ROC-AUC score
  - Confusion Matrix
- [ ] Compare models and select best performer
- [ ] Analyze prediction errors and edge cases

### Phase 4: Feature Importance & Interpretability
- [ ] Generate SHAP values for all features
- [ ] Identify top churn factors
- [ ] Create feature importance visualizations
- [ ] Understand model decision boundaries

### Phase 5: Business Insights & Recommendations
- [ ] Extract actionable insights
- [ ] Create customer segmentation
- [ ] Develop retention strategies:
  - Target high-risk customer groups
  - Personalized offers
  - Service improvement recommendations
- [ ] Present findings to stakeholders

### Phase 6: Deployment & Monitoring
- [ ] Create prediction API/service
- [ ] Deploy to production environment
- [ ] Monitor model performance
- [ ] Set up alerts for performance degradation
- [ ] Implement model retraining pipeline

---

## 📈 Expected Model Performance Targets
- **Accuracy:** >85%
- **Precision (Churn):** >75%
- **Recall (Churn):** >70%
- **F1-Score:** >72%
- **ROC-AUC:** >85%

## 📝 Notes & Important Details

### Data Pipeline Features
- **Structured Learning Approach:** Clear day-wise sections with learning progression
- **Code Comments:** Each code block is well-documented with explanations
- **Data Validation:** Checks at each stage ensure data quality
- **Reproducibility:** Fixed random_state=42 ensures consistent results
- **Stratified Splitting:** Maintains class distribution across train/test sets
- **ColumnTransformer:** Scalable approach to handle mixed data types

### Best Practices Implemented
1. **Data Cleaning:** Handles missing values appropriately
2. **Feature Engineering:** One-hot encoding for categorical variables
3. **Train-Test Separation:** Prevents data leakage
4. **Class Imbalance Awareness:** Ready for SMOTE application
5. **Modular Code:** Functions and transformers for reusability

### Files in This Project
```
Customer Churn Analysis/
├── customer_churn_anaylsis.ipynb      # Main analysis notebook (253 cells)
├── Telco-Customer-Churn.csv           # Dataset (7,042 rows × 21 columns)
├── README.md                          # This comprehensive documentation
└── requirements.txt                   # (Optional) Python dependencies
```

### Key Metrics & Numbers
| Metric | Value |
|--------|-------|
| Original Records | 7,042 |
| Cleaned Records | 7,032 |
| Records Removed | 10 (0.14%) |
| Features (Initial) | 21 |
| Features (After Cleaning) | 19 |
| Features (After Encoding) | 40+ |
| Training Samples | 5,625 (80%) |
| Testing Samples | 1,407 (20%) |
| Churn Customers | ~27% |
| Retained Customers | ~73% |

### Code Structure
- **Cells 1-10:** Initial setup and basic operations
- **Cells 11-35:** Data import and exploration
- **Cells 36-55:** Data cleaning and validation
- **Cells 56-75:** Feature preparation and target encoding
- **Cells 76-100:** Train-test splitting with stratification
- **Cells 101-253:** Feature engineering, one-hot encoding, and final preprocessing

## 👨‍💻 Author
Student - NBICT ML Program (4-1 Semester)

## 📄 License
This project is for educational purposes.

---

**Last Updated:** December 2025
