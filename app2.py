# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# Ensure specific versions of libraries are installed
!pip uninstall -y scikit-learn
!pip install scikit-learn==1.5.2

import xgboost
import sklearn
print(xgboost.__version__)  # Check XGBoost version
print(sklearn.__version__)  # Check Scikit-learn version

# Load dataset from a CSV file
data = pd.read_csv('/content/loan_data_set.csv')

# Step 1: Data Cleaning
# Remove irrelevant column if present
if 'Loan_ID' in data.columns:
    data = data.drop('Loan_ID', axis=1)

# Fill missing values for categorical and numerical columns
categorical_cols = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History']
numerical_cols = ['LoanAmount', 'Loan_Amount_Term']
for col in categorical_cols:
    data[col] = data[col].fillna(data[col].mode()[0])  # Fill missing categorical values with mode
for col in numerical_cols:
    data[col] = data[col].fillna(data[col].median())  # Fill missing numerical values with median

# Step 2: Data Preprocessing
# Encode categorical variables into numeric format
label_encoders = {}
for col in data.select_dtypes(include='object').columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Step 3: Feature Engineering
# Create new features like total income and debt-to-income ratio
data['Total_Income'] = data['ApplicantIncome'] + data['CoapplicantIncome']
data['Debt_To_Income_Ratio'] = data['LoanAmount'] / (data['Total_Income'] + 1e-6)
data = data.drop(['ApplicantIncome', 'CoapplicantIncome'], axis=1)

# Step 4: Outlier Removal
# Remove outliers from numerical columns using the IQR method
for col in ['LoanAmount', 'Total_Income']:
    q1 = data[col].quantile(0.25)
    q3 = data[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]

# Step 5: Normalize the dataset
# Scale numerical columns to have zero mean and unit variance
scaler = StandardScaler()
data[['LoanAmount', 'Loan_Amount_Term', 'Total_Income', 'Debt_To_Income_Ratio']] = scaler.fit_transform(
    data[['LoanAmount', 'Loan_Amount_Term', 'Total_Income', 'Debt_To_Income_Ratio']]
)

# Step 6: Data Transformation
# Split dataset into features (X) and target variable (y)
X = data.drop('Loan_Status', axis=1)
y = data['Loan_Status']

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Step 7: Model Building with Hyperparameter Tuning
# Define parameter grid for XGBoost
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [3, 6],
    'learning_rate': [0.01, 0.1],
}

# Initialize XGBoost classifier
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# Perform grid search with cross-validation
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=StratifiedKFold(n_splits=5), scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Retrieve the best model from grid search
best_model = grid_search.best_estimator_

# Evaluate model using cross-validation
cv_scores = cross_val_score(best_model, X_train, y_train, cv=StratifiedKFold(n_splits=5), scoring='accuracy')
print("Cross-validation Accuracy:", np.mean(cv_scores))

# Make predictions and evaluate model performance
y_pred = best_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Function to predict loan eligibility based on user input
def predict_loan_eligibility():
    print("Enter the following details:")
    user_input = {
        'Gender': input("Gender (Male/Female): "),
        'Married': input("Married (Yes/No): "),
        'Dependents': input("Dependents (0/1/2/3+): "),
        'Education': input("Education (Graduate/Not Graduate): "),
        'Self_Employed': input("Self Employed (Yes/No): "),
        'ApplicantIncome': float(input("Applicant Income: ")),
        'CoapplicantIncome': float(input("Coapplicant Income: ")),
        'LoanAmount': float(input("Loan Amount: ")),
        'Loan_Amount_Term': float(input("Loan Amount Term: ")),
        'Credit_History': float(input("Credit History (1.0/0.0): ")),
        'Property_Area': input("Property Area (Urban/Semiurban/Rural): ")
    }

    # Convert user input to DataFrame
    input_df = pd.DataFrame([user_input])

    # Encode categorical variables using pre-trained label encoders
    for col, le in label_encoders.items():
        if col in input_df.columns:
            input_df[col] = le.transform(input_df[col])

    # Compute additional features
    input_df['Total_Income'] = input_df['ApplicantIncome'] + input_df['CoapplicantIncome']
    input_df['Debt_To_Income_Ratio'] = input_df['LoanAmount'] / (input_df['Total_Income'] + 1e-6)
    input_df = input_df.drop(['ApplicantIncome', 'CoapplicantIncome'], axis=1)

    # Ensure feature alignment with training data
    input_df = input_df.reindex(columns=X.columns, fill_value=0)

    # Scale numerical features
    num_features = ['LoanAmount', 'Loan_Amount_Term', 'Total_Income', 'Debt_To_Income_Ratio']
    input_df[num_features] = scaler.transform(input_df[num_features])

    # Predict loan eligibility
    prediction = best_model.predict(input_df)
    return "Loan Approved" if prediction[0] == 1 else "Loan Denied"

# Prompt user for input and display prediction result
result = predict_loan_eligibility()
print("Prediction:", result)
