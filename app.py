import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv('loan_data_set.csv')

    # Data cleaning and preprocessing steps
    if 'Loan_ID' in data.columns:
        data = data.drop('Loan_ID', axis=1)

    categorical_cols = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History']
    numerical_cols = ['LoanAmount', 'Loan_Amount_Term']
    for col in categorical_cols:
        data[col] = data[col].fillna(data[col].mode()[0])
    for col in numerical_cols:
        data[col] = data[col].fillna(data[col].median())

    label_encoders = {}
    for col in data.select_dtypes(include='object').columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    data['Total_Income'] = data['ApplicantIncome'] + data['CoapplicantIncome']
    data = data.drop(['ApplicantIncome', 'CoapplicantIncome'], axis=1)

    scaler = StandardScaler()
    data[['LoanAmount', 'Loan_Amount_Term', 'Total_Income']] = scaler.fit_transform(
        data[['LoanAmount', 'Loan_Amount_Term', 'Total_Income']]
    )

    return data, label_encoders, scaler


# Load the data
data, label_encoders, scaler = load_data()

X = data.drop('Loan_Status', axis=1)
y = data['Loan_Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Streamlit UI
st.title("Loan Eligibility Predictor")
st.write("### Model Accuracy")
st.write(f"Accuracy: {accuracy:.2f}")

# Add a separator for clean sectioning
st.markdown("---")

# User input form for prediction
def user_input_form():
    st.sidebar.header("Enter Loan Applicant Details")
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    married = st.sidebar.selectbox("Marital Status", ["Married", "Not Married"])
    dependents = st.sidebar.selectbox("Dependents", [0, 1, 2, 3])
    education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
    applicant_income = st.sidebar.number_input("Applicant Income", min_value=0)
    coapplicant_income = st.sidebar.number_input("Coapplicant Income", min_value=0)
    loan_amount = st.sidebar.number_input("Loan Amount", min_value=0)
    loan_amount_term = st.sidebar.number_input("Loan Amount Term", min_value=0)
    credit_history = st.sidebar.selectbox("Credit History", [1.0, 0.0])
    property_area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    user_input = {
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_amount_term,
        "Credit_History": credit_history,
        "Property_Area": property_area,
    }

    return user_input

# Display form and make prediction
user_input = user_input_form()
if st.sidebar.button("Predict Loan Eligibility"):
    # Convert user input into the right format
    input_df = pd.DataFrame([user_input])

    # Encode categorical variables
    for col, le in label_encoders.items():
        if col in input_df.columns:
            if input_df[col].iloc[0] not in le.classes_:
                input_df[col] = le.transform([le.classes_[0]])  # Use a default class, or handle missing
            else:
                input_df[col] = le.transform(input_df[col])

    # Combine incomes for Total_Income feature
    input_df['Total_Income'] = input_df['ApplicantIncome'] + input_df['CoapplicantIncome']
    input_df = input_df.drop(['ApplicantIncome', 'CoapplicantIncome'], axis=1)

    # Normalize numerical features
    numerical_features = ['LoanAmount', 'Loan_Amount_Term', 'Total_Income']
    input_df[numerical_features] = scaler.transform(input_df[numerical_features])

    # Ensure all features are present in the input
    input_df = input_df.reindex(columns=X.columns, fill_value=0)

    # Make prediction
    prediction = model.predict(input_df)
    
    # Display loan status with an icon
    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Denied")
