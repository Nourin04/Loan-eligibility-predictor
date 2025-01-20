import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load and preprocess the data
@st.cache_data  # Caches the data loading function for better performance
def load_data():
    # Load the dataset
    data = pd.read_csv('loan_data_set.csv')

    # Drop the 'Loan_ID' column if it exists (not relevant for prediction)
    if 'Loan_ID' in data.columns:
        data = data.drop('Loan_ID', axis=1)

    # Define columns for categorical and numerical data
    categorical_cols = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History']
    numerical_cols = ['LoanAmount', 'Loan_Amount_Term']

    # Handle missing values for categorical and numerical columns
    for col in categorical_cols:
        data[col] = data[col].fillna(data[col].mode()[0])  # Fill with mode
    for col in numerical_cols:
        data[col] = data[col].fillna(data[col].median())  # Fill with median

    # Encode categorical variables using Label Encoding
    label_encoders = {}
    for col in data.select_dtypes(include='object').columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    # Create a new feature 'Total_Income' by combining Applicant and Coapplicant Income
    data['Total_Income'] = data['ApplicantIncome'] + data['CoapplicantIncome']
    data = data.drop(['ApplicantIncome', 'CoapplicantIncome'], axis=1)

    # Standardize numerical features using StandardScaler
    scaler = StandardScaler()
    data[['LoanAmount', 'Loan_Amount_Term', 'Total_Income']] = scaler.fit_transform(
        data[['LoanAmount', 'Loan_Amount_Term', 'Total_Income']]
    )

    return data, label_encoders, scaler


# Load the data and preprocess it
data, label_encoders, scaler = load_data()

# Split data into features (X) and target (y)
X = data.drop('Loan_Status', axis=1)
y = data['Loan_Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Streamlit App UI
st.title("Loan Eligibility Predictor")

# Project Overview Section
st.markdown("""
### Project Overview:
This project predicts loan eligibility using machine learning. Users input their personal and financial details, and the app predicts whether their loan will be approved or denied.
""")

# Model Accuracy Section
st.write("### Model Accuracy")
st.write(f"Accuracy: {accuracy:.2f}")

# Form for user input
def user_input_form():
    st.sidebar.header("Enter Loan Applicant Details")
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    married = st.sidebar.selectbox("Marital Status", ["Married", "Not Married"])
    dependents = st.sidebar.selectbox("Dependents", [0, 1, 2, 3+])
    education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
    applicant_income = st.sidebar.number_input("Applicant Income", min_value=0)
    coapplicant_income = st.sidebar.number_input("Coapplicant Income", min_value=0)
    loan_amount = st.sidebar.number_input("Loan Amount", min_value=0)
    loan_amount_term = st.sidebar.number_input("Loan Amount Term", min_value=0)
    credit_history = st.sidebar.selectbox("Credit History", [1.0, 0.0])
    property_area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    # Collect all user inputs into a dictionary
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

# Display chatbot for quick assistance
def chatbot_conversation():
    user_input = st.text_input("Chat with us! Ask a question:", "")

    if user_input:
        # Rule-based chatbot responses
        if "loan eligibility" in user_input.lower():
            st.chat_message("bot").markdown("**Bot:** Provide your details above to check eligibility.")
        elif "apply for loan" in user_input.lower():
            st.chat_message("bot").markdown("**Bot:** Fill out the form above to apply.")
        else:
            st.chat_message("bot").markdown("**Bot:** Please ask about eligibility or loan criteria.")

st.sidebar.subheader("Loan Eligibility Chatbot")
chatbot_conversation()

# Get user input and make predictions
user_input = user_input_form()
if st.sidebar.button("Predict Loan Eligibility"):
    input_df = pd.DataFrame([user_input])

    # Encode user input and preprocess it
    for col, le in label_encoders.items():
        if col in input_df.columns:
            if input_df[col].iloc[0] not in le.classes_:
                input_df[col] = le.transform([le.classes_[0]])  # Handle unseen data
            else:
                input_df[col] = le.transform(input_df[col])

    input_df['Total_Income'] = input_df['ApplicantIncome'] + input_df['CoapplicantIncome']
    input_df = input_df.drop(['ApplicantIncome', 'CoapplicantIncome'], axis=1)

    numerical_features = ['LoanAmount', 'Loan_Amount_Term', 'Total_Income']
    input_df[numerical_features] = scaler.transform(input_df[numerical_features])

    # Match input with model features
    input_df = input_df.reindex(columns=X.columns, fill_value=0)

    # Make prediction
    prediction = model.predict(input_df)

    # Display prediction result
    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Denied")
