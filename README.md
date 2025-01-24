
# Loan Eligibility Predictor

This project is a Loan Eligibility Predictor built using **machine learning** and **Streamlit**. It predicts whether a loan applicant is eligible for a loan based on various factors such as income, credit history, and loan details.  

## Features
- Interactive Streamlit web application.
- Accepts user input for loan applicant details.
- Uses a **Random Forest Classifier** to predict loan eligibility.
- Displays model accuracy and loan eligibility status.

## Table of Contents
- [Overview](#overview)
- [Project Workflow](#project-workflow)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Streamlit Link](#streamlit-link)
- [Future Enhancements](#future-enhancements)
- [Contact](#contact)

---

## Overview
This Loan Eligibility Predictor application aims to demonstrate the practical application of machine learning in the financial domain. It enables users to:
- Predict whether their loan will be approved.
- Gain insights into the loan eligibility process.

---

## Project Workflow
1. **Data Cleaning & Preprocessing**  
   - Missing values were handled for both categorical and numerical variables.
   - Label encoding was applied to categorical variables for model compatibility.  

2. **Feature Engineering**  
   - Combined `ApplicantIncome` and `CoapplicantIncome` into a new feature `Total_Income`.
   - Normalized numerical features using **StandardScaler**.  

3. **Model Development**  
   - A **Random Forest Classifier** was used as the machine learning model.
   - Model accuracy: Displayed in the web app for transparency.  

4. **Web Application**  
   - Built an interactive web interface using **Streamlit**.
   - Users can input their loan details and view prediction results.

---

## Dataset
The dataset for this project contains loan applicant information, such as:
- Gender, Marital Status, Dependents, Education, Employment Status, Income, Credit History, and Loan Details.

### Sample Dataset Columns
- `Gender`  
- `Married`  
- `Dependents`  
- `Education`  
- `Self_Employed`  
- `ApplicantIncome`  
- `CoapplicantIncome`  
- `LoanAmount`  
- `Loan_Amount_Term`  
- `Credit_History`  
- `Property_Area`  
- `Loan_Status`  

You can replace the file path with your own dataset if needed.  

---

## Installation
### Prerequisites
- Python 3.8 or above
- pip (Python package manager)

### Steps
1. Clone this repository:  
   ```bash
   git clone https://github.com/your-username/loan-eligibility-predictor.git
   cd loan-eligibility-predictor
   ```

2. Install the required packages:  
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:  
   ```bash
   streamlit run app.py
   ```

---

## Usage
1. Open the Streamlit app using the link displayed in your terminal after running `streamlit run app.py`.  
2. Use the sidebar to input loan applicant details, including income, loan amount, and credit history.  
3. Click the **"Predict Loan Eligibility"** button to view the prediction.  

---

## Streamlit Link
[Loan Eligibility Predictor]([https://your-deployment-link](https://63vg5ogruyh9xspmayv6fj.streamlit.app/
))  


### Deploying to Streamlit Cloud
1. Push your code to a GitHub repository.  
2. Go to [Streamlit Cloud](https://streamlit.io/cloud).  
3. Connect your repository and deploy the app.

---

## Future Enhancements
- Add more machine learning models for comparison.
- Enhance the UI for a better user experience.
- Include explainability features like **SHAP** for model interpretability.
- Expand dataset for better generalization.  

---

## Contact
For queries or collaborations, please feel free to reach out:  
**Your Name:** Noureen  
**Email:** your-email@example.com  


--- 


