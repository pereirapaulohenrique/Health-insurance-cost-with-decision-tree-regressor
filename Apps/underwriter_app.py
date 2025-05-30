import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Load the Saved Model and Columns ---
# Ensure these files are in the same directory as your Streamlit app, or provide the correct path.
try:
    model = joblib.load('enhanced_decision_tree_model.joblib')
    model_columns = joblib.load('model_columns.joblib')
except FileNotFoundError:
    st.error("Model files not found! Make sure 'enhanced_decision_tree_model.joblib' and 'model_columns.joblib' are in the app directory.")
    st.stop() # Stop execution if model files are missing

# --- Helper Function to Preprocess Inputs (Same as in cost_estimator_app.py) ---
def preprocess_input(age, sex, bmi, children, smoker, region, all_model_columns):
    raw_data = {
        'age': age, 'sex': sex, 'bmi': bmi, 
        'children': children, 'smoker': smoker, 'region': region
    }
    input_df_raw = pd.DataFrame([raw_data])

    input_df_processed = input_df_raw.copy()
    input_df_processed['sex_male'] = 1 if sex == 'male' else 0
    input_df_processed['smoker_yes'] = 1 if smoker == 'yes' else 0
    
    input_df_processed['region_northwest'] = 1 if region == 'northwest' else 0
    input_df_processed['region_southeast'] = 1 if region == 'southeast' else 0
    input_df_processed['region_southwest'] = 1 if region == 'southwest' else 0

    columns_to_drop_after_encoding = ['sex', 'smoker', 'region']
    for col in columns_to_drop_after_encoding:
        if col in input_df_processed.columns:
            input_df_processed = input_df_processed.drop(columns=[col])
    
    input_df_processed['age_bmi_interaction'] = input_df_processed['age'] * input_df_processed['bmi']
    input_df_processed['smoker_bmi_interaction'] = input_df_processed['smoker_yes'] * input_df_processed['bmi']
    input_df_processed['smoker_age_interaction'] = input_df_processed['smoker_yes'] * input_df_processed['age']
    input_df_processed['age_squared'] = input_df_processed['age']**2
    
    final_input_df = pd.DataFrame(np.zeros((1, len(all_model_columns))), columns=all_model_columns)
    
    for col in input_df_processed.columns:
        if col in final_input_df.columns:
            final_input_df[col] = input_df_processed[col].values 
    
    final_input_df = final_input_df.fillna(0) 
    return final_input_df

# --- Function to Determine Risk Category and Key Factors ---
def assess_risk_profile(prediction, age, bmi, smoker):
    risk_category = ""
    key_factors_summary = []

    # Define risk thresholds (these are illustrative, adjust as needed based on your data's charge distribution)
    if prediction < 8000: # Example threshold
        risk_category = "Low Risk"
    elif prediction < 20000: # Example threshold
        risk_category = "Medium Risk"
    else:
        risk_category = "High Risk"

    # Simplified key factors (based on general knowledge from your model)
    if smoker == 'yes':
        key_factors_summary.append("Applicant is a smoker.")
    if age > 50: # Example threshold
        key_factors_summary.append(f"Applicant age ({age}) is in a higher-cost bracket.")
    if bmi > 30: # Example threshold (Obese category)
        key_factors_summary.append(f"Applicant BMI ({bmi:.2f}) is in the obese category.")
    
    if not key_factors_summary and risk_category == "Low Risk":
        key_factors_summary.append("Applicant profile suggests lower costs based on primary factors.")
    elif not key_factors_summary:
         key_factors_summary.append("Predicted cost influenced by a combination of factors.")


    return risk_category, key_factors_summary

# --- Streamlit App Interface ---
st.set_page_config(layout="wide")
st.title("Underwriter's Assistant: Risk Profiler 🛡️")

# Use columns for layout
col1, col2 = st.columns([1, 2]) # Input column, Output column

with col1:
    st.header("Applicant Details:")
    age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
    
    st.subheader("BMI Calculation")
    height_cm = st.number_input("Height (cm)", min_value=50, max_value=250, value=170, step=1)
    weight_kg = st.number_input("Weight (kg)", min_value=20.0, max_value=200.0, value=70.0, step=0.1, format="%.1f")

    calculated_bmi = 0.0
    if height_cm > 0:
        height_m = height_cm / 100
        calculated_bmi = weight_kg / (height_m ** 2)
        st.metric(label="Calculated BMI", value=f"{calculated_bmi:.2f}")
    else:
        st.warning("Valid height needed for BMI.")

    sex = st.selectbox("Sex", ('male', 'female'), key='sex_underwriter')
    children = st.selectbox("Number of Children", (0, 1, 2, 3, 4, 5), key='children_underwriter')
    smoker = st.selectbox("Smoker", ('no', 'yes'), key='smoker_underwriter') # Default to 'no'
    region = st.selectbox("Region", ('northeast', 'northwest', 'southeast', 'southwest'), key='region_underwriter')

# "Assess Risk" button in the input column
if col1.button("Assess Applicant Risk Profile"):
    if height_cm <= 0:
        col2.error("Height must be greater than 0 cm to calculate BMI and assess risk.")
    else:
        processed_input_df = preprocess_input(age, sex, calculated_bmi, children, smoker, region, model_columns)
        log_prediction = model.predict(processed_input_df)
        prediction = np.expm1(log_prediction)[0]
        
        risk_category, key_factors = assess_risk_profile(prediction, age, calculated_bmi, smoker)
        
        with col2:
            st.header("Risk Assessment Output:")
            st.subheader("Predicted Annual Cost:")
            st.markdown(f"<h3 style='color: blue;'>${prediction:,.2f}</h3>", unsafe_allow_html=True)
            
            st.subheader("Risk Category:")
            if risk_category == "High Risk":
                st.markdown(f"<h3 style='color: red;'>{risk_category}</h3>", unsafe_allow_html=True)
            elif risk_category == "Medium Risk":
                st.markdown(f"<h3 style='color: orange;'>{risk_category}</h3>", unsafe_allow_html=True)
            else: # Low Risk
                st.markdown(f"<h3 style='color: green;'>{risk_category}</h3>", unsafe_allow_html=True)

            st.subheader("Key Contributing Factors / Observations:")
            for factor in key_factors:
                st.markdown(f"- {factor}")
            
            st.markdown("---")
            st.markdown("""
            **Note:** This tool provides an estimate and a general risk profile based on a predictive model. 
            It should be used as one of several inputs for underwriting decisions.
            """)
else:
    with col2:
        st.info("Enter applicant details on the left and click 'Assess Applicant Risk Profile'.")