import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path # Import Path

# --- THIS MUST BE THE FIRST STREAMLIT COMMAND ---
st.set_page_config(layout="wide")

# --- Get the directory of the current script (which is inside 'Apps') ---
SCRIPT_DIR = Path(__file__).resolve().parent

# --- Go one level up to the project root, then into 'Models' ---
MODEL_DIR = SCRIPT_DIR.parent / "Models"

MODEL_PATH = MODEL_DIR / 'enhanced_decision_tree_model.joblib'
MODEL_COLUMNS_PATH = MODEL_DIR / 'model_columns.joblib'

# --- Load the Saved Model and Columns ---
try:
    model = joblib.load(MODEL_PATH)
    model_columns = joblib.load(MODEL_COLUMNS_PATH)
except FileNotFoundError:
    st.error(
        f"Model files not found! Searched for these absolute paths:\n"
        f"- Model: {MODEL_PATH.resolve()}\n"
        f"- Columns: {MODEL_COLUMNS_PATH.resolve()}\n"
        f"Please ensure a 'Models' folder exists at the project root (same level as your 'Apps' folder) and contains the .joblib files."
    )
    st.stop()

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
def assess_risk_profile(prediction, age, bmi_value, smoker_status): # Renamed bmi and smoker for clarity
    risk_category = ""
    key_factors_summary = []

    # Define risk thresholds (these are illustrative, adjust based on your data's charge distribution)
    if prediction < 8000: 
        risk_category = "Low Risk"
    elif prediction < 20000: 
        risk_category = "Medium Risk"
    else:
        risk_category = "High Risk"

    # Simplified key factors
    if smoker_status == 'yes':
        key_factors_summary.append("Applicant is a **smoker**.")
    if age > 50: 
        key_factors_summary.append(f"Applicant age ({age}) is **over 50**.")
    if bmi_value > 30: 
        key_factors_summary.append(f"Applicant BMI ({bmi_value:.2f}) is in the **obese category (>30)**.")
    
    if not key_factors_summary and risk_category == "Low Risk":
        key_factors_summary.append("Profile suggests lower costs based on primary factors.")
    elif not key_factors_summary: # For Medium risk with no obvious flags
         key_factors_summary.append("Predicted cost influenced by a combination of factors.")

    return risk_category, key_factors_summary

# --- Streamlit App Interface ---
st.title("Underwriter's Assistant: Risk Profiler 🛡️")

col1, col2 = st.columns([1, 1.5]) # Adjusted column widths

with col1:
    st.header("Applicant Details:")
    age_input = st.number_input("Age", min_value=18, max_value=100, value=30, step=1, key='age_uw')
    
    st.subheader("BMI Calculation")
    height_cm_input = st.number_input("Height (cm)", min_value=50, max_value=250, value=170, step=1, key='height_uw')
    weight_kg_input = st.number_input("Weight (kg)", min_value=20.0, max_value=200.0, value=70.0, step=0.1, format="%.1f", key='weight_uw')

    calculated_bmi_uw = 0.0
    if height_cm_input > 0:
        height_m_uw = height_cm_input / 100
        calculated_bmi_uw = weight_kg_input / (height_m_uw ** 2)
        st.metric(label="Calculated BMI", value=f"{calculated_bmi_uw:.2f}")
    else:
        st.warning("Valid height needed for BMI.")

    sex_input = st.selectbox("Sex", ('male', 'female'), key='sex_uw')
    children_input = st.selectbox("Number of Children", (0, 1, 2, 3, 4, 5), key='children_uw')
    smoker_input = st.selectbox("Smoker", ('no', 'yes'), key='smoker_uw')
    region_input = st.selectbox("Region", ('northeast', 'northwest', 'southeast', 'southwest'), key='region_uw')

if col1.button("Assess Applicant Risk Profile"):
    if height_cm_input <= 0:
        col2.error("Height must be greater than 0 cm to calculate BMI and assess risk.")
    else:
        processed_input_df_uw = preprocess_input(age_input, sex_input, calculated_bmi_uw, children_input, smoker_input, region_input, model_columns)
        log_prediction_uw = model.predict(processed_input_df_uw)
        prediction_uw = np.expm1(log_prediction_uw)[0]
        
        risk_category_uw, key_factors_uw = assess_risk_profile(prediction_uw, age_input, calculated_bmi_uw, smoker_input)
        
        with col2:
            st.header("Risk Assessment Output:")
            st.subheader("Predicted Annual Cost:")
            st.markdown(f"<h3 style='color: blue;'>${prediction_uw:,.2f}</h3>", unsafe_allow_html=True)
            
            st.subheader("Risk Category:")
            risk_color = "green" # Default for Low Risk
            if risk_category_uw == "High Risk": risk_color = "red"
            elif risk_category_uw == "Medium Risk": risk_color = "orange"
            st.markdown(f"<h3 style='color: {risk_color};'>{risk_category_uw}</h3>", unsafe_allow_html=True)

            st.subheader("Key Contributing Factors / Observations:")
            if key_factors_uw:
                for factor in key_factors_uw:
                    st.markdown(f"- {factor}")
            else: # Should be covered by logic in assess_risk_profile
                st.markdown("- No single dominant high-risk factor identified; cost driven by overall profile.")
            
            st.markdown("---")
            st.markdown("""
            **Note:** This tool provides an estimate and a general risk profile based on a predictive model. 
            It should be used as one of several inputs for underwriting decisions.
            """)
else:
    with col2:
        st.info("Enter applicant details on the left and click 'Assess Applicant Risk Profile'.")