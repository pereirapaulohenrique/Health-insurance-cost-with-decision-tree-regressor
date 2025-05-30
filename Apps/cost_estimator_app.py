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


# --- Helper Function to Preprocess Inputs (Using the more robust version) ---
def preprocess_input(age, sex, bmi, children, smoker, region, all_model_columns):
    raw_data = {
        'age': age,
        'sex': sex, # Will be dropped after one-hot encoding
        'bmi': bmi, # Now calculated and passed in
        'children': children,
        'smoker': smoker, # Will be dropped
        'region': region  # Will be dropped
    }
    input_df_raw = pd.DataFrame([raw_data])

    input_df_processed = input_df_raw.copy()
    input_df_processed['sex_male'] = 1 if sex == 'male' else 0
    input_df_processed['smoker_yes'] = 1 if smoker == 'yes' else 0
    
    # Region encoding - adjust if your base region was different (e.g., if 'region_northeast' was dropped)
    input_df_processed['region_northwest'] = 1 if region == 'northwest' else 0
    input_df_processed['region_southeast'] = 1 if region == 'southeast' else 0
    input_df_processed['region_southwest'] = 1 if region == 'southwest' else 0
    # 'region_northeast' would be all zeros if it was the base category (drop_first=True during training)

    columns_to_drop_after_encoding = ['sex', 'smoker', 'region']
    for col in columns_to_drop_after_encoding:
        if col in input_df_processed.columns:
            input_df_processed = input_df_processed.drop(columns=[col])
    
    # Feature Engineering
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

# --- Streamlit App Interface ---
st.set_page_config(layout="wide") # Use wide layout
st.title("Health Insurance Cost Estimator 🩺")

st.sidebar.header("Enter Your Details:")

# Input fields
age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30, step=1)

# --- BMI Calculation Inputs ---
st.sidebar.subheader("BMI Calculation")
height_cm = st.sidebar.number_input("Height (cm)", min_value=50, max_value=250, value=170, step=1)
weight_kg = st.sidebar.number_input("Weight (kg)", min_value=20.0, max_value=200.0, value=70.0, step=0.1, format="%.1f")

calculated_bmi = 0.0
if height_cm > 0:
    height_m = height_cm / 100
    calculated_bmi = weight_kg / (height_m ** 2)
    st.sidebar.metric(label="Calculated BMI", value=f"{calculated_bmi:.2f}")
else:
    st.sidebar.warning("Please enter a valid height.")

sex = st.sidebar.selectbox("Sex", ('male', 'female'))
children = st.sidebar.selectbox("Number of Children", (0, 1, 2, 3, 4, 5)) # Using selectbox for consistency
smoker = st.sidebar.selectbox("Smoker", ('no', 'yes')) # Changed order for 'no' as default
region = st.sidebar.selectbox("Region", ('northeast', 'northwest', 'southeast', 'southwest'))


# Predict button
if st.sidebar.button("Estimate Cost"):
    if height_cm <= 0: # Basic validation for height
        st.error("Height must be greater than 0 cm to calculate BMI.")
    else:
        # Preprocess the input using the calculated BMI
        processed_input_df = preprocess_input(age, sex, calculated_bmi, children, smoker, region, model_columns)
        
        # Make prediction (model predicts on log scale)
        log_prediction = model.predict(processed_input_df)
        
        # Convert prediction back to original scale
        prediction = np.expm1(log_prediction)
        
        st.subheader("Estimated Annual Insurance Cost:")
        st.markdown(f"<h2 style='text-align: center; color: green;'>${prediction[0]:,.2f}</h2>", unsafe_allow_html=True)
        
        st.balloons()

        st.markdown("---")
        st.markdown("""
        **Disclaimer:** This is an estimate based on a machine learning model and historical data. 
        It is not an official insurance quote. Actual costs may vary.
        """)
else:
    st.info("Please enter your details in the sidebar and click 'Estimate Cost'.")

st.markdown("---")
st.markdown("This app uses an Enhanced Decision Tree Regressor model. The model was trained on data including engineered features and a log-transformed target variable to improve prediction accuracy, especially for varied cost ranges.")