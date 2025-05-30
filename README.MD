# Health Insurance Cost Prediction Project

## 🚀 Objective
This project aims to develop an accurate regression model to predict annual health insurance costs for individuals. The goal is to understand the key factors influencing these costs and to create a reliable predictive tool that can be leveraged by both individuals and businesses for better financial planning and risk assessment.

## 📊 Dataset
The analysis is based on a publicly available dataset simulating health insurance information, which includes features such as:
* **Age:** Age of the primary beneficiary.
* **Sex:** Gender of the primary beneficiary.
* **BMI:** Body mass index.
* **Children:** Number of children covered by health insurance / Number of dependents.
* **Smoker:** Whether the person smokes.
* **Region:** The beneficiary's residential area in the US.
* **Charges:** Individual medical costs billed by health insurance (Target Variable).

The dataset typically contains around 1338 records and was sourced from Kaggle.

## 🛠️ Methodology

The project followed a structured machine learning workflow:

1.  **Exploratory Data Analysis (EDA):** Investigated data distributions (notably the right-skew of `charges`), feature correlations, and initial insights into cost drivers.
2.  **Data Preprocessing:** Handled categorical features using one-hot encoding and split the data into training and testing sets.
3.  **Iterative Model Development:**
    * Started with a **Baseline Decision Tree Regressor** to establish initial performance.
    * Performed **Hyperparameter Tuning** using `GridSearchCV` to optimize the Decision Tree.
    * Explored ensemble methods by implementing and tuning **Random Forest Regressor** and **XGBoost Regressor**.
    * Returned to the Decision Tree and significantly improved its performance through **Target Transformation** (log transform on `charges`) and **Feature Engineering** (creating interaction and polynomial features). This "Enhanced Tuned Decision Tree" emerged as the champion model.
4.  **Model Evaluation:** Consistently evaluated models using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R-squared (R²) on a held-out test set. MAE was prioritized for its direct interpretability in dollar terms.
5.  **Advanced Error Analysis:** Conducted a detailed analysis of the champion model's errors, including performance on data subsets (e.g., by smoker status, age group) and the impact of error outliers.

## ✨ Key Results & Insights

* The **Enhanced Tuned Decision Tree Regressor** (with log-transformed target and engineered features) was the best-performing model, achieving:
    * **Test Set MAE: $1881.73**
    * Test Set RMSE: $4763.58
    * Test Set R-squared: 0.84
* This MAE indicates that, on average, the model can predict annual insurance charges within approximately $1,882 of the actual cost, a significant improvement from the data's inherent variability (original standard deviation of charges: ~$12,110).
* **Key Cost Drivers:**
    * **Smoking status** consistently emerged as the most dominant predictor.
    * **Age** and **BMI** were also highly influential, especially in the more nuanced, tuned models.
    * Feature engineering and target transformation were crucial in achieving the best results, highlighting that data representation is as important as model complexity.
* **Error Analysis Insights:** The model performs exceptionally well for the majority of cases (MAE of ~$512 when excluding error outliers). Specific segments (e.g., non-smokers, older individuals) showed different error characteristics, offering avenues for future specialized modeling.

## 🚀 Applications Built

Two Streamlit web applications were developed to showcase the model's utility:

1.  **Individual Cost Estimator:** A user-facing tool where individuals can input their details (age, BMI, smoker status, etc.) and receive an instant estimate of their potential annual health insurance costs.
    * Screenshot to be added...
    * Link to Deployed Individual Estimator App to be added...
2.  **Underwriter's Assistant Tool / Risk Profiler:** A business-facing tool designed to help insurance underwriters assess the risk profile of new applicants. It provides a predicted cost, a risk category (Low, Medium, High), and highlights simplified key factors influencing the prediction.
    * Screenshot to be added...
    * Link to Deployed Underwriter's Assistant App to be added...

## 💻 Technologies Used

* **Python:** Core programming language.
* **Pandas & NumPy:** Data manipulation and numerical operations.
* **Scikit-learn:** Machine learning (DecisionTreeRegressor, RandomForestRegressor, train_test_split, GridSearchCV, metrics).
* **XGBoost:** Gradient Boosting model.
* **Matplotlib & Seaborn:** Data visualization.
* **Joblib:** Saving and loading the trained model.
* **Streamlit:** Building and deploying interactive web applications.
* **Jupyter Notebook:** Development environment.
* **Git & GitHub:** Version control and project hosting.

## ⚙️ Setup & How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/](https://github.com/)[your-github-username]/[your-repository-name].git
    cd [your-repository-name]
    ```
2.  **Create and activate a Conda environment:**
    The `environment.yml` file can be used to recreate the environment (if you create one). Alternatively, set up manually:
    ```bash
    conda create --name health_insurance_env python=3.9
    conda activate health_insurance_env
    pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib streamlit jupyter ipykernel
    # Optional: For Jupyter kernel
    python -m ipykernel install --user --name=health_insurance_env --display-name="Python (health_insurance_env)"
    ```
3.  **Run the Jupyter Notebook:**
    Open the `Health Insurance Cost Prediction.ipynb` notebook in Jupyter Notebook or JupyterLab to see the analysis and model development steps.
4.  **Run the Streamlit Applications:**
    Navigate to the directory containing the app files and run:
    ```bash
    # For the Individual Cost Estimator
    streamlit run cost_estimator_app.py

    # For the Underwriter's Assistant Tool
    streamlit run underwriter_app.py
    ```
    Ensure the saved model (`enhanced_decision_tree_model.joblib`) and columns file (`model_columns.joblib`) are in the same directory as the Streamlit app files.

## 🔗 Portfolio & Contact
* **My Portfolio Page:** In Build...
* **LinkedIn:** https://www.linkedin.com/in/pauloh-pereira/

## 🔮 Future Work (Optional)
* Explore more advanced feature engineering techniques.
* Experiment with other regression algorithms or more extensive tuning for ensemble methods like XGBoost/LightGBM.
* Develop the "mixture of experts" (sub-modeling) approach further based on error analysis for specific segments.
* Incorporate uncertainty estimation for predictions in the applications.

---