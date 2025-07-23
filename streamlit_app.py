import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, PowerTransformer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

@st.cache_resource
def load_model_and_features():
    df = pd.read_csv('parkinsons_disease_data.csv')
    df.drop_duplicates(inplace=True)
    df.drop(["DoctorInCharge", "PatientID"], axis=1, inplace=True)

    numerical_columns = [
        'Age', 'BMI', 'SystolicBP', 'DiastolicBP', 'CholesterolTotal',
        'CholesterolLDL', 'CholesterolHDL', 'CholesterolTriglycerides', 'UPDRS',
        'MoCA', 'FunctionalAssessment',
        'Gender', 'Ethnicity', 'EducationLevel', 'Smoking', 'AlcoholConsumption',
        'PhysicalActivity', 'DietQuality', 'SleepQuality', 'FamilyHistoryParkinsons',
        'TraumaticBrainInjury', 'Hypertension', 'Diabetes', 'Depression', 'Stroke',
        'Tremor', 'Rigidity', 'Bradykinesia', 'PosturalInstability', 'SpeechProblems',
        'SleepDisorders', 'Constipation'
    ]

    # Remove outliers from numerical features
    for col in numerical_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    # Power transformation on all numerical columns
    pt = PowerTransformer(method='yeo-johnson')
    df[numerical_columns] = pt.fit_transform(df[numerical_columns])

    # Scale numerical features
    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    # Split into features and label
    X = df.drop("Diagnosis", axis=1)
    y = df["Diagnosis"]
    feature_order = X.columns.tolist()

    # Oversample minority class
    sm = SMOTE(random_state=300)
    X_resampled, y_resampled = sm.fit_resample(X, y)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    return model, numerical_columns, scaler, pt, feature_order


# ---------------- Streamlit UI ---------------- #
st.title("🧠 Parkinson’s Disease Prediction App")
st.write("Enter the values for each feature below:")

# Load model and preprocessing tools
model, numerical_columns, scaler, pt, feature_order = load_model_and_features()

st.subheader("Enter your medical and demographic information below:")

input_data = {}

# Collect numerical input
for col in numerical_columns:
    val = st.number_input(f"{col}", min_value=-1000.0, value=0.0, step=0.01)
    input_data[col] = val

# Predict button
if st.button("Predict"):
    try:
        input_df = pd.DataFrame([input_data])

        # Reorder columns to match training feature order
        input_df = input_df[feature_order]

        # Apply PowerTransformer and scaler
        input_df[numerical_columns] = pt.transform(input_df[numerical_columns])
        input_df[numerical_columns] = scaler.transform(input_df[numerical_columns])

        # Predict
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        # Display result
        if prediction == 1:
            st.error("⚠️ Likely Parkinson’s Disease detected.")
        else:
            st.success("✅ Unlikely to have Parkinson’s Disease.")

        st.info(f"🧮 Model confidence: {probability:.2%}")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
