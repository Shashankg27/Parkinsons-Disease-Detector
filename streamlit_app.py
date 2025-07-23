import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, PowerTransformer
from imblearn.over_sampling import SMOTE

st.set_page_config(page_title="Parkinson’s Disease Prediction App", layout="centered")

@st.cache_resource
def load_model_and_preprocessors():
    df = pd.read_csv("parkinsons_disease_data.csv")
    df.drop_duplicates(inplace=True)
    df.drop(["DoctorInCharge", "PatientID"], axis=1, errors="ignore", inplace=True)

    feature_columns = df.columns.tolist()
    feature_columns.remove("Diagnosis")

    # Remove outliers
    for col in feature_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    # Transform and scale
    pt = PowerTransformer(method="yeo-johnson")
    df[feature_columns] = pt.fit_transform(df[feature_columns])
    scaler = StandardScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])

    # Prepare model
    X = df[feature_columns]
    y = df["Diagnosis"]
    sm = SMOTE(random_state=42)
    X_resampled, y_resampled = sm.fit_resample(X, y)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_resampled, y_resampled)

    return model, feature_columns, scaler, pt

model, feature_order, scaler, pt = load_model_and_preprocessors()

st.title("🧠 Parkinson’s Disease Prediction App")
st.markdown("Choose one of the following options to predict Parkinson’s Disease:")
option = st.radio("Select Input Method", ["📤 Upload CSV", "✍️ Manual Entry"])

# --------- CSV Upload Path ---------
if option == "📤 Upload CSV":
    st.subheader("Upload a CSV file")
    uploaded_file = st.file_uploader("Upload test dataset CSV", type=["csv"])

    if uploaded_file:
        try:
            full_df = pd.read_csv(uploaded_file)
            metadata_cols = [col for col in ["PatientID", "DoctorInCharge"] if col in full_df.columns]
            metadata_df = full_df[metadata_cols] if metadata_cols else pd.DataFrame()

            full_df.drop(columns=["Diagnosis"], inplace=True, errors="ignore")

            feature_cols_in_file = full_df.columns.intersection(feature_order)
            missing = set(feature_order) - set(feature_cols_in_file)

            if missing:
                st.error(f"❌ Missing required features: {missing}")
                st.stop()

            feature_df = full_df[feature_order].copy()

            for col in feature_order:
                feature_df[col] = feature_df[col].astype(str).str.strip()
                feature_df[col] = pd.to_numeric(feature_df[col], errors='coerce')

            invalid_rows = feature_df.isnull().any(axis=1)
            if invalid_rows.any():
                feature_df = feature_df[~invalid_rows]

            if feature_df.empty:
                st.error("❌ No valid rows left after cleaning.")
                st.stop()

            transformed = pd.DataFrame(pt.transform(feature_df), columns=feature_order)
            scaled = pd.DataFrame(scaler.transform(transformed), columns=feature_order)

            predictions = model.predict(scaled)
            probs = model.predict_proba(scaled)[:, 1]

            result_df = pd.concat([metadata_df.reset_index(drop=True), 
                                   feature_df.reset_index(drop=True)], axis=1)
            result_df["Prediction"] = predictions
            result_df["Confidence"] = pd.Series((probs * 100).round(2)).map(lambda x: f"{x}%")

            st.success("✅ Prediction completed.")
            st.dataframe(result_df)

            csv = result_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Results as CSV", csv, "predictions.csv", "text/csv")

        except Exception as e:
            st.error("⚠️ Error during prediction:")
            st.exception(e)

# --------- Manual Entry Path ---------
elif option == "✍️ Manual Entry":
    st.subheader("Enter feature values manually")

    user_input = {}
    for col in feature_order:
        user_input[col] = st.number_input(f"{col}", value=0.0, step=0.01)

    if st.button("Predict"):
        try:
            input_df = pd.DataFrame([user_input])[feature_order]
            transformed = pd.DataFrame(pt.transform(input_df), columns=feature_order)
            scaled = pd.DataFrame(scaler.transform(transformed), columns=feature_order)

            prediction = model.predict(scaled)[0]
            confidence = model.predict_proba(scaled)[0][1]

            if prediction == 1:
                st.error("⚠️ Likely Parkinson’s Disease detected.")
            else:
                st.success("✅ Unlikely to have Parkinson’s Disease.")

            st.info(f"🧮 Model confidence: {confidence:.2%}")

        except Exception as e:
            st.error("Error during prediction:")
            st.exception(e)
