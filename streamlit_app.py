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

    # Feature columns
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

    # Power transformation
    pt = PowerTransformer(method="yeo-johnson")
    df[feature_columns] = pt.fit_transform(df[feature_columns])

    # Scaling
    scaler = StandardScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])

    # Features and labels
    X = df[feature_columns]
    y = df["Diagnosis"]

    # Handle imbalance
    sm = SMOTE(random_state=42)
    X_resampled, y_resampled = sm.fit_resample(X, y)

    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_resampled, y_resampled)

    return model, feature_columns, scaler, pt


# Load model and preprocessors
model, feature_order, scaler, pt = load_model_and_preprocessors()


# ------------------------ Streamlit Interface ------------------------ #
st.title("🧠 Parkinson’s Disease Prediction App")

st.markdown("Choose one of the following options to predict Parkinson’s Disease:")
option = st.radio("Select Input Method", ["📤 Upload CSV", "✍️ Manual Entry"])

if option == "📤 Upload CSV":
    st.subheader("Upload a CSV file")
    uploaded_file = st.file_uploader("Upload test dataset CSV", type=["csv"])

    if uploaded_file:
        try:
            full_df = pd.read_csv(uploaded_file)

            # Preserve metadata columns if available
            metadata_cols = [col for col in ["PatientID", "DoctorInCharge"] if col in full_df.columns]
            metadata_df = full_df[metadata_cols] if metadata_cols else pd.DataFrame()

            # Drop target label if present
            full_df.drop(columns=["Diagnosis"], inplace=True, errors="ignore")

            # Ensure required features are present
            feature_cols_in_file = full_df.columns.intersection(feature_order)
            missing = set(feature_order) - set(feature_cols_in_file)
            extra = set(full_df.columns) - set(feature_order) - set(metadata_cols)

            if missing:
                st.error(f"❌ Missing required features: {missing}")
            else:
                feature_df = full_df[feature_order]
                original_feature_df = feature_df.copy()

                # Preprocess
                transformed = pd.DataFrame(pt.transform(feature_df), columns=feature_order)
                scaled = pd.DataFrame(scaler.transform(transformed), columns=feature_order)

                # Predict
                predictions = model.predict(scaled)
                probs = model.predict_proba(scaled)[:, 1]

                # Merge all outputs
                result_df = pd.concat([metadata_df, original_feature_df], axis=1)
                result_df["Prediction"] = predictions
                result_df["Confidence"] = (probs * 100).round(2).astype(str) + "%"

                st.success("✅ Prediction completed.")
                st.dataframe(result_df)

                csv = result_df.to_csv(index=False).encode("utf-8")
                st.download_button("Download Results as CSV", csv, "predictions.csv", "text/csv")

        except Exception as e:
            st.error(f"⚠️ Error processing file: {e}")

elif option == "✍️ Manual Entry":
    st.subheader("Enter feature values manually")

    user_input = {}
    for col in feature_order:
        user_input[col] = st.number_input(f"{col}", value=0.0, step=0.01)

    if st.button("Predict"):
        try:
            input_df = pd.DataFrame([user_input])[feature_order]
            original_input = input_df.copy()

            # Preprocess
            input_df = pd.DataFrame(pt.transform(input_df), columns=feature_order)
            input_df = pd.DataFrame(scaler.transform(input_df), columns=feature_order)

            prediction = model.predict(input_df)[0]
            confidence = model.predict_proba(input_df)[0][1]

            if prediction == 1:
                st.error("⚠️ Likely Parkinson’s Disease detected.")
            else:
                st.success("✅ Unlikely to have Parkinson’s Disease.")

            st.info(f"🧮 Model confidence: {confidence:.2%}")

        except Exception as e:
            st.error(f"Error during prediction: {e}")
