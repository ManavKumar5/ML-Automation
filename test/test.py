import streamlit as st
import pickle
import numpy as np


def load_model_from_upload(uploaded_file):
    return pickle.load(uploaded_file)


def main():
    st.set_page_config(page_title="LinearRegression Predictor", layout="centered")
    st.title("🤖 LinearRegression — Prediction App")
    st.caption("Auto-generated frontend · Model: `Advertising.pkl`")
    st.divider()

    # ✅ User uploads their downloaded .pkl file — no hardcoded path needed
    st.subheader("📂 Upload Your Model")
    uploaded_file = st.file_uploader(
        "Upload the `Advertising.pkl` file you downloaded",
        type=["pkl"],
        help="Download the .pkl file from the Model Training page and upload it here.",
    )

    if uploaded_file is None:
        st.info("⬆️ Please upload your `.pkl` model file to get started.")
        st.stop()

    model = load_model_from_upload(uploaded_file)
    st.success("✅ Model loaded successfully!")
    st.divider()

    st.subheader("Enter Feature Values")
    with st.form("prediction_form"):
        Unnamed: 0 = st.number_input("Unnamed: 0", value=0.0, key="Unnamed: 0")
        TV = st.number_input("Tv", value=0.0, key="TV")
        Radio = st.number_input("Radio", value=0.0, key="Radio")
        Newspaper = st.number_input("Newspaper", value=0.0, key="Newspaper")

        submitted = st.form_submit_button("Predict", use_container_width=True)

    input_array = np.array([[Unnamed: 0, TV, Radio, Newspaper]])

    if submitted:
        st.divider()
        st.subheader("Result")
        try:
            prediction = model.predict(input_array)[0]
            st.success(f"**Predicted Value:** {prediction:.4f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")


if __name__ == "__main__":
    main()