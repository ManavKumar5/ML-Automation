import streamlit as st
import os
import pickle
import numpy as np


def extract_model_metadata(model_path):
    """Extract feature names, types, and model info from a trained .pkl file."""
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    metadata = {
        "feature_names": None,
        "feature_types": None,
        "model_type": type(model).__name__,
        "task": "classification",
        "classes": None,
    }

    if hasattr(model, "feature_names_in_"):
        metadata["feature_names"] = list(model.feature_names_in_)
    elif hasattr(model, "named_steps"):
        for step_name, step in model.named_steps.items():
            if hasattr(step, "feature_names_in_"):
                metadata["feature_names"] = list(step.feature_names_in_)
                break
            if hasattr(step, "get_feature_names_out"):
                try:
                    metadata["feature_names"] = list(step.get_feature_names_out())
                    break
                except Exception:
                    pass
    elif hasattr(model, "n_features_in_"):
        metadata["feature_names"] = [
            f"feature_{i}" for i in range(model.n_features_in_)
        ]

    if hasattr(model, "predict_proba") or hasattr(model, "classes_"):
        metadata["task"] = "classification"
        if hasattr(model, "classes_"):
            metadata["classes"] = list(model.classes_)
    elif hasattr(model, "predict") and not hasattr(model, "predict_proba"):
        metadata["task"] = "regression"

    if hasattr(model, "named_steps"):
        final_step = list(model.named_steps.values())[-1]
        if hasattr(final_step, "classes_"):
            metadata["task"] = "classification"
            metadata["classes"] = list(final_step.classes_)
        elif "Regressor" in type(final_step).__name__:
            metadata["task"] = "regression"
        elif "Classifier" in type(final_step).__name__:
            metadata["task"] = "classification"

    if "Regressor" in metadata["model_type"] or "SVR" in metadata["model_type"]:
        metadata["task"] = "regression"
    elif "Classifier" in metadata["model_type"] or "SVC" in metadata["model_type"]:
        metadata["task"] = "classification"

    return metadata


def generate_frontend_code(model_filename, metadata):
    """Generate a real, data-aware Streamlit frontend from model metadata."""

    feature_names = metadata.get("feature_names") or []
    task = metadata.get("task", "classification")
    model_type = metadata.get("model_type", "Model")
    classes = metadata.get("classes")

    # Build input fields block (8 spaces = inside with st.form)
    input_fields_lines = []
    for feat in feature_names:
        label = feat.replace("_", " ").title()
        input_fields_lines.append(
            f'        {feat} = st.number_input("{label}", value=0.0, key="{feat}")'
        )
    input_fields_code = (
        "\n".join(input_fields_lines)
        if input_fields_lines
        else '        feature_0 = st.number_input("Feature 0", value=0.0)'
    )

    # Build numpy array from feature list
    if feature_names:
        features_list = ", ".join(feature_names)
        array_code = f"    input_array = np.array([[{features_list}]])"
    else:
        array_code = "    input_array = np.array([[feature_0]])"

    # Build result display block — 12 spaces to nest correctly inside if submitted > try
    if task == "classification":
        if classes:
            classes_repr = repr(classes)
            result_block = (
                "            proba = model.predict_proba(input_array)[0]\n"
                f"            class_labels = {classes_repr}\n"
                "            prediction = model.predict(input_array)[0]\n"
                '            st.success(f"**Predicted Class:** {prediction}")\n'
                '            st.write("**Prediction Probabilities:**")\n'
                '            prob_dict = {str(label): float(f"{p:.4f}") for label, p in zip(class_labels, proba)}\n'
                "            st.bar_chart(prob_dict)"
            )
        else:
            result_block = (
                "            prediction = model.predict(input_array)[0]\n"
                '            st.success(f"**Predicted Class:** {prediction}")'
            )
    else:
        result_block = (
            "            prediction = model.predict(input_array)[0]\n"
            '            st.success(f"**Predicted Value:** {prediction:.4f}")'
        )

    code = f"""import streamlit as st
import pickle
import numpy as np


def load_model_from_upload(uploaded_file):
    return pickle.load(uploaded_file)


def main():
    st.set_page_config(page_title="{model_type} Predictor", layout="centered")
    st.title("{model_type} — Prediction App")
    st.caption("Auto-generated frontend · Model: `{model_filename}`")
    st.divider()

    #  User uploads their downloaded .pkl file — no hardcoded path needed
    st.subheader(" Upload Your Model")
    uploaded_file = st.file_uploader(
        "Upload the `{model_filename}` file you downloaded",
        type=["pkl"],
        help="Download the .pkl file from the Model Training page and upload it here.",
    )

    if uploaded_file is None:
        st.info(" Please upload your `.pkl` model file to get started.")
        st.stop()

    model = load_model_from_upload(uploaded_file)
    st.success("Model loaded successfully!")
    st.divider()

    st.subheader("Enter Feature Values")
    with st.form("prediction_form"):
{input_fields_code}

        submitted = st.form_submit_button("Predict", use_container_width=True)

{array_code}

    if submitted:
        st.divider()
        st.subheader("Result")
        try:
{result_block}
        except Exception as e:
            st.error(f"Prediction failed: {{e}}")


if __name__ == "__main__":
    main()
"""
    return code


def display_frontend_code():
    st.set_page_config(page_title="ML Frontend Generator", layout="centered")
    st.title("⚙️ ML Model Frontend Generator")
    st.write(
        "Select a trained `.pkl` model to auto-generate a Streamlit prediction app."
    )

    models_dir = "models"
    if not os.path.exists(models_dir):
        st.warning("No `models/` folder found. Please train a model first.")
        return

    model_files = [f for f in os.listdir(models_dir) if f.endswith(".pkl")]
    if not model_files:
        st.warning("No `.pkl` files found in the `models/` folder.")
        return

    selected_model_file = st.selectbox("Select Model File:", model_files)

    if selected_model_file:
        model_path = os.path.join(models_dir, selected_model_file)

        with st.spinner("Reading model metadata..."):
            try:
                metadata = extract_model_metadata(model_path)
            except Exception as e:
                st.error(f"Could not load model: {e}")
                return

        col1, col2, col3 = st.columns(3)
        col1.metric("Model Type", metadata["model_type"])
        col2.metric("Task", metadata["task"].title())
        n_features = (
            len(metadata["feature_names"]) if metadata["feature_names"] else "?"
        )
        col3.metric("Features", n_features)

        if metadata["feature_names"]:
            with st.expander("Detected Features"):
                st.write(metadata["feature_names"])

        if metadata["classes"]:
            with st.expander("Target Classes"):
                st.write(metadata["classes"])

        st.divider()

        frontend_code = generate_frontend_code(selected_model_file, metadata)

        st.write("### Generated Streamlit App Code:")
        st.code(frontend_code, language="python")

        st.download_button(
            label="Download frontend_app.py",
            data=frontend_code.encode("utf-8"),
            file_name=f"{selected_model_file.split('.')[0]}_app.py",
            mime="text/plain",
            use_container_width=True,
        )


if __name__ == "__main__":
    display_frontend_code()
