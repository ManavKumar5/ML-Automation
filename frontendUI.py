import streamlit as st
import os
import pickle
import numpy as np


def load_model(model_path):
    with open(model_path, "rb") as f:
        return pickle.load(f)


def extract_model_metadata(model):
    """Extract feature names and task type directly from model object."""
    metadata = {
        "feature_names": None,
        "model_type": type(model).__name__,
        "task": "classification",
        "classes": None,
    }

    # Extract feature names
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

    # Detect task type
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


def frontend_ui():
    st.title("🖥️ Frontend UI — Live Prediction")

    # ── Model source selection ──────────────────────────────────────────
    st.write("### Select Model Source ")
    model_source = st.radio(
        "How would you like to load your model?",
        ["Select from saved models", "Upload a .pkl file"],
        horizontal=True,
    )

    model = None
    model_name = None

    # ── Load from saved models folder ──────────────────────────────────
    if model_source == "Select from saved models":
        models_dir = "models"
        if not os.path.exists(models_dir):
            st.warning(
                "No `models/` folder found. Please train and save a model first."
            )
            return

        model_files = [f for f in os.listdir(models_dir) if f.endswith(".pkl")]
        if not model_files:
            st.warning("No `.pkl` files found. Please train and save a model first.")
            return

        selected_file = st.selectbox("Select a saved model:", model_files)
        if selected_file:
            model_path = os.path.join(models_dir, selected_file)
            try:
                model = load_model(model_path)
                model_name = selected_file
                st.success(f"✅ Loaded `{selected_file}` successfully!")
            except Exception as e:
                st.error(f"Could not load model: {e}")
                return

    # ── Load from uploaded .pkl ─────────────────────────────────────────
    elif model_source == "Upload a .pkl file":
        uploaded_file = st.file_uploader(
            "Upload your `.pkl` model file",
            type=["pkl"],
            help="Download the .pkl from Model Training page and upload here.",
        )
        if uploaded_file is None:
            st.info(" Please upload a `.pkl` file to continue.")
            return
        try:
            model = pickle.load(uploaded_file)
            model_name = uploaded_file.name
            st.success(f"✅ Loaded `{uploaded_file.name}` successfully!")
        except Exception as e:
            st.error(f"Could not load model: {e}")
            return

    if model is None:
        return

    # ── Extract metadata ────────────────────────────────────────────────
    metadata = extract_model_metadata(model)
    feature_names = metadata.get("feature_names") or []
    task = metadata.get("task", "classification")
    model_type = metadata.get("model_type", "Model")
    classes = metadata.get("classes")

    st.divider()

    # ── Model info cards ────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    col1.metric("Model Type", model_type)
    col2.metric("Task", task.title())
    col3.metric("Features", len(feature_names) if feature_names else "?")

    if feature_names:
        with st.expander("📋 Detected Features"):
            st.write(feature_names)
    if classes:
        with st.expander("🏷️ Target Classes"):
            st.write(classes)

    st.divider()

    # ── Input form ──────────────────────────────────────────────────────
    st.subheader("Enter Feature Values")

    if not feature_names:
        st.warning("Could not detect feature names from the model.")
        return

    with st.form("live_prediction_form"):
        input_values = {}
        # Display inputs in 3 columns for cleaner UI
        cols = st.columns(3)
        for i, feat in enumerate(feature_names):
            label = feat.replace("_", " ").title()
            with cols[i % 3]:
                input_values[feat] = st.number_input(label, value=0.0, key=feat)

        submitted = st.form_submit_button("🔮 Predict", use_container_width=True)

    # ── Prediction ──────────────────────────────────────────────────────
    if submitted:
        input_array = np.array([[input_values[f] for f in feature_names]])
        st.divider()
        st.subheader("Result")
        try:
            if task == "classification":
                prediction = model.predict(input_array)[0]
                st.success(f"**Predicted Class:** {prediction}")

                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(input_array)[0]
                    st.write("**Prediction Probabilities:**")
                    if classes:
                        prob_dict = {
                            str(label): float(f"{p:.4f}")
                            for label, p in zip(classes, proba)
                        }
                    else:
                        prob_dict = {
                            f"Class {i}": float(f"{p:.4f}") for i, p in enumerate(proba)
                        }
                    st.bar_chart(prob_dict)
            else:
                prediction = model.predict(input_array)[0]
                st.success(f"**Predicted Value:** {prediction:.4f}")

        except Exception as e:
            st.error(f"Prediction failed: {e}")


if __name__ == "__main__":
    frontend_ui()
