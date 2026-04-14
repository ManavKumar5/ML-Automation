import streamlit as st
import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report


def model_training():
    st.title("Model Training")

    preprocessed_files = [f for f in os.listdir("preprocessed") if f.endswith(".csv")]
    if not preprocessed_files:
        st.warning(
            "No preprocessed files available. Please preprocess a dataset first."
        )
        return

    st.write("### Select a preprocessed CSV file ")
    selected_file = st.selectbox("Select :", preprocessed_files)
    if not selected_file:
        return

    df = pd.read_csv(os.path.join("preprocessed", selected_file))
    st.write("### Select the target (Y) column ")
    y_column = st.selectbox("Select :", df.columns)
    st.write("### Select problem type ")
    problem_type = st.selectbox("Select:", ["Classification", "Regression"])

    if y_column and problem_type:
        X = df.drop(columns=[y_column])
        y = df[y_column]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        models = {}
        if problem_type == "Classification":
            models = {
                "Logistic Regression": LogisticRegression(),
                "KNN": KNeighborsClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "SVM": SVC(),
            }
        elif problem_type == "Regression":
            models = {
                "Linear Regression": LinearRegression(),
                "Ridge Regression": Ridge(alpha=1.0),
                "Lasso Regression": Lasso(alpha=0.1),
                "SVM (Regression)": SVR(),
            }

        accuracies = {}
        classification_reports = {}
        cv_scores = {}
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            if problem_type == "Classification":
                accuracies[model_name] = accuracy_score(y_test, y_pred)
                classification_reports[model_name] = classification_report(
                    y_test, y_pred, output_dict=True
                )
                kfold = KFold(n_splits=5, shuffle=True, random_state=42)
                cv_score = cross_val_score(model, X, y, cv=kfold, scoring="accuracy")
                cv_scores[model_name] = cv_score.mean()
            else:
                accuracies[model_name] = mean_squared_error(y_test, y_pred)

        st.write("### Model Performance ")
        for model_name, score in accuracies.items():
            st.write(f"- Accuracy of {model_name} Model: {score:.4f}")
            if problem_type == "Classification":
                st.write(
                    f"- Cross-Validation Accuracy for {model_name}: {cv_scores[model_name]:.4f}"
                )

        if problem_type == "Classification":
            st.write("### Classification Report   ")
            report_df = pd.DataFrame(classification_reports[model_name]).transpose()
            st.dataframe(report_df)

        st.write("### Select a model to save ")
        chosen_model_name = st.selectbox("Select :", list(models.keys()))
        save_cv = st.checkbox("Save Cross-Validation Accuracy")

        chosen_model = models[chosen_model_name]
        selected_fileA = selected_file.split(".")
        model_filename = f"{selected_fileA[0]}.pkl"

        # Serialize model to bytes for download
        model_bytes = pickle.dumps(chosen_model)

        col1, col2 = st.columns(2)

        # ✅ Save model to disk
        with col1:
            if st.button("Save Model", use_container_width=True):
                os.makedirs("models", exist_ok=True)
                model_path = os.path.join("models", model_filename)
                model_data = {"model": chosen_model}
                if save_cv and problem_type == "Classification":
                    model_data["cv_accuracy"] = cv_scores.get(chosen_model_name, None)
                with open(model_path, "wb") as f:
                    pickle.dump(chosen_model, f)
                st.success(f"Model saved as {model_path}")

        # ✅ Download model as .pkl
        with col2:
            st.download_button(
                label=" Download Model (.pkl)",
                data=model_bytes,
                file_name=model_filename,
                mime="application/octet-stream",
                use_container_width=True,
            )
