import pandas as pd
import os
import streamlit as st
from fileManagement import listFiles
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import json
import matplotlib.pyplot as plt
import seaborn as sns


# function to remove outliers
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter out outliers
    df_cleaned = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    return df_cleaned


# defining function for plotting graph


def plot_boxplot(df, column):
    """Function to plot a boxplot"""
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.boxplot(y=df[column].dropna(), ax=ax)
    plt.ylabel(column)
    st.pyplot(fig)


def plot_barplot(df, column):
    """Function to plot a barplot"""
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.countplot(x=df[column].dropna(), ax=ax)
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    st.pyplot(fig)


def plot_histogram(df, column):
    """Function to plot a histogram"""
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df[column].dropna(), bins=30, kde=True, ax=ax)
    plt.xlabel(column)
    plt.ylabel("Frequency")
    st.pyplot(fig)


def plot_selected_graph(df, plot_type, column):
    """Function to select the graph type and plot it"""
    if plot_type == "Boxplot":
        plot_boxplot(df, column)
    elif plot_type == "Barplot":
        plot_barplot(df, column)
    elif plot_type == "Histogram":
        plot_histogram(df, column)


# Function to perform under-sampling
def under_sample_data(df, target_column):
    X = df.drop(columns=[target_column])  # Features
    y = df[target_column]  # Target column

    undersampler = RandomUnderSampler(sampling_strategy="auto", random_state=42)
    X_resampled, y_resampled = undersampler.fit_resample(X, y)

    # Combine back into a DataFrame
    df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    df_resampled[target_column] = y_resampled

    return df_resampled


# Function to perform over-sampling
def over_sample_data(df, target_column):
    X = df.drop(columns=[target_column])  # Features
    y = df[target_column]  # Target column

    oversampler = RandomOverSampler(sampling_strategy="auto", random_state=42)
    X_resampled, y_resampled = oversampler.fit_resample(X, y)

    # Combine back into a DataFrame
    df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    df_resampled[target_column] = y_resampled

    return df_resampled


# Function to display class distribution
def display_class_distribution(df, target_column):
    st.write("### Class Distribution:")
    st.write(df[target_column].value_counts())

    # Show class distribution plot
    plt.figure(figsize=(6, 4))
    sns.countplot(x=target_column, data=df)
    st.pyplot(plt)


# Main preprocess function
def preprocess_data():
    st.title("Preprocess Data")

    csv_files = listFiles()
    if not csv_files:
        st.warning("No CSV files available. Please upload a file first.")
    else:
        st.write("### Select a CSV file to preprocess 📄")
        selected_file = st.selectbox("Select:", csv_files)

        if selected_file:
            df = pd.read_csv(os.path.join("data", selected_file))
            st.write("### Data Preview  ")
            st.dataframe(df.head())

            st.write("### Visualize Data  ")

            # User selects the type of graph
            plot_type = st.selectbox(
                "Choose a plot type:", ["Boxplot", "Barplot", "Histogram"]
            )

            # User selects the column for the chosen plot type
            if plot_type == "Boxplot":
                numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
                if not numeric_cols.empty:
                    column = st.selectbox("Select column for Boxplot:", numeric_cols)
                    if column:
                        plot_selected_graph(df, plot_type, column)
                else:
                    st.warning("No numeric columns available for Boxplot.")

            elif plot_type == "Barplot":
                categorical_cols = df.select_dtypes(include=["object"]).columns
                if not categorical_cols.empty:
                    column = st.selectbox(
                        "Select column for Barplot:", categorical_cols
                    )
                    if column:
                        plot_selected_graph(df, plot_type, column)
                else:
                    st.warning("No categorical columns available for Barplot.")

            elif plot_type == "Histogram":
                numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
                if not numeric_cols.empty:
                    column = st.selectbox("Select column for Histogram:", numeric_cols)
                    if column:
                        plot_selected_graph(df, plot_type, column)
                else:
                    st.warning("No numeric columns available for Histogram.")

            # Multi-column selection for removing outliers
            st.write("### Select columns to remove outliers ")
            col1, col2 = st.columns([3, 1])
            columns_to_clean = col1.multiselect("Select:", df.columns)
            if st.button("Remove Outliers"):
                if columns_to_clean:
                    df_cleaned = df.copy()  # Create a copy of df to modify
                    for column in columns_to_clean:
                        df_cleaned = remove_outliers(
                            df_cleaned, column
                        )  # Remove outliers column-wise
                        if df_cleaned.empty:
                            st.warning(
                                "All values were removed! Try adjusting your dataset or method."
                            )
                        else:
                            df = df_cleaned  # Update df globally
                            st.write(
                                f"### Data after removing outliers from {', '.join(columns_to_clean)}:"
                            )
                            st.dataframe(df)  # Display updated DataFrame
                else:
                    st.warning("Please select at least one column.")

            # Preprocess buttons and actions
            st.write("### Analyze your data  ")
            if st.button(" Analyze Data"):
                null_counts = df.isnull().sum()
                null_info = pd.DataFrame(
                    {"Column": df.columns, "Null Values": null_counts.values}
                )
                st.write("### Null Values per Column:")
                st.dataframe(null_info)

                categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
                st.write("### Categorical Columns:")
                st.write(
                    categorical_cols
                    if categorical_cols
                    else "No categorical columns found."
                )

            # Preprocess Data button
            st.write("### Preprocess your data  ")
            if st.button("Preprocess Data"):
                label_encoders = {}
                # Handle missing values and encode categorical features
                for col in df.columns:
                    if df[col].isnull().sum() > 0:
                        df[col].fillna(df[col].mode()[0], inplace=True)

                    if df[col].dtype == "object":
                        le = LabelEncoder()
                        df[col] = le.fit_transform(df[col])
                        label_encoders[col] = {
                            str(k): int(v)
                            for k, v in zip(le.classes_, le.transform(le.classes_))
                        }

                # Save preprocessed file
                preprocessed_folder = "preprocessed"
                os.makedirs(preprocessed_folder, exist_ok=True)
                preprocessed_filename = f"{selected_file}"
                df.to_csv(
                    os.path.join("preprocessed", preprocessed_filename), index=False
                )
                st.success(f"Preprocessed file saved as {preprocessed_filename}")

                # Save label encoding mappings
                mapping_folder = "mapping"
                os.makedirs(mapping_folder, exist_ok=True)
                selected_fileA = selected_file.split(".")
                mapping_file = os.path.join(mapping_folder, f"{selected_fileA[0]}.json")
                with open(mapping_file, "w") as f:
                    json.dump(label_encoders, f, indent=4)
                st.success(f"Label encoding mappings saved in {mapping_file}")

            ########adding under sampling and over sampling ###############################
            if not df.empty and not df.columns.empty:
                # Default to the last column if no target column is pre-selected
                default_column = df.columns[-1]  # Automatically selects the last column

                st.write("### Select target column for balancing ")

                # Selectbox to allow user to choose the target column
                target_column = st.selectbox(
                    "Select target column:",
                    df.columns.tolist(),
                    index=(
                        df.columns.get_loc(default_column)
                        if default_column in df.columns
                        else 0
                    ),
                )

                st.write(f"Selected Target Column: **{target_column}**")

                # Display original class distribution
                display_class_distribution(df, target_column)

            ################################ UNDER SAMPLING ###################################################
            col1, col2 = st.columns([1, 1])
            col1.write("### Apply Under Sampling ")
            if col1.button("Under-Sampling  "):
                df_under = under_sample_data(df, target_column)
                st.write("Under-Sampling Applied.")
                display_class_distribution(df_under, target_column)
                st.dataframe(df_under.head())
                # else:
                #     st.warning("Please select a target column first.")

                # Save preprocessed file for under_sampling
                preprocessed_folder_after_under_sampling = "under_sampling"
                os.makedirs(preprocessed_folder_after_under_sampling, exist_ok=True)
                preprocessed_filename_after_under_sampling = f"{selected_file}"
                df.to_csv(
                    os.path.join(
                        "under_sampling", preprocessed_filename_after_under_sampling
                    ),
                    index=False,
                )
                st.success(
                    f"Preprocessed file after under sampling is saved as {preprocessed_filename_after_under_sampling}"
                )

            ####################################### OVER SAMPLING #####################################

            col2.write("### Apply Over Sampling ")
            if col2.button("Over-Sampling  "):
                df_over = over_sample_data(df, target_column)
                st.write("Over-Sampling Applied.")
                display_class_distribution(df_over, target_column)
                st.dataframe(df_over.head())
                # else:
                #     st.warning("Please select a target column first.")

                # Save preprocessed file for over_sampling
                preprocessed_folder_after_over_sampling = "over_sampling"
                os.makedirs(preprocessed_folder_after_over_sampling, exist_ok=True)
                preprocessed_filename_after_over_sampling = f"{selected_file}"
                df.to_csv(
                    os.path.join(
                        "over_sampling", preprocessed_filename_after_over_sampling
                    ),
                    index=False,
                )
                st.success(
                    f"Preprocessed file after over sampling is saved as {preprocessed_filename_after_over_sampling}"
                )
