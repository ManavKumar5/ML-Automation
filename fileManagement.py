import streamlit as st
import os
import pandas as pd

SAVE_DIR = "data"
os.makedirs(SAVE_DIR, exist_ok=True)


def upload_csv():
    uploaded_file = st.file_uploader("📎Upload a new document (CSV):", type=["csv"])
    # type=["csv"] --> Accepts only CSV files
    if uploaded_file is not None:
        file_path = os.path.join(SAVE_DIR, uploaded_file.name)
        # uploaded_file.name --> retrieves the original filename
        # os.path.join --> creates valid path for different Operating System
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            # getbuffer-->extracts the file contents as bytes
        st.success("  File Uploaded Successfully")
        # Show a preview of the CSV file
        df = pd.read_csv(file_path)
        st.dataframe(df.head())


def listFiles():
    st.subheader("Uploaded Files ")
    files = os.listdir(SAVE_DIR)
    # os.listdir(SAVE_DIR) --> return list of all files from SAVE_DIR directory
    if files:
        for file in files:
            col1, col2 = st.columns([4, 1])
            # splits screen in two column with ratio 4:1
            # col1 --> takes 4/5 width of screen
            # col2 --> takes 1/5 width of screen
            col1.markdown(f" **{file}**")
            # displays file in col1
            if col2.button("delete ", key=file):
                # creates delete button in front of every file in col2
                # key is used for giving unique identity to each file in case of multiple files
                os.remove(os.path.join(SAVE_DIR, file))
                # os.remove --> deletes the selected file from SAVE_DIR
                # os.path.join(SAVE_DIR, file) -- safely gives path to delete file
                st.success(f"Deleted {file}")
                st.rerun()
    else:
        st.info("ℹ No files uploaded")
    return files
