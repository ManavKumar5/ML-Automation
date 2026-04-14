# ML-Automation

# 🧠 ML Automation Studio

An end-to-end **Machine Learning Automation Platform** built with Streamlit that enables users to upload datasets, preprocess data, train models, and generate real-time predictions — all through an intuitive UI.

---

## 🚀 Live Demo

👉 https://querymind-14.streamlit.app/

---

## ✨ Features

### 📂 File Management

* Upload CSV datasets
* Preview data instantly
* Manage and delete uploaded files

### 🛠️ Data Preprocessing

* Automatic handling of missing values
* Encoding categorical variables
* Feature scaling

### 🤖 Model Training

* Train multiple ML models
* Supports both **classification & regression**
* Automatic evaluation metrics (Accuracy, RMSE, etc.)

### 🧠 Frontend UI (Live Prediction)

* Upload trained `.pkl` model
* Auto-detect features from model
* Real-time prediction interface
* Probability visualization (for classification)

---

## 🧱 Tech Stack

* **Frontend:** Streamlit
* **Backend:** Python
* **Libraries:**

  * Pandas
  * NumPy
  * Scikit-learn
  * Pickle

---

## 📸 Screenshots

<img width="1836" height="872" alt="image" src="https://github.com/user-attachments/assets/243a5b1c-97c0-4d42-942f-5c0df3a0b9fa" />


---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/ml-automation.git
cd ml-automation
```

### 2️⃣ Create virtual environment

```bash
python -m venv venv
source venv/bin/activate   # For Mac/Linux
venv\Scripts\activate      # For Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the app

```bash
streamlit run app.py/ python -m streamlit run app.py
```

---

## 📁 Project Structure

```
ml-automation/
│
├── app.py                  # Main Streamlit app
├── fileManagement.py       # File upload & management
├── Preprocess.py           # Data preprocessing
├── modelTraining.py        # Model training logic
├── Frontend_app.py         # code for the specific model
├── frontend_ui.py          # Live prediction UI
├── models/                 # Saved trained models
├── data/                   # Uploaded datasets
└── requirements.txt
```

---

## 🎯 Use Cases

* Automate ML workflows without coding
* Rapid prototyping for ML models
* Portfolio project for Data Science roles
* Internal tools for data teams

---

## 💡 Future Enhancements

*  Advanced visualizations (Plotly dashboards)
*  AutoML model selection
*  Cloud model deployment
*  User authentication system
*  Model performance tracking

---

## 📬 Contact

**Manav Kumar**
Portfolio: https://manavkumar5.github.io
LinkedIn: ("www.linkedin.com/in/kumarmannavv")

---

## ⭐ Support

If you found this project helpful, please ⭐ the repository!

---

