# ML-Automation

# 🧠 ML Automation Studio

An end-to-end **Machine Learning Automation Platform** built with Streamlit that enables users to upload datasets, preprocess data, train models, and generate real-time predictions — all through an intuitive UI.

---

## 🚀 Live Demo

👉 SOON

---

## ✨ Features

### 📂 File Management

* Upload CSV datasets
* Preview data instantly
* Manage and delete uploaded files

---

### 🛠️ Data Preprocessing

* Automatic handling of missing values
* Encoding categorical variables
* Feature scaling

---

### 🤖 Model Training

* Train multiple ML models
* Supports both **classification & regression**
* Automatic evaluation metrics (Accuracy, RMSE, etc.)
* Save/Download trained models as `.pkl` files

---

### ⚙️ Frontend Code Generator

* Automatically generates **ready-to-use frontend code**
* Helps in deploying ML models quickly
* Reduces manual coding effort for prediction interfaces

---

### 🧠 Frontend UI (Live Prediction)

* USE Saved Models/Upload trained `.pkl` model
* Auto-detect feature names from model
* Dynamic input form generation
* Real-time predictions
* Probability visualization (for classification models)

---

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

FileManagement <img width="1836" height="872" alt="image" src="https://github.com/user-attachments/assets/243a5b1c-97c0-4d42-942f-5c0df3a0b9fa" />

Preprocess Data <img width="1811" height="764" alt="image" src="https://github.com/user-attachments/assets/f4d98084-db77-402c-a916-7941833cacc5" />
<img width="1804" height="507" alt="image" src="https://github.com/user-attachments/assets/c5cf9234-0ac3-4485-89b7-b1296fd0accb" />
<img width="1782" height="440" alt="image" src="https://github.com/user-attachments/assets/dffeac72-131b-4c36-bd26-f263048b8198" />
<img width="1786" height="875" alt="image" src="https://github.com/user-attachments/assets/a26ae5df-1c4c-4e11-8c73-427f5408aaee" />
<img width="1782" height="891" alt="image" src="https://github.com/user-attachments/assets/aa260cb9-e74c-415d-bdee-2d7ce56a6633" />
Before UnderSampling
<img width="1787" height="644" alt="image" src="https://github.com/user-attachments/assets/006d89d5-2c11-49d3-8d17-6f53c5a19aae" />
<img width="1774" height="634" alt="image" src="https://github.com/user-attachments/assets/35b9c475-f282-4361-a059-14a2faa2dd36" />
After UnderSampling
<img width="1770" height="616" alt="image" src="https://github.com/user-attachments/assets/e8b043db-82b4-4d46-a600-8ed6a28e580b" />
<img width="1702" height="866" alt="image" src="https://github.com/user-attachments/assets/49c00e31-205d-4403-8594-b250181c506b" />
Model Training <img width="1821" height="870" alt="image" src="https://github.com/user-attachments/assets/de07c2e6-abf7-40d5-8dc5-b63583ed56ac" />










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

