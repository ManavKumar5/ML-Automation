import streamlit as st
from fileManagement import upload_csv, listFiles
from Preprocess import preprocess_data
from modelTraining import model_training
from frontendUI import frontend_ui
from Frontend_app import display_frontend_code

import streamlit as st

st.set_page_config(
    page_title="AI ML Studio", layout="wide", initial_sidebar_state="expanded"
)

st.markdown(
    """
<style>

/* ===== MAIN APP BACKGROUND (QueryMind Style) ===== */
.stApp {
    background: radial-gradient(circle at 20% 20%, rgba(139, 92, 246, 0.25), transparent 40%),
                radial-gradient(circle at 80% 0%, rgba(168, 85, 247, 0.2), transparent 40%),
                radial-gradient(circle at 50% 100%, rgba(124, 58, 237, 0.2), transparent 40%),
                linear-gradient(135deg, #0B0F19 0%, #05070F 100%);
    color: #E5E7EB;
}

/* ===== SIDEBAR ===== */
section[data-testid="stSidebar"] {
    background: rgba(10, 12, 25, 0.9);
    backdrop-filter: blur(10px);
    border-right: 1px solid rgba(139, 92, 246, 0.2);
}

/* ===== GLASS EFFECT CARDS ===== */
.custom-card {
    background: rgba(17, 24, 39, 0.6);
    backdrop-filter: blur(12px);
    padding: 20px;
    border-radius: 16px;
    border: 1px solid rgba(139, 92, 246, 0.2);
    box-shadow: 0 8px 30px rgba(139, 92, 246, 0.15);
}

/* ===== BUTTONS ===== */
.stButton>button {
    background: linear-gradient(135deg, #8B5CF6, #6D28D9);
    color: white;
    border-radius: 12px;
    padding: 10px 18px;
    border: none;
    font-weight: 500;
    transition: all 0.3s ease;
}

.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 20px rgba(139, 92, 246, 0.6);
}

/* ===== INPUTS ===== */
.stTextInput input, .stNumberInput input {
    background-color: rgba(17, 24, 39, 0.7);
    color: white;
    border-radius: 10px;
    border: 1px solid rgba(139, 92, 246, 0.3);
}

/* ===== METRICS ===== */
[data-testid="metric-container"] {
    background: rgba(17, 24, 39, 0.6);
    border-radius: 16px;
    padding: 15px;
    border: 1px solid rgba(139, 92, 246, 0.2);
}

/* ===== HEADINGS ===== */
h1, h2, h3 {
    color: #F9FAFB;
}

/* ===== SUBTLE GLOW TEXT ===== */
h1 {
    text-shadow: 0px 0px 20px rgba(139, 92, 246, 0.4);
}

/* ===== REMOVE DEFAULT STREAMLIT WHITE BLOCK ===== */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<hr style="border: 1px solid rgba(139, 92, 246, 0.3);">
""",
    unsafe_allow_html=True,
)


# 🌌 Purple SaaS Theme
st.markdown(
    """
<style>

/* ===== GLOBAL ===== */
.stApp {
    background: linear-gradient(135deg, #0B0F19, #111827);
    color: #E5E7EB;
    font-family: 'Inter', sans-serif;
}

/* ===== SIDEBAR ===== */
section[data-testid="stSidebar"] {
    background: #070B14;
    border-right: 1px solid #1F2937;
}

/* Sidebar text */
section[data-testid="stSidebar"] * {
    color: #D1D5DB !important;
}

/* ===== BUTTONS ===== */
.stButton>button {
    background: linear-gradient(135deg, #8B5CF6, #6D28D9);
    color: white;
    border-radius: 12px;
    padding: 10px 18px;
    border: none;
    font-weight: 500;
    transition: 0.3s ease;
}

.stButton>button:hover {
    transform: scale(1.03);
    background: linear-gradient(135deg, #7C3AED, #5B21B6);
}

/* ===== INPUTS ===== */
.stTextInput input, .stNumberInput input {
    background-color: #111827;
    color: white;
    border-radius: 10px;
    border: 1px solid #374151;
}

/* ===== SELECTBOX ===== */
.stSelectbox div {
    background-color: #111827 !important;
    border-radius: 10px;
}

/* ===== METRICS ===== */
[data-testid="metric-container"] {
    background: #111827;
    border-radius: 16px;
    padding: 18px;
    border: 1px solid #1F2937;
}

/* ===== CARDS ===== */
.custom-card {
    background: #111827;
    padding: 20px;
    border-radius: 16px;
    border: 1px solid #1F2937;
    box-shadow: 0px 4px 20px rgba(139, 92, 246, 0.15);
}

/* ===== HEADINGS ===== */
h1, h2, h3 {
    color: #F9FAFB;
}

/* ===== SCROLLBAR ===== */
::-webkit-scrollbar {
    width: 6px;
}
::-webkit-scrollbar-thumb {
    background: #8B5CF6;
    border-radius: 10px;
}

</style>
""",
    unsafe_allow_html=True,
)


st.set_page_config(
    page_title="ML Automation",
    layout="wide",
    # page_icon="",
    initial_sidebar_state="expanded",
)

# st.sidebar.image("")
# page = st.sidebar.selectbox(
#     "Select an option:", ["File Management", "Preprocess Data", "Model Training"]
# )

# Sidebar Menu Options with Icons
options = {
    "File Management": "📂 File Management",
    "Preprocess Data": "🛠️ Preprocess Data",
    "Model Training": "🤖 Model Training",
    "Frontend Application": " Frontend Application",
    "Frontedn UI": "</> Frontend UI",
}

# Sidebar Navigation
selected_option = st.sidebar.selectbox("📌 **Select an option:**", options.values())

# Reverse mapping to get the key (without emoji) for processing
page = [key for key, value in options.items() if value == selected_option][0]


if page == "File Management":
    st.title("📂File Management System")
    upload_csv()
    listFiles()

elif page == "Preprocess Data":
    st.title("🛠️ Data Preprocessing")
    preprocess_data()

elif page == "Model Training":
    st.title("🤖 Training Your Model")
    model_training()

elif page == "Frontend Application":
    st.title("Frontend Application for Model")
    display_frontend_code()

elif selected_option == "</> Frontend UI":
    frontend_ui()


# ... other imports

# pages = {
#     "Frontend UI": frontend_ui,
#     # ... other pages
# }

# selected_page = st.sidebar.selectbox("Select an option:", list(pages.keys()))
# pages[selected_page]()
