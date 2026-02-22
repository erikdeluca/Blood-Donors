import streamlit as st
import warnings
from app import config
from app import logic

# --- GLOBAL SETUP ---
C = config.CONFIG
warnings.filterwarnings("ignore", message=".*weights_only=False.*")

st.set_page_config(
    page_title="Blood Donors Prediction",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- PRE-LOADING IN CACHE ---
try:
    with st.spinner("Initializing system and loading AI models..."):
        logic.load_and_preprocess_data(C["DATA_PATH"], C["COVID_YEARS"], C["AGE_BINS"])
        logic.load_model_resources(C["MODEL_PATH"])
except Exception as e:
    st.error(f"Initialization Error: {e}")
    st.stop()

if "lang" not in st.session_state:
    st.session_state.lang = "EN"

st.sidebar.header("Global Settings")
selected_lang = st.sidebar.radio(
    "Language / Lingua",
    ["EN", "IT"],
    horizontal=True,
    index=0 if st.session_state.lang == "EN" else 1,
)
st.session_state.lang = selected_lang
T = config.TEXT[st.session_state.get("lang", "EN")]

pg = st.navigation(
    [
        st.Page("app/pages/home.py", title="Home", icon="🏠"),
        st.Page("app/pages/model_on_db.py", title=T["page_db"], icon="📊"),
        st.Page("app/pages/model_on_new_data.py", title=T["page_sim"], icon="🧪"),
    ]
)
pg.run()
