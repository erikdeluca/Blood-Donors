import streamlit as st
from app import config

C = config.CONFIG
T = config.TEXT[st.session_state.get("lang", "EN")]


st.title(T["title_app"])
st.markdown("---")
st.write(T["app_description"])
st.markdown("---")

col1, col2 = st.columns([1, 1])

col1.page_link("app/pages/model_on_db.py", label=T["page_db"])
col2.page_link("app/pages/model_on_new_data.py", label=T["page_sim"])
