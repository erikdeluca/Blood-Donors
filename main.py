# app.py
import streamlit as st
import pandas as pd
import warnings


# Importa i tuoi nuovi moduli
from app import config
from app import logic
from app import plots

# --- SETUP ---
# Usa config per le costanti
C = config.CONFIG
warnings.filterwarnings("ignore", message=".*weights_only=False.*")

st.set_page_config(
    page_title="Blood Donors Prediction",
    # page_icon="img/logo_project.png",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- DATA LOADING ---
try:
    # Chiamiamo le funzioni dal modulo logic
    df, tensors_db, choices_map, uid_to_idx, stats = logic.load_and_preprocess_data(
        C["DATA_PATH"], C["COVID_YEARS"], C["AGE_BINS"]
    )
    model_params = logic.load_model_resources(C["MODEL_PATH"])
    beta_em = model_params[4]
except Exception as e:
    st.error(f"Initialization Error: {e}")
    st.stop()

# --- SIDEBAR & LANGUAGES ---
st.sidebar.header("Settings")
selected_lang = st.sidebar.radio("Language / Lingua", ["EN", "IT"], horizontal=True)
T = config.TEXT[selected_lang]
T["tooltip_exp"] = T["metric_exp"]

st.sidebar.markdown("---")
st.sidebar.title(T["nav_title"])
page = st.sidebar.radio(
    "Go to", [T["page_db"], T["page_sim"]], label_visibility="collapsed"
)

# --------------------------
# PAGE 1: DATABASE
# --------------------------
if page == T["page_db"]:
    st.title(T["title_db"])

    selected_label = st.selectbox(T["search_label"], list(choices_map.keys()))
    selected_uid = choices_map[selected_label]
    idx = uid_to_idx[str(selected_uid)]

    # Logica spostata in logic.py
    path_states, pred = logic.get_donor_path_and_pred(
        idx, tensors_db, C["MODEL_PATH"], beta_em
    )

    donor_row = df[df["unique_number"].astype(str) == str(selected_uid)].iloc[0]
    years = stats["years_num"]
    donations = donor_row[stats["year_cols"]].values.astype(int)

    df_long = pd.DataFrame(
        {
            "year": years,
            "donations": donations,
            "state": path_states,
            "state_cat": path_states.astype(int).astype(str),
        }
    )

    with st.expander(T["view_data"]):
        st.markdown(f"**{T['static_info']}**")
        col1, col2, col3 = st.columns(3)
        col1.info(f"ID: {selected_uid}")
        col2.info(f"{T['input_gender']}: {donor_row['gender']}")
        col3.info(f"{T['input_birth']}: {int(donor_row['birth_year'])}")
        st.dataframe(df_long, width="stretch")

    # Plotting spostato in plots.py
    fig = plots.plot_donor_stepped_line(df_long, pred, T)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    last_state_id = int(pred["last_state"])
    last_state_name = T["state_names"].get(last_state_id, f"State {last_state_id}")

    c1, c2, c3 = st.columns(3)
    c1.metric(T["metric_state"], last_state_name)
    c2.metric(T["metric_prob"], f"{pred['prob_donate_next']:.1%}")
    c3.metric(T["metric_exp"], f"{pred['expected_next']:.3f}")


# --------------------------
# PAGE 2: SIMULATOR
# --------------------------
else:
    st.title(T["title_sim"])
    st.caption(T["subtitle_sim"])

    with st.form("sim_form"):
        c1, c2 = st.columns(2)
        with c1:
            sim_gender = st.selectbox(T["input_gender"], ["M", "F"])
        with c2:
            sim_birth = st.number_input(
                T["input_birth"], min_value=1920, max_value=2010, value=1980
            )

        st.subheader(T["input_history"])
        default_years = stats["years_num"]
        input_df = pd.DataFrame(
            [0] * len(default_years), index=default_years, columns=[T["tooltip_don"]]
        )
        edited_df = st.data_editor(input_df.T, width="stretch")
        submitted = st.form_submit_button(T["btn_simulate"])

    if submitted:
        sim_donations = edited_df.iloc[0].values.astype(int)

        if len(sim_donations) != len(stats["years_num"]):
            st.error(T["error_len"])
        else:
            # Logica complessa delegata a logic.py
            sim_tensors = logic.prepare_manual_tensors(
                sim_donations,
                sim_gender,
                sim_birth,
                stats,
                C["COVID_YEARS"],
                C["AGE_BINS"],
            )

            path_states, pred = logic.get_donor_path_and_pred(
                0, sim_tensors, C["MODEL_PATH"], beta_em
            )

            df_sim_long = pd.DataFrame(
                {
                    "year": stats["years_num"],
                    "donations": sim_donations,
                    "state": path_states,
                    "state_cat": path_states.astype(int).astype(str),
                }
            )

            fig = plots.plot_donor_stepped_line(df_sim_long, pred, T)
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

            last_state_id = int(pred["last_state"])
            last_state_name = T["state_names"].get(
                last_state_id, f"State {last_state_id}"
            )

            c1, c2, c3 = st.columns(3)
            c1.metric(T["metric_state"], last_state_name)
            c2.metric(T["metric_prob"], f"{pred['prob_donate_next']:.1%}")
            c3.metric(T["metric_exp"], f"{pred['expected_next']:.3f}")
