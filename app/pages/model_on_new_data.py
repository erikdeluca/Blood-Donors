import streamlit as st
import pandas as pd
from app import config
from app import logic
from app import plots

# Lang setup
C = config.CONFIG
T = config.TEXT[st.session_state.get("lang", "EN")]
T["tooltip_exp"] = T.get("metric_exp", "Expected Value")

st.markdown(f"## {T['title_sim']}")
st.caption(T["subtitle_sim"])

# --- Reload data from the cache ---
_, _, _, _, stats = logic.load_and_preprocess_data(
    C["DATA_PATH"], C["COVID_YEARS"], C["AGE_BINS"]
)
model_params = logic.load_model_resources(C["MODEL_PATH"])
beta_em = model_params[4]

# --- UI ---
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
        sim_tensors = logic.prepare_manual_tensors(
            sim_donations,
            sim_gender,
            sim_birth,
            stats,
            C["COVID_YEARS"],
            C["AGE_BINS"],
        )

        path_states, pred = logic.get_donor_path_and_pred(
            0, sim_tensors, model_params, beta_em
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
        