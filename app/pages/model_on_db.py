import streamlit as st
import pandas as pd
from app import config
from app import logic
from app import plots

# language setup
C = config.CONFIG
T = config.TEXT[st.session_state.get("lang", "EN")]
T["tooltip_exp"] = T.get("metric_exp", "Expected Value")

st.markdown(f'## {T["title_db"]}')

# --- reload data from cache ---
df, tensors_db, choices_map, uid_to_idx, stats = logic.load_and_preprocess_data(
    C["DATA_PATH"], C["COVID_YEARS"], C["AGE_BINS"]
)
model_params = logic.load_model_resources(C["MODEL_PATH"])
beta_em = model_params[4]

# --- UI ---
selected_label = st.selectbox(T["search_label"], list(choices_map.keys()))
selected_uid = choices_map[selected_label]
idx = uid_to_idx[str(selected_uid)]

path_states, pred = logic.get_donor_path_and_pred(
    idx, tensors_db, model_params, beta_em
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

fig = plots.plot_donor_stepped_line(df_long, pred, T)
st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

last_state_id = int(pred["last_state"])
last_state_name = T["state_names"].get(last_state_id, f"State {last_state_id}")

c1, c2, c3 = st.columns(3)
c1.metric(T["metric_state"], last_state_name)
c2.metric(T["metric_prob"], f"{pred['prob_donate_next']:.1%}")
c3.metric(T["metric_exp"], f"{pred['expected_next']:.3f}")
