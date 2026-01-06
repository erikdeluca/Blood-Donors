import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import torch
import sys

# project modules
sys.path.append("python")
import hmm_glm_model as hmm_glm  # noqa: E402
import hmm_glm_viterbi as viterbi  # noqa: E402

# ---------------------------------------------------------------------
# 0. Multi-language Configuration
# ---------------------------------------------------------------------
TEXT = {
    "EN": {
        "nav_title": "Navigation",
        "page_db": "Database Analysis",
        "page_sim": "Manual Simulator",
        "loading_data": "Loading data...",
        "loading_model": "Loading model...",
        "title_db": "Donor Database Analysis",
        "title_sim": "Donor Simulator",
        "subtitle_sim": "Simulate a donor's path based on manual inputs.",
        "input_gender": "Gender",
        "input_birth": "Birth Year",
        "input_history": "Donation History (Edit values below)",
        "btn_simulate": "Run Simulation",
        "view_data": "View Raw Data & Sociodemographics",
        "chart_title": "Donation History & Forecast",
        "axis_x": "Year",
        "axis_y": "Donations",
        "legend_trend": "Trend",
        "legend_pred": "Prediction (Next Year)",
        "metric_state": "Last State",
        "metric_prob": "Future Prob.",
        "metric_exp": "Expected (Lambda)",
        "static_info": "Sociodemographic Info",
        "error_len": "History length mismatch.",
        "state_prefix": "State",
        "tooltip_year": "Year",
        "tooltip_don": "Donations",
        "tooltip_state": "State",
        "search_label": "Search Donor",
    },
    "IT": {
        "nav_title": "Navigazione",
        "page_db": "Analisi Database",
        "page_sim": "Simulatore Manuale",
        "loading_data": "Caricamento dati...",
        "loading_model": "Caricamento modello...",
        "title_db": "Analisi Database Donatori",
        "title_sim": "Simulatore Donatore",
        "subtitle_sim": "Simula il percorso di un donatore inserendo i dati manualmente.",
        "input_gender": "Genere",
        "input_birth": "Anno di Nascita",
        "input_history": "Storia Donazioni (Modifica i valori sotto)",
        "btn_simulate": "Esegui Simulazione",
        "view_data": "Vedi Dati Grezzi e Sociodemografici",
        "chart_title": "Storia Donazioni e Previsione",
        "axis_x": "Anno",
        "axis_y": "Donazioni",
        "legend_trend": "Andamento",
        "legend_pred": "Predizione (Anno Prossimo)",
        "metric_state": "Ultimo Stato",
        "metric_prob": "Prob. Futura",
        "metric_exp": "Atteso (Lambda)",
        "static_info": "Info Sociodemografiche",
        "error_len": "Lunghezza storico errata.",
        "state_prefix": "Stato",
        "tooltip_year": "Anno",
        "tooltip_don": "Donazioni",
        "tooltip_state": "Stato",
        "search_label": "Cerca Donatore",
    },
}

# ---------------------------------------------------------------------
# 1. Global Configurations
# ---------------------------------------------------------------------
CONFIG = {
    "DATA_PATH": "data/recent_donations.csv",
    "MODEL_PATH": "models/hmm_glm_full.pt",
    "COVID_YEARS": (2020, 2021, 2022),
    "AGE_BINS": [18, 25, 35, 45, 55, 60, 65, 75],
}


# ---------------------------------------------------------------------
# 2. Load Data (Updated to return stats)
# ---------------------------------------------------------------------
@st.cache_data(show_spinner="Loading...")
def load_and_preprocess_data(data_path, covid_years, age_bins):
    df = pd.read_csv(data_path)

    # Statistics needed for normalization (Saved for Simulator)
    stats = {
        "birth_mean": df["birth_year"].mean(),
        "birth_std": df["birth_year"].std(),
        "year_cols": sorted([c for c in df.columns if c.startswith("y_")]),
    }
    stats["years_num"] = np.array([int(c[2:]) for c in stats["year_cols"]])

    # ... (Processing Standard - same as before) ...
    obs = df[stats["year_cols"]].fillna(0).astype(int).values
    N = obs.shape[0]
    T_len = len(stats["year_cols"])

    gender_code = np.where(df["gender"] == "F", 1, 0)
    birth_year_norm = (df["birth_year"] - stats["birth_mean"]) / stats["birth_std"]

    cov_init = np.stack([np.ones(N), birth_year_norm, gender_code], axis=1)

    ages = stats["years_num"][None, :] - df["birth_year"].values[:, None]
    covid_mask = np.isin(stats["years_num"], list(covid_years)).astype(float)
    covid_years_tile = np.tile(covid_mask, (N, 1))

    n_agebins = len(age_bins) - 1
    ages_binned = np.digitize(ages, age_bins, right=False)
    ages_binned = np.clip(ages_binned, 1, n_agebins)
    ages_onehot = np.eye(n_agebins)[ages_binned - 1][:, :, 1:]

    intercept_tile = np.ones((N, T_len, 1))

    cov_tran = np.concatenate(
        [intercept_tile, ages_onehot, covid_years_tile[:, :, None]], axis=2
    )

    gender_code_tile = np.repeat(gender_code[:, None], T_len, axis=1)
    cov_emiss = np.concatenate(
        [
            intercept_tile,
            gender_code_tile[:, :, None],
            ages_onehot,
            covid_years_tile[:, :, None],
        ],
        axis=2,
    )

    tensors = {
        "obs": torch.tensor(obs, dtype=torch.long),
        "cov_init": torch.tensor(cov_init, dtype=torch.float32),
        "cov_tran": torch.tensor(cov_tran, dtype=torch.float32),
        "cov_emiss": torch.tensor(cov_emiss, dtype=torch.float32),
    }

    choices_map = {
        f"{int(r.unique_number)} - {r.gender} ({int(r.birth_year)})": str(
            r.unique_number
        )
        for r in df.itertuples()
    }
    uid_to_idx = {str(r.unique_number): i for i, r in enumerate(df.itertuples())}

    return df, tensors, choices_map, uid_to_idx, stats


# ---------------------------------------------------------------------
# 3. Model Loader
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading Model...")
def load_model_resources(model_path):
    params = hmm_glm.load_hmm_params(model_path)  # W_pi, W_A, pi_base, A_base, beta_em
    return params


# ---------------------------------------------------------------------
# 4. Core Logic: Prediction & Path Finding
# ---------------------------------------------------------------------
def get_donor_path_and_pred(idx, tensors, model_path, beta_em):
    # Wrapper per gestire sia tensori completi che singoli (simulati)
    # Assicura le dimensioni corrette
    obs = tensors["obs"]
    if obs.dim() == 1:
        obs = obs.unsqueeze(0)

    cov_init = tensors["cov_init"]
    if cov_init.dim() == 1:
        cov_init = cov_init.unsqueeze(0)

    cov_tran = tensors["cov_tran"]
    if cov_tran.dim() == 2:
        cov_tran = cov_tran.unsqueeze(0)

    cov_emiss = tensors["cov_emiss"]
    if cov_emiss.dim() == 2:
        cov_emiss = cov_emiss.unsqueeze(0)

    # Calcolo Viterbi
    paths = viterbi.viterbi_paths_glm(obs, cov_init, cov_tran, cov_emiss, model_path)

    # Estrazione ultimo stato e predizione
    # Se è simulazione, idx è sempre 0
    z_last = int(paths[idx, -1])
    x_em_last = cov_emiss[idx, -1, :].cpu().numpy()

    beta_k = beta_em[z_last, :]
    log_mu = float((x_em_last * beta_k).sum())
    mu = float(np.exp(log_mu))
    prob_donate = float(1.0 - np.exp(-mu))

    return paths[idx, :].cpu().numpy(), {
        "last_state": z_last,
        "expected_next": mu,
        "prob_donate_next": prob_donate,
    }


# ---------------------------------------------------------------------
# 5. Helper: Manual Tensor Builder (Simulatore)
# ---------------------------------------------------------------------
def prepare_manual_tensors(
    donations_list, gender, birth_year, stats, covid_years, age_bins
):
    """
    Costruisce i tensori PyTorch per un singolo donatore simulato (N=1).
    """
    years_num = stats["years_num"]
    T_len = len(years_num)

    # 1. Obs
    obs = np.array(donations_list, dtype=int).reshape(1, T_len)

    # 2. Init
    gender_code = 1 if gender == "F" else 0
    birth_norm = (birth_year - stats["birth_mean"]) / stats["birth_std"]
    cov_init = np.array([[1.0, birth_norm, gender_code]], dtype=np.float32)  # (1, 3)

    # 3. Features per Tran/Emiss
    ages = years_num - birth_year  # (T,)
    n_agebins = len(age_bins) - 1
    ages_binned = np.digitize(ages, age_bins, right=False)
    ages_binned = np.clip(ages_binned, 1, n_agebins)
    # One hot: (T, n_bins-1)
    ages_onehot = np.eye(n_agebins)[ages_binned - 1][:, 1:]

    covid_mask = np.isin(years_num, list(covid_years)).astype(float)  # (T,)

    # Tile (Repeat for time steps if needed, but here we build by stacking time)
    # Shape target: (1, T, Feat)

    # A. Cov Tran [Intercept, Age, Covid]
    intercepts = np.ones((T_len, 1))
    covid_col = covid_mask[:, None]

    # Concateniamo su asse 1 (features) poi aggiungiamo asse 0 (batch)
    ct_t = np.concatenate([intercepts, ages_onehot, covid_col], axis=1)
    cov_tran = ct_t[None, :, :].astype(np.float32)

    # B. Cov Emiss [Intercept, Gender, Age, Covid]
    gender_col = np.full((T_len, 1), gender_code)
    ce_t = np.concatenate([intercepts, gender_col, ages_onehot, covid_col], axis=1)
    cov_emiss = ce_t[None, :, :].astype(np.float32)

    return {
        "obs": torch.tensor(obs, dtype=torch.long),
        "cov_init": torch.tensor(cov_init, dtype=torch.float32),
        "cov_tran": torch.tensor(cov_tran, dtype=torch.float32),
        "cov_emiss": torch.tensor(cov_emiss, dtype=torch.float32),
    }


# ---------------------------------------------------------------------
# 6. Plotting Function (Shared)
# ---------------------------------------------------------------------
def plot_donor_stepped_line(df_long, pred_data, T_dict):
    fig = go.Figure()

    # Linea Grigia Sfondo
    fig.add_trace(
        go.Scatter(
            x=df_long["year"],
            y=df_long["donations"],
            mode="lines",
            line_shape="hvh",
            line=dict(color="lightgrey", width=2),
            name=T_dict["legend_trend"],
            hoverinfo="skip",
        )
    )

    # Punti Colorati
    color_map = {
        "0": "#1f77b4",
        "1": "#ff7f0e",
        "2": "#2ca02c",
        "3": "#d62728",
        "4": "#9467bd",
    }
    for state in sorted(df_long["state_cat"].unique()):
        subset = df_long[df_long["state_cat"] == state]
        fig.add_trace(
            go.Scatter(
                x=subset["year"],
                y=subset["donations"],
                mode="markers",
                marker=dict(
                    size=12,
                    color=color_map.get(state, "black"),
                    line=dict(width=2, color="white"),
                ),
                name=f"{T_dict['state_prefix']} {state}",
                hovertemplate=f"<b>{T_dict['tooltip_year']}:</b> %{{x}}<br><b>{T_dict['tooltip_don']}:</b> %{{y}}<br><b>{T_dict['tooltip_state']}:</b> {state}",
            )
        )

    # Predizione
    last_year = df_long["year"].max()
    last_val = df_long.loc[df_long["year"] == last_year, "donations"].values[0]
    next_year = last_year + 1
    predicted_val = pred_data["expected_next"]

    fig.add_trace(
        go.Scatter(
            x=[last_year, next_year],
            y=[last_val, predicted_val],
            mode="lines",
            line=dict(color="gray", width=2, dash="dash"),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[next_year],
            y=[predicted_val],
            mode="markers",
            marker=dict(
                size=14,
                symbol="square",
                color="#e377c2",
                line=dict(width=2, color="white"),
            ),
            name=T_dict["legend_pred"],
            hovertemplate=f"<b>{T_dict['tooltip_year']}:</b> {next_year}<br><b>{T_dict['tooltip_exp']}:</b> %{{y:.3f}}<extra></extra>",
        )
    )

    fig.update_layout(
        title=T_dict["chart_title"],
        xaxis_title=T_dict["axis_x"],
        yaxis_title=T_dict["axis_y"],
        dragmode=False,
        xaxis=dict(tickmode="linear", showgrid=False, fixedrange=True),
        yaxis=dict(
            dtick=1,
            range=[-0.5, max(4.5, predicted_val + 0.5)],
            showgrid=True,
            gridcolor="#eee",
            fixedrange=True,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    return fig


# ---------------------------------------------------------------------
# 7. MAIN APP
# ---------------------------------------------------------------------

# Init Data & Model
try:
    df, tensors_db, choices_map, uid_to_idx, stats = load_and_preprocess_data(
        CONFIG["DATA_PATH"], CONFIG["COVID_YEARS"], CONFIG["AGE_BINS"]
    )
    model_params = load_model_resources(CONFIG["MODEL_PATH"])
    beta_em = model_params[4]
except Exception as e:
    st.error(f"Initialization Error: {e}")
    st.stop()

# --- SIDEBAR CONFIGURATION ---

# 1. Language Selector (Spostato nella Sidebar)
st.sidebar.header("Settings")
selected_lang = st.sidebar.radio("Language / Lingua", ["EN", "IT"], horizontal=True)
T = TEXT[selected_lang]
# Patch for tooltip key missing in dict above
T["tooltip_exp"] = T["metric_exp"]

st.sidebar.markdown("---")  # Separatore visivo

# 2. Navigation
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

    # Calcolo Percorso
    path_states, pred = get_donor_path_and_pred(
        idx, tensors_db, CONFIG["MODEL_PATH"], beta_em
    )

    # Preparazione DF Long
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

    # View Raw Data (Enhanced)
    with st.expander(T["view_data"]):
        # Sezione 1: Sociodemografiche
        st.markdown(f"**{T['static_info']}**")
        col_sd1, col_sd2, col_sd3 = st.columns(3)
        col_sd1.info(f"ID: {selected_uid}")
        col_sd2.info(f"{T['input_gender']}: {donor_row['gender']}")
        col_sd3.info(f"{T['input_birth']}: {int(donor_row['birth_year'])}")

        # Sezione 2: Tabella
        st.dataframe(df_long, use_container_width=True)

    # Plot
    fig = plot_donor_stepped_line(df_long, pred, T)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Metriche
    c1, c2, c3 = st.columns(3)
    c1.metric(T["metric_state"], f"{T['state_prefix']} {pred['last_state']}")
    c2.metric(T["metric_prob"], f"{pred['prob_donate_next']:.1%}")
    c3.metric(T["metric_exp"], f"{pred['expected_next']:.3f}")

# --------------------------
# PAGE 2: SIMULATOR
# --------------------------
else:
    st.title(T["title_sim"])
    st.caption(T["subtitle_sim"])

    # Inputs Simulator
    with st.form("sim_form"):
        c1, c2 = st.columns(2)
        with c1:
            sim_gender = st.radio(T["input_gender"], ["M", "F"], horizontal=True)
        with c2:
            sim_birth = st.number_input(
                T["input_birth"], min_value=1920, max_value=2010, value=1990
            )

        st.subheader(T["input_history"])
        # Data Editor per inserimento veloce
        default_years = stats["years_num"]
        input_df = pd.DataFrame(
            [0] * len(default_years), index=default_years, columns=[T["tooltip_don"]]
        )

        # Trasponiamo per avere gli anni in orizzontale (più compatto)
        edited_df = st.data_editor(input_df.T, use_container_width=True)

        submitted = st.form_submit_button(T["btn_simulate"])

    if submitted:
        # Recupero valori manuali
        # edited_df è orizzontale, i valori sono nella prima riga
        sim_donations = edited_df.iloc[0].values.astype(int)

        # Controllo lunghezza
        if len(sim_donations) != len(stats["years_num"]):
            st.error(T["error_len"])
        else:
            # Creazione Tensori Manuali
            sim_tensors = prepare_manual_tensors(
                sim_donations,
                sim_gender,
                sim_birth,
                stats,
                CONFIG["COVID_YEARS"],
                CONFIG["AGE_BINS"],
            )

            # Calcolo Modello (Index 0 perché c'è un solo utente nel batch)
            path_states, pred = get_donor_path_and_pred(
                0, sim_tensors, CONFIG["MODEL_PATH"], beta_em
            )

            # Costruzione DF per grafico
            df_sim_long = pd.DataFrame(
                {
                    "year": stats["years_num"],
                    "donations": sim_donations,
                    "state": path_states,
                    "state_cat": path_states.astype(int).astype(str),
                }
            )

            # Grafico
            fig = plot_donor_stepped_line(df_sim_long, pred, T)
            st.plotly_chart(
                fig, use_container_width=True, config={"displayModeBar": False}
            )

            # Metriche
            c1, c2, c3 = st.columns(3)
            c1.metric(T["metric_state"], f"{T['state_prefix']} {pred['last_state']}")
            c2.metric(T["metric_prob"], f"{pred['prob_donate_next']:.1%}")
            c3.metric(T["metric_exp"], f"{pred['expected_next']:.3f}")
