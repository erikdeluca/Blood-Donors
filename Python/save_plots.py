# python/save_hmm_plots.py

import os
import matplotlib.pyplot as plt
from pyprojroot import here

# data + model loading
import pandas as pd
import polars as pl
import torch
import numpy as np

import hmm_glm_model as hmm_glm
import hmm_glm_plots as hmm_pl
import hmm_glm_viterbi as viterbi
import hmm_glm_prediction as pred

# ----------------------
# Settings
# ----------------------
IMG_DIR = here("thesis/img/hmm")
os.makedirs(IMG_DIR, exist_ok=True)

# Theme colors
SITE_BGCOLOR = "#F4ECE2"
state_cols = ["#8c1c13ff", "#df9457ff", "#86ba90ff", "#54403bff"]

# ----------------------
# Load data (as in your script)
# ----------------------
data = pd.read_csv(here("data/recent_donations.csv"))
df = pl.from_pandas(data)

# collect donation numbers along years
year_cols = sorted([c for c in df.columns if c.startswith("y_")])
T = len(year_cols)
obs = (
    df.select(year_cols)
      .fill_null(0)
      .to_numpy()
      .astype(int)  # (N,T)
)

# prepare fixed covariates for pi
df = df.with_columns([
    (pl.col("gender") == "F").cast(pl.Int8).alias("gender_code"),
    ((pl.col("birth_year") - pl.col("birth_year").mean()) /
     pl.col("birth_year").std()).alias("birth_year_norm")
])

birth_year_mean = df["birth_year"].to_numpy().mean()
birth_year_std = df["birth_year"].to_numpy().std()
birth_year_norm = df["birth_year_norm"].to_numpy()  # (N,)
gender_code     = df["gender_code"].to_numpy()      # (N,)
intercept = np.ones_like(birth_year_norm)

cov_init = np.stack([intercept, birth_year_norm, gender_code], axis=1)  # (N,2)

# dynamic base: ages (N,T) and covid dummy (N,T)
years_num  = np.array([int(c[2:]) for c in year_cols])  # e.g. [2009, …, 2023]
ages       = years_num[None, :] - df["birth_year"].to_numpy()[:, None]  # (N,T)
ages_squared = ages ** 2

covid_mask  = np.isin(years_num, [2020, 2021, 2022]).astype(float)  # (T,)
covid_years = np.tile(covid_mask, (df.height, 1))                   # (N,T)

# age bins over the FULL df (N,T) -> one-hot (N,T,7)
# bins: [0,25), [25,35), [35,45), [45,55), [55,65), [65,75), [75,120]
age_bins = np.array([18, 25, 35, 45, 55, 60, 65, 75])
ages_binned = np.digitize(ages, age_bins, right=False)   # 1..7
n_agebins = len(age_bins) - 1
ages_binned = np.clip(ages_binned, 1, n_agebins)

ages_onehot = np.eye(n_agebins)[ages_binned - 1][:, :, 1:] # drop the baseline 18-24

intercept_tile = np.ones((ages_onehot.shape[0], ages_onehot.shape[1], 1))  

cov_tran = np.concatenate([
    intercept_tile,
    ages_onehot,                       # (N,T,6)
    covid_years[:, :, None]            # (N,T,1) -> expand with none
], axis=2)

# emission covariates (N,T,9): gender + 7 age-bin dummies + covid
gender_code_tile = np.repeat(gender_code[:, None], T, axis=1)       # (N,T)
cov_emission = np.concatenate([
    intercept_tile,
    gender_code_tile[:, :, None],     # (N,T,1)
    ages_onehot,                      # (N,T,7)
    # ages_squared[:, :, None],
    # ages[:, :, None],
    covid_years[:, :, None],          # (N,T,1)
], axis=2)      
# ----------------------
# Load model parameters
# ----------------------
W_pi, W_A, pi_base, A_base, beta_em = hmm_glm.load_hmm_params(here("models/hmm_glm_full.pt"))

# Dummy state/coeff names
S = beta_em.shape[0]
C = beta_em.shape[1]
state_names = [f"State {i}" for i in range(S)]
age_years_levels = ["18-24","25-34","35-44","45-54","55-59","60-64", "+65"] # TODO: #2 improve age bins  
ref_age_level = "18-24"

cov_names_pi = [
    "intercept",
    "birth_year_norm",
    "gender_code"
]


age_fac_cols = hmm_pl.expand_factor_names("age_years", age_years_levels, ref_level=ref_age_level)
cov_names_A = ["intercept"] + age_fac_cols + ["covid_years"]
# cov_names_A = ["intercept"] + ["covid_years"]

em_age_fac_cols = hmm_pl.expand_factor_names("age_years", age_years_levels, ref_level=ref_age_level)
# cov_names_em = ["intercept", "gender"] +  ["covid_years"]
# cov_names_em = ["intercept", "gender", "ages_squared" + "ages" + "covid_years"]
cov_names_em = ["intercept", "gender"] + em_age_fac_cols + ["covid_years"]

# ----------------------
# Helper function to save plots in two themes
# ----------------------
def save_plot(fig, name: str):
    """Save matplotlib figure in white and themed background versions."""
    path_white = os.path.join(IMG_DIR, f"{name}_white.png")
    path_theme = os.path.join(IMG_DIR, f"{name}_theme.png")

    # white / transparent
    fig.savefig(path_white, dpi=300, bbox_inches="tight", transparent=True)
    # theme background
    fig.savefig(path_theme, dpi=300, bbox_inches="tight", transparent=True)

    print(f"Saved: {path_white}, {path_theme}")

# ----------------------
# Plot and save HMM params
# ----------------------
# Get colors for states
colors = hmm_pl.colors_for_states(len(pi_base))

fig, ax = plt.subplots(figsize=(6, 4))
hmm_pl.plot_initial_probs(initial_probs=pi_base, state_names=state_names, colors=colors, ax=ax)
save_plot(fig, "hmm_init")
plt.close(fig)  # chiude la figura per non mostrarla

# -----------------------
# Transition matrix
# -----------------------
fig, ax = plt.subplots(figsize=(6, 5))
hmm_pl.plot_transition_matrix(transitions=A_base, state_names=state_names, ax=ax)
save_plot(fig, "hmm_trans")
plt.close(fig)

# -----------------------
# Emission coefficients
# -----------------------
fig, ax = plt.subplots(figsize=(10, 4))
hmm_pl.plot_emission_coeffs(beta_em=beta_em, state_names=state_names, coeff_names=cov_names_em, colors=colors, ax=ax)
save_plot(fig, "hmm_em")
plt.close(fig)


# to torch
obs_torch       = torch.tensor(obs,          dtype=torch.long)   # (N,T)
cov_init_torch  = torch.tensor(cov_init,     dtype=torch.float)  # (N,2)
cov_tran_torch  = torch.tensor(cov_tran,     dtype=torch.float)  # (N,T,8)
cov_emiss_torch = torch.tensor(cov_emission, dtype=torch.float)  # (N,T,9)

paths = viterbi.viterbi_paths_glm(
    obs_torch,
    cov_init_torch,    # (N,C_pi)
    cov_tran_torch,    # (N,T,C_A)
    cov_emiss_torch,    # (N,T,C_em)
    here("models/hmm_glm_full.pt")
)
switch_rate = (paths[:, 1:] != paths[:, :-1]).any(1).float().mean()
print(f"switch rate = {switch_rate:.2%}")

from plotnine import (
    ggplot, aes, geom_line,
    scale_color_manual, scale_x_continuous, scale_y_continuous,
    labs, theme_minimal, theme, element_text, element_rect
)

# Convert to numpy if torch
paths_np = paths.detach().cpu().numpy() if hasattr(paths, "detach") else np.asarray(paths)
N, T = paths_np.shape
K = int(paths_np.max()) + 1 if paths_np.size else 1

# Counts per state over time -> (K,T)
counts = np.stack([(paths_np == k).sum(axis=0) for k in range(K)], axis=0)
props = counts / np.maximum(counts.sum(axis=0, keepdims=True), 1)  # avoid div-by-zero

rows = []
for k in range(K):
    for t in range(T):
        rows.append({"t": (t + 2009), "state": f"state {k}", "share": float(props[k, t])})
df = pd.DataFrame(rows)

# Base plot
base_plot = (
    ggplot(df, aes("t", "share", color="state"))
    + geom_line(size=1.1)
    + scale_color_manual(values=state_cols, name="state")
    + scale_x_continuous(breaks=list(range(T+2009)))
    + scale_y_continuous(limits=(0, 1))
    + labs(x="year index", y="population share", title="State occupancy over time")
    + theme_minimal()
    + theme(axis_text_x=element_text(rotation=0))
)

# ---- save helper
def save_plotnine(plot, name, width=8, height=4):
    path_white = here(f"thesis/img/hmm/{name}_white.png")
    path_theme = here(f"thesis/img/hmm/{name}_theme.png")

    # White background
    plot.save(path_white, width=width, height=height, dpi=300, transparent=True)

    plot.save(path_theme, width=width, height=height, dpi=300)

    print(f"Saved: {path_white}, {path_theme}")

# ---- save the state occupancy plot
save_plotnine(base_plot, "hmm_state_occupancy")


# Plot W_pi
fig_pi = hmm_pl.plot_W_pi_heat(W_pi, cov_names_pi)
fig_pi.savefig(
    here("thesis/img/hmm/pi_coeff.png"),
    dpi=300,
    bbox_inches="tight",
    transparent=True
)

# Plot W_A
fig_A = hmm_pl.plot_W_A_heat(W_A, cov_names_A)
fig_A.savefig(
    here("thesis/img/hmm/trans_coeff.png"),
    dpi=600,
    bbox_inches="tight",
    transparent=True
)

log_pi0 = np.log(np.clip(pi_base, 1e-30, None))
fig_pi_birth = hmm_pl.plot_pi_vs_cov_orig(
    df=df,
    ages=ages,
    var="birth_year_norm",
    cov_names_pi=cov_names_pi,
    W_pi=W_pi,
    log_pi0=log_pi0,
    x_pi_data=cov_init,
    factor_specs_pi={}
)
fig_pi_birth.savefig(
    here("thesis/img/hmm/pi_age.png"),
    dpi=300,
    bbox_inches="tight",
    transparent=True
)


# prediction
donors_predict=[4011, 3012, 8005, 2002]
for i in donors_predict:
    years_hist = years_num[:-1].tolist()
    year_next  = int(years_num[-1])
    counts_hist = obs_torch[i, :len(years_hist)].detach().cpu().numpy().tolist()

    birth_year_i = int(data["birth_year"][i])
    gender_i     = data["gender"][i]

    prediction = pred.predict_donor(
        birth_year=birth_year_i,
        gender=gender_i,
        history_years=years_hist,
        history_counts=counts_hist,
        next_year=year_next,
        max_k=4,
        model_path="models/hmm_glm_train.pt",
        birth_year_mean=birth_year_mean,
        birth_year_std=birth_year_std
    )

    T_hist = len(years_hist)
    N = obs_torch.shape[0]
    paths = np.zeros((N, T_hist), dtype=int)
    paths[i, :] = np.asarray(prediction["viterbi_states"], dtype=int)

    print("Years:", prediction["years"])
    print("Counts:", prediction["counts"])
    print("Viterbi states:", prediction["viterbi_states"])
    print("Next year:", prediction["next_year"])
    print("Next-state probabilities:", np.round(prediction["next_state_probs"], 3))
    print("Expected next:", round(prediction["expected_next"], 3))
    print("Prob donate next:", round(prediction["prob_donate_next"], 3))
    print("PMF next:", {k: round(v, 4) for k, v in prediction["pmf_next"].items()})

    y_true_next = int(obs_torch[i, T_hist].detach().cpu().item())
    p = hmm_pl.plot_donor_gg(
        idx=i,
        obs_torch=obs_torch[:, :T_hist],
        paths=paths,
        years=years_hist,
        expected_next=prediction["expected_next"],
        y_true_next=y_true_next,
        next_year=year_next,
        y_max=4
    )
    p.save(
        here(f"thesis/img/hmm/predict_{i}.png"),
        dpi=300,
        width=4,
        height=3,
        transparent=True
    )