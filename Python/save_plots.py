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

# ----------------------
# Settings
# ----------------------
IMG_DIR = here("thesis/img/hmm")
os.makedirs(IMG_DIR, exist_ok=True)

# Theme colors
SITE_BGCOLOR = "#F4ECE2"

# ----------------------
# Load data (as in your script)
# ----------------------
data = pd.read_csv(here("data/recent_donations.csv"))
df = pl.from_pandas(data)

year_cols = sorted([c for c in df.columns if c.startswith("y_")])
T = len(year_cols)
obs = (
    df.select(year_cols)
      .fill_null(0)
      .to_numpy()
      .astype(int)
)

# covariates (shortened, you can keep the full preprocessing pipeline if needed)
df = df.with_columns([
    (pl.col("gender") == "F").cast(pl.Int8).alias("gender_code"),
    ((pl.col("birth_year") - pl.col("birth_year").mean()) / pl.col("birth_year").std()).alias("birth_year_norm")
])

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
    fig.savefig(path_white, dpi=300, bbox_inches="tight", facecolor="white")
    # theme background
    fig.savefig(path_theme, dpi=300, bbox_inches="tight", facecolor=SITE_BGCOLOR)

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
        rows.append({"t": t, "state": f"state {k}", "share": float(props[k, t])})
df = pd.DataFrame(rows)

# Base plot
base_plot = (
    ggplot(df, aes("t", "share", color="state"))
    + geom_line(size=1.1)
    + scale_color_manual(values=state_cols, name="state")
    + scale_x_continuous(breaks=list(range(T)))
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
    plot.save(path_white, width=width, height=height, dpi=300)

    # Site theme background
    plot_theme = plot + theme(
        figure_background=element_rect(fill="#F4ECE2", color=None),
        panel_background=element_rect(fill="#F4ECE2", color=None),
    )
    plot_theme.save(path_theme, width=width, height=height, dpi=300)

    print(f"Saved: {path_white}, {path_theme}")

# ---- save the state occupancy plot
save_plotnine(base_plot, "hmm_state_occupancy")