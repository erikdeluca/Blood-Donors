from pyprojroot import here
import pandas as pd
import polars as pl
import numpy as np
import torch

# Shiny plotting helpers
from plotnine import ggplot
import matplotlib.pyplot as plt

# Project modules (already in your repo)
import hmm_glm_model as hmm_glm
import hmm_glm_plots as hmm_pl
import hmm_glm_viterbi as viterbi
import hmm_glm_prediction as pred

# -----------------------------
# Data and model configuration
# -----------------------------
DATA_PATH = here("data/recent_donations.csv")
MODEL_PATH = here("models/hmm_glm_full.pt")  # use full/prod model
COVID_YEARS = (2020, 2021, 2022)

# Theme/colors
STATE_COLS = ["#8c1c13", "#df9457", "#86ba90", "#54403b"]

# -----------------------------
# Load data
# -----------------------------
data_pd = pd.read_csv(DATA_PATH)
df = pl.from_pandas(data_pd)

year_cols = sorted([c for c in df.columns if c.startswith("y_")])
T = len(year_cols)
years_num = np.array([int(c[2:]) for c in year_cols])  # [2009,..,2023]

# Observed counts (N,T)
obs = (
    df.select(year_cols)
      .fill_null(0)
      .to_numpy()
      .astype(int)
)
N = obs.shape[0]

# Fixed covariates for π
df = df.with_columns([
    (pl.col("gender") == "F").cast(pl.Int8).alias("gender_code"),
    ((pl.col("birth_year") - pl.col("birth_year").mean()) / pl.col("birth_year").std()).alias("birth_year_norm"),
])
birth_year_mean = df["birth_year"].to_numpy().mean()
birth_year_std  = df["birth_year"].to_numpy().std()
birth_year_norm = df["birth_year_norm"].to_numpy()
gender_code     = df["gender_code"].to_numpy()
intercept_vec   = np.ones_like(birth_year_norm)

cov_init = np.stack([intercept_vec, birth_year_norm, gender_code], axis=1)  # (N,3)

# Dynamic covariates for transitions (age bins + covid)
ages = years_num[None, :] - df["birth_year"].to_numpy()[:, None]           # (N,T)
covid_mask = np.isin(years_num, list(COVID_YEARS)).astype(float)           # (T,)
covid_years = np.tile(covid_mask, (df.height, 1))                          # (N,T)

# Age bins: 18–24 (baseline), 25–34, 35–44, 45–54, 55–59, 60–64, 65+
age_bins = np.array([18, 25, 35, 45, 55, 60, 65, 75])
ages_binned = np.digitize(ages, age_bins, right=False)     # 1..7
n_agebins = len(age_bins) - 1
ages_binned = np.clip(ages_binned, 1, n_agebins)
ages_onehot = np.eye(n_agebins)[ages_binned - 1][:, :, 1:]  # drop baseline 18–24 → 6 dummies

intercept_tile = np.ones((N, T, 1))
cov_tran = np.concatenate([intercept_tile, ages_onehot, covid_years[:, :, None]], axis=2)  # (N,T,8)

# Emission covariates: intercept + gender + same age dummies + covid → (N,T,9)
gender_code_tile = np.repeat(gender_code[:, None], T, axis=1)
cov_emission = np.concatenate(
    [intercept_tile, gender_code_tile[:, :, None], ages_onehot, covid_years[:, :, None]],
    axis=2
)

# Torch tensors
obs_torch       = torch.tensor(obs,          dtype=torch.long)
cov_init_torch  = torch.tensor(cov_init,     dtype=torch.float32)
cov_tran_torch  = torch.tensor(cov_tran,     dtype=torch.float32)
cov_emiss_torch = torch.tensor(cov_emission, dtype=torch.float32)

# Keep donor registry for UI
donors_df = data_pd[["unique_number", "gender", "birth_year"]].copy()
donors_df["label"] = donors_df.apply(
    lambda r: f'{int(r["unique_number"])} — {r["gender"]} ({int(r["birth_year"])})', axis=1
)
UIDS = donors_df["unique_number"].astype(str).tolist()
UID_TO_IDX = {str(u): i for i, u in enumerate(donors_df["unique_number"].tolist())}

# --------------------------------
# Load trained model + Viterbi all
# --------------------------------
W_pi, W_A, pi_base, A_base, beta_em = hmm_glm.load_hmm_params(MODEL_PATH)
PATHS_ALL = viterbi.viterbi_paths_glm(
    obs_torch,
    cov_init_torch,    # (N,C_pi)
    cov_tran_torch,    # (N,T,C_A)
    cov_emiss_torch,   # (N,T,C_em)
    MODEL_PATH
)  # torch.LongTensor (N,T)

# --------------------------------
# Helpers
# --------------------------------
def donor_index_by_uid(uid_str: str) -> int:
    return UID_TO_IDX[str(uid_str)]

def year_axis():
    return years_num.tolist()

def pmf_to_df(pmf_dict: dict) -> pd.DataFrame:
    # order keys as 0..4, '>=5' if present
    keys = [k for k in ["0","1","2","3","4"] if k in pmf_dict] + [k for k in pmf_dict.keys() if k.startswith(">=")]
    return pd.DataFrame({"k": keys, "prob": [pmf_dict[k] for k in keys]})

def plotnine_to_mpl(g: ggplot):
    """Render a plotnine ggplot into a Matplotlib Figure for Shiny."""
    fig = g.draw()
    # Tight layout to avoid clipping labels
    try:
        fig.tight_layout()
    except Exception:
        pass
    return fig

def build_donor_plot(idx: int, prediction: dict | None = None):
    """
    Returns a Matplotlib Figure showing the donor series colored by latent states,
    optionally annotated with next-year expectation from 'prediction'.
    """
    # Last observed year
    years_hist = years_num.tolist()
    # decoded paths for this donor
    z = PATHS_ALL[idx].cpu().numpy()
    # expected_next and y_true_next
    expected_next = None
    y_true_next = None
    if prediction is not None:
        expected_next = prediction.get("expected_next", None)
        # if we have the next observed year in data, show it (last column)
        if len(years_hist) == obs_torch.shape[1]:
            y_true_next = None  # typically unknown; set if you have hold-out
    g = hmm_pl.plot_donor_gg(
        idx=idx,
        obs_torch=obs_torch,
        paths=PATHS_ALL,
        years=years_hist,
        expected_next=expected_next,
        y_true_next=y_true_next,
        next_year=int(years_hist[-1] + 1),
        y_max=4
    )
    return plotnine_to_mpl(g)

def predict_for_index(i: int) -> dict:
    """Call project’s predictor for a single donor i using their own history."""
    years_hist = years_num[:-1].tolist()  # up to T-1, predict T
    year_next  = int(years_num[-1])
    counts_hist = obs_torch[i, :len(years_hist)].cpu().numpy().astype(int).tolist()
    birth_year_i = int(data_pd["birth_year"].iloc[i])
    gender_i     = str(data_pd["gender"].iloc[i])
    out = pred.predict_donor(
        birth_year=birth_year_i,
        gender=gender_i,
        history_years=years_hist,
        history_counts=counts_hist,
        next_year=year_next,
        max_k=4,
        model_path=str(MODEL_PATH),
        birth_year_mean=birth_year_mean,
        birth_year_std=birth_year_std
    )
    return out

def predict_from_manual(birth_year: int, gender: str, years: list[int], counts: list[int]) -> dict:
    next_year = int(years[-1]) + 1
    out = pred.predict_donor(
        birth_year=birth_year,
        gender=gender,
        history_years=years,
        history_counts=counts,
        next_year=next_year,
        max_k=4,
        model_path=str(MODEL_PATH),
        birth_year_mean=birth_year_mean,
        birth_year_std=birth_year_std
    )
    return out

def parse_counts_csv(counts_text: str) -> list[int]:
    """Parse '0,1,0,2,...' or '0 1 0 2' into a list of ints, ignoring empty tokens."""
    if counts_text is None:
        return []
    txt = counts_text.replace("\n", " ").replace(",", " ")
    tokens = [t for t in txt.split(" ") if t.strip() != ""]
    try:
        vals = [int(t) for t in tokens]
        return vals
    except Exception:
        return []