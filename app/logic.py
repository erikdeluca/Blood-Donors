import streamlit as st
import pandas as pd
import numpy as np
import torch
import sys
import os

# project modules path handling
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "python"))
import hmm_glm_model as hmm_glm  # noqa: E402
import hmm_glm_viterbi as viterbi  # noqa: E402


# load data
@st.cache_data(show_spinner="Loading...")
def load_and_preprocess_data(data_path, covid_years, age_bins):
    df = pd.read_csv(data_path)

    # statistics needed for normalization (Saved for Simulator)
    stats = {
        "birth_mean": df["birth_year"].mean(),
        "birth_std": df["birth_year"].std(),
        "year_cols": sorted([c for c in df.columns if c.startswith("y_")]),
    }
    stats["years_num"] = np.array([int(c[2:]) for c in stats["year_cols"]])

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

    # move to tensors
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


# load model parameters
@st.cache_resource(show_spinner="Loading Model...")
def load_model_resources(model_path):
    params = hmm_glm.load_hmm_params(model_path)  # W_pi, W_A, pi_base, A_base, beta_em
    return params


# get the donor path via viterbi and predictions
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

    paths = viterbi.viterbi_paths_glm(obs, cov_init, cov_tran, cov_emiss, model_path)

    # extract last state and compute prediction
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


def prepare_manual_tensors(
    donations_list, gender, birth_year, stats, covid_years, age_bins
):
    """
    Prepares tensors for a manually input donor for simulation.
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
