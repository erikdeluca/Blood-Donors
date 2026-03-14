import pyro
import torch
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.special import logsumexp

import config as C


# utility functions ----------------------------------------------------------------
def _order_by_emissions(beta_em: np.ndarray) -> np.ndarray:
    """
    Sort the states by the intercept
      - state 0 = low (Non-Donor)
      - state 1 = medium (Occasional Donor)
      - state 2 = high (Frequent Donor)
    """
    beta_em = np.asarray(beta_em)

    intercepts = beta_em[:, 0]

    # argsort return the index that sort the array
    s_asc = np.argsort(intercepts)

    return s_asc.astype(int)


def reorder_params(order, pi_base, A_base, W_pi, W_A, beta_em):
    idx = np.asarray(order)
    inv = np.empty_like(idx)
    inv[idx] = np.arange(len(idx))
    pi_base_ = pi_base[idx]
    A_base_ = A_base[idx][:, idx]
    W_pi_ = W_pi[idx]
    W_A_ = W_A[idx][:, idx, :]
    beta_em_ = beta_em[idx]
    return pi_base_, A_base_, W_pi_, W_A_, beta_em_, inv


def log_softmax_logits(logits, axis=-1):
    return logits - logsumexp(logits, axis=axis, keepdims=True)


def softmax_row(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


# export functions ----------------------------------------------------------------


def load_hmm_params(paramfile=None):
    # Add the constraints used for training the model
    safe = [
        torch.distributions.constraints._Real,
        torch.distributions.constraints._Simplex,
        torch.distributions.constraints._GreaterThan,
    ]
    try:
        torch.serialization.add_safe_globals(safe)
    except Exception:
        pass
    if paramfile is not None:
        pyro.clear_param_store()
        pyro.get_param_store().load(paramfile)

    W_pi = pyro.param("W_pi").detach().cpu().numpy()
    W_A = pyro.param("W_A").detach().cpu().numpy()
    pi_base = pyro.param("pi_base_map").detach().cpu().numpy()
    A_base = pyro.param("A_base_map").detach().cpu().numpy()
    beta_em = pyro.param("beta_em").detach().cpu().numpy()

    idx = _order_by_emissions(beta_em)
    pi_base, A_base, W_pi, W_A, beta_em, _ = reorder_params(
        idx, pi_base, A_base, W_pi, W_A, beta_em
    )

    return W_pi, W_A, pi_base, A_base, beta_em


def forward_loglik_cov(obs, x_pi, x_A, x_em, model_path):
    """
    Calculate the log-likelihood forward.
    """
    W_pi, W_A, pi_base, A_base, beta_em = load_hmm_params(model_path)

    N, T = obs.shape

    # Emission log-probs
    log_mu = np.einsum("ntc,kc->ntk", x_em, beta_em)
    emis_log = stats.poisson.logpmf(obs[:, :, np.newaxis], np.exp(log_mu))

    # Initial state log-probs
    log_pi = log_softmax_logits(np.log(pi_base) + x_pi @ W_pi.T, axis=1)
    log_alpha = log_pi + emis_log[:, 0]

    # Forward pass
    log_A_base = np.log(A_base)
    for t in range(1, T):
        x_t = x_A[:, t, :]
        logits = log_A_base[np.newaxis, :, :] + np.sum(
            W_A[np.newaxis, :, :, :] * x_t[:, np.newaxis, np.newaxis, :], axis=-1
        )
        log_A = log_softmax_logits(logits, axis=2)
        log_alpha = (
            logsumexp(log_alpha[:, :, np.newaxis] + log_A, axis=1) + emis_log[:, t]
        )

    return logsumexp(log_alpha, axis=1)


def one_step_ahead(obs, x_pi, x_A, x_em, model_path, pick_one_state=False):
    """
    Computes One-Step-Ahead predictions for the entire dataset.
    Returns step-by-step predicted values, probabilities, and true values.
    """
    # Load parameters
    W_pi, W_A, pi_base, A_base, beta_em = load_hmm_params(model_path)

    N, T = obs.shape
    K = pi_base.shape[0]

    # Global emission log-probs
    log_mu = np.einsum("ntc,kc->ntk", x_em, beta_em)
    emis_log = stats.poisson.logpmf(obs[:, :, np.newaxis], np.exp(log_mu))

    # Initialize alpha
    log_pi = log_softmax_logits(np.log(pi_base + 1e-30) + x_pi @ W_pi.T, axis=1)
    log_alpha = np.empty((N, T, K))
    log_alpha[:, 0] = log_pi + emis_log[:, 0]
    alpha_norm = np.exp(
        log_alpha[:, 0] - logsumexp(log_alpha[:, 0], axis=1, keepdims=True)
    )

    preds_mean, preds_pdon, logscore, y_next_all, alphas_pred = [], [], [], [], []
    log_A_base = np.log(A_base + 1e-30)

    # Step-by-step forward prediction
    for t in range(0, T - 1):
        x_t1 = x_A[:, t + 1, :]
        logits = log_A_base[np.newaxis, :, :] + np.sum(
            W_A[np.newaxis, :, :, :] * x_t1[:, np.newaxis, np.newaxis, :], axis=-1
        )
        log_A = log_softmax_logits(logits, axis=2)

        # Predicted state probabilities at time t+1
        P_next = np.einsum("nk,nkj->nj", alpha_norm, np.exp(log_A))
        alphas_pred.append(P_next)

        # Expected lambda for each state at t+1
        log_mu_t1 = np.einsum("nc,kc->nk", x_em[:, t + 1, :], beta_em)
        lam_ntk = np.exp(log_mu_t1)

        if pick_one_state:
            # Hard assignment: use only the emissions of the most probable state
            final_state = np.argmax(P_next, axis=1)
            Ey = lam_ntk[np.arange(N), final_state]
            P0 = np.exp(-lam_ntk)[np.arange(N), final_state]

            # Log-score based only on the chosen state
            pmf_k = np.stack(
                [
                    np.exp(stats.poisson.logpmf(obs[:, t + 1], lam_ntk[:, k]))
                    for k in range(K)
                ],
                axis=1,
            )
            mix_pmf = np.clip(pmf_k[np.arange(N), final_state], 1e-12, None)
        else:
            # Soft assignment: weighted average across all states (Bayesian mixture)
            Ey = np.sum(P_next * lam_ntk, axis=1)
            P0 = np.sum(P_next * np.exp(-lam_ntk), axis=1)

            # Log-score based on the mixture
            pmf_k = np.stack(
                [
                    np.exp(stats.poisson.logpmf(obs[:, t + 1], lam_ntk[:, k]))
                    for k in range(K)
                ],
                axis=1,
            )
            mix_pmf = np.clip(np.sum(P_next * pmf_k, axis=1), 1e-12, None)

        Pdon = 1.0 - P0
        y_next = obs[:, t + 1]
        ls = np.log(mix_pmf)

        preds_mean.append(Ey)
        preds_pdon.append(Pdon)
        logscore.append(ls)
        y_next_all.append(y_next)

        # Update alpha with the actual observation at t+1 (Filtering)
        log_alpha[:, t + 1] = np.log(P_next + 1e-40) + emis_log[:, t + 1]
        alpha_norm = np.exp(
            log_alpha[:, t + 1] - logsumexp(log_alpha[:, t + 1], axis=1, keepdims=True)
        )
        # log_alpha[:, t + 1] = logsumexp(log_alpha[:, t, :, np.newaxis] + log_A, axis=1) + emis_log[:, t + 1]
        # alpha_norm = np.exp(log_alpha[:, t + 1] - logsumexp(log_alpha[:, t + 1], axis=1, keepdims=True))

    # Return matrices of shape (N, T-1)
    return {
        "y_expected": np.stack(preds_mean, axis=1),
        "p_donate": np.stack(preds_pdon, axis=1),
        "log_score": np.stack(logscore, axis=1),
        "alpha_pred": np.stack(alphas_pred, axis=1),
        "y_true": np.stack(y_next_all, axis=1),
    }


def one_step_ahead_metrics(obs, x_pi, x_A, x_em, model_path, pick_one_state=False):
    """
    Calls one_step_ahead() and computes aggregated predictive metrics (MAE, RMSE, Brier, NLL).
    """
    # 1. Get raw prediction matrices
    results = one_step_ahead(obs, x_pi, x_A, x_em, model_path, pick_one_state)

    # 2. Flatten arrays to compute global metrics over all donors and time steps
    Ey_flat = results["y_expected"].flatten()
    Y_flat = results["y_true"].flatten()
    Pd_flat = results["p_donate"].flatten()
    LS_flat = results["log_score"].flatten()

    # 3. Compute metrics
    mae = np.mean(np.abs(Ey_flat - Y_flat))
    rmse = np.sqrt(np.mean((Ey_flat - Y_flat) ** 2))
    brier = np.mean((Pd_flat - (Y_flat > 0).astype(float)) ** 2)
    nll = -np.mean(LS_flat)

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "Brier(y>0)": round(brier, 4),
        "NLL": round(nll, 4),
    }


def prepare_donor_data(
    birth_year,
    gender,
    first_donation_year,
    history_years,
    history_counts,
    birth_year_mean,
    birth_year_std,
    first_don_mean,
    first_don_std,
    donation_py_init=0,
    next_year=None,
    covid_years=(2020, 2021, 2022),
):
    """
    Transform the donor data and its donation time serie to be ready for the model HMM-GLM.
    It normalize the continuous variables given the mean and sd obtained in data wrangling of the model.

    """
    years = np.array(history_years, dtype=int)
    yvals = np.array(history_counts, dtype=int)
    if next_year is None:
        next_year = int(years[-1] + 1)

    g_code = 1.0 if gender.upper().startswith("F") else 0.0
    by_norm = (birth_year - birth_year_mean) / birth_year_std
    fd_norm = (first_donation_year - first_don_mean) / first_don_std
    x_pi = np.array([[fd_norm, by_norm, g_code]], dtype=np.float32)

    hist_donations_py = np.concatenate(([donation_py_init], yvals[:-1]))
    next_donation_py = np.array([yvals[-1]])

    def get_dynamic_covs(target_years, don_py_array):
        target_years = np.atleast_1d(target_years)
        don_py_array = np.atleast_1d(don_py_array)[:, None]  # (T, 1)
        ages = target_years - birth_year

        age_bins = np.array([18, 25, 35, 45, 55, 60, 65, 75])
        n_agebins = len(age_bins) - 1
        ages_binned = np.clip(np.digitize(ages, age_bins, right=False), 1, n_agebins)
        ages_onehot = np.eye(n_agebins)[ages_binned - 1][:, 1:]

        covid_flag = np.isin(target_years, covid_years).astype(float)[:, None]
        intercept = np.ones((len(target_years), 1), dtype=np.float32)
        gender_col = np.full((len(target_years), 1), g_code, dtype=np.float32)

        A_covs = np.concatenate(
            [
                ages_onehot.astype(np.float32),
                covid_flag.astype(np.float32),
                don_py_array.astype(np.float32),
            ],
            axis=1,
        )

        em_covs = np.concatenate(
            [
                intercept,
                gender_col,
                ages_onehot.astype(np.float32),
                covid_flag.astype(np.float32),
            ],
            axis=1,
        )

        return A_covs, em_covs

    x_A_hist, x_em_hist = get_dynamic_covs(years, hist_donations_py)
    x_A_next, x_em_next = get_dynamic_covs(next_year, next_donation_py)

    obs = yvals[np.newaxis, :]

    return obs, x_pi, x_A_hist, x_em_hist, x_A_next[0], x_em_next[0], int(next_year)


def prepare_donor_data_thesis(
    birth_year,
    gender,
    history_years,
    history_counts,
    birth_year_mean,
    birth_year_std,
    next_year=None,
    covid_years=(2020, 2021, 2022),
):
    """
    Transform donor data for the OLD HMM-GLM model, the model used in the thesis.
    (Without first_donation_year and previous_donation logic).
    """
    years = np.array(history_years, dtype=int)
    yvals = np.array(history_counts, dtype=int)
    if next_year is None:
        next_year = int(years[-1] + 1)

    g_code = 1.0 if gender.upper().startswith("F") else 0.0
    by_norm = (birth_year - birth_year_mean) / birth_year_std
    x_pi = np.array([[by_norm, g_code]], dtype=np.float32)

    def get_dynamic_covs(target_years):
        target_years = np.atleast_1d(target_years)
        ages = target_years - birth_year

        age_bins = np.array([18, 25, 35, 45, 55, 60, 65, 75])
        n_agebins = len(age_bins) - 1
        ages_binned = np.clip(np.digitize(ages, age_bins, right=False), 1, n_agebins)
        ages_onehot = np.eye(n_agebins)[ages_binned - 1][:, 1:]

        covid_flag = np.isin(target_years, covid_years).astype(float)[:, None]
        intercept = np.ones((len(target_years), 1), dtype=np.float32)
        gender_col = np.full((len(target_years), 1), g_code, dtype=np.float32)

        A_covs = np.concatenate(
            [ages_onehot.astype(np.float32), covid_flag.astype(np.float32)], axis=1
        )

        em_covs = np.concatenate(
            [
                intercept,
                gender_col,
                ages_onehot.astype(np.float32),
                covid_flag.astype(np.float32),
            ],
            axis=1,
        )

        return A_covs, em_covs

    x_A_hist, x_em_hist = get_dynamic_covs(years)
    x_A_next, x_em_next = get_dynamic_covs(next_year)

    obs = yvals[np.newaxis, :]

    return obs, x_pi, x_A_hist, x_em_hist, x_A_next[0], x_em_next[0], int(next_year)


def predict_hmm_donor(
    obs,
    x_pi,
    x_A_hist,
    x_em_hist,
    x_A_next,
    x_em_next,
    model_params,
    years: np.array,
    max_k=4,
):
    """
    Execute the Forward algorithm (for the prediction) and Viterbi's algorithm (for the historical states)
    """
    W_pi, W_A, pi_base, A_base, beta_em = model_params
    K = pi_base.shape[0]

    # acept every dimension and the flat to 1D or 2D
    obs = np.atleast_1d(obs).squeeze()  # (T,)
    x_pi = np.atleast_1d(x_pi).squeeze()  # (C_pi,)
    x_A_hist = np.atleast_2d(x_A_hist)  # (T, C_A)
    x_em_hist = np.atleast_2d(x_em_hist)  # (T, C_em)
    x_A_next = np.atleast_1d(x_A_next).squeeze()  # (C_A,)
    x_em_next = np.atleast_1d(x_em_next).squeeze()  # (C_em,)

    T_hist = obs.shape[0]

    # historical emission probabilities
    log_mu = x_em_hist @ beta_em.T
    emis_log = stats.poisson.logpmf(obs[:, np.newaxis], np.exp(log_mu))  # (T, K)

    # initialization Forward and Viterbi
    log_pi0 = np.log(pi_base + 1e-30) + (W_pi @ x_pi)  # (K,)
    log_pi = log_softmax_logits(log_pi0, axis=0)  # (K,)

    log_alpha = np.empty((T_hist, K), dtype=float)
    log_alpha[0] = log_pi + emis_log[0]

    delta = log_pi + emis_log[0]
    psi = np.zeros((T_hist, K), dtype=int)

    # iteration: Forward + Viterbi
    log_A_base = np.log(A_base + 1e-30)
    for t in range(1, T_hist):
        x_t = x_A_hist[t]  # (C_A,)

        # transiction at time t: (K, K)
        logits = log_A_base + (W_A @ x_t)
        log_A = log_softmax_logits(logits, axis=1)

        # Forward
        log_alpha[t] = (
            logsumexp(log_alpha[t - 1][:, np.newaxis] + log_A, axis=0) + emis_log[t]
        )

        # Viterbi
        trans_probs = delta[:, np.newaxis] + log_A
        psi[t, :] = np.argmax(trans_probs, axis=0)
        delta = np.max(trans_probs, axis=0) + emis_log[t]

    # normalization forward
    alpha_T_log = log_alpha[-1] - logsumexp(log_alpha[-1])
    alpha_T = np.exp(alpha_T_log)  # (K,)

    # backtracking for Viterbi
    v_path = np.zeros(T_hist, dtype=int)
    last_state = np.argmax(delta)
    v_path[-1] = last_state
    for t in range(T_hist - 1, 0, -1):
        last_state = psi[t, last_state]
        v_path[t - 1] = last_state

    # prediction for the next year
    logits_next = np.log(A_base + 1e-30) + (W_A @ x_A_next)
    A_next = softmax_row(logits_next)
    p_next = alpha_T @ A_next  # (K,)

    lam_next = np.exp(beta_em @ x_em_next)  # (K,)
    expected_next = float((p_next * lam_next).sum())
    prob_donate_next = 1.0 - float((p_next * np.exp(-lam_next)).sum())

    # PMF Poisson
    pmf0k = np.array(
        [(p_next * stats.poisson.pmf(k, lam_next)).sum() for k in range(max_k + 1)]
    )
    tail = float(max(0.0, 1.0 - pmf0k.sum()))

    pmf_dict = {str(k): round(float(pmf0k[k]), 4) for k in range(max_k)}
    pmf_dict[f">={max_k}"] = round(float(pmf0k[max_k]) + tail, 4)

    v_path_names = [C.STATE_NAMES[s] for s in v_path]

    df_history = pd.DataFrame(
        [obs.tolist(), v_path_names],
        columns=years[:T_hist],
        index=["Donations", "State"],
    )

    next_yr_val = years[T_hist] if len(years) > T_hist else years[-1] + 1

    return {
        "history_df": df_history,
        "next_year": int(next_yr_val),
        "next_state_probs": [round(p, 4) for p in p_next.tolist()],
        "expected_next": round(expected_next, 4),
        "prob_donate_next": round(prob_donate_next, 4),
        "pmf_next": pmf_dict,
    }


def build_hmm_metrics(y_pred: np.array, y_true: np.array):
    """
    Calculate the metrics for comparing the HMM against a GLM
    """
    return {
        "pred mean": np.mean(y_pred),
        "obs mean": np.mean(y_true),
        "MSE": np.mean((y_pred - y_true) ** 2),
        "Accuracy (round)": np.mean(np.round(y_pred) == y_true),
    }


def pct_rel_diff(val_base, val_new):
    return 100.0 * (val_new - val_base) / (abs(val_base) + 1e-12)


def print_comparison_table(metrics_glm, metrics_hmm):
    """
    print a table
    """
    print(
        f"{'Metric':<20} {'GLM':>12} {'HMM':>12} {'Diff (HMM-GLM)':>16} {'Rel diff %':>12}"
    )
    print("-" * 75)

    order = ["pred mean", "obs mean", "MSE", "Accuracy (round)"]
    for key in order:
        g = metrics_glm[key]
        h = metrics_hmm[key]

        if key == "Accuracy (round)":
            g_disp = 100.0 * g
            h_disp = 100.0 * h
            abs_disp = h_disp - g_disp
            rel_pct = pct_rel_diff(g, h)
            print(
                f"{key:<20} {g_disp:11.2f}% {h_disp:11.2f}% {abs_disp:13.2f} pp {rel_pct:11.2f}%"
            )

        elif key == "MSE":
            abs_diff = h - g
            rel_pct = pct_rel_diff(g, h)
            print(f"{key:<20} {g:12.4f} {h:12.4f} {abs_diff:16.4f} {rel_pct:11.2f}%")

        else:
            abs_diff = h - g
            rel_pct = pct_rel_diff(g, h)
            print(f"{key:<20} {g:12.2f} {h:12.2f} {abs_diff:16.2f} {rel_pct:11.2f}%")
