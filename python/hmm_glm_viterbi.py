import python.hmm_glm_model as hmm_glm
import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.special import logsumexp


def log_softmax_logits(logits, axis=-1):
    return logits - logsumexp(logits, axis=axis, keepdims=True)


def viterbi_paths_glm(obs, x_pi, x_A, x_em, model_path=None, model_params=None):
    """
    Viterbi decoding for HMM with covariate-dependent pi, A,
    and Poisson-GLM emissions.

    Parameters
    ----------
    obs  : (N, T) array-like
        Observed counts.
    x_pi : (N, C_pi) array-like
        Covariates for initial state distribution.
    x_A  : (N, T, C_A) array-like
        Covariates for transition probabilities.
    x_em : (N, T, C_em) array-like
        Covariates for emission GLM (already includes intercept).
    model_path : str or Path, optional
        Path to saved HMM parameters.
    model_params : tuple/dict
        Parameters preloaded (W_pi, W_A, pi_base, A_base, beta_em).
        To use in substitution of 'model_path'.

    Returns
    -------
    paths : (N, T) numpy ndarray
        Most likely latent state sequence for each sequence.
    """
    # ---------------- load parameters ----------------
    if model_params is not None:
        W_pi, W_A, pi_base, A_base, beta_em = model_params
    elif model_path is not None:
        W_pi, W_A, pi_base, A_base, beta_em = hmm_glm.load_hmm_params(model_path)
    else:
        raise ValueError("model_parms and model_path are missings. insert one of them")

    N, T = obs.shape
    K = pi_base.shape[0]

    # ---------------- emission log-probs ----------------
    eta = np.einsum("ntc,kc->ntk", x_em, beta_em)  # (N,T,K)
    emis_log = stats.poisson.logpmf(obs[:, :, np.newaxis], np.exp(eta))  # (N,T,K)

    # ---------------- initial distribution ----------------
    log_pi_base = np.log(pi_base + 1e-30)  # (K,)
    logits0 = log_pi_base[np.newaxis, :] + (x_pi @ W_pi.T)  # (N,K)
    log_pi = log_softmax_logits(logits0, axis=1)  # (N,K)

    delta = log_pi + emis_log[:, 0]  # (N,K)
    psi = np.zeros((N, T, K), dtype=int)

    # ---------------- forward DP ----------------
    log_A_base = np.log(A_base + 1e-30)  # (K,K)

    for t in range(1, T):
        x_t = x_A[:, t, :]  # (N,C_A)

        slope = np.einsum("ijc,nc->nij", W_A, x_t)
        logits = log_A_base[np.newaxis, :, :] + slope  # (N,K,K)
        log_A = log_softmax_logits(logits, axis=2)  # (N,K,K)

        trans_probs = delta[:, :, np.newaxis] + log_A  # (N,K,K)

        # 'dim=1' -> previous state
        score = np.max(trans_probs, axis=1)  # (N,K)
        idx = np.argmax(trans_probs, axis=1)  # (N,K)

        psi[:, t, :] = idx
        delta = score + emis_log[:, t]

    # ---------------- backtracking ----------------
    paths = np.empty((N, T), dtype=int)

    last_state = np.argmax(delta, axis=1)
    paths[:, -1] = last_state

    batch_idx = np.arange(N)

    for t in range(T - 1, 0, -1):
        last_state = psi[batch_idx, t, last_state]
        paths[:, t - 1] = last_state

    return paths


def viterbi_print_donor_info(idx, paths, obs, x_A, x_em, cov_names_A, beta_em):
    T = obs.shape[1]
    states = paths[idx]

    expected_donations = np.zeros(T)
    for t in range(T):
        k = states[t]
        log_mu = np.dot(x_em[idx, t, :], beta_em[k, :])
        expected_donations[t] = np.exp(log_mu)

    df_dict = {
        "Time (t)": np.arange(T),
        "Viterbi State": states,
        "Obs. Donations": obs[idx],
        "Exp. Donations": expected_donations.round(3),
    }

    names_A = list(cov_names_A.values())
    for c, name in enumerate(names_A):
        df_dict[name] = x_A[idx, :, c]

    return pd.DataFrame(df_dict)
