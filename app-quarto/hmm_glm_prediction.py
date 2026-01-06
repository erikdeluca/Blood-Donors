import torch
import pyro.distributions as dist
import numpy as np
from pyprojroot import here

import hmm_glm_model as hmm_glm
import hmm_glm_viterbi as viterbi


def _softmax_row(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


@torch.no_grad()
def predict_donor(
    birth_year: int,
    gender: str,
    history_years: list,
    history_counts: list,
    next_year: int | None = None,
    max_k: int = 4,
    covid_years=(2020, 2021, 2022),
    model_path="models/hmm_glm_train.pt",
    birth_year_mean=float,
    birth_year_std=float,
):
    years = np.array(history_years, dtype=int)
    yvals = np.array(history_counts, dtype=int)
    if next_year is None:
        next_year = int(years[-1] + 1)

    # load the model and reorder the states for a better readability
    W_pi, W_A, pi_base, A_base, beta_em = hmm_glm.load_hmm_params(here(model_path))
    order = hmm_glm._simple_order_by_pi0(pi_base)
    pi_base, A_base, W_pi, W_A, beta_em, inv = hmm_glm.reorder_params(
        order, pi_base, A_base, W_pi, W_A, beta_em
    )

    K = int(pi_base.shape[0])

    g_code = 1.0 if gender.upper().startswith("F") else 0.0
    by_norm = (birth_year - birth_year_mean) / birth_year_std

    age_bins = np.array([18, 25, 35, 45, 55, 60, 65, 75])
    ages_arr = years - birth_year
    ages_binned = np.digitize(ages_arr, age_bins, right=False)
    n_agebins = len(age_bins) - 1
    ages_binned = np.clip(ages_binned, 1, n_agebins)
    ages_onehot = np.eye(n_agebins)[ages_binned - 1][:, 1:]
    covid_flag = np.isin(years, covid_years).astype(float)[:, None]

    x_pi_np = np.array([[1.0, by_norm, g_code]], dtype=np.float32)
    intercept = np.ones((len(years), 1), dtype=np.float32)

    x_A_hist = np.concatenate(
        [intercept, ages_onehot.astype(np.float32), covid_flag.astype(np.float32)],
        axis=1,
    )
    age_next = np.array([next_year - birth_year])
    ages_binned_next = np.digitize(age_next, age_bins, right=False)
    ages_binned_next = np.clip(ages_binned_next, 1, n_agebins)
    ages_onehot_next = np.eye(n_agebins)[ages_binned_next - 1][:, 1:]
    covid_next_flag = np.array(
        [[1.0 if next_year in covid_years else 0.0]], dtype=np.float32
    )
    x_A_next = np.concatenate(
        [
            np.ones((1, 1), dtype=np.float32),
            ages_onehot_next.astype(np.float32),
            covid_next_flag,
        ],
        axis=1,
    )[0]

    gender_col = np.full((len(years), 1), g_code, dtype=np.float32)
    x_em_hist = np.concatenate(
        [
            intercept,
            gender_col,
            ages_onehot.astype(np.float32),
            covid_flag.astype(np.float32),
        ],
        axis=1,
    )
    x_em_next = np.concatenate(
        [
            np.ones((1, 1), dtype=np.float32),
            np.array([[g_code]], dtype=np.float32),
            ages_onehot_next.astype(np.float32),
            covid_next_flag,
        ],
        axis=1,
    )[0]

    obs_te = torch.tensor(yvals[None, :], dtype=torch.long)
    xpi_te = torch.tensor(x_pi_np, dtype=torch.float32)
    xA_te = torch.tensor(x_A_hist[None, :, :], dtype=torch.float32)
    xem_te = torch.tensor(x_em_hist[None, :, :], dtype=torch.float32)

    paths_old = viterbi.viterbi_paths_glm(
        obs_te, xpi_te, xA_te, xem_te, model_path=here(model_path)
    )
    # let's try to adjust the order states
    v_path = inv[paths_old.cpu().numpy()[0]].tolist()
    # v_path = paths_old.cpu().numpy()[0].tolist()

    B = torch.tensor(beta_em, dtype=torch.float32)
    emis_log = dist.Poisson(rate=torch.einsum("ntc,kc->ntk", xem_te, B).exp()).log_prob(
        obs_te.unsqueeze(-1)
    )

    log_pi0 = torch.tensor(
        np.log(pi_base + 1e-30) + (x_pi_np @ W_pi.T), dtype=torch.float32
    )
    log_pi = log_pi0 - torch.logsumexp(log_pi0, dim=1, keepdim=True)

    K = pi_base.shape[0]
    T_hist = obs_te.size(1)
    log_alpha = torch.empty(1, T_hist, K, dtype=torch.float32)
    log_alpha[:, 0] = log_pi + emis_log[:, 0]

    logA0 = torch.tensor(np.log(A_base + 1e-30), dtype=torch.float32)
    W_A_t = torch.tensor(W_A, dtype=torch.float32)
    for t in range(1, T_hist):
        x_t = xA_te[:, t, :]
        logits = logA0.unsqueeze(0) + (W_A_t.unsqueeze(0) * x_t[:, None, None, :]).sum(
            -1
        )
        log_A = logits - torch.logsumexp(logits, dim=2, keepdim=True)
        log_alpha[:, t] = (
            torch.logsumexp(log_alpha[:, t - 1].unsqueeze(2) + log_A, dim=1)
            + emis_log[:, t]
        )
    alpha_T = (
        (log_alpha[:, -1] - torch.logsumexp(log_alpha[:, -1], dim=1, keepdim=True))
        .exp()
        .cpu()
        .numpy()[0]
    )

    logits_next = np.log(A_base + 1e-30) + np.tensordot(W_A, x_A_next, axes=([2], [0]))
    A_next = _softmax_row(logits_next)
    p_next = alpha_T @ A_next

    lam_next = np.exp(beta_em @ x_em_next)
    expected_next = float((p_next * lam_next).sum())
    p0 = float((p_next * np.exp(-lam_next)).sum())
    prob_donate_next = 1.0 - p0

    from scipy.stats import poisson as _po

    pmf0k = np.array(
        [(p_next * _po.pmf(k, lam_next)).sum() for k in range(max_k)], dtype=float
    )
    tail = float(max(0.0, 1.0 - pmf0k.sum()))
    pmf_dict = {str(k): float(pmf0k[k]) for k in range(max_k)}
    pmf_dict[f">={max_k}"] = tail

    return {
        "years": history_years,
        "counts": history_counts,
        "viterbi_states": v_path,
        "next_year": int(next_year),
        "next_state_probs": p_next.tolist(),
        "expected_next": expected_next,
        "prob_donate_next": prob_donate_next,
        "pmf_next": pmf_dict,
    }
