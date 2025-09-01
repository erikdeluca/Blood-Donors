import torch, pyro, pyro.distributions as dist
import numpy as np
from plotnine import (
    ggplot, aes, geom_point, geom_step, geom_vline, geom_text,
    scale_color_manual, scale_x_continuous, scale_y_continuous,
    labs, theme_minimal, theme, element_text, guides, guide_legend
)

def log_softmax_logits(logits, dim=-1):
    return logits - torch.logsumexp(logits, dim=dim, keepdim=True)

def _get_params_as_torch(device=None):
    """
    Load parameters from Pyro's param store as torch tensors on a common device.
    Returns keys: pi_base, A_base, W_pi, W_A, rates (opt), beta_em (opt).
    """
    ps = pyro.get_param_store()
    P = {
        "pi_base_map": ps["pi_base_map"],
        "A_base_map":  ps["A_base_map"],
        "W_pi":        ps["W_pi"],
        "W_A":         ps["W_A"],
    }
    P["rates"]   = ps["rates"]   if "rates"   in ps else None
    P["beta_em"] = ps["beta_em"] if "beta_em" in ps else None

    if device is None:
        device = P["W_pi"].device
    for k, v in P.items():
        if v is not None and v.device != device:
            P[k] = v.to(device)

    return dict(
        pi_base=P["pi_base_map"],
        A_base=P["A_base_map"],
        W_pi=P["W_pi"],
        W_A=P["W_A"],
        rates=P["rates"],
        beta_em=P["beta_em"],
    )

@torch.no_grad()
def forward_loglik_cov(obs, x_pi, x_A, x_em=None):
    """
    Per-sequence log-likelihood under point estimates.
    If 'beta_em' and x_em are provided, uses GLM emissions; else uses 'rates'.
    """
    device = obs.device
    P = _get_params_as_torch(device)
    pi_base, A_base, W_pi, W_A = P["pi_base"], P["A_base"], P["W_pi"], P["W_A"]
    rates, beta_em = P["rates"], P["beta_em"]

    N, T = obs.shape
    K = pi_base.shape[0]

    # Build emission log-probabilities (N,T,K)
    if beta_em is not None and x_em is not None:
        b0 = beta_em[:, 0]                 # (K,)
        B  = beta_em[:, 1:]                # (K, C_em)
        log_mu = torch.einsum("ntc,kc->ntk", x_em.to(device), B) + b0.view(1, 1, K)
        emis_log = dist.Poisson(rate=log_mu.exp()).log_prob(obs.unsqueeze(-1))  # (N,T,K)
    else:
        assert rates is not None, "Need either 'rates' or ('beta_em' and x_em)."
        emis_log = torch.stack([dist.Poisson(r).log_prob(obs) for r in rates], dim=-1)  # (N,T,K)

    # Initial distribution
    log_pi = log_softmax_logits(pi_base.log() + x_pi @ W_pi.T, dim=1)  # (N,K)
    log_alpha = log_pi + emis_log[:, 0]                                 # (N,K)

    # Forward recursion with covariate-driven transitions
    log_A0 = A_base.log()  # (K,K)
    for t in range(1, T):
        x_t = x_A[:, t, :]  # (N, C_A)
        logits = log_A0.unsqueeze(0) + (W_A.unsqueeze(0) * x_t[:, None, None, :]).sum(-1)  # (N,K,K)
        log_A = log_softmax_logits(logits, dim=2)  # (N,K,K)
        log_alpha = torch.logsumexp(log_alpha.unsqueeze(2) + log_A, dim=1) + emis_log[:, t]

    return torch.logsumexp(log_alpha, dim=1)  # (N,)

@torch.no_grad()
def one_step_ahead_metrics(obs, x_pi, x_A, x_em=None):
    """
    One-step-ahead predictive metrics:
      - MAE and RMSE of expected counts
      - Brier score for donation (y>0)
      - Negative log-likelihood of the predictive mixture pmf
    Works with constant-rate or GLM emissions.
    """
    import numpy as np

    device = obs.device
    P = _get_params_as_torch(device)
    pi_base, A_base, W_pi, W_A = P["pi_base"], P["A_base"], P["W_pi"], P["W_A"]
    rates, beta_em = P["rates"], P["beta_em"]

    N, T = obs.shape
    K = pi_base.shape[0]

    # Emission log-probabilities for filtering
    if beta_em is not None and x_em is not None:
        b0 = beta_em[:, 0]
        B  = beta_em[:, 1:]
        log_mu = torch.einsum("ntc,kc->ntk", x_em.to(device), B) + b0.view(1, 1, K)
        emis_log = dist.Poisson(rate=log_mu.exp()).log_prob(obs.unsqueeze(-1))
    else:
        assert rates is not None, "Need either 'rates' or ('beta_em' and x_em)."
        emis_log = torch.stack([dist.Poisson(r).log_prob(obs) for r in rates], dim=-1)

    # Filtering init
    log_pi = log_softmax_logits(pi_base.log() + x_pi @ W_pi.T, dim=1)  # (N,K)
    log_alpha = torch.empty(N, T, K, device=device)
    log_alpha[:, 0] = log_pi + emis_log[:, 0]
    alpha_norm = (log_alpha[:, 0] - torch.logsumexp(log_alpha[:, 0], dim=1, keepdim=True)).exp()

    preds_mean, preds_pdon, logscore, y_next_all = [], [], [], []
    log_A0 = A_base.log()

    for t in range(0, T - 1):
        # Transition at t+1
        x_t1 = x_A[:, t + 1, :]  # (N, C_A)
        logits = log_A0.unsqueeze(0) + (W_A.unsqueeze(0) * x_t1[:, None, None, :]).sum(-1)  # (N,K,K)
        log_A = log_softmax_logits(logits, dim=2)  # (N,K,K)

        # Predictive state distribution at t+1
        P_next = torch.exp(torch.log(alpha_norm + 1e-40).unsqueeze(1) + log_A).sum(dim=2)  # (N,K)

        # Emission rates at t+1
        if beta_em is not None and x_em is not None:
            log_mu_t1 = torch.einsum("nc,kc->nk", x_em[:, t + 1, :].to(device), B) + b0  # (N,K)
            lam_ntk = log_mu_t1.exp()
        else:
            lam_ntk = rates.view(1, K).expand(N, K)

        # Predictive mean and prob(donation)
        Ey  = (P_next * lam_ntk).sum(dim=1)                # (N,)
        P0  = (P_next * torch.exp(-lam_ntk)).sum(dim=1)    # (N,)
        Pdon = 1.0 - P0

        # Predictive mixture pmf for observed y_{t+1}
        y_next = obs[:, t + 1]  # (N,)
        pmf_k = torch.stack(
            [torch.exp(dist.Poisson(lam_ntk[:, k]).log_prob(y_next)) for k in range(K)],
            dim=1
        )  # (N,K)
        mix_pmf = (P_next * pmf_k).sum(dim=1).clamp_min(1e-12)
        ls = torch.log(mix_pmf)

        preds_mean.append(Ey.cpu().numpy())
        preds_pdon.append(Pdon.cpu().numpy())
        logscore.append(ls.cpu().numpy())
        y_next_all.append(y_next.cpu().numpy())

        # Filter update
        log_alpha[:, t + 1] = torch.logsumexp(log_alpha[:, t].unsqueeze(2) + log_A, dim=1) + emis_log[:, t + 1]
        alpha_norm = (log_alpha[:, t + 1] - torch.logsumexp(log_alpha[:, t + 1], dim=1, keepdim=True)).exp()

    Ey = np.concatenate(preds_mean)
    Pd = np.concatenate(preds_pdon)
    LS = np.concatenate(logscore)
    Y  = np.concatenate(y_next_all)

    mae = np.mean(np.abs(Ey - Y))
    rmse = np.sqrt(np.mean((Ey - Y) ** 2))
    brier = np.mean((Pd - (Y > 0).astype(float)) ** 2)
    nll = -np.mean(LS)

    return {"MAE": mae, "RMSE": rmse, "Brier(y>0)": brier, "NLL": nll}


@torch.no_grad()
def predict_donor(birth_year: int,
                  gender: str,
                  history_years: list,
                  history_counts: list,
                  next_year: int | None = None,
                  max_k: int = 4,
                  x_em_builder=None,
                  x_A_builder=None,
                  covid_years=(2020, 2021, 2022)):
    """
    Predict next-year donation distribution and decode past latent states (Viterbi)
    for the HMM with covariate-dependent emissions (GLM).

    Parameters
    ----------
    birth_year : int
        Birth year.
    gender : str
        'M' or 'F'.
    history_years : list[int]
        Observed years in order.
    history_counts : list[int]
        Counts per year, same length as history_years.
    next_year : int or None
        Year to predict; if None, uses last(history_years)+1.
    max_k : int
        PMF will be returned for 0..max_k and a tail '>=max_k+1'.
    x_em_builder : callable or None
        Function (years: np.ndarray[int], birth_year: int, gender_code: float) -> np.ndarray[T, C_em]
        to build emission covariates on the same scale used in training.
        If None, defaults to [gender_code, ages_norm, covid_flag] with normalization globals.
    x_A_builder : callable or None
        Function (years: np.ndarray[int], birth_year: int, gender_code: float) -> np.ndarray[T, C_A]
        to build transition covariates on the same scale used in training.
        If None, defaults to [ages_norm, covid_flag] with normalization globals.
    covid_years : tuple[int]
        Years considered as covid indicator = 1.0.

    Returns
    -------
    dict
        {
          "years": history_years,
          "counts": history_counts,
          "viterbi_states": list[int],
          "next_year": int,
          "next_state_probs": list[float] (length K),
          "expected_next": float,
          "prob_donate_next": float,
          "pmf_next": dict[str -> float]  # keys "0"..str(max_k), ">=max_k+1"
        }
    """
    assert len(history_years) == len(history_counts), "history mismatch"
    years = np.array(history_years, dtype=int)
    yvals = np.array(history_counts, dtype=int)
    if next_year is None:
        next_year = int(years[-1] + 1)

    # Retrieve learned parameters
    pi_base = pyro.param("pi_base_map").detach().cpu().numpy()   # (K,) simplex
    A_base  = pyro.param("A_base_map").detach().cpu().numpy()    # (K,K) row-simplex
    W_pi    = pyro.param("W_pi").detach().cpu().numpy()          # (K, C_pi)
    W_A     = pyro.param("W_A").detach().cpu().numpy()           # (K, K, C_A)
    beta_em = pyro.param("beta_em").detach().cpu().numpy()       # (K, 1 + C_em)

    K      = pi_base.shape[0]
    C_pi   = W_pi.shape[1]
    C_A    = W_A.shape[2]
    C_em   = beta_em.shape[1] - 1

    # Basic features shared across builders
    g_code = 1.0 if gender.upper().startswith("F") else 0.0

    # Normalization stats expected to exist in scope (computed during training):
    #   birth_year_mean, birth_year_std, ages_mean, ages_std
    by_norm = (birth_year - birth_year_mean) / birth_year_std

    ages_arr   = years - birth_year
    ages_norm  = (ages_arr - ages_mean) / ages_std
    covid_flag = np.isin(years, covid_years).astype(float)

    # Build x_pi (initial covariates) on training scale
    if C_pi == 2:
        x_pi_np = np.array([[by_norm, g_code]], dtype=np.float32)
    else:
        raise ValueError(f"x_pi dimension mismatch: model expects {C_pi} features.")

    # Build x_A (transition covariates) for history and next year
    if x_A_builder is None:
        if C_A != 2:
            raise ValueError(f"x_A dimension mismatch ({C_A}). Provide x_A_builder(years, birth_year, gender_code)->(T,{C_A}).")
        x_A_hist = np.stack([ages_norm, covid_flag], axis=1).astype(np.float32)  # (T,2)
        age_next_norm   = (next_year - birth_year - ages_mean) / ages_std
        covid_next_flag = 1.0 if next_year in covid_years else 0.0
        x_A_next = np.array([age_next_norm, covid_next_flag], dtype=np.float32)  # (2,)
    else:
        x_A_hist = np.asarray(x_A_builder(years, birth_year, g_code), dtype=np.float32)  # (T, C_A)
        if x_A_hist.shape[1] != C_A:
            raise ValueError(f"x_A_builder returned {x_A_hist.shape[1]} features, expected {C_A}.")
        x_A_next = np.asarray(x_A_builder(np.array([next_year], dtype=int), birth_year, g_code), dtype=np.float32)[0]  # (C_A,)

    # Build x_em (emission covariates) for history and next year
    if x_em_builder is None:
        if C_em != 3:
            raise ValueError(f"x_em dimension mismatch ({C_em}). Provide x_em_builder(years, birth_year, gender_code)->(T,{C_em}).")
        x_em_hist = np.stack([np.repeat(g_code, len(years)), ages_norm, covid_flag], axis=1).astype(np.float32)  # (T,3)
        x_em_next = np.array([g_code, age_next_norm, covid_next_flag], dtype=np.float32)                           # (3,)
    else:
        x_em_hist = np.asarray(x_em_builder(years, birth_year, g_code), dtype=np.float32)  # (T, C_em)
        if x_em_hist.shape[1] != C_em:
            raise ValueError(f"x_em_builder returned {x_em_hist.shape[1]} features, expected {C_em}.")
        x_em_next = np.asarray(x_em_builder(np.array([next_year], dtype=int), birth_year, g_code), dtype=np.float32)[0]  # (C_em,)

    # Torch tensors
    obs_te  = torch.tensor(yvals[None, :], dtype=torch.long)                 # (1,T)
    xpi_te  = torch.tensor(x_pi_np, dtype=torch.float32)                     # (1,C_pi)
    xA_te   = torch.tensor(x_A_hist[None, :, :], dtype=torch.float32)        # (1,T,C_A)
    xem_te  = torch.tensor(x_em_hist[None, :, :], dtype=torch.float32)       # (1,T,C_em)

    # Precompute emission log-probabilities for history
    b0 = torch.tensor(beta_em[:, 0], dtype=torch.float32)                    # (K,)
    B  = torch.tensor(beta_em[:, 1:], dtype=torch.float32)                   # (K,C_em)
    log_mu_hist = torch.einsum("ntc,kc->ntk", xem_te, B) + b0.view(1, 1, K)  # (1,T,K)
    emis_log = dist.Poisson(rate=log_mu_hist.exp()).log_prob(obs_te.unsqueeze(-1))  # (1,T,K)

    # Initial logits for z_0
    log_pi0 = torch.tensor(np.log(pi_base + 1e-30) + (x_pi_np @ W_pi.T), dtype=torch.float32)  # (1,K)
    log_pi  = log_pi0 - torch.logsumexp(log_pi0, dim=1, keepdim=True)                          # (1,K)

    # Viterbi decoding over history with covariate transitions and GLM emissions
    T_hist = obs_te.size(1)
    delta = log_pi + emis_log[:, 0]                         # (1,K)
    psi   = torch.zeros(1, T_hist, K, dtype=torch.long)     # backpointers

    logA0 = torch.tensor(np.log(A_base + 1e-30), dtype=torch.float32)       # (K,K)
    W_A_t = torch.tensor(W_A, dtype=torch.float32)                           # (K,K,C_A)

    for t in range(1, T_hist):
        x_t = xA_te[:, t, :]                                                 # (1,C_A)
        logits = logA0.unsqueeze(0) + (W_A_t.unsqueeze(0) * x_t[:, None, None, :]).sum(-1)  # (1,K,K)
        log_A  = logits - torch.logsumexp(logits, dim=2, keepdim=True)       # (1,K,K)
        score, idx = (delta.unsqueeze(2) + log_A).max(dim=1)                 # (1,K), (1,K)
        psi[:, t] = idx
        delta = score + emis_log[:, t]

    # Backtrack Viterbi path
    paths = torch.empty(1, T_hist, dtype=torch.long)
    last_state = delta.argmax(dim=1)
    paths[:, -1] = last_state
    for t in range(T_hist - 1, 0, -1):
        last_state = psi[torch.arange(1), t, last_state]
        paths[:, t - 1] = last_state
    v_path = paths[0].cpu().numpy().tolist()

    # Forward filtering to get alpha_T for prediction
    log_alpha = torch.empty(1, T_hist, K, dtype=torch.float32)
    log_alpha[:, 0] = log_pi + emis_log[:, 0]
    for t in range(1, T_hist):
        x_t = xA_te[:, t, :]
        logits = logA0.unsqueeze(0) + (W_A_t.unsqueeze(0) * x_t[:, None, None, :]).sum(-1)  # (1,K,K)
        log_A  = logits - torch.logsumexp(logits, dim=2, keepdim=True)
        log_alpha[:, t] = torch.logsumexp(log_alpha[:, t - 1].unsqueeze(2) + log_A, dim=1) + emis_log[:, t]
    alpha_T = (log_alpha[:, -1] - torch.logsumexp(log_alpha[:, -1], dim=1, keepdim=True)).exp().cpu().numpy()[0]  # (K,)

    # Next-year transition and emission
    logits_next = np.log(A_base + 1e-30) + np.tensordot(W_A, x_A_next, axes=([2], [0]))  # (K,K)
    A_next = _softmax_np(logits_next)                                                     # (K,K)
    p_next = alpha_T @ A_next                                                             # (K,)

    lam_next = np.exp(beta_em[:, 0] + (beta_em[:, 1:] @ x_em_next))                       # (K,)

    # Predictive mixture for next year
    expected_next = float((p_next * lam_next).sum())
    p0 = float((p_next * np.exp(-lam_next)).sum())
    prob_donate_next = 1.0 - p0

    from scipy.stats import poisson as _po
    pmf0k = np.array([(p_next * _po.pmf(k, lam_next)).sum() for k in range(max_k + 1)], dtype=float)
    tail  = float(max(0.0, 1.0 - pmf0k.sum()))
    pmf_dict = {str(k): float(pmf0k[k]) for k in range(max_k + 1)}
    pmf_dict[f">={max_k+1}"] = tail

    return {
        "years": history_years,
        "counts": history_counts,
        "viterbi_states": v_path,
        "next_year": int(next_year),
        "next_state_probs": p_next.tolist(),
        "expected_next": expected_next,
        "prob_donate_next": prob_donate_next,
        "pmf_next": pmf_dict
    }


# 5) Optional: custom builders if your model expects C_em != 3 or C_A != 2
def make_xA_builder(CA):
    # Example: if CA==2 we mimic [age_norm, covid_flag]; otherwise fill zeros for extra dims
    def _builder(years, by, gcode):
        years = np.asarray(years, dtype=int)
        age_norm = (years - by - ages_mean) / ages_std
        covid = np.isin(years, (2020, 2021, 2022)).astype(float)
        base = np.stack([age_norm, covid], axis=1).astype(np.float32)
        if CA > 2:
            extra = np.zeros((len(years), CA - 2), dtype=np.float32)
            base = np.concatenate([base, extra], axis=1)
        return base
    return _builder

def make_xem_builder(Cem):
    # Example emission builder:
    #   - starts with [gender, age_norm, covid]
    #   - pads with zeros if more features are required
    def _builder(years, by, gcode):
        years = np.asarray(years, dtype=int)
        age_norm = (years - by - ages_mean) / ages_std
        covid = np.isin(years, (2020, 2021, 2022)).astype(float)
        base = np.stack([np.repeat(gcode, len(years)), age_norm, covid], axis=1).astype(np.float32)
        if Cem > 3:
            extra = np.zeros((len(years), Cem - 3), dtype=np.float32)
            base = np.concatenate([base, extra], axis=1)
        return base
    return _builder

# 3) Builder per x_A: primi 2 slot = [age_norm, covid], resto = medie (o 0)
def make_xA_builder_with_means(CA, feature_means=None, covid_years=(2020, 2021, 2022)):
    means = feature_means if feature_means is not None else np.zeros(CA, dtype=np.float32)
    def _builder(years, birth_year, gcode):
        years = np.asarray(years, dtype=int)
        T = len(years)
        age_norm = (years - birth_year - ages_mean) / ages_std
        covid = np.isin(years, covid_years).astype(float)
        X = np.tile(means, (T, 1)).astype(np.float32)
        # Assunzione: col 0=age_norm, col 1=covid
        X[:, 0] = age_norm.astype(np.float32)
        X[:, 1] = covid.astype(np.float32)
        return X
    return _builder

# 4) Builder per x_em se C_em != 3 (padding con zeri)
def make_xem_builder_pad(Cem, covid_years=(2020, 2021, 2022)):
    def _builder(years, birth_year, gcode):
        years = np.asarray(years, dtype=int)
        T = len(years)
        age_norm = (years - birth_year - ages_mean) / ages_std
        covid = np.isin(years, covid_years).astype(float)
        base = np.stack([np.repeat(gcode, T), age_norm, covid], axis=1).astype(np.float32)
        if Cem > 3:
            extra = np.zeros((T, Cem - 3), dtype=np.float32)
            return np.concatenate([base, extra], axis=1)
        else:
            return base[:, :Cem].astype(np.float32)
    return _builder


def plot_donor_gg(idx,
                  obs_torch,
                  paths,
                  years,
                  expected_next=None,    # float, predicted expected donations for next year
                  y_true_next=None,      # int, actual donations next year (if available)
                  next_year=None,        # int, defaults to years[-1] + 1
                  state_cols=None,       # list of colors for states
                  title_prefix="Donor",
                  y_max=4):
    """
    Plot observed yearly donations colored by latent state (Viterbi),
    plus markers for predicted and actual next-year donations.

    Parameters
    ----------
    idx : int
        Donor index.
    obs_torch : array-like or torch.Tensor (N, T)
        Observed counts.
    paths : array-like or torch.Tensor (N, T)
        Latent states (e.g., Viterbi), ints in 0..K-1.
    years : array-like (T,)
        Calendar years for the x-axis (must match T).
    expected_next : float or None
        Predicted expected donations for next_year.
    y_true_next : int or None
        Actual number of donations at next_year (if available).
    next_year : int or None
        Year for the prediction point. Defaults to years[-1] + 1.
    state_cols : list[str] or None
        Color palette for states; if None, uses an extended Set1-like palette.
    title_prefix : str
        Plot title prefix.
    y_max : int
        Top of y-axis (default 4 → shows 0..4 and clamps visually).

    Returns
    -------
    plotnine.ggplot
        The assembled ggplot object.
    """
    # --- Extract donor slice as numpy ---
    x = obs_torch[idx].detach().cpu().numpy() if hasattr(obs_torch, "detach") else np.asarray(obs_torch[idx])
    z = paths[idx].detach().cpu().numpy()     if hasattr(paths, "detach")     else np.asarray(paths[idx], dtype=int)
    years = np.asarray(years, dtype=int)

    T = len(x)
    if len(years) != T:
        raise ValueError("Length of 'years' must match the time dimension T for the donor.")

    # --- Build state palette and labels ---
    K = int(np.max(z)) + 1 if z.size > 0 else 1
    if state_cols is None:
        default_cols = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33',
                        '#a65628', '#f781bf', '#999999']
        state_cols = default_cols[:K]
    state_labels = [f"State {k}" for k in range(K)]
    z_labs = [state_labels[s] for s in z]

    # --- Observed data frame ---
    df_obs = pd.DataFrame({
        "year": years,
        "donations": x,
        "state": z_labs
    })

    # --- Prediction annotations ---
    if next_year is None:
        next_year = int(years[-1] + 1)

    rows_pred = []
    if expected_next is not None:
        rows_pred.append({"year": next_year, "donations": expected_next, "kind": "Predicted"})
    if y_true_next is not None:
        rows_pred.append({"year": next_year, "donations": y_true_next, "kind": "Actual"})
    df_pred = pd.DataFrame(rows_pred) if rows_pred else pd.DataFrame(columns=["year", "donations", "kind"])

    # --- Axis limits and breaks ---
    y_low, y_high = -0.5, float(y_max) + 0.5
    x_breaks = list(np.unique(np.concatenate([years, np.array([next_year])])))

    # --- Base plot ---
    p = (
        ggplot(df_obs, aes("year", "donations"))
        + geom_step(direction="mid", color="black", alpha=0.35)
        + geom_point(aes(color="state"), size=2.5)
        + scale_color_manual(values=state_cols, name="latent state",
                             breaks=state_labels, labels=state_labels)
        + scale_x_continuous(breaks=x_breaks)
        + scale_y_continuous(limits=(y_low, y_high), breaks=list(range(0, int(y_max) + 1)))
        + labs(title=f"{title_prefix} {idx}", x="year", y="# donations")
        + theme_minimal()
        + theme(
            axis_text_x=element_text(rotation=45, ha="right"),
            legend_title=element_text(size=10),
            legend_text=element_text(size=9),
            plot_title=element_text(weight="bold")
        )
        + guides(color=guide_legend(title="latent state"))
    )

    # --- Next-year markers and labels ---
    if not df_pred.empty:
        p = p + geom_vline(xintercept=next_year, linetype="dashed", alpha=0.6)

        if (df_pred["kind"] == "Predicted").any():
            p = p + geom_point(
                mapping=aes("year", "donations"),
                data=df_pred[df_pred["kind"] == "Predicted"],
                color="black",
                size=3.2,
                show_legend=False
            ) + geom_text(
                mapping=aes("year", "donations"),
                data=df_pred[df_pred["kind"] == "Predicted"],
                label="pred",
                nudge_y=0.25,
                size=8,
                color="black",
                show_legend=False
            )

        if (df_pred["kind"] == "Actual").any():
            p = p + geom_point(
                mapping=aes("year", "donations"),
                data=df_pred[df_pred["kind"] == "Actual"],
                color="#d62728",
                size=3.5,
                shape="x",
                show_legend=False
            ) + geom_text(
                mapping=aes("year", "donations"),
                data=df_pred[df_pred["kind"] == "Actual"],
                label="actual",
                nudge_y=0.25,
                size=8,
                color="#d62728",
                show_legend=False
            )
    return p


def hmm_forward_predict(
    obs_so_far, xpi, xA, A_base, W_pi, W_A, pi_base, beta_em, cov_emission,
    steps_ahead=1
):
    """
    Predicts E[y_{T+steps_ahead}] with covariate-driven transitions and GLM emissions.

    Parameters
    ----------
    obs_so_far : (N, T_obs) array or None
        Observed history (used only for its length; no filtering).
    xpi : (N, C_pi) float
        Initial-state covariates.
    xA : (N, T_total, C_A) float
        Transition covariates for all times (must cover prediction index).
    A_base : (K, K) float, rows simplex
        Base transition matrix.
    W_pi : (K, C_pi) float
        Initial-state slopes.
    W_A : (K, K, C_A) float
        Transition slopes (row-wise).
    pi_base : (K,) float, simplex
        Base initial state.
    beta_em : (K, 1 + C_em) float
        Emission GLM coefficients (intercept + slopes) per state.
    cov_emission : (N, T_total, C_em) float
        Emission covariates at each time.
    steps_ahead : int
        How many steps ahead from T_obs to predict (>=1).

    Returns
    -------
    y_expected : (N,) float
        Predicted expected donations at prediction time.
    state_dist : (N, K) float
        Belief on hidden state at prediction time (after propagation).
    """
    EPS = 1e-30

    N = xpi.shape[0]
    K = beta_em.shape[0]
    T_obs = 0 if (obs_so_far is None) else int(obs_so_far.shape[1])

    # 1) Initial alpha via softmax(log π_base + xπ Wπ^T)
    logits_pi = np.log(pi_base + EPS)[None, :] + xpi @ W_pi.T            # (N,K)
    alpha = softmax_row(logits_pi)                                       # (N,K)

    # 2) Propagation for steps_ahead
    #    For each t in [T_obs, T_obs+steps_ahead-1] build Aprob(n) = softmax_row(log A_base + W_A ⋅ xA[n,t,:])
    logA0 = np.log(A_base + EPS)                                         # (K,K)
    for t in range(T_obs, T_obs + steps_ahead):
        x_t = xA[:, t, :]                                                # (N,C_A)
        # slope logits: tensordot over C_A → (N,K,K)
        slope = np.tensordot(x_t, W_A, axes=([1], [2]))                  # (N,K,K)
        logits = logA0[None, :, :] + slope                               # (N,K,K)
        Aprob = softmax_row(logits)                                      # (N,K,K) row-softmax over axis=-1
        alpha = np.einsum('nk,nkj->nj', alpha, Aprob)                    # (N,K)

    # 3) Emission at prediction time
    t_pred = T_obs + steps_ahead - 1
    x_em = cov_emission[:, t_pred, :]                                    # (N,C_em)
    # eta = b0 + X_em @ B^T → (N,K)
    b0 = beta_em[:, 0][None, :]                                          # (1,K)
    B  = beta_em[:, 1:]                                                  # (K,C_em)
    eta = b0 + x_em @ B.T                                                # (N,K)
    lam = np.exp(eta)                                                    # (N,K)

    # Expected count under mixture
    y_expected = (alpha * lam).sum(axis=1)                               # (N,)

    return y_expected, alpha