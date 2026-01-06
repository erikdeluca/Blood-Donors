import torch
import pyro.distributions as dist
import hmm_glm_model as hmm_glm


def log_softmax_logits(logits, dim=-1):
    # Stable log-softmax from logits
    return logits - logits.logsumexp(dim=dim, keepdim=True)


def _coerce_to_torch(x, device, dtype=torch.float32):
    # Convert numpy/number to torch tensor on device; pass-through torch tensors
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.to(
            device=device, dtype=dtype if x.dtype.is_floating_point else x.dtype
        )
    return torch.tensor(x, device=device, dtype=dtype)


@torch.no_grad()
def viterbi_paths_glm(obs, x_pi, x_A, x_em, model_path=None):
    """
    Viterbi decoding for HMM with covariate-dependent pi, A,
    and Poisson-GLM emissions (intercept already included in covariates).

    Parameters
    ----------
    obs  : (N, T) long
        Observed counts.
    x_pi : (N, C_pi) float
        Covariates for initial state distribution.
    x_A  : (N, T, C_A) float
        Covariates for transition probabilities.
    x_em : (N, T, C_em) float
        Covariates for emission GLM (already includes intercept).
    model_path : str or Path, optional
        Path to saved HMM parameters.

    Returns
    -------
    paths : (N, T) long tensor on CPU
        Most likely latent state sequence for each sequence.
    """
    # ---------------- load parameters ----------------
    W_pi, W_A, pi_base, A_base, beta_em = hmm_glm.load_hmm_params(model_path)

    # ---------------- canonical order by pi0 ----------------
    order = hmm_glm._simple_order_by_pi0(pi_base)
    pi_base, A_base, W_pi, W_A, beta_em, inv = hmm_glm.reorder_params(
        order, pi_base, A_base, W_pi, W_A, beta_em
    )

    # ---------------- device alignment ----------------
    device = obs.device
    obs = obs.to(device=device)
    x_pi = x_pi.to(device=device, dtype=torch.float32)
    x_A = x_A.to(device=device, dtype=torch.float32)
    x_em = x_em.to(device=device, dtype=torch.float32)

    W_pi = _coerce_to_torch(W_pi, device)
    W_A = _coerce_to_torch(W_A, device)
    pi_base = _coerce_to_torch(pi_base, device)
    A_base = _coerce_to_torch(A_base, device)
    beta_em = _coerce_to_torch(beta_em, device)

    N, T = obs.shape
    K = int(pi_base.shape[0])

    # ---------------- emission log-probs ----------------
    # eta[n,t,k] = x_em[n,t,:] @ beta_em[k,:]
    eta = torch.einsum("ntc,kc->ntk", x_em, beta_em)  # (N,T,K)
    emis_log = dist.Poisson(rate=eta.exp()).log_prob(obs.unsqueeze(-1))  # (N,T,K)

    # ---------------- initial distribution ----------------
    log_pi_base = torch.log(pi_base + 1e-30)  # (K,)
    logits0 = log_pi_base.view(1, K) + x_pi @ W_pi.T  # (N,K)
    log_pi = log_softmax_logits(logits0, dim=1)  # (N,K)

    delta = log_pi + emis_log[:, 0]  # (N,K)
    psi = torch.zeros(N, T, K, dtype=torch.long, device=device)

    # ---------------- forward DP ----------------
    log_A_base = torch.log(A_base + 1e-30)  # (K,K)
    for t in range(1, T):
        x_t = x_A[:, t, :]  # (N,C_A)
        slope = (W_A.unsqueeze(0) * x_t[:, None, None, :]).sum(-1)  # (N,K,K)
        logits = log_A_base.unsqueeze(0) + slope  # (N,K,K)
        log_A = log_softmax_logits(logits, dim=2)  # (N,K,K)

        score, idx = (delta.unsqueeze(2) + log_A).max(dim=1)  # (N,K)
        psi[:, t] = idx
        delta = score + emis_log[:, t]

    # ---------------- backtracking ----------------
    paths = torch.empty(N, T, dtype=torch.long, device=device)
    last_state = delta.argmax(dim=1)
    paths[:, -1] = last_state
    for t in range(T - 1, 0, -1):
        last_state = psi[torch.arange(N, device=device), t, last_state]
        paths[:, t - 1] = last_state

    return paths.cpu()
