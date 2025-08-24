import torch
import pyro
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
        return x.to(device=device, dtype=dtype if x.dtype.is_floating_point else x.dtype)
    return torch.tensor(x, device=device, dtype=dtype)


@torch.no_grad()
def viterbi_paths_cov(obs, x_pi, x_A, x_em=None, model_path=None):
    """
    Viterbi for HMM with covariate-dependent pi and A, and optional Poisson-GLM emissions.

    Inputs
    ------
    obs  : (N, T) long
    x_pi : (N, C_pi) float
    x_A  : (N, T, C_A) float
    x_em : (N, T, C_em) float, required if GLM emissions are present in the checkpoint

    Returns
    -------
    paths : (N, T) long tensor on CPU
    """
    # ------------------------------------------------------------
    # Load parameters (support both dict and tuple signatures)
    # ------------------------------------------------------------
    params = hmm_glm.load_hmm_params(model_path)  # may return dict or tuple

    if isinstance(params, dict):
        W_pi = params.get("W_pi", None)
        W_A = params.get("W_A", None)
        pi_base = params.get("pi_base_map", None)
        A_base = params.get("A_base_map", None)
        beta_em = params.get("beta_em", None)
    else:
        # Tuple convention: W_pi, W_A, pi_base, A_base, beta_em
        W_pi, W_A, pi_base, A_base, beta_em = params

    # ------------------------------------------------------------
    # Device/dtype alignment
    # ------------------------------------------------------------
    device = obs.device  # anchor to obs' device
    obs = obs.to(device=device)
    x_pi = x_pi.to(device=device, dtype=torch.float32)
    x_A = x_A.to(device=device, dtype=torch.float32)

    # Coerce loaded params to torch on the same device
    W_pi = _coerce_to_torch(W_pi, device)
    W_A = _coerce_to_torch(W_A, device)
    pi_base = _coerce_to_torch(pi_base, device)
    A_base = _coerce_to_torch(A_base, device)
    beta_em = _coerce_to_torch(beta_em, device)

    # ------------------------------------------------------------
    # Emission branch: GLM if beta_em provided, else constant-rate
    # ------------------------------------------------------------
    use_glm = beta_em is not None or ("beta_em" in pyro.get_param_store())

    if use_glm:
        # Ensure beta_em is a torch tensor (either from loader or ParamStore)
        if beta_em is None:
            beta_em = pyro.param("beta_em").to(device)
        if x_em is None:
            raise ValueError("x_em is required because GLM emission 'beta_em' is present.")
        x_em = x_em.to(device=device, dtype=torch.float32)

    N, T = obs.shape
    K = int(pi_base.shape[0])

    # ------------------------------------------------------------
    # Emission log-probabilities: log p(y_t | z_t=k)
    # ------------------------------------------------------------
    if use_glm:
        # beta_em: (K, 1 + C_em) = [intercept, slopes...]
        b0 = beta_em[:, 0]                # (K,)
        B = beta_em[:, 1:]                # (K, C_em)
        # eta[n,t,k] = b0[k] + x_em[n,t,:] @ B[k,:]
        eta = torch.einsum("ntc,kc->ntk", x_em, B) + b0.view(1, 1, K)
        emis_log = dist.Poisson(rate=eta.exp()).log_prob(obs.unsqueeze(-1))  # (N,T,K)
    else:
        # Constant-rate emissions per state
        if "rates" in pyro.get_param_store():
            rates = pyro.param("rates").to(device)
        else:
            raise ValueError("No GLM beta_em and no 'rates' found in ParamStore for emissions.")
        emis_log = torch.stack(
            [dist.Poisson(l).log_prob(obs) for l in rates]  # (K,N,T)
        ).permute(1, 2, 0)  # (N,T,K)

    # ------------------------------------------------------------
    # Initial step t = 0: log π(x_pi) + emission
    # π(x) ∝ π_base ⊙ exp(W_pi x)  on log-scale: log π_base + W_pi x, then log-softmax
    # ------------------------------------------------------------
    log_pi_base = torch.log(pi_base + 1e-30)       # (K,)
    logits0 = log_pi_base.view(1, K) + x_pi @ W_pi.T  # (N,K)
    log_pi = log_softmax_logits(logits0, dim=1)    # (N,K)

    delta = log_pi + emis_log[:, 0]                # (N,K)
    psi = torch.zeros(N, T, K, dtype=torch.long, device=device)

    # ------------------------------------------------------------
    # Forward pass: dynamic programming over t = 1..T-1
    # A(x) row-wise softmax of log A_base + W_A · x_t
    # ------------------------------------------------------------
    log_A_base = torch.log(A_base + 1e-30)         # (K,K)
    for t in range(1, T):
        x_t = x_A[:, t, :]  # (N,C_A)
        # slope[n,k_prev,k_next] = <W_A[k_prev,k_next,:], x_t[n,:]>
        slope = (W_A.unsqueeze(0) * x_t[:, None, None, :]).sum(-1)  # (N,K,K)
        logits = log_A_base.unsqueeze(0) + slope                     # (N,K,K)
        log_A = log_softmax_logits(logits, dim=2)                    # normalize across next-state

        score, idx = (delta.unsqueeze(2) + log_A).max(dim=1)         # max over prev-state
        psi[:, t] = idx
        delta = score + emis_log[:, t]

    # ------------------------------------------------------------
    # Back-tracking
    # ------------------------------------------------------------
    paths = torch.empty(N, T, dtype=torch.long, device=device)
    last_state = delta.argmax(dim=1)
    paths[:, -1] = last_state
    for t in range(T - 1, 0, -1):
        last_state = psi[torch.arange(N, device=device), t, last_state]
        paths[:, t - 1] = last_state

    return paths.cpu()
