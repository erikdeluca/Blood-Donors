import pyro
import torch
import numpy as np
def _simple_order_by_pi0(pi_base: np.ndarray) -> np.ndarray:
    """
    Ordina gli stati in modo fisso:
      - stato 0 = più alta probabilità iniziale
      - stato 1 = più bassa probabilità iniziale
      - stato 2 = quella intermedia (se K=3)
    Per K diverso da 3: ordine decrescente (alto → basso).
    """
    pi_base = np.asarray(pi_base)
    s_asc = np.argsort(pi_base)  # crescente
    if pi_base.shape[0] == 3:
        return np.array([s_asc[-1], s_asc[0], s_asc[1]], dtype=int)  # [high, low, mid]
    else:
        return s_asc[::-1].astype(int)  # fallback generale: alto → basso

def reorder_params(order, pi_base, A_base, W_pi, W_A, beta_em):
    idx = np.asarray(order)
    inv = np.empty_like(idx); inv[idx] = np.arange(len(idx))
    # riordina righe/colonne
    pi_base_ = pi_base[idx]
    A_base_  = A_base[idx][:, idx]
    W_pi_    = W_pi[idx]
    W_A_     = W_A[idx][:, idx, :]
    beta_em_ = beta_em[idx]
    return pi_base_, A_base_, W_pi_, W_A_, beta_em_, inv

# ─────────────────────────────────────────────────────────────
# load the model (con ordinamento fisso sugli stati)
# ─────────────────────────────────────────────────────────────
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

    W_pi    = pyro.param("W_pi").detach().cpu().numpy()
    W_A     = pyro.param("W_A").detach().cpu().numpy()
    pi_base = pyro.param("pi_base_map").detach().cpu().numpy()
    A_base  = pyro.param("A_base_map").detach().cpu().numpy()
    beta_em = pyro.param("beta_em").detach().cpu().numpy()

    # ORDINE SEMPLICE: stato 0 = high pi0, stato 1 = low pi0, stato 2 = mid pi0
    idx = _simple_order_by_pi0(pi_base)
    pi_base, A_base, W_pi, W_A, beta_em, _ = reorder_params(idx, pi_base, A_base, W_pi, W_A, beta_em)

    return W_pi, W_A, pi_base, A_base, beta_em

    
def get_W_A_and_logA(paramfile=None):
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
    if paramfile != None:
        pyro.clear_param_store()
        pyro.get_param_store().load(paramfile)
    W_A = pyro.param("W_A").detach().cpu().numpy()          # (K,K,C_A)
    A0  = pyro.param("A_base_map").detach().cpu().numpy()   # (K,K)
    log_A0 = np.log(np.clip(A0, 1e-30, None))
    return W_A, log_A0

def get_pi_params(paramfile=None):
    import torch
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
    if paramfile != None:
        pyro.clear_param_store()
    W_pi   = pyro.param("W_pi").detach().cpu().numpy()                # (K,C_pi)
    pi_base = pyro.param("pi_base_map").detach().cpu().numpy()        # (K,)
    log_pi0 = np.log(np.clip(pi_base, 1e-30, None))
    return W_pi, log_pi0

def get_emission_params_and_ref(cov_emiss_torch):
    has_beta = "beta_em" in pyro.get_param_store()
    if not has_beta:
        return None, None, None
    beta_em = pyro.param("beta_em").detach().cpu().numpy()            # (K,1+C_em)
    X = cov_emiss_torch.detach().cpu().numpy()                        # (N,T,C_em)
    x_em_ref = X.mean(axis=(0,1))                                     # (C_em,)
    return beta_em, x_em_ref, X.shape[-1]
