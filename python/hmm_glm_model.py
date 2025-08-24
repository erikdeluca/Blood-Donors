import pyro
import torch
import numpy as np
# ─────────────────────────────────────────────────────────────
# load the model
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
    if paramfile != None:
        pyro.clear_param_store()
        pyro.get_param_store().load(paramfile)
    W_pi    = pyro.param("W_pi").detach().cpu().numpy()
    W_A     = pyro.param("W_A").detach().cpu().numpy()
    pi_base = pyro.param("pi_base_map").detach().cpu().numpy()
    A_base  = pyro.param("A_base_map").detach().cpu().numpy()
    beta_em = pyro.param("beta_em").detach().cpu().numpy()
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