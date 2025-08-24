# ───────────────────────────────────────────────────────────────
#  Poisson-HMM con Dirichlet asimmetriche
# ───────────────────────────────────────────────────────────────
import torch, pyro, pyro.distributions as dist
from   pyro.infer  import SVI, TraceEnum_ELBO, config_enumerate


K        = 3
C_pi     = 2        # birth_year_norm , gender_code
C_A      = 2        # ages_norm , covid_years

#  a priori asimmetriche 
alpha_pi = torch.tensor([5., 2., 1.])
alpha_A  = torch.tensor([[6.,1.,1.], 
                         [1.,6.,1.],
                         [1.,1.,6.]])

@config_enumerate
def model(obs, x_pi, x_A):
    N, T = obs.shape

    # 1) Poisson rates
    rates = pyro.param("rates",
                       0.5*torch.ones(K),
                       constraint=dist.constraints.positive)

    # 2) Dirichlet priors 
    pi_base  = pyro.sample("pi_base", dist.Dirichlet(alpha_pi))
    A_base   = pyro.sample("A_base",
                           dist.Dirichlet(alpha_A).to_event(1))     # shape (K,K)

    log_pi_base = pi_base.log()           # (K,)
    log_A_base  = A_base.log()            # (K,K)

    # 3) slope coefficients for covariates 
    W_pi = pyro.param("W_pi", torch.zeros(K, C_pi))
    W_A  = pyro.param("W_A",  torch.zeros(K, K, C_A))

    with pyro.plate("seqs", N):
        # T=0
        logits0 = log_pi_base + (x_pi @ W_pi.T)             # (N,K)
        z_prev  = pyro.sample("z_0",
                              dist.Categorical(logits=logits0),
                              infer={"enumerate": "parallel"})
        pyro.sample("y_0", dist.Poisson(rates[z_prev]), obs=obs[:,0])

        # T>0
        for t in range(1, T):
            x_t   = x_A[:, t, :]                            # (N,2)
            logitsT = (log_A_base[z_prev] +
                       (W_A[z_prev] * x_t[:,None,:]).sum(-1))
            z_t = pyro.sample(f"z_{t}",
                              dist.Categorical(logits=logitsT),
                              infer={"enumerate": "parallel"})
            pyro.sample(f"y_{t}", dist.Poisson(rates[z_t]), obs=obs[:,t])
            z_prev = z_t

# ─────────────────────────────────────────────────────────────
#  GUIDE 
# ─────────────────────────────────────────────────────────────
def guide(obs, x_pi, x_A):
    # pi base
    pi_q = pyro.param(
        "pi_base_map",
        torch.tensor([0.6, 0.3, 0.1]),   
        constraint=dist.constraints.simplex
    )

    # A base
    A_init = torch.eye(K) * (K - 1.) + 1.      # helps the diagonal
    A_init = A_init / A_init.sum(-1, keepdim=True) # softmax, x>o and sum(x)=1

    A_q = pyro.param(
        "A_base_map",
        A_init,
        constraint=dist.constraints.simplex # x>o and sum(x)=1
    )

    # fix new pi_base and A_base for the next training iteration
    # Delta is a trick to fix this values with sample 
    pyro.sample("pi_base", dist.Delta(pi_q).to_event(1))  
    pyro.sample("A_base",  dist.Delta(A_q ).to_event(2))  


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
    lam     = pyro.param("rates").detach().cpu().numpy()  
    return W_pi, W_A, pi_base, A_base, lam
