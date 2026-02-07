import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam


# ---------------------------
# Poisson GLM with log link
# ---------------------------
def glm_poisson_model(X, y=None, param_name="glm_beta"):
    N, C = X.shape
    beta = pyro.param(
        param_name, torch.zeros(C + 1, device=X.device)
    )  # intercept + C slopes
    log_mu = beta[0] + (X @ beta[1:].unsqueeze(-1)).squeeze(-1)  # (N,)
    mu = log_mu.exp()
    with pyro.plate("data", N):
        pyro.sample("obs", dist.Poisson(mu), obs=y)


def glm_poisson_guide(X, y=None, param_name="glm_beta"):
    # Mean-field guide is empty: we use point estimates via pyro.param in the model.
    pass


# ---------------------------
# Train
# ---------------------------
def fit_glm_poisson(
    X,
    y,
    steps=3000,
    lr=1e-2,
    param_name="glm_beta",
    verbose_every=500,
    save_path="glm_poisson_params.pt",
):
    X = X.detach()
    y = y.detach().float()  # Poisson log_prob accetta float
    pyro.clear_param_store()
    svi = SVI(
        lambda X_, y_: glm_poisson_model(X_, y_, param_name),
        lambda X_, y_: glm_poisson_guide(X_, y_, param_name),
        Adam({"lr": lr}),
        loss=Trace_ELBO(),
    )
    loss_hist = []
    for s in range(steps):
        loss = svi.step(X, y)
        if verbose_every and s % verbose_every == 0:
            print(f"step {s:4d}: loss={loss:.3f}")
        loss_hist.append(loss)
    # Save only the GLM params (store currently contains only them if you cleared before)
    pyro.get_param_store().save(save_path)
    return loss_hist


# ---------------------------
# Predict and metrics
# ---------------------------
@torch.no_grad()
def glm_poisson_predict(X, param_name="glm_beta", return_proba=True):
    X = X.detach()
    beta = pyro.param(param_name)
    log_mu = beta[0] + (X @ beta[1:].unsqueeze(-1)).squeeze(-1)
    mu = log_mu.exp()  # expected counts
    if return_proba:
        p_donate = 1.0 - torch.exp(-mu)  # P(Y>0 | mu)
        return mu, p_donate
    return mu


@torch.no_grad()
def glm_poisson_evaluate(X, y, param_name="glm_beta"):
    X = X.detach()
    y = y.detach().float()
    mu = glm_poisson_predict(X, param_name, return_proba=False)
    # Metrics
    mae = torch.mean(torch.abs(mu - y)).item()
    rmse = torch.sqrt(torch.mean((mu - y) ** 2)).item()
    # Poisson NLL per observation: -log p(y|mu) = mu - y*log(mu) + log(y!)
    # Use dist.Poisson for stable log_prob
    nll = -torch.mean(dist.Poisson(mu).log_prob(y)).item()
    brier = torch.mean(((1.0 - torch.exp(-mu)) - (y > 0).float()) ** 2).item()
    return {"MAE": mae, "RMSE": rmse, "NLL": nll, "Brier(y>0)": brier}
