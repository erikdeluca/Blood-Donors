import pyro
from pyro.infer import SVI, TraceEnum_ELBO
from pyro.optim import Adam
import hmm_model
import runpy
from pyprojroot import here

# import enviroment
ns = runpy.run_path(here("python/hmm_import_data.py"))
globals().update(ns)

# bayesian hmm
pyro.clear_param_store()
svi = SVI(
    hmm_model.model,
    hmm_model.guide,
    Adam({"lr": 2e-2}),
    loss=TraceEnum_ELBO(max_plate_nesting=1),
)

for step in range(800):
    loss = svi.step(obs_torch, cov_init_torch, cov_tran_torch)  # noqa: F821
    if step % 50 == 0:
        print(f"{step:4d}  ELBO = {loss:,.0f}")

param_path = here("models/hmm_full.pt")
pyro.get_param_store().save(param_path)
print(f"ParamStore saved in: {param_path}")
