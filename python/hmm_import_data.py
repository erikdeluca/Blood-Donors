import pandas as pd
import numpy as np
from pyprojroot import here
import torch
import polars as pl

data = pd.read_csv(here("data/recent_donations.csv"))
data

# load data in polars format
df = pl.from_pandas(data)


year_cols = sorted([c for c in df.columns if c.startswith("y_")])
T = len(year_cols)
# recolect donation observations
obs = (df.select(year_cols)
         .fill_null(0)
         .to_numpy()
         .astype(int))                   # (N,T)

# gender to integer and normalize birth year
df = df.with_columns([
    (pl.col("gender") == "F").cast(pl.Int8).alias("gender_code"),
    ((pl.col("birth_year") - pl.col("birth_year").mean()) /
     pl.col("birth_year").std()).alias("birth_year_norm")
])

# store in numpy objects
birth_year_norm = df["birth_year_norm"].to_numpy()        # (N,)
gender_code     = df["gender_code"].to_numpy()            # (N,)

# covariate matrix for initial probabilites (pi)
cov_init = np.stack([birth_year_norm, gender_code], axis=1)

# dynamic covariates for transition matrix
# normalize age
years_num  = np.array([int(c[2:]) for c in year_cols])    # [2009, …, 2023]
ages       = years_num[None, :] - df["birth_year"].to_numpy()[:, None]
ages_norm  = (ages - ages.mean()) / ages.std()            # (N,T)

# create dummy variable for covid years
covid_mask = np.isin(years_num, [2020, 2021, 2022]).astype(float)  # (T,)
covid_years = np.tile(covid_mask, (df.height, 1))          # (N,T)

# A-covariate tensor (N,T,2)
cov_tran = np.stack([ages_norm, covid_years], axis=2)

# store in torch objects
obs_torch      = torch.tensor(obs,      dtype=torch.long)
cov_init_torch = torch.tensor(cov_init, dtype=torch.float)   # (N,2)
cov_tran_torch = torch.tensor(cov_tran, dtype=torch.float)   # (N,T,2)

# print("obs        :", obs_torch.shape)      # (N,T)
# print("π covs     :", cov_init_torch.shape) # (N,2)
# print("A covs     :", cov_tran_torch.shape) # (N,T,2)