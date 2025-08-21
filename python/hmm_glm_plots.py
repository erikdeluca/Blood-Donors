# -*- coding: utf-8 -*-
# NOTE: This version uses Matplotlib/Seaborn only (no plotnine).
#       It draws three panels:
#         1) Initial state probabilities (bar)
#         2) Transition matrix (heatmap)
#         3) Emission coefficients by state (grouped bar; x=coefficient, y=value, color=state)

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import hmm_glm_model as hmm_glm
from typing import Callable, Iterable, Optional, Tuple

## HMM with coefficients

def plot_hmm_params_with_coeffs(
    transitions,
    initial_probs,
    beta_em,
    state_names=None,
    coeff_names=None,
    include_intercept=True,
    intercept_name="Intercept",
    figsize=(16, 4),
    annot_transitions=True,
    show=True
):
    """
    Visual summary of HMM parameters with emission GLM coefficients.

    Parameters
    ----------
    transitions : array-like (S, S)
        Transition probability matrix (rows sum to 1).
    initial_probs : array-like (S,)
        Initial state probabilities.
    beta_em : array-like (S, C) or (S, C+1 if include_intercept=True)
        Emission GLM coefficients per state. If include_intercept=True, the first column is intercept.
    state_names : list[str], optional
        Names of states, length S. Defaults to ["State 0", ..., "State S-1"].
    coeff_names : list[str], optional
        Names of non-intercept coefficients (length C). If None, auto-generates as ["x0", ...].
    include_intercept : bool, default True
        Whether the first column of beta_em is an intercept.
    intercept_name : str, default "Intercept"
        Name to use for the intercept coefficient (if include_intercept=True).
    figsize : tuple, default (16, 4)
        Figure size for the 1×3 layout.
    annot_transitions : bool, default True
        Whether to annotate transition heatmap values.
    show : bool, default True
        If True, calls plt.show() at the end.

    Returns
    -------
    dict
        {
            "fig": matplotlib.figure.Figure,
            "axs": np.ndarray of Axes (shape (3,)),
            "data": {
                "init": pd.DataFrame with columns ["state","prob"],
                "trans": pd.DataFrame with columns ["from","to","prob"],
                "coeffs": pd.DataFrame with columns ["state","coef_name","value"]
            }
        }
    """
    # ----------------- validate and coerce -----------------
    transitions = np.asarray(transitions, dtype=float)
    initial_probs = np.asarray(initial_probs, dtype=float)
    beta_em = np.asarray(beta_em, dtype=float)

    S = initial_probs.shape[0]
    if transitions.shape != (S, S):
        raise ValueError(f"transitions must be shape (S,S); got {transitions.shape} vs S={S}")
    if beta_em.shape[0] != S:
        raise ValueError(f"beta_em first dim must be S={S}; got {beta_em.shape}")

    if state_names is None:
        state_names = [f"State {i}" for i in range(S)]
    if len(state_names) != S:
        raise ValueError(f"state_names must have length S={S}")

    # ----------------- build coefficient names -----------------
    if include_intercept:
        C = beta_em.shape[1] - 1
        if C < 0:
            raise ValueError("beta_em must have at least 1 column when include_intercept=True")
        if coeff_names is None:
            coeff_names = [f"x{i}" for i in range(C)]
        if len(coeff_names) != C:
            raise ValueError(f"coeff_names must have length C={C}")
        coef_full_names = [intercept_name] + coeff_names
        beta_plot = beta_em
    else:
        C = beta_em.shape[1]
        if coeff_names is None:
            coeff_names = [f"x{i}" for i in range(C)]
        if len(coeff_names) != C:
            raise ValueError(f"coeff_names must have length C={C}")
        coef_full_names = coeff_names
        beta_plot = beta_em

    # ----------------- prepare dataframes for plotting -----------------
    df_init = pd.DataFrame({"state": state_names, "prob": initial_probs})

    rows, cols, vals = [], [], []
    for i in range(S):
        for j in range(S):
            rows.append(state_names[i])
            cols.append(state_names[j])
            vals.append(float(transitions[i, j]))
    df_trans = pd.DataFrame({"from": rows, "to": cols, "prob": vals})

    data_coef = []
    for s in range(S):
        for c_idx, name in enumerate(coef_full_names):
            data_coef.append({
                "state": state_names[s],
                "coef_name": name,
                "value": float(beta_plot[s, c_idx])
            })
    df_coef = pd.DataFrame(data_coef)

    # ----------------- figure and axes -----------------
    fig, axs = plt.subplots(1, 3, figsize=figsize)

    # Panel 1: Initial probabilities (bar)
    # ---------------------------------------------------
    ax0 = axs[0]
    palette = sns.color_palette("Set1", n_colors=S)
    ax0.bar(np.arange(S), df_init["prob"].values, color=palette)
    ax0.set_title("Initial State Probabilities")
    ax0.set_xlabel("State")
    ax0.set_ylabel("Probability")
    ax0.set_xticks(np.arange(S))
    ax0.set_xticklabels(state_names, rotation=0)
    ax0.set_ylim(0, max(1.0, df_init["prob"].max() * 1.1))
    ax0.grid(axis="y", alpha=0.3)

    # Panel 2: Transition matrix heatmap
    # ---------------------------------------------------
    ax1 = axs[1]
    mat = transitions.copy()
    sns.heatmap(
        mat,
        ax=ax1,
        cmap="Greens",
        annot=annot_transitions,
        fmt=".2f" if annot_transitions else "",
        cbar=False,
        xticklabels=state_names,
        yticklabels=state_names,
        vmin=0.0,
        vmax=max(1.0, mat.max())
    )
    ax1.set_title("Transition Probabilities")
    ax1.set_xlabel("Next State")
    ax1.set_ylabel("Current State")

    # Panel 3: Emission coefficients by state (grouped bar)
    # ---------------------------------------------------
    # x-axis = coefficient names; color = state; y = coefficient value
    ax2 = axs[2]
    coef_labels = coef_full_names
    M = len(coef_labels)  # number of coefficients (including intercept if present)
    x = np.arange(M)
    width = 0.8 / S

    for s_idx, s_name in enumerate(state_names):
        vals = df_coef.loc[df_coef["state"] == s_name, "value"].values
        if len(vals) != M:
            raise ValueError("Mismatch when slicing coefficients for grouped bars.")
        x_positions = x - 0.4 + width * (s_idx + 0.5)
        ax2.bar(x_positions, vals, width=width, color=palette[s_idx], label=s_name)

    ax2.axhline(0.0, color="black", linewidth=0.8)  # zero reference for signed coefficients
    ax2.set_title("Emission Coefficients by State")
    ax2.set_xlabel("Coefficient")
    ax2.set_ylabel("Value")
    ax2.set_xticks(x)
    ax2.set_xticklabels(coef_labels, rotation=90)
    ax2.legend(title="State", fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if show:
        plt.show()

    return {
        "fig": fig,
        "axs": axs,
        "data": {"init": df_init, "trans": df_trans, "coeffs": df_coef}
    }


from plotnine import (  # noqa: E402
    ggplot, aes, geom_point, geom_step,
    scale_color_manual, scale_x_continuous, scale_y_continuous,
    labs, theme_minimal, theme, element_text, element_line, element_blank,
    guides, guide_legend
)

def plot_one_gg(idx, obs_torch, paths, years=None, state_cols=None, y_max=4, title_prefix="Donor"):
    """
    Recreates the original scatter+step plot with plotnine:
      - observed counts (step + points)
      - points colored by discrete latent states with a clean legend
      - x-ticks labeled with years, y-limits clamped to [-0.5, y_max+0.5]
    """

    K_default   = 3
    state_cols_default = ['#e41a1c', '#377eb8', '#4daf4a']  # 3 Set1 colors

    # -------- extract donor data --------
    x = obs_torch[idx].detach().cpu().numpy() if hasattr(obs_torch, "detach") else np.asarray(obs_torch[idx])
    z = paths[idx].detach().cpu().numpy()     if hasattr(paths, "detach")     else np.asarray(paths[idx], dtype=int)
    years = np.asarray(years, dtype=int)

    T = len(x)
    if len(years) != T:
        raise ValueError("years length must match T for the selected donor.")

    # -------- palette and labels --------
    K = int(np.max(z)) + 1 if z.size > 0 else (state_cols and len(state_cols)) or K_default
    if state_cols is None:
        state_cols = state_cols_default[:K]
    state_labels = [f"State {k}" for k in range(K)]
    z_labs = [state_labels[s] for s in z]

    # -------- build dataframe ----------
    df = pd.DataFrame({
        "t": np.arange(T),
        "year": years,
        "donations": x,
        "state": z_labs
    })

    # -------- assemble plot ------------
    p = (
        ggplot(df, aes("t", "donations"))
        + geom_step(direction="mid", color="black", alpha=0.35)
        + geom_point(aes(color="state"), size=2.6)
        + scale_color_manual(values=state_cols, name="latent state",
                             breaks=state_labels, labels=state_labels)
        + scale_x_continuous(breaks=list(range(T)), labels=[str(y) for y in years])
        + scale_y_continuous(limits=(-0.5, float(y_max) + 0.5), breaks=list(range(0, int(y_max) + 1)))
        + labs(title=f"{title_prefix} {idx}", x="year", y="# donations")
        + theme_minimal()
        + theme(
            axis_text_x=element_text(rotation=45, ha="right"),
            legend_title=element_text(size=10),
            legend_text=element_text(size=9),
            plot_title=element_text(weight="bold"),
            # mimic dotted y-grid only
            panel_grid_major_y=element_line(linetype="dotted", alpha=0.4),
            panel_grid_major_x=element_blank(),
            panel_grid_minor=element_blank()
        )
        + guides(color=guide_legend(title="latent state"))
    )
    return p


# ============ 1) Utility per nomi covariate con fattori ===================
def expand_factor_names(var_name, levels, ref_level=None, prefix="[", suffix="]"):
    """
    Restituisce i nomi delle colonne dummy per un fattore.
    Se ref_level è dato, lo esclude (schema one-hot con base).
    """
    names = [f"{var_name}{prefix}{lev}{suffix}" for lev in levels]
    if ref_level is not None:
        names = [n for n in names if not n.endswith(f"{ref_level}{suffix}")]
    return names

# ============ 2) Heatmap W_pi (slopes su π) ===============================
def plot_W_pi_heat(W_pi, cov_names_pi=None, title="W_pi – slopes on log π"):
    K, C = W_pi.shape
    if not cov_names_pi or len(cov_names_pi) != C:
        cov_names_pi = [f"cov_{i}" for i in range(C)]

    plt.figure(figsize=(max(3, 0.9*C), max(2.8, 0.5*K + 1)))
    sns.heatmap(
        W_pi,
        annot=True, fmt=".2f",
        xticklabels=cov_names_pi,
        yticklabels=[f"S{k}" for k in range(K)],
        cmap="coolwarm", center=0
    )
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

# ============ 3) Heatmap W_A (slopes su transizioni) ======================
def plot_W_A_heat(W_A, cov_names_A=None, title="W_A – transition slopes"):
    K, _, C = W_A.shape
    if not cov_names_A or len(cov_names_A) != C:
        cov_names_A = [f"cov_{i}" for i in range(C)]

    fig, axes = plt.subplots(
        K, K,
        figsize=(max(2.0, 0.55*C)*K, max(2.0, 0.8)*K),
        sharex=True, sharey=True
    )
    vmin, vmax = W_A.min(), W_A.max()

    for i in range(K):
        for j in range(K):
            ax = axes[i, j] if K > 1 else axes
            mat = W_A[i, j].reshape(1, -1)  # (1,C)
            sns.heatmap(
                mat, ax=ax, vmin=vmin, vmax=vmax,
                cmap="coolwarm", cbar=False,
                xticklabels=cov_names_A, yticklabels=[]
            )
            ax.set_title(f"{i}→{j}", fontsize=8)
            ax.tick_params(axis="x", labelrotation=60)

    plt.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.show()

# ============ 4) Heatmap beta_em (GLM emissioni per stato) ================
def plot_beta_em_heat(beta_em, cov_names_em=None, title="beta_em – GLM emission coefficients"):
    """
    beta_em: (K, 1 + C_em)  [intercetta, slopes...]
    """
    K, P = beta_em.shape
    C_em = P - 1
    if not cov_names_em or len(cov_names_em) != C_em:
        cov_names_em = [f"em_{i}" for i in range(C_em)]
    colnames = ["Intercept"] + cov_names_em

    plt.figure(figsize=(max(3, 0.7*P), max(2.8, 0.5*K + 1)))
    sns.heatmap(
        beta_em,
        annot=True, fmt=".2f",
        xticklabels=colnames,
        yticklabels=[f"S{k}" for k in range(K)],
        cmap="coolwarm", center=0
    )
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def build_factor_cols(cov_names, factor_name, levels, ref_level):
    """
    Ricostruisce l'indice delle colonne dummificate per un fattore:
    cerca pattern 'factor_name[LEVEL]' in cov_names.
    Ritorna: dict {level -> col_index or None se livello è la base (reference)}.
    """
    name_to_idx = {n: i for i, n in enumerate(cov_names)}
    mapping = {}
    for lev in levels:
        col_name = f"{factor_name}[{lev}]"
        if lev == ref_level:
            mapping[lev] = None  # livello di riferimento → tutte dummies a 0
        else:
            if col_name not in name_to_idx:
                raise ValueError(f"Manca la colonna dummy {col_name} in cov_names.")
            mapping[lev] = name_to_idx[col_name]
    return mapping

def set_factor_level_in_vector(x_vec, factor_map, all_dummy_indices, level):
    """
    Imposta in-place il livello del fattore in un vettore design.
    - azzera tutte le dummies del fattore
    - se level ha una colonna dedicata, la pone a 1 (altrimenti resta tutto 0 → riferimento)
    """
    x_vec[all_dummy_indices] = 0.0
    idx = factor_map[level]
    if idx is not None:
        x_vec[idx] = 1.0
def original_values(
    var_name: str,
    df: pl.DataFrame | pd.DataFrame,
    ages_matrix: np.ndarray | None = None,
    factor_levels: dict | None = None,
) -> np.ndarray:
    """Return ORIGINAL values for the requested covariate."""
    if not isinstance(var_name, str):
        raise TypeError("var_name must be a string (did you swap arguments order?).")

    # Continuous variables
    if var_name == "birth_year_norm":
        if isinstance(df, pl.DataFrame):
            return df.get_column("birth_year").to_numpy()
        elif isinstance(df, pd.DataFrame):
            return df["birth_year"].to_numpy()
        else:
            raise TypeError("df must be a Polars or Pandas DataFrame.")

    if var_name == "ages_norm":
        if ages_matrix is None:
            raise ValueError("ages_matrix must be provided for var_name == 'ages_norm'.")
        return np.asarray(ages_matrix).reshape(-1)

    # Binary variables
    if var_name == "gender_code":
        return np.array([0.0, 1.0])
    if var_name == "covid_years":
        return np.array([0.0, 1.0])

    # Factor variables via declared levels
    if factor_levels and var_name in factor_levels:
        return np.array(factor_levels[var_name], dtype=object)

    # Fallback for Pandas categorical/object
    if isinstance(df, pd.DataFrame) and var_name in df.columns:
        col = df[var_name]
        if pd.api.types.is_categorical_dtype(col) or pd.api.types.is_object_dtype(col):
            return np.array(sorted(pd.unique(col.astype(str))), dtype=object)

    raise ValueError(f"Unknown covariate {var_name} (provide factor_levels if it is a factor).")


def to_norm(var_name: str, x_orig: np.ndarray, stats: dict | None = None) -> np.ndarray:
    """Convert ORIGINAL values to z-scored scale for continuous covariates."""
    if stats is None:
        stats = globals().get("stats", {})
    if var_name in stats:
        mu, sd = stats[var_name]
        sd = 1.0 if (sd is None or sd == 0.0) else sd
        return (np.asarray(x_orig) - mu) / sd
    return np.asarray(x_orig)


def plot_pi_vs_cov_orig(
    df,
    ages,
    var,
    cov_names_pi,
    W_pi,
    log_pi0,
    x_pi_data=None,               # (N, C_pi) already normalised
    x_pi_ref=None,                # (C_pi,) optional
    factor_specs_pi=None,         # {"factor_name": {"levels":[...], "ref":"..."}}
    grid_orig=None,
    state_cols=None,
    title_prefix="Initial-state probability vs "
):
    """Plot π_k(x) against one covariate on its original scale (continuous or factor)."""
    # --- helpers used below ---
    def softmax(v):
        v = v - v.max()
        e = np.exp(v)
        return e / e.sum()

    def set_factor_level_in_vector(x_vec, factor_map, all_dummy_indices, level):
        x_vec[all_dummy_indices] = 0.0
        idx = factor_map[level]
        if idx is not None:
            x_vec[idx] = 1.0

    def build_factor_cols(cov_names, factor_name, levels, ref_level):
        name_to_idx = {n: i for i, n in enumerate(cov_names)}
        mapping = {}
        for lev in levels:
            col_name = f"{factor_name}[{lev}]"
            if lev == ref_level:
                mapping[lev] = None
            else:
                if col_name not in name_to_idx:
                    raise ValueError(f"Missing dummy column {col_name} in cov_names.")
                mapping[lev] = name_to_idx[col_name]
        return mapping

    # --- setup ---
    K = W_pi.shape[0]
    if state_cols is None:
        state_cols = ['#e41a1c', '#377eb8', '#4daf4a'][:K]

    # Reference vector (normalised)
    if x_pi_ref is None:
        if x_pi_data is None:
            if "cov_init_torch" in globals():
                x_pi_data = cov_init_torch.detach().cpu().numpy()  # noqa: F821
            else:
                raise ValueError("Provide x_pi_data or x_pi_ref.")
        x_ref_norm = x_pi_data.mean(0)   # (C_pi,)
    else:
        x_ref_norm = np.asarray(x_pi_ref).copy()

    # Factor branch
    is_factor = factor_specs_pi is not None and var in factor_specs_pi
    if is_factor:
        levels = factor_specs_pi[var]["levels"]
        ref    = factor_specs_pi[var].get("ref", None)
        factor_map = build_factor_cols(cov_names_pi, var, levels, ref)
        dummy_indices = [idx for idx in factor_map.values() if idx is not None]
        grid_levels = levels if grid_orig is None else list(grid_orig)

        curves = []
        for lev in grid_levels:
            x_norm = x_ref_norm.copy()
            set_factor_level_in_vector(x_norm, factor_map, np.array(dummy_indices, dtype=int), lev)
            logits = log_pi0 + W_pi @ x_norm
            curves.append(softmax(logits))
        curves = np.vstack(curves)  # (G, K)

        fig, ax = plt.subplots(figsize=(7, 3.2))
        x_pos = np.arange(len(grid_levels))
        width = 0.8 / K
        for k, c in enumerate(state_cols):
            ax.bar(x_pos + k*width - 0.4 + width*K/2, curves[:, k], width=width, color=c, label=f"state {k}")
        ax.set_xticks(x_pos); ax.set_xticklabels(grid_levels, rotation=45, ha="right")
        ax.set_ylabel("π_k(x)")
        ax.set_title(f"{title_prefix}{var}")
        ax.grid(axis="y", ls=":", alpha=0.4)
        ax.legend()
        plt.tight_layout(); plt.show()
        return

    # Continuous/binary branch
    if var not in cov_names_pi:
        raise ValueError(f"{var} not in cov_names_pi and not declared as factor.")
    idx = cov_names_pi.index(var)

    # Correct call order (var first, then df, then ages)
    col_orig = original_values(var_name=var, df=df, ages_matrix=ages, factor_levels=None)

    # Build ORIGINAL grid
    if grid_orig is None:
        uniq = np.unique(col_orig)
        grid_orig = uniq if len(uniq) <= 6 else np.linspace(col_orig.min(), col_orig.max(), 41)

    curves = []
    for v_orig in grid_orig:
        x_norm = x_ref_norm.copy()
        x_norm[idx] = to_norm(var, v_orig)
        logits = log_pi0 + W_pi @ x_norm
        curves.append(softmax(logits))
    curves = np.vstack(curves)

    # Plot
    for k, c in enumerate(state_cols):
        plt.plot(grid_orig, curves[:, k], color=c, label=f"state {k}")
    plt.xlabel(var.replace("_norm", ""));  plt.ylabel("π_k(x)")
    plt.title(f"{title_prefix}{var.replace('_norm', '')}")
    plt.grid(ls=":", alpha=0.5);  plt.legend();  plt.tight_layout();  plt.show()

# ==============================================================
# 6) λ_k(x_em) del GLM emissioni vs covariata (fattori/continui)
# ==============================================================

def plot_lambda_em_vs_cov(
    var_em,
    beta_em,                # (K, 1 + C_em)
    cov_names_em,           # lunghezza = C_em
    x_em_data=None,         # (N, T, C_em) o (N, C_em) per media di riferimento
    x_em_ref=None,          # (C_em,) opzionale
    factor_specs_em=None,   # dict: { "age_years": {"levels":[...], "ref":"<25"} , ... }
    grid_orig=None,
    state_cols=None,
    title_prefix="Emission GLM λ_k vs "
):
    """
    Plotta la λ_k = exp(b0_k + x_em · B_k) variando una sola covariata di emissione.
    Gestisce sia continui che fattori (dummies con riferimento).
    """
    K, P = beta_em.shape
    C_em = P - 1
    if state_cols is None:
        state_cols = ['#e41a1c', '#377eb8', '#4daf4a'][:K]

    # reference su x_em
    if x_em_ref is None:
        if x_em_data is None:
            if "cov_emiss_torch" in globals():
                X = cov_emiss_torch.cpu().numpy()  # noqa: F821
                x_em_ref = X.mean(axis=(0, 1))  # media su N,T
            else:
                raise ValueError("Serve x_em_data o x_em_ref per definire il vettore di riferimento.")
        else:
            X = x_em_data
            if X.ndim == 3:
                x_em_ref = X.mean(axis=(0, 1))
            elif X.ndim == 2:
                x_em_ref = X.mean(axis=0)
            else:
                raise ValueError("x_em_data deve essere (N,T,C_em) o (N,C_em).")

    b0 = beta_em[:, 0]      # (K,)
    B  = beta_em[:, 1:]     # (K, C_em)

    # fattore?
    is_factor = factor_specs_em is not None and var_em in factor_specs_em

    if is_factor:
        levels = factor_specs_em[var_em]["levels"]
        ref    = factor_specs_em[var_em]["ref"]
        # ricostruisci mappa livello->indice colonna
        fac_map = build_factor_cols(cov_names_em, var_em, levels, ref)
        dummy_idx = [i for i in fac_map.values() if i is not None]
        grid_levels = levels if grid_orig is None else grid_orig

        lam = np.zeros((len(grid_levels), K))
        for g, lev in enumerate(grid_levels):
            x_ref = x_em_ref.copy()
            # azzera dummies e setta il livello
            if dummy_idx:
                x_ref[dummy_idx] = 0.0
            idx = fac_map[lev]
            if idx is not None:
                x_ref[idx] = 1.0
            eta = b0 + B @ x_ref
            lam[g] = np.exp(eta)

        # plot bar per stato
        x_pos = np.arange(len(grid_levels))
        width = 0.8 / K
        fig, ax = plt.subplots(figsize=(7, 3.2))
        for k, c in enumerate(state_cols):
            ax.bar(x_pos + k*width - 0.4 + width*K/2, lam[:, k], width=width, color=c, label=f"state {k}")
        ax.set_xticks(x_pos); ax.set_xticklabels(grid_levels, rotation=45, ha="right")
        ax.set_ylabel("λ_k")
        ax.set_title(f"{title_prefix}{var_em}")
        ax.grid(axis="y", ls=":", alpha=0.4)
        ax.legend()
        plt.tight_layout(); plt.show()

    else:
        # continuo / binario singolo
        if var_em not in cov_names_em:
            raise ValueError(f"{var_em} non è in cov_names_em e non risulta specificato come fattore.")
        j = cov_names_em.index(var_em)
        # griglia ORIGINAL (niente z-score: emissioni GLM usano la scala del design)
        if grid_orig is None:
            # prova a inferire dal campione se disponibile
            if x_em_data is not None:
                X = x_em_data
                vals = X.reshape(-1, X.shape[-1])[:, j]
                uniq = np.unique(vals)
                grid_orig = uniq if len(uniq) <= 6 else np.linspace(vals.min(), vals.max(), 41)
            else:
                # fallback generico
                grid_orig = np.linspace(-2, 2, 41)

        lam = np.zeros((len(grid_orig), K))
        for g, v in enumerate(grid_orig):
            x_ref = x_em_ref.copy()
            x_ref[j] = v
            eta = b0 + B @ x_ref
            lam[g] = np.exp(eta)

        for k, c in enumerate(state_cols):
            plt.plot(grid_orig, lam[:, k], color=c, label=f"state {k}")
        plt.xlabel(var_em); plt.ylabel("λ_k")
        plt.title(f"{title_prefix}{var_em}")
        plt.grid(ls=":", alpha=0.5); plt.legend(); plt.tight_layout(); plt.show()

# ---------- helpers ---------------------------------------------------------
def softmax(v):
    v = v - v.max()
    e = np.exp(v)
    return e / e.sum()



# Palette default
def default_state_cols(K):
    base = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33',
            '#a65628', '#f781bf', '#999999']
    return (base * ((K + len(base) - 1)//len(base)))[:K]

# =====================================================================
# Transition probabilities vs covariata (ORIGINAL scale, fattori ok)
# =====================================================================
def plot_trans_vs_cov_orig(
    var,
    prev_state=0,
    grid_orig=None,
    cov_names_A=None,
    x_A_data=None,        # (N,T,C_A) per vettore di riferimento (media)
    W_A=None, log_A0=None,
    factor_specs_A=None,  # es.: {"age_years": {"levels": [...], "ref": None}}
    state_cols=None
):
    # Recuperi di default se non passati
    if cov_names_A is None:
        raise ValueError("Serve cov_names_A (lista nomi colonne di A).")
    if (W_A is None) or (log_A0 is None):
        W_A, log_A0 = hmm_glm.get_W_A_and_logA()
    if x_A_data is None:
        # prova da tensore globale
        if "cov_tran_torch" in globals():
            x_A_data = cov_tran_torch.detach().cpu().numpy()  # noqa: F821
        else:
            raise ValueError("Serve x_A_data (N,T,C_A) o cov_tran_torch globale.")
    K = W_A.shape[0]
    if state_cols is None:
        state_cols = default_state_cols(K)
    if prev_state >= K:
        raise ValueError("prev_state out of range")

    # vettore di riferimento (media su N,T)
    x_ref = x_A_data.mean(axis=(0,1)).copy()   # (C_A,)

    # Caso FATTORIALE (var è un fattore, es. "age_years")
    is_factor = (factor_specs_A is not None) and (var in factor_specs_A)
    if is_factor:
        levels = factor_specs_A[var]["levels"]
        ref    = factor_specs_A[var].get("ref", None)
        factor_map, all_dummy_idx = build_factor_cols(cov_names_A, var, levels, ref)

        mats = np.zeros((len(levels), K))
        for g, lev in enumerate(levels):
            x_vec = x_ref.copy()
            set_factor_level_in_vector(x_vec, factor_map, all_dummy_idx, lev)
            logits = log_A0[prev_state] + (W_A[prev_state] @ x_vec)   # (K,)
            mats[g] = softmax(logits)

        # barplot raggruppato per j
        fig, ax = plt.subplots(figsize=(8, 3.2))
        x_pos = np.arange(len(levels))
        width = 0.8 / K
        for j, c in enumerate(state_cols):
            ax.bar(x_pos + j*width - 0.4 + width*K/2, mats[:, j], width=width, color=c, label=f"{prev_state}→{j}")
        ax.set_xticks(x_pos); ax.set_xticklabels(levels, rotation=45, ha="right")
        ax.set_ylabel("transition prob."); ax.set_xlabel(var)
        ax.set_title(f"Transition from state {prev_state} vs {var}")
        ax.grid(axis="y", ls=":", alpha=0.4); ax.legend(); plt.tight_layout(); plt.show()
        return

    # Caso CONTINUO/BINARIO (var è una singola colonna di A)
    if var not in cov_names_A:
        raise ValueError(f"{var} non trovato in cov_names_A e non specificato come fattore.")
    idx = cov_names_A.index(var)

    # grid (ORIGINALE): se binaria → {0,1}
    if grid_orig is None:
        # prova a inferire dai dati
        vals = x_A_data[..., idx].reshape(-1)
        uniq = np.unique(vals)
        grid_orig = uniq if len(uniq) <= 6 else np.linspace(vals.min(), vals.max(), 41)

    mats = np.zeros((len(grid_orig), K))
    for g, v in enumerate(grid_orig):
        x_vec = x_ref.copy()
        x_vec[idx] = v
        logits = log_A0[prev_state] + (W_A[prev_state] @ x_vec)
        mats[g] = softmax(logits)

    # line plot
    for j, c in enumerate(state_cols):
        plt.plot(grid_orig, mats[:, j], color=c, label=f"{prev_state}→{j}")
    plt.xlabel(var); plt.ylabel("transition prob.")
    plt.title(f"Transition from state {prev_state} vs {var}")
    plt.grid(ls=":"); plt.legend(); plt.tight_layout(); plt.show()

# =====================================================================
# Expected E[y0 | x] vs π-covariata (ORIG scale) con emissioni GLM
# =====================================================================
def expected_y_orig(
    var: str,
    cov_names_pi: Iterable[str],
    W_pi: np.ndarray,                 # (K, C_pi)
    log_pi0: np.ndarray,              # (K,)
    *,
    # Emission parameters (choose one branch)
    rates: Optional[np.ndarray] = None,       # (K,)
    beta_em: Optional[np.ndarray] = None,     # (K, 1 + C_em)
    x_em_ref: Optional[np.ndarray] = None,    # (C_em,) required if beta_em is provided
    # Reference for π(x)
    x_pi_data: Optional[np.ndarray] = None,   # (N, C_pi) normalized; used to build mean reference if x_pi_ref is None
    x_pi_ref: Optional[np.ndarray] = None,    # (C_pi,) normalized; overrides x_pi_data
    # Grid and helpers
    grid_orig: Optional[np.ndarray] = None,   # values on ORIGINAL scale for the selected var
    to_norm_fn: Optional[Callable[[str, np.ndarray], np.ndarray]] = None,   # original → normalized
    original_values_fn: Optional[Callable[[str], np.ndarray]] = None,       # returns original values for var
    # Plotting
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    show: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute and plot E[y0 | x(var)] as the selected covariate varies on its ORIGINAL scale.

    Parameters
    ----------
    var : str
        Covariate name to vary (must be present in cov_names_pi).
    cov_names_pi : list[str]
        Names of π covariates (order must match columns of W_pi / x_pi_data).
    W_pi : (K, C_pi)
        Slopes for initial-state logits.
    log_pi0 : (K,)
        Log of base initial-state probabilities (same base used during training).
    rates : (K,), optional
        Poisson rates per state (used if beta_em is None).
    beta_em : (K, 1 + C_em), optional
        Emission GLM coefficients per state. First column is intercept.
    x_em_ref : (C_em,), optional
        Reference emission covariates to compute λ_k = exp(b0_k + x_em_ref · B_k).
        Required when beta_em is provided.
    x_pi_data : (N, C_pi), optional
        Normalized π covariates; used only to compute a mean reference vector if x_pi_ref is None.
    x_pi_ref : (C_pi,), optional
        Normalized reference vector for π; if provided, overrides x_pi_data.
    grid_orig : array, optional
        Values on original scale for the selected covariate. If None, tries original_values_fn(var).
    to_norm_fn : callable, optional
        Function mapping original → normalized scale: to_norm_fn(var, x_orig) -> x_norm.
        If None, identity is used (assumes inputs are already normalized).
    original_values_fn : callable, optional
        Function returning original values for var: original_values_fn(var) -> array.
        Used only to build a default grid when grid_orig is None.
    ax : matplotlib Axes, optional
        Axes object to plot on. If None, a new figure is created.
    title : str, optional
        Custom plot title. If None, a default title is used.
    show : bool, default True
        If True, calls plt.show() at the end.

    Returns
    -------
    grid_orig : np.ndarray
        Grid on original scale used for plotting.
    exp_vals : np.ndarray
        Expected values E[y0 | x] for each grid point.
    """
    cov_names_pi = list(cov_names_pi)
    if var not in cov_names_pi:
        raise ValueError(f"{var} is not present in cov_names_pi.")

    K, C_pi = W_pi.shape
    if log_pi0.shape[0] != K:
        raise ValueError(f"log_pi0 must have shape (K,), got {log_pi0.shape} for K={K}.")

    # Emission: choose GLM if beta_em is provided; else use rates
    use_glm = beta_em is not None
    if use_glm:
        if x_em_ref is None:
            raise ValueError("x_em_ref is required when beta_em is provided.")
        if beta_em.shape[0] != K:
            raise ValueError(f"beta_em first dimension must be K={K}, got {beta_em.shape}.")
        b0 = beta_em[:, 0]      # (K,)
        B  = beta_em[:, 1:]     # (K, C_em)
        if B.shape[1] != x_em_ref.shape[0]:
            raise ValueError("beta_em and x_em_ref have incompatible shapes.")
        lam = np.exp(b0 + B @ x_em_ref)  # (K,)
    else:
        if rates is None:
            raise ValueError("Provide either beta_em (with x_em_ref) or rates.")
        if rates.shape[0] != K:
            raise ValueError(f"rates must have shape (K,), got {rates.shape}.")
        lam = rates.copy()

    # Reference vector for π(x) on normalized scale
    if x_pi_ref is None:
        if x_pi_data is None:
            raise ValueError("Provide x_pi_ref or x_pi_data to build the reference for π.")
        x_pi_ref = np.asarray(x_pi_data, dtype=float).mean(axis=0)
    else:
        x_pi_ref = np.asarray(x_pi_ref, dtype=float)
    if x_pi_ref.shape[0] != C_pi:
        raise ValueError(f"x_pi_ref must be length C_pi={C_pi}, got {x_pi_ref.shape}.")

    # Build ORIGINAL grid for the selected covariate
    j = cov_names_pi.index(var)
    if grid_orig is None:
        if original_values_fn is not None:
            raw_vals = np.asarray(original_values_fn(var))
            if raw_vals.dtype.kind in ("U", "S", "O"):
                # Factor/categorical: use unique sorted levels
                uniq = np.unique(raw_vals)
                grid_orig = uniq
            else:
                # Continuous: dense grid over observed range
                lo, hi = float(np.min(raw_vals)), float(np.max(raw_vals))
                grid_orig = np.linspace(lo, hi, 41)
        else:
            # Fallback: use a reasonable numeric grid
            grid_orig = np.linspace(-2.0, 2.0, 41)

    grid_orig = np.asarray(grid_orig)
    is_categorical = grid_orig.dtype.kind in ("U", "S", "O")

    # Define softmax over a vector
    def softmax_vec(v: np.ndarray) -> np.ndarray:
        v = v - np.max(v)
        e = np.exp(v)
        return e / np.sum(e)

    # Convert original value → normalized using provided function (identity if None)
    def to_norm_value(var_name: str, x_orig) -> float:
        if to_norm_fn is None:
            # Identity mapping: assumes x_orig is already normalized
            return float(x_orig)
        x_arr = np.asarray([x_orig], dtype=float)
        v = to_norm_fn(var_name, x_arr)
        return float(v[0]) if np.ndim(v) > 0 else float(v)

    # Compute expected values along the grid
    exp_vals = np.zeros(len(grid_orig), dtype=float)
    for g, v_orig in enumerate(grid_orig):
        x_vec = x_pi_ref.copy()
        if is_categorical:
            # Categorical var in π is expected to be dummy-coded in cov_names_pi.
            # We turn off all dummies of this factor and set the one matching v_orig to 1.
            prefix = f"{var}["
            all_dummy_idx = [k for k, name in enumerate(cov_names_pi) if name.startswith(prefix) and name.endswith("]")]
            if all_dummy_idx:
                x_vec[np.array(all_dummy_idx, dtype=int)] = 0.0
            level_name = f"{var}[{v_orig}]"
            if level_name in cov_names_pi:
                x_vec[cov_names_pi.index(level_name)] = 1.0
        else:
            x_vec[j] = to_norm_value(var, v_orig)

        logits = log_pi0 + W_pi @ x_vec
        pi_x = softmax_vec(logits)  # (K,)
        exp_vals[g] = float(np.dot(pi_x, lam))

    # Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 3.0))
    ax.plot(grid_orig, exp_vals, "-o")
    ax.set_xlabel(var.replace("_norm", ""))
    ax.set_ylabel("E[y₀ | x]")
    ax.set_title(title or f"Expected count at t=0 vs {var.replace('_norm','')}")
    ax.grid(ls=":", alpha=0.5)
    plt.tight_layout()
    if show:
        plt.show()

    return grid_orig, exp_vals