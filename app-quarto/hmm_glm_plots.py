# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import polars as pl
from pyprojroot import here
import sys


from typing import Callable, Iterable, Optional, Tuple

# data visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotnine as pn

# plot settings
from matplotlib import font_manager as fm
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap

python_dir = str(here("python"))
if python_dir not in sys.path:
    sys.path.insert(0, python_dir)

plt.rcParams["axes.prop_cycle"] = cycler(color=["#8c1c13ff", "#86ba90ff", "#54403bff"])
plt.rcParams["figure.facecolor"] = "#F4ECE2"
fm.fontManager.addfont(here("python/Figtree-Regular.ttf"))
palette = ["#8c1c13ff", "#df9457ff", "#86ba90ff", "#54403bff"]
STATE_PALETTE = {
    0: "#8c1c13",  # es. rosso scuro  (state 0)
    1: "#df9457",  # es. arancione     (state 1)
    2: "#86ba90",  # es. verde         (state 2)
}


def colors_for_states(K: int, mapping: dict[int, str] = STATE_PALETTE) -> list[str]:
    return [mapping.get(k, "#999999") for k in range(K)]


# Custom colormaps from theme colors
TRANS_CMAP = LinearSegmentedColormap.from_list("trans_cmap", ["#E5F0E7", "#4A8255"])
EMISS_CMAP = LinearSegmentedColormap.from_list("emiss_cmap", ["#f4ece2", "#8c1c13ff"])
COEFF_CMAP = LinearSegmentedColormap.from_list(
    "emiss_cmap", ["#8c1c13ff", "#f4ece2", "#86ba90"]
)


# ---------- Subplots ----------
def plot_initial_probs(ax, initial_probs, state_names, colors):
    """
    Plot the initial state probabilities as a bar chart.
    """
    S = len(initial_probs)
    ax.bar(np.arange(S), initial_probs, color=colors[:S])
    ax.set_title("Initial State Probabilities")
    ax.set_xlabel("State")
    ax.set_ylabel("Probability")
    ax.set_xticks(np.arange(S))
    ax.set_xticklabels(state_names)
    ax.set_ylim(0, max(1.0, initial_probs.max() * 1.1))
    ax.grid(axis="y", alpha=0.3)


def plot_transition_matrix(ax, transitions, state_names, annot=True):
    """
    Plot the transition probability matrix as a heatmap using custom theme colors.
    """
    sns.heatmap(
        transitions,
        ax=ax,
        cmap=TRANS_CMAP,
        annot=annot,
        fmt=".2f" if annot else "",
        cbar=False,
        xticklabels=state_names,
        yticklabels=state_names,
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_title("Transition Probabilities")
    ax.set_xlabel("Next State")
    ax.set_ylabel("Current State")


def plot_emission_coeffs(ax, beta_em, state_names, coeff_names, colors):
    """
    Plot emission GLM coefficients for each state as grouped bar chart.
    """
    S, C = beta_em.shape
    x = np.arange(C)
    width = 0.8 / S

    for s_idx, s_name in enumerate(state_names):
        vals = beta_em[s_idx, :]
        x_positions = x - 0.4 + width * (s_idx + 0.5)
        ax.bar(x_positions, vals, width=width, color=colors[s_idx], label=s_name)

    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Emission Coefficients by State")
    ax.set_xlabel("Coefficient")
    ax.set_ylabel("Value")
    ax.set_xticks(x)
    ax.set_xticklabels(coeff_names, rotation=90)
    ax.legend(title="State", fontsize=9)
    ax.grid(axis="y", alpha=0.3)


# ---------- Wrapper ----------
def plot_hmm_params_with_coeffs(
    transitions,
    initial_probs,
    beta_em,
    state_names=None,
    coeff_names=None,
    figsize=(16, 4),
    annot_transitions=True,
    show=True,
):
    """
    Visual summary of HMM parameters with emission GLM coefficients.
    """
    transitions = np.asarray(transitions, dtype=float)
    initial_probs = np.asarray(initial_probs, dtype=float)
    beta_em = np.asarray(beta_em, dtype=float)

    S = initial_probs.shape[0]
    if state_names is None:
        state_names = [f"State {i}" for i in range(S)]
    if coeff_names is None:
        coeff_names = [f"x{i}" for i in range(beta_em.shape[1])]

    colors = colors_for_states(S)

    fig, axs = plt.subplots(1, 3, figsize=figsize)

    # Call subplots
    plot_initial_probs(axs[0], initial_probs, state_names, colors)
    plot_transition_matrix(axs[1], transitions, state_names, annot=annot_transitions)
    plot_emission_coeffs(axs[2], beta_em, state_names, coeff_names, colors)

    plt.tight_layout()
    if show:
        plt.show()

    return {
        "fig": fig,
        "axs": axs,
        "data": {
            "init": pd.DataFrame({"state": state_names, "prob": initial_probs}),
            "trans": pd.DataFrame(
                {
                    "from": np.repeat(state_names, S),
                    "to": np.tile(state_names, S),
                    "prob": transitions.flatten(),
                }
            ),
            "coeffs": pd.DataFrame(
                [
                    {
                        "state": state_names[s],
                        "coef_name": coeff_names[c],
                        "value": beta_em[s, c],
                    }
                    for s in range(S)
                    for c in range(len(coeff_names))
                ]
            ),
        },
    }


from plotnine import (  # noqa: E402
    ggplot,
    aes,
    geom_point,
    geom_step,
    scale_color_manual,
    scale_x_continuous,
    scale_y_continuous,
    labs,
    theme_minimal,
    theme,
    element_text,
    element_line,
    element_blank,
    guides,
    guide_legend,
)


def plot_one_gg(
    idx, obs_torch, paths, years=None, state_cols=None, y_max=4, title_prefix="Donor"
):
    """
    Recreates the original scatter+step plot with plotnine:
      - observed counts (step + points)
      - points colored by discrete latent states with a clean legend
      - x-ticks labeled with years, y-limits clamped to [-0.5, y_max+0.5]
    """
    # -------- extract donor data --------
    x = (
        obs_torch[idx].detach().cpu().numpy()
        if hasattr(obs_torch, "detach")
        else np.asarray(obs_torch[idx])
    )
    z = (
        paths[idx].detach().cpu().numpy()
        if hasattr(paths, "detach")
        else np.asarray(paths[idx], dtype=int)
    )
    years = np.asarray(years, dtype=int)

    unique_states = np.unique(z)
    # Costruisci etichette e dizionario label->colore coerente con gli indici reali
    state_to_label = {int(s): f"State {int(s)}" for s in unique_states}
    z_labs = [state_to_label[int(s)] for s in z]
    label_to_color = {
        state_to_label[k]: STATE_PALETTE.get(int(k), "#999999") for k in unique_states
    }
    state_labels = [state_to_label[int(s)] for s in unique_states]  # ordine per legenda

    T = len(x)
    if len(years) != T:
        raise ValueError("years length must match T for the selected donor.")

    df = pd.DataFrame(
        {"t": np.arange(T), "year": years, "donations": x, "state": z_labs}
    )

    # -------- assemble plot ------------
    p = (
        ggplot(df, aes("t", "donations"))
        + geom_step(direction="mid", color="black", alpha=0.35)
        + geom_point(aes(color="state"), size=2.6)
        + scale_color_manual(
            values=label_to_color, breaks=state_labels, labels=state_labels
        )
        + scale_x_continuous(breaks=list(range(T)), labels=[str(y) for y in years])
        + scale_y_continuous(
            limits=(-0.5, float(y_max) + 0.5), breaks=list(range(0, int(y_max) + 1))
        )
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
            panel_grid_minor=element_blank(),
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

    plt.figure(figsize=(max(3, 0.9 * C), max(2.8, 0.5 * K + 1)))
    sns.heatmap(
        W_pi,
        annot=True,
        fmt=".2f",
        xticklabels=cov_names_pi,
        yticklabels=[f"S{k}" for k in range(K)],
        cmap=COEFF_CMAP,
        center=0,
    )
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    return plt


# ============ 3) Heatmap W_A (slopes su transizioni) ======================
def plot_W_A_heat(W_A, cov_names_A=None, title="W_A – transition slopes"):
    K, _, C = W_A.shape
    if not cov_names_A or len(cov_names_A) != C:
        cov_names_A = [f"cov_{i}" for i in range(C)]

    fig, axes = plt.subplots(
        K,
        K,
        figsize=(max(2.0, 0.55 * C) * K, max(2.0, 0.8) * K),
        sharex=True,
        sharey=True,
    )
    vmin, vmax = W_A.min(), W_A.max()

    for i in range(K):
        for j in range(K):
            ax = axes[i, j] if K > 1 else axes
            mat = W_A[i, j].reshape(1, -1)  # (1,C)
            sns.heatmap(
                mat,
                ax=ax,
                vmin=vmin,
                vmax=vmax,
                annot=True,
                fmt=".1f",
                cmap=COEFF_CMAP,
                cbar=False,
                xticklabels=cov_names_A,
                yticklabels=[],
            )
            ax.set_title(f"{i}→{j}", fontsize=8)
            ax.tick_params(axis="x", labelrotation=60)

    plt.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.show()

    return plt


# ============ 4) Heatmap beta_em (GLM emissioni per stato) ================
def plot_beta_em_heat(
    beta_em, cov_names_em=None, title="beta_em – GLM emission coefficients"
):
    K, P = beta_em.shape
    C_em = P
    if not cov_names_em or len(cov_names_em) != C_em:
        cov_names_em = [f"em_{i}" for i in range(C_em)]

    plt.figure(figsize=(max(3, 0.7 * P), max(2.8, 0.5 * K + 1)))
    sns.heatmap(
        beta_em,
        annot=True,
        fmt=".2f",
        xticklabels=cov_names_em,
        yticklabels=[f"S{k}" for k in range(K)],
        cmap=COEFF_CMAP,
        center=0,
    )
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


# ==============================================================
# 2) Gestione fattori (one-hot con riferimento)
# ==============================================================
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


# ==============================================================
# 3) ORIGINAL values per covariate (continui, binari, fattori)
# ==============================================================
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
            raise ValueError(
                "ages_matrix must be provided for var_name == 'ages_norm'."
            )
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

    raise ValueError(
        f"Unknown covariate {var_name} (provide factor_levels if it is a factor)."
    )


def to_norm(var_name: str, x_orig: np.ndarray, stats: dict | None = None) -> np.ndarray:
    """Convert ORIGINAL values to z-scored scale for continuous covariates."""
    if stats is None:
        stats = globals().get("stats", {})
    if var_name in stats:
        mu, sd = stats[var_name]
        sd = 1.0 if (sd is None or sd == 0.0) else sd
        return (np.asarray(x_orig) - mu) / sd
    return np.asarray(x_orig)


def _softmax_vec(v: np.ndarray) -> np.ndarray:
    v = v - np.max(v)
    e = np.exp(v)
    return e / np.sum(e)


def _default_state_cols(K: int):
    base = palette
    return (base * ((K + len(base) - 1) // len(base)))[:K]


def _df_columns(df):
    try:
        return list(df.columns)
    except Exception:
        return []


def _has_col(df, name: str) -> bool:
    try:
        return name in df.columns
    except Exception:
        try:
            df[name]
            return True
        except Exception:
            try:
                df.get_column(name)
                return True
            except Exception:
                return False


def _col_to_numpy(df, name: str) -> np.ndarray:
    # Try Pandas then Polars accessors
    try:
        arr = df[name].to_numpy()
        return np.asarray(arr)
    except Exception:
        try:
            arr = df.get_column(name).to_numpy()
            return np.asarray(arr)
        except Exception:
            raise KeyError(
                f"Column '{name}' not found or cannot be converted to numpy."
            )


def _infer_years_num(df) -> Optional[np.ndarray]:
    cols = _df_columns(df)
    year_cols = [c for c in cols if isinstance(c, str) and c.startswith("y_")]
    years = []
    for c in year_cols:
        try:
            years.append(int(c[2:]))
        except Exception:
            pass
    if years:
        years = sorted(set(years))
        return np.array(years, dtype=int)
    return None


def plot_pi_vs_cov_orig(
    *,
    df,
    ages: np.ndarray,  # (N,T) already computed
    var: str,
    cov_names_pi: Iterable[str],
    W_pi: np.ndarray,  # (K, C_pi)
    log_pi0: np.ndarray,  # (K,)
    x_pi_data: np.ndarray,  # (N, C_pi) to build reference
    factor_specs_pi: Optional[dict] = None,  # not used here but kept for API
    grid_orig: Optional[np.ndarray] = None,
    to_norm_fn: Optional[Callable[[str, np.ndarray], np.ndarray]] = None,
    state_cols: Optional[list[str]] = None,
    title_prefix: str = "π_k(x) vs ",
) -> Tuple[np.ndarray, np.ndarray]:
    cov_names_pi = list(cov_names_pi)
    K, C_pi = W_pi.shape
    if state_cols is None:
        state_cols = _default_state_cols(K)

    # Reference (normalized)
    x_ref = np.asarray(x_pi_data, dtype=float).mean(axis=0).copy()
    if x_ref.shape[0] != C_pi:
        raise ValueError(
            f"x_pi_data has C={x_ref.shape[0]} but W_pi expects C_pi={C_pi}"
        )

    # Only continuous/binary handled here (no factor for π in your setup)
    if var not in cov_names_pi:
        raise ValueError(
            f"{var} non trovato in cov_names_pi e non specificato come fattore."
        )
    j = cov_names_pi.index(var)

    # Build ORIGINAL grid for birth_year_norm (from df or reconstruct), else use normalized grid
    def default_to_norm(var_name: str, x_orig: np.ndarray) -> np.ndarray:
        x_orig = np.asarray(x_orig, dtype=float)
        # If we vary birth_year_norm on original birth_year
        if var_name == "birth_year_norm":
            # Prefer df['birth_year'] if present; else reconstruct from ages and year_cols
            if _has_col(df, "birth_year"):
                col = _col_to_numpy(df, "birth_year").astype(float)
            else:
                years_num = _infer_years_num(df)
                if (
                    years_num is not None
                    and ages is not None
                    and ages.ndim == 2
                    and ages.shape[1] >= 1
                ):
                    # reconstruct: birth_year_i ≈ years_num[0] - ages[i,0]
                    col = (years_num[0] - ages[:, 0]).astype(float)
                else:
                    # No original available → assume already normalized
                    return x_orig
            mu, sd = (
                float(np.mean(col)),
                float(np.std(col) if np.std(col) != 0 else 1.0),
            )
            return (x_orig - mu) / sd
        # Default: assume already normalized
        return x_orig

    to_norm = to_norm_fn if to_norm_fn is not None else default_to_norm

    if grid_orig is None:
        if var == "birth_year_norm":
            if _has_col(df, "birth_year"):
                raw_vals = _col_to_numpy(df, "birth_year").astype(float)
            else:
                years_num = _infer_years_num(df)
                if (
                    years_num is not None
                    and ages is not None
                    and ages.ndim == 2
                    and ages.shape[1] >= 1
                ):
                    raw_vals = (years_num[0] - ages[:, 0]).astype(float)
                else:
                    # Fallback: vary directly on normalized scale
                    vals = x_pi_data[:, j]
                    grid_orig = np.linspace(
                        np.percentile(vals, 1), np.percentile(vals, 99), 41
                    )
                    # Mark that this is normalized scale
                    xlabel = var
                    use_norm_scale = True
                    raw_vals = None
            if grid_orig is None:
                lo, hi = (
                    float(np.percentile(raw_vals, 1)),
                    float(np.percentile(raw_vals, 99)),
                )
                grid_orig = np.linspace(lo, hi, 41)
                use_norm_scale = False
                xlabel = "birth_year"
        else:
            # Binary or numeric already
            vals = x_pi_data[:, j]
            uniq = np.unique(vals)
            grid_orig = (
                uniq
                if len(uniq) <= 6
                else np.linspace(np.percentile(vals, 1), np.percentile(vals, 99), 41)
            )
            xlabel = var
            use_norm_scale = True
    else:
        # User provided; decide label
        xlabel = "birth_year" if var == "birth_year_norm" else var
        use_norm_scale = var != "birth_year_norm"

    grid_orig = np.asarray(grid_orig)
    if grid_orig.dtype.kind in ("U", "S", "O"):
        raise ValueError("Valori categorici passati per una variabile non fattoriale.")

    # Compute π along grid
    pi_grid = np.zeros((len(grid_orig), K), dtype=float)
    for g, v_orig in enumerate(grid_orig):
        x = x_ref.copy()
        # map original -> normalized when needed
        if var == "birth_year_norm" and not use_norm_scale:
            v_norm = float(to_norm(var, np.array([v_orig]))[0])
        else:
            v_norm = float(v_orig)  # already normalized
        x[j] = v_norm
        logits = log_pi0 + (W_pi @ x)
        pi_grid[g] = _softmax_vec(logits)

    # Plot
    plt.figure(figsize=(4, 3))
    for k, col in enumerate(state_cols):
        plt.plot(grid_orig, pi_grid[:, k], color=col, label=f"state {k}")
    plt.xlabel(xlabel)
    plt.ylabel("π_k")
    ttl_x = xlabel if xlabel != var else var.replace("_norm", "")
    plt.title(f"{title_prefix}{ttl_x}")
    plt.grid(ls=":", alpha=0.5)
    plt.tight_layout()
    plt.legend()
    plt.show()

    return plt


# ==============================================================
# 6) λ_k(x_em) del GLM emissioni vs covariata (fattori/continui)
# ==============================================================


def plot_lambda_em_vs_cov(
    var_em,
    beta_em,  # (K, 1 + C_em)
    cov_names_em,  # lunghezza = C_em
    x_em_data=None,  # (N, T, C_em) o (N, C_em) per media di riferimento
    x_em_ref=None,  # (C_em,) opzionale
    factor_specs_em=None,  # dict: { "age_years": {"levels":[...], "ref":"<25"} , ... }
    grid_orig=None,
    state_cols=None,
    title_prefix="Emission GLM λ_k vs ",
):
    """
    Plotta la λ_k = exp(b0_k + x_em · B_k) variando una sola covariata di emissione.
    Gestisce sia continui che fattori (dummies con riferimento).
    """
    K, P = beta_em.shape
    if state_cols is None:
        state_cols = ["#e41a1c", "#377eb8", "#4daf4a"][:K]

    # reference su x_em
    if x_em_ref is None:
        if x_em_data is None:
            if "cov_emiss_torch" in globals():
                X = cov_emiss_torch.cpu().numpy()  # noqa: F821
                x_em_ref = X.mean(axis=(0, 1))  # media su N,T
            else:
                raise ValueError(
                    "Serve x_em_data o x_em_ref per definire il vettore di riferimento."
                )
        else:
            X = x_em_data
            if X.ndim == 3:
                x_em_ref = X.mean(axis=(0, 1))
            elif X.ndim == 2:
                x_em_ref = X.mean(axis=0)
            else:
                raise ValueError("x_em_data deve essere (N,T,C_em) o (N,C_em).")

    b0 = beta_em[:, 0]  # (K,)
    B = beta_em[:, 1:]  # (K, C_em)

    # fattore?
    is_factor = factor_specs_em is not None and var_em in factor_specs_em

    if is_factor:
        levels = factor_specs_em[var_em]["levels"]
        ref = factor_specs_em[var_em]["ref"]
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
            ax.bar(
                x_pos + k * width - 0.4 + width * K / 2,
                lam[:, k],
                width=width,
                color=c,
                label=f"state {k}",
            )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(grid_levels, rotation=45, ha="right")
        ax.set_ylabel("λ_k")
        ax.set_title(f"{title_prefix}{var_em}")
        ax.grid(axis="y", ls=":", alpha=0.4)
        ax.legend()
        plt.tight_layout()
        plt.show()

    else:
        # continuo / binario singolo
        if var_em not in cov_names_em:
            raise ValueError(
                f"{var_em} non è in cov_names_em e non risulta specificato come fattore."
            )
        j = cov_names_em.index(var_em)
        # griglia ORIGINAL (niente z-score: emissioni GLM usano la scala del design)
        if grid_orig is None:
            # prova a inferire dal campione se disponibile
            if x_em_data is not None:
                X = x_em_data
                vals = X.reshape(-1, X.shape[-1])[:, j]
                uniq = np.unique(vals)
                grid_orig = (
                    uniq if len(uniq) <= 6 else np.linspace(vals.min(), vals.max(), 41)
                )
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
        plt.xlabel(var_em)
        plt.ylabel("λ_k")
        plt.title(f"{title_prefix}{var_em}")
        plt.grid(ls=":", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()


# ---------- helpers ---------------------------------------------------------
def softmax_vec(v: np.ndarray) -> np.ndarray:
    v = v - np.max(v)
    e = np.exp(v)
    return e / np.sum(e)


# def build_factor_cols(
#     cov_names: list[str],
#     factor_name: str,
#     levels: Iterable,
#     ref_level: Optional[str] = None,
# ):
#     """
#     Map factor levels to column indices in a full one-hot design.

#     Returns
#     -------
#     factor_map : dict
#         level -> column index (or None if 'ref_level' for reference coding)
#     all_dummy_idx : list[int]
#         all dummy column indices (excluding the reference if provided)
#     """
#     name_to_idx = {n: i for i, n in enumerate(cov_names)}
#     factor_map = {}
#     all_dummy_idx = []
#     for lev in levels:
#         col_name = f"{factor_name}[{lev}]"
#         if (ref_level is not None) and (lev == ref_level):
#             factor_map[lev] = None
#         else:
#             if col_name not in name_to_idx:
#                 raise ValueError(
#                     f"Missing dummy column '{col_name}' in cov_names for factor '{factor_name}'."
#                 )
#             idx = name_to_idx[col_name]
#             factor_map[lev] = idx
#             all_dummy_idx.append(idx)
#     return factor_map, all_dummy_idx


# def set_factor_level_in_vector(
#     x_vec: np.ndarray, factor_map: dict, all_dummy_idx: list[int], level
# ) -> None:
#     """Zero all factor dummies, then set the one for 'level' to 1 (if it has a column)."""
#     if all_dummy_idx:
#         x_vec[np.array(all_dummy_idx, dtype=int)] = 0.0
#     idx = factor_map.get(level, None)
#     if idx is not None:
#         x_vec[idx] = 1.0


# =====================================================================
# Transition probabilities vs covariata (ORIGINAL scale, fattori ok)
# =====================================================================
def plot_trans_vs_cov_orig(
    var: str,
    prev_state: int = 0,
    *,
    cov_names_A: list[str],
    x_A_data: np.ndarray,  # (N,T,C_A) per vettore di riferimento (media)
    W_A: np.ndarray,  # (K,K,C_A)
    log_A0: np.ndarray,  # (K,K)
    factor_specs_A: Optional[
        dict
    ] = None,  # {"age_years": {"levels": [...], "ref": None}}
    grid_orig: Optional[np.ndarray] = None,
    state_cols: Optional[list[str]] = None,
):
    K = W_A.shape[0]
    if state_cols is None:
        state_cols = _default_state_cols(K)
    if prev_state >= K:
        raise ValueError("prev_state out of range")

    # vettore di riferimento (media su N,T)
    x_ref = x_A_data.mean(axis=(0, 1)).copy()  # (C_A,)

    # Caso FATTORIALE
    is_factor = (factor_specs_A is not None) and (var in factor_specs_A)
    if is_factor:
        levels = factor_specs_A[var]["levels"]
        ref = factor_specs_A[var].get("ref", None)
        factor_map, all_dummy_idx = build_factor_cols(cov_names_A, var, levels, ref)

        mats = np.zeros((len(levels), K))
        for g, lev in enumerate(levels):
            x_vec = x_ref.copy()
            set_factor_level_in_vector(x_vec, factor_map, all_dummy_idx, lev)
            logits = log_A0[prev_state] + (W_A[prev_state] @ x_vec)  # (K,)
            mats[g] = softmax_vec(logits)

        # barplot raggruppato per j
        fig, ax = plt.subplots(figsize=(8, 3.2))
        x_pos = np.arange(len(levels))
        width = 0.8 / K
        for j, c in enumerate(state_cols):
            ax.bar(
                x_pos + j * width - 0.4 + width * K / 2,
                mats[:, j],
                width=width,
                color=c,
                label=f"{prev_state}→{j}",
            )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(levels, rotation=45, ha="right")
        ax.set_ylabel("transition prob.")
        ax.set_xlabel(var)
        ax.set_title(f"Transition from state {prev_state} vs {var}")
        ax.grid(axis="y", ls=":", alpha=0.4)
        ax.legend()
        plt.tight_layout()
        plt.show()
        return mats

    # Caso CONTINUO/BINARIO (var è una singola colonna di A)
    if var not in cov_names_A:
        raise ValueError(
            f"{var} non trovato in cov_names_A e non specificato come fattore."
        )
    idx = cov_names_A.index(var)

    # grid (ORIGINALE): se binaria → {0,1}
    if grid_orig is None:
        vals = x_A_data[..., idx].reshape(-1)
        uniq = np.unique(vals)
        grid_orig = (
            uniq
            if len(uniq) <= 6
            else np.linspace(np.percentile(vals, 1), np.percentile(vals, 99), 41)
        )

    mats = np.zeros((len(grid_orig), K))
    for g, v in enumerate(grid_orig):
        x_vec = x_ref.copy()
        x_vec[idx] = v
        logits = log_A0[prev_state] + (W_A[prev_state] @ x_vec)
        mats[g] = softmax_vec(logits)

    # line plot
    for j, c in enumerate(state_cols):
        plt.plot(grid_orig, mats[:, j], color=c, label=f"{prev_state}→{j}")
    plt.xlabel(var)
    plt.ylabel("transition prob.")
    plt.title(f"Transition from state {prev_state} vs {var}")
    plt.grid(ls=":")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return mats


# # =====================================================================
# # Emission λ_k vs covariata (ORIGINAL scale, fattori ok)
# # =====================================================================
# def plot_lambda_em_vs_cov(
#     var_em: str,
#     *,
#     beta_em: np.ndarray,  # (K, 1 + C_em)
#     cov_names_em: list[str],
#     x_em_data: np.ndarray,  # (N, T, C_em)
#     x_em_ref: Optional[np.ndarray] = None,  # (C_em,)
#     factor_specs_em: Optional[dict] = None,
#     grid_orig: Optional[np.ndarray] = None,
#     state_cols: Optional[list[str]] = None,
#     title_prefix: str = "Emission rate λ_k vs ",
# ):
#     K = beta_em.shape[0]
#     b0 = beta_em[:, 0]  # (K,)
#     B = beta_em[:, :]  # (K, C_em)
#     C_em = B.shape[1]
#     if x_em_ref is None:
#         x_em_ref = x_em_data.mean(axis=(0, 1))  # (C_em,)
#     if x_em_ref.shape[0] != C_em:
#         raise ValueError(f"x_em_ref must have length {C_em}")

#     if state_cols is None:
#         state_cols = _default_state_cols(K)

#     # Factor branch
#     if factor_specs_em is not None and var_em in factor_specs_em:
#         levels = factor_specs_em[var_em]["levels"]
#         ref = factor_specs_em[var_em].get("ref", None)
#         factor_map, dummy_idx = build_factor_cols(cov_names_em, var_em, levels, ref)

#         grid_levels = levels if grid_orig is None else list(grid_orig)
#         lam = np.zeros((len(grid_levels), K), dtype=float)

#         for g, lev in enumerate(grid_levels):
#             x = x_em_ref.copy()
#             if dummy_idx:
#                 x[np.array(dummy_idx, dtype=int)] = 0.0
#             idx = factor_map.get(lev, None)
#             if idx is not None:
#                 x[idx] = 1.0
#             lam[g] = np.exp(b0 + B @ x)

#         # Bar plot per state
#         fig, ax = plt.subplots(figsize=(7.0, 3.2))
#         x_pos = np.arange(len(grid_levels))
#         width = 0.8 / K
#         for k, col in enumerate(state_cols):
#             ax.bar(
#                 x_pos + (k - (K - 1) / 2) * width,
#                 lam[:, k],
#                 width=width,
#                 color=col,
#                 label=f"state {k}",
#             )
#         ax.set_xticks(x_pos)
#         ax.set_xticklabels(grid_levels, rotation=45, ha="right")
#         ax.set_ylabel("λ_k")
#         ax.set_title(f"{title_prefix}{var_em}")
#         ax.grid(axis="y", ls=":", alpha=0.4)
#         ax.legend()
#         plt.tight_layout()
#         plt.show()
#         return grid_levels, lam

#     # Continuous/binary branch
#     if var_em not in cov_names_em:
#         raise ValueError(f"{var_em} not in cov_names_em and not declared as factor.")

#     j = cov_names_em.index(var_em)
#     col = x_em_data[:, :, j].reshape(-1)
#     if grid_orig is None:
#         uniq = np.unique(col)
#         grid_orig = (
#             uniq
#             if len(uniq) <= 6
#             else np.linspace(np.percentile(col, 1), np.percentile(col, 99), 41)
#         )
#     grid_orig = np.asarray(grid_orig, dtype=float)

#     lam = np.zeros((len(grid_orig), K), dtype=float)
#     for g, v in enumerate(grid_orig):
#         x = x_em_ref.copy()
#         x[j] = v
#         lam[g] = np.exp(b0 + B @ x)

#     # Line plot
#     for k, colc in enumerate(state_cols):
#         plt.plot(grid_orig, lam[:, k], color=colc, label=f"state {k}")
#     plt.xlabel(var_em)
#     plt.ylabel("λ_k")
#     plt.title(f"{title_prefix}{var_em}")
#     plt.grid(ls=":", alpha=0.5)
#     plt.tight_layout()
#     plt.legend()
#     plt.show()

#     return grid_orig, lam


# =====================================================================
# Expected E[y0 | x] vs π-covariata (ORIG scale) con emissioni GLM
# =====================================================================
def expected_y_orig(
    var: str,
    cov_names_pi: Iterable[str],
    W_pi: np.ndarray,  # (K, C_pi)
    log_pi0: np.ndarray,  # (K,)
    *,
    # Emission parameters (choose one branch)
    rates: Optional[np.ndarray] = None,  # (K,)
    beta_em: Optional[np.ndarray] = None,  # (K, 1 + C_em)
    x_em_ref: Optional[np.ndarray] = None,  # (C_em,) required if beta_em is provided
    # Reference for π(x)
    x_pi_data: Optional[
        np.ndarray
    ] = None,  # (N, C_pi) normalized; used to build mean reference if x_pi_ref is None
    x_pi_ref: Optional[np.ndarray] = None,  # (C_pi,) normalized; overrides x_pi_data
    # Grid and helpers
    grid_orig: Optional[
        np.ndarray
    ] = None,  # values on ORIGINAL scale for the selected var
    to_norm_fn: Optional[
        Callable[[str, np.ndarray], np.ndarray]
    ] = None,  # original → normalized
    original_values_fn: Optional[
        Callable[[str], np.ndarray]
    ] = None,  # returns original values for var
    # Plotting
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    show: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    cov_names_pi = list(cov_names_pi)
    if var not in cov_names_pi:
        raise ValueError(f"{var} is not present in cov_names_pi.")

    K, C_pi = W_pi.shape
    if log_pi0.shape[0] != K:
        raise ValueError(
            f"log_pi0 must have shape (K,), got {log_pi0.shape} for K={K}."
        )

    # Emission: choose GLM if beta_em is provided; else use rates
    use_glm = beta_em is not None
    if use_glm:
        if x_em_ref is None:
            raise ValueError("x_em_ref is required when beta_em is provided.")
        if beta_em.shape[0] != K:
            raise ValueError(
                f"beta_em first dimension must be K={K}, got {beta_em.shape}."
            )
        b0 = beta_em[:, 0]  # (K,)
        B = beta_em[:, 1:]  # (K, C_em)
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
            raise ValueError(
                "Provide x_pi_ref or x_pi_data to build the reference for π."
            )
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

    # Convert original value → normalized using provided function (identity if None)
    def to_norm_value(var_name: str, x_orig) -> float:
        if to_norm_fn is None:
            return float(x_orig)  # assumes already normalized
        x_arr = np.asarray([x_orig], dtype=float)
        v = to_norm_fn(var_name, x_arr)
        return float(v[0]) if np.ndim(v) > 0 else float(v)

    # Compute expected values along the grid
    exp_vals = np.zeros(len(grid_orig), dtype=float)
    for g, v_orig in enumerate(grid_orig):
        x_vec = x_pi_ref.copy()
        if is_categorical:
            # Categorical var in π is expected to be dummy-coded in cov_names_pi.
            prefix = f"{var}["
            all_dummy_idx = [
                k
                for k, name in enumerate(cov_names_pi)
                if name.startswith(prefix) and name.endswith("]")
            ]
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


def plot_donor_gg(
    idx,
    obs_torch,
    paths,
    years,
    expected_next=None,  # float, predicted expected donations for next year
    y_true_next=None,  # int, actual donations next year (if available)
    next_year=None,  # int, defaults to years[-1] + 1
    state_cols=None,  # list of colors for states
    state_to_label=None,  # list of states names e.g. donatore frequente
    predicted_state_next=None,
    colors=None,
    title_prefix="Donor",
    y_max=4,
):
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
        Latent states (e.g., Viterbi), ints in 0..K-1 (or any integer labels).
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
    x = (
        obs_torch[idx].detach().cpu().numpy()
        if hasattr(obs_torch, "detach")
        else np.asarray(obs_torch[idx])
    )
    z = (
        paths[idx].detach().cpu().numpy()
        if hasattr(paths, "detach")
        else np.asarray(paths[idx], dtype=int)
    )
    years = np.asarray(years, dtype=int)

    # --- Basic checks ---
    T = len(x)
    if len(years) != T:
        raise ValueError(
            "Length of 'years' must match the time dimension T for the donor."
        )
    if z.shape[0] != T:
        raise ValueError(
            "Length of 'paths[idx]' must match the time dimension T for the donor."
        )

    # --- Unique states and labels (robusto anche se gli stati non sono 0..K-1) ---
    if predicted_state_next is None:
        unique_states = np.unique(z)
    else:
        unique_states = np.unique(np.append(z, predicted_state_next))
    # state_to_label = {int(s): f"State {int(s)}" for s in unique_states}
    if state_to_label is None:
        state_to_label = {int(s): f"State {int(s)}" for s in unique_states}
    elif len(state_to_label) < len(unique_states):
        raise ValueError(
            "Length of state_to_label must match the number of latent states."
        )

    z_labs = [state_to_label[int(s)] for s in z]
    state_labels = [state_to_label[int(s)] for s in unique_states]

    # set colors
    if colors is None:
        label_to_color = {
            state_to_label[k]: STATE_PALETTE.get(int(k), "#999999")
            for k in unique_states
        }
    else:
        # label_to_color = {state_to_label[k]: c for k, c in zip(unique_states, colors)}
        label_to_color = colors

    # --- Observed data frame ---
    df_obs = pd.DataFrame({"year": years, "donations": x, "state": z_labs})

    # --- Prediction annotations ---
    if next_year is None:
        next_year = int(years[-1] + 1)

    rows_pred = []
    if expected_next is not None:
        rows_pred.append(
            {"year": next_year, "donations": expected_next, "kind": "Predicted"}
        )
    if y_true_next is not None:
        rows_pred.append(
            {"year": next_year, "donations": y_true_next, "kind": "Actual"}
        )
    df_pred = (
        pd.DataFrame(rows_pred)
        if rows_pred
        else pd.DataFrame(columns=["year", "donations", "kind"])
    )

    # --- Axis limits and breaks ---
    y_low, y_high = -0.5, float(y_max) + 0.5
    x_breaks = list(np.unique(np.concatenate([years, np.array([next_year])])))

    # --- Base plot ---
    p = (
        pn.ggplot(df_obs, aes("year", "donations"))
        + pn.geom_step(direction="mid", color="black", alpha=0.35)
        + pn.geom_point(aes(color="state"), size=2.5)
        + pn.scale_color_manual(
            values=label_to_color, breaks=state_labels, labels=state_labels
        )
        + pn.scale_x_continuous(breaks=x_breaks, minor_breaks=None)
        + pn.scale_y_continuous(
            limits=(y_low, y_high),
            breaks=list(range(0, int(y_max) + 1)),
            minor_breaks=None,
        )
        # + pn.labs(title=f"{title_prefix} {idx}", x="", y="# donations")
        + pn.labs(x="")
        + pn.theme_minimal()
        + pn.theme(
            axis_text_x=element_text(rotation=45, ha="right"),
            # legend_title=element_text(size=10),
            legend_title=element_blank(),
            legend_text=element_text(size=9),
            legend_position="bottom",
            axis_ticks_minor_x=element_blank(),
            axis_ticks_minor_y=element_blank(),
            axis_ticks_major=element_blank(),
            # axis_ticks_minor_y=element_blank(),
            plot_title=element_text(weight="bold"),
        )
        + pn.guides(color=guide_legend(title=""))
    )

    # --- Next-year markers and labels ---
    if not df_pred.empty:
        p = p + pn.geom_vline(xintercept=next_year, linetype="dashed", alpha=0.6)

        if (df_pred["kind"] == "Predicted").any():
            if predicted_state_next is not None:
                pred_label = state_to_label.get(predicted_state_next, None)
                pred_color = label_to_color.get(pred_label, "black")
                pred_shape = "^"  # triangolo
            else:
                pred_color = "black"
                pred_shape = "^"

            p = (
                p
                + pn.geom_point(
                    mapping=aes("year", "donations"),
                    data=df_pred[df_pred["kind"] == "Predicted"],
                    color=pred_color,
                    size=3.5,
                    shape=pred_shape,
                    show_legend=False,
                )
                + pn.geom_text(
                    mapping=aes("year", "donations"),
                    data=df_pred[df_pred["kind"] == "Predicted"],
                    label="pred",
                    nudge_y=0.25,
                    size=8,
                    color=pred_color,
                    show_legend=False,
                )
            )

        if (df_pred["kind"] == "Actual").any():
            p = (
                p
                + pn.geom_point(
                    mapping=aes("year", "donations"),
                    data=df_pred[df_pred["kind"] == "Actual"],
                    color="#d62728",
                    size=3.5,
                    shape="x",
                    show_legend=False,
                )
                + pn.geom_text(
                    mapping=aes("year", "donations"),
                    data=df_pred[df_pred["kind"] == "Actual"],
                    label="actual",
                    nudge_y=0.25,
                    size=8,
                    color="#d62728",
                    show_legend=False,
                )
            )
    return p


# per la dashboard, non utilizza più il df ma calcola da 0 tutto
def plot_donor_simple(
    years,
    donations,
    states=None,  # opzionale: stati latenti o None
    expected_next=None,  # opzionale: float
    y_true_next=None,  # opzionale: int
    next_year=None,  # opzionale: default years[-1] + 1
    state_labels=None,  # opzionale: dict {state_value: "label"}
    colors=None,  # opzionale: dict {label: "#hex"}
    title=None,
    y_max=4,
    show_legend=True,
):
    import numpy as np
    import pandas as pd
    import plotnine as pn

    years = np.asarray(years, dtype=int)
    donations = np.asarray(donations, dtype=float)

    if next_year is None:
        next_year = int(years[-1] + 1)

    df_obs = pd.DataFrame({"year": years, "donations": donations})

    # Stati opzionali
    if states is not None:
        states = np.asarray(states)
        uniq = list(pd.unique(states))
        if state_labels is None:
            state_labels = {int(s): f"State {int(s)}" for s in uniq}
        df_obs["state"] = [state_labels.get(int(s), str(s)) for s in states]
        # Colori
        if colors is None:
            # Prendi i label in order e popola STATE_PALETTE (loop su uniq)
            label_order = [state_labels[int(s)] for s in uniq]
            palette = STATE_PALETTE
            colors = {
                label: palette.get(int(s), "#888")
                for s, label in zip(uniq, label_order)
            }
    else:
        df_obs["state"] = "Observed"

    # Prossimo anno
    rows_pred = []
    if expected_next is not None:
        rows_pred.append(
            {"year": next_year, "donations": float(expected_next), "kind": "Pred"}
        )
    if y_true_next is not None:
        rows_pred.append(
            {"year": next_year, "donations": float(y_true_next), "kind": "Actual"}
        )
    df_pred = pd.DataFrame(rows_pred)

    # Assi
    y_low, y_high = -0.5, float(y_max) + 0.5
    x_breaks = sorted(pd.unique(np.concatenate([years, np.array([next_year])])))

    # Base plot
    p = pn.ggplot(df_obs, pn.aes("year", "donations")) + pn.geom_step(
        direction="mid", color="black", alpha=0.35
    )

    if states is not None:
        p = p + pn.geom_point(pn.aes(color="state"), size=2.5)
        p = p + pn.scale_color_manual(values=colors)
        if not show_legend:
            p = p + pn.guides(color=None)
    else:
        p = p + pn.geom_point(color="black", size=2.5)
        p = p + pn.guides(color=None)

    if not df_pred.empty:
        p = p + pn.geom_vline(xintercept=next_year, linetype="dashed", alpha=0.6)
        if (df_pred["kind"] == "Pred").any():
            p = (
                p
                + pn.geom_point(
                    pn.aes("year", "donations"),
                    data=df_pred[df_pred["kind"] == "Pred"],
                    color="black",
                    size=3.5,
                    shape="^",
                    show_legend=False,
                )
                + pn.geom_text(
                    pn.aes("year", "donations"),
                    data=df_pred[df_pred["kind"] == "Pred"],
                    label="pred",
                    nudge_y=0.25,
                    size=8,
                    color="black",
                    show_legend=False,
                )
            )
        if (df_pred["kind"] == "Actual").any():
            p = (
                p
                + pn.geom_point(
                    pn.aes("year", "donations"),
                    data=df_pred[df_pred["kind"] == "Actual"],
                    color="#d62728",
                    size=3.5,
                    shape="x",
                    show_legend=False,
                )
                + pn.geom_text(
                    pn.aes("year", "donations"),
                    data=df_pred[df_pred["kind"] == "Actual"],
                    label="actual",
                    nudge_y=0.25,
                    size=8,
                    color="#d62728",
                    show_legend=False,
                )
            )

    p = (
        p
        + pn.scale_x_continuous(breaks=x_breaks)
        + pn.scale_y_continuous(
            limits=(y_low, y_high), breaks=list(range(0, int(y_max) + 1))
        )
        + pn.labs(title=title or "", x="", y="# donazioni")
        + pn.theme_minimal()
    )
    return p


def plot_accuracy(glm_acc, hmm_acc, show: bool = True):
    """
    Plotta le accuracy di GLM e HMM.

    Args:
        glm_acc (float): Accuracy del GLM (in percento o frazione).
        hmm_acc (float): Accuracy dell'HMM (in percento o frazione).
    """
    # Se i valori sono frazioni (0-1), li porto in percento
    if glm_acc <= 1 and hmm_acc <= 1:
        glm_acc *= 100
        hmm_acc *= 100

    methods = ["GLM", "HMM"]
    values = [glm_acc, hmm_acc]

    plt.figure(figsize=(5, 4))
    bars = plt.bar(methods, values, color=palette, edgecolor="black")

    # Aggiungo etichette sopra le barre
    for bar, val in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.5,
            f"{val:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.ylabel("Accuracy (%)")
    plt.title("Confronto Accuracy: GLM vs HMM-GLM")
    plt.ylim(0, max(values) + 10)
    if show:
        plt.show()

    return plt


def plot_error_distribution(
    errors, title, ax, theme_minimal: bool = False, with_title: bool = True
):
    full_range = pd.Series(range(-4, 5), name="error")
    freq = errors.value_counts(normalize=True).sort_index() * 100
    freq_df = freq.reset_index()
    freq_df.columns = ["error", "percent"]
    freq_df = full_range.to_frame().merge(freq_df, on="error", how="left").fillna(0)
    pivot_df = freq_df.pivot_table(index="error", values="percent")
    sns.heatmap(
        pivot_df.T,
        annot=True,
        fmt=".2f",
        cmap=COEFF_CMAP,
        cbar=False if theme_minimal else True,
        cbar_kws={"label": "Percentuale (%)"} if not theme_minimal else None,
        linewidths=0.5,
        linecolor="gray",
        ax=ax,
        vmin=0,
        vmax=freq_df["percent"].max(),  # uniform color scale
    )
    if theme_minimal:
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.set_yticks([])
    else:
        ax.set_xlabel("Errore arrotondato")
        ax.set_ylabel("")
        ax.set_yticks([])
        ax.set_xticklabels(range(-4, 5))
    if with_title:
        ax.set_title(f"Distribuzione percentuale errori {title}")
    return ax
