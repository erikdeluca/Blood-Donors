import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson
import pandas as pd
from plotnine import (
    ggplot,
    aes,
    geom_col,
    geom_tile,
    geom_text,
    position_dodge,
    scale_fill_brewer,
    scale_fill_gradient,
    theme_bw,
    theme,
    element_text,
    labs,
    scale_x_discrete,
    scale_y_discrete,
)


def plot_hmm_params(
    transitions,
    initial_probs,
    emissions,
    state_names=None,
    emission_names=None,
    nrows=1,
    ncols=3,
    figsize_width=15,
    figsize_height=3,
):
    """
    Plotta in una riga:
    - Matrice di transizione [S, S]
    - Prob iniziali [S]
    - Matrice emissioni [S, K]
    """
    S = len(initial_probs)
    K = emissions.shape[1]
    if state_names is None:
        state_names = [f"State {i}" for i in range(S)]
    if emission_names is None:
        emission_names = [str(i) for i in range(K)]

    fig, axs = plt.subplots(nrows, ncols, figsize=(figsize_width, figsize_height))

    # Initial probabilities
    axs[0].bar(np.arange(S), initial_probs, color="royalblue")
    axs[0].set_title("Initial State Probabilities")
    axs[0].set_xlabel("State")
    axs[0].set_ylabel("Probability")
    axs[0].set_xticks(np.arange(S))
    axs[0].set_xticklabels(state_names)
    axs[0].grid(axis="y", alpha=0.3)

    # Transition matrix
    sns.heatmap(
        transitions,
        annot=True,
        fmt=".2f",
        cmap="Greens",
        xticklabels=state_names,
        yticklabels=state_names,
        ax=axs[1],
        cbar=False,
    )
    axs[1].set_title("Transition Probabilities")
    axs[1].set_xlabel("Next State")
    axs[1].set_ylabel("Current State")

    # Emission probabilities/matrix
    sns.heatmap(
        emissions,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=emission_names,
        yticklabels=state_names,
        ax=axs[2],
        cbar=False,
    )
    axs[2].set_title("Emission Probabilities")
    axs[2].set_xlabel("Donations in a Year")
    axs[2].set_ylabel("Latent State")

    plt.tight_layout()
    plt.show()


def build_emission_matrix_truncated_poisson(rates, max_k=4):
    S = len(rates)
    K = max_k + 1  # da 0 a max_k incluso
    emissions = np.zeros((S, K))
    for s in range(S):
        for k in range(max_k):
            emissions[s, k] = poisson.pmf(k, rates[s])
        # L'ultimo raccoglie la coda (tutto >= max_k)
        emissions[s, max_k] = 1 - poisson.cdf(max_k - 1, rates[s])
    return emissions


def build_emission_matrix_trunc_poisson(rates, max_k=4):
    S, G = len(rates), max_k + 1
    M = np.zeros((S, G))
    for s in range(S):
        for k in range(max_k):
            M[s, k] = poisson.pmf(k, rates[s])
        M[s, max_k] = 1 - poisson.cdf(max_k - 1, rates[s])
    return M


def plot_hmm_params_with_coeffs(
    transitions,
    initial_probs,
    beta_em,
    state_names=None,
    coeff_names=None,
    include_intercept=True,
    intercept_name="Intercept",
):
    """
    Plot three panels using plotnine:
      1) Initial state probabilities [S]
      2) Transition matrix heatmap [S x S]
      3) Emission coefficients by state (barplot; y=value, x=coefficient name, color=state)

    Parameters
    ----------
    transitions : array-like (S, S)
        Transition probability matrix (rows sum to 1).
    initial_probs : array-like (S,)
        Initial state probabilities.
    beta_em : array-like (S, C) or (S, C+1 if include_intercept=True)
        Emission GLM coefficients per state. If include_intercept=True, the first column is intercept.
    state_names : list[str], optional
        Names of states, length S.
    coeff_names : list[str], optional
        Names of coefficients (excluding intercept), length C.
    include_intercept : bool, default True
        Whether the first column of beta_em is an intercept.
    intercept_name : str, default 'Intercept'
        Name to use for the intercept coefficient.

    Returns
    -------
    dict
        Dictionary with plotnine objects: {'init': p_init, 'trans': p_trans, 'coeffs': p_coef}
    """
    transitions = np.asarray(transitions)
    initial_probs = np.asarray(initial_probs)
    beta_em = np.asarray(beta_em)

    S = initial_probs.shape[0]
    assert transitions.shape == (S, S), "transitions must be (S, S)"
    assert (
        beta_em.shape[0] == S
    ), "beta_em first dimension must match number of states S"

    # Build state names if not provided
    if state_names is None:
        state_names = [f"State {i}" for i in range(S)]

    # Build coefficient names
    if include_intercept:
        C = beta_em.shape[1] - 1
        if coeff_names is None:
            coeff_names = [f"x{i}" for i in range(C)]
        coef_full_names = [intercept_name] + coeff_names
        beta_plot = beta_em
    else:
        C = beta_em.shape[1]
        if coeff_names is None:
            coeff_names = [f"x{i}" for i in range(C)]
        coef_full_names = coeff_names
        beta_plot = beta_em

    # Panel 1: Initial probabilities (bar plot)
    df_init = pd.DataFrame({"state": state_names, "prob": initial_probs})
    p_init = (
        ggplot(df_init, aes(x="state", y="prob", fill="state"))
        + geom_col()
        + scale_fill_brewer(type="qual", palette="Set1")
        + theme_bw()
        + theme(axis_text_x=element_text(rotation=0), legend_position="none")
        + labs(title="Initial State Probabilities", x="State", y="Probability")
    )

    # Panel 2: Transition matrix (heatmap with labels)
    rows, cols, vals = [], [], []
    for i in range(S):
        for j in range(S):
            rows.append(state_names[i])
            cols.append(state_names[j])
            vals.append(float(transitions[i, j]))
    df_trans = pd.DataFrame({"from": rows, "to": cols, "prob": vals})
    df_trans["label"] = df_trans["prob"].round(2).astype(str)

    p_trans = (
        ggplot(df_trans, aes(x="to", y="from", fill="prob"))
        + geom_tile()
        + geom_text(aes(label="label"), size=9)
        + scale_fill_gradient(low="#e5f5e0", high="#238b45")
        + theme_bw()
        + theme(axis_text_x=element_text(rotation=0))
        + labs(
            title="Transition Probabilities",
            x="Next State",
            y="Current State",
            fill="Prob",
        )
        + scale_x_discrete(limits=state_names)
        + scale_y_discrete(limits=state_names)
    )

    # Panel 3: Emission coefficients by state (grouped bar plot)
    data_coef = []
    for s in range(S):
        for c_idx, name in enumerate(coef_full_names):
            data_coef.append(
                {
                    "state": state_names[s],
                    "coef_name": name,
                    "value": float(beta_plot[s, c_idx]),
                }
            )
    df_coef = pd.DataFrame(data_coef)

    p_coef = (
        ggplot(df_coef, aes(x="coef_name", y="value", fill="state"))
        + geom_col(position=position_dodge(width=0.8))
        + scale_fill_brewer(type="qual", palette="Set1")
        + theme_bw()
        + theme(axis_text_x=element_text(rotation=90, ha="right"))
        + labs(
            title="Emission Coefficients by State",
            x="Coefficient",
            y="Value",
            fill="State",
        )
    )

    return {"init": p_init, "trans": p_trans, "coeffs": p_coef}
