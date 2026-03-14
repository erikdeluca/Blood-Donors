import numpy as np
import pandas as pd
from pyprojroot import here
import sys
from sklearn.metrics import roc_curve, roc_auc_score


import config as C

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


# Custom colormaps from theme colors
TRANS_CMAP = LinearSegmentedColormap.from_list("trans_cmap", ["#E5F0E7", "#4A8255"])
EMISS_CMAP = LinearSegmentedColormap.from_list("emiss_cmap", ["#f4ece2", "#8c1c13ff"])
COEFF_CMAP = LinearSegmentedColormap.from_list(
    "emiss_cmap", ["#8c1c13ff", "#f4ece2", "#86ba90"]
)


# ---------- Subplots ----------
def plot_initial_probs(ax, initial_probs, state_names, colors: dict = C.STATE_PALETTE):
    """
    Plot the initial state probabilities as a bar chart.
    """
    K = len(initial_probs)
    ax.bar(np.arange(K), initial_probs, color=colors.values())
    ax.set_title("Initial State Probabilities")
    ax.set_xlabel("State")
    ax.set_ylabel("Probability")
    ax.set_xticks(np.arange(K))
    ax.set_xticklabels(state_names)
    ax.set_ylim(0, max(1.0, initial_probs.max() * 1.1))
    ax.grid(axis="y", alpha=0.3)


def plot_transition_matrix(ax, transitions, state_names, colors=TRANS_CMAP, annot=True):
    """
    Plot the transition probability matrix as a heatmap using custom theme colors.
    """
    sns.heatmap(
        transitions,
        ax=ax,
        cmap=colors,
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


def plot_emission_coeffs(
    ax, beta_em, state_names, coeff_names, colors: dict = C.STATE_PALETTE
):
    """
    Plot emission GLM coefficients for each state as grouped bar chart.
    """
    K, C = beta_em.shape
    x = np.arange(C)
    width = 0.8 / K

    for k_idx, s_name in enumerate(state_names):
        vals = beta_em[k_idx, :]
        x_positions = x - 0.4 + width * (k_idx + 0.5)
        ax.bar(x_positions, vals, width=width, color=colors[k_idx], label=s_name)

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
    transitions: np.ndarray,
    initial_probs: np.ndarray,
    beta_em: np.ndarray,
    state_names=None,
    coeff_names=None,
    figsize=(16, 4),
    annot_transitions=True,
    show=True,
):
    """
    Visual summary of HMM parameters with emission GLM coefficients.
    """

    K = initial_probs.shape[0]

    if state_names is None:
        state_names = [f"State {i}" for i in range(K)]
    else:
        state_names = list(state_names)
    if coeff_names is None:
        coeff_names = [f"x{i}" for i in range(beta_em.shape[1])]

    fig, axs = plt.subplots(1, 3, figsize=figsize)

    # Call subplots
    plot_initial_probs(axs[0], initial_probs, state_names, C.STATE_PALETTE)
    plot_transition_matrix(axs[1], transitions, state_names, annot=annot_transitions)
    plot_emission_coeffs(axs[2], beta_em, state_names, coeff_names, C.STATE_PALETTE)

    plt.tight_layout()
    if show:
        plt.show()
        return
    else:
        return {
            "fig": fig,
            "axs": axs,
            "data": {
                "init": pd.DataFrame({"state": state_names, "prob": initial_probs}),
                "trans": pd.DataFrame(
                    {
                        "from": np.repeat(state_names, K),
                        "to": np.tile(state_names, K),
                        "prob": transitions.flatten(),
                    }
                ),
                "coeffs": pd.DataFrame(beta_em, columns=coeff_names, index=state_names)
                .rename_axis("state")
                .reset_index(),
            },
        }


def plot_one_gg(
    idx: int,
    obs: np.ndarray,
    paths: np.ndarray,
    years: np.ndarray,
    title_prefix="Donor",
):
    """
    plot the donation of a donor across its time serie.
    """

    T = len(obs[idx])
    if len(years) != T:
        raise ValueError("years length must match T for the selected donor.")

    plot_data = pd.DataFrame(
        {
            "t": np.arange(T),
            "year": years,
            "donations": obs[idx],
            "state": [C.STATE_NAMES[s] for s in paths[idx]],
        }
    )

    p = (
        pn.ggplot(plot_data, pn.aes("t", "donations", color="factor(state)"))
        + pn.geom_step(direction="mid", color="black", alpha=0.35)
        + pn.geom_point(pn.aes(color="state"), size=2.6)
        + pn.scale_color_manual(
            values=list(C.STATE_PALETTE.values()),
            breaks=list(C.STATE_NAMES.values()),
            name="State",
        )
        + pn.scale_x_continuous(breaks=list(range(T)), labels=[str(y) for y in years])
        + pn.scale_y_continuous(
            limits=(-0.5, float(C.MAX_N_DONATIONS) + 0.5),
            breaks=list(range(0, int(C.MAX_N_DONATIONS) + 1)),
        )
        + pn.labs(title=f"{title_prefix} {idx}", x="year", y="# donations")
        + pn.theme_minimal()
        + pn.theme(
            axis_text_x=pn.element_text(rotation=0, ha="right"),
            legend_title=pn.element_text(size=10),
            legend_text=pn.element_text(size=9),
            legend_position="bottom",
            plot_title=pn.element_text(weight="bold"),
            # mimic dotted y-grid only
            panel_grid_major_y=pn.element_line(linetype="dotted", alpha=0.4),
            panel_grid_major_x=pn.element_blank(),
            panel_grid_minor=pn.element_blank(),
        )
        + pn.guides(color=pn.guide_legend(title="latent state"))
    )
    return p


def plot_W_pi_heat(W_pi, cov_names_pi=None, title=r"$W_{\pi}$ - slopes on log $\pi$"):
    K, num_covs = W_pi.shape
    if not cov_names_pi or len(cov_names_pi) != num_covs:
        cov_names_pi = [f"cov_{i}" for i in range(num_covs)]

    plt.figure(figsize=(max(3, 0.9 * num_covs), max(2.8, 0.5 * K + 1)))
    sns.heatmap(
        W_pi,
        annot=True,
        fmt=".2f",
        xticklabels=cov_names_pi,
        yticklabels=[name.replace(" ", "\n") for name in list(C.STATE_NAMES.values())],
        cmap=COEFF_CMAP,
        center=0,
    )
    plt.title(title)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()

    return plt


def plot_W_A_heat(W_A, cov_names_A=None, title=r"$W_A$ - transition slopes"):
    K, _, num_covs = W_A.shape
    if not cov_names_A or len(cov_names_A) != num_covs:
        cov_names_A = [f"cov_{i}" for i in range(num_covs)]

    fig, axes = plt.subplots(
        K,
        K,
        figsize=(max(2.0, 0.55 * num_covs) * K, max(2.0, 0.8) * K),
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
            ax.set_title(rf"${i} \rightarrow {j}$", fontsize=8)
            ax.set_xticklabels(
                ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor"
            )

    plt.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.show()

    return plt


def plot_beta_em_heat(
    beta_em, cov_names_em=None, title=r"$\beta_{em}$ - GLM emission coefficients"
):
    K, num_covs = beta_em.shape
    if not cov_names_em or len(cov_names_em) != num_covs:
        cov_names_em = [f"em_{i}" for i in range(num_covs)]

    plt.figure(figsize=(max(3, 0.7 * num_covs), max(2.8, 0.5 * K + 1)))
    sns.heatmap(
        beta_em,
        annot=True,
        fmt=".2f",
        xticklabels=cov_names_em,
        yticklabels=[name.replace(" ", "\n") for name in list(C.STATE_NAMES.values())],
        cmap=COEFF_CMAP,
        center=0,
    )
    plt.title(title)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()


def softmax_row(v):
    e = np.exp(v - np.max(v, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def plot_pi_vs_cov_orig(
    var: str,
    cov_names_pi: dict,
    W_pi: np.ndarray,
    pi_base: np.ndarray,
    x_pi_data: np.ndarray,
    mean_norm: np.float64 = None,
    sd_norm: np.float64 = None,
    axes_label: list = None,
    title_prefix: str = r"$\pi_k(x)$ vs ",
):
    # Add a minimum value to avoid log0
    log_pi0 = np.log(np.clip(pi_base, 1e-30, None))
    K, C_pi = W_pi.shape

    # Reference data
    x_ref = x_pi_data.mean(axis=0)

    # Index column of the variable to analyze
    j = list(cov_names_pi.keys()).index(var)

    # Make a grid with the possible values
    grid = np.unique(x_pi_data[:, j])

    pi_grid = np.zeros((len(grid), K), dtype=float)
    for g, val in enumerate(grid):
        x = x_ref.copy()
        x[j] = val
        logits_grid = log_pi0 + (x @ W_pi.T)
        pi_grid[g] = softmax_row(logits_grid)

    # Calculate the original values for the standardized variables
    if mean_norm is not None and sd_norm is not None:
        grid = grid * sd_norm + mean_norm

    plt.figure(figsize=(5, 3.5))

    # Switch between categorical and continuous variables
    if len(grid) <= 3:
        # Stacked horizontal bar chart
        y_pos = np.arange(len(grid))

        if axes_label is None:
            axes_label = [str(int(g)) for g in grid]

        left_starts = np.zeros(len(grid))
        for k in range(K):
            plt.barh(
                y_pos,
                pi_grid[:, k],
                left=left_starts,
                color=C.STATE_PALETTE[k],
                label=C.STATE_NAMES[k],
                edgecolor="white",
                height=0.6,
            )
            left_starts += pi_grid[:, k]

        plt.yticks(y_pos, axes_label)
        plt.xlabel(r"Initial Probability ($\pi_k$)")
        plt.ylabel(cov_names_pi[var])
        plt.xlim(0, 1)

    else:
        # Line chart for continuous variables
        for k in range(K):
            plt.plot(
                grid,
                pi_grid[:, k],
                color=C.STATE_PALETTE[k],
                label=C.STATE_NAMES[k],
                lw=2,
            )
        plt.grid(ls=":", alpha=0.5)
        plt.xlabel(cov_names_pi[var])
        plt.ylabel(r"$\pi_k$")

    plt.title(f"{title_prefix}{cov_names_pi[var]}")
    plt.tight_layout()
    # Move legend outside to avoid covering bars
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.show()

    return plt


def plot_lambda_em_vs_cov_categorical(
    vars_list: list,
    cov_names_em: dict,
    beta_em: np.ndarray,
    x_em_data: np.ndarray,
    axes_label: list = None,
    title_prefix: str = r"Expected Donations ($\lambda_k$) vs ",
    var_title: str = "Age Groups",  # Titolo custom per il grafico
):
    """
    Barchart grouped for categorical variables
    """
    K, C_em = beta_em.shape
    x_ref = x_em_data.mean(axis=(0, 1))

    keys_list = list(cov_names_em.keys())
    dummy_indices = [keys_list.index(v) for v in vars_list]

    num_levels = 1 + len(vars_list)
    lam_grid = np.zeros((num_levels, K))

    # Baseline
    x = x_ref.copy()
    x[dummy_indices] = 0.0
    lam_grid[0] = np.exp(beta_em @ x)

    # Other levels
    for i, idx in enumerate(dummy_indices):
        x = x_ref.copy()
        x[dummy_indices] = 0.0
        x[idx] = 1.0
        lam_grid[i + 1] = np.exp(beta_em @ x)

    # Labels
    if axes_label is None:
        axes_label = ["Baseline"] + [cov_names_em[v] for v in vars_list]

    plt.figure(figsize=(9, 4))
    x_pos = np.arange(num_levels)
    width = 0.8 / K

    for k in range(K):
        offset = (k - (K - 1) / 2) * width
        plt.bar(
            x_pos + offset,
            lam_grid[:, k],
            width=width,
            color=C.STATE_PALETTE[k],
            label=C.STATE_NAMES[k],
            edgecolor="white",
        )

    plt.xticks(x_pos, axes_label, rotation=30, ha="right")
    plt.xlabel("Categorical Levels")
    plt.ylabel(r"Expected Donations ($\lambda_k$)")
    plt.title(f"{title_prefix}{var_title}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

    return plt


def plot_lambda_em_vs_cov_continuous(
    var: str,
    cov_names_em: dict,
    beta_em: np.ndarray,
    x_em_data: np.ndarray,
    axes_label: list = None,
    title_prefix: str = r"Expected Donations ($\lambda_k$) vs ",
):
    """
    Line plot for continuous covariates or binary
    """
    K, C_em = beta_em.shape
    x_ref = x_em_data.mean(axis=(0, 1))

    keys_list = list(cov_names_em.keys())
    j = keys_list.index(var)

    # Lookup the values grid
    vals = x_em_data[:, :, j].flatten()
    grid = np.unique(vals)

    if len(grid) > 10:
        grid = np.linspace(vals.min(), vals.max(), 40)

    lam_grid = np.zeros((len(grid), K))
    for g, val in enumerate(grid):
        x = x_ref.copy()
        x[j] = val
        lam_grid[g] = np.exp(beta_em @ x)

    plt.figure(figsize=(6, 4))

    # binary variables
    if len(grid) <= 3:
        x_pos = np.arange(len(grid))
        width = 0.8 / K
        display_labels = axes_label if axes_label else [str(int(g)) for g in grid]

        for k in range(K):
            offset = (k - (K - 1) / 2) * width
            plt.bar(
                x_pos + offset,
                lam_grid[:, k],
                width=width,
                color=C.STATE_PALETTE[k],
                label=C.STATE_NAMES[k],
                edgecolor="white",
            )
        plt.xticks(x_pos, display_labels)
    # continuous variables
    else:
        for k in range(K):
            plt.plot(
                grid,
                lam_grid[:, k],
                color=C.STATE_PALETTE[k],
                label=C.STATE_NAMES[k],
                lw=2,
            )
        plt.grid(ls=":", alpha=0.5)

    plt.xlabel(cov_names_em[var])
    plt.ylabel(r"Expected Donations ($\lambda_k$)")
    plt.title(f"{title_prefix}{cov_names_em[var]}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

    return plt


def plot_trans_vs_cov_categorical(
    vars_list: list,
    cov_names_A: dict,
    W_A: np.ndarray,
    log_A0: np.ndarray,
    x_A_data: np.ndarray,
    axes_label: list = None,
    var_title: str = "Factor",
):
    """
    Side-by-side horizontal stacked bar charts showing transition probabilities
    for each previous state across categorical dummy levels.
    """
    K = W_A.shape[0]

    keys_list = list(cov_names_A.keys())
    dummy_indices = [keys_list.index(v) for v in vars_list]

    # reference donor
    x_ref = x_A_data.mean(axis=(0, 1))

    num_levels = 1 + len(vars_list)

    if axes_label is None:
        axes_label = ["Baseline"] + [cov_names_A[v] for v in vars_list]

    # create subplots side by side
    fig, axes = plt.subplots(1, K, figsize=(4.5 * K, 4), sharey=True)
    if K == 1:
        axes = [axes]

    y_pos = np.arange(num_levels)

    # iterate over all possible previous states
    for prev_state in range(K):
        trans_grid = np.zeros((num_levels, K))

        # calculate baseline
        x = x_ref.copy()
        x[dummy_indices] = 0.0
        logits = log_A0[prev_state] + (W_A[prev_state] @ x)
        trans_grid[0] = softmax_row(logits)

        # calculate other levels
        for i, idx in enumerate(dummy_indices):
            x = x_ref.copy()
            x[dummy_indices] = 0.0
            x[idx] = 1.0
            logits = log_A0[prev_state] + (W_A[prev_state] @ x)
            trans_grid[i + 1] = softmax_row(logits)

        # plot on the specific subplot
        ax = axes[prev_state]
        left_starts = np.zeros(num_levels)

        for j in range(K):
            # assign label only on the first subplot to avoid duplicated legends
            current_label = (
                rf"Actual State $\rightarrow$ {C.STATE_NAMES[j]}"
                if prev_state == 0
                else ""
            )
            ax.barh(
                y_pos,
                trans_grid[:, j],
                left=left_starts,
                color=C.STATE_PALETTE[j],
                label=current_label,
                edgecolor="white",
                height=0.6,
            )
            left_starts += trans_grid[:, j]

        # formatting for each subplot
        ax.set_yticks(y_pos)
        if prev_state == 0:
            ax.set_yticklabels(axes_label)

        ax.set_xlabel("Transition Probability")
        ax.set_xlim(0, 1)
        ax.set_title(f"From {C.STATE_NAMES[prev_state]}")

    # overall title and legend
    fig.suptitle(f"Transition Probabilities vs {var_title}", y=1.05, fontsize=14)
    fig.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=K,
        frameon=False,
        fontsize=10,
    )

    plt.tight_layout()
    # plt.show()

    return fig


def plot_trans_vs_cov_continuous(
    var: str,
    cov_names_A: dict,
    W_A: np.ndarray,
    log_A0: np.ndarray,
    x_A_data: np.ndarray,
    axes_label: list = None,
    title_prefix: str = "Transition Probabilities vs ",
):
    """
    Side-by-side plots (line charts or horizontal stacked bar charts) showing
    transition probabilities across a continuous or binary covariate.
    """
    K = W_A.shape[0]

    keys_list = list(cov_names_A.keys())
    idx = keys_list.index(var)

    # reference donor
    x_ref = x_A_data.mean(axis=(0, 1))

    # extract grid of values
    vals = x_A_data[..., idx].flatten()
    grid = np.unique(vals)
    if len(grid) > 10:
        grid = np.linspace(np.percentile(vals, 1), np.percentile(vals, 99), 40)

    # create subplots side by side
    fig, axes = plt.subplots(1, K, figsize=(5 * K, 4.5), sharey=True)
    if K == 1:
        axes = [axes]

    is_categorical = len(grid) <= 3
    if is_categorical:
        y_pos = np.arange(len(grid))
        if axes_label is None:
            axes_label = [str(int(g)) for g in grid]

    # iterate over all possible previous states
    for prev_state in range(K):
        ax = axes[prev_state]

        # compute probabilities for the current previous state
        trans_grid = np.zeros((len(grid), K))
        for g, v in enumerate(grid):
            x = x_ref.copy()
            x[idx] = v
            logits = log_A0[prev_state] + (W_A[prev_state] @ x)
            trans_grid[g] = softmax_row(logits)

        if is_categorical:
            # stacked bar chart for binary variables
            left_starts = np.zeros(len(grid))

            for j in range(K):
                # assign label only on the first subplot for the global legend
                current_label = (
                    rf"Actual State $\rightarrow$ {C.STATE_NAMES[j]}"
                    if prev_state == 0
                    else ""
                )
                ax.barh(
                    y_pos,
                    trans_grid[:, j],
                    left=left_starts,
                    color=C.STATE_PALETTE[j],
                    label=current_label,
                    edgecolor="white",
                    height=0.6,
                )
                left_starts += trans_grid[:, j]

            # formatting
            ax.set_yticks(y_pos)
            if prev_state == 0:
                ax.set_yticklabels(axes_label)
                ax.set_ylabel(cov_names_A[var])

            ax.set_xlabel("Transition Probability")
            ax.set_xlim(0, 1)

        else:
            # line chart for continuous variables
            for j in range(K):
                current_label = (
                    rf"Actual State $\rightarrow$ {C.STATE_NAMES[j]}"
                    if prev_state == 0
                    else ""
                )
                ax.plot(
                    grid,
                    trans_grid[:, j],
                    color=C.STATE_PALETTE[j],
                    label=current_label,
                    lw=2,
                )
            ax.grid(ls=":", alpha=0.5)
            ax.set_xlabel(axes_label)

            ax.set_ylim(-0.05, 1.05)
            if prev_state == 0:
                ax.set_ylabel("Transition Probability")

        ax.set_title(f"From {C.STATE_NAMES[prev_state]}")

    # overall title
    fig.suptitle(f"{title_prefix}{cov_names_A[var]}", y=1.02, fontsize=14)

    # global legend at the bottom in a single row
    fig.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=K,
        frameon=False,
        fontsize=10,
    )

    plt.tight_layout()
    plt.show()

    return fig


def plot_donor_gg(
    results_dict,
    y_true_next=None,
):
    """
    Plot observed yearly donations colored by latent state (Viterbi),
    plus markers for predicted and actual next-year donations.

    Parameters
    ----------
    results_dict : dict
        Output from the `predict_hmm_donor` function (contains history_df, expected_next, etc.).
    y_true_next : int, optional
        Actual number of donations in the predicted year (if available in ground truth).
    """

    # extract data from the dict and transpose it in a new df
    df_hist = results_dict["history_df"].T.reset_index()
    df_hist.columns = ["year", "donations", "state"]
    df_hist["year"] = df_hist["year"].astype(int)
    df_hist["donations"] = df_hist["donations"].astype(int)

    # identify the predicted state
    pred_state_idx = np.argmax(results_dict["next_state_probs"])
    pred_color = C.STATE_PALETTE[pred_state_idx]

    next_year = int(results_dict["next_year"])
    expected_next = results_dict["expected_next"]

    rows_pred = [{"year": next_year, "donations": expected_next, "kind": "Predicted"}]
    if y_true_next is not None:
        rows_pred.append(
            {"year": next_year, "donations": y_true_next, "kind": "Actual"}
        )
    df_pred = pd.DataFrame(rows_pred)

    y_low, y_high = -0.5, C.MAX_N_DONATIONS + 0.5
    x_breaks = df_hist["year"].tolist() + [next_year]
    plot_colors = {C.STATE_NAMES[k]: color for k, color in C.STATE_PALETTE.items()}
    plot_colors["Predicted"] = pred_color
    plot_colors["Actual"] = "#54403bff"

    legend_breaks = [C.STATE_NAMES[k] for k in C.STATE_PALETTE.keys()]
    if y_true_next is not None:
        legend_breaks.append("Actual")

    p = (
        pn.ggplot(df_hist, pn.aes(x="year", y="donations"))
        + pn.geom_step(direction="mid", color="black", alpha=0.35)
        + pn.geom_point(pn.aes(color="state"), size=2.5)
        + pn.scale_color_manual(values=plot_colors, breaks=legend_breaks)
        + pn.scale_x_continuous(breaks=x_breaks, minor_breaks=None)
        + pn.scale_y_continuous(
            limits=(y_low, y_high),
            breaks=list(range(0, C.MAX_N_DONATIONS + 1)),
            minor_breaks=None,
        )
        + pn.labs(x="", y="Donations")
        + pn.theme_minimal()
        + pn.theme(
            # axis_text_x=pn.element_text(rotation=30, ha="right"),
            legend_title=pn.element_blank(),
            legend_text=pn.element_text(size=9),
            legend_position="bottom",
            panel_grid_minor=pn.element_blank(),
        )
        + pn.guides(color=pn.guide_legend(title=""))
    )

    p += pn.geom_vline(xintercept=next_year, linetype="dotted", alpha=0.3)

    p += pn.geom_point(
        data=df_pred,
        mapping=pn.aes(x="year", y="donations", color="kind"),
        size=3.5,
        shape="^",
        show_legend=False,
    )

    p += pn.geom_text(
        data=df_pred,
        mapping=pn.aes(x="year", y="donations", label="kind", color="kind"),
        size=8,
        show_legend=False,
        nudge_y=0.1,
        # adjust_text={
        #     'expand_text': (1.5, 1.5),  # Costringe i BOX di testo ad allontanarsi tra di loro
        #     'expand_points': (1.5, 1.5), # Li allontana dai triangoli
        #     'force_text': (0.5, 1.0),   # Applica una forza di repulsione vettoriale
        #     'arrowprops': {'arrowstyle': '-', 'color': 'gray', 'lw': 0.5}
        # }
    )

    return p


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


def plot_accuracy(glm_acc, hmm_acc):
    glm_acc *= 100
    hmm_acc *= 100

    methods = ["GLM", "HMM"]
    values = [glm_acc, hmm_acc]

    plt.figure(figsize=(5, 4))
    bars = plt.bar(methods, values, color=C.STATE_PALETTE.values(), edgecolor="black")

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
    plt.title("Accuracy: GLM vs HMM")
    plt.ylim(0, max(values) + 10)
    plt.show()


def plot_roc_auc(
    y_test: np.array,
    y_pred: list,
    names: list = ["GLM", "HMM-GLM one-state", "HMM-GLM multi-state"],
    figsize: list = (10, 4),
):
    """
    Visual summary of HMM parameters with emission GLM coefficients.
    """

    # transform the list into a matrix
    y_pred = np.column_stack(y_pred)
    # to get shape[1] even if only one model predctions are loaded
    y_pred = np.atleast_2d(y_pred)

    fig, axs = plt.subplots(1, y_pred.shape[1], figsize=figsize)

    y_true_binary = np.where(y_test > 0, 1, 0)

    for model in range(y_pred.shape[1]):
        # calculate the metrics
        fpr, tpr, thresholds = roc_curve(y_true_binary, y_pred[:, model])
        auc = roc_auc_score(y_true_binary, y_pred[:, model])

        axs[model].plot(
            fpr, tpr, color=C.STATE_PALETTE[model], label=f"AUC = {auc:0.3f}"
        )
        axs[model].plot([0, 1], [0, 1], color="grey", linestyle="--")

        axs[model].set_xlabel("False Positive Rate")
        axs[model].set_ylabel("True Positive Rate")
        axs[model].set_title(names[model])
        axs[model].legend()

    fig.suptitle("ROC Curve")

    fig.tight_layout()
    plt.show()
