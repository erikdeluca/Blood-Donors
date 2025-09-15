from shiny import reactive, render
from shiny import ui as sui

from global import (
    donors_df, UIDS, donor_index_by_uid, predict_for_index, predict_from_manual,
    pmf_to_df, build_donor_plot, parse_counts_csv, PATHS_ALL, obs_torch, years_num
)

def app_server(input, output, session):
    # Populate donor select once
    @reactive.effect
    def _init_choices():
        # label choices like "12345 — M (1978)"
        choices = dict(zip(donors_df["label"].tolist(), donors_df["unique_number"].astype(str).tolist()))
        session.send_input_message("donor_uid", {"choices": choices, "selected": donors_df["unique_number"].astype(str).iloc[0]})

    # --- DONOR BROWSER TAB ---

    @reactive.calc
    def sel_index():
        uid = input.donor_uid()
        return donor_index_by_uid(uid) if uid else 0

    @reactive.calc
    def donor_prediction():
        # recompute on refresh or change in selection
        _ = input.refresh()
        return predict_for_index(sel_index())

    @render.ui
    def _vb_state():
        i = sel_index()
        last_state = int(PATHS_ALL[i, -1].item())
        return sui.value_box("Stato latente (ultimo)", sui.h3(f"State {last_state}"))

    @render.ui
    def _vb_mean():
        p = donor_prediction()
        return sui.value_box("E[y next]", sui.h3(f"{p.get('expected_next', float('nan')):.2f}"))

    @render.ui
    def _vb_prob():
        p = donor_prediction()
        prob = p.get("prob_donate_next", float("nan"))
        return sui.value_box("P(donare next)", sui.h3(f"{prob*100:.1f}%"))

    # Wire value boxes into the column wrap (they’re declared in UI via placeholders)
    output["_vb_state"] = _vb_state
    output["_vb_mean"]  = _vb_mean
    output["_vb_prob"]  = _vb_prob

    @render.plot
    def donor_plot():
        return build_donor_plot(sel_index(), donor_prediction())

    @render.data_frame
    def pmf_table():
        p = donor_prediction()
        return render.DataGrid(pmf_to_df(p["pmf_next"]))

    # --- WHAT-IF SIMULATOR TAB ---

    @reactive.calc
    def whatif_inputs():
        # Parse and validate
        by = int(input.birth_year() or 1985)
        g  = str(input.gender() or "M")
        y0 = int(input.start_year() or 2009)
        y1 = int(input.end_year() or 2023)
        if y1 < y0:
            y0, y1 = y1, y0
        years = list(range(y0, y1 + 1))
        counts = parse_counts_csv(input.counts_csv() or "")
        # pad/trim counts to match years length
        if len(counts) < len(years):
            counts = counts + [0]*(len(years)-len(counts))
        else:
            counts = counts[:len(years)]
        return {"birth_year": by, "gender": g, "years": years, "counts": counts}

    @reactive.calc
    def whatif_prediction():
        # trigger only on simulate click
        _ = input.simulate()
        w = whatif_inputs().copy()
        return predict_from_manual(w["birth_year"], w["gender"], w["years"], w["counts"])

    @render.ui
    def _vb_wi_mean():
        try:
            p = whatif_prediction()
            return sui.value_box("E[y next]", sui.h3(f"{p.get('expected_next', float('nan')):.2f}"))
        except Exception:
            return sui.value_box("E[y next]", sui.h3("—"))

    @render.ui
    def _vb_wi_prob():
        try:
            p = whatif_prediction()
            return sui.value_box("P(donare next)", sui.h3(f"{p.get('prob_donate_next', 0.0)*100:.1f}%"))
        except Exception:
            return sui.value_box("P(donare next)", sui.h3("—"))

    @render.ui
    def _vb_wi_state():
        try:
            # decode last state of history via a tiny Viterbi on-the-fly
            p = whatif_prediction()
            states = p.get("viterbi_states", [])
            last = states[-1] if states else None
            return sui.value_box("Stato latente finale", sui.h3(f"State {last if last is not None else '—'}"))
        except Exception:
            return sui.value_box("Stato latente finale", sui.h3("—"))

    output["_vb_wi_mean"]  = _vb_wi_mean
    output["_vb_wi_prob"]  = _vb_wi_prob
    output["_vb_wi_state"] = _vb_wi_state

    @render.plot
    def whatif_plot():
        # Attempt to plot provided history; we don’t have decoded states for arbitrary history,
        # but predict_donor already returns viterbi_states over the input years, so we can plot them.
        try:
            p = whatif_prediction()
            years = p["years"]
            counts = p["counts"]
            z = np.array(p["viterbi_states"], dtype=int)
            # Build a minimal matplotlib plot
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.step(years, counts, where="mid", color="#333", alpha=0.4)
            sc = ax.scatter(years, counts, c=z, cmap="Set1", s=60, zorder=3)
            ax.set_ylim(-0.5, 4.5)
            ax.set_xlabel("Anno"); ax.set_ylabel("Donazioni")
            ax.set_title(f"Traiettoria simulata – E[y next]={p['expected_next']:.2f}, P(don)={p['prob_donate_next']*100:.1f}%")
            return fig
        except Exception:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.text(0.5, 0.5, "Inserisci input validi e premi Simula", ha="center", va="center")
            ax.axis("off")
            return fig

    @render.data_frame
    def whatif_pmf():
        try:
            p = whatif_prediction()
            return render.DataGrid(pmf_to_df(p["pmf_next"]))
        except Exception:
            return render.DataGrid(
                pmf_to_df({"0": 1.0})
            )