import numpy as np
import matplotlib.pyplot as plt


def plot_regime_horizon_map(
    horizon_panels,
    out_path=None,
    *,
    show_legend=True,
    show_transitions=True,
    transition_style="ticks",
    annotate_shocks=True
):
    fig, ax = plt.subplots(figsize=(14, 6))

    horizon_panels = sorted(
        horizon_panels,
        key=lambda p: int(str(p.get("label", "")).strip()[:-1]) if str(p.get("label", "")).strip().endswith("D") and str(p.get("label", "")).strip()[:-1].isdigit() else 10**9
    )

    COLORS = {1: "green", -1: "red", 0: "gray"}
    BASE_ALPHA = {1: 0.30, -1: 0.30, 0: 0.10}
    ALPHA_GAIN = {1: 0.60, -1: 0.60, 0: 0.25}

    band_height = 0.90
    amplitude = 0.85
    tick_half = 0.18

    for i, panel in enumerate(horizon_panels):
        x = panel["index"]
        mu = panel["mu"].astype(float)
        vol = panel["vol"].astype(float)
        regime = panel["regime"].astype(int)

        mag = np.abs(mu) / (np.nanmax(np.abs(mu)) + 1e-12)
        unc = vol / (np.nanmax(vol) + 1e-12)
        y_base = i

        for j in range(1, len(x)):
            r = int(regime.iloc[j])
            c = COLORS.get(r, "gray")
            alpha = BASE_ALPHA.get(r, 0.12) + ALPHA_GAIN.get(r, 0.30) * float(unc.iloc[j])
            alpha = float(np.clip(alpha, 0.05, 0.95))
            y1 = y_base + float(np.sign(mu.iloc[j]) * mag.iloc[j] * amplitude)

            ax.fill_between(
                [x[j - 1], x[j]],
                [y_base, y_base],
                [y1, y1],
                color=c,
                alpha=alpha,
                linewidth=0
            )

        if show_transitions:
            mark_only_uncertainty = True
            min_gap_by_label = {"5D": 10, "20D": 20, "60D": 30}
            lbl = str(panel.get("label", "")).strip()
            min_gap = min_gap_by_label.get(lbl, 15)

            changes = regime.ne(regime.shift(1)).fillna(False)
            change_times = x[changes.values]
            change_idx = np.where(changes.values)[0]

            last_mark = -10**9
            for idx_change, t in zip(change_idx, change_times):
                if idx_change - last_mark < min_gap:
                    continue

                prev_r = int(regime.iloc[idx_change - 1]) if idx_change - 1 >= 0 else int(regime.iloc[idx_change])
                new_r = int(regime.iloc[idx_change])

                if mark_only_uncertainty and not (prev_r == 0 or new_r == 0):
                    continue

                last_mark = idx_change

                if transition_style == "lines":
                    ax.axvline(t, linewidth=0.5, alpha=0.15, color="black")
                else:
                    ax.plot(
                        [t, t],
                        [y_base - tick_half, y_base + tick_half],
                        color="black",
                        linewidth=0.7,
                        alpha=0.35
                    )

    n = len(horizon_panels)
    ax.set_yticks(list(range(n)))

    expanded_labels = []
    for p in horizon_panels:
        lbl = str(p.get("label", "")).strip()
        if lbl.endswith("D") and lbl[:-1].isdigit():
            expanded_labels.append(f"{lbl[:-1]}-Day Horizon")
        else:
            expanded_labels.append(lbl if lbl else "Horizon")
    ax.set_yticklabels(expanded_labels)

    ax.set_xlabel("Date")
    ax.set_ylabel("Horizon (Rolling Window)")

    ax.set_title(
        "Regime–Horizon Map for SPY\n"
        "Green: Positive Drift | Red: Negative Drift | Gray: High Uncertainty : Volatility",
        fontsize=12
    )

    ax.grid(True, axis="x", alpha=0.18)
    ax.grid(False, axis="y")
    ax.set_ylim(-0.8, n - 1 + band_height)

    if show_legend:
        handles = []
        labels = []
        for k, name in [(1, "Positive Drift"), (-1, "Negative Drift"), (0, "High Uncertainty")]:
            h = ax.plot([], [], color=COLORS[k], linewidth=6, alpha=0.6)[0]
            handles.append(h)
            labels.append(name)
        ax.legend(handles, labels, loc="upper left", frameon=False, fontsize=9)

    if annotate_shocks:
        try:
            years = np.array([d.year for d in horizon_panels[-1]["index"]])
            if (years.min() <= 2020) and (years.max() >= 2020):
                import pandas as pd
                shock_date = pd.Timestamp("2020-03-16")
                ax.annotate(
                    "COVID-19 volatility shock",
                    xy=(shock_date, n - 1),
                    xytext=(shock_date, n - 1 + 0.55),
                    arrowprops=dict(arrowstyle="->", lw=0.7, alpha=0.55),
                    fontsize=9,
                    ha="left",
                    va="bottom",
                    alpha=0.85
                )
        except Exception:
            pass

    fig.tight_layout()

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=240)

    return fig
