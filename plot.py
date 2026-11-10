"""Compare separation with Component A on a single play."""

from textwrap import fill

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from metric import FPS


def plot_play(frames, scores, description, path):
    play = scores[0]
    end = (play["terminal_frame"] - play["snap_frame"]) / FPS
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    fig.subplots_adjust(left=0.11, right=0.95, bottom=0.12, top=0.69, hspace=0.38)
    fig.text(0.11, 0.95, "Separation and pass-option value", fontsize=20, weight="bold")
    fig.text(
        0.11, 0.90, f'{play["gameId"]}  /  Play {play["playId"]}  ·  {play["team"]}'
        f'  ·  Down {play["down"]} and {play["yards_to_go"]}  ·  {play["yards_to_goal"]:g} yd to goal',
        fontsize=10,
    )
    fig.text(0.11, 0.87, fill(description, width=110, max_lines=2, placeholder=" …"),
             fontsize=10, va="top", color="#475569")

    for index, receiver in enumerate(sorted(scores, key=lambda row: row["position"])):
        series = frames[frames["nflId"].eq(receiver["nflId"])].set_index("frameId")
        color = plt.get_cmap("tab10")(index % 10)
        for axis, column, peak in [
            (axes[0], "nearest_defender_yards", "separation_peak_frame"),
            (axes[1], "positive_option_value_proxy", "peak_frame"),
        ]:
            axis.plot(series["seconds_after_snap"], series[column], lw=2, color=color,
                      label=f'{receiver["player"]} ({receiver["position"]})')
            peak_frame = receiver[peak]
            if series.at[peak_frame, column] > 0:
                axis.scatter(series.at[peak_frame, "seconds_after_snap"],
                             series.at[peak_frame, column], s=24, color=color, zorder=3)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper left",
               bbox_to_anchor=(0.10, 0.80), frameon=False, ncol=min(3, len(scores)), fontsize=9)
    axes[0].set_title("Nearest-defender separation", loc="left", fontsize=11, weight="bold")
    axes[1].set_title("Component A pass-option value", loc="left", fontsize=11, weight="bold")
    axes[0].set(ylabel="Yards", ylim=(0, max(0.1, frames["nearest_defender_yards"].max() * 1.12)))
    axes[1].set(ylabel="Point proxy", xlabel="Seconds after snap",
                ylim=(0, max(0.1, frames["positive_option_value_proxy"].max() * 1.12)))
    for axis in axes:
        axis.axvline(end, color="#64748b", linestyle=":", lw=1)
        axis.grid(axis="y", alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    axes[1].set_xlim(0, end * 1.02)
    event = play["terminal_event"].replace("_", " ")
    axes[1].text(1, 1.05, f"Dotted line marks {event}", transform=axes[1].transAxes,
                 ha="right", fontsize=8, color="#475569")
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)
