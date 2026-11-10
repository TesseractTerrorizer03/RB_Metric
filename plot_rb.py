"""Render the RB leaderboard and a third-down outlet example."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd

HERE = Path(__file__).resolve().parent
BLUE = "#2160ac"
ORANGE = "#b44923"
INK = "#18253b"


def save(fig, name):
    out = HERE / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / name, dpi=160, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


def leaderboard(results):
    players = pd.read_csv(results / "player_summary.csv")
    top = players.query("position == 'RB' and routes >= 50").nlargest(10, "mean_peak_option_value_proxy")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(top.player, top.mean_peak_option_value_proxy, color=BLUE)
    for bar, row in zip(ax.patches, top.itertuples()):
        ax.text(bar.get_width() + .007, bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.3f} · {row.routes} routes", va="center", fontsize=9)
    ax.invert_yaxis()
    ax.set(xlim=(0, .52), xlabel="Mean peak pass-option value per route · expected points")
    ax.set_title("RB receiving opportunity · 2021 Weeks 1–8", loc="left", weight="bold", pad=14)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    save(fig, "rb_leaderboard.png")


def outlet(trace_path, results):
    routes = pd.read_csv(results / "receiver_scores.csv")
    route = routes.query("gameId == 2021102403 and playId == 3868 and player == 'Ty Johnson'").iloc[0]
    frames = pd.read_csv(trace_path)
    frames = frames[frames.nflId.eq(route.nflId)].sort_values("frameId")
    last = frames.iloc[-1]
    fig, (ax, gains) = plt.subplots(2, 1, figsize=(9, 6.5))
    fig.subplots_adjust(hspace=.55, top=.82)
    fig.suptitle("An easy catch can still leave fourth down", x=.125, ha="left",
                 fontsize=20, weight="bold", color=INK)
    fig.text(.125, .88, "Jets at Patriots · Third-and-7 · Ty Johnson caught the pass for four yards")
    for field, color, label in [
        ("fitted_p", BLUE, "Catch"),
        ("first_down_probability", ORANGE, "First down or touchdown"),
    ]:
        ax.plot(frames.seconds_after_snap, frames[field], color=color, lw=2.5, label=label)
        ax.scatter(last.seconds_after_snap, last[field], color=color)
        ax.annotate(f"{last[field]:.0%}", (last.seconds_after_snap, last[field]),
                    xytext=(9, -3), textcoords="offset points", color=color, weight="bold")
    ax.set(xlabel="Seconds after snap", ylabel="Chance if targeted", ylim=(0, 1.07),
           xlim=(0, last.seconds_after_snap + .24))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.legend(frameon=False, loc="center left")
    counts = last.filter(regex=r"^completed_gain_\d+$").astype(int).value_counts().sort_index()
    gains.bar(counts.index, counts, color=[ORANGE if gain >= 7 else "#9aabbf" for gain in counts.index])
    for x, label, color in [(4, "Actual gain", BLUE), (7, "First down", ORANGE)]:
        gains.axvline(x, color=color, ls="--", lw=1.5)
        gains.text(x + .15, counts.max() + .12, label, color=color, fontsize=9)
    gains.set(title="25 possible gains if caught · last frame before release",
              xlabel="Yards gained", ylabel="Scenarios", ylim=(0, counts.max() + 1))
    for panel in (ax, gains):
        panel.spines[["top", "right"]].set_visible(False)
        panel.set_axisbelow(True)
        panel.grid(axis="y", alpha=.2)
    save(fig, "ty_johnson_outlet.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=HERE / "results" / "v2")
    parser.add_argument("--frames", type=Path, help="V2 per-frame export for game 2021102403, play 3868")
    args = parser.parse_args()
    leaderboard(args.results_dir)
    if args.frames:
        outlet(args.frames, args.results_dir)
