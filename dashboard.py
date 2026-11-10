"""Build a self-contained dashboard from a Component A player summary."""

import argparse
import base64
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, help="Read and write dashboard data in this run directory")
    parser.add_argument("--output", type=Path, help="Override the generated HTML path")
    args = parser.parse_args()
    results_dir = args.results_dir or HERE / "results" / "v2"
    summary = results_dir / "player_summary.csv"
    if not summary.is_file():
        parser.error(f"missing {summary}. Run validation/run_distribution.py to score Component A v2 first.")
    players = pd.read_csv(summary).rename(columns={
        "nflId": "id",
        "player": "name",
        "mean_peak_option_value_proxy": "score",
        "positive_window_share": "positive",
        "mean_peak_nearest_defender_yards": "separation",
        "mean_first_down_probability": "first_down",
        "mean_touchdown_probability": "touchdown",
    })
    players = players.loc[players["position"].eq("RB")]
    if players.empty:
        parser.error(f"no running backs in {summary}")
    fields = ["id", "name", "routes", "score", "positive", "separation", "first_down", "touchdown"]
    charts = {}
    for name in ("ty_johnson_outlet", "rb_leaderboard"):
        path = HERE / "figures" / f"{name}.png"
        if path.is_file():
            charts[name] = base64.b64encode(path.read_bytes()).decode("ascii")
    data = {"players": players[fields].to_dict("records"), "charts": charts}
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("<", r"\u003c")
    template = (HERE / "dashboard.html.in").read_text(encoding="utf-8")
    output = args.output or (args.results_dir or HERE) / "receiver_dashboard_v2.html"
    output.write_text(template.replace("%%DATA%%", payload), encoding="utf-8")
    print(output.resolve())


if __name__ == "__main__":
    main()
