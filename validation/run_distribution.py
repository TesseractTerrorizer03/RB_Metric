"""Score possible completed gains and cross-fit Component A by game."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from expected_points import load_model, outcome_deltas
from run_validation import CONTEXT, DEFAULT_DATA, KEYS, fit_models, predict, read_pbp, tracking_examples

DRAW_COUNT = 25
GAIN_MIN, GAIN_MAX = -100, 100
WEEKS = ((1, 2, 3), (4, 5), (6, 7, 8))
SCORES = ["original_value", "mean_gain_ep", "distribution_ep"]


def gain_groups(rows):
    depth = np.digitize(rows["fitted_depth"], [3, 10, 20])
    yac = np.digitize(rows["fitted_yac"], [3, 6])
    return depth, yac


def fit_residuals(train):
    """Build empirical gain errors from inner game-held-out predictions."""
    samples = []
    for fit, check in GroupKFold(n_splits=3).split(train, groups=train["gameId"]):
        complete = train.iloc[check].query("complete_pass == 1").dropna(subset=["receiving_yards"])
        samples.append(predict(fit_models(train.iloc[fit]), complete))
    scored = pd.concat(samples)
    depth, yac = gain_groups(scored)
    residual = (scored["receiving_yards"] - scored["fitted_depth"] - scored["fitted_yac"]).to_numpy()
    grid = (np.arange(DRAW_COUNT) + 0.5) / DRAW_COUNT
    pooled = np.quantile(residual, grid)
    by_group = {}
    for a in range(4):
        same_depth = depth == a
        depth_draws = np.quantile(residual[same_depth], grid) if same_depth.sum() >= 60 else pooled
        for b in range(3):
            same_group = same_depth & (yac == b)
            by_group[a, b] = (
                np.quantile(residual[same_group], grid) if same_group.sum() >= 60 else depth_draws
            )
    return by_group


def completed_gains(rows, residuals):
    """Return legal completed-gain draws and the fitted point estimate."""
    depth, yac = gain_groups(rows)
    table = np.array([[residuals[a, b] for b in range(3)] for a in range(4)])
    center = rows["fitted_depth"].to_numpy() + rows["fitted_yac"].to_numpy()
    raw = np.floor(center[:, None] + table[depth, yac])
    goal = rows["yardline_100"].to_numpy()
    gains = np.clip(raw, goal[:, None] - 100, goal[:, None]).astype(np.int16)
    mean_gain = np.clip(np.floor(center), goal - 100, goal).astype(np.int16)
    return gains, mean_gain


def expected_value(rows, residuals, booster):
    """Average EP over gain draws, using one lookup per distinct play/spot."""
    gains, mean_gain = completed_gains(rows, residuals)
    goal = rows["yardline_100"].to_numpy()
    to_go = rows["ydstogo"].to_numpy()
    p = rows["fitted_p"].to_numpy()

    play_index = pd.MultiIndex.from_frame(rows[KEYS])
    codes, plays = pd.factorize(play_index, sort=False)
    first = np.flatnonzero(~play_index.duplicated())
    needed = np.zeros((len(plays), GAIN_MAX - GAIN_MIN + 1), dtype=bool)
    needed[codes, -GAIN_MIN] = True
    needed[codes, mean_gain - GAIN_MIN] = True
    for gain in gains.T:
        needed[codes, gain - GAIN_MIN] = True
    play_code, gain_code = np.nonzero(needed)
    lookup = np.zeros(needed.shape, dtype=np.float32)
    lookup[play_code, gain_code] = outcome_deltas(
        booster, rows.iloc[first[play_code]], gain_code + GAIN_MIN,
    )
    incomplete = lookup[codes, -GAIN_MIN]
    mean_complete = lookup[codes, mean_gain - GAIN_MIN]
    average_complete = lookup[codes[:, None], gains - GAIN_MIN].mean(axis=1)
    return rows.assign(
        original_value=rows["option_value_proxy"],
        mean_gain_ep=p * mean_complete + (1 - p) * incomplete,
        distribution_ep=p * average_complete + (1 - p) * incomplete,
        expected_gain=gains.mean(axis=1),
        gain_p10=np.quantile(gains, 0.1, axis=1),
        gain_p90=np.quantile(gains, 0.9, axis=1),
        first_down_probability=p * (gains >= np.minimum(to_go, goal)[:, None]).mean(axis=1),
        touchdown_probability=p * (gains >= goal[:, None]).mean(axis=1),
        mean_first_down_probability=p * (mean_gain >= np.minimum(to_go, goal)),
        mean_touchdown_probability=p * (mean_gain >= goal),
    )


def metrics(test, booster):
    sample = test.query("interception == 0 and fumble_lost == 0").copy()
    completed = sample["complete_pass"].eq(1)
    actual_gain = sample["receiving_yards"].where(completed, 0)
    actual = outcome_deltas(booster, sample, actual_gain)
    actual_first = completed & actual_gain.ge(sample[["ydstogo", "yardline_100"]].min(axis=1))
    actual_td = completed & actual_gain.ge(sample["yardline_100"])
    results = []
    for label, value, first, td in [
        ("mean gain", "mean_gain_ep", "mean_first_down_probability", "mean_touchdown_probability"),
        ("gain distribution", "distribution_ep", "first_down_probability", "touchdown_probability"),
    ]:
        results.append({
            "estimate": label, "attempts": len(sample),
            "state_value_mae": mean_absolute_error(actual, sample[value]),
            "state_value_rmse": np.sqrt(mean_squared_error(actual, sample[value])),
            "first_down_brier": brier_score_loss(actual_first, sample[first]),
            "touchdown_brier": brier_score_loss(actual_td, sample[td]),
        })
    return pd.DataFrame(results)


def summarize_routes(frames, players):
    # Average over three consecutive tracking frames to reduce one-frame spikes.
    keys = [*KEYS, "nflId"]
    frames = frames.sort_values([*keys, "frameId"]).reset_index(drop=True)
    grouped = frames.groupby(keys, sort=False)
    recent = grouped["distribution_ep"].rolling(3, min_periods=3).mean().reset_index(level=keys, drop=True)
    consecutive = frames["frameId"] - grouped["frameId"].shift(2) == 2
    frames["window_value"] = recent.where(consecutive)
    has_window = frames.groupby(keys, sort=False)["window_value"].transform("count").gt(0)
    # For shorter or fragmented tracking windows, use the mean of the observed frames.
    fallback = grouped["distribution_ep"].transform("mean")
    frames["window_value"] = frames["window_value"].where(has_window, fallback)
    peak = frames.dropna(subset=["window_value"]).groupby(keys, sort=False)["window_value"].idxmax()
    cols = keys + [
        "week", "frameId", "seconds_after_snap", "window_value", "fitted_p", "fitted_depth",
        "fitted_yac", "expected_gain", "gain_p10", "gain_p90", "first_down_probability",
        "touchdown_probability",
    ]
    routes = frames.loc[peak, cols].rename(columns={"frameId": "peak_frame"})
    routes = routes.join(grouped["nearest_defender_yards"].max().rename("peak_nearest_defender_yards"), on=keys)
    routes = routes.join(grouped["original_value"].max().clip(lower=0).rename("original_peak_value"), on=keys)
    routes["peak_option_value_proxy"] = routes["window_value"].clip(lower=0)
    routes["player"] = routes["nflId"].map(lambda n: players[n]["displayName"])
    routes["position"] = routes["nflId"].map(lambda n: players[n]["officialPosition"])
    return routes


def summarize_players(routes):
    summary = routes.assign(positive=routes["peak_option_value_proxy"].gt(0)).groupby(
        ["nflId", "player", "position"], as_index=False,
    ).agg(
        routes=("peak_option_value_proxy", "size"),
        mean_peak_option_value_proxy=("peak_option_value_proxy", "mean"),
        positive_window_share=("positive", "mean"),
        mean_peak_nearest_defender_yards=("peak_nearest_defender_yards", "mean"),
        mean_first_down_probability=("first_down_probability", "mean"),
        mean_touchdown_probability=("touchdown_probability", "mean"),
    )
    summary = summary.sort_values(["position", "mean_peak_option_value_proxy", "player"],
                                  ascending=[True, False, True])
    summary.insert(0, "rank_within_position", summary.groupby("position").cumcount() + 1)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pbp", required=True, type=Path)
    parser.add_argument("--players", required=True, type=Path)
    parser.add_argument("--ep-model", required=True, type=Path)
    parser.add_argument("--data-dir", default=DEFAULT_DATA, type=Path)
    parser.add_argument("--output-dir", default=Path(__file__).resolve().parents[1] / "results" / "v2", type=Path)
    parser.add_argument("--example", nargs=2, type=int, metavar=("GAME_ID", "PLAY_ID"),
                        help="export per-frame forecasts and completed-gain draws for this play")
    args = parser.parse_args()
    pbp = read_pbp(args.pbp, args.players)
    attempts, frames, players = tracking_examples(args.data_dir, pbp, all_games=True)
    frames = frames.dropna(subset=CONTEXT)
    booster = load_model(args.ep_model)
    results, checks = [], []
    with threadpool_limits(limits=4):
        for weeks in WEEKS:
            train = attempts[~attempts["week"].isin(weeks)]
            test = attempts[attempts["week"].isin(weeks)]
            assert set(train["gameId"]).isdisjoint(test["gameId"])
            models, residuals = fit_models(train), fit_residuals(train)
            test = expected_value(predict(models, test), residuals, booster)
            checks.append(test)
            heldout = frames[frames["week"].isin(weeks)]
            scored = expected_value(predict(models, heldout), residuals, booster)
            results.append(summarize_routes(scored, players))
            if args.example:
                game, play = args.example
                example = scored[scored["gameId"].eq(game) & scored["playId"].eq(play)]
                if not example.empty:
                    columns = [*KEYS, "nflId", "frameId", "seconds_after_snap", "fitted_p", "fitted_yac",
                               "first_down_probability", "touchdown_probability", "distribution_ep"]
                    trace = example[columns].copy()
                    draws, _ = completed_gains(example, residuals)
                    for index, gain in enumerate(draws.T, 1):
                        trace[f"completed_gain_{index:02d}"] = gain
                    args.output_dir.mkdir(parents=True, exist_ok=True)
                    trace.to_csv(args.output_dir / f"frames_{game}_{play}.csv", index=False,
                                 float_format="%.6f")
            print(f"Weeks {weeks}: {len(test):,} matched throws and {len(results[-1]):,} scored routes.", flush=True)

    routes = pd.concat(results, ignore_index=True)
    attempts = pd.concat(checks, ignore_index=True)
    report = metrics(attempts[attempts["week"].ge(6)], booster)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    routes.to_csv(args.output_dir / "receiver_scores.csv", index=False, float_format="%.6f")
    summarize_players(routes).to_csv(args.output_dir / "player_summary.csv", index=False, float_format="%.9f")
    report.to_csv(args.output_dir / "heldout_metrics.csv", index=False, float_format="%.6f")
    fields = [*KEYS, "nflId", "week", "air_yards", "yards_after_catch", "receiving_yards",
              "complete_pass", "fitted_depth", "fitted_yac", "fitted_p", "expected_gain",
              "gain_p10", "gain_p90", "first_down_probability", "touchdown_probability", *SCORES]
    attempts[fields].to_csv(args.output_dir / "attempt_predictions.csv", index=False, float_format="%.6f")
    print(report.to_string(index=False))
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
