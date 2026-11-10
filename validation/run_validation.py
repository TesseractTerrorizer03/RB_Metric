"""Validate Component A on observed 2021 throws, then rescore held-out routes."""

import argparse
import os
from pathlib import Path
import sys

# Avoid oversubscribing the small per-play prediction batches.
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metric import outcome_values
from run import DEFAULT_DATA, KEYS, TRACKING_COLUMNS, score_play
from expected_points import expected_points, load_model, outcome_deltas

PBP_FIELDS = [
    "old_game_id", "play_id", "season", "week", "season_type", "posteam", "home_team",
    "roof", "location", "half_seconds_remaining", "posteam_timeouts_remaining",
    "defteam_timeouts_remaining", "down", "ydstogo", "yardline_100", "ep",
    "pass_attempt", "sack", "qb_spike", "penalty", "fumble_lost", "interception",
    "receiver_player_id", "complete_pass", "air_yards", "yards_after_catch", "receiving_yards",
]
GEOMETRY = [
    "catch_depth", "throw_distance", "receiver_speed", "receiver_downfield_speed",
    "projected_separation", "sideline_distance", "lane_clearance",
]
RUNWAY = ["ahead_clearance", "defenders_in_runway"]
CONTEXT = ["down", "ydstogo", "yardline_100"]
VARIANTS = ["original", "depth_only", "yac_only", "completion_only", "all_inputs", "all_inputs_nflfastR"]


def read_pbp(path, crosswalk):
    pbp = pd.read_parquet(path, columns=PBP_FIELDS).rename(columns={
        "old_game_id": "gameId", "play_id": "playId",
    })
    pbp = pbp.query("season == 2021 and week <= 8 and season_type == 'REG'").copy()
    pbp[KEYS] = pbp[KEYS].astype("int64")
    ids = pd.read_csv(crosswalk, usecols=["gsis_id", "nfl_id"]).dropna().set_index("gsis_id")
    pbp["target_nflId"] = pbp["receiver_player_id"].map(ids["nfl_id"])
    pbp["labeled_attempt"] = (
        pbp["pass_attempt"].eq(1) & pbp["sack"].eq(0) & pbp["qb_spike"].eq(0)
        & pbp["penalty"].eq(0) & pbp["air_yards"].notna()
        & pbp["target_nflId"].notna() & pbp["complete_pass"].notna()
    )
    return pbp.set_index(KEYS, verify_integrity=True)


def tracking_examples(data, pbp, all_games=False):
    plays = pd.read_csv(data / "plays.csv").set_index(KEYS)
    players = pd.read_csv(data / "players.csv").set_index("nflId").to_dict("index")
    scouting = pd.read_csv(data / "pffScoutingData.csv", usecols=[*KEYS, "nflId", "pff_role"])
    scouting = scouting[scouting["pff_role"].isin(["Pass", "Pass Route"])]
    roles = {
        key: {role: rows["nflId"].tolist() for role, rows in group.groupby("pff_role")}
        for key, group in scouting.groupby(KEYS, sort=False)
    }
    samples, heldout = [], []
    for path in sorted((data / "tracking").glob("tracking_*.csv")):
        game = int(path.stem.split("_")[-1])
        game_pbp = pbp.loc[game]
        test_game = all_games or game_pbp["week"].iloc[0] >= 6
        tracking = pd.read_csv(path, usecols=TRACKING_COLUMNS)
        for key, play in tracking.groupby(KEYS, sort=False):
            if key not in plays.index or key[1] not in game_pbp.index:
                continue
            context = game_pbp.loc[key[1]]
            if not test_game and not context.labeled_attempt:
                continue
            _, records, frames = score_play(
                play, plays.loc[key], key, roles.get(key, {}), players,
                include_frames=True, validation_features=True,
            )
            if frames is None:
                continue
            if test_game:
                heldout.append(frames)
            release = records[0]["terminal_event"] in ("pass_forward", "autoevent_passforward")
            if context.labeled_attempt and release:
                final_frame = records[0]["terminal_frame"] - 1
                target = frames[
                    frames["nflId"].eq(context.target_nflId) & frames["frameId"].eq(final_frame)
                ]
                if not target.empty:
                    samples.append(target)
    def with_context(parts, context):
        frames = pd.concat(parts, ignore_index=True)
        return frames.join(context, on=KEYS, validate="many_to_one")
    state_fields = [
        *CONTEXT, "week", "roof", "location", "posteam", "home_team",
        "half_seconds_remaining", "posteam_timeouts_remaining", "defteam_timeouts_remaining",
    ]
    return with_context(samples, pbp), with_context(heldout, pbp[state_fields]), players


def fit_regression(rows, fields, outcome):
    model = HistGradientBoostingRegressor(
        max_iter=120, max_leaf_nodes=12, min_samples_leaf=60,
        l2_regularization=3, early_stopping=False, random_state=7,
    )
    return model.fit(rows[fields], rows[outcome])


def fit_models(train):
    complete = train[train["complete_pass"].eq(1) & train["yards_after_catch"].notna()]
    depth = fit_regression(train, GEOMETRY + CONTEXT, "air_yards")
    yac_geometry = fit_regression(complete, GEOMETRY + CONTEXT, "yards_after_catch")
    yac_runway = fit_regression(complete, GEOMETRY + CONTEXT + RUNWAY, "yards_after_catch")
    logistic = make_pipeline(StandardScaler(), LogisticRegression(solver="liblinear", max_iter=1000))
    logistic.fit(completion_features(train), train["complete_pass"])
    return depth, yac_geometry, yac_runway, logistic


def completion_features(rows):
    geometry = rows[GEOMETRY].fillna({"lane_clearance": 10})
    p = rows["completion_proxy"].clip(1e-5, 1 - 1e-5)
    return geometry.assign(original_logit=np.log(p / (1 - p)))


def predict(models, rows):
    depth, yac_geometry, yac_runway, completion = models
    scale, logistic = completion.steps
    features = scale[1].transform(completion_features(rows))
    logits = (features * logistic[1].coef_[0]).sum(axis=1) + logistic[1].intercept_[0]
    return rows.assign(
        fitted_depth=depth.predict(rows[GEOMETRY + CONTEXT]),
        fitted_yac_geometry=yac_geometry.predict(rows[GEOMETRY + CONTEXT]),
        fitted_yac=yac_runway.predict(rows[GEOMETRY + CONTEXT + RUNWAY]),
        fitted_p=np.where(rows["completion_proxy"].gt(0),
                          expit(logits), 0.0),
    )


def signed_values(rows, booster):
    gain = rows["completion_gain"].to_numpy()
    fitted_gain = np.floor(rows["fitted_depth"] + rows["fitted_yac"])
    gains = [
        gain, np.floor(rows["fitted_depth"] + rows["yac_proxy"]),
        np.floor(rows["catch_depth"] + rows["fitted_yac"]), gain, fitted_gain,
    ]
    down = rows["down"].to_numpy(dtype=int)
    to_go, spot = rows["ydstogo"].to_numpy(), rows["yardline_100"].to_numpy()
    p = rows["completion_proxy"].to_numpy()
    fitted_p = rows["fitted_p"].to_numpy()
    for variant, candidate, prob in zip(VARIANTS, gains, [p, p, p, fitted_p, fitted_p]):
        complete, incomplete = outcome_values(down, to_go, spot, candidate)
        rows[variant] = prob * complete + (1 - prob) * incomplete
    complete = outcome_deltas(booster, rows, fitted_gain)
    incomplete = outcome_deltas(booster, rows, np.zeros(len(rows)))
    rows["all_inputs_nflfastR"] = fitted_p * complete + (1 - fitted_p) * incomplete
    return rows


def evaluate(test, booster):
    metrics = []
    completed = test["complete_pass"].eq(1) & test["yards_after_catch"].notna()
    for label, actual, cohort, predictions in [
        ("air_yards_completed", "air_yards", test[completed], ["catch_depth", "fitted_depth"]),
        ("intended_air_yards_all", "air_yards", test, ["catch_depth", "fitted_depth"]),
        ("yac", "yards_after_catch", test[completed],
         ["yac_proxy", "training_mean_yac", "fitted_yac_geometry", "fitted_yac"]),
        ("gain_completed", "receiving_yards", test[completed],
         ["completion_gain", "fitted_gain"]),
    ]:
        for pred in predictions:
            error = cohort[pred] - cohort[actual]
            metrics.append((label, pred, len(cohort), error.abs().mean(),
                            np.sqrt(np.mean(error**2)), error.mean()))
    for pred in ["training_completion_rate", "completion_proxy", "fitted_p"]:
        metrics.append(("completion_brier_logloss", pred, len(test),
                        brier_score_loss(test["complete_pass"], test[pred]),
                        log_loss(test["complete_pass"], test[pred]),
                        (test[pred] - test["complete_pass"]).mean()))
    binary = test[test["interception"].eq(0) & test["fumble_lost"].eq(0)].copy()
    realized_gain = binary["receiving_yards"].where(binary["complete_pass"].eq(1), 0)
    down = binary["down"].to_numpy(dtype=int)
    to_go, spot = binary["ydstogo"].to_numpy(), binary["yardline_100"].to_numpy()
    realized_manual, _ = outcome_values(down, to_go, spot, realized_gain)
    realized_official = outcome_deltas(booster, binary, realized_gain)
    for pred in VARIANTS:
        actual = realized_official if pred.endswith("nflfastR") else realized_manual
        metrics.append(("throw_value_own_point_scale", pred, len(binary),
                        mean_absolute_error(actual, binary[pred]),
                        np.sqrt(mean_squared_error(actual, binary[pred])),
                        (binary[pred] - actual).mean()))
    return pd.DataFrame(metrics, columns=["task", "estimate", "n", "error_1", "error_2", "bias"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pbp", required=True, type=Path)
    parser.add_argument("--players", required=True, type=Path)
    parser.add_argument("--ep-model", required=True, type=Path)
    parser.add_argument("--data-dir", default=DEFAULT_DATA, type=Path)
    parser.add_argument("--output-dir", default=Path(__file__).resolve().parents[1] / "results" / "validation", type=Path)
    args = parser.parse_args()
    pbp = read_pbp(args.pbp, args.players)
    labels, frames, players = tracking_examples(args.data_dir, pbp)
    train = labels[labels["week"].le(5)]
    test = labels[labels["week"].ge(6)].copy()
    assert set(train["gameId"]).isdisjoint(test["gameId"])
    assert not labels.duplicated(KEYS).any()
    print(f"Train {len(train):,} attempts / {train.gameId.nunique()} games. "
          f"Test {len(test):,} attempts / {test.gameId.nunique()} games.", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    booster = load_model(args.ep_model)
    ep_check = test.dropna(subset=["ep"])
    predicted = expected_points(booster, ep_check, ep_check["down"], ep_check["ydstogo"],
                                ep_check["yardline_100"])
    ep_error = mean_absolute_error(ep_check["ep"], predicted)
    if ep_error > 0.015:
        raise ValueError(f"EP model does not reproduce nflverse snap EP (MAE {ep_error:.4f})")
    print(f"EP snap parity MAE {ep_error:.6f}.", flush=True)
    with threadpool_limits(limits=4):
        models = fit_models(train)
        test = signed_values(predict(models, test), booster)
        test["training_mean_yac"] = train["yards_after_catch"].mean()
        test["training_completion_rate"] = train["complete_pass"].mean()
        test["fitted_gain"] = np.floor(test["fitted_depth"] + test["fitted_yac"])
        metrics = evaluate(test, booster)
        frames = frames.dropna(subset=CONTEXT)
        frames = signed_values(predict(models, frames), booster)
    route_values = frames.groupby([*KEYS, "nflId"], as_index=False)[VARIANTS].max()
    route_values[VARIANTS] = route_values[VARIANTS].clip(lower=0)
    route_values["player"] = route_values["nflId"].map(lambda n: players[n]["displayName"])
    route_values["position"] = route_values["nflId"].map(lambda n: players[n]["officialPosition"])
    grouped = route_values.query("position == 'WR'").groupby(["nflId", "player"])
    rankings = grouped[VARIANTS].mean()
    rankings.insert(0, "routes", grouped.size())
    rankings = rankings.query("routes >= 40").copy()
    for variant in VARIANTS:
        rankings[f"{variant}_rank"] = rankings[variant].rank(method="min", ascending=False).astype(int)
    rankings.sort_values("original_rank").to_csv(args.output_dir / "heldout_wr_rankings.csv", float_format="%.6f")
    metrics.to_csv(args.output_dir / "heldout_metrics.csv", index=False, float_format="%.6f")
    calibration = []
    for estimate in ["completion_proxy", "fitted_p"]:
        bins = pd.cut(test[estimate], np.linspace(0, 1, 11), include_lowest=True)
        by_bin = test.groupby(bins, observed=True).agg(
            attempts=("complete_pass", "size"),
            mean_predicted=(estimate, "mean"),
            observed_completion_rate=("complete_pass", "mean"),
        )
        calibration.append(by_bin.rename_axis("probability_bin").reset_index().assign(estimate=estimate))
    pd.concat(calibration).to_csv(args.output_dir / "heldout_completion_calibration.csv", index=False)
    test[[*KEYS, "nflId", "week", "complete_pass", "interception", "air_yards", "yards_after_catch",
          "receiving_yards", "catch_depth", "yac_proxy", "completion_gain", "completion_proxy",
          "fitted_depth", "fitted_yac", "fitted_gain", "fitted_p", *VARIANTS]].to_csv(
        args.output_dir / "heldout_attempts.csv", index=False, float_format="%.6f")
    print(metrics.round(4).to_string(index=False))
    print(f"Saved validation to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
