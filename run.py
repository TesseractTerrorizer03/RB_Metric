"""Score Component A pass options from the NYU Big Data Bowl tracking files."""

import argparse
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from metric import FIELD_LENGTH, FIELD_WIDTH, FPS, OPP_GOAL, OWN_GOAL, score_options

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE.parent / "nfl-big-data-bowl-regional-event-data-main" / "data"
SKILL_POSITIONS = {"WR", "TE", "RB", "FB"}
TRACKING_COLUMNS = [
    "gameId", "playId", "nflId", "frameId", "team", "playDirection", "x", "y", "s", "dir",
    "event",
]
KEYS = ["gameId", "playId"]
THROW_EVENTS = ("pass_forward", "autoevent_passforward", "autoevent_passinterrupted")
TERMINAL_EVENTS = {
    "C": THROW_EVENTS, "I": THROW_EVENTS, "IN": THROW_EVENTS,
    "S": ("qb_sack", "qb_strip_sack"), "R": ("run",),
}


def anchors(ball, result):
    """Prefer explicit snap/release tags over automated fallbacks."""
    snap = ball.loc[ball["event"].eq("ball_snap"), "frameId"].min()
    if pd.isna(snap):
        snap = ball.loc[ball["event"].eq("autoevent_ballsnap"), "frameId"].min()
    if pd.isna(snap):
        return None
    after_snap = ball[ball["frameId"] > snap]
    for event in TERMINAL_EVENTS.get(result, ()):
        terminal = after_snap.loc[after_snap["event"].eq(event), "frameId"].min()
        if pd.notna(terminal):
            return int(snap), int(terminal), event
    return None


def player_tracks(window, player_ids, snap, terminal, direction):
    """Align the passer, receivers and defenders on the same tracking frames."""
    frames = pd.RangeIndex(snap, terminal, name="frameId")
    index = pd.MultiIndex.from_product([frames, player_ids], names=["frameId", "nflId"])
    rows = window[window["nflId"].isin(player_ids)].set_index(["frameId", "nflId"])
    if not rows.index.is_unique:
        return None
    tracks = rows.reindex(index)[["x", "y", "s", "dir"]].to_numpy()
    tracks = tracks.reshape(len(frames), len(player_ids), 4)
    valid = np.isfinite(tracks).all(axis=(1, 2))
    if not valid[0] or valid.sum() < 2:
        return None
    xy = tracks[valid, :, :2]
    speed = tracks[valid, :, 2]
    angle = np.deg2rad(tracks[valid, :, 3])
    velocity = speed[..., None] * np.stack([np.sin(angle), np.cos(angle)], axis=-1)
    if direction == "left":
        xy = np.array([FIELD_LENGTH, FIELD_WIDTH]) - xy
        velocity = -velocity
    return frames.to_numpy()[valid], xy, velocity


def summarize(metadata, routes, players, frame_ids, options):
    value = options["positive_option_value_proxy"]
    signed = options["option_value_proxy"]
    separation = options["nearest_defender_yards"]
    records = []
    for col, player_id in enumerate(routes):
        peak = signed[:, col].argmax()
        separation_peak = separation[:, col].argmax()
        records.append({
            **metadata,
            "nflId": player_id,
            "player": players[player_id]["displayName"],
            "position": players[player_id]["officialPosition"],
            "frames_scored": len(frame_ids),
            "peak_option_value_proxy": value[peak, col],
            "peak_signed_option_value_proxy": signed[peak, col],
            "peak_frame": frame_ids[peak],
            "peak_seconds_after_snap": (frame_ids[peak] - frame_ids[0]) / FPS,
            "catch_depth_at_peak": options["catch_depth"][peak, col],
            "completion_gain_at_peak": options["completion_gain"][peak, col],
            "completion_proxy_at_peak": options["completion_proxy"][peak, col],
            "nearest_defender_yards_at_peak": separation[peak, col],
            "peak_nearest_defender_yards": separation[separation_peak, col],
            "separation_peak_frame": frame_ids[separation_peak],
            "completion_gain_at_separation_peak": options["completion_gain"][separation_peak, col],
            "option_value_at_separation_peak_proxy": value[separation_peak, col],
        })
    return records


def score_play(tracking, play, key, roles, players, include_frames, validation_features=False):
    game_id, play_id = key
    ball = tracking[tracking["team"].eq("football")]
    window_bounds = anchors(ball, play.passResult)
    if window_bounds is None:
        return "missing snap or terminal", [], None
    snap, terminal, terminal_event = window_bounds
    direction = ball.loc[ball["frameId"].eq(snap), "playDirection"].iloc[0]
    if (play.down not in (1, 2, 3, 4) or not play.yardsToGo > 0
            or direction not in ("left", "right")):
        return "invalid play context", [], None
    line_x = play.absoluteYardlineNumber
    if direction == "left":
        line_x = FIELD_LENGTH - line_x
    if not (OWN_GOAL < line_x < OPP_GOAL):
        return "invalid line of scrimmage", [], None

    window = tracking[tracking["frameId"].between(snap, terminal - 1)]
    offense = set(window.loc[window["team"].eq(play.possessionTeam), "nflId"].dropna())
    defenders = sorted(
        window.loc[window["team"].eq(play.defensiveTeam), "nflId"].dropna().unique()
    )
    passer = [player for player in roles.get("Pass", ()) if player in offense]
    routes = [
        player for player in roles.get("Pass Route", ())
        if player in offense and players.get(player, {}).get("officialPosition") in SKILL_POSITIONS
    ]
    if len(passer) != 1 or not routes or not defenders:
        return "missing passer, routes or defenders", [], None
    tracks = player_tracks(window, [*passer, *routes, *defenders], snap, terminal, direction)
    if tracks is None:
        return "incomplete player tracking", [], None
    frame_ids, xy, velocity = tracks
    crossed = np.flatnonzero(xy[:, 0, 0] > line_x)
    if crossed.size:
        stop = crossed[0]
        if stop < 2:
            return "passer crossed line before a scoring window", [], None
        terminal, terminal_event = int(frame_ids[stop]), "passer_crossed_line"
        frame_ids, xy, velocity = frame_ids[:stop], xy[:stop], velocity[:stop]
    receiver_end = 1 + len(routes)
    options = score_options(
        xy[:, 0], xy[:, 1:receiver_end], velocity[:, 1:receiver_end],
        xy[:, receiver_end:], velocity[:, receiver_end:], line_x, play.down, play.yardsToGo,
        validation_features=validation_features,
    )
    metadata = {
        "gameId": game_id, "playId": play_id, "team": play.possessionTeam,
        "down": play.down, "yards_to_go": play.yardsToGo, "yards_to_goal": OPP_GOAL - line_x,
        "pass_result": play.passResult, "snap_frame": snap, "terminal_frame": terminal,
        "terminal_event": terminal_event,
    }
    records = summarize(metadata, routes, players, frame_ids, options)
    frames = None
    if include_frames:
        count = len(routes)
        frames = pd.DataFrame({
            "gameId": game_id,
            "playId": play_id,
            "frameId": np.repeat(frame_ids, count),
            "nflId": np.tile(routes, len(frame_ids)),
            "seconds_after_snap": np.repeat((frame_ids - snap) / FPS, count),
            **{name: values.ravel() for name, values in options.items()},
        })
    return "scored", records, frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    games = parser.add_mutually_exclusive_group()
    games.add_argument("--games", type=int, help="limit to the first N tracking files")
    games.add_argument("--game-id", type=int, help="score a particular game")
    parser.add_argument("--play-id", type=int, help="score a particular play within --game-id")
    parser.add_argument("--frames", action="store_true", help="export the frame-level inputs and scores")
    parser.add_argument("--plot", action="store_true", help="draw a timeline for --game-id and --play-id")
    args = parser.parse_args()
    if args.games is not None and args.games < 1:
        parser.error("--games must be positive")
    if args.play_id is not None and args.game_id is None:
        parser.error("--play-id requires --game-id")
    if args.plot and args.play_id is None:
        parser.error("--plot requires --play-id and --game-id")

    tracking_dir = args.data_dir / "tracking"
    files = sorted(tracking_dir.glob("tracking_*.csv"))[:args.games]
    if args.game_id is not None:
        files = [path for path in files if path.stem == f"tracking_{args.game_id}"]
    if not files:
        parser.error(f"no matching tracking files in {tracking_dir}")
    plays = pd.read_csv(args.data_dir / "plays.csv").set_index(KEYS)
    players = pd.read_csv(args.data_dir / "players.csv").set_index("nflId").to_dict("index")
    scouting = pd.read_csv(args.data_dir / "pffScoutingData.csv", usecols=[
        *KEYS, "nflId", "pff_role",
    ])
    scouting = scouting[scouting["pff_role"].isin(("Pass", "Pass Route"))]
    if args.game_id is not None:
        scouting = scouting[scouting["gameId"].eq(args.game_id)]
    roles_by_play = {
        key: {role: rows["nflId"].tolist() for role, rows in group.groupby("pff_role")}
        for key, group in scouting.groupby(KEYS, sort=False)
    }

    records = []
    counts = Counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Stage CSVs so interrupted computation cannot publish a partial table.
    with TemporaryDirectory(dir=args.output_dir) as temp:
        staging = Path(temp)
        frame_path = staging / "frame_scores.csv"
        for index, path in enumerate(files, 1):
            tracking = pd.read_csv(path, usecols=TRACKING_COLUMNS)
            if args.play_id is not None:
                tracking = tracking[tracking["playId"].eq(args.play_id)]
            for key, group in tracking.groupby(KEYS, sort=False):
                if key not in plays.index:
                    counts["missing play metadata"] += 1
                    continue
                status, scored, frames = score_play(
                    group, plays.loc[key], key, roles_by_play.get(key, {}), players,
                    args.frames or args.plot,
                )
                counts[status] += 1
                records.extend(scored)
                if args.frames and frames is not None:
                    frames.to_csv(frame_path, mode="a", header=not frame_path.exists(),
                                  index=False, float_format="%.6f")
                if args.plot and frames is not None:
                    from plot import plot_play

                    plot_play(frames, scored, plays.loc[key, "playDescription"], staging / "play.png")
            if index % 10 == 0 or index == len(files):
                print(f"{index}/{len(files)} games · {len(records):,} receiver-plays", flush=True)

        if not records:
            parser.error(f"no receiver-plays scored ({dict(counts)})")
        scores = pd.DataFrame(records).sort_values([*KEYS, "nflId"])
        scores.to_csv(staging / "receiver_scores.csv", index=False, float_format="%.6f")
        summary = scores.assign(positive=scores["peak_option_value_proxy"].gt(0)).groupby(
            ["nflId", "player", "position"], as_index=False
        ).agg(
            routes=("peak_option_value_proxy", "size"),
            mean_peak_option_value_proxy=("peak_option_value_proxy", "mean"),
            positive_window_share=("positive", "mean"),
            mean_peak_nearest_defender_yards=("peak_nearest_defender_yards", "mean"),
        )
        summary = summary.sort_values(
            ["position", "mean_peak_option_value_proxy", "player", "nflId"],
            ascending=[True, False, True, True],
        )
        summary.insert(0, "rank_within_position", summary.groupby("position").cumcount() + 1)
        summary.to_csv(staging / "player_summary.csv", index=False, float_format="%.9f")
        for file in staging.iterdir():
            file.replace(args.output_dir / file.name)
    print(f"{dict(counts)}\nWrote results to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
