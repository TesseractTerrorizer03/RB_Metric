"""Score 2021 hypothetical states using the public fastrmodels EP booster."""

import lzma

import numpy as np
import pandas as pd
import xgboost as xgb


def load_model(path):
    # fastrmodels 2.1.0 stores the UBJSON booster as a named R raw vector.
    data = lzma.decompress(path.read_bytes())
    marker = b"ep_model\x00\x00\x00\x18"
    start = data.index(marker) + len(marker)
    length = int.from_bytes(data[start:start + 4], "big")
    booster = xgb.Booster(params={"nthread": 4})
    booster.load_model(bytearray(data[start + 4:start + 4 + length]))
    booster.set_param({"nthread": 4})
    return booster


def expected_points(booster, plays, down, to_go, spot, turnover=False):
    """Mirror nflfastR's 18 EP inputs for the post-2017 era."""
    roof = plays["roof"].replace({"open": "retractable", "closed": "retractable"})
    roof = roof.fillna("retractable")
    home = plays["posteam"].eq(plays["home_team"]).to_numpy()
    home = np.where(turnover, ~home, home) & plays["location"].ne("Neutral").to_numpy()
    own_timeouts = plays["posteam_timeouts_remaining"].to_numpy()
    opp_timeouts = plays["defteam_timeouts_remaining"].to_numpy()
    zero = np.zeros(len(plays))
    one = np.ones(len(plays))
    matrix = np.column_stack([
        plays["half_seconds_remaining"], spot, home,
        roof.eq("retractable"), roof.eq("dome"), roof.eq("outdoors"), to_go,
        zero, zero, zero, zero, one,
        *[down == n for n in range(1, 5)],
        np.where(turnover, opp_timeouts, own_timeouts),
        np.where(turnover, own_timeouts, opp_timeouts),
    ]).astype(np.float32)
    weights = np.array([7, -7, 3, -3, 2, -2, 0], dtype=np.float32)
    return np.sum(booster.inplace_predict(matrix) * weights, axis=1)


def outcome_deltas(booster, states, gain):
    """Value the spot after a gain; time and timeouts stay at the snap state."""
    # Collapse repeated frame outcomes before running the booster.
    states = states.assign(gain=gain)
    keys = ["gameId", "playId", "gain"]
    unique = states.drop_duplicates(keys).reset_index(drop=True)
    down, to_go, spot, gain = (
        unique[name].to_numpy() for name in ("down", "ydstogo", "yardline_100", "gain")
    )
    before = expected_points(booster, unique, down, to_go, spot)
    converted = gain >= to_go
    turnover = (down == 4) & ~converted
    next_down = np.where(converted | turnover, 1, down + 1)
    next_spot = np.clip(spot - gain, 1, 99)
    next_spot = np.where(turnover, 100 - next_spot, next_spot)
    next_to_go = np.where(next_down == 1, np.minimum(10, next_spot), to_go - gain)
    after = expected_points(booster, unique, next_down, next_to_go, next_spot, turnover)
    after = np.where(turnover, -after, after)
    after = np.where(gain >= spot, 7.0, after)
    after = np.where(spot - gain >= 100, -2.0, after)
    delta = pd.Series(after - before, index=pd.MultiIndex.from_frame(unique[keys]))
    return delta.reindex(pd.MultiIndex.from_frame(states[keys])).to_numpy()
