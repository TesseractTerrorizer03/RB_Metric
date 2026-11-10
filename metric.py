"""Vectorized Component A pass-option scores for one dropback."""

import numpy as np

FIELD_LENGTH = 120.0
FIELD_WIDTH = 160.0 / 3
OWN_GOAL = 10.0
OPP_GOAL = 110.0
FPS = 10

BALL_SPEED = 23.0  # yards per second
MAX_PROJECTION = 0.55  # seconds of constant-velocity projection


def sigmoid(value):
    return 1 / (1 + np.exp(-value))


def state_value(down, to_go, yards_to_goal):
    """Transparent, hand-set EP-style field-position and down/distance proxy."""
    down_cost = np.array([0.0, 0.55, 1.15, 1.85])[np.asarray(down) - 1]
    standard_distance = np.minimum(10, yards_to_goal)
    return (
        -1.1 + 0.076 * (100 - yards_to_goal)
        - down_cost - 0.045 * (to_go - standard_distance)
    )


def outcome_values(down, to_go, yards_to_goal, gain):
    """Return complete/incomplete changes from the same pre-play state."""
    before = state_value(down, to_go, yards_to_goal)
    spot = np.clip(yards_to_goal - gain, 1, 99)
    converted = gain >= to_go
    first_down = state_value(1, np.minimum(10, spot), spot)
    # Turnover on downs flips the sign and the field position.
    next_down = np.minimum(4, down + 1)
    opponent_spot = 100 - spot
    opponent_start = 100 - yards_to_goal
    short = np.where(
        down == 4,
        -state_value(1, np.minimum(10, opponent_spot), opponent_spot),
        state_value(next_down, np.maximum(1, to_go - gain), spot),
    )
    incomplete = np.where(
        down == 4,
        -state_value(1, np.minimum(10, opponent_start), opponent_start),
        state_value(next_down, to_go, yards_to_goal),
    )
    complete = np.where(converted, first_down, short)
    complete = np.where(gain >= yards_to_goal, 7.0, complete)
    complete = np.where(yards_to_goal - gain >= 100, -2.0, complete)
    return complete - before, incomplete - before


def score_options(qb_xy, receiver_xy, receiver_velocity, defender_xy,
                  defender_velocity, line_x, down, to_go, validation_features=False):
    """Evaluate one candidate catch point per frame and receiver."""
    # Arrays are frame × player × xy. Catch points use only this frame's data.
    distance_now = np.linalg.norm(receiver_xy - qb_xy[:, None, :], axis=-1)
    projection = np.minimum(distance_now / BALL_SPEED, MAX_PROJECTION)
    catch_xy = receiver_xy + projection[..., None] * receiver_velocity
    catch_xy = np.clip(catch_xy, [0, 0], [FIELD_LENGTH, FIELD_WIDTH])
    projected_defense = (
        defender_xy[:, None, :, :]
        + projection[..., None, None] * defender_velocity[:, None, :, :]
    )
    separation = np.linalg.norm(
        projected_defense - catch_xy[:, :, None, :], axis=-1
    ).min(axis=-1)
    separation_now = np.linalg.norm(
        defender_xy[:, None, :, :] - receiver_xy[:, :, None, :], axis=-1
    ).min(axis=-1)

    # Check the middle of the throw segment at an approximate interception time.
    throw = catch_xy - qb_xy[:, None, :]
    throw_distance = np.linalg.norm(throw, axis=-1)
    from_qb = defender_xy[:, None, :, :] - qb_xy[:, None, None, :]
    length_squared = np.maximum(throw_distance[..., None] ** 2, 1e-6)
    fraction = np.sum(from_qb * throw[:, :, None, :], axis=-1) / length_squared
    lane_time = np.clip(fraction, 0, 1) * throw_distance[..., None] / BALL_SPEED
    from_qb = (
        from_qb
        + np.minimum(lane_time, MAX_PROJECTION)[..., None]
        * defender_velocity[:, None, :, :]
    )
    fraction = np.sum(from_qb * throw[:, :, None, :], axis=-1) / length_squared
    perpendicular = from_qb - fraction[..., None] * throw[:, :, None, :]
    clearance = np.linalg.norm(perpendicular, axis=-1)
    lane_clearance = np.where((fraction > 0.10) & (fraction < 0.90),
                              clearance, np.inf).min(axis=-1)
    lane_penalty = 1.7 * np.exp(-0.5 * (lane_clearance / 1.1) ** 2)
    completion = sigmoid(
        0.35 + 0.95 * (separation - 1.5) - 0.055 * throw_distance - lane_penalty
    )
    inbounds = ((receiver_xy >= [0, 0])
                & (receiver_xy <= [FIELD_LENGTH, FIELD_WIDTH])).all(axis=-1)
    completion = np.where(inbounds, completion, 0.0)

    catch_depth = catch_xy[..., 0] - line_x
    yac = np.clip(separation - 1.5, 0, 4)
    # Once a catch is in the end zone its value is a TD. A player catching in
    # their own end zone may still advance out of it in this simple YAC model.
    gain = np.floor(catch_depth + yac)
    complete, incomplete = outcome_values(down, to_go, OPP_GOAL - line_x, gain)
    signed_value = completion * complete + (1 - completion) * incomplete
    options = {
        "completion_proxy": completion,
        "catch_depth": catch_depth,
        "completion_gain": gain,
        "projected_separation": separation,
        "nearest_defender_yards": separation_now,
        "lane_clearance": np.where(np.isfinite(lane_clearance), lane_clearance, np.nan),
        "delta_complete_proxy": complete,
        "delta_incomplete_proxy": np.full_like(complete, incomplete),
        "option_value_proxy": signed_value,
        "positive_option_value_proxy": np.maximum(signed_value, 0.0),
    }
    if validation_features:
        offset = projected_defense - catch_xy[:, :, None, :]
        ahead = offset[..., 0] > 0
        distance = np.linalg.norm(offset, axis=-1)
        options.update({
            "throw_distance": throw_distance,
            "receiver_speed": np.linalg.norm(receiver_velocity, axis=-1),
            "receiver_downfield_speed": receiver_velocity[..., 0],
            "sideline_distance": np.minimum(catch_xy[..., 1], FIELD_WIDTH - catch_xy[..., 1]),
            "yac_proxy": yac,
            "ahead_clearance": np.minimum(
                np.where(ahead, distance, np.inf).min(axis=-1), 25,
            ),
            "defenders_in_runway": (ahead & (offset[..., 0] < 12)
                                    & (np.abs(offset[..., 1]) < 5)).sum(axis=-1),
        })
    return options
