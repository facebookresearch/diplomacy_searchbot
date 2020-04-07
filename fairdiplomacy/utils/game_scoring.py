from typing import Dict, Sequence
import collections

from fairdiplomacy.models.consts import POWERS, N_SCS


GameScores = collections.namedtuple(
    "GameScores",
    [
        "center_ratio",  # Ratio of the player's SCs to N_SCS.
        "square_ratio",  # Ratio of the squure of the SCs to sum of squares.
        "square_score",  # Same as square_ratio, but 1 if is_clear_win.
        "is_complete_unroll",  # 0/1 whether last phase is complete.
        "is_clear_win",  # 0/1 whether the player has more than half SC.
        "is_leader",  # 0/1 whether the player has at least as many SCs as anyone else.
    ],
)


def compute_game_scores(power_id: int, game_json: Dict) -> GameScores:
    last_phase_centers = game_json["phases"][-1]["state"]["centers"]
    center_counts = [len(last_phase_centers[p]) for p in POWERS]
    center_suqares = [x ** 2 for x in center_counts]
    complete_unroll = game_json["phases"][-1]["name"] == "COMPLETED"
    is_clear_win = center_counts[power_id] > N_SCS / 2
    metrics = dict(
        center_ratio=center_counts[power_id] / N_SCS,
        square_ratio=center_suqares[power_id] / sum(center_suqares, 1e-5),
        is_complete_unroll=float(complete_unroll),
        is_clear_win=float(is_clear_win),
        is_leader=float(center_counts[power_id] == max(center_counts)),
    )
    metrics["square_score"] = 1.0 if is_clear_win else metrics["square_ratio"]
    return GameScores(**metrics)


def average_game_scores(many_games_scores: Sequence[GameScores]) -> GameScores:
    assert many_games_scores, "Must be non_empty"
    result = {}
    for key in GameScores._fields:
        result[key] = sum(getattr(scores, key) for scores in many_games_scores) / len(
            many_games_scores
        )
    return GameScores(**result)
