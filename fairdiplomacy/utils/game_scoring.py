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
        "is_clear_loss",  # 0/1 whether another player has more than half SC.
        "is_leader",  # 0/1 whether the player has at least as many SCs as anyone else.
        "num_games",  # Number of games being averaged
    ],
)


def compute_phase_scores(power_id: int, phase_json: Dict) -> GameScores:
    return compute_game_scores_from_state(power_id, phase_json["state"])


def compute_game_scores(power_id: int, game_json: Dict) -> GameScores:
    return compute_game_scores_from_state(power_id, game_json["phases"][-1]["state"])


def compute_game_scores_from_state(power_id: int, game_state: Dict) -> GameScores:
    center_counts = [len(game_state["centers"].get(p, [])) for p in POWERS]
    center_squares = [x ** 2 for x in center_counts]
    complete_unroll = game_state["name"] == "COMPLETED"
    is_clear_win = center_counts[power_id] > N_SCS / 2
    is_clear_loss = center_counts[power_id] == 0 or (
        not is_clear_win and any(c > N_SCS / 2 for c in center_counts)
    )
    metrics = dict(
        center_ratio=center_counts[power_id] / N_SCS,
        square_ratio=center_squares[power_id] / sum(center_squares, 1e-5),
        is_complete_unroll=float(complete_unroll),
        is_clear_win=float(is_clear_win),
        is_clear_loss=float(is_clear_loss),
        is_leader=float(center_counts[power_id] == max(center_counts)),
    )
    metrics["square_score"] = (
        1.0 if is_clear_win else (0 if is_clear_loss else metrics["square_ratio"])
    )
    return GameScores(**metrics, num_games=1)


def compute_game_scores(power_id: int, game_json: Dict) -> GameScores:
    return compute_phase_scores(power_id, game_json["phases"][-1])


def average_game_scores(many_games_scores: Sequence[GameScores]) -> GameScores:
    assert many_games_scores, "Must be non_empty"
    result = {}
    tot_n_games = sum(scores.num_games for scores in many_games_scores)
    for key in GameScores._fields:
        if key == "num_games":
            continue
        result[key] = (
            sum(getattr(scores, key) * scores.num_games for scores in many_games_scores)
            / tot_n_games
        )
    return GameScores(**result, num_games=tot_n_games)
