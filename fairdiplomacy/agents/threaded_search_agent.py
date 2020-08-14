import logging
from collections import Counter
from typing import List, Tuple, Dict, Union, Sequence

import numpy as np
import torch

from fairdiplomacy.game import sort_phase_key
import pydipcc
from fairdiplomacy.agents.base_search_agent import (
    BaseSearchAgent,
    model_output_transform,
    filter_keys,
    are_supports_coordinated,
    safe_idx,
    n_move_phases_later,
    get_square_scores_from_game,
    average_score_dicts,
)
from fairdiplomacy.agents.dipnet_agent import encode_batch_inputs
from fairdiplomacy.data.dataset import DataFields
from fairdiplomacy.models.consts import MAX_SEQ_LEN, POWERS
from fairdiplomacy.models.dipnet.load_model import load_dipnet_model
from fairdiplomacy.models.dipnet.order_vocabulary import (
    EOS_IDX,
    get_order_vocabulary,
    get_order_vocabulary_idxs_len,
)
from fairdiplomacy.utils.parse_device import parse_device
from fairdiplomacy.utils.timing_ctx import TimingCtx, DummyCtx
from fairdiplomacy.utils.cat_pad_sequences import cat_pad_sequences
from fairdiplomacy.utils.game_scoring import compute_game_scores_from_state

ORDER_VOCABULARY_TO_IDX = {order: idx for idx, order in enumerate(get_order_vocabulary())}


class ThreadedSearchAgent(BaseSearchAgent):
    def __init__(
        self,
        *,
        model_path,
        max_batch_size,
        max_rollout_length=3,
        rollout_temperature,
        rollout_top_p=1.0,
        n_rollout_procs=70,
        device=0,
        mix_square_ratio_scoring=0,
    ):
        super().__init__()
        self.n_rollout_procs = n_rollout_procs
        self.rollout_temperature = rollout_temperature
        self.rollout_top_p = rollout_top_p
        self.max_batch_size = max_batch_size
        self.max_rollout_length = max_rollout_length
        self.mix_square_ratio_scoring = mix_square_ratio_scoring
        self.device = parse_device(device) if torch.cuda.is_available() else "cpu"

        self.model = load_dipnet_model(model_path, map_location=self.device, eval=True)
        self.thread_pool = pydipcc.ThreadPool(
            n_rollout_procs, ORDER_VOCABULARY_TO_IDX, get_order_vocabulary_idxs_len()
        )

    def do_model_request(
        self,
        x: DataFields,
        temperature: float,
        top_p: float,
        timings=DummyCtx(),
        values_only: bool = False,
    ):
        with timings("model.pre"):
            B = x["x_board_state"].shape[0]
            x["temperature"] = torch.zeros(B, 1).fill_(temperature)
            x["top_p"] = torch.zeros(B, 1).fill_(top_p)

        # model forward / transform
        with torch.no_grad():
            with timings("to_cuda"):
                x = {k: v.to(self.device) for k, v in x.items()}
            with timings("model"):
                y = self.model(**x, values_only=values_only)
            if values_only:
                with timings("to_cpu"):
                    return y.cpu()
            with timings("transform"):
                y = model_output_transform(x, y)
            with timings("to_cpu"):
                y = tuple(x.to("cpu") for x in y)

        with timings("model.numpy"):
            order_idxs, order_logprobs, final_scores = y
            assert x["x_board_state"].shape[0] == final_scores.shape[0], (
                x["x_board_state"].shape[0],
                final_scores.shape[0],
            )
            if hasattr(final_scores, "numpy"):
                final_scores = final_scores.numpy()

        with timings("model.decode"):
            decoded = self.thread_pool.decode_order_idxs(order_idxs)
        with timings("model.decode.tuple"):
            decoded = [[tuple(orders) for orders in powers_orders] for powers_orders in decoded]

        return (decoded, order_logprobs, final_scores)

    def do_rollouts(self, game_init, set_orders_dicts, average_n_rollouts=1, log_timings=False):
        timings = TimingCtx()

        with timings("setup"):
            games = [
                pydipcc.Game(game_init)  # clones
                for _ in set_orders_dicts
                for _ in range(average_n_rollouts)
            ]
            for i in range(len(games)):
                games[i].game_id += f"_{i}"
            est_final_scores = {}  # game id -> np.array len=7

            # set orders if specified
            for game, set_orders_dict in zip(games, repeat(set_orders_dicts, average_n_rollouts)):
                for power, orders in set_orders_dict.items():
                    game.set_orders(power, list(orders))

            other_powers = {
                game.game_id: [p for p in POWERS if p not in set_orders_dict]
                for game, set_orders_dict in zip(
                    games, repeat(set_orders_dicts, average_n_rollouts)
                )
            }

            has_other_powers = any(other_powers.values())

        rollout_start_phase = game_init.current_short_phase
        rollout_end_phase = n_move_phases_later(rollout_start_phase, self.max_rollout_length)

        while True:
            # exit loop if all games are done before max_rollout_length
            ongoing_game_phases = [
                game.current_short_phase for game in games if not game.is_game_done
            ]
            if len(ongoing_game_phases) == 0:
                break

            # step games together at the pace of the slowest game, e.g. process
            # games with retreat phases alone before moving on to the next move phase
            min_phase = min(ongoing_game_phases, key=sort_phase_key)

            with timings("encoding"):
                games_to_encode = [
                    game
                    for game in games
                    if not game.is_game_done and game.current_short_phase == min_phase
                ]
                batch_inputs = encode_batch_inputs(self.thread_pool, games_to_encode)
                batch_data = list(
                    zip(games_to_encode, games_to_encode)
                )  # FIXME: remove zip, only need games

            with timings("cat_pad"):
                # xs = [inputs for game, inputs in batch_data]
                # batch_inputs = self.cat_pad_inputs(xs)
                pass

            if not has_other_powers:
                # We already know action for each power.
                if min_phase == rollout_end_phase:
                    # Need to compute values though.
                    batch_est_final_scores = self.do_model_request(
                        batch_inputs,
                        self.rollout_temperature,
                        self.rollout_top_p,
                        timings=timings,
                        values_only=True,
                    )
                    batch_orders = None
                else:
                    # Don't need anything. Probably first phase.
                    batch_est_final_scores = batch_orders = None

            else:
                batch_orders, _, batch_est_final_scores = self.do_model_request(
                    batch_inputs, self.rollout_temperature, self.rollout_top_p, timings=timings
                )

            with timings("score.max_rollout_len"):
                if min_phase == rollout_end_phase:
                    for game_idx, (game, _) in enumerate(batch_data):
                        est_final_scores[game.game_id] = np.array(batch_est_final_scores[game_idx])

                    # skip env step and exit loop once we've accumulated the estimated
                    # scores for all games up to max_rollout_length
                    break

            with timings("env.set_orders"):
                if has_other_powers:
                    assert len(batch_data) == len(batch_orders)
                    for (game, _), power_orders in zip(batch_data, batch_orders):
                        power_orders = dict(zip(POWERS, power_orders))
                        for other_power in other_powers[game.game_id]:
                            game.set_orders(other_power, list(power_orders[other_power]))
                        assert game.current_short_phase == min_phase

                for game in games:
                    other_powers[game.game_id] = POWERS  # no set orders on subsequent turns
                has_other_powers = True

            with timings("env.step"):
                self.thread_pool.process_multi([game for game, _ in batch_data])

            with timings("score.gameover"):
                for (game, _) in batch_data:
                    if game.is_game_done:
                        final_scores = np.array(get_square_scores_from_game(game))
                        est_final_scores[game.game_id] = final_scores

        # out of loop: rollouts are done

        with timings("final_scores.0"):
            final_game_scores = [
                dict(zip(POWERS, est_final_scores[game.game_id])) for game in games
            ]

        with timings("final_scores.mix"):
            # mix in current sum of squares ratio to encourage losing powers to try hard
            # get GameScores objects for current game state
            if self.mix_square_ratio_scoring > 0:

                for game, final_scores in zip(games, final_game_scores):
                    current_scores = game.get_square_scores()
                    for pi, p in enumerate(POWERS):
                        final_scores[p] = (1 - self.mix_square_ratio_scoring) * final_scores[p] + (
                            self.mix_square_ratio_scoring * current_scores[pi]
                        )

        with timings("final_scores.average"):
            r = [
                (set_orders_dict, average_score_dicts(scores_dicts))
                for set_orders_dict, scores_dicts in zip(
                    set_orders_dicts, groups_of(final_game_scores, average_n_rollouts)
                )
            ]

        if log_timings:
            timings.pprint(logging.getLogger("timings").info)

        return r

    @classmethod
    def cat_pad_inputs(cls, xs: List[DataFields]) -> DataFields:
        return DataFields({k: torch.cat([x[k] for x in xs]) for k in xs[0].keys()})


def repeat(seq, n):
    """Yield each element in seq 'n' times"""
    for e in seq:
        for _ in range(n):
            yield e


def groups_of(seq, n):
    """Yield len(seq)/n groups of `n` elements each"""
    for i in range(0, len(seq), n):
        yield seq[i : (i + n)]
