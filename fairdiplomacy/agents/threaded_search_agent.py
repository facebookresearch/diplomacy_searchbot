import logging
from collections import Counter
from typing import List, Tuple, Dict, Union, Sequence

import numpy as np
import torch

from fairdiplomacy.game import sort_phase_key
from fairdiplomacy import pydipcc
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
        value_model_path=None,
        max_batch_size,
        max_rollout_length=3,
        rollout_temperature,
        rollout_top_p=1.0,
        n_rollout_procs=70,
        device=0,
        mix_square_ratio_scoring=0,
        clear_old_all_possible_orders=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_rollout_procs = n_rollout_procs
        self.rollout_temperature = rollout_temperature
        self.rollout_top_p = rollout_top_p
        self.max_batch_size = max_batch_size
        self.max_rollout_length = max_rollout_length
        self.mix_square_ratio_scoring = mix_square_ratio_scoring
        self.clear_old_all_possible_orders = clear_old_all_possible_orders
        self.device = parse_device(device) if torch.cuda.is_available() else "cpu"

        self.model = load_dipnet_model(model_path, eval=True)
        # Loading model to gpu right away will load optimizer state we don't care about.
        self.model.to(self.device)
        if value_model_path is not None:
            self.value_model = load_dipnet_model(value_model_path, eval=True)
            # Loading model to gpu right away will load optimizer state we don't care about.
            self.value_model.to(self.device)
        else:
            self.value_model = self.model

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
                if values_only:
                    y = self.value_model(**x, values_only=values_only)
                else:
                    y = self.model(**x, values_only=values_only)
            if values_only:
                with timings("to_cpu"):
                    return y.cpu().numpy()
            with timings("transform"):
                y = model_output_transform(x, y)
            with timings("to_cpu"):
                y = tuple(x.to("cpu") for x in y)

        order_idxs, order_logprobs, final_scores = y

        with timings("model.decode"):
            decoded = self.thread_pool.decode_order_idxs(order_idxs)
        with timings("model.decode.tuple"):
            decoded = [[tuple(orders) for orders in powers_orders] for powers_orders in decoded]

        # Returning None for values. Must call with values_only to get values.
        return (decoded, order_logprobs, None)

    def get_values(self, game) -> np.ndarray:
        batch_inputs = encode_batch_inputs(self.thread_pool, [game])
        batch_est_final_scores = self.do_model_request(
            batch_inputs, self.rollout_temperature, self.rollout_top_p, values_only=True
        )
        return batch_est_final_scores[0]

    def do_rollouts(
        self, game_init, set_orders_dicts, average_n_rollouts=1, timings=None, log_timings=False
    ):
        if timings is None:
            timings = TimingCtx()

        if self.clear_old_all_possible_orders:
            with timings("clear_old_orders"):
                game_init = pydipcc.Game(game_init)
                game_init.clear_old_all_possible_orders()
        with timings("clone"):
            games = [
                pydipcc.Game(game_init)  # clones
                for _ in set_orders_dicts
                for _ in range(average_n_rollouts)
            ]
        with timings("setup"):
            for i in range(len(games)):
                games[i].game_id += f"_{i}"
            est_final_scores = {}  # game id -> np.array len=7

            # set orders if specified
            for game, set_orders_dict in zip(games, repeat(set_orders_dicts, average_n_rollouts)):
                for power, orders in set_orders_dict.items():
                    game.set_orders(power, list(orders))

            # for each game, a list of powers whose orders need to be generated
            # by the model on the first phase.
            missing_start_orders = {
                game.game_id: frozenset(p for p in POWERS if p not in set_orders_dict)
                for game, set_orders_dict in zip(
                    games, repeat(set_orders_dicts, average_n_rollouts)
                )
            }

        if self.max_rollout_length > 0:
            rollout_end_phase_id = sort_phase_key(
                n_move_phases_later(game_init.current_short_phase, self.max_rollout_length)
            )
            max_steps = 1000000
        else:
            # Really far ahead.
            rollout_end_phase_id = sort_phase_key(
                n_move_phases_later(game_init.current_short_phase, 10)
            )
            max_steps = 1

        # This loop steps the games until one of the conditions is true:
        #   - all games are done
        #   - at least one game was stepped for max_steps steps
        #   - all games are either completed or reach a phase such that
        #     sort_phase_key(phase) >= rollout_end_phase_id
        for step_id in range(max_steps):
            ongoing_game_phases = [
                game.current_short_phase for game in games if not game.is_game_done
            ]

            if len(ongoing_game_phases) == 0:
                # all games are done
                break

            # step games together at the pace of the slowest game, e.g. process
            # games with retreat phases alone before moving on to the next move phase
            min_phase = min(ongoing_game_phases, key=sort_phase_key)

            if sort_phase_key(min_phase) >= rollout_end_phase_id:
                break

            games_to_step = [
                game
                for game in games
                if not game.is_game_done and game.current_short_phase == min_phase
            ]

            if step_id > 0 or any(missing_start_orders.values()):
                with timings("encoding"):
                    batch_inputs = encode_batch_inputs(self.thread_pool, games_to_step)

                batch_orders, _, _ = self.do_model_request(
                    batch_inputs, self.rollout_temperature, self.rollout_top_p, timings=timings
                )

                with timings("env.set_orders"):
                    assert len(games_to_step) == len(batch_orders)
                    for game, orders_per_power in zip(games_to_step, batch_orders):
                        for power, orders in zip(POWERS, orders_per_power):
                            if step_id == 0 and power not in missing_start_orders[game.game_id]:
                                continue
                            game.set_orders(power, list(orders))

            with timings("env.step"):
                self.thread_pool.process_multi([game for game in games_to_step])

        # Compute SoS for done game and query the net for not-done games.
        not_done_games = [game for game in games if not game.is_game_done]
        if not_done_games:
            with timings("encoding"):
                batch_inputs = encode_batch_inputs(self.thread_pool, not_done_games)

            batch_est_final_scores = self.do_model_request(
                batch_inputs,
                self.rollout_temperature,
                self.rollout_top_p,
                timings=timings,
                values_only=True,
            )
            for game_idx, game in enumerate(not_done_games):
                est_final_scores[game.game_id] = np.array(batch_est_final_scores[game_idx])
        for game in games:
            if game.is_game_done:
                est_final_scores[game.game_id] = np.array(get_square_scores_from_game(game))

        with timings("final_scores"):
            final_game_scores = [
                dict(zip(POWERS, est_final_scores[game.game_id])) for game in games
            ]

            # mix in current sum of squares ratio to encourage losing powers to try hard
            # get GameScores objects for current game state
            if self.mix_square_ratio_scoring > 0:

                for game, final_scores in zip(games, final_game_scores):
                    current_scores = game.get_square_scores()
                    for pi, p in enumerate(POWERS):
                        final_scores[p] = (1 - self.mix_square_ratio_scoring) * final_scores[p] + (
                            self.mix_square_ratio_scoring * current_scores[pi]
                        )

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
