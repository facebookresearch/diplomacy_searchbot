# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Tuple
import logging
import numpy as np
import torch

from conf import agents_cfgs
from fairdiplomacy import pydipcc
from fairdiplomacy.agents.base_search_agent import (
    n_move_phases_later,
    average_score_dicts,
    get_square_scores_from_game,
)
from fairdiplomacy.agents.model_wrapper import ModelWrapper
from fairdiplomacy.game import sort_phase_key
from fairdiplomacy.models.consts import POWERS
from fairdiplomacy.utils.thread_pool_encoding import FeatureEncoder
from fairdiplomacy.utils.timing_ctx import TimingCtx
from fairdiplomacy.typedefs import Action, JointAction, JointActionValues, Power


class ModelRollouts:
    def __init__(self, model: ModelWrapper, cfg: agents_cfgs.ModelRollouts):
        self.model = model
        self.feature_encoder = FeatureEncoder(cfg.n_threads)

        assert cfg.temperature >= 0, "Set rollout_cfg.temperature"
        assert cfg.max_rollout_length >= 0, "Set rollout_cfg.max_rollout_length"

        self.temperature = cfg.temperature
        self.top_p = cfg.top_p
        self.max_rollout_length = cfg.max_rollout_length
        self.mix_square_ratio_scoring = cfg.mix_square_ratio_scoring
        self.clear_old_all_possible_orders = cfg.clear_old_all_possible_orders
        self.average_n_rollouts = cfg.average_n_rollouts

    def do_rollouts(
        self, game_init, set_orders_dicts: List[JointAction], timings=None, log_timings=False,
    ) -> List[Tuple[JointAction, JointActionValues]]:
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
                for _ in range(self.average_n_rollouts)
            ]
        with timings("setup"):
            for i in range(len(games)):
                games[i].game_id += f"_{i}"
            est_final_scores = {}  # game id -> np.array len=7

            # set orders if specified
            for game, set_orders_dict in zip(
                games, repeat(set_orders_dicts, self.average_n_rollouts)
            ):
                for power, orders in set_orders_dict.items():
                    game.set_orders(power, list(orders))

            # for each game, a list of powers whose orders need to be generated
            # by the model on the first phase.
            missing_start_orders = {
                game.game_id: frozenset(p for p in POWERS if p not in set_orders_dict)
                for game, set_orders_dict in zip(
                    games, repeat(set_orders_dicts, self.average_n_rollouts)
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
                    batch_inputs = self.feature_encoder.encode_inputs(games_to_step)

                batch_orders, _, _ = self.model.do_model_request(
                    batch_inputs, self.temperature, self.top_p, timings=timings
                )

                with timings("env.set_orders"):
                    assert len(games_to_step) == len(batch_orders)
                    for game, orders_per_power in zip(games_to_step, batch_orders):
                        for power, orders in zip(POWERS, orders_per_power):
                            if step_id == 0 and power not in missing_start_orders[game.game_id]:
                                continue
                            game.set_orders(power, list(orders))

            with timings("env.step"):
                self.feature_encoder.process_multi([game for game in games_to_step])

        # Compute SoS for done game and query the net for not-done games.
        not_done_games = [game for game in games if not game.is_game_done]
        if not_done_games:
            with timings("encoding"):
                batch_inputs = self.feature_encoder.encode_inputs_state_only(not_done_games)

            batch_est_final_scores = self.model.do_model_request(
                batch_inputs, self.temperature, self.top_p, timings=timings, values_only=True
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
                    set_orders_dicts, groups_of(final_game_scores, self.average_n_rollouts)
                )
            ]

        if log_timings:
            timings.pprint(logging.getLogger("timings").info)

        return r


def repeat(seq, n):
    """Yield each element in seq 'n' times"""
    for e in seq:
        for _ in range(n):
            yield e


def groups_of(seq, n):
    """Yield len(seq)/n groups of `n` elements each"""
    for i in range(0, len(seq), n):
        yield seq[i : (i + n)]
