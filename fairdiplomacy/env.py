# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import random
import time

import numpy as np
import torch

from fairdiplomacy.agents.searchbot_agent import SearchBotAgent

from fairdiplomacy.pydipcc import Game
from fairdiplomacy.models.consts import POWERS


PYDIPCC_MAX_YEAR = 1935


class OneSixPolicyProfile:
    """A combination of independent agents to predict all power moves."""

    def __init__(self, agent_one, agent_six, agent_one_power, seed=0):
        self._agent_one_power = agent_one_power
        self._six_powers = [p for p in POWERS if p != agent_one_power]
        self._agent_one = agent_one
        self._agent_six = agent_six

    def get_all_power_orders(self, game):
        logging.debug("Starting turn {}".format(game.phase))
        orders = {}
        orders.update(self._agent_one.get_orders_many_powers(game, [self._agent_one_power]))
        orders.update(self._agent_six.get_orders_many_powers(game, self._six_powers))
        return orders


class SharedPolicyProfile:
    """Single agent that predicts orders for all powers."""

    def __init__(self, agent):
        self.agent = agent

    def get_all_power_orders(self, game):
        return self.agent.get_orders_many_powers(game, POWERS)


class Env:
    def __init__(
        self, policy_profile, seed=0, cf_agent=None, max_year=PYDIPCC_MAX_YEAR, game_obj=None
    ):
        self.game = Game(game_obj) if game_obj is not None else Game()

        # set random seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.policy_profile = policy_profile
        self.cf_agent = cf_agent
        assert (
            max_year <= PYDIPCC_MAX_YEAR
        ), f"pydipcc doesn't allow to go beyond {PYDIPCC_MAX_YEAR}"
        self.max_year = max_year

    def process_turn(self, timeout=10):
        logging.debug("Starting turn {}".format(self.game.phase))

        power_orders = self.policy_profile.get_all_power_orders(self.game)

        for power, orders in power_orders.items():
            if not self.game.get_orderable_locations().get(power):
                logging.debug(f"Skipping orders for {power}")
                continue
            logging.info(
                "Set orders {} {} {}".format(self.game.current_short_phase, power, orders)
            )
            if self.cf_agent:
                cf_orders = self.cf_agent.get_orders(self.game, power)
                logging.debug(
                    "CF  orders {} {} {}".format(self.game.current_short_phase, power, cf_orders)
                )
            self.game.set_orders(power, orders)

        self.game.process()

    def process_all_turns(self, max_turns=0):
        """Process all turns until game is over

        Returns a dict mapping power -> supply count
        """
        turn_id = 0
        while not self.game.is_game_done:
            if max_turns and turn_id >= max_turns:
                break
            _, year, _ = self.game.phase.split()
            if int(year) > self.max_year:
                break
            self.process_turn()
            turn_id += 1

        return {k: len(v) for k, v in self.game.get_state()["centers"].items()}

    def save(self, output_path):
        logging.info("Saving to {}".format(output_path))
        with open(output_path, "w") as stream:
            stream.write(self.game.to_json())
