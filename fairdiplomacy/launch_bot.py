# ==============================================================================
# Copyright 2019 - Philip Paquette
#
# NOTICE:  Permission is hereby granted, free of charge, to any person obtaining
#   a copy of this software and associated documentation files (the "Software"),
#   to deal in the Software without restriction, including without limitation the
#   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
#   sell copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in all
#   copies or substantial portions of the Software.
# ==============================================================================
#
# edited by jsgray to accomodate for different bots, work with heyhi cfg
#
# ==============================================================================
import argparse
import diplomacy
import logging
from diplomacy import connect
from diplomacy.utils import constants, exceptions, strings
from tornado import gen, ioloop

from fairdiplomacy.agents import build_agent_from_cfg

LOGGER = logging.getLogger("diplomacy_research.scripts.launch_bot")
PERIOD_SECONDS = 2


class Bot:
    """ Bot class. Properties:
        - host: name of host to connect
        - port: port to connect in host
        - username: name of user to connect to server. By default, private bot username.
        - password: password of user to connect to server. BY default, private bot password.
        - period_seconds: time (in seconds) between 2 queries in server to look for powers to order.
            By default, 10 seconds.
        - player_builder: a callable (without arguments) to be used to create a "player"
            which is responsible for generating orders for a single power in a game.
            Can be a class. By default, class RandomPlayer.
        - buffer_size: number of powers this bot will ask to manage to server.
    """

    def __init__(self, host, port, agent, *, period_seconds=PERIOD_SECONDS, buffer_size=128):
        """ Initialize the bot.
            :param host: (required) name of host to connect
            :param port: (required) port to connect
            :param port: (required) Agent object with get_orders function
            :param period_seconds: time in second between two consecutive bot queries on server. Default 10 seconds.
            :param buffer_size: number of powers to ask to server.
        """
        self.host = host
        self.port = port
        self.agent = agent
        self.username = constants.PRIVATE_BOT_USERNAME
        self.password = constants.PRIVATE_BOT_PASSWORD
        self.period_seconds = period_seconds
        self.player = None
        self.game_to_phase = {}
        self.buffer_size = buffer_size

    @gen.coroutine
    def run(self):
        """ Main bot code. """

        # Connecting to server
        connection = yield connect(self.host, self.port)
        LOGGER.info("Connected to %s", connection.url)
        LOGGER.info("Opening a channel.")
        channel = yield connection.authenticate(self.username, self.password)
        LOGGER.info("Connected as user %s.", self.username)

        while True:
            try:
                all_dummy_power_names = yield channel.get_dummy_waiting_powers(
                    buffer_size=self.buffer_size
                )

                # Getting orders for the dummy powers
                if all_dummy_power_names:
                    LOGGER.info("Managing %d game(s).", len(all_dummy_power_names))
                    yield [
                        self.generate_orders(channel, game_id, dummy_power_names)
                        for game_id, dummy_power_names in all_dummy_power_names.items()
                    ]
                yield gen.sleep(self.period_seconds)

            # Server error - Logging, but continuing
            except (exceptions.DiplomacyException, RuntimeError) as error:
                LOGGER.error(error)

    @gen.coroutine
    def generate_orders(self, channel, game_id, dummy_power_names):
        """ Generate orders for a list of power names in a network game.
            :param channel: a channel connected to a server.
            :param game_id: ID of network game to join.
            :param dummy_power_names: a sequence of power names waiting
                for orders in network game to join.
            :type channel: diplomacy.client.channel.Channel
            :type game_channel: diplomacy.client.channel.Channel
        """
        try:
            # Join powers.
            yield channel.join_powers(game_id=game_id, power_names=dummy_power_names)

            # Join all games
            games = yield {
                power_name: channel.join_game(game_id=game_id, power_name=power_name)
                for power_name in dummy_power_names
            }

            # Retrieves and submits all orders
            yield [
                self.submit_orders(games[power_name], power_name)
                for power_name in dummy_power_names
            ]

        except exceptions.ResponseException as exc:
            LOGGER.error("Exception occurred while working on game %s: %s", game_id, exc)

    @gen.coroutine
    def submit_orders(self, game, power_name):
        """ Retrieves and submits orders for a power
            :param game: An instance of the game object.
            :param power_name: The name of the power submitting orders (e.g. 'FRANCE')
            :type game: diplomacy.client.network_game.NetworkGame
        """
        with game.current_state():
            game_copy = diplomacy.Game()
            phase_history = yield game.get_phase_history()
            game_copy.set_phase_data(phase_history + [game.get_phase_data()])

            orders = self.agent.get_orders(game_copy, power_name)
            should_draw = False

            # Setting vote
            vote = strings.YES if should_draw else strings.NO
            if game.get_power(power_name).vote != vote:
                yield game.vote(power_name=power_name, vote=vote)

            # Setting orders
            yield game.set_orders(power_name=power_name, orders=orders, wait=False)

            # Printing log message
            LOGGER.info(
                "%s/%s/%s/orders: %s",
                game.game_id,
                game.current_short_phase,
                power_name,
                ", ".join(orders) if orders else "(empty)",
            )


def run_with_cfg(cfg):
    agent = build_agent_from_cfg(cfg.agent)
    bot = Bot(cfg.host, cfg.port, agent, period_seconds=cfg.period, buffer_size=cfg.buffer_size)
    io_loop = ioloop.IOLoop.instance()
    while True:
        try:
            io_loop.run_sync(bot.run)
        except KeyboardInterrupt:
            LOGGER.error("Bot interrupted.")
            break
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.error(exc)
            LOGGER.info("Restarting bot...")