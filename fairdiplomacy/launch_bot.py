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
import diplomacy
import logging
from copy import deepcopy
from diplomacy import connect
from diplomacy.utils import constants, exceptions, strings
from tornado import gen, ioloop

import torch
from fairdiplomacy.game import Game
from fairdiplomacy.agents import build_agent_from_cfg
from concurrent.futures import ThreadPoolExecutor

LOGGER = logging.getLogger("diplomacy_research.scripts.launch_bot")
LOGGER.setLevel(logging.DEBUG)
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

    def __init__(
        self,
        host,
        port,
        agent_cfg,
        *,
        period_seconds=PERIOD_SECONDS,
        buffer_size=128,
        reuse_agent=False,
        reuse_model_servers=0,
        game_id="",
        power="",
    ):
        """ Initialize the bot.
            :param host: (required) name of host to connect
            :param port: (required) port to connect
            :param agent_cfg: heyhi config object for agent
            :param period_seconds: time in second between two consecutive bot queries on server. Default 10 seconds.
            :param buffer_size: number of powers to ask to server.
            :param reuse_agent: if True, agent_fn is called a single time and
                the agent object is used for all powers
            :param reuse_model_servers: int, if non-zero then start this number of
                model servers and reuse them for all agents
        """
        self.host = host
        self.port = port
        self.agent_cfg = agent_cfg
        self.username = constants.PRIVATE_BOT_USERNAME
        self.password = constants.PRIVATE_BOT_PASSWORD
        self.period_seconds = period_seconds
        self.player = None
        self.game_to_phase = {}
        self.buffer_size = buffer_size
        self.reuse_model_servers = reuse_model_servers
        self.game_id = game_id
        self.power = power

        self.agents = {}  # (channel, power) -> Agent
        self.cfgs_with_server = []
        self.executor = ThreadPoolExecutor(8)

    @gen.coroutine
    def run(self):
        """ Main bot code. """

        # Connecting to server
        LOGGER.warning("About to connect")
        connection = yield connect(self.host, self.port)
        LOGGER.info("Connected to %s", connection.url)
        LOGGER.info("Opening a channel.")
        channel = yield connection.authenticate(self.username, self.password)
        LOGGER.info("Connected as user %s.", self.username)

        while True:
            try:
                get_dummy_waiting_powers_kwargs = {"buffer_size": self.buffer_size}
                if self.game_id:
                    get_dummy_waiting_powers_kwargs["only_game_id"] = self.game_id
                if self.power:
                    get_dummy_waiting_powers_kwargs["only_power"] = self.power
                all_dummy_power_names = yield channel.get_dummy_waiting_powers(
                    **get_dummy_waiting_powers_kwargs
                )
                LOGGER.debug(f"all_dummy_power_names={all_dummy_power_names}")

                # Getting orders for the dummy powers
                if all_dummy_power_names:
                    LOGGER.info("Managing %d game(s).", len(all_dummy_power_names))
                    yield [
                        self.generate_orders(channel, game_id, dummy_power_names)
                        for game_id, dummy_power_names in all_dummy_power_names.items()
                        if not self.game_id or game_id == self.game_id
                    ]
                yield gen.sleep(self.period_seconds)

            # Server error - Logging, but continuing
            except (exceptions.DiplomacyException, RuntimeError) as error:
                LOGGER.error(error)
                raise

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
            if self.power:
                if self.power in dummy_power_names:
                    LOGGER.info(f"Overriding power={self.power}, prev={dummy_power_names}")
                    dummy_power_names = [self.power]
                else:
                    # power is not waiting for orders
                    return

            # Join powers.
            LOGGER.info(f"Joining powers game_id={game_id}, powers={dummy_power_names}")
            yield channel.join_powers(game_id=game_id, power_names=dummy_power_names)

            # Init agents
            keys = [(channel, power) for power in dummy_power_names]
            missing_keys = [k for k in keys if k not in self.agents]
            LOGGER.info(f"Missing agents for keys: {missing_keys}")
            if self.reuse_model_servers <= 0:
                # don't reuse model servers
                futures = [
                    self.executor.submit(build_agent_from_cfg, self.agent_cfg)
                    for _ in missing_keys
                ]
                for key, future in zip(missing_keys, futures):
                    self.agents[key] = yield future
            else:
                # reuse model servers: first launch agents with unique servers
                assert self.reuse_model_servers <= torch.cuda.device_count()
                n_missing_server_agents = min(
                    len(missing_keys), self.reuse_model_servers - len(self.cfgs_with_server)
                )
                LOGGER.info(f"Launching {n_missing_server_agents} new server agents")
                new_server_agents = yield self.launch_n_server_agents(n_missing_server_agents)

                # then launch agents reusing servers
                n_missing_serverless_agents = len(missing_keys) - len(self.cfgs_with_server)
                LOGGER.info(f"Launching {n_missing_serverless_agents} new serverless agents")
                new_serverless_agents = yield self.launch_n_serverless_agents(
                    n_missing_serverless_agents
                )

                # assign agents to powers
                assert len(missing_keys) == len(new_server_agents) + len(
                    new_serverless_agents
                ), "Launched wrong number of agents, {} != {} + {}".format(
                    len(missing_keys), len(new_server_agents), len(new_serverless_agents)
                )

                for key, agent in zip(missing_keys, new_server_agents + new_serverless_agents):
                    self.agents[key] = agent

            # Join all games
            games = yield {
                power_name: channel.join_game(game_id=game_id, power_name=power_name)
                for power_name in dummy_power_names
            }

            # Retrieves and submits all orders
            yield [
                self.submit_orders(
                    games[power_name], power_name, self.agents[(channel, power_name)]
                )
                for power_name in dummy_power_names
            ]

        except exceptions.ResponseException as exc:
            LOGGER.error("Exception occurred while working on game %s: %s", game_id, exc)

    @gen.coroutine
    def submit_orders(self, game, power_name, agent):
        """ Retrieves and submits orders for a power
            :param game: An instance of the game object.
            :param power_name: The name of the power submitting orders (e.g. 'FRANCE')
            :param agent: Agent object
            :type game: diplomacy.client.network_game.NetworkGame
        """
        with game.current_state():
            game_copy = Game()
            phase_history = yield game.get_phase_history()
            game_copy.set_phase_data(phase_history + [game.get_phase_data()])

            orders = yield self.executor.submit(agent.get_orders, game_copy, power_name)
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

    @gen.coroutine
    def launch_n_server_agents(self, n):
        """Return a list of Agent objects"""
        futures = []
        for i in range(n):
            cfg = deepcopy(self.agent_cfg)
            getattr(cfg, cfg.WhichOneof("agent")).device = i  # use different gpus
            future = self.executor.submit(build_agent_from_cfg, cfg)
            futures.append(future)
        agents = yield futures

        # save server addrs in reusable cfgs for serverless agents
        server_addrs = [agent.hostports[0] for agent in agents]
        cfgs = [deepcopy(self.agent_cfg) for _ in server_addrs]
        for addr, cfg in zip(server_addrs, cfgs):
            getattr(cfg, cfg.WhichOneof("agent")).use_server_addr = addr
        self.cfgs_with_server.extend(cfgs)

        LOGGER.info(f"Launched {len(cfgs)} new server agents, {len(self.cfgs_with_server)} total")

        return agents

    @gen.coroutine
    def launch_n_serverless_agents(self, n):
        agents = yield [
            self.executor.submit(
                build_agent_from_cfg, self.cfgs_with_server[i % len(self.cfgs_with_server)]
            )
            for i in range(n)
        ]
        LOGGER.info(f"Launched {len(agents)} new serverless agents")
        return agents


def run_with_cfg(cfg):
    bot = Bot(
        cfg.host,
        cfg.port,
        cfg.agent,
        period_seconds=cfg.period,
        buffer_size=cfg.buffer_size,
        reuse_model_servers=cfg.reuse_model_servers,
        game_id=cfg.game_id,
        power=cfg.power,
    )
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
