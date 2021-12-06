# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Generator, Dict, Optional, Sequence, Tuple
import collections
import dataclasses
import datetime
import io
import logging
import pathlib
import random

import numpy as np
import torch

from fairdiplomacy import pydipcc
from fairdiplomacy.action_exploration import compute_evs_fva, double_oracle_fva
from fairdiplomacy.agents.model_wrapper import compute_action_logprobs
from fairdiplomacy.data.dataset import DataFields
from fairdiplomacy.models.consts import POWERS, MAX_SEQ_LEN
from fairdiplomacy.models.diplomacy_model.order_vocabulary import EOS_IDX
from fairdiplomacy.selfplay.ckpt_syncer import build_search_agent_with_syncs
from fairdiplomacy.selfplay.search_utils import (
    all_power_prob_distributions_to_tensors,
    create_research_targets_single_rollout,
    perform_retry_loop,
    unparse_device,
)
from fairdiplomacy.typedefs import Policy
from fairdiplomacy.utils.batching import batched_forward
from fairdiplomacy.utils.sampling import sample_p_dict
from fairdiplomacy.utils.order_idxs import (
    ORDER_VOCABULARY,
    action_strs_to_global_idxs,
    global_order_idxs_to_local,
)
from fairdiplomacy.utils.timing_ctx import TimingCtx
from fairdiplomacy.utils.thread_pool_encoding import FeatureEncoder
import nest
import postman

import conf.agents_cfgs
import conf.conf_cfgs

# ReSearchRollout = collections.namedtuple("ReSearchRollout", "game_json, observations, first_phase")
ReSearchRolloutBatch = collections.namedtuple(
    "ReSearchRolloutBatch",
    [
        "observations",
        "rewards",  # Tensor [T, 7]. Final rewards for the episode.
        "done",  # Tensor bool [T]. False everywhere but last phase.
        "is_explore",  # Tensor bool [T, 7]. True if the agent deviated at this phase.
        "explored_on_the_right",  # Tensor bool [T, 7]. True if the agent will deviate.
        "scores",  # [T, 7], SoS scores at the end of each phase.
        "values",  # [T, 7], Values for all states from the value net.
        "targets",  # [T, 7], bootstapped targets.
        "is_search_policy_valid",  # [T]. search_policy_probs, search_policy_orders, and
        # blueprint_probs should be ingored if this is 0.
        "state_hashes",  # [T] long hash of the state
        "years",  # [T]
        "phase_type",  # Byte tensor [T], contains ord("M"), ord("A"), or ord("R").
        "search_policy_evs",  # Float tensor [T, 7, max_actions] or zeros of [T] if not collect_search_evs.
        "search_policy_probs",  # [T, 7, max_actions] or zeros of [T].
        "search_policy_orders",  # [T, 7, max_actions, MAX_SEQ_LEN] or zeros of [T].
        "blueprint_probs",  # [T, 7, max_actions] or zeros of [T]. Probabilities of search_policy_orders under blueprint.
    ],
)


@dataclasses.dataclass(frozen=True)
class RollloutResult:
    batch: ReSearchRolloutBatch
    # Info about the initial state rollout produced on
    game_meta: Dict  # Info about the produced game and initial state.
    # Info about the checkpoint that was used for the rollout. One per syncer.
    last_ckpt_metas: Optional[Dict[str, Dict]]


def yield_game(seed: int, game_json_paths: Optional[Sequence[str]], game_kwargs: Optional[Dict]):
    rng = np.random.RandomState(seed=seed)
    while True:
        if game_json_paths is None:
            game = pydipcc.Game()
            game_id = "std"
        else:
            p = game_json_paths[rng.choice(len(game_json_paths))]
            with open(p) as stream:
                game_serialized = stream.read()
            game = pydipcc.Game.from_json(game_serialized)
            game_id = pathlib.Path(p).name.rsplit(".", 1)[0]
        if game_kwargs is not None:
            for k, v in game_kwargs.items():
                getattr(game, f"set_{k}")(v)
        yield game_id, game


def yield_rollouts(
    *,
    device: str,
    game_json_paths: Optional[Sequence[str]],
    searchbot_cfg: conf.agents_cfgs.SearchBotAgent,
    seed=None,
    ckpt_sync_path=None,
    eval_mode=False,
    game_kwargs=None,
    stats_server: Optional[str] = None,
    extra_params_cfg: conf.conf_cfgs.ExploitTask.SearchRollout.ExtraRolloutParams,
    collect_game_logs=False,
) -> Generator[RollloutResult, None, None]:
    """Do non-stop rollout for 7 search agents.

    This method can safely be called in a subprocess

    Arguments:
    - model_path: str
    - game_jsons: either None or a list of paths to json-serialized games.
    - searchbot_cfg: message of type SearchBotAgent.
    - seed: seed for _some_ randomness within the rollout.
    - ckpt_sync_path: if not None, will load the model from this folder.
    - eval_mode: whether rollouts are for train or eval, e.g., to exploration in eval.
    - game_kwargs: Optional a dict of modifiers for the game.
    - stats_server: if not None, a postman server where to send stats about the game
    - extra_params_cfg: Misc flags straight from the proto.
    - collect_game_logs: if True, the game json will collect logs for each phase.

    yields a RollloutResult.

    """
    if seed is not None:
        torch.manual_seed(seed)

    fake_gen = extra_params_cfg.fake_gen or 1

    # ---- Creating searchbot config to use in this worker.
    agent, do_sync_fn = build_search_agent_with_syncs(
        searchbot_cfg,
        ckpt_sync_path=ckpt_sync_path,
        use_trained_value=extra_params_cfg.use_trained_value,
        use_trained_policy=extra_params_cfg.use_trained_policy,
        device_id=unparse_device(device),
    )

    assert not (
        extra_params_cfg.independent_explore and extra_params_cfg.explore_all_but_one
    ), "Mutually exclusive flags"

    input_encoder = FeatureEncoder()

    if extra_params_cfg.max_year is not None:
        logging.warning("Will simuluate only up to %s", extra_params_cfg.max_year)

    max_plausible_orders = agent.n_plausible_orders

    if stats_server:
        stats_client = postman.Client(stats_server)
        stats_client.connect()
    else:
        stats_client = None

    # ---- Logging.
    logger = logging.getLogger("")
    if collect_game_logs:
        logger.info("Collecting logs into game json")
        _set_all_logger_handlers_to_level(logger, logging.WARNING)
    game_log_handler = None

    rng = np.random.RandomState(seed=seed)
    last_ckpt_metas = None
    for rollout_id, (game_id, game) in enumerate(yield_game(seed, game_json_paths, game_kwargs)):
        if extra_params_cfg.max_year:
            if extra_params_cfg.randomize_max_year:
                game_max_year = rng.randint(1902, extra_params_cfg.max_year + 1)
            else:
                game_max_year = extra_params_cfg.max_year
        else:
            game_max_year = None
        if rollout_id < 1 or (rollout_id & (rollout_id + 1)) == 0:
            _set_all_logger_handlers_to_level(logger, logging.INFO)
        else:
            logging.info("Skipping logs for rollout %s", rollout_id)
            _set_all_logger_handlers_to_level(logger, logging.WARNING)
            # Keep the game log handler at info level for writing into game json
            if game_log_handler is not None:
                game_log_handler.setLevel(logging.INFO)
        game_meta = {"start_phase": game.current_short_phase, "game_id": game_id}
        timings = TimingCtx()
        with timings("ckpt_sync"):
            if ckpt_sync_path is not None:
                last_ckpt_metas = do_sync_fn()

        if extra_params_cfg.grow_action_space_till:
            assert ckpt_sync_path
            epoch = max(d["epoch"] for d in last_ckpt_metas.values())
            agent.n_plausible_orders = int(
                _linear_grows(
                    t=epoch,
                    start_t=100,
                    end_t=extra_params_cfg.grow_action_space_till,
                    start_value=5,
                    end_value=searchbot_cfg.n_plausible_orders,
                )
            )
            logging.info(
                "Setting n_plausible_orders=%d for epoch=%d", agent.n_plausible_orders, epoch
            )

        with timings("rl.init"):
            # Need to do some measurement to make Lost time tracking work.
            stats = collections.defaultdict(float)
            observations = []
            is_explore = []
            scores = []
            phases = []
            years = []
            phase_type = []
            state_hashes = []

            is_search_policy_valid_flags = []  # Only filled if collect_search_policies.
            search_policies = []  # Only filled if collect_search_policies.
            search_policy_evs = []  # Only filled if collect_search_evs.
            target_evs = []  # Only filled if collect_search_evs.
        while not game.is_game_done:
            if game_max_year and int(game.current_short_phase[1:-1]) > game_max_year:
                break

            if collect_game_logs:
                if game_log_handler is not None:
                    logger.removeHandler(game_log_handler)
                log_stream = io.StringIO()
                game_log_handler = logging.StreamHandler(log_stream)
                game_log_handler.setLevel(logging.INFO)
                logger.addHandler(game_log_handler)
            else:
                log_stream = None

            state_hashes.append(game.compute_board_hash() % 2 ** 63)
            phases.append(game.current_short_phase)
            years.append(int(game.current_short_phase[1:-1]))
            phase_type.append(ord(game.current_short_phase[-1]))
            observations.append(input_encoder.encode_inputs([game]))

            alive_power_ids = [
                i for i, score in enumerate(game.get_square_scores()) if score > 1e-3
            ]

            if game.current_short_phase == "S1901M" and extra_params_cfg.explore_s1901m_eps > 0:
                phase_explore_eps = extra_params_cfg.explore_s1901m_eps
            elif game.current_short_phase == "F1901M" and extra_params_cfg.explore_f1901m_eps > 0:
                phase_explore_eps = extra_params_cfg.explore_f1901m_eps
            else:
                phase_explore_eps = extra_params_cfg.explore_eps

            if extra_params_cfg.independent_explore:
                do_explore = [
                    not eval_mode and phase_explore_eps > 0 and phase_explore_eps > rng.uniform()
                    for _ in POWERS
                ]
            else:
                someone_explores = (
                    not eval_mode and phase_explore_eps > 0 and phase_explore_eps > rng.uniform()
                )
                if someone_explores:
                    do_explore = [True] * len(POWERS)
                    if extra_params_cfg.explore_all_but_one:
                        do_explore[random.choice(alive_power_ids)] = False
                else:
                    do_explore = [False] * len(POWERS)

            for i in range(len(POWERS)):
                if i not in alive_power_ids:
                    # Dead never wander.
                    do_explore[i] = False

            with timings("plausible_orders"):
                if extra_params_cfg.random_policy:
                    plausible_orders_policy = _random_plausible_orders(
                        observations[-1]["x_possible_actions"], max_plausible_orders
                    )
                else:
                    plausible_orders_policy = agent.get_plausible_orders_policy(game)

            run_double_oracle: bool
            is_search_policy_valid: bool
            if eval_mode or extra_params_cfg.do is None:
                run_double_oracle = False
                is_search_policy_valid = True
            else:
                is_search_policy_valid = random.random() <= extra_params_cfg.run_do_prob
                run_double_oracle = is_search_policy_valid and game.current_short_phase.endswith(
                    "M"
                )
            inner_timings = TimingCtx()
            if not run_double_oracle:
                all_power_prob_distributions = agent.get_all_power_prob_distributions(
                    game, timings=inner_timings, bp_policy=plausible_orders_policy
                )
            else:
                all_power_prob_distributions, do_stats = double_oracle_fva(
                    game,
                    agent,
                    double_oracle_cfg=extra_params_cfg.do,
                    need_actions_only=False,
                    initial_plausible_orders_policy=plausible_orders_policy,
                    timings=inner_timings,
                )
                stats["do_attempts"] += 1
                stats["do_successes"] += do_stats["num_changes"]
                # Truncating to max_plausible_orders.
                all_power_prob_distributions = {
                    power: truncate_policy(policy, max_plausible_orders)
                    for power, policy in all_power_prob_distributions.items()
                }

            timings += inner_timings
            # Making a move: sampling from the policies.
            power_orders = {}
            for i, power in enumerate(POWERS):
                if not all_power_prob_distributions[power]:
                    power_orders[power] = tuple()
                elif do_explore[i]:
                    # Uniform sampling from plausible actions.
                    power_orders[power] = _choice(
                        rng, list(plausible_orders_policy[power].keys()) or [tuple()]
                    )
                else:
                    power_orders[power] = sample_p_dict(
                        all_power_prob_distributions[power], rng=rng
                    )

            if extra_params_cfg.collect_search_policies:
                is_search_policy_valid_flags.append(is_search_policy_valid)
                if is_search_policy_valid:
                    with timings("rl.collect_policies"):
                        phase_orders, phase_probs = all_power_prob_distributions_to_tensors(
                            all_power_prob_distributions,
                            max_plausible_orders,
                            observations[-1]["x_possible_actions"].squeeze(0),
                            observations[-1]["x_in_adj_phase"].item(),
                        )
                        phase_blueprint_probs = _compute_action_probs(
                            agent.model.model,
                            game,
                            all_power_prob_distributions,
                            max_actions=max_plausible_orders,
                            batch_size=searchbot_cfg.plausible_orders_cfg.batch_size,
                            half_precision=searchbot_cfg.half_precision,
                        )
                        search_policies.append(
                            dict(
                                probs=phase_probs,
                                orders=phase_orders,
                                blueprint_probs=phase_blueprint_probs,
                            )
                        )
                else:
                    search_policies.append(
                        dict(
                            probs=torch.empty(
                                (len(POWERS), max_plausible_orders), dtype=torch.long
                            ),
                            orders=torch.empty((len(POWERS), max_plausible_orders, MAX_SEQ_LEN)),
                            blueprint_probs=torch.empty((len(POWERS), max_plausible_orders)),
                        )
                    )
            if extra_params_cfg.collect_search_evs:
                with timings("rl.collect_evs"):
                    # FIXME(akhti): This will die for non-FvA games.
                    phase_evs, phase_action_evs = _compute_evs_as_tensors(
                        game,
                        agent,
                        all_power_prob_distributions,
                        max_actions=max_plausible_orders,
                    )
                target_evs.append(phase_evs)
                search_policy_evs.append(phase_action_evs)

            is_explore.append(do_explore)
            for power, orders in power_orders.items():
                game.set_orders(power, orders)
            if collect_game_logs:
                game.add_log(log_stream.getvalue())
            game.process()
            scores.append(game.get_square_scores())

        timings.start("rl.aggregate")
        # Must do before merging.
        rollout_length = len(observations)
        # Convert everything to tensors.
        observations = DataFields.cat(observations)
        scores = torch.as_tensor(scores)
        rewards = scores[-1].unsqueeze(0).repeat(rollout_length, 1)
        is_explore = torch.as_tensor(is_explore)
        done = torch.zeros(rollout_length, dtype=torch.bool)
        done[-1] = True

        if extra_params_cfg.collect_search_policies:
            is_search_policy_valid_flags = torch.as_tensor(
                is_search_policy_valid_flags, dtype=torch.bool
            )
            search_policies = nest.map_many(lambda x: torch.stack(x, 0), *search_policies)
            search_policy_probs = search_policies["probs"]
            search_policy_orders = search_policies["orders"]
            blueprint_probs = search_policies["blueprint_probs"]
        else:
            assert not is_search_policy_valid_flags
            assert not search_policies
            is_search_policy_valid_flags = torch.zeros(rollout_length, dtype=torch.bool)
            search_policy_probs = search_policy_orders = blueprint_probs = torch.zeros(
                rollout_length
            )

        if extra_params_cfg.collect_search_evs:
            search_policy_evs = torch.stack(search_policy_evs)
        else:
            assert not search_policy_evs
            assert not target_evs
            search_policy_evs = torch.zeros(rollout_length)

        values = torch.as_tensor(
            agent.model.do_model_request(
                x=observations, temperature=-1, top_p=-1, values_only=True
            )
        )
        values = values.float()

        # [False, True, False, False] -> [ True, True, False, False].
        explored_on_the_right = (
            torch.flip(torch.cumsum(torch.flip(is_explore.long(), [0]), 0), [0]) > 0
        )

        if extra_params_cfg.use_ev_targets:
            assert extra_params_cfg.collect_search_evs
            targets = torch.stack(target_evs)
        else:
            targets = create_research_targets_single_rollout(
                is_explore, scores[-1], values, scores > 1e-3, extra_params_cfg.discounting
            )
        timings.stop()
        timings.pprint(logging.getLogger("timings.search_rollout").info)

        if stats_client is not None and last_ckpt_metas is not None:
            # Also here be tolerarant of the other side hanging due to things like
            # torch.distributed.init_process_group
            aggregated = _aggregate_stats(game, stats, last_ckpt_metas, timings=timings)

            def send_to_client():
                stats_client.add_stats(aggregated)

            perform_retry_loop(send_to_client, max_tries=20, sleep_seconds=10)

        game_meta["game"] = game

        research_batch = ReSearchRolloutBatch(
            observations=observations,
            rewards=rewards,
            done=done,
            is_explore=is_explore,
            explored_on_the_right=explored_on_the_right,
            scores=scores,
            values=values,
            targets=targets,
            state_hashes=torch.tensor(state_hashes, dtype=torch.long),
            years=torch.tensor(years),
            phase_type=torch.ByteTensor(phase_type),
            is_search_policy_valid=is_search_policy_valid_flags,
            search_policy_evs=search_policy_evs,
            search_policy_probs=search_policy_probs,
            search_policy_orders=search_policy_orders,
            blueprint_probs=blueprint_probs,
        )
        for key, value in research_batch._asdict().items():
            if key != "observations":
                assert value.shape[0] == rollout_length, nest.map(
                    lambda x: x.shape, research_batch._asdict()
                )
        for _ in range(fake_gen):
            yield RollloutResult(
                batch=research_batch, game_meta=game_meta, last_ckpt_metas=last_ckpt_metas
            )


def _set_all_logger_handlers_to_level(logger, level):
    for handler in logger.handlers[:]:
        handler.setLevel(level)


def _compute_evs_as_tensors(
    game, agent, policies, *, max_actions: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """This function computes EVs for FvA game and format it as 7p EVs.

    It will die if the game is not FvA.

    Returns tuple (tensor [7], tensor [7, max_actions])
    """
    ev_target_aus, ev_aus, ev_fra = compute_evs_fva(game, agent, policies, min_action_prob=0.01)
    ev_target = ev_aus.new_zeros([len(POWERS)])
    ev_target[POWERS.index("AUSTRIA")] = ev_target_aus
    ev_target[POWERS.index("FRANCE")] = 1 - ev_target_aus

    result = ev_aus.new_full((len(POWERS), max_actions), -1)
    result[POWERS.index("AUSTRIA"), : len(ev_aus)] = ev_aus
    result[POWERS.index("FRANCE"), : len(ev_fra)] = ev_fra
    return ev_target, result


def _aggregate_stats(
    game, raw_stats: Dict, last_ckpt_metas: Dict, timings: TimingCtx
) -> Dict[str, float]:
    stats = {}
    # Metrics.
    if "do_attempts" in raw_stats:
        stats["rollouter/do_success_rate"] = raw_stats["do_successes"] / raw_stats["do_attempts"]
    stats["rollouter/phases"] = len(game.get_phase_history())
    # Timings.
    total = sum(v for k, v in timings.items())
    for k, v in timings.items():
        stats[f"rollouter_timings_pct/{k}"] = v / (total + 1e-100)
    stats.update((f"rollouter/epoch_{k}", d["epoch"]) for k, d in last_ckpt_metas.items())
    # Global step.
    stats["epoch"] = max(d["epoch"] for d in last_ckpt_metas.values())
    # PostMan hack.
    stats = {k: torch.tensor(v).view(1, 1) for k, v in stats.items()}
    return stats


def truncate_policy(policy: Policy, max_items: int) -> Policy:
    if len(policy) <= max_items:
        return policy
    actions, probs = zip(*collections.Counter(policy).most_common(max_items))
    total_prob = sum(probs)
    probs = [x / total_prob for x in probs]
    return dict(zip(actions, probs))


def _compute_action_probs(
    model, game, plausible_actions, max_actions, batch_size, half_precision
) -> torch.Tensor:
    power_action_pairs = []
    per_power_inputs = []
    observations = FeatureEncoder().encode_inputs([game])
    for power, actions in plausible_actions.items():
        assert len(actions) <= max_actions, (power, len(actions), max_actions)
        power_id = POWERS.index(power)
        inputs = nest.map(lambda x: x.expand(len(actions), *x.shape[1:]), observations)
        inputs["teacher_force_orders"] = torch.full(
            (len(actions), MAX_SEQ_LEN), EOS_IDX, dtype=torch.long
        )
        for i, action in enumerate(actions):
            # Have to convert first and the use its size to handle joined build orders.
            t = torch.as_tensor(action_strs_to_global_idxs(action))
            inputs["teacher_force_orders"][i, : len(t)] = t
        inputs["x_power"] = torch.full((len(actions), MAX_SEQ_LEN), power_id)
        inputs["x_loc_idxs"] = inputs["x_loc_idxs"][:, power_id]
        inputs["x_possible_actions"] = inputs["x_possible_actions"][:, power_id]

        per_power_inputs.append(inputs)
        power_action_pairs.extend((power_id, i) for i in range(len(actions)))

    def f(batch):
        model_batch = DataFields(**batch)
        model_batch["teacher_force_orders"] = batch["teacher_force_orders"].clamp(min=0)
        if half_precision:
            model_batch = model_batch.to_half_precision()
        _, _, logits, _ = model(**model_batch, need_value=False, temperature=1.0)
        try:
            local_indices = global_order_idxs_to_local(
                batch["teacher_force_orders"], batch["x_possible_actions"]
            )
        except Exception:
            local_indices = global_order_idxs_to_local(
                batch["teacher_force_orders"], batch["x_possible_actions"], ignore_missing=True
            )
            shuttle = dict(game=game.to_json(), plausible_actions=plausible_actions, batch=batch)
            torch.save(shuttle, "debug_global_order.%s.pt" % datetime.datetime.now())

        # Hack to make resample_duplicate_disbands_inplace work. Adding POWER dimension.
        logprobs = compute_action_logprobs(local_indices.unsqueeze(1), logits.unsqueeze(1))
        # Remove fake power.
        logprobs = logprobs.squeeze(1)
        return logprobs.exp()

    joined_input = nest.map_many(lambda x: torch.cat(x, 0), *per_power_inputs)
    device = next(iter(model.parameters())).device
    probs = batched_forward(f, joined_input, batch_size=batch_size, device=device)

    prob_tensor = torch.zeros((len(POWERS), max_actions))
    for (power_id, action_id), prob in zip(power_action_pairs, probs.cpu().tolist()):
        prob_tensor[power_id, action_id] = prob
    return prob_tensor


def _choice(rng, sequence):
    return sequence[rng.randint(0, len(sequence))]


def _linear_grows(t, start_t, end_t, start_value, end_value):
    alpha = (t - start_t) / (end_t - start_t)
    alpha = max(min(1.0, alpha), 0.0)
    return alpha * (end_value - start_value) + start_value


def _random_plausible_orders(x_possible_actions, max_plausible_orders):
    assert x_possible_actions.shape[1] == 7, x_possible_actions.shape
    x_possible_actions = x_possible_actions.squeeze(0)
    plausible_orders = {}
    for i, power in enumerate(POWERS):
        power_possible_actions = x_possible_actions[i]
        if (power_possible_actions == -1).all():
            plausible_orders[power] = {tuple(): -1.0}
            continue
        plausible_orders[power] = {}
        for _ in range(max_plausible_orders):
            action = []
            for row in power_possible_actions:
                if (row == -1).all():
                    continue
                row = row[row != -1]
                action.append(ORDER_VOCABULARY[random.choice(row)])
            if len(action) == 1 and ";" in action[0]:
                action = action[0].split(";")
            plausible_orders[power][tuple(action)] = -1.0

    return plausible_orders
