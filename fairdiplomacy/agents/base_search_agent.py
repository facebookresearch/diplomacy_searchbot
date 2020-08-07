import logging
import numpy as np
from collections import Counter
from typing import List, Dict, Union, Sequence, Tuple
from fairdiplomacy.agents.dipnet_agent import encode_inputs, encode_batch_inputs

from fairdiplomacy.agents.base_agent import BaseAgent
from fairdiplomacy.agents.dipnet_agent import resample_duplicate_disbands_inplace
from fairdiplomacy.models.consts import POWERS
from fairdiplomacy.utils.game_scoring import compute_game_scores_from_state


class BaseSearchAgent(BaseAgent):
    def get_plausible_orders(
        self,
        game,
        *,
        n=1000,
        temperature=1.0,
        limit: Union[int, Sequence[int]],  # limit, or list of limits per power
        batch_size=500,
        top_p=1.0,
    ) -> Dict[str, Dict[Tuple[str], float]]:
        assert n % batch_size == 0, f"{n}, {batch_size}"

        # limits is a list of 7 limits
        limits = [limit] * 7 if type(limit) == int else limit
        assert len(limits) == 7
        del limit

        # trivial return case: all powers have at most `limit` actions
        # orderable_locs = game.get_orderable_locations()
        # if max(map(len, orderable_locs.values())) <= 1:
        #     all_orders = game.get_all_possible_orders()
        #     pow_orders = {
        #         p: all_orders[orderable_locs[p][0]] if orderable_locs[p] else [] for p in POWERS
        #     }
        #     if all(len(pow_orders[p]) <= limit for p, limit in zip(POWERS, limits)):
        #         return {p: set((x,) for x in orders) for p, orders in pow_orders.items()}

        # non-trivial return case: query model
        counters = {p: Counter() for p in POWERS}

        # FIXME: this if statement is hacky and awful
        if hasattr(self, "thread_pool"):
            x = [encode_batch_inputs(self.thread_pool, [game])] * n
        else:
            x = [encode_inputs(game)] * n

        orders_to_logprobs = {}
        for x_chunk in [x[i : i + batch_size] for i in range(0, n, batch_size)]:
            batch_inputs = self.cat_pad_inputs(x_chunk)
            batch_orders, batch_order_logprobs, _ = self.do_model_request(
                batch_inputs, temperature, top_p
            )
            batch_orders = list(zip(*batch_orders))  # power -> list[orders]
            batch_order_logprobs = batch_order_logprobs.t()  # [7 x B]
            for p, power in enumerate(POWERS):
                counters[power].update(batch_orders[p])

            # slow and steady
            for power_orders, power_scores in zip(batch_orders, batch_order_logprobs):
                for order, score in zip(power_orders, power_scores):
                    if order not in orders_to_logprobs:
                        orders_to_logprobs[order] = score
                    assert (
                        abs(orders_to_logprobs[order] - score) < 1e-2
                    ), f"{order} : {orders_to_logprobs[order]} != {score}"

        logging.info(
            "get_plausible_orders(n={}, t={}) found {} unique sets, choosing top {}".format(
                n, temperature, list(map(len, counters.values())), limits
            )
        )

        # filter out badly-coordinated actions
        counters = {
            power: (
                filter_keys(counter, are_supports_coordinated) if len(counter) > limit else counter
            )
            for (power, counter), limit in zip(counters.items(), limits)
        }

        most_common = {
            power: sorted(counter.most_common(), key=lambda o: -orders_to_logprobs[o[0]])[:limit]
            for (power, counter), limit in zip(counters.items(), limits)
        }

        # # choose most common
        # most_common = {
        #     power: counter.most_common(limit)
        #     for (power, counter), limit in zip(counters.items(), limits)
        # }

        try:
            logging.info(
                "get_plausible_orders filtered down to {} unique sets, n_0={}, n_cut={}".format(
                    list(map(len, counters.values())),
                    [safe_idx(most_common[p], 0, default=(None, None))[1] for p in POWERS],
                    [
                        safe_idx(most_common[p], limit - 1, default=(None, None))[1]
                        for (p, limit) in zip(POWERS, limits)
                    ],
                )
            )
        except:
            # TODO: remove this if not seen in production
            logging.warning("error in get_plausible_orders logging")

        logging.info("Plausible orders:")
        logging.info("        count,count_frac,prob")
        for power, orders_and_counts in most_common.items():
            logging.info(f"    {power}")
            for orders, count in orders_and_counts:
                logging.info(
                    f"        {count:5d} {count/n:10.5f} {np.exp(orders_to_logprobs[orders]):10.5f}  {orders}"
                )

        return {
            power: {orders: orders_to_logprobs[orders] for orders, _ in orders_and_counts}
            for power, orders_and_counts in most_common.items()
        }


def compute_sampled_logprobs(sampled_idxs, logits):
    sampled_idxs = sampled_idxs[:, :, : logits.shape[-2]]  # trim off excess seq dim
    invalid_mask = sampled_idxs < 0
    sampled_idxs = sampled_idxs.clamp(
        min=0
    )  # otherwise gather(-1) will blow up. We'll mask these out later
    logprobs = logits.log_softmax(-1)
    sampled_logprobs = logprobs.gather(-1, sampled_idxs.unsqueeze(-1)).squeeze(-1)
    sampled_logprobs[invalid_mask] = 0
    total_logprobs = sampled_logprobs.sum(-1)
    return total_logprobs


def model_output_transform(x, y):
    order_idxs, sampled_idxs, logits, final_sos = y
    resample_duplicate_disbands_inplace(
        order_idxs, sampled_idxs, logits, x["x_possible_actions"], x["x_in_adj_phase"]
    )
    return order_idxs, compute_sampled_logprobs(sampled_idxs, logits), final_sos


def average_score_dicts(score_dicts: List[Dict]) -> Dict:
    return {p: sum(d.get(p, 0) for d in score_dicts) / len(score_dicts) for p in POWERS}


def filter_keys(d, fn, log_warn=False):
    """Return a copy of a dict-like input containing the subset of keys where fn(k) is truthy"""
    r = type(d)()
    for k, v in d.items():
        if fn(k):
            r[k] = v
        elif log_warn:
            logging.warning(f"filtered bad key: {k}")
    return r


def are_supports_coordinated(orders: List[str]) -> bool:
    """Return False if any supports or convoys are not properly coordinated

    e.g. if "F BLA S A SEV - RUM", return False if "A SEV" is not ordered "A SEV - RUM"
             0  1  2 3  4  5  6
    """
    required = {}
    ordered = {}

    for order in orders:
        split = order.split()
        ordered[split[1]] = split  # save by location
        if split[2] in ("S", "C"):
            if split[4] in required and required[split[4]] != split[3:]:
                # an order is already required of this unit, but it contradicts this one
                return False
            else:
                required[split[4]] = split[3:]

    for req_loc, req_order in required.items():
        if req_loc not in ordered:
            # supporting a foreign unit is always allowed, since we can't
            # control the coordination
            continue

        actual_order = ordered[req_loc]

        if len(req_order) == 2 and actual_order[2] == "-":
            # we supported a hold, but it tried to move
            return False
        elif (
            len(req_order) > 2
            and req_order[2] == "-"
            and (actual_order[2] != "-" or actual_order[3][:3] != req_order[3][:3])
        ):
            # we supported a move, but the order given was (1) not a move, or
            # (2) a move to the wrong destination
            return False

    # checks passed, return True
    return True


def safe_idx(seq, idx, default=None):
    try:
        return seq[idx]
    except IndexError:
        return default


def n_move_phases_later(from_phase, n):
    year_idx = int(from_phase[1:-1]) - 1901
    season = from_phase[0]
    from_move_phase_idx = 2 * year_idx + (1 if season in "FW" else 0)
    to_move_phase_idx = from_move_phase_idx + n
    to_move_phase_year = to_move_phase_idx // 2 + 1901
    to_move_phase_season = "S" if to_move_phase_idx % 2 == 0 else "F"
    return f"{to_move_phase_season}{to_move_phase_year}M"


def get_square_scores_from_game(game):
    return [
        compute_game_scores_from_state(power_idx, game.get_state()).square_score
        for power_idx in range(len(POWERS))
    ]
