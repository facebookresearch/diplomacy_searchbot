import json
import logging
from collections import defaultdict
from pydipcc import Game
from fairdiplomacy.models.consts import POWERS
import heyhi


def order_prob(prob_distributions, *expected_orders):
    total = 0
    for pd in prob_distributions.values():
        for orders, prob in pd.items():
            if all(x in orders for x in expected_orders):
                total += prob
    return total


def fragment_prob(prob_distributions, power, fragment):
    total = 0
    for orders, prob in prob_distributions[power].items():
        if any(fragment in x for x in orders):
            total += prob
    return total


def run_situation_check(meta, agent):
    results = {}
    for name, config in meta.items():
        logging.info("=" * 80)
        comment = config.get("comment", "")
        logging.info(f"{name}: {comment} (phase={config.get('phase')})")
        # If path is not absolute, treat as relative to code root.
        game_path = heyhi.PROJ_ROOT / config["game_path"]
        logging.info(f"path: {game_path}")
        with open(game_path) as f:
            game = Game.from_json(f.read())
        if "phase" in config:
            game.rollback_to_phase(config["phase"])

        if hasattr(agent, "get_all_power_prob_distributions"):
            prob_distributions = agent.get_all_power_prob_distributions(game)  # FIXME: early exit
            logging.info("CFR strategy:")
        else:
            # this is a supervised agent, sample N times to get a distribution
            NUM_ROLLOUTS = 100
            prob_distributions = {p: defaultdict(float) for p in POWERS}
            for power in POWERS:
                for N in range(NUM_ROLLOUTS):
                    orders = agent.get_orders(game, power)
                    prob_distributions[power][tuple(orders)] += 1 / NUM_ROLLOUTS

        for power in POWERS:
            pd = prob_distributions[power]
            pdl = sorted(list(pd.items()), key=lambda x: -x[1])
            logging.info(f"   {power}")

            for order, prob in pdl:
                if prob < 0.02:
                    break
                logging.info(f"       {prob:5.2f} {order}")

        for i, (test_desc, test_func_str) in enumerate(config.get("tests", {}).items()):
            test_func = eval(test_func_str)
            passed = test_func(prob_distributions)
            results[f"{name}.{i}"] = int(passed)
            res_string = "PASSED" if passed else "FAILED"
            logging.info(f"Result: {res_string:8s}  {name:20s} {test_desc}")
            logging.info(f"        {test_func_str}")
    logging.info("Passed: %d/%d", sum(results.values()), len(results))
    logging.info("JSON: %s", results)
    return results
