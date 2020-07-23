import json
import logging
from fairdiplomacy.game import Game
from fairdiplomacy.models.consts import POWERS


def order_prob(prob_distributions, *expected_orders):
    total = 0
    for pd in prob_distributions.values():
        for orders, prob in pd.items():
            if all(x in orders for x in expected_orders):
                total += prob
    return total


def run_situation_check(meta, agent):
    results = {}
    for name, config in meta.items():
        logging.info("=" * 80)
        comment = config.get("comment", "")
        logging.info(f"{name}: {comment} (phase={config.get('phase')})")
        logging.info(f"path: {config['game_path']}")
        with open(config["game_path"]) as f:
            j = json.load(f)
        game = Game.from_saved_game_format(j)
        if "phase" in config:
            game = Game.clone_from(game, up_to_phase=config["phase"])

        prob_distributions = agent.get_all_power_prob_distributions(game)  # FIXME: early exit
        logging.info("CFR strategy:")
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
