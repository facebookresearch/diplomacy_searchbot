import argparse
import torch
import time
import logging
import json

from fairdiplomacy.game import Game
from fairdiplomacy.agents.dipnet_agent import encode_inputs
from fairdiplomacy.models.dipnet.load_model import load_dipnet_model, new_model

# logging.basicConfig(format="%(asctime)s [%(levelname)s]: %(message)s", level=logging.DEBUG)


B = [2 ** x for x in range(11)]
N = 20


def load_late_game():
    with open("/checkpoint/jsgray/diplomacy/slurm/cmp_mila__mila/game_TUR.2.json", "r") as f:
        late_game = Game.from_saved_game_format(json.load(f))
    late_game.set_phase_data(late_game.get_phase_history()[-2])
    return late_game


def profile_model(model_path):
    late_game = load_late_game()

    model = load_dipnet_model(model_path, map_location="cuda", eval=True)

    for game_name, game in [("new_game", Game()), ("late_game", late_game)]:
        print("\n#", game_name)
        inputs = encode_inputs(game)
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        for batch_size in B:
            b_inputs = {
                k: v.repeat((batch_size,) + (1,) * (len(v.shape) - 1)) for k, v in inputs.items()
            }
            with torch.no_grad():
                tic = time.time()
                for _ in range(N):
                    order_idxs, order_scores, cand_scores, final_scores = model(
                        **b_inputs, temperature=1.0
                    )
                toc = time.time() - tic

                print(
                    f"[B={batch_size}] {toc}s / {N}, latency={1000*toc/N}ms, throughput={N*batch_size/toc}/s"
                )


# USEFUL SNIPPETS

# with torch.autograd.profiler.profile(use_cuda=True, record_shapes=True) as prof:
# print(prof.key_averages().table(sort_by="self_cpu_time_total"))
