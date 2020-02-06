import logging
import time

import diplomacy

from fairdiplomacy.agents.dipnet_agent import DipnetAgent


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s [%(levelname)s]: %(message)s", level=logging.DEBUG)

    agent = DipnetAgent("/checkpoint/jsgray/dipnet.pth")
    game = diplomacy.Game()
    orders = agent.get_orders(game, "ITALY")
    logging.info("Submit orders: {}".format(orders))

    b, N = 1000, 100

    tic = time.time()
    for i in range(N):
        orders = agent.get_orders(game, "ITALY")
    timing = time.time() - tic
    print(f"batch {b} N {N} : forward'd {b*N} in {timing} s. {b*N/timing} forwards/s")

    # with torch.autograd.profiler.profile(use_cuda=True, record_shapes=True) as prof:
    #     agent.get_orders(game, "ITALY", temperature=args.temperature, batch_size=400)
    # print(prof.table(row_limit=10000))i
