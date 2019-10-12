import argparse
import torch

from fairdiplomacy.data.dataset import Dataset
from fairdiplomacy.models.consts import ADJACENCY_MATRIX
from dipnet import DipNetEncoder

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-dataloader-workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    A = torch.from_numpy(ADJACENCY_MATRIX).float()
    net = DipNetEncoder(35, 40, 120, 7, 3, 16, A)

    print("Loading dataset...")
    dataset = Dataset(
        ["/Users/jsgray/code/fairdiplomacy/fairdiplomacy/data/out/game_3232.json"] * 16
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, shuffle=True, num_workers=args.num_dataloader_workers, batch_size=args.batch_size
    )

    for batch_i, batch in enumerate(dataloader):
        print("Starting batch {}".format(batch_i))

        for i, tensor in enumerate(batch):
            batch[i] = tensor.reshape(-1, *tensor.shape[2:])  # flatten batch of batches

        x_state, x_orders, x_power, x_season, y_actions = batch

        y_guess = net(x_state, x_orders, x_power, x_season)

        print(y_guess.shape)

        break
