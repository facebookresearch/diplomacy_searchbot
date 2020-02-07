import glob
import os
import tempfile
import time
import torch

from fairdiplomacy.data.dataset import Dataset

if __name__ == "__main__":
    game_jsons = glob.glob("/checkpoint/jsgray/diplomacy/mila_dataset/tiny_data/*.json")

    k = 1
    tic = time.time()
    for _ in range(k):
        cache = Dataset(game_jsons)
    delta = time.time() - tic
    print(f"Encoded {k*len(game_jsons)} games in {delta}s. {k*len(game_jsons)/delta} games/s")

    feats = cache[torch.tensor([100, 2], dtype=torch.long)]

    with tempfile.TemporaryDirectory() as tmpd:
        torch.save(cache, os.path.join(tmpd, "test_cache.pt"))
        cache2 = torch.load(os.path.join(tmpd, "test_cache.pt"))

    feats2 = cache[torch.tensor([100, 2], dtype=torch.long)]
    assert all((f1 == f2).all() for f1, f2 in zip(feats, feats2))

    N = len(cache2)
    tic = time.time()
    B, k = 1000, 10
    for i in range(k):
        idx = (torch.rand(B) * 0.99 * N).to(torch.long)
        data = cache2[idx]
    delta = time.time() - tic
    print(f"Looked up {B*k} in {delta} s. {B*k/delta} / s")
