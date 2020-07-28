"""Creates cache using slurm array of jobs.

Usage:
    python bin/create_cache.py \
        /checkpoint/jsgray/diplomacy/mila_dataset/data \
        /checkpoint/yolo/fairdiplomacy/tmp/par_data

"""
import glob
import pathlib
import os
import random
import time

import submitit
import torch


def create_dataset(game_jsons, only_with_min_final_score, out_path):
    from fairdiplomacy.data.dataset import Dataset
    import torch

    # Todo(apjacob): Fix dataset
    dataset = Dataset(game_jsons, only_with_min_final_score=only_with_min_final_score, n_jobs=10,)
    torch.save(dataset, out_path)


def main(src_dir, work_folder, only_with_min_final_score, chunks_train, chunks_val, partition):
    from fairdiplomacy.data.dataset import Dataset

    val_set_pct = 0.01
    random.seed(0)
    game_jsons = glob.glob(os.path.join(src_dir, "*.json"))
    assert len(game_jsons) > 0
    print(f"Found dataset of {len(game_jsons)} games...")
    val_game_jsons = sorted(random.sample(game_jsons, max(1, int(len(game_jsons) * val_set_pct))))
    train_game_jsons = sorted(set(game_jsons) - set(val_game_jsons))

    work_folder.mkdir(exist_ok=True, parents=True)
    chunk_folder = work_folder / "chunks"
    chunk_folder.mkdir(exist_ok=True, parents=True)

    chunks_train = min(len(train_game_jsons), chunks_train)
    chunks_val = min(len(val_game_jsons), chunks_val)

    tasks = []
    for i in range(chunks_train):
        out_path = chunk_folder / f"train_{i:06d}"
        if not out_path.exists():
            tasks.append((train_game_jsons[i::chunks_train], only_with_min_final_score, out_path))
    for i in range(chunks_val):
        out_path = chunk_folder / f"val_{i:06d}"
        if not out_path.exists():
            tasks.append((val_game_jsons[i::chunks_val], only_with_min_final_score, out_path))

    executor = submitit.AutoExecutor(folder=work_folder / "submitit")
    executor.update_parameters(
        slurm_partition=partition, slurm_constraint="pascal", cpus_per_task=10, slurm_time=24
    )

    if tasks:
        active_jobs = executor.map_array(create_dataset, *zip(*tasks))
    else:
        active_jobs = []

    while active_jobs:
        print(f"Waiting for {len(active_jobs)} jobs to finish")
        time.sleep(60)
        active_jobs = [x for x in active_jobs if not x.done()]

    def load_merge(prefix, size):
        chunks = []
        for i in range(size):
            p = chunk_folder / f"{prefix}_{i:06d}"
            d = torch.load(p)
            print(f"-- loaded {p} size={len(d)}")
            chunks.append(d)
        return Dataset.from_merge(chunks)

    print("Merging train")
    train = load_merge("train", chunks_train)
    print("Merging val")
    val = load_merge("val", chunks_val)
    outpath = work_folder / "cache.pth"
    print("Saving to", outpath)
    torch.save((train, val), str(outpath))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("src_dir")
    parser.add_argument(
        "work_folder", type=pathlib.Path, help="Where to store intermidiate fils and final result"
    )
    parser.add_argument("--only_with_min_final_score", type=int, default=0)
    parser.add_argument("--chunks_train", type=int, default=256)
    parser.add_argument("--chunks_val", type=int, default=10)
    parser.add_argument("--partition", default="learnfair")
    main(**vars(parser.parse_args()))
