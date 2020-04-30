import glob
import torch
import logging

from fairdiplomacy.agents import build_agent_from_cfg
from fairdiplomacy.data.dataset import Dataset

logging.getLogger().setLevel(logging.DEBUG)


def build_db_cache_from_cfg(cfg):
    assert cfg.glob, cfg
    assert cfg.out_path, cfg

    game_json_paths = glob.glob(cfg.glob)

    dataset = Dataset(
        game_json_paths,
        only_with_min_final_score=cfg.only_with_min_final_score,
        cf_agent=(None if cfg.cf_agent is None else build_agent_from_cfg(cfg.cf_agent)),
        n_cf_agent_samples=cfg.n_cf_agent_samples,
        n_jobs=cfg.n_parallel_jobs
    )

    torch.save(dataset, cfg.out_path)
