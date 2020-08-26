"""Inregration tests for heyhi based flows.

Note: on devfair some extra test cases will be enabled that use real data.

Use `notetest <path>` to execute this file.
"""
import unittest

from parameterized import parameterized
import torch

import fairdiplomacy.models.dipnet.load_model
import heyhi
import integration_tests.build_test_cache
import integration_tests.heyhi_utils

CACHE_PATH = integration_tests.build_test_cache.GAMES_CACHE_ROOT
TMP_DIR = integration_tests.heyhi_utils.OUTPUT_ROOT


def _load_task_cfg(cfg_path, overides=tuple()):
    task_name, cfg = heyhi.load_cfg(cfg_path, overides)
    return getattr(cfg, task_name)


def _create_supervised_model(sup_model_path):
    sup_model_path.parent.mkdir(exist_ok=True, parents=True)
    sl_cfg = _load_task_cfg(
        heyhi.CONF_ROOT / "c02_sup_train" / "sl.prototxt", ["num_encoder_blocks=1"]
    )
    model = fairdiplomacy.models.dipnet.load_model.new_model(sl_cfg)
    ckpt = {"model": model.state_dict(), "args": sl_cfg}
    print(model)
    torch.save(ckpt, sup_model_path)


@parameterized([("sl.prototxt",), ("sl_20200717.prototxt",)])
def test_train_configs(cfg_name):
    cfg_path = heyhi.CONF_ROOT / "c02_sup_train" / cfg_name
    integration_tests.heyhi_utils.run_config(
        cfg=cfg_path, overrides=[f"data_cache={CACHE_PATH}", "num_epochs=1", "debug_no_mp=1"]
    )


@parameterized([("sl.prototxt",), ("sl_20200717.prototxt",)])
def test_train_configs_real_data(cfg_name):
    if not heyhi.is_devfair():
        raise unittest.SkipTest("Can run only on devfair")
    cfg_path = heyhi.CONF_ROOT / "c02_sup_train" / cfg_name
    integration_tests.heyhi_utils.run_config(
        cfg=cfg_path, overrides=["num_epochs=1", "debug_no_mp=1", "epoch_max_batches=10"]
    )


@parameterized([("exploit_06.prototxt",), ("selfplay_01.prototxt",)])
def test_rl_configs(cfg_name):
    sup_model_path = TMP_DIR / "sup_model_for_rl.pth"
    _create_supervised_model(sup_model_path)

    integration_tests.heyhi_utils.run_config(
        cfg=heyhi.CONF_ROOT / "c04_exploit" / cfg_name,
        overrides=[
            f"model_path={sup_model_path}",
            "trainer.max_epochs=1",
            "rollout.num_rollout_processes=-1",
            "trainer.epoch_size=1",
        ],
    )


@parameterized([("exploit_06.prototxt",), ("selfplay_01.prototxt",)])
def test_rl_configs_real_data(cfg_name):
    if not heyhi.is_devfair():
        raise unittest.SkipTest("Can run only on devfair")
    integration_tests.heyhi_utils.run_config(
        cfg=heyhi.CONF_ROOT / "c04_exploit" / cfg_name,
        overrides=[
            "trainer.max_epochs=1",
            "rollout.num_rollout_processes=-1",
            "trainer.epoch_size=1",
        ],
    )


def test_situation_check_configs():
    sup_model_path = TMP_DIR / "sup_model_for_test_sit.pth"
    _create_supervised_model(sup_model_path)

    situation_json = heyhi.PROJ_ROOT / "integration_tests/data/test_situations.json"
    integration_tests.heyhi_utils.run_config(
        cfg=heyhi.CONF_ROOT / "c06_situation_check" / "cmp.prototxt",
        overrides=[
            f"situation_json={situation_json}",
            "I.agent=agents/cfr1p",
            f"agent.cfr1p.model_path={sup_model_path}",
            "agent.cfr1p.n_rollouts=1",
            "agent.cfr1p.n_rollout_procs=1",
        ],
    )


def test_situation_check_configs_length0():
    sup_model_path = TMP_DIR / "sup_model_for_test_sit.pth"
    _create_supervised_model(sup_model_path)

    situation_json = heyhi.PROJ_ROOT / "integration_tests/data/test_situations.json"
    integration_tests.heyhi_utils.run_config(
        cfg=heyhi.CONF_ROOT / "c06_situation_check" / "cmp.prototxt",
        overrides=[
            f"situation_json={situation_json}",
            "I.agent=agents/cfr1p",
            f"agent.cfr1p.model_path={sup_model_path}",
            "agent.cfr1p.n_rollouts=1",
            "agent.cfr1p.max_rollout_length=0",
            "agent.cfr1p.n_rollout_procs=1",
        ],
    )


def test_build_cache():
    out_path = TMP_DIR / "test_build_cache" / "cache.out"
    overrides = [
        f"glob={integration_tests.build_test_cache.GAMES_ROOT.absolute()}/*.json",
        f"out_path={out_path}",
    ]
    integration_tests.heyhi_utils.run_config(
        cfg=integration_tests.build_test_cache.BUILD_DB_CONF, overrides=overrides
    )


def test_compare_agents():
    out_path = TMP_DIR / "c01_ag_cmp" / "model.pth"
    _create_supervised_model(out_path)
    integration_tests.heyhi_utils.run_config(
        cfg=heyhi.CONF_ROOT / "c01_ag_cmp" / "cmp.prototxt",
        overrides=[
            "I.agent_one=agents/dipnet",
            "I.agent_six=agents/dipnet",
            f"agent_one.dipnet.model_path={out_path}",
            f"agent_six.dipnet.model_path={out_path}",
            "max_turns=1",
        ],
    )
