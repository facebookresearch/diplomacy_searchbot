"""Tools to run h2h over several seeds and aggregate results over such evals."""
from typing import List, Tuple, Optional
import enum
import dataclasses
import getpass
import json
import logging
import os
import pathlib

from fairdiplomacy.typedefs import Power
from fairdiplomacy.data.build_dataset import GameVariant
from fairdiplomacy.models.consts import POWERS
from fairdiplomacy.utils import game_scoring
from slurm.tasks import Task, run_locally_or_on_slurm
import heyhi
import conf.conf_cfgs


@dataclasses.dataclass
class EvalRun:
    cfg: Optional[conf.conf_cfgs.CompareAgentsTask]
    num_seeds: int
    initial_seed: int
    out_dir: pathlib.Path
    done_evals: List[pathlib.Path]
    active_powers: List[Power]


DEFAULT_CFG = heyhi.CONF_ROOT / "c01_ag_cmp" / "cmp.prototxt"
# Name of a file to create a started eval.
STARTED_FILE = "started"

DEFAULT_MODE = "START"
DEFAULT_VARIANT = "CLASSIC"

PREFIX2POWER = {("game_" + power[:3]): power for power in POWERS}
PREFIX_LEN = 8
assert all([len(k) == PREFIX_LEN for k in PREFIX2POWER.keys()])


def get_default_output_dir(username: Optional[str] = None) -> pathlib.Path:
    return pathlib.Path("./compare_agents_array_results/")


class Mode(enum.Enum):
    # Launch if the folder didn't exist.
    START = enum.auto()
    # Launch only new/failed seeds.
    START_MISSING = enum.auto()
    # Only check config is valid and exit.
    CHECK = enum.auto()


def get_eval_game_file_name(power: Power, seed: int) -> str:
    power_short = power[:3]
    return f"game_{power_short}.{seed}.json"


def get_power_scores_from_folder(
    game_dir: pathlib.Path, all_agents: bool = False
) -> List[Tuple[Power, game_scoring.GameScores]]:
    """Given a dir with h2h game files like game_{POWER}.{SEED}.{info,json,html} load scores from it.

    See fairdiplomacy.env for the code that records hvh game files in this way.

    Parameters:
    game_dir: Path to the dir that contains the game json and info files.

    Returns: A list of all the (power, scores for that power) results across all games.
"""
    pairs = []
    for path in game_dir.glob("game*.json"):
        if all_agents:
            powers = POWERS
        else:
            powers = [PREFIX2POWER[path.name[:PREFIX_LEN]]]
        for power in powers:
            info_path = pathlib.Path(str(path).replace(".json", ".info"))
            game_scores = None
            if info_path.exists():
                with info_path.open() as stream:
                    info = json.load(stream)
                game_scores = game_scoring.GameScores(**(info["game_scores"][power]))
            else:
                with path.open() as stream:
                    game = json.load(stream)
                game_scores = game_scoring.compute_game_scores(POWERS.index(power), game)

            pairs.append((power, game_scores))

    return pairs


def _run_game(
    out_dir: pathlib.Path, power: Power, seed: int, eval_cfg_path: pathlib.Path,
):
    game_json_name = get_eval_game_file_name(power, seed)
    log_dir = (out_dir / game_json_name.rsplit(".", 1)[0]).absolute()
    log_dir.mkdir(exist_ok=True, parents=True)
    tokens = [
        "python",
        f"{heyhi.PROJ_ROOT}/run.py",
        "--mode=start_continue",
        f"--cfg={eval_cfg_path}",
        f"--out={log_dir}",
        f"power_one={power}",
        f"seed={seed}",
        f"out={out_dir}/{game_json_name}",
        ">>",
        f"{log_dir}/stdout.log",
        "2>>",
        f"{log_dir}/stderr.log",
    ]
    logging.info("Running command: " + " ".join(tokens))
    cmd = os.system(" ".join(tokens))


def launch(
    missing_jobs: List[Tuple[Power, int]],
    out_dir: pathlib.Path,
    cfg: conf.conf_cfgs.MetaCfg,
):
    assert len(missing_jobs) > 0
    pid = os.getpid()
    eval_cfg_path = out_dir / f"eval_cfg.{pid}.prototxt"
    with eval_cfg_path.open("w") as stream:
        stream.write(str(cfg))

    tasks = []
    for (power, seed) in missing_jobs:
        task = Task(
            target_file_name=get_eval_game_file_name(power, seed),
            task_kwargs=dict(
                out_dir=out_dir, power=power, seed=seed, eval_cfg_path=eval_cfg_path,
            ),
            handler=_run_game,
            # We don't want the file pre-opened for us, since _run_game itself
            # already is responsble for producing the game json file.
            no_open_target_file=True,
        )
        tasks.append(task)

    run_locally_or_on_slurm(
        tasks=tasks,
        results_dir=out_dir,
        slurm_dir=out_dir / "slurm",
    )


def run_evals(
    *,
    cfg_path: pathlib.Path = DEFAULT_CFG,
    mode: Mode = getattr(Mode, DEFAULT_MODE),
    variant: GameVariant = getattr(GameVariant, DEFAULT_VARIANT),
    num_seeds: int,
    initial_seed: int,
    out_dir: pathlib.Path,
    overrides: List[str],
    active_powers: Optional[List[Power]] = None,
    capture_logs: bool = False,
    share_strategy: bool = False,
) -> EvalRun:
    assert cfg_path.exists(), cfg_path
    if active_powers is not None:
        pass
    elif variant == GameVariant.FVA:
        active_powers = ["FRANCE", "AUSTRIA"]
    else:
        assert variant == GameVariant.CLASSIC
        active_powers = POWERS

    if variant == GameVariant.FVA:
        overrides.append(
            "start_game=./fva_starting_position.json"
        )
    if capture_logs:
        overrides.append("capture_logs=1")
    if share_strategy:
        overrides.append("share_strategy=1")

    cfg: Optional[conf.conf_cfgs.MetaCfg]
    try:
        cfg = heyhi.load_root_config(cfg_path, overrides)
    except Exception as e:
        logging.warning(
            "Failed to result agent config. Stale flag?\nCfg: %s\nOverrides: %s\nError: %s",
            cfg_path,
            overrides,
            e,
        )
        if mode != Mode.CHECK:
            logging.error("Not in check mode, gonna raise now!")
            raise
        else:
            logging.warning("Check mode. Will proceed without a config")
        cfg = None
    else:
        assert cfg.which_task == "compare_agents", cfg
        logging.info("Eval cfg (seed and output will be overwritten):\n%s", cfg.compare_agents)

    assert cfg is not None
    if cfg.compare_agents.use_shared_agent:
        logging.info("CompareAgentsTask.share_strategy is true -> will run evals only for AUS")
        assert "AUSTRIA" in active_powers, active_powers
        active_powers = ["AUSTRIA"]

    knows_files = frozenset(x.name for x in out_dir.iterdir()) if out_dir.exists() else frozenset()
    missing_jobs: List[Tuple[Power, int]] = []
    found_games: List[pathlib.Path] = []
    # Iterate over seed first to get more representative partial results.
    for seed in range(initial_seed, initial_seed + num_seeds):
        for power in active_powers:
            name = get_eval_game_file_name(power, seed)
            if name in knows_files:
                found_games.append(out_dir / name)
            else:
                missing_jobs.append((power, seed))

    if mode != Mode.CHECK:
        out_dir.mkdir(exist_ok=True, parents=True)
        started_file = out_dir / STARTED_FILE
        if not started_file.exists() or mode == mode.START_MISSING:
            if not missing_jobs:
                logging.info("No missing evals!")
            else:
                assert (
                    cfg is not None
                ), "Need to launch jobs but the config/overrides is not valid!"
                launch(missing_jobs, out_dir, cfg)

                with started_file.open("w"):
                    pass
        else:
            logging.info(
                "Eval in the folder exists. Not running again."
                " Use --mode=start_missing to compute missing"
            )
    else:
        logging.info("Check mode, not running anything")

    logging.info("Out dir: %s", out_dir)
    logging.info("Logs:\n\ttail -n5 %s/*/*err.log", out_dir)
    logging.info("Done %d out of %d", len(found_games), len(found_games) + len(missing_jobs))
    return EvalRun(
        out_dir=out_dir,
        done_evals=found_games,
        cfg=cfg.compare_agents if cfg is not None else None,
        active_powers=active_powers,
        num_seeds=num_seeds,
        initial_seed=initial_seed,
    )
